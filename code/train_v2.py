# -*- coding: utf-8 -*-
"""U-KAN + redesigned routing (v2) training script.

Identical protocol to train_dg.py (which itself mirrors Seg_UKAN/train.py):
  seed_torch(1029) fixed init; --dataseed controls only the 80/20 split; batch 8 drop_last;
  Adam lr 1e-4 wd 1e-4 (KAN fc layers lr 1e-2); CosineAnnealingLR eta_min 1e-5; 400 epochs;
  RandomRotate90 + Flip + Resize + Normalize; best-val-IoU checkpoint + model.pth each epoch;
  cudnn.benchmark = True after seeding (same as the public code).

Differences (all additive, nothing in train_dg.py / archs_dg.py is touched):
  * model  = archs_v2.build_from_config(config)           (arch UKAN_V2; all switches off == archs.UKAN bit-identical)
  * criterion = losses_v2.build_criterion(config)         callable(model_out, target) -> (loss, logits, terms: dict)
  * log.csv gains one column per auxiliary loss term / monitor (mean over the epoch, train_ and val_ prefixed)
  * config.yml records n_params, and copies of train_v2.py / archs_v2.py / losses_v2.py are saved into the run dir
  * generic switch names so the cumulative ablation is  --use_ctrl / +--use_fusion / +--use_context / +--use_objective
  * --opt key=value ... passes design-specific hyper-parameters through to archs_v2 / losses_v2 (stored in config['opt'])
"""
import argparse
import json
import os
import time
from collections import OrderedDict
from glob import glob
import random
import numpy as np

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

from albumentations.augmentations import transforms
from albumentations.augmentations import geometric
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize

import archs_v2
import losses_v2
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool
from tensorboardX import SummaryWriter
import shutil


def list_type(s):
    return [int(a) for a in s.split(',')]


def parse_opt(items):
    """--opt k=v k2=v2  ->  dict with int/float/bool auto-casting."""
    out = {}
    for it in items or []:
        k, v = it.split('=', 1)
        vl = v.lower()
        if vl in ('true', 'false'):
            out[k] = vl == 'true'
            continue
        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(v)
            continue
        except ValueError:
            pass
        out[k] = v
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('--dataseed', default=2981, type=int)
    # model
    parser.add_argument('--arch', '-a', default='UKAN_V2')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_w', default=256, type=int)
    parser.add_argument('--input_h', default=256, type=int)
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])
    # v2 switches (cumulative ablation order)
    parser.add_argument('--use_ctrl', action='store_true', help='controller (shared routing state) + its supervision')
    parser.add_argument('--use_fusion', action='store_true', help='routed skip fusion at the 4 scales (needs ctrl)')
    parser.add_argument('--use_context', action='store_true', help='image-wide consensus fields in every router (implies --use_fusion)')
    parser.add_argument('--use_objective', action='store_true', help='controller-linked auxiliary objective (needs ctrl)')
    parser.add_argument('--lam_ctrl', default=0.25, type=float, help='weight of the controller supervision term')
    parser.add_argument('--lam_obj', default=0.5, type=float, help='weight of the controller-linked objective term')
    parser.add_argument('--opt', nargs='*', default=[], help='design-specific options: key=value ...')
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss')
    # dataset
    parser.add_argument('--dataset', default='busi')
    parser.add_argument('--data_dir', default='inputs')
    parser.add_argument('--split', default='p1', choices=['p1', 'p2'],
                        help='p1: public 80/20 (train/val). p2: outer 80/20 identical to p1, the 20%% is halved '
                             'into val (selection) and test (reporting only, untouched during training)')
    parser.add_argument('--output_dir', default='outputs')
    # optimizer
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)
    parser.add_argument('--kan_lr', default=1e-2, type=float)
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float)
    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--no_kan', action='store_true')
    parser.add_argument('--kan_mask', default='K,K,K', type=str)
    args = parser.parse_args()
    args.opt = parse_opt(args.opt)
    return args


class TermMeters:
    """AverageMeter per auxiliary term name (created lazily)."""

    def __init__(self):
        self.m = OrderedDict()

    def update(self, terms, n):
        for k, v in terms.items():
            if k not in self.m:
                self.m[k] = AverageMeter()
            self.m[k].update(float(v), n)

    def avg(self):
        return OrderedDict((k, m.avg) for k, m in self.m.items())


def _grad_group(name):
    """Coarse parameter groups for the epoch-0 gradient-norm probe (SPEC_v1 §0.2 row 9)."""
    if name.startswith('ctrl.head'):
        return 'ctrl.head'
    if name.startswith('ctrl.'):
        return 'ctrl.trunk'
    for r in ('route1', 'route2', 'route3', 'route4'):
        if name.startswith(r + '.'):
            return r
    if 'layer' in name.lower() and 'fc' in name.lower():
        return 'kan'
    return 'backbone'


def module_norms(model):
    """Weight norms of the routers' expert last layers and detail gains (gate D3 of SPEC_v1 §8)."""
    out = OrderedDict()
    for r in ('route1', 'route2', 'route3', 'route4'):
        m = getattr(model, r, None)
        if m is None:
            continue
        out[r + '_wc'] = m.commit_mix[-1].weight.norm().item()
        out[r + '_wa'] = m.abstain_mix[-1].weight.norm().item()
        if getattr(m, 'detail', False):
            out[r + '_beta'] = m.beta.norm().item()
    return out


def train(config, train_loader, model, criterion, optimizer, grad_probe=None):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    tm = TermMeters()
    model.train()
    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        model_out = model(input)
        loss, logits, terms = criterion(model_out, target)
        iou, dice, _ = iou_score(logits, target)

        optimizer.zero_grad()
        loss.backward()
        if grad_probe is not None:  # epoch-0 only: per-group mean gradient norm over the epoch
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    g = grad_probe.setdefault(_grad_group(name), [0.0, 0])
                    g[0] += p.grad.norm().item(); g[1] += 1
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        tm.update(terms, input.size(0))
        pbar.set_postfix(OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg)]))
        pbar.update(1)
    pbar.close()
    out = OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg)])
    out.update(tm.avg())
    return out


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter()}
    tm = TermMeters()
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()
            model_out = model(input)
            loss, logits, terms = criterion(model_out, target)
            iou, dice, _ = iou_score(logits, target)
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            tm.update(terms, input.size(0))
            pbar.set_postfix(OrderedDict([('loss', avg_meters['loss'].avg),
                                          ('iou', avg_meters['iou'].avg),
                                          ('dice', avg_meters['dice'].avg)]))
            pbar.update(1)
        pbar.close()
    out = OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg),
                       ('dice', avg_meters['dice'].avg)])
    out.update(tm.avg())
    return out


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    config = vars(parse_args())
    exp_name = config.get('name')
    output_dir = config.get('output_dir')

    if config['name'] is None:
        config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
        exp_name = config['name']

    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)
    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')

    cudnn.benchmark = True

    # model
    model = archs_v2.build_from_config(config).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    config['n_params'] = int(n_params)
    print(f'#params: {n_params/1e6:.3f}M')

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)
    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f)

    # criterion: (model_out, target) -> (loss, logits, terms)
    criterion = losses_v2.build_criterion(config)

    # param groups: identical rule to train_dg.py.  archs_v2 must not name any new parameter with both
    # 'layer' and 'fc' in it (that would silently give it the KAN lr 1e-2); asserted in archs_v2.
    # Router kernel logits (.theta) and gain coefficients (.beta) are shape parameters, not weights: no weight decay
    # (SPEC_v1 §0.2 row 21). Same lr as everything else.
    param_groups = []
    n_kan = n_nowd = 0
    for name, param in model.named_parameters():
        if 'layer' in name.lower() and 'fc' in name.lower():
            param_groups.append({'params': param, 'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay']})
            n_kan += 1
        elif name.endswith(('.theta', '.beta')):
            param_groups.append({'params': param, 'lr': config['lr'], 'weight_decay': 0.0})
            n_nowd += 1
        else:
            param_groups.append({'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']})
    print(f'#param tensors in KAN-lr group: {n_kan}; in no-weight-decay group: {n_nowd}')

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)
    else:
        optimizer = optim.SGD(param_groups, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'],
                                                   patience=config['patience'], verbose=1,
                                                   min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    else:
        scheduler = None

    here = os.path.dirname(os.path.abspath(__file__))
    for fn in ('train_v2.py', 'archs_v2.py', 'losses_v2.py'):
        shutil.copy2(os.path.join(here, fn), f'{output_dir}/{exp_name}/')

    dataset_name = config['dataset']
    img_ext = '.png'
    mask_ext = '_mask.png' if dataset_name == 'busi' else '.png'

    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    if config.get('split') == 'p2':
        from split_p2 import split_p2
        train_img_ids, val_img_ids, test_img_ids = split_p2(img_ids, config['dataseed'])
        with open(os.path.join(config['output_dir'], exp_name, 'split_ids_p2.txt'), 'w') as f:
            f.write('val: ' + ' '.join(val_img_ids) + '\n')
            f.write('test: ' + ' '.join(test_img_ids) + '\n')
    else:
        train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])

    train_transform = Compose([RandomRotate90(), geometric.transforms.Flip(),
                               Resize(config['input_h'], config['input_w']), transforms.Normalize()])
    val_transform = Compose([Resize(config['input_h'], config['input_w']), transforms.Normalize()])

    train_dataset = Dataset(img_ids=train_img_ids,
                            img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
                            mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
                            img_ext=img_ext, mask_ext=mask_ext,
                            num_classes=config['num_classes'], transform=train_transform)
    val_dataset = Dataset(img_ids=val_img_ids,
                          img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
                          mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
                          img_ext=img_ext, mask_ext=mask_ext,
                          num_classes=config['num_classes'], transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'],
                                               shuffle=True, num_workers=config['num_workers'], drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'],
                                             shuffle=False, num_workers=config['num_workers'], drop_last=False)

    log = OrderedDict([('epoch', []), ('lr', []), ('loss', []), ('iou', []),
                       ('val_loss', []), ('val_iou', []), ('val_dice', []), ('epoch_sec', [])])
    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))
        t0 = time.time()
        grad_probe = {} if epoch == 0 else None
        train_log = train(config, train_loader, model, criterion, optimizer, grad_probe)
        if grad_probe:
            with open(f'{output_dir}/{exp_name}/grad_norms_epoch0.json', 'w') as f:
                json.dump({k: v[0] / max(v[1], 1) for k, v in grad_probe.items()}, f, indent=2)
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou'], val_log['dice']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['epoch_sec'].append(time.time() - t0)
        # auxiliary terms / monitors
        for k, v in train_log.items():
            if k in ('loss', 'iou'):
                continue
            log.setdefault('train_' + k, []).append(v)
        for k, v in val_log.items():
            if k in ('loss', 'iou', 'dice'):
                continue
            log.setdefault('val_' + k, []).append(v)
        for k, v in module_norms(model).items():  # D3: expert last-layer / detail-gain norms
            log.setdefault('norm_' + k, []).append(v)
        pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/log.csv', index=False)

        my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)
        my_writer.add_scalar('val/iou', val_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/dice', val_log['dice'], global_step=epoch)

        torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model.pth')

        if val_log['iou'] > best_iou:
            print('=> saved best model')
            torch.save(model.state_dict(), f'{output_dir}/{exp_name}/best_model.pth')
            best_iou = val_log['iou']
            trigger = 0

        if config['early_stopping'] >= 0:
            trigger += 1
            if trigger >= config['early_stopping']:
                print('=> early stopping')
                break
        torch.cuda.empty_cache()
    print('DONE best_val_iou %.4f' % best_iou)


if __name__ == '__main__':
    main()
