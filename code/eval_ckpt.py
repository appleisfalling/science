# -*- coding: utf-8 -*-
"""Per-image evaluation of a finished run directory (read-only w.r.t. existing files).

Usage (on server, from /data/kimi_repro/code/ukan_dg):
  python eval_ckpt.py --run /data/kimi_repro/outputs/ukan_dg/glas_A4full_s2981 [--ckpt best_model.pth]
                      [--data_dir inputs] [--latency] [--out eval_metrics.json]

Writes <run>/<out> (new file). Reports
  * batch-aggregated IoU/Dice exactly as train_dg.py/validate() (sanity check against log.csv)
  * per-image IoU, Dice, precision, recall, HD95 (px), ASSD (px), boundary F-score (tol=2 px)
  * optional latency (batch 1, fp32, RTX), peak memory, params/GFLOPs (calflops)
"""
import argparse, os, sys, json, glob, time, math
import numpy as np, cv2, torch, yaml
import warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from albumentations import Compose, Resize
from albumentations.augmentations import transforms as A_tf
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import archs

try:
    from medpy.metric.binary import hd95 as _hd95, assd as _assd
except Exception:  # pragma: no cover
    _hd95 = _assd = None


def build_model(cfg):
    arch = cfg['arch']
    emb = cfg.get('input_list', [128, 160, 256])
    common = dict(num_classes=cfg['num_classes'], input_channels=cfg['input_channels'],
                  deep_supervision=cfg.get('deep_supervision', False), embed_dims=emb,
                  no_kan=cfg.get('no_kan', False), kan_mask=cfg.get('kan_mask', 'K,K,K'))
    if arch == 'UKAN':
        return archs.UKAN(**common)
    if arch == 'UKAN_DG':
        import archs_dg
        return archs_dg.UKAN_DG(use_sid=cfg['use_sid'], use_cdfa=cfg['use_cdfa'],
                                use_cc=cfg['use_cc'], use_boundary=cfg['use_boundary'], **common)
    if arch == 'UKAN_V2':
        import archs_v2
        return archs_v2.build_from_config(cfg)
    raise ValueError(arch)


def logits_of(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def boundary(mask):
    """1-px boundary of a binary mask (erosion-based)."""
    m = mask.astype(bool)
    er = ndimage.binary_erosion(m, structure=np.ones((3, 3)), border_value=0)
    return m & ~er


def bf_score(pred, gt, tol=2):
    bp, bg = boundary(pred), boundary(gt)
    if bp.sum() == 0 and bg.sum() == 0:
        return 1.0
    if bp.sum() == 0 or bg.sum() == 0:
        return 0.0
    dg = ndimage.distance_transform_edt(~bg)
    dp = ndimage.distance_transform_edt(~bp)
    prec = (dg[bp] <= tol).mean()
    rec = (dp[bg] <= tol).mean()
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def per_image(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    inter, union = (p & g).sum(), (p | g).sum()
    ps, gs = p.sum(), g.sum()
    r = {}
    r['iou'] = 1.0 if union == 0 else inter / union
    r['dice'] = 1.0 if ps + gs == 0 else 2 * inter / (ps + gs)
    r['precision'] = 1.0 if ps == 0 and gs == 0 else (0.0 if ps == 0 else inter / ps)
    r['recall'] = 1.0 if gs == 0 and ps == 0 else (0.0 if gs == 0 else inter / gs)
    r['bf2'] = bf_score(p, g, 2)
    if ps == 0 and gs == 0:
        r['hd95'] = 0.0; r['assd'] = 0.0
    elif ps == 0 or gs == 0 or _hd95 is None:
        r['hd95'] = float('nan'); r['assd'] = float('nan')
    else:
        r['hd95'] = float(_hd95(p, g)); r['assd'] = float(_assd(p, g))
    r['pred_empty'] = bool(ps == 0); r['gt_empty'] = bool(gs == 0)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--ckpt', default=None, help='default: best_model.pth if present else model.pth')
    ap.add_argument('--data_dir', default='inputs')
    ap.add_argument('--out', default='eval_metrics.json')
    ap.add_argument('--latency', action='store_true')
    ap.add_argument('--save_pred', default=None, help='dir to dump per-image prob maps (.npy, uint8*255)')
    a = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(a.run, 'config.yml')))
    ckpt = a.ckpt or ('best_model.pth' if os.path.exists(os.path.join(a.run, 'best_model.pth')) else 'model.pth')
    dev = 'cuda'
    model = build_model(cfg)
    sd = torch.load(os.path.join(a.run, ckpt), map_location='cpu')
    model.load_state_dict(sd, strict=True)
    model = model.to(dev).eval()

    ds = cfg['dataset']
    img_ext = '.png'; mask_ext = '_mask.png' if ds == 'busi' else '.png'
    ids = sorted(os.path.splitext(os.path.basename(p))[0]
                 for p in glob.glob(os.path.join(a.data_dir, ds, 'images', '*' + img_ext)))
    if cfg.get('split') == 'p2':
        from split_p2 import split_p2
        _, val_ids, test_ids = split_p2(ids, cfg['dataseed'])
        partitions = [('eval_metrics.json', val_ids, 'val'), ('eval_test_metrics.json', test_ids, 'test')]
    else:
        _, val_ids = train_test_split(ids, test_size=0.2, random_state=cfg['dataseed'])
        partitions = [(a.out, val_ids, 'val')]
    tf = Compose([Resize(cfg['input_h'], cfg['input_w']), A_tf.Normalize()])
    bs = cfg.get('batch_size', 8)

    def eval_partition(pids, save_pred):
        per = {}
        # batch-aggregated replica of train_dg.validate (batch 8, drop_last=False)
        agg_iou_sum = 0.0; agg_dice_sum = 0.0; n_seen = 0
        probs_all = {}
        with torch.no_grad():
            for b0 in range(0, len(pids), bs):
                chunk = pids[b0:b0 + bs]
                xs, ys = [], []
                for iid in chunk:
                    img = cv2.imread(os.path.join(a.data_dir, ds, 'images', iid + img_ext))
                    m = cv2.imread(os.path.join(a.data_dir, ds, 'masks', '0', iid + mask_ext), cv2.IMREAD_GRAYSCALE)[..., None]
                    aug = tf(image=img, mask=m)
                    x = aug['image'].astype('float32') / 255; x = x.transpose(2, 0, 1)
                    y = aug['mask'].astype('float32') / 255; y = y.transpose(2, 0, 1)
                    if y.max() < 1: y[y > 0] = 1.0
                    xs.append(x); ys.append(y)
                x = torch.from_numpy(np.stack(xs)).to(dev); y = torch.from_numpy(np.stack(ys)).to(dev)
                prob = torch.sigmoid(logits_of(model(x))).cpu().numpy()
                yn = y.cpu().numpy()
                pb = prob > 0.5; gb = yn > 0.5
                smooth = 1e-5
                iou_b = ((pb & gb).sum() + smooth) / ((pb | gb).sum() + smooth)
                dice_b = 2 * iou_b / (iou_b + 1)
                agg_iou_sum += iou_b * len(chunk); agg_dice_sum += dice_b * len(chunk); n_seen += len(chunk)
                for k, iid in enumerate(chunk):
                    per[iid] = per_image(pb[k, 0], gb[k, 0])
                    if save_pred:
                        probs_all[iid] = (prob[k, 0] * 255).astype(np.uint8)
        return per, (agg_iou_sum / n_seen if n_seen else None), (agg_dice_sum / n_seen if n_seen else None), probs_all

    keys = ['iou', 'dice', 'precision', 'recall', 'bf2', 'hd95', 'assd']
    for pi, (out_name, pids, pname) in enumerate(partitions):
        per, agg_iou, agg_dice, probs_all = eval_partition(pids, a.save_pred if pi == 0 else None)
        summary = {}
        for k in keys:
            v = np.array([per[i][k] for i in pids], dtype=float)
            ok = ~np.isnan(v)
            summary[k] = {'mean': float(np.nanmean(v)) if ok.any() else None,
                          'std': float(np.nanstd(v, ddof=1)) if ok.sum() > 1 else None,
                          'median': float(np.nanmedian(v)) if ok.any() else None,
                          'n_valid': int(ok.sum())}
        summary['n_images'] = len(pids)
        summary['n_pred_empty'] = int(sum(per[i]['pred_empty'] for i in pids))
        summary['n_gt_empty'] = int(sum(per[i]['gt_empty'] for i in pids))
        summary['batch_agg_iou'] = agg_iou
        summary['batch_agg_dice'] = agg_dice
        summary['ckpt'] = ckpt
        summary['partition'] = pname
        summary['n_params'] = sum(p.numel() for p in model.parameters())

        if a.latency and pi == 0:
            x = torch.randn(1, 3, cfg['input_h'], cfg['input_w'], device=dev)
            torch.backends.cudnn.benchmark = True
            with torch.no_grad():
                for _ in range(20): model(x)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                ts = []
                for _ in range(100):
                    t0 = time.perf_counter(); model(x); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
            summary['latency_ms_b1'] = {'mean': 1000 * float(np.mean(ts)), 'std': 1000 * float(np.std(ts))}
            summary['peak_mem_MB_b1'] = torch.cuda.max_memory_allocated() / 2**20
            try:
                from calflops import calculate_flops
                m_cpu = build_model(cfg); m_cpu.load_state_dict(sd); m_cpu.eval()
                f, p, _ = calculate_flops(model=m_cpu, input_shape=(1, 3, cfg['input_h'], cfg['input_w']),
                                          output_as_string=False, print_results=False, print_detailed=False)
                summary['gflops'] = f / 1e9; summary['params_M_calflops'] = p / 1e6
            except Exception as e:  # pragma: no cover
                summary['gflops_error'] = str(e)

        out = {'run': a.run, 'config': {k: cfg[k] for k in ('arch', 'dataset', 'dataseed', 'name')},
               'summary': summary, 'per_image': per}
        with open(os.path.join(a.run, out_name), 'w') as f:
            json.dump(out, f, indent=1)
        if a.save_pred and pi == 0:
            os.makedirs(a.save_pred, exist_ok=True)
            np.savez_compressed(os.path.join(a.save_pred, cfg['name'] + '_probs.npz'), **probs_all)
        print('[%s]' % pname, json.dumps({k: summary[k] for k in ('batch_agg_iou', 'batch_agg_dice', 'n_images', 'n_pred_empty')}),
              '| iou %.4f dice %.4f hd95 %s assd %s bf2 %.4f' % (
                  summary['iou']['mean'], summary['dice']['mean'],
                  None if summary['hd95']['mean'] is None else '%.2f' % summary['hd95']['mean'],
                  None if summary['assd']['mean'] is None else '%.2f' % summary['assd']['mean'],
                  summary['bf2']['mean']))


if __name__ == '__main__':
    main()
