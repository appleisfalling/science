# -*- coding: utf-8 -*-
"""Dump qualitative material for a finished UKAN_V2 run (and, optionally, a matched baseline run).

Usage (server, from /data/kimi_repro/code/ukan_dg):
  python viz_v2.py --run /data/kimi_repro/outputs/ukan_v2/glas_v1_full_2981 \
                   --base /data/kimi_repro/outputs/ukan/glas_UKAN_s2981 \
                   --out  /data/kimi_repro/outputs/ukan_v2/viz/glas_2981 [--n_pick 4] [--all]

Writes into --out (new directory; existing files are never overwritten):
  cases.json            per-image IoU of both models on the SAME validation split, sorted by (ours - base),
                        plus the picked ids: best gain, median, worst (failure) and the one with largest abstention
  <id>.npz              image (uint8 HxWx3, BGR as read), gt (uint8), prob_ours, prob_base (float16),
                        q, rho, m_fg, m_bg, m_a (float16, H/8 grid) for the picked ids (all ids with --all)
Nothing outside --out is touched.
"""
import argparse, glob, json, os, sys
import numpy as np, cv2, torch, yaml
import warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from albumentations import Compose, Resize
from albumentations.augmentations import transforms as A_tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ckpt import build_model, logits_of


def load(run, dev):
    cfg = yaml.safe_load(open(os.path.join(run, 'config.yml')))
    ckpt = 'best_model.pth' if os.path.exists(os.path.join(run, 'best_model.pth')) else 'model.pth'
    m = build_model(cfg)
    m.load_state_dict(torch.load(os.path.join(run, ckpt), map_location='cpu'), strict=True)
    return cfg, m.to(dev).eval()


def iou(p, g):
    p, g = p > 0.5, g > 0.5
    return float(((p & g).sum() + 1e-5) / ((p | g).sum() + 1e-5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--base', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data_dir', default='inputs')
    ap.add_argument('--n_pick', type=int, default=4)
    ap.add_argument('--all', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg, model = load(a.run, dev)
    base = load(a.base, dev)[1] if a.base else None
    ds = cfg['dataset']
    img_ext = '.png'; mask_ext = '_mask.png' if ds == 'busi' else '.png'
    ids = sorted(os.path.splitext(os.path.basename(p))[0]
                 for p in glob.glob(os.path.join(a.data_dir, ds, 'images', '*' + img_ext)))
    _, val_ids = train_test_split(ids, test_size=0.2, random_state=cfg['dataseed'])
    tf = Compose([Resize(cfg['input_h'], cfg['input_w']), A_tf.Normalize()])

    rows, store = [], {}
    with torch.no_grad():
        for iid in val_ids:
            img = cv2.imread(os.path.join(a.data_dir, ds, 'images', iid + img_ext))
            m = cv2.imread(os.path.join(a.data_dir, ds, 'masks', '0', iid + mask_ext), cv2.IMREAD_GRAYSCALE)[..., None]
            aug = tf(image=img, mask=m)
            x = torch.from_numpy((aug['image'].astype('float32') / 255).transpose(2, 0, 1)[None]).to(dev)
            y = (aug['mask'][..., 0].astype('float32') / 255)
            if y.max() < 1: y[y > 0] = 1.0
            out = model(x)
            prob = torch.sigmoid(logits_of(out))[0, 0].cpu().numpy()
            rec = {'id': iid, 'iou_ours': iou(prob, y)}
            item = {'gt': (y * 255).astype(np.uint8), 'prob_ours': prob.astype(np.float16),
                    'image': cv2.resize(img, (cfg['input_w'], cfg['input_h']))}
            if isinstance(out, (tuple, list)) and out[1][1] is not None:
                p, masses = out[1]
                q = torch.sigmoid(p)[0, 0]
                mfg, mbg, ma = masses[0, 0], masses[0, 1], masses[0, 2]
                mu = 1 - (2 * q - 1).abs()
                rho = torch.where(mu > 1e-4, ma / mu.clamp_min(1e-4), torch.full_like(ma, float('nan')))
                for k, v in (('q', q), ('rho', rho), ('m_fg', mfg), ('m_bg', mbg), ('m_a', ma)):
                    item[k] = v.cpu().numpy().astype(np.float16)
                rec['ma_mean'] = float(ma.mean()); rec['ma_max'] = float(ma.max())
                rec['cell_iou'] = iou(q.cpu().numpy(), cv2.resize(y, q.shape[::-1], interpolation=cv2.INTER_AREA))
            if base is not None:
                pb = torch.sigmoid(logits_of(base(x)))[0, 0].cpu().numpy()
                item['prob_base'] = pb.astype(np.float16)
                rec['iou_base'] = iou(pb, y); rec['gain'] = rec['iou_ours'] - rec['iou_base']
            rows.append(rec); store[iid] = item

    key = 'gain' if base is not None else 'iou_ours'
    rows.sort(key=lambda r: r[key], reverse=True)
    n = len(rows)
    picks = {'best_gain': rows[0]['id'], 'median': rows[n // 2]['id'], 'worst': rows[-1]['id']}
    if 'ma_mean' in rows[0]:
        picks['most_abstain'] = max(rows, key=lambda r: r['ma_mean'])['id']
    if a.n_pick > 4:
        for k, r in enumerate(rows[1:a.n_pick - 3]):
            picks['gain_%d' % (k + 2)] = r['id']
    summary = {'run': a.run, 'base': a.base, 'dataset': ds, 'dataseed': cfg['dataseed'], 'n_val': n,
               'mean_iou_ours': float(np.mean([r['iou_ours'] for r in rows])),
               'mean_iou_base': float(np.mean([r['iou_base'] for r in rows])) if base is not None else None,
               'n_improved': int(sum(r.get('gain', 0) > 0 for r in rows)) if base is not None else None,
               'picks': picks, 'rows': rows}
    cp = os.path.join(a.out, 'cases.json')
    if not os.path.exists(cp):
        json.dump(summary, open(cp, 'w'), indent=1)
    for iid in (store if a.all else set(picks.values())):
        fp = os.path.join(a.out, iid + '.npz')
        if not os.path.exists(fp):
            np.savez_compressed(fp, **store[iid])
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
