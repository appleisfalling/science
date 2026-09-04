# -*- coding: utf-8 -*-
"""Multi-dataset qualitative grid for the JBHI v2 paper (one figure* for all datasets).

Usage:
  python make_qual_grid.py --viz glas=/…/results_cache/viz/glas_2981 busi=/…/viz/busi_2981 \
                           cvc=/…/viz/cvc_2981 kvasir=/…/viz/kvasir_2981 \
                           --out "/…/U-KAN-DG_JBHI/figures/fig11_qual_grid.png" [--picks median worst]

Rows = datasets (in the order given); one block of 4 panels per pick:
  input + GT contour | error map U-KAN | error map + ACRE-C | abstention M_a (magma over grey image, cyan GT contour)
Error maps: yellow TP, red FP, green FN; per-image IoU printed in the error panels.
The prediction panels of make_qual_fig.py are dropped because the error maps already carry prediction and GT.
Datasets whose viz dir is missing are skipped (printed). Existing output is backed up to *.bak.
"""
import argparse, json, os, shutil
import numpy as np, cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DS_NAME = {'busi': 'BUSI', 'glas': 'GlaS', 'cvc': 'CVC-ClinicDB', 'kvasir': 'Kvasir-SEG'}
PICK_NAME = {'median': 'median case', 'worst': 'worst case', 'best_gain': 'largest gain', 'most_abstain': 'most abstention'}


def gt_contours(gt):
    cs, _ = cv2.findContours((gt > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cs


def error_map(prob, gt):
    p, g = prob > 0.5, gt > 127
    out = np.zeros(p.shape + (3,), np.uint8) + 30
    out[p & g] = (255, 215, 0)
    out[p & ~g] = (220, 30, 30)
    out[~p & g] = (40, 180, 60)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--viz', nargs='+', required=True, help='ds=path pairs')
    ap.add_argument('--out', required=True)
    ap.add_argument('--picks', nargs='+', default=['median', 'worst'])
    a = ap.parse_args()
    rows = []
    for spec in a.viz:
        ds, path = spec.split('=', 1)
        if not os.path.isfile(os.path.join(path, 'cases.json')):
            print('skip', ds, '(no cases.json in', path + ')')
            continue
        rows.append((ds, path, json.load(open(os.path.join(path, 'cases.json')))))
    if not rows:
        raise SystemExit('nothing to draw')
    npan = 4
    ncol = npan * len(a.picks)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(0.95 * ncol + 0.5, 1.0 * len(rows) + 0.35))
    axes = np.atleast_2d(axes)
    hm = None
    for r, (ds, path, cases) in enumerate(rows):
        by_id = {x['id']: x for x in cases['rows']}
        for b, pick in enumerate(a.picks):
            iid = cases['picks'].get(pick)
            if iid is None:
                continue
            d = np.load(os.path.join(path, iid + '.npz'))
            img, gt = d['image'], d['gt']
            po = d['prob_ours'].astype(np.float32)
            pb = d['prob_base'].astype(np.float32) if 'prob_base' in d else None
            rec = by_id[iid]
            c0 = b * npan
            ax = axes[r, c0]
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for cnt in gt_contours(gt):
                cnt = cnt[:, 0, :]
                ax.plot(np.r_[cnt[:, 0], cnt[0, 0]], np.r_[cnt[:, 1], cnt[0, 1]], color='yellow', lw=0.7)
            ax.set_xlim(0, gt.shape[1]); ax.set_ylim(gt.shape[0], 0)
            panels = [(error_map(pb, gt) if pb is not None else np.zeros_like(img), rec.get('iou_base')),
                      (error_map(po, gt), rec['iou_ours'])]
            for k, (im, v) in enumerate(panels):
                ax = axes[r, c0 + 1 + k]
                ax.imshow(im)
                if v is not None:
                    ax.text(3, gt.shape[0] - 4, '%.1f' % (100 * v), color='w', fontsize=5.5, va='bottom',
                            bbox=dict(facecolor='k', alpha=0.55, pad=1.0, lw=0))
            ax = axes[r, c0 + 3]
            ma = cv2.resize(d['m_a'].astype(np.float32), gt.shape[::-1], interpolation=cv2.INTER_LINEAR)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0, vmax=255)
            hm = ax.imshow(ma, cmap='magma', vmin=0, vmax=1, alpha=0.65)
            for cnt in gt_contours(gt):
                cnt = cnt[:, 0, :]
                ax.plot(np.r_[cnt[:, 0], cnt[0, 0]], np.r_[cnt[:, 1], cnt[0, 1]], color='cyan', lw=0.6)
            ax.set_xlim(0, gt.shape[1]); ax.set_ylim(gt.shape[0], 0)
            if r == 0:
                for k, t in enumerate(['input + GT', 'U-KAN', '+ ACRE-C', '$M_a$']):
                    axes[0, c0 + k].set_title(t, fontsize=6.5, pad=2)
        for c in range(ncol):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for sp in axes[r, c].spines.values():
                sp.set_linewidth(0.3)
        axes[r, 0].set_ylabel(DS_NAME.get(ds, ds), fontsize=7)
    # block titles above the column titles
    for b, pick in enumerate(a.picks):
        x0 = axes[0, b * npan].get_position().x0; x1 = axes[0, b * npan + npan - 1].get_position().x1
        fig.text((x0 + x1) / 2, 0.985, '(%s) %s' % ('abcd'[b], PICK_NAME.get(pick, pick)), ha='center', va='top', fontsize=7.5)
    plt.subplots_adjust(wspace=0.03, hspace=0.05, left=0.035, right=0.95, top=0.88, bottom=0.01)
    if hm is not None:
        cax = fig.add_axes([0.958, 0.25, 0.008, 0.45])
        cb = fig.colorbar(hm, cax=cax)
        cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    if os.path.exists(a.out):
        shutil.copy2(a.out, a.out + '.bak')
    fig.savefig(a.out, dpi=300)
    print('wrote', a.out, 'rows:', [r[0] for r in rows])


if __name__ == '__main__':
    main()
