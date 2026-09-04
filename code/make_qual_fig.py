# -*- coding: utf-8 -*-
"""Qualitative figure for the JBHI v2 paper from viz_v2.py dumps.

Usage:
  python make_qual_fig.py --viz /Users/shiyayong/ukan_dg_work/results_cache/viz/glas_2981 \
                          --out "/Users/shiyayong/Documents/ChatGPT/专利/U-KAN-DG_JBHI/figures/fig11_qual_glas.png" \
                          [--rows best_gain median worst most_abstain] [--title GlaS]

One row per case, columns:
  input with ground-truth contour | U-KAN | + ACRE-C | error map of ours (yellow TP, red FP, green FN) | abstention M_a
The row label carries the per-image IoU of both models. Existing output files are backed up to *.bak.
"""
import argparse, json, os, shutil
import numpy as np, cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROW_NAMES = {'best_gain': 'largest gain', 'median': 'median', 'worst': 'worst (failure)', 'most_abstain': 'most abstention'}


def contour_overlay(img_bgr, gt):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
    cs, _ = cv2.findContours((gt > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, cs, -1, (0, 255, 255), 2)
    return rgb


def error_map(prob, gt):
    p, g = prob > 0.5, gt > 127
    out = np.zeros(p.shape + (3,), np.uint8) + 30
    out[p & g] = (255, 215, 0)     # TP yellow
    out[p & ~g] = (220, 30, 30)    # FP red
    out[~p & g] = (40, 180, 60)    # FN green
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--viz', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--rows', nargs='+', default=['best_gain', 'median', 'worst', 'most_abstain'])
    ap.add_argument('--title', default='')
    a = ap.parse_args()
    cases = json.load(open(os.path.join(a.viz, 'cases.json')))
    by_id = {r['id']: r for r in cases['rows']}
    rows = [(k, cases['picks'][k]) for k in a.rows if k in cases['picks']]
    has_abst = any('m_a' in np.load(os.path.join(a.viz, iid + '.npz')) for _, iid in rows)
    ncol = 5 if has_abst else 4
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.05 * ncol, 2.15 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (kind, iid) in enumerate(rows):
        d = np.load(os.path.join(a.viz, iid + '.npz'))
        img, gt = d['image'], d['gt']
        po = d['prob_ours'].astype(np.float32)
        pb = d['prob_base'].astype(np.float32) if 'prob_base' in d else None
        rec = by_id[iid]
        panels = [(contour_overlay(img, gt), 'input + GT contour'),
                  ((pb > 0.5).astype(np.uint8) * 255 if pb is not None else np.zeros_like(gt), 'U-KAN (%.1f)' % (100 * rec.get('iou_base', float('nan')))),
                  ((po > 0.5).astype(np.uint8) * 255, '+ ACRE-C (%.1f)' % (100 * rec['iou_ours'])),
                  (error_map(po, gt), 'error map (ours)')]
        for c, (im, ttl) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(im, cmap='gray' if im.ndim == 2 else None, vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(ttl.split(' (')[0], fontsize=8)
            if c in (1, 2): ax.text(4, 250, ttl.split('(')[1].rstrip(')'), color='w', fontsize=7,
                                    bbox=dict(facecolor='k', alpha=0.5, pad=1.5, lw=0))
        if has_abst:
            ax = axes[r, 4]
            ma = cv2.resize(d['m_a'].astype(np.float32), gt.shape[::-1], interpolation=cv2.INTER_LINEAR)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0, vmax=255)
            hm = ax.imshow(ma, cmap='magma', vmin=0, vmax=1, alpha=0.65)
            cs, _ = cv2.findContours((gt > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cs:
                cnt = cnt[:, 0, :]
                ax.plot(np.r_[cnt[:, 0], cnt[0, 0]], np.r_[cnt[:, 1], cnt[0, 1]], color='cyan', lw=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title('abstention $M_a$', fontsize=8)
        axes[r, 0].set_ylabel('(%s) %s' % ('abcdefgh'[r], ROW_NAMES.get(kind, kind)), fontsize=8)
    if a.title:
        fig.suptitle(a.title, fontsize=9)
    plt.subplots_adjust(wspace=0.04, hspace=0.08, left=0.04, right=0.93, top=0.94 if a.title else 0.96, bottom=0.01)
    if has_abst:
        cax = fig.add_axes([0.94, 0.3, 0.012, 0.4])
        cb = fig.colorbar(hm, cax=cax)
        cb.ax.tick_params(labelsize=7)
    if os.path.exists(a.out):
        shutil.copy2(a.out, a.out + '.bak')
    fig.savefig(a.out, dpi=300)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
