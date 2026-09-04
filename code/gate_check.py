# -*- coding: utf-8 -*-
"""Evaluate SPEC_v1 §8 screening gates from finished v2 run directories (read-only).

Usage (server or local copy of the run dirs):
  python gate_check.py --out /data/kimi_repro/outputs/ukan_v2 [--json gates.json]

Reads <run>/log.csv (+ DONE marker, config.yml) for every run dir under --out and prints
  * best val IoU / final-epoch IoU / epoch of best, per run
  * ladder (GlaS split 2981):  base -> ctrl -> ctrl_fusion -> full  (G5 monotone within 0.2 pt)
  * G1-G4 thresholds (best val IoU, %):  GlaS >= 85.5, CVC >= 87.0, Kvasir >= 78.5, BUSI >= 71.5
  * D1 ctrl_bce / cell_iou drift  (last 50 epochs vs epochs 50-100), D2 rho_std > 0.05, D3 finite norms < 10
  * U1/U2: full - abst_margin >= 0.2, full - abst_none >= 0.2
Nothing is written unless --json is given (new file only).
"""
import argparse, glob, json, math, os
import pandas as pd

GATES = {'glas': 85.5, 'cvc': 87.0, 'kvasir': 78.5, 'busi': 71.5}
BASE_2981 = {'glas': 84.66, 'cvc': 86.59, 'busi': 70.66, 'kvasir': 75.69}     # U-KAN re-trained, split 2981
DG_2981 = {'glas': 85.77, 'cvc': 87.24, 'busi': 71.86, 'kvasir': 78.93}       # old U-KAN-DG, split 2981


def summarise(run):
    csv = os.path.join(run, 'log.csv')
    if not os.path.isfile(csv):
        return None
    df = pd.read_csv(csv)
    if len(df) == 0:
        return None
    s = {'run': os.path.basename(run), 'epochs': int(len(df)), 'done': os.path.isfile(os.path.join(run, 'DONE'))}
    s['best_val_iou'] = float(df['val_iou'].max() * 100)
    s['best_epoch'] = int(df['val_iou'].idxmax())
    s['final_val_iou'] = float(df['val_iou'].iloc[-1] * 100)
    s['best_val_dice'] = float(df['val_dice'].max() * 100)
    s['nan'] = bool(df.isna().any().any() or (df.select_dtypes('number').abs() == float('inf')).any().any())
    # dynamics (val_ monitors exist only when the controller is on)
    if 'val_ctrl_bce' in df and len(df) >= 150:
        early, late = df.iloc[50:100], df.iloc[-50:]
        s['D1_ctrl_bce_rise'] = float(late['val_ctrl_bce'].mean() - early['val_ctrl_bce'].mean())
        s['D1_cell_iou_drop'] = float((early['val_cell_iou'].mean() - late['val_cell_iou'].mean()) * 100)
        s['D1_pass'] = s['D1_ctrl_bce_rise'] <= 0.02 and s['D1_cell_iou_drop'] <= 1.0
    if 'val_rho_std' in df:
        s['D2_rho_std_last50'] = float(df['val_rho_std'].iloc[-50:].mean())
        s['D2_pass'] = s['D2_rho_std_last50'] > 0.05
        s['ma_mean_last50'] = float(df['val_ma_mean'].iloc[-50:].mean()) if 'val_ma_mean' in df else None
    norm_cols = [c for c in df.columns if c.startswith('norm_route')]
    if norm_cols:
        last = df[norm_cols].iloc[-50:]
        s['D3_max_norm'] = float(last.max().max())
        s['D3_pass'] = (not s['nan']) and math.isfinite(s['D3_max_norm']) and s['D3_max_norm'] < 10
        s['norms_final'] = {c.replace('norm_', ''): round(float(df[c].iloc[-1]), 3) for c in norm_cols}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/data/kimi_repro/outputs/ukan_v2')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    runs = sorted(glob.glob(os.path.join(a.out, '*')))
    S = {}
    for r in runs:
        s = summarise(r)
        if s:
            S[s['run']] = s
    if not S:
        print('no runs with log.csv under', a.out); return
    print(f"{'run':32s} {'ep':>4s} {'best':>6s} {'@ep':>4s} {'final':>6s} {'D1':>4s} {'D2':>4s} {'D3':>4s}  done")
    for k, s in S.items():
        f = lambda key: ('ok' if s[key] else 'FAIL') if key in s else '-'
        print(f"{k:32s} {s['epochs']:4d} {s['best_val_iou']:6.2f} {s['best_epoch']:4d} {s['final_val_iou']:6.2f} "
              f"{f('D1_pass'):>4s} {f('D2_pass'):>4s} {f('D3_pass'):>4s}  {'Y' if s['done'] else 'n'}")
    print()
    # G1-G4 (full config, split 2981)
    for ds, thr in GATES.items():
        k = f'{ds}_v1_full_2981'
        if k in S:
            v = S[k]['best_val_iou']
            print(f"G[{ds:6s}] full={v:6.2f}  gate>={thr:5.2f}  base={BASE_2981[ds]:5.2f}  oldDG={DG_2981[ds]:5.2f}  "
                  f"-> {'PASS' if v >= thr else 'fail'}  (vs base {v - BASE_2981[ds]:+.2f}, vs oldDG {v - DG_2981[ds]:+.2f})"
                  + ('' if S[k]['done'] else '  [running]'))
    # G5 ladder on GlaS
    lad = ['glas_v1_base_2981', 'glas_v1_ctrl_2981', 'glas_v1_ctrl_fusion_2981', 'glas_v1_full_2981']
    vals = [S[k]['best_val_iou'] if k in S else None for k in lad]
    if all(v is not None for v in vals):
        mono = all(vals[i + 1] >= vals[i] - 0.2 for i in range(3))
        print('G5 ladder GlaS:', ' -> '.join(f'{v:.2f}' for v in vals), '=>', 'PASS' if mono else 'fail (non-monotone > 0.2)')
    else:
        print('G5 ladder GlaS (partial):', ' -> '.join('--' if v is None else f'{v:.2f}' for v in vals))
    # U1/U2 abstention usefulness
    full = S.get('glas_v1_full_2981', {}).get('best_val_iou')
    for tag, name in (('U1', 'glas_v1_abst_margin_2981'), ('U2', 'glas_v1_abst_none_2981')):
        if full is not None and name in S:
            d = full - S[name]['best_val_iou']
            print(f"{tag} full - {name.split('_v1_')[1]}: {d:+.2f} => {'PASS' if d >= 0.2 else 'fail'}")
    for name in ('glas_v1_nodetail_2981', 'glas_v1_focus_gt_2981', 'glas_v1_ctrl_obj_2981'):
        if full is not None and name in S:
            print(f"side  full - {name.split('_v1_')[1]}: {full - S[name]['best_val_iou']:+.2f}")
    if a.json:
        if os.path.exists(a.json):
            raise SystemExit(f'refusing to overwrite {a.json}')
        with open(a.json, 'w') as f:
            json.dump(S, f, indent=2)
        print('wrote', a.json)


if __name__ == '__main__':
    main()
