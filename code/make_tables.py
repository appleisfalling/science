# -*- coding: utf-8 -*-
"""Build the LaTeX result tables for the JBHI v2 paper from cached run directories (read-only).

Usage:
  python make_tables.py --cache /Users/shiyayong/ukan_dg_work/results_cache \
                        --tex_dir "/Users/shiyayong/Documents/ChatGPT/专利/U-KAN-DG_JBHI/tables_v2"

Inputs (per run dir <cache>/{ukan,ukan_dg,ukan_v2}/<run>/):
  log.csv           training log (val_iou, val_dice, ..., dynamics columns for v2 runs)
  eval_metrics.json optional; produced by eval_ckpt.py --run <dir>  (per-image IoU/Dice/HD95/ASSD ...)

Run-name conventions
  baseline U-KAN      {ds}_UKAN_s{seed}   (busi/cvc/glas)   |  kvasir_UKANoff_s{seed}
  old U-KAN-DG        {ds}_A4full_s{seed}                    (internal reference only, not in the paper)
  ACRE-C (v2)         {ds}_v1_{variant}_{seed}   variant in {full, base, ctrl, ctrl_fusion, abst_margin,
                      abst_none, nodetail, focus_gt, ctrl_obj, ...}

Outputs (written only into --tex_dir, which is created if absent; existing files are backed up to *.bak):
  tab_main.tex        4 datasets x {U-KAN, ACRE-C}: IoU / Dice mean±std over the seeds available
  tab_perseed.tex     per-seed best-val IoU for U-KAN -> ACRE-C
  tab_boundary.tex    HD95 / ASSD / BF2 from eval_metrics.json (only rows where both models have it)
  tab_ladder.tex      GlaS split-2981 ladder + side ablations (abstention policy, detail trust, focus, loss-only)
  tab_dynamics.tex    D1-D3 polarity-drift / abstention / norm monitors for every v2 run with a controller
  numbers.json        every number used, for the text (\TODO fill-in)
Missing runs are rendered as \TODO{...} cells, so the script can be run before Round 2 finishes.
"""
import argparse, glob, json, os, re, shutil, statistics as st

import pandas as pd

DATASETS = ['busi', 'glas', 'cvc', 'kvasir']
DS_NAME = {'busi': 'BUSI', 'glas': 'GlaS', 'cvc': 'CVC-ClinicDB', 'kvasir': 'Kvasir-SEG'}
SEEDS = [2981, 6142, 1187]
LADDER = [('base', 'U-KAN (all switches off)'), ('ctrl', '+ commit controller, $\\mathcal{L}_{\\mathrm{ctrl}}$'),
          ('ctrl_fusion', '+ consensus routers (local)'), ('full', '+ image-wide consensus, $\\mathcal{L}_{\\mathrm{foc}}$ (full)')]
SIDE = [('abst_margin', 'abstention fixed: \\emph{margin} ($\\rho\\equiv1$)'),
        ('abst_none', 'abstention fixed: \\emph{none} ($\\rho\\equiv0$)'),
        ('nodetail', 'full without detail trust (Step~1)'),
        ('focus_gt', 'focus weight $z_{\\mathrm{gt}}$ only (\\emph{focus-gt})'),
        ('ctrl_obj', 'both losses, no routers (\\emph{loss-only})')]


def todo(s):
    return '\\TODO{%s}' % str(s).replace('_', '\\_')


def fmt(v, nd=2):
    return '--' if v is None else ('%.*f' % (nd, v))


def mean_std(vals, nd=2):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return '%.*f' % (nd, vals[0])
    # population std over the seeds, as in the mainline draft (sections/04) so the U-KAN row matches it
    return '%.*f$\\,\\pm\\,$%.*f' % (nd, st.mean(vals), nd, st.stdev(vals))


# ----------------------------------------------------------------------------- loading
def load_run(d):
    csv = os.path.join(d, 'log.csv')
    if not os.path.isfile(csv):
        return None
    df = pd.read_csv(csv)
    if len(df) == 0:
        return None
    r = {'dir': d, 'name': os.path.basename(d), 'epochs': int(len(df)),
         'done': os.path.isfile(os.path.join(d, 'DONE')) or len(df) >= 400,
         'best_iou': float(df['val_iou'].max() * 100), 'best_epoch': int(df['val_iou'].idxmax()),
         'dice_at_best': float(df['val_dice'].iloc[int(df['val_iou'].idxmax())] * 100),
         'final_iou': float(df['val_iou'].iloc[-1] * 100),
         'final50_iou': float(df['val_iou'].iloc[-50:].mean() * 100)}
    em = os.path.join(d, 'eval_metrics.json')
    if os.path.isfile(em):
        s = json.load(open(em))['summary']
        for k in ('iou', 'dice', 'hd95', 'assd', 'bf2', 'precision', 'recall'):
            if k in s:
                r['em_' + k] = float(s[k]['mean']) * (100 if k in ('iou', 'dice', 'bf2', 'precision', 'recall') else 1)
    if 'val_ctrl_bce' in df:
        n = len(df)
        early = df.iloc[50:100] if n >= 100 else df.iloc[: max(1, n // 2)]
        late = df.iloc[-50:]
        r['D1_ctrl_bce_rise'] = float(late['val_ctrl_bce'].mean() - early['val_ctrl_bce'].mean())
        r['D1_cell_iou_drop'] = float((early['val_cell_iou'].mean() - late['val_cell_iou'].mean()) * 100)
        r['cell_iou_last50'] = float(late['val_cell_iou'].mean() * 100)
        r['D2_rho_std_last50'] = float(late['val_rho_std'].mean())
        r['ma_mean_last50'] = float(late['val_ma_mean'].mean())
        norm_cols = [c for c in df.columns if c.startswith('norm_route')]
        r['D3_max_norm'] = float(df[norm_cols].iloc[-50:].max().max()) if norm_cols else None
        r['norms_final'] = {c.replace('norm_', ''): float(df[c].iloc[-1]) for c in norm_cols}
    return r


def parse_name(name):
    m = re.match(r'^(busi|glas|cvc|kvasir)_(UKAN|UKANoff)_s(\d+)$', name)
    if m:
        return m.group(1), 'ukan', int(m.group(3))
    m = re.match(r'^(busi|glas|cvc|kvasir)_A4full_s(\d+)$', name)
    if m:
        return m.group(1), 'dg_old', int(m.group(2))
    m = re.match(r'^(busi|glas|cvc|kvasir)_v\d+_(.+)_(\d{4})$', name)
    if m:
        return m.group(1), 'v2:' + m.group(2), int(m.group(3))
    return None


def load_all(cache):
    runs = {}
    for sub in ('ukan', 'ukan_dg', 'ukan_v2'):
        for d in sorted(glob.glob(os.path.join(cache, sub, '*'))):
            p = parse_name(os.path.basename(d))
            if not p:
                continue
            r = load_run(d)
            if r:
                runs[p] = r
    return runs


# ----------------------------------------------------------------------------- tables
def get(runs, ds, variant, seed, key='best_iou', require_done=True):
    r = runs.get((ds, variant, seed))
    if r is None or (require_done and not r['done']):
        return None
    return r.get(key)


def tab_main(runs, nums):
    L = ['\\begin{table*}[!t]', '\\caption{Validation Results Under the Frozen Protocol (\\%, Mean$\\pm$Standard Deviation Over Three Data Seeds). Left: Best-Validation-IoU Checkpoint. Right: Mean IoU Over the Last 50 Epochs, Without Checkpoint Selection.}',
         '\\label{tab:main}', '\\centering\\footnotesize', '\\begin{threeparttable}', '\\setlength{\\tabcolsep}{4.5pt}',
         '\\begin{tabular}{@{}lccc ccc ccc@{}}', '\\toprule',
         'Dataset & \\multicolumn{3}{c}{IoU (best)} & \\multicolumn{3}{c}{Dice (at best IoU)} & \\multicolumn{3}{c}{IoU (last 50 epochs)} \\\\',
         '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}',
         ' & U-KAN & + ACRE-C & $\\Delta$ & U-KAN & + ACRE-C & $\\Delta$ & U-KAN & + ACRE-C & $\\Delta$ \\\\', '\\midrule']
    all_d = {'best_iou': [], 'dice_at_best': [], 'final50_iou': []}
    for ds in DATASETS:
        cells = []
        for key in ('best_iou', 'dice_at_best', 'final50_iou'):
            a = [get(runs, ds, 'ukan', s, key) for s in SEEDS]
            b = [get(runs, ds, 'v2:full', s, key) for s in SEEDS]
            na, nb = len([v for v in a if v is not None]), len([v for v in b if v is not None])
            nums['%s_%s' % (ds, key)] = {'ukan': a, 'acre': b}
            cells.append(mean_std(a) or todo(''))
            cells.append((mean_std(b) or todo('')) + ('$^{%d}$' % nb if 0 < nb < 3 else ''))
            if na and nb:
                d = st.mean([v for v in b if v is not None]) - st.mean([v for v in a if v is not None])
                nums['delta_%s_%s' % (ds, key)] = d
                all_d[key].append(d)
                cells.append('$%+.2f$' % d)
            else:
                cells.append(todo(''))
        L.append('%s & %s \\\\' % (DS_NAME[ds], ' & '.join(cells)))
    L.append('\\midrule')
    cells = []
    for key in ('best_iou', 'dice_at_best', 'final50_iou'):
        cells += ['', '', ('$%+.2f$' % st.mean(all_d[key])) if len(all_d[key]) == 4 else todo('')]
    L.append('Mean $\\Delta$ over datasets & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\begin{tablenotes}[flushleft]\\footnotesize',
          '\\item All runs: $256\\times256$, batch 8, Adam $10^{-4}$, cosine schedule, 400 epochs, identical split and mini-batch order for both models within a seed. Parameters: U-KAN 6.36\\,M, + ACRE-C 6.75\\,M. A superscript marks the number of seeds when fewer than three were available.',
          '\\end{tablenotes}', '\\end{threeparttable}', '\\end{table*}']
    return '\n'.join(L) + '\n'


def tab_perseed(runs, nums):
    L = ['\\begin{table}[!t]', '\\caption{Per-Seed Best Validation IoU (\\%): U-KAN $\\rightarrow$ U-KAN + ACRE-C}',
         '\\label{tab:perseed}', '\\centering\\footnotesize', '\\begin{tabular}{@{}lccc@{}}', '\\toprule',
         'Dataset & ' + ' & '.join('Seed %d' % s for s in SEEDS) + ' \\\\', '\\midrule']
    for ds in DATASETS:
        cells = []
        for s in SEEDS:
            a, b = get(runs, ds, 'ukan', s), get(runs, ds, 'v2:full', s)
            if a is None and b is None:
                cells.append(todo(''))
            else:
                cells.append('%s$\\,\\rightarrow\\,$%s' % (fmt(a), fmt(b) if b is not None else todo('')))
        L.append('%s & %s \\\\' % (DS_NAME[ds], ' & '.join(cells)))
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return '\n'.join(L) + '\n'


def tab_boundary(runs, nums):
    L = ['\\begin{table}[!t]', '\\caption{Boundary Metrics From the Best Checkpoints (Mean Over Seeds; HD95 and ASSD in Pixels at $256\\times256$, BF$_2$ in \\%)}',
         '\\label{tab:boundary}', '\\centering\\footnotesize', '\\begin{tabular}{@{}llccc@{}}', '\\toprule',
         'Dataset & Method & HD95 $\\downarrow$ & ASSD $\\downarrow$ & BF$_2$ $\\uparrow$ \\\\', '\\midrule']
    for ds in DATASETS:
        for variant, label in (('ukan', 'U-KAN'), ('v2:full', '+ ACRE-C')):
            hd = [get(runs, ds, variant, s, 'em_hd95') for s in SEEDS]
            asd = [get(runs, ds, variant, s, 'em_assd') for s in SEEDS]
            bf = [get(runs, ds, variant, s, 'em_bf2') for s in SEEDS]
            nums['%s_%s_hd95' % (variant, ds)] = hd
            L.append('%s & %s & %s & %s & %s \\\\' % (DS_NAME[ds] if variant == 'ukan' else '', label,
                                                      mean_std(hd) or todo(''), mean_std(asd) or todo(''), mean_std(bf) or todo('')))
        if ds != DATASETS[-1]:
            L.append('\\addlinespace[2pt]')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return '\n'.join(L) + '\n'


def tab_ladder(runs, nums):
    L = ['\\begin{table}[!t]', '\\caption{Component Ladder and Side Ablations on GlaS (Split 2981, 400 Epochs, Validation IoU/Dice at the Best Epoch and Mean IoU over the Last 50 Epochs, \\%)}',
         '\\label{tab:ablation}', '\\centering\\footnotesize', '\\setlength{\\tabcolsep}{3.0pt}', '\\begin{tabular}{@{}lcccc@{}}', '\\toprule',
         'Configuration & IoU & Dice & $\\Delta$IoU & IoU$_{50}$ \\\\', '\\midrule']
    full = get(runs, 'glas', 'v2:full', 2981)
    nums['ladder'] = {}

    def row(var, label):
        v = get(runs, 'glas', 'v2:' + var, 2981)
        d = get(runs, 'glas', 'v2:' + var, 2981, 'dice_at_best')
        f = get(runs, 'glas', 'v2:' + var, 2981, 'final50_iou')
        nums['ladder'][var] = {'best_iou': v, 'dice_at_best': d, 'final50_iou': f}
        delta = ('%+.2f' % (v - full)) if (v is not None and full is not None) else '--'
        return '%s & %s & %s & %s & %s \\\\' % (label, fmt(v) if v is not None else todo(var), fmt(d) if d is not None else '',
                                            delta, fmt(f) if f is not None else '')
    for var, label in LADDER:
        L.append(row(var, label))
    L.append('\\midrule')
    for var, label in SIDE:
        L.append(row(var, label))
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return '\n'.join(L) + '\n'


def tab_dynamics(runs, nums):
    L = ['\\begin{table}[!t]', '\\caption{Controller Dynamics (Validation, Last-50-Epoch Means). $\\Delta$BCE / $\\Delta$IoU: Change of the Cell-Level Polarity BCE / IoU From Epochs 50--100 to the Last 50.}',
         '\\label{tab:dynamics}', '\\centering\\scriptsize', '\\setlength{\\tabcolsep}{3pt}', '\\begin{tabular}{@{}lcccccc@{}}', '\\toprule',
         'Run & Cell IoU & $\\Delta$BCE & $\\Delta$IoU & $\\bar{M}_a$ & std$(\\rho)$ & max $\\lVert W\\rVert$ \\\\', '\\midrule']
    rows = []
    keep = {'full', 'abst_margin', 'abst_none', 'ctrl'}
    for (ds, var, seed), r in sorted(runs.items(), key=lambda kv: (DATASETS.index(kv[0][0]), kv[0][1], kv[0][2])):
        if not var.startswith('v2:') or 'D1_ctrl_bce_rise' not in r or var in ('v2:base',):
            continue
        if var[3:] not in keep:
            continue
        name = '%s / %s / %d' % ({'busi': 'BUSI', 'glas': 'GlaS', 'cvc': 'CVC', 'kvasir': 'Kvasir'}[ds], var[3:].replace('_', '\\_'), seed)
        rows.append('%s & %.1f & %+.3f & %+.2f & %.2f & %.3f & %s%s \\\\' % (
            name, r['cell_iou_last50'], r['D1_ctrl_bce_rise'], -r['D1_cell_iou_drop'], r['ma_mean_last50'],
            r['D2_rho_std_last50'], ('%.2f' % r['D3_max_norm']) if r['D3_max_norm'] is not None else '--',
            '' if r['done'] else '$^{\\dagger}$'))
        nums.setdefault('dynamics', {})[r['name']] = {k: r[k] for k in ('cell_iou_last50', 'D1_ctrl_bce_rise', 'D1_cell_iou_drop', 'ma_mean_last50', 'D2_rho_std_last50', 'D3_max_norm', 'done')}
    L += rows or ['\\multicolumn{7}{c}{%s} \\\\' % todo('no v2 runs with controller logs yet')]
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return '\n'.join(L) + '\n'


def write(path, text):
    if os.path.exists(path):
        shutil.copy2(path, path + '.bak')
    with open(path, 'w') as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/Users/shiyayong/ukan_dg_work/results_cache')
    ap.add_argument('--tex_dir', default='/Users/shiyayong/Documents/ChatGPT/专利/U-KAN-DG_JBHI/tables_v2')
    a = ap.parse_args()
    runs = load_all(a.cache)
    print('loaded %d runs' % len(runs))
    for k in sorted(runs, key=lambda k: (DATASETS.index(k[0]), k[1], k[2])):
        r = runs[k]
        print('  %-28s ep=%3d done=%d best=%.2f@%d final50=%.2f' % (r['name'], r['epochs'], r['done'], r['best_iou'], r['best_epoch'], r['final50_iou']))
    os.makedirs(a.tex_dir, exist_ok=True)
    nums = {}
    for fn, f in (('tab_main.tex', tab_main), ('tab_perseed.tex', tab_perseed), ('tab_boundary.tex', tab_boundary),
                  ('tab_ladder.tex', tab_ladder), ('tab_dynamics.tex', tab_dynamics)):
        write(os.path.join(a.tex_dir, fn), f(runs, nums))
    write(os.path.join(a.tex_dir, 'numbers.json'), json.dumps(nums, indent=1, default=str))
    print('tables written to', a.tex_dir)


if __name__ == '__main__':
    main()
