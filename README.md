# ACRE-C: Commit-or-Abstain Consensus Routing of Skip Evidence for Medical Image Segmentation

Code, complete training logs, and evaluation outputs for the paper
**"Commit or Abstain: Consensus Routing of Skip Evidence for Medical Image Segmentation"** (manuscript).

ACRE-C treats encoder--decoder skip fusion as a vote: a controller formed once on the deepest
convolutional feature assigns each cell a supervised polarity and an abstention with no target;
commitment-weighted normalised convolution forms foreground/background consensuses at every decoder
scale, and the abstention routes evidence to one of two zero-initialised experts. The backbone
(U-KAN) is unchanged; the router adds 0.39M parameters (+16% MACs).

## Layout
- `code/` — training / evaluation / analysis code (`train_v2.py`, `archs_v2.py`, `losses_v2.py`,
  `eval_ckpt.py`, `gate_check.py`, `viz_v2.py`, `split_p2.py`, queue runner, and all queue files).
  Python environment: PyTorch 2.11, CUDA 12.8.
- `results/ukan_v2/` — one directory per run: `config.yml`, per-epoch `log.csv`, `eval_metrics.json`
  (validation partition), and for the 80/10/10 `p2` protocol runs `eval_test_metrics.json` (test
  partition, never used for training or model selection) plus the auditable `split_ids_p2.txt`.
- `results/baselines_ukan/`, `results/baselines_kvasir_ukanoff/` — re-trained U-KAN baseline logs.
- `results/viz/` — per-image qualitative dumps (probability/error/abstention maps, `.npz`).
- `results/tables/` — auto-generated LaTeX tables and `numbers.json` used in the paper.
- `paper/` — LaTeX source of the manuscript.

## Protocols
- `p1` — the backbone's public protocol: random 80/20 train/validation split per data seed
  (2981 / 6142 / 1187); 400 epochs, batch 8, Adam 1e-4, best-validation-IoU checkpoint, plus the
  last-50-epoch mean as a selection-free check.
- `p2` — 80/10/10: the outer 80/20 boundary is identical to `p1` (same training set); the held-out
  20% is halved with a derived seed into validation (model selection) and test (reporting only).

## Datasets
The four public datasets (BUSI, GlaS, CVC-ClinicDB, Kvasir-SEG) are **not** redistributed here;
download them from their original sources and arrange under `inputs/<dataset>/images` and
`inputs/<dataset>/masks/0` (BUSI mask files carry a `_mask` suffix).

## Reproducing
`code/test_unit_v2.py` verifies the identity-at-initialisation property of the router
(maximum deviation from U-KAN with identical weights < 1e-7). A full run:
`python train_v2.py --name <run> --dataset <ds> --dataseed 2981 --split p2 --use_ctrl --use_fusion --use_context --use_objective`

## License
Code: MIT. Third-party dataset licenses apply to the datasets referenced above.
