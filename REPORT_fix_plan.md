# Collapse Fix Plan Report

## Root Cause Hypothesis
- kappa input scale is too small or masked/normalized away, so the model ignores conditioning and collapses to a near-constant predictor.

## Fixes Implemented
- Global stats script computes dataset-wide `kappa_mean/std` and `w_mean/std`, and fails if std <= 0.
- Dataset enforces stats file presence, standardizes the kappa input channel, and ships means/stds into each batch (no silent fallback).
- Loss is computed in standardized space (mean/std), with Huber delta defined in that space and weighted loss ratios logged regularly.
- Curriculum stays HF off by default; Stage A trains only w, Stage B ramps kappa (and optional HF) via warmup.
- Training diagnostics now log `mask.sum()`, kappa_in mean/std, and batch max|a1-a2|; verify script captures input diffs and loss breakdown.

## Verification Checklist
- `a_pred_max_abs_diff >= 1e-3`
- `w_amplitude_ratio >= 0.3`
- `weighted_loss_kappa : weighted_loss_w` stays within ~1:10 to 10:1
- `mask.sum() > 0` and `kappa_in` mean/std ~ 0/1

## Commands
1) Compute stats:
   `python scripts/compute_dataset_stats.py --config project/configs/train.yaml`
2) Train Stage A (w-only):
   `python project/scripts/train.py --config project/configs/train_stageA.yaml`
3) Train Stage B (ramp kappa):
   `python project/scripts/train.py --config project/configs/train_stageB.yaml`
4) Verify:
   `python tools_diagnose/verify_collapse_fix.py --checkpoint runs/<run_name>/checkpoint.pt`
