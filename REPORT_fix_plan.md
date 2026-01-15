# Collapse Fix Plan Report

## Entry Points
- Training entry: `project/scripts/train.py`
- Trainer: `project/src/train/trainer.py`
- Dataset: `project/src/datasets/fiber_sequence.py`
- Loss: `project/src/losses/losses.py`

## Key Mapping
- kappa_true: `batch["X"][..., 4]`
- w_true: `batch["w_points"]`
- a_pred: `outputs["a"]`

## Changes Made
- Added dataset-wide stats computation script: `project/scripts/compute_dataset_stats.py`.
- Added stats injection into batches (`kappa_std`, `w_std`) in `project/src/datasets/fiber_sequence.py`.
- Enforced required stds and added raw/weighted loss outputs + logging in `project/src/losses/losses.py`.
- Added curriculum scheduling and loss logging in `project/src/train/trainer.py`.
- Updated configs with curriculum defaults, loss logging, and normalized-space Huber delta notes in `project/configs/train.yaml`.
- Added Stage A/B configs: `project/configs/train_stageA.yaml`, `project/configs/train_stageB.yaml`.
- Enhanced diagnostics output in `tools_diagnose/verify_collapse_fix.py`.

## Why loss_w Was Near Zero
- kappa loss was always active and dominated the total (no curriculum warmup).
- std fallback could silently use 1.0, masking true scale and keeping kappa larger.
- w loss used fewer valid points (mask) and smaller raw magnitude, so its weighted share stayed tiny.

## Why This Fix Should Work
- Required stds + asserts guarantee normalization is actually applied.
- Raw vs weighted losses + logging make imbalance visible and debuggable.
- Stage A trains only w first; Stage B ramps kappa (and optionally hf) to avoid early collapse.

## Expected Metrics After Fix
- `a_pred max|diff|` should rise above `1e-3`.
- `loss_w` share should rise above ~10%.
- `max|w_pred|/max|w_true|` should increase toward `>= 0.3`.

## How To Run
1) Compute stats:
   `python project/scripts/compute_dataset_stats.py --config project/configs/train.yaml`
2) Train Stage A (w-only):
   `python project/scripts/train.py --config project/configs/train_stageA.yaml`
3) Train Stage B (ramp kappa/hf):
   `python project/scripts/train.py --config project/configs/train_stageB.yaml`
4) Verify:
   `python tools_diagnose/verify_collapse_fix.py --checkpoint runs/<run_name>/checkpoint.pt --device cuda`
