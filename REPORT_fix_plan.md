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
- Standardized kappa/w losses and added `lambda_kappa` in `project/src/losses/losses.py`.
- Added curriculum scheduling for `lambda_kappa`/`lambda_hf` in `project/src/train/trainer.py`.
- Updated config defaults with `stats_path`, `loss.lambda_kappa`, and `loss.lambda_hf=0` in `project/configs/train.yaml`.
- Added collapse verification script: `tools_diagnose/verify_collapse_fix.py`.

## Why This Fixes Collapse
- Standardizing kappa/w by global std prevents scale dominance (kappa/hf overpowering w).
- `lambda_kappa` decouples kappa loss weight from the default 1.0.
- Default `lambda_hf=0` removes high-frequency suppression early on.
- Curriculum keeps Stage A focused on w-supervision, then ramps kappa/hf to avoid early collapse.

## How To Run
1) Compute stats:
   `python project/scripts/compute_dataset_stats.py --config project/configs/train.yaml`
2) Train (Stage A → Stage B via curriculum):
   `python project/scripts/train.py --config project/configs/train.yaml`
   - Set `curriculum.enabled=true`, `curriculum.stageA_steps`, `curriculum.stageB_warmup_steps` in `project/configs/train.yaml`.
3) Verify:
   `python tools_diagnose/verify_collapse_fix.py --checkpoint runs/<run_name>/checkpoint.pt --device cuda`
   PYTHONPATH=../project python ../tools_diagnose/verify_collapse_fix.py --checkpoint runs/20260115-152854/checkpoint.pt --device cuda

