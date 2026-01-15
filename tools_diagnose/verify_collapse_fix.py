#!/usr/bin/env python
"""
Verify collapse mitigation by comparing two samples and loss breakdown.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Add project root + src to path for direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "project"))
sys.path.insert(0, str(ROOT_DIR / "project" / "src"))

from datasets import FiberSequenceDataset, fiber_sequence_collate
from models import MambaCoeffNet
from train import Trainer
from utils import setup_logger

logger = setup_logger("verify_collapse_fix")


def _masked_values(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.reshape(-1)
    mask_f = mask > 0
    if values.ndim == mask_f.ndim + 1:
        mask_f = mask_f.unsqueeze(-1).expand_as(values)
    return values[mask_f]


def _masked_stats(values: torch.Tensor, mask: torch.Tensor | None) -> dict[str, float]:
    vals = _masked_values(values, mask)
    if vals.numel() == 0:
        return {"abs_max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "abs_max": float(vals.abs().max().item()),
        "mean": float(vals.mean().item()),
        "std": float(vals.std(unbiased=False).item()),
    }


def _masked_min_max(values: torch.Tensor, mask: torch.Tensor | None) -> tuple[float, float]:
    vals = _masked_values(values, mask)
    if vals.numel() == 0:
        return float("nan"), float("nan")
    return float(vals.min().item()), float(vals.max().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify collapse fix diagnostics")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu|cuda|auto")
    parser.add_argument("--global_step", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="diagnostic_logs")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(config_device: Any, override: str | None) -> torch.device:
    device_value = override if override is not None else config_device
    if isinstance(device_value, dict):
        device_value = device_value.get("type", "auto")
    device_value = device_value or "auto"
    if device_value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_value)


def build_model(config: dict[str, Any]) -> MambaCoeffNet:
    model_cfg = config.get("model", {})
    basis_cfg = config.get("basis", {})
    out_coeffs = int(model_cfg.get("out_coeffs", basis_cfg.get("M") * basis_cfg.get("N")))
    model_cfg["out_coeffs"] = out_coeffs
    return MambaCoeffNet(
        in_features=int(model_cfg.get("in_features", 5)),
        d_model=int(model_cfg.get("d_model", 128)),
        n_layers=int(model_cfg.get("n_layers", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        out_coeffs=out_coeffs,
        encoder_type=str(model_cfg.get("encoder_type", "auto")),
    )


def find_latest_checkpoint(run_root: Path) -> Path:
    candidates = list(run_root.rglob("checkpoint.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {run_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> int:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint(Path("runs"))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config_path = (
        Path(args.config)
        if args.config is not None
        else checkpoint_path.parent / "config.yaml"
    )
    config = load_config(config_path)

    if args.manifest is not None:
        config.setdefault("data", {})["manifest"] = args.manifest
    if args.data_dir is not None:
        config.setdefault("data", {})["samples_dir"] = args.data_dir
    if args.device is not None:
        config["device"] = args.device

    dataset = FiberSequenceDataset(
        manifest_path=config.get("data", {}).get("manifest"),
        samples_dir=config.get("data", {}).get("samples_dir"),
        df_list=config.get("data", {}).get("df_list") if config.get("data", {}).get("use_dataframe") else None,
        stats_path=config.get("stats_path"),
        coord_scale=float(config.get("data", {}).get("coord_scale", 1.0)),
        kappa_meas_scale=float(config.get("data", {}).get("kappa_meas_scale", 1.0)),
    )
    if len(dataset) < 2:
        raise SystemExit("Dataset must contain at least 2 samples.")
    device = resolve_device(config.get("device", "auto"), args.device)
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer, config, device)
    checkpoint = trainer.load_checkpoint(checkpoint_path)
    global_step_source = "checkpoint"
    if args.global_step is not None:
        trainer.global_step = int(args.global_step)
        global_step_source = "arg"
    elif "global_step" not in checkpoint:
        global_step_source = "default"

    rng = np.random.default_rng()
    indices = rng.choice(len(dataset), size=2, replace=False)
    samples = [dataset[int(i)] for i in indices]
    batch = fiber_sequence_collate(samples)
    batch = trainer._move_batch(batch)
    outputs = trainer._forward(batch)
    loss_dict = trainer._step_loss(batch, outputs)

    a_pred = outputs["a"].detach().cpu().numpy()
    a1 = a_pred[0].reshape(-1)
    a2 = a_pred[1].reshape(-1)
    a_diff = float(np.max(np.abs(a1 - a2)))

    x1 = batch["X"][0]
    x2 = batch["X"][1]
    x_diff = float((x1 - x2).abs().max().item())
    kappa_in_1 = batch["X"][0, :, 4]
    kappa_in_2 = batch["X"][1, :, 4]
    kappa_in_diff = float((kappa_in_1 - kappa_in_2).abs().max().item())
    kappa_true_1 = batch.get("kappa_meas", batch["X"][:, :, 4])[0]
    kappa_true_2 = batch.get("kappa_meas", batch["X"][:, :, 4])[1]
    kappa_true_diff = float((kappa_true_1 - kappa_true_2).abs().max().item())

    mask = batch.get("mask")
    mask_union = None
    if mask is not None:
        mask_union = (mask[0] > 0) | (mask[1] > 0)

    if mask_union is None:
        masked_x_diff = x_diff
        masked_kappa_in_diff = kappa_in_diff
    else:
        x_masked_vals = (x1 - x2).abs()[mask_union]
        masked_x_diff = float(x_masked_vals.max().item()) if x_masked_vals.numel() else float("nan")
        kappa_masked_vals = (kappa_in_1 - kappa_in_2).abs()[mask_union]
        masked_kappa_in_diff = (
            float(kappa_masked_vals.max().item()) if kappa_masked_vals.numel() else float("nan")
        )

    x_vals = batch.get("x", batch["X"][..., 0])
    y_vals = batch.get("y", batch["X"][..., 1])
    x_min, x_max = _masked_min_max(x_vals, mask)
    y_min, y_max = _masked_min_max(y_vals, mask)
    x_range = x_max - x_min if np.isfinite(x_min) and np.isfinite(x_max) else float("nan")
    y_range = y_max - y_min if np.isfinite(y_min) and np.isfinite(y_max) else float("nan")
    x_range_ratio = x_range / float(trainer.Lx) if trainer.Lx != 0 else float("nan")
    y_range_ratio = y_range / float(trainer.Ly) if trainer.Ly != 0 else float("nan")

    kappa_meas = batch.get("kappa_meas", batch["X"][..., 4])
    kappa_pred = outputs["kappa_pred"]
    kappa_meas_stats = _masked_stats(kappa_meas, mask)
    kappa_pred_stats = _masked_stats(kappa_pred, mask)
    ratio_eps = 1e-12
    scale_ratio_std = kappa_pred_stats["std"] / (kappa_meas_stats["std"] + ratio_eps)
    scale_ratio_max = kappa_pred_stats["abs_max"] / (kappa_meas_stats["abs_max"] + ratio_eps)

    kappa_mean_t = torch.as_tensor(
        batch.get("kappa_mean", float("nan")),
        device=kappa_pred.device,
        dtype=kappa_pred.dtype,
    )
    kappa_std_t = torch.as_tensor(
        batch.get("kappa_std", float("nan")),
        device=kappa_pred.device,
        dtype=kappa_pred.dtype,
    )
    eps_t = torch.tensor(1e-12, device=kappa_pred.device, dtype=kappa_pred.dtype)
    kappa_pred_hat = (kappa_pred - kappa_mean_t) / (kappa_std_t + eps_t)
    kappa_meas_hat = (kappa_meas - kappa_mean_t) / (kappa_std_t + eps_t)
    kappa_pred_hat_stats = _masked_stats(kappa_pred_hat, mask)
    kappa_meas_hat_stats = _masked_stats(kappa_meas_hat, mask)

    w_ratio = None
    if "w_points" in batch and outputs.get("w_pred_points") is not None:
        w_pred = outputs["w_pred_points"].detach().cpu().numpy()
        w_true = batch["w_points"].detach().cpu().numpy()
        max_true = float(np.max(np.abs(w_true)))
        max_pred = float(np.max(np.abs(w_pred)))
        w_ratio = float(max_pred / max_true) if max_true > 0 else float("inf")

    total_loss = float(loss_dict["loss"].item())
    loss_kappa = float(loss_dict.get("loss_kappa", torch.tensor(0.0)).item())
    loss_w = float(loss_dict.get("loss_w", torch.tensor(0.0)).item()) if "loss_w" in loss_dict else None
    loss_hf = float(loss_dict.get("loss_hf", torch.tensor(0.0)).item())

    raw_loss_kappa = float(loss_dict.get("raw_loss_kappa", torch.tensor(0.0)).item())
    raw_loss_w = (
        float(loss_dict.get("raw_loss_w", torch.tensor(0.0)).item()) if "raw_loss_w" in loss_dict else None
    )
    raw_loss_hf = float(loss_dict.get("raw_loss_hf", torch.tensor(0.0)).item())
    weighted_loss_kappa = float(loss_dict.get("weighted_loss_kappa", torch.tensor(0.0)).item())
    weighted_loss_w = (
        float(loss_dict.get("weighted_loss_w", torch.tensor(0.0)).item())
        if "weighted_loss_w" in loss_dict
        else None
    )
    weighted_loss_hf = float(loss_dict.get("weighted_loss_hf", torch.tensor(0.0)).item())

    weighted_total = weighted_loss_kappa + weighted_loss_hf
    if weighted_loss_w is not None:
        weighted_total += weighted_loss_w
    ratios = {
        "weighted_loss_kappa": weighted_loss_kappa / weighted_total if weighted_total > 0 else float("nan"),
        "weighted_loss_hf": weighted_loss_hf / weighted_total if weighted_total > 0 else float("nan"),
    }
    if weighted_loss_w is not None:
        ratios["weighted_loss_w"] = weighted_loss_w / weighted_total if weighted_total > 0 else float("nan")

    lambda_kappa, lambda_hf = trainer._scheduled_lambdas()
    schedule_progress = 1.0
    if trainer.curriculum_enabled:
        if trainer.global_step < trainer.stageA_steps:
            schedule_progress = 0.0
        elif trainer.stageB_warmup_steps <= 0:
            schedule_progress = 1.0
        else:
            schedule_progress = (
                (trainer.global_step - trainer.stageA_steps + 1)
                / float(trainer.stageB_warmup_steps)
            )
            schedule_progress = max(0.0, min(1.0, schedule_progress))

    if mask is None:
        mask_sum = float(batch["X"].shape[0] * batch["X"].shape[1])
        kappa_vals = batch["X"][..., 4].reshape(-1)
    else:
        mask_f = mask > 0
        mask_sum = float(mask_f.sum().item())
        kappa_vals = batch["X"][..., 4][mask_f]
    kappa_in_mean = float(kappa_vals.mean().item()) if kappa_vals.numel() else float("nan")
    kappa_in_std = float(kappa_vals.std(unbiased=False).item()) if kappa_vals.numel() else float("nan")

    sample_ids = batch.get("sample_id")
    if sample_ids is not None and len(set(sample_ids)) < 2:
        raise ValueError(f"Expected distinct sample_ids, got {sample_ids}")
    coord_scale = float(config.get("data", {}).get("coord_scale", 1.0))
    kappa_meas_scale = float(config.get("data", {}).get("kappa_meas_scale", 1.0))
    unit_summary_lines = [
        f"coord_scale={coord_scale:g}, kappa_meas_scale={kappa_meas_scale:g}",
        f"x_range/Lx={x_range_ratio:.4f}, y_range/Ly={y_range_ratio:.4f}",
        f"scale_ratio_std={scale_ratio_std:.3e}, scale_ratio_max={scale_ratio_max:.3e}",
    ]
    unit_warnings = []
    if np.isfinite(x_range_ratio) and (x_range_ratio > 2.0 or x_range_ratio < 0.5):
        unit_warnings.append(
            f"WARNING: x_range/Lx={x_range_ratio:.3f} far from 1; possible unit mismatch."
        )
    if np.isfinite(y_range_ratio) and (y_range_ratio > 2.0 or y_range_ratio < 0.5):
        unit_warnings.append(
            f"WARNING: y_range/Ly={y_range_ratio:.3f} far from 1; possible unit mismatch."
        )
    if scale_ratio_std > 1e3 or scale_ratio_std < 1e-3:
        unit_warnings.append(
            "WARNING: kappa_pred vs kappa_meas scale differs by >1e3; "
            "check x/y vs Lx/Ly units (kx=m*pi/Lx, d2 includes kx^2) and kappa_meas scaling."
        )
    unit_summary_lines.extend(unit_warnings)
    unit_sanity_summary = "\n".join(unit_summary_lines)

    global_step_warning = None
    if global_step_source == "default":
        global_step_warning = "WARNING: global_step=0 (checkpoint missing); schedule may be incorrect."

    report = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "sample_ids": sample_ids,
        "a_pred_sample1_first10": [float(v) for v in a1[:10]],
        "a_pred_sample2_first10": [float(v) for v in a2[:10]],
        "a_pred_max_abs_diff": a_diff,
        "x_input_max_abs_diff": x_diff,
        "masked_x_input_max_abs_diff": masked_x_diff,
        "kappa_in_max_abs_diff": kappa_in_diff,
        "masked_kappa_in_max_abs_diff": masked_kappa_in_diff,
        "kappa_channel_max_abs_diff": kappa_in_diff,
        "kappa_true_max_abs_diff": kappa_true_diff,
        "kappa_in_mean": kappa_in_mean,
        "kappa_in_std": kappa_in_std,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "x_range_over_Lx": x_range_ratio,
        "y_range_over_Ly": y_range_ratio,
        "kappa_meas_abs_max": kappa_meas_stats["abs_max"],
        "kappa_meas_mean": kappa_meas_stats["mean"],
        "kappa_meas_std": kappa_meas_stats["std"],
        "kappa_pred_abs_max": kappa_pred_stats["abs_max"],
        "kappa_pred_mean": kappa_pred_stats["mean"],
        "kappa_pred_std": kappa_pred_stats["std"],
        "scale_ratio_std": scale_ratio_std,
        "scale_ratio_max": scale_ratio_max,
        "kappa_meas_hat_abs_max": kappa_meas_hat_stats["abs_max"],
        "kappa_meas_hat_mean": kappa_meas_hat_stats["mean"],
        "kappa_meas_hat_std": kappa_meas_hat_stats["std"],
        "kappa_pred_hat_abs_max": kappa_pred_hat_stats["abs_max"],
        "kappa_pred_hat_mean": kappa_pred_hat_stats["mean"],
        "kappa_pred_hat_std": kappa_pred_hat_stats["std"],
        "mask_sum": mask_sum,
        "w_amplitude_ratio": w_ratio,
        "loss_total": total_loss,
        "loss_kappa": loss_kappa,
        "loss_w": loss_w,
        "loss_hf": loss_hf,
        "raw_loss_kappa": raw_loss_kappa,
        "raw_loss_w": raw_loss_w,
        "raw_loss_hf": raw_loss_hf,
        "weighted_loss_kappa": weighted_loss_kappa,
        "weighted_loss_w": weighted_loss_w,
        "weighted_loss_hf": weighted_loss_hf,
        "loss_ratio": ratios,
        "kappa_mean": float(batch.get("kappa_mean", torch.tensor(float("nan")))),
        "kappa_std": float(batch.get("kappa_std", torch.tensor(float("nan")))),
        "w_mean": float(batch.get("w_mean", torch.tensor(float("nan")))),
        "w_std": float(batch.get("w_std", torch.tensor(float("nan")))),
        "global_step": int(trainer.global_step),
        "global_step_source": global_step_source,
        "global_step_warning": global_step_warning,
        "curriculum_enabled": bool(trainer.curriculum_enabled),
        "stageA_steps": int(trainer.stageA_steps),
        "stageB_warmup_steps": int(trainer.stageB_warmup_steps),
        "schedule_progress": float(schedule_progress),
        "lambda_kappa": float(lambda_kappa),
        "lambda_w": float(trainer.lambda_w),
        "lambda_hf": float(lambda_hf),
        "coord_scale": coord_scale,
        "kappa_meas_scale": kappa_meas_scale,
        "unit_sanity_summary": unit_sanity_summary,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "verify_fix.json"
    txt_path = output_dir / "verify_fix.txt"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("Verify collapse fix report\n")
        handle.write(f"checkpoint: {checkpoint_path}\n")
        handle.write(f"sample_ids: {sample_ids}\n")
        handle.write(f"a_pred sample1 first10: {report['a_pred_sample1_first10']}\n")
        handle.write(f"a_pred sample2 first10: {report['a_pred_sample2_first10']}\n")
        handle.write(f"a_pred max|diff|: {a_diff:.6f}\n")
        handle.write(f"X max|diff|: {x_diff:.6f}\n")
        handle.write(f"X masked max|diff|: {masked_x_diff:.6f}\n")
        handle.write(f"kappa_in max|diff|: {kappa_in_diff:.6f}\n")
        handle.write(f"kappa_in masked max|diff|: {masked_kappa_in_diff:.6f}\n")
        handle.write(f"kappa_true max|diff|: {kappa_true_diff:.6f}\n")
        handle.write(f"kappa_in mean/std: {kappa_in_mean:.6f} / {kappa_in_std:.6f}\n")
        handle.write(f"x min/max: {x_min:.6f} / {x_max:.6f}\n")
        handle.write(f"y min/max: {y_min:.6f} / {y_max:.6f}\n")
        handle.write(f"x_range/Lx: {x_range_ratio:.6f}\n")
        handle.write(f"y_range/Ly: {y_range_ratio:.6f}\n")
        handle.write(
            "kappa_meas abs_max/mean/std: "
            f"{kappa_meas_stats['abs_max']:.6f} / {kappa_meas_stats['mean']:.6f} / "
            f"{kappa_meas_stats['std']:.6f}\n"
        )
        handle.write(
            "kappa_pred abs_max/mean/std: "
            f"{kappa_pred_stats['abs_max']:.6f} / {kappa_pred_stats['mean']:.6f} / "
            f"{kappa_pred_stats['std']:.6f}\n"
        )
        handle.write(f"scale_ratio_std: {scale_ratio_std:.6e}\n")
        handle.write(f"scale_ratio_max: {scale_ratio_max:.6e}\n")
        handle.write(
            "kappa_meas_hat abs_max/mean/std: "
            f"{kappa_meas_hat_stats['abs_max']:.6f} / {kappa_meas_hat_stats['mean']:.6f} / "
            f"{kappa_meas_hat_stats['std']:.6f}\n"
        )
        handle.write(
            "kappa_pred_hat abs_max/mean/std: "
            f"{kappa_pred_hat_stats['abs_max']:.6f} / {kappa_pred_hat_stats['mean']:.6f} / "
            f"{kappa_pred_hat_stats['std']:.6f}\n"
        )
        handle.write(f"mask.sum(): {mask_sum:.2f}\n")
        handle.write(f"w amplitude ratio: {w_ratio}\n")
        handle.write(f"global_step: {trainer.global_step} (source={global_step_source})\n")
        if global_step_warning:
            handle.write(f"{global_step_warning}\n")
        handle.write(
            "schedule progress: "
            f"{schedule_progress:.6f} (stageA_steps={trainer.stageA_steps}, "
            f"stageB_warmup_steps={trainer.stageB_warmup_steps})\n"
        )
        handle.write(f"loss total: {total_loss:.6f}\n")
        handle.write(f"loss_kappa: {loss_kappa:.6f}\n")
        handle.write(f"loss_w: {loss_w}\n")
        handle.write(f"loss_hf: {loss_hf:.6f}\n")
        handle.write(f"raw_loss_kappa: {raw_loss_kappa:.6f}\n")
        handle.write(f"raw_loss_w: {raw_loss_w}\n")
        handle.write(f"raw_loss_hf: {raw_loss_hf:.6f}\n")
        handle.write(f"weighted_loss_kappa: {weighted_loss_kappa:.6f}\n")
        handle.write(f"weighted_loss_w: {weighted_loss_w}\n")
        handle.write(f"weighted_loss_hf: {weighted_loss_hf:.6f}\n")
        handle.write(f"loss ratios: {ratios}\n")
        handle.write("unit sanity summary:\n")
        for line in unit_sanity_summary.splitlines():
            handle.write(f"- {line}\n")

    logger.info("Saved diagnostics to %s and %s", json_path, txt_path)
    logger.info("a_pred max|diff|: %.6f", a_diff)
    logger.info("w amplitude ratio: %s", w_ratio)
    logger.info("loss ratios: %s", ratios)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
