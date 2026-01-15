#!/usr/bin/env python
"""
Run inference from a checkpoint and save predictions + plots.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import FiberSequenceDataset, fiber_sequence_collate
from models import MambaCoeffNet
from train import Trainer
from utils import setup_logger

logger = setup_logger("predict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict from a fiber checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu|cuda|auto")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_plots", type=int, default=5)
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


def build_dataloader(config: dict[str, Any]) -> tuple[DataLoader, FiberSequenceDataset]:
    data_cfg = config.get("data", {})
    dataset = FiberSequenceDataset(
        manifest_path=data_cfg.get("manifest"),
        samples_dir=data_cfg.get("samples_dir"),
        df_list=data_cfg.get("df_list") if data_cfg.get("use_dataframe") else None,
        stats_path=config.get("stats_path"),
    )
    train_cfg = config.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 8))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=fiber_sequence_collate,
    )
    return dataloader, dataset


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


def _sample_names(dataset: FiberSequenceDataset) -> list[str]:
    if getattr(dataset, "_use_df", False):
        return [f"sample_{i:04d}" for i in range(len(dataset))]
    paths = getattr(dataset, "_paths", [])
    if not paths:
        return [f"sample_{i:04d}" for i in range(len(dataset))]
    return [Path(path).stem for path in paths]


def _save_parquet(path: Path, rows: Iterable[dict]) -> None:
    df = pd.DataFrame(list(rows))
    df.to_parquet(path, index=False)


def _plot_kappa_scatter(df: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        logger.warning("matplotlib unavailable; skip scatter plot")
        return
    if df.empty:
        logger.warning("No rows to plot for kappa scatter")
        return
    plt.figure(figsize=(5, 5))
    plt.scatter(df["kappa_meas"], df["kappa_pred"], s=6, alpha=0.4, edgecolors="none")
    vmin = min(df["kappa_meas"].min(), df["kappa_pred"].min())
    vmax = max(df["kappa_meas"].max(), df["kappa_pred"].max())
    plt.plot([vmin, vmax], [vmin, vmax], color="black", linewidth=1)
    plt.xlabel("kappa_meas")
    plt.ylabel("kappa_pred")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_curve(
    *,
    x: np.ndarray,
    y_meas: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    title: str,
    ylabel: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        logger.warning("matplotlib unavailable; skip curve plot")
        return
    plt.figure(figsize=(6, 3.5))
    plt.plot(x, y_meas, label="meas", linewidth=1.2)
    plt.plot(x, y_pred, label="pred", linewidth=1.2)
    plt.title(title)
    plt.xlabel("index")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_3d_scatter(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    path: Path,
    title: str,
    zlabel: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        logger.warning("matplotlib unavailable; skip 3D plot")
        return
    if x.size == 0:
        logger.warning("No points for 3D plot: %s", path.name)
        return
    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, s=6, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(zlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
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
        config.setdefault("data", {})["samples_dir"] = None
    elif args.data_dir is not None:
        # Force samples_dir to take precedence over any existing manifest.
        config.setdefault("data", {})["manifest"] = None
        config.setdefault("data", {})["samples_dir"] = args.data_dir
    if args.device is not None:
        config["device"] = args.device

    device = resolve_device(config.get("device", "auto"), None)
    dataloader, dataset = build_dataloader(config)
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer, config, device)
    trainer.load_checkpoint(checkpoint_path)

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_names = _sample_names(dataset)
    plot_limit = max(0, int(args.max_plots))

    prediction_rows: list[dict] = []
    coeff_rows: list[dict] = []
    w_rows: list[dict] = []
    w_compare_rows: list[dict] = []
    metrics_rows: list[dict] = []
    total_kappa_sse = 0.0
    total_kappa_count = 0
    total_w_sse = 0.0
    total_w_count = 0
    plot_cache: dict[int, dict[str, np.ndarray]] = {}

    model.eval()
    sample_offset = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_size = batch["X"].shape[0]
            mask = batch.get("mask")
            if mask is None:
                mask = torch.ones_like(batch["X"][..., 0])
            mask = mask > 0

            batch_device = trainer._move_batch(batch)
            outputs = trainer._forward(batch_device)

            w_pred_points = outputs.get("w_pred_points")
            if w_pred_points is None:
                a_reshaped = outputs["a_reshaped"]
                x_for_w = batch_device.get("x", batch_device["X"][..., 0])
                y_for_w = batch_device.get("y", batch_device["X"][..., 1])
                w_pred_points = trainer._compute_w(a_reshaped, x_for_w, y_for_w)

            kappa_pred = outputs["kappa_pred"].detach().cpu().numpy()
            kappa_meas = batch.get("kappa_meas", batch["X"][..., 4]).detach().cpu().numpy()
            x_vals = batch.get("x", batch["X"][..., 0]).detach().cpu().numpy()
            y_vals = batch.get("y", batch["X"][..., 1]).detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy()

            w_true = batch.get("w_points")
            w_pred = w_pred_points.detach().cpu().numpy()
            if w_true is not None:
                w_true = w_true.detach().cpu().numpy()

            coeff = outputs["a"].detach().cpu().numpy()

            for i in range(batch_size):
                sample_idx = sample_offset + i
                sample_name = (
                    sample_names[sample_idx]
                    if sample_idx < len(sample_names)
                    else f"sample_{sample_idx:04d}"
                )

                valid = mask_np[i]
                valid_idx = np.where(valid)[0]
                if valid_idx.size == 0:
                    continue
                point_ids = valid_idx.astype(int)

                for pid in point_ids:
                    prediction_rows.append(
                        {
                            "sample_id": sample_idx,
                            "sample_name": sample_name,
                            "point_id": int(pid),
                            "x": float(x_vals[i, pid]),
                            "y": float(y_vals[i, pid]),
                            "kappa_meas": float(kappa_meas[i, pid]),
                            "kappa_pred": float(kappa_pred[i, pid]),
                        }
                    )

                    if w_pred is not None:
                        row = {
                            "sample_id": sample_idx,
                            "sample_name": sample_name,
                            "point_id": int(pid),
                            "w_pred": float(w_pred[i, pid]),
                        }
                        if w_true is not None:
                            row["w_true"] = float(w_true[i, pid])
                        w_rows.append(row)
                        if w_true is not None:
                            w_p = float(w_pred[i, pid])
                            w_t = float(w_true[i, pid])
                            w_diff = w_p - w_t
                            w_compare_rows.append(
                                {
                                    "sample_id": sample_idx,
                                    "sample_name": sample_name,
                                    "point_id": int(pid),
                                    "w_pred": w_p,
                                    "w_true": w_t,
                                    "w_diff": w_diff,
                                    "w_abs_error": abs(w_diff),
                                    "w_sq_error": w_diff * w_diff,
                                }
                            )

                kappa_diff = kappa_pred[i, valid] - kappa_meas[i, valid]
                kappa_mse = float(np.mean(kappa_diff**2)) if kappa_diff.size else float("nan")
                kappa_mae = float(np.mean(np.abs(kappa_diff))) if kappa_diff.size else float("nan")
                total_kappa_sse += float(np.sum(kappa_diff**2))
                total_kappa_count += int(kappa_diff.size)

                w_mse = float("nan")
                w_mae = float("nan")
                if w_true is not None:
                    w_diff = w_pred[i, valid] - w_true[i, valid]
                    if w_diff.size:
                        w_mse = float(np.mean(w_diff**2))
                        w_mae = float(np.mean(np.abs(w_diff)))
                        total_w_sse += float(np.sum(w_diff**2))
                        total_w_count += int(w_diff.size)

                metrics_rows.append(
                    {
                        "sample_id": sample_idx,
                        "sample_name": sample_name,
                        "n_points": int(valid.sum()),
                        "kappa_mse": kappa_mse,
                        "kappa_mae": kappa_mae,
                        "w_mse": w_mse,
                        "w_mae": w_mae,
                    }
                )

                coeff_rows.append(
                    {
                        "sample_id": sample_idx,
                        "sample_name": sample_name,
                        **{f"a_{j}": float(coeff[i, j]) for j in range(coeff.shape[1])},
                    }
                )

                if sample_idx < plot_limit:
                    plot_cache[sample_idx] = {
                        "x": x_vals[i, valid],
                        "y": y_vals[i, valid],
                        "kappa_meas": kappa_meas[i, valid],
                        "kappa_pred": kappa_pred[i, valid],
                        "w_pred": w_pred[i, valid],
                        "w_true": None if w_true is None else w_true[i, valid],
                    }

            sample_offset += batch_size

    _save_parquet(output_dir / "predictions.parquet", prediction_rows)
    _save_parquet(output_dir / "coeffs.parquet", coeff_rows)
    if w_rows:
        _save_parquet(output_dir / "w_predictions.parquet", w_rows)
    if w_compare_rows:
        pd.DataFrame(w_compare_rows).to_csv(output_dir / "w_compare.csv", index=False)
    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(output_dir / "prediction_metrics.csv", index=False)

    df_pred = pd.DataFrame(prediction_rows)
    _plot_kappa_scatter(df_pred, output_dir / "kappa_scatter.png")

    for sample_idx, payload in plot_cache.items():
        x_axis = np.arange(payload["kappa_meas"].shape[0])
        sample_name = (
            sample_names[sample_idx]
            if sample_idx < len(sample_names)
            else f"sample_{sample_idx:04d}"
        )
        _plot_curve(
            x=x_axis,
            y_meas=payload["kappa_meas"],
            y_pred=payload["kappa_pred"],
            path=output_dir / f"kappa_curve_{sample_name}.png",
            title=f"kappa_t: {sample_name}",
            ylabel="kappa_t",
        )
        _plot_3d_scatter(
            x=payload["x"],
            y=payload["y"],
            z=payload["w_pred"],
            path=output_dir / f"w_pred_3d_{sample_name}.png",
            title=f"w_pred: {sample_name}",
            zlabel="w_pred",
        )
        if payload["w_true"] is not None:
            _plot_3d_scatter(
                x=payload["x"],
                y=payload["y"],
                z=payload["w_true"],
                path=output_dir / f"w_true_3d_{sample_name}.png",
                title=f"w_true: {sample_name}",
                zlabel="w_true",
            )
        if payload["w_true"] is not None:
            _plot_curve(
                x=x_axis,
                y_meas=payload["w_true"],
                y_pred=payload["w_pred"],
                path=output_dir / f"w_curve_{sample_name}.png",
                title=f"w: {sample_name}",
                ylabel="w",
            )

    overall_kappa_mse = (
        total_kappa_sse / total_kappa_count if total_kappa_count else float("nan")
    )
    overall_w_mse = total_w_sse / total_w_count if total_w_count else float("nan")

    summary = {
        "num_samples": len(sample_names),
        "num_points": int(len(prediction_rows)),
        "kappa_mse": overall_kappa_mse,
        "w_mse": overall_w_mse,
        "output_dir": str(output_dir),
    }
    with (output_dir / "prediction_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Saved predictions to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
