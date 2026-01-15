#!/usr/bin/env python
"""
Collect a diagnostic pack for investigating kappa_pred scale issues.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

# Add project root + src to path for direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "project"))
sys.path.insert(0, str(ROOT_DIR / "project" / "src"))

from basis import hessian_from_coeff, make_mode_frequencies, w_from_coeff
from datasets import FiberSequenceDataset, fiber_sequence_collate
from models import MambaCoeffNet
from operators import kappa_t_from_coeff
from train import Trainer
from utils import setup_logger
from viz.plotting import plot_error_heatmap, plot_surface_3d

logger = setup_logger("collect_kappa_debug_pack")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect kappa debug diagnostic pack")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu|cuda|auto")
    parser.add_argument("--sample_id", type=int, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--grid_nx", type=int, default=64)
    parser.add_argument("--grid_ny", type=int, default=64)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument(
        "--allow_cnn_fallback",
        action="store_true",
        help="Allow CNN fallback when Mamba weights cannot be used (results may be invalid).",
    )
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


def _to_numpy(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values
    return values.detach().cpu().numpy()


def _masked(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    flat = values.reshape(-1)
    if mask is None:
        return flat
    mask_flat = mask.reshape(-1) > 0
    return flat[mask_flat]


def _masked_stats(values: np.ndarray, mask: np.ndarray | None) -> dict[str, float]:
    vals = _masked(values, mask)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"abs_max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "abs_max": float(np.max(np.abs(vals))),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
    }


def _finite_stats(values: np.ndarray) -> dict[str, float]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return {"abs_max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "abs_max": float(np.max(np.abs(vals))),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
    }


def _diff_stats(diff: np.ndarray) -> dict[str, float]:
    vals = diff[np.isfinite(diff)]
    if vals.size == 0:
        return {"max_abs": float("nan"), "mean_abs": float("nan")}
    return {
        "max_abs": float(np.max(np.abs(vals))),
        "mean_abs": float(np.mean(np.abs(vals))),
    }


def _grid_nearest(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    v_arr = np.asarray(values, dtype=float).ravel()
    valid = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(v_arr)
    if not np.any(valid):
        Xg, _ = np.meshgrid(xg, yg)
        return np.full_like(Xg, np.nan, dtype=float)

    pts = np.column_stack([x_arr[valid], y_arr[valid]])
    vals = v_arr[valid]
    Xg, Yg = np.meshgrid(xg, yg)
    grid_pts = np.column_stack([Xg.ravel(), Yg.ravel()])
    out = np.empty(grid_pts.shape[0], dtype=float)
    for start in range(0, grid_pts.shape[0], chunk_size):
        chunk = grid_pts[start : start + chunk_size]
        diff = chunk[:, None, :] - pts[None, :, :]
        dist2 = np.sum(diff**2, axis=-1)
        idx = np.argmin(dist2, axis=1)
        out[start : start + chunk_size] = vals[idx]
    return out.reshape(Xg.shape)


def _finite_difference_hessian(
    w_grid: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    w = np.asarray(w_grid, dtype=float)
    if w.ndim != 2:
        raise ValueError(f"w_grid must be 2D, got shape {w.shape}")

    dx = float(xg[1] - xg[0]) if len(xg) > 1 else float("nan")
    dy = float(yg[1] - yg[0]) if len(yg) > 1 else float("nan")
    w_xx = np.full_like(w, np.nan, dtype=float)
    w_yy = np.full_like(w, np.nan, dtype=float)
    w_xy = np.full_like(w, np.nan, dtype=float)

    if w.shape[0] >= 3 and w.shape[1] >= 3 and np.isfinite(dx) and np.isfinite(dy):
        w_xx[1:-1, 1:-1] = (w[1:-1, 2:] - 2.0 * w[1:-1, 1:-1] + w[1:-1, :-2]) / (dx**2)
        w_yy[1:-1, 1:-1] = (w[2:, 1:-1] - 2.0 * w[1:-1, 1:-1] + w[:-2, 1:-1]) / (dy**2)
        w_xy[1:-1, 1:-1] = (
            w[2:, 2:] - w[2:, :-2] - w[:-2, 2:] + w[:-2, :-2]
        ) / (4.0 * dx * dy)

    meta = {
        "dx": dx,
        "dy": dy,
        "boundary_handling": "central difference on interior; boundary left as NaN",
    }
    return w_xx, w_xy, w_yy, meta


def _save_grid_csv(path: Path, xg: np.ndarray, yg: np.ndarray, columns: dict[str, np.ndarray]) -> None:
    Xg, Yg = np.meshgrid(xg, yg)
    data = {"x": Xg.ravel(), "y": Yg.ravel()}
    for key, values in columns.items():
        data[key] = np.asarray(values, dtype=float).ravel()
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def _grid_bounds(x_vals: np.ndarray, y_vals: np.ndarray, Lx: float, Ly: float) -> tuple[float, float, float, float]:
    if x_vals.size == 0 or y_vals.size == 0:
        return 0.0, float(Lx), 0.0, float(Ly)
    return float(np.min(x_vals)), float(np.max(x_vals)), float(np.min(y_vals)), float(np.max(y_vals))


def _resolve_split_indices(
    dataset: FiberSequenceDataset,
    config: dict[str, Any],
    split: str,
) -> tuple[list[int], str]:
    data_cfg = config.get("data", {})
    splits = data_cfg.get("splits")
    if isinstance(splits, dict) and split in splits:
        indices = splits[split]
        if isinstance(indices, (list, tuple, np.ndarray)):
            indices_list = [int(v) for v in indices]
            return indices_list, "config.data.splits"
    return list(range(len(dataset))), "full_dataset"


def _issue_hint(stats: dict[str, Any]) -> str:
    scale_ratio_std = float(stats.get("scale_ratio_std", float("nan")))
    t_norm_mean = float(stats.get("t_norm_mean", float("nan")))
    t_norm_abs_max = float(stats.get("t_norm_abs_max", float("nan")))

    hessian_rel = []
    for key in ("w_xx", "w_yy", "w_xy"):
        rel = stats.get("hessian_fd_error", {}).get(key, {}).get("rel_max")
        if rel is not None and np.isfinite(rel):
            hessian_rel.append(float(rel))
    hessian_rel_max = max(hessian_rel) if hessian_rel else float("nan")

    kappa_rel = float(
        stats.get("kappa_fd_error", {}).get("rel_max", float("nan"))
    )

    if np.isfinite(hessian_rel_max) and hessian_rel_max > 0.2:
        return "解析 Hessian 尺度"
    if np.isfinite(kappa_rel) and kappa_rel > 0.2:
        return "解析 Hessian 尺度"
    if np.isfinite(scale_ratio_std) and (scale_ratio_std > 1e3 or scale_ratio_std < 1e-3):
        return "kappa_meas 比例"
    if np.isfinite(t_norm_mean) and (
        abs(t_norm_mean - 1.0) > 0.1 or (np.isfinite(t_norm_abs_max) and t_norm_abs_max > 1.5)
    ):
        return "tx,ty 归一化"
    return "其他"


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if "model_state" not in checkpoint and "state_dict" not in checkpoint:
        raise ValueError("Checkpoint missing model_state/state_dict.")
    return checkpoint


def _checkpoint_has_mamba(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(key.startswith("mamba_layers.") for key in state_dict.keys())


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

    checkpoint_payload = _load_checkpoint_payload(checkpoint_path)
    state_dict = checkpoint_payload.get("model_state", checkpoint_payload.get("state_dict"))
    if state_dict is None:
        raise ValueError("Checkpoint missing model_state/state_dict.")
    checkpoint_has_mamba = _checkpoint_has_mamba(state_dict)

    model_cfg = config.setdefault("model", {})
    encoder_type = str(model_cfg.get("encoder_type", "auto"))
    forced_encoder = None
    if not checkpoint_has_mamba and encoder_type in {"auto", "mamba"}:
        model_cfg["encoder_type"] = "cnn"
        forced_encoder = "cnn"

    data_cfg = config.get("data", {})
    dataset = FiberSequenceDataset(
        manifest_path=data_cfg.get("manifest"),
        samples_dir=data_cfg.get("samples_dir"),
        df_list=data_cfg.get("df_list") if data_cfg.get("use_dataframe") else None,
        stats_path=config.get("stats_path"),
        coord_scale=float(data_cfg.get("coord_scale", 1.0)),
        kappa_meas_scale=float(data_cfg.get("kappa_meas_scale", 1.0)),
    )

    split_indices, split_source = _resolve_split_indices(dataset, config, args.split)
    if args.sample_id < 0 or args.sample_id >= len(split_indices):
        raise ValueError(f"sample_id out of range for split '{args.split}' (size={len(split_indices)})")

    dataset_idx = int(split_indices[args.sample_id])
    sample = dataset[dataset_idx]
    batch = fiber_sequence_collate([sample])

    device = resolve_device(config.get("device", "auto"), args.device)
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer, config, device)
    model_has_mamba = model.mamba_layers is not None
    will_use_mamba = model.encoder_type in {"auto", "mamba"} and model_has_mamba

    fallback_reason = None
    load_strict = True
    if checkpoint_has_mamba and not model_has_mamba:
        fallback_reason = "checkpoint_requires_mamba"
        load_strict = False
    if will_use_mamba and device.type != "cuda":
        fallback_reason = fallback_reason or "mamba_cpu_fallback"
    if fallback_reason and not args.allow_cnn_fallback:
        hint = (
            "Install mamba-ssm and run with CUDA, "
            "or rerun with --allow_cnn_fallback (results may be invalid)."
        )
        raise RuntimeError(
            "Checkpoint/model mismatch: "
            f"checkpoint_has_mamba={checkpoint_has_mamba}, model_has_mamba={model_has_mamba}, "
            f"device={device.type}. {hint}"
        )

    load_result = trainer.model.load_state_dict(state_dict, strict=load_strict)
    if not load_strict:
        logger.warning(
            "Loaded checkpoint with strict=False; missing=%s unexpected=%s",
            load_result.missing_keys,
            load_result.unexpected_keys,
        )
    if "optimizer_state" in checkpoint_payload:
        trainer.optimizer.load_state_dict(checkpoint_payload["optimizer_state"])
    if "global_step" in checkpoint_payload:
        trainer.global_step = int(checkpoint_payload["global_step"])
    trainer.model.eval()

    batch_device = trainer._move_batch(batch)
    with torch.no_grad():
        outputs = trainer._forward(batch_device)

    x = batch_device.get("x", batch_device["X"][..., 0])
    y = batch_device.get("y", batch_device["X"][..., 1])
    tx = batch_device["X"][..., 2]
    ty = batch_device["X"][..., 3]
    mask = batch_device.get("mask")
    kappa_meas = batch_device.get("kappa_meas", batch_device["X"][..., 4])

    if outputs.get("w_pred_points") is None:
        w_pred_points = trainer._compute_w(outputs["a_reshaped"], x, y)
    else:
        w_pred_points = outputs["w_pred_points"]

    w_true_points = batch_device.get("w_points")
    w_xx_pts, w_xy_pts, w_yy_pts = hessian_from_coeff(
        outputs["a_reshaped"][0], x[0], y[0], Lx=trainer.Lx, Ly=trainer.Ly
    )

    x_np = _to_numpy(x[0])
    y_np = _to_numpy(y[0])
    tx_np = _to_numpy(tx[0])
    ty_np = _to_numpy(ty[0])
    mask_np = _to_numpy(mask[0]) if mask is not None else None
    kappa_meas_np = _to_numpy(kappa_meas[0])
    kappa_pred_np = _to_numpy(outputs["kappa_pred"][0])
    w_pred_np = _to_numpy(w_pred_points[0])
    w_true_np = _to_numpy(w_true_points[0]) if w_true_points is not None else None
    w_xx_np = _to_numpy(w_xx_pts).reshape(-1)
    w_xy_np = _to_numpy(w_xy_pts).reshape(-1)
    w_yy_np = _to_numpy(w_yy_pts).reshape(-1)

    valid_mask = None if mask_np is None else (mask_np > 0)
    if valid_mask is None:
        valid_mask = np.ones_like(x_np, dtype=bool)
    valid_mask &= np.isfinite(x_np) & np.isfinite(y_np)

    point_ids = np.where(valid_mask)[0]
    point_pack = {
        "sample_id": np.full(point_ids.shape, dataset_idx, dtype=int),
        "point_id": point_ids.astype(int),
        "x": x_np[valid_mask],
        "y": y_np[valid_mask],
        "tx": tx_np[valid_mask],
        "ty": ty_np[valid_mask],
        "mask": mask_np[valid_mask] if mask_np is not None else np.ones_like(point_ids, dtype=float),
        "w_pred": w_pred_np[valid_mask],
        "kappa_meas": kappa_meas_np[valid_mask],
        "kappa_pred": kappa_pred_np[valid_mask],
        "w_xx": w_xx_np[valid_mask],
        "w_xy": w_xy_np[valid_mask],
        "w_yy": w_yy_np[valid_mask],
    }
    if w_true_np is not None:
        point_pack["w_true"] = w_true_np[valid_mask]

    x_min, x_max, y_min, y_max = _grid_bounds(x_np[valid_mask], y_np[valid_mask], trainer.Lx, trainer.Ly)
    xg = np.linspace(x_min, x_max, int(args.grid_nx))
    yg = np.linspace(y_min, y_max, int(args.grid_ny))

    Xg, Yg = np.meshgrid(xg, yg)

    with torch.no_grad():
        w_pred_grid_t = w_from_coeff(outputs["a_reshaped"][0], Xg, Yg, Lx=trainer.Lx, Ly=trainer.Ly)
        w_xx_grid_t, w_xy_grid_t, w_yy_grid_t = hessian_from_coeff(
            outputs["a_reshaped"][0], Xg, Yg, Lx=trainer.Lx, Ly=trainer.Ly
        )
    w_pred_grid = _to_numpy(w_pred_grid_t)
    w_xx_grid = _to_numpy(w_xx_grid_t)
    w_xy_grid = _to_numpy(w_xy_grid_t)
    w_yy_grid = _to_numpy(w_yy_grid_t)

    tx_grid = _grid_nearest(x_np[valid_mask], y_np[valid_mask], tx_np[valid_mask], xg, yg)
    ty_grid = _grid_nearest(x_np[valid_mask], y_np[valid_mask], ty_np[valid_mask], xg, yg)
    kappa_meas_grid = _grid_nearest(
        x_np[valid_mask], y_np[valid_mask], kappa_meas_np[valid_mask], xg, yg
    )
    w_true_grid = (
        _grid_nearest(x_np[valid_mask], y_np[valid_mask], w_true_np[valid_mask], xg, yg)
        if w_true_np is not None
        else None
    )

    with torch.no_grad():
        kappa_pred_grid_t = kappa_t_from_coeff(
            outputs["a_reshaped"][0],
            Xg,
            Yg,
            tx_grid,
            ty_grid,
            Lx=trainer.Lx,
            Ly=trainer.Ly,
        )
    kappa_pred_grid = _to_numpy(kappa_pred_grid_t)

    w_xx_fd, w_xy_fd, w_yy_fd, fd_meta = _finite_difference_hessian(w_pred_grid, xg, yg)
    kappa_fd_grid = (tx_grid**2) * w_xx_fd + 2.0 * tx_grid * ty_grid * w_xy_fd + (ty_grid**2) * w_yy_fd

    w_true_fd = None
    if w_true_grid is not None:
        w_true_fd = _finite_difference_hessian(w_true_grid, xg, yg)[:3]

    run_name = checkpoint_path.parent.name
    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else Path("diagnostic_pack") / f"{run_name}_sid{args.sample_id}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_grid_csv(out_dir / "w_pred_grid.csv", xg, yg, {"w_pred": w_pred_grid})
    if w_true_grid is not None:
        _save_grid_csv(out_dir / "w_true_grid.csv", xg, yg, {"w_true": w_true_grid})
    _save_grid_csv(out_dir / "kappa_meas_grid.csv", xg, yg, {"kappa_meas": kappa_meas_grid})
    _save_grid_csv(out_dir / "kappa_pred_grid.csv", xg, yg, {"kappa_pred": kappa_pred_grid})
    _save_grid_csv(
        out_dir / "hessian_analytic_grid.csv",
        xg,
        yg,
        {"w_xx": w_xx_grid, "w_xy": w_xy_grid, "w_yy": w_yy_grid},
    )
    _save_grid_csv(
        out_dir / "hessian_fd_grid.csv",
        xg,
        yg,
        {"w_xx_fd": w_xx_fd, "w_xy_fd": w_xy_fd, "w_yy_fd": w_yy_fd},
    )
    if w_true_fd is not None:
        _save_grid_csv(
            out_dir / "hessian_fd_true_grid.csv",
            xg,
            yg,
            {"w_xx_fd": w_true_fd[0], "w_xy_fd": w_true_fd[1], "w_yy_fd": w_true_fd[2]},
        )

    pd.DataFrame(point_pack).to_csv(out_dir / "point_pack.csv", index=False)

    if w_true_grid is not None:
        plot_surface_3d(w_true_grid, xg, yg, out_dir / "w_true_surface.png", title="w_true")
    plot_surface_3d(w_pred_grid, xg, yg, out_dir / "w_pred_surface.png", title="w_pred")
    plot_error_heatmap(kappa_meas_grid, xg, yg, out_dir / "kappa_meas_heatmap.png", title="kappa_meas")
    plot_error_heatmap(kappa_pred_grid, xg, yg, out_dir / "kappa_pred_heatmap.png", title="kappa_pred")
    plot_error_heatmap(
        np.abs(w_xx_grid - w_xx_fd),
        xg,
        yg,
        out_dir / "hessian_error_heatmap.png",
        title="|w_xx - w_xx_fd|",
    )

    kappa_meas_stats = _masked_stats(kappa_meas_np, mask_np)
    kappa_pred_stats = _masked_stats(kappa_pred_np, mask_np)
    ratio_eps = 1e-12
    scale_ratio_std = kappa_pred_stats["std"] / (kappa_meas_stats["std"] + ratio_eps)
    scale_ratio_max = kappa_pred_stats["abs_max"] / (kappa_meas_stats["abs_max"] + ratio_eps)

    x_range = x_max - x_min
    y_range = y_max - y_min
    x_range_ratio = x_range / float(trainer.Lx) if trainer.Lx != 0 else float("nan")
    y_range_ratio = y_range / float(trainer.Ly) if trainer.Ly != 0 else float("nan")

    coeff = _to_numpy(outputs["a"][0].reshape(-1))
    coeff_stats = _finite_stats(coeff)

    kx, ky = make_mode_frequencies(trainer.M, trainer.N, trainer.Lx, trainer.Ly)
    kx_np = _to_numpy(kx)
    ky_np = _to_numpy(ky)

    hess_err_xx = _diff_stats(w_xx_grid - w_xx_fd)
    hess_err_xy = _diff_stats(w_xy_grid - w_xy_fd)
    hess_err_yy = _diff_stats(w_yy_grid - w_yy_fd)
    hess_abs_xx = _finite_stats(w_xx_grid)
    hess_abs_xy = _finite_stats(w_xy_grid)
    hess_abs_yy = _finite_stats(w_yy_grid)

    def _rel_stats(err: dict[str, float], base: dict[str, float]) -> dict[str, float]:
        denom = base["abs_max"] + ratio_eps
        return {"rel_max": err["max_abs"] / denom if np.isfinite(denom) else float("nan")}

    hessian_fd_error = {
        "w_xx": {**hess_err_xx, **_rel_stats(hess_err_xx, hess_abs_xx)},
        "w_xy": {**hess_err_xy, **_rel_stats(hess_err_xy, hess_abs_xy)},
        "w_yy": {**hess_err_yy, **_rel_stats(hess_err_yy, hess_abs_yy)},
    }

    kappa_fd_err = _diff_stats(kappa_pred_grid - kappa_fd_grid)
    kappa_pred_grid_stats = _finite_stats(kappa_pred_grid)
    kappa_fd_error = {
        **kappa_fd_err,
        "rel_max": kappa_fd_err["max_abs"] / (kappa_pred_grid_stats["abs_max"] + ratio_eps),
    }

    t_norm = tx_np**2 + ty_np**2
    t_norm_vals = t_norm[valid_mask]
    t_norm_vals = t_norm_vals[np.isfinite(t_norm_vals)]
    t_norm_mean = float(np.mean(t_norm_vals)) if t_norm_vals.size else float("nan")
    t_norm_std = float(np.std(t_norm_vals)) if t_norm_vals.size else float("nan")
    t_norm_abs_max = float(np.max(np.abs(t_norm_vals))) if t_norm_vals.size else float("nan")

    stats = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "device": str(device),
        "model_encoder_type": str(model.encoder_type),
        "forced_encoder": forced_encoder,
        "checkpoint_has_mamba": bool(checkpoint_has_mamba),
        "model_has_mamba": bool(model_has_mamba),
        "allow_cnn_fallback": bool(args.allow_cnn_fallback),
        "cnn_fallback_reason": fallback_reason,
        "load_strict": bool(load_strict),
        "split": args.split,
        "split_source": split_source,
        "sample_id": int(args.sample_id),
        "dataset_index": dataset_idx,
        "out_dir": str(out_dir),
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "x_range_over_Lx": x_range_ratio,
        "y_range_over_Ly": y_range_ratio,
        "coord_scale": float(data_cfg.get("coord_scale", 1.0)),
        "kappa_meas_scale": float(data_cfg.get("kappa_meas_scale", 1.0)),
        "kappa_meas_abs_max": kappa_meas_stats["abs_max"],
        "kappa_meas_mean": kappa_meas_stats["mean"],
        "kappa_meas_std": kappa_meas_stats["std"],
        "kappa_pred_abs_max": kappa_pred_stats["abs_max"],
        "kappa_pred_mean": kappa_pred_stats["mean"],
        "kappa_pred_std": kappa_pred_stats["std"],
        "scale_ratio_std": scale_ratio_std,
        "scale_ratio_max": scale_ratio_max,
        "coeff_abs_max": coeff_stats["abs_max"],
        "coeff_std": coeff_stats["std"],
        "kx_max": float(np.max(np.abs(kx_np))) if kx_np.size else float("nan"),
        "ky_max": float(np.max(np.abs(ky_np))) if ky_np.size else float("nan"),
        "kx_sq_max": float(np.max(kx_np**2)) if kx_np.size else float("nan"),
        "ky_sq_max": float(np.max(ky_np**2)) if ky_np.size else float("nan"),
        "hessian_fd_error": hessian_fd_error,
        "kappa_fd_error": kappa_fd_error,
        "grid_nx": int(args.grid_nx),
        "grid_ny": int(args.grid_ny),
        "grid_dx": fd_meta["dx"],
        "grid_dy": fd_meta["dy"],
        "fd_boundary_handling": fd_meta["boundary_handling"],
        "t_norm_mean": t_norm_mean,
        "t_norm_std": t_norm_std,
        "t_norm_abs_max": t_norm_abs_max,
    }

    stats_path = out_dir / "pack_stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    if fallback_reason:
        print(
            "WARNING: CNN fallback in use (results may be invalid). "
            f"reason={fallback_reason}"
        )
    print(json.dumps(stats, indent=2))
    hint = _issue_hint(stats)
    print(f"最可能问题位于：{hint}")

    logger.info("Saved diagnostic pack to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
