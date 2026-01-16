#!/usr/bin/env python
"""
Generate reconstruction diagnostics for kappa and w alignment issues.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Add project root + src to path for direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "project"))
sys.path.insert(0, str(ROOT_DIR / "project" / "src"))

from basis import w_from_coeff
from datasets import FiberSequenceDataset, fiber_sequence_collate
from models import MambaCoeffNet
from operators import kappa_t_from_coeff
from train import Trainer
from utils import setup_logger
from viz.plotting import plot_error_heatmap, scatter_to_grid

logger = setup_logger("report_recon_diagnostics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruction diagnostics for kappa/w.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--sample_id", type=int, default=None)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--all", action="store_true", help="Process all samples in the split.")
    parser.add_argument("--device", type=str, default=None, help="cpu|cuda|auto")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: diagnostic_reports/<run_name>/)",
    )
    parser.add_argument("--grid", action="store_true", help="Use grid heatmaps instead of scatter.")
    parser.add_argument("--no_plots", action="store_true", help="Skip generating plots.")
    parser.add_argument(
        "--detect_collapse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run conditional collapse detection over the split.",
    )
    parser.add_argument(
        "--collapse_samples",
        type=int,
        default=50,
        help="Number of split samples to evaluate for collapse detection.",
    )
    parser.add_argument(
        "--do_baseline_lstsq",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Solve baseline coefficients via linear least squares.",
    )
    parser.add_argument("--do_baseline_opt", action="store_true")
    parser.add_argument(
        "--baseline_target",
        type=str,
        default="both",
        choices=["w", "kappa", "both"],
        help="Baseline target for least squares (or optimization when enabled).",
    )
    parser.add_argument(
        "--do_sensitivity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run kappa_in sensitivity test (x0/x2).",
    )
    parser.add_argument(
        "--case_tol_small_w_rmse",
        type=float,
        default=0.1,
        help="Small-w RMSE threshold as a factor of std(w_true).",
    )
    parser.add_argument(
        "--case_tol_large_w_rmse",
        type=float,
        default=0.5,
        help="Large-w RMSE threshold as a factor of std(w_true).",
    )
    parser.add_argument(
        "--case_tol_high_kappa_rmse",
        type=float,
        default=5.0,
        help="High-kappa RMSE threshold as a factor of median(|kappa_meas|).",
    )
    parser.add_argument(
        "--case_tol_low_corr_kappa",
        type=float,
        default=0.2,
        help="Low correlation threshold for corr(kappa_pred, kappa_meas).",
    )
    parser.add_argument(
        "--case_ratio_kappa_rmse",
        type=float,
        default=5.0,
        help="Ratio threshold for kappa RMSE comparisons (e.g., kappa_rmse_w / kappa_rmse_k).",
    )
    parser.add_argument(
        "--case_ratio_w_rmse",
        type=float,
        default=5.0,
        help="Ratio threshold for w RMSE comparisons (e.g., w_rmse_k / w_rmse_w).",
    )
    parser.add_argument(
        "--case_ratio_model_w_rmse",
        type=float,
        default=2.0,
        help="Model-vs-LS w RMSE ratio threshold for training-issue detection.",
    )
    parser.add_argument(
        "--case_ratio_model_kappa_rmse",
        type=float,
        default=2.0,
        help="Model-vs-LS kappa RMSE ratio threshold for training-issue detection.",
    )
    parser.add_argument(
        "--lstsq_ridge_lambda",
        type=float,
        default=1e-6,
        help="Tikhonov ridge lambda for underdetermined LS systems.",
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
    loss_cfg = config.get("loss", {})
    out_coeffs = int(model_cfg.get("out_coeffs", basis_cfg.get("M") * basis_cfg.get("N")))
    model_cfg["out_coeffs"] = out_coeffs
    return MambaCoeffNet(
        in_features=int(model_cfg.get("in_features", 5)),
        d_model=int(model_cfg.get("d_model", 128)),
        n_layers=int(model_cfg.get("n_layers", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        out_coeffs=out_coeffs,
        encoder_type=str(model_cfg.get("encoder_type", "auto")),
        embedding_norm=bool(model_cfg.get("embedding_norm", True)),
        pooling=str(model_cfg.get("pooling", "mean")),
        kappa_scale_learnable=bool(loss_cfg.get("kappa_scale_learnable", False)),
        kappa_scale_init=float(loss_cfg.get("kappa_scale_init", 1.0)),
    )


def find_latest_checkpoint(run_root: Path) -> Path:
    candidates = list(run_root.rglob("checkpoint.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {run_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


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


def _masked_flat(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(-1)
    if mask is None:
        return flat
    mask_flat = np.asarray(mask, dtype=float).reshape(-1) > 0
    return flat[mask_flat]


def _masked_pair(
    a: np.ndarray, b: np.ndarray, mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    a_flat = _masked_flat(a, mask)
    b_flat = _masked_flat(b, mask)
    valid = np.isfinite(a_flat) & np.isfinite(b_flat)
    return a_flat[valid], b_flat[valid]


def _masked_values(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    vals = _masked_flat(values, mask)
    return vals[np.isfinite(vals)]


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    a_mean = float(np.mean(a))
    b_mean = float(np.mean(b))
    da = a - a_mean
    db = b - b_mean
    denom = float(np.sqrt(np.sum(da**2) * np.sum(db**2)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(da * db) / denom)


def _error_stats(diff: np.ndarray) -> dict[str, float]:
    vals = diff[np.isfinite(diff)]
    if vals.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "max_abs": float("nan")}
    return {
        "rmse": float(np.sqrt(np.mean(vals**2))),
        "mae": float(np.mean(np.abs(vals))),
        "max_abs": float(np.max(np.abs(vals))),
    }


def _fit_scale_bias(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 2 or y.size < 2:
        return float("nan"), float("nan")
    A = np.column_stack([x, np.ones_like(x)])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def _value_stats(values: np.ndarray) -> dict[str, float]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "abs_max": float("nan")}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "abs_max": float(np.max(np.abs(vals))),
    }


def _max_abs_diff(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None
) -> float:
    diff = (a - b).reshape(-1)
    if mask is not None:
        diff = diff[mask.reshape(-1) > 0]
    if diff.numel() == 0:
        return float("nan")
    return float(diff.abs().max().item())


def _safe_get(stats: dict[str, Any], *keys: str) -> Any:
    value: Any = stats
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _align_mean(
    w_pred: np.ndarray, w_true: np.ndarray, mask: np.ndarray | None
) -> tuple[np.ndarray, float]:
    diff = w_true - w_pred
    diff_vals = _masked_values(diff, mask)
    if diff_vals.size == 0:
        return w_pred.copy(), float("nan")
    offset = float(np.mean(diff_vals))
    return w_pred + offset, offset


def _align_plane(
    x: np.ndarray,
    y: np.ndarray,
    w_pred: np.ndarray,
    w_true: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, float]]:
    x_flat = np.asarray(x, dtype=float).reshape(-1)
    y_flat = np.asarray(y, dtype=float).reshape(-1)
    diff = (w_true - w_pred).reshape(-1)
    if mask is None:
        valid = np.ones_like(diff, dtype=bool)
    else:
        valid = np.asarray(mask, dtype=float).reshape(-1) > 0
    valid &= np.isfinite(x_flat) & np.isfinite(y_flat) & np.isfinite(diff)
    if valid.sum() < 3:
        return w_pred.copy(), {"a": float("nan"), "b": float("nan"), "c": float("nan"), "success": 0.0}
    A = np.column_stack([x_flat[valid], y_flat[valid], np.ones(int(valid.sum()), dtype=float)])
    coeffs, _, _, _ = np.linalg.lstsq(A, diff[valid], rcond=None)
    plane = coeffs[0] * x_flat + coeffs[1] * y_flat + coeffs[2]
    aligned = w_pred.reshape(-1) + plane
    return aligned.reshape(w_pred.shape), {
        "a": float(coeffs[0]),
        "b": float(coeffs[1]),
        "c": float(coeffs[2]),
        "success": 1.0,
    }


def _plot_scatter_map(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    out_path: Path,
    *,
    title: str,
    cmap: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sc = ax.scatter(x, y, c=values, s=10, cmap=cmap, alpha=0.9)
    fig.colorbar(sc, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_kappa_scatter(kappa_meas: np.ndarray, kappa_pred: np.ndarray, out_path: Path) -> None:
    if kappa_meas.size == 0 or kappa_pred.size == 0:
        return
    finite = np.isfinite(kappa_meas) & np.isfinite(kappa_pred)
    if not np.any(finite):
        return
    meas = kappa_meas[finite]
    pred = kappa_pred[finite]
    vmin = float(np.min([meas.min(), pred.min()]))
    vmax = float(np.max([meas.max(), pred.max()]))
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(meas, pred, s=8, alpha=0.6)
    ax.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1.2)
    ax.set_xlabel("kappa_meas")
    ax.set_ylabel("kappa_pred")
    ax.set_title("kappa_meas vs kappa_pred")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _grid_from_scatter(
    x: np.ndarray, y: np.ndarray, values: np.ndarray, grid_n: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x.size == 0 or y.size == 0:
        return np.array([]), np.array([]), np.array([[]])
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if not np.isfinite([x_min, x_max, y_min, y_max]).all():
        return np.array([]), np.array([]), np.array([[]])
    xg = np.linspace(x_min, x_max, grid_n)
    yg = np.linspace(y_min, y_max, grid_n)
    grid = scatter_to_grid(x, y, values, xg, yg, method="linear")
    return xg, yg, grid


def _build_design_matrix_w(
    x: torch.Tensor,
    y: torch.Tensor,
    M: int,
    N: int,
    *,
    Lx: float,
    Ly: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    basis = torch.eye(M * N, device=x.device, dtype=dtype).reshape(M * N, M, N)
    w_basis = w_from_coeff(basis, x, y, Lx=Lx, Ly=Ly)
    return w_basis.transpose(0, 1)


def _build_design_matrix_kappa(
    x: torch.Tensor,
    y: torch.Tensor,
    tx: torch.Tensor,
    ty: torch.Tensor,
    M: int,
    N: int,
    *,
    Lx: float,
    Ly: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    basis = torch.eye(M * N, device=x.device, dtype=dtype).reshape(M * N, M, N)
    kappa_basis = kappa_t_from_coeff(basis, x, y, tx, ty, Lx=Lx, Ly=Ly)
    return kappa_basis.transpose(0, 1)


def _solve_lstsq_system(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    ridge_lambda: float,
    label: str,
) -> dict[str, Any]:
    if A.numel() == 0 or b.numel() == 0:
        return {"success": 0.0, "reason": "empty_system"}

    num_rows, num_cols = int(A.shape[0]), int(A.shape[1])
    if num_rows < num_cols:
        logger.warning(
            "LS %s: num_rows < num_cols (P=%d, MN=%d); using ridge lambda=%.2e",
            label,
            num_rows,
            num_cols,
            ridge_lambda,
        )
        AtA = A.T @ A
        Atb = A.T @ b
        reg = ridge_lambda * torch.eye(num_cols, device=A.device, dtype=A.dtype)
        solution = torch.linalg.solve(AtA + reg, Atb)
        return {
            "success": 1.0,
            "solution": solution,
            "solver": "ridge",
            "num_rows": num_rows,
            "num_cols": num_cols,
            "rank": None,
            "ridge_lambda": float(ridge_lambda),
        }

    lstsq_out = torch.linalg.lstsq(A, b)
    solution = lstsq_out.solution
    if torch.isnan(solution).any() or torch.isinf(solution).any():
        logger.warning("LS %s: lstsq produced NaN/Inf; falling back to ridge.", label)
        AtA = A.T @ A
        Atb = A.T @ b
        reg = ridge_lambda * torch.eye(num_cols, device=A.device, dtype=A.dtype)
        solution = torch.linalg.solve(AtA + reg, Atb)
        return {
            "success": 1.0,
            "solution": solution,
            "solver": "ridge",
            "num_rows": num_rows,
            "num_cols": num_cols,
            "rank": None,
            "ridge_lambda": float(ridge_lambda),
        }

    rank = getattr(lstsq_out, "rank", None)
    if isinstance(rank, torch.Tensor):
        rank_value = int(rank.item()) if rank.numel() == 1 else int(rank.numel())
    elif isinstance(rank, (int, float)):
        rank_value = int(rank)
    else:
        rank_value = None

    return {
        "success": 1.0,
        "solution": solution,
        "solver": "lstsq",
        "num_rows": num_rows,
        "num_cols": num_cols,
        "rank": rank_value,
        "ridge_lambda": 0.0,
    }


def _eval_predictions(
    *,
    w_pred: np.ndarray,
    kappa_pred: np.ndarray,
    w_true: np.ndarray | None,
    kappa_meas: np.ndarray,
    mask: np.ndarray | None,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    compute_aligned_plane: bool = False,
) -> dict[str, float]:
    kappa_pred_vals, kappa_meas_vals = _masked_pair(kappa_pred, kappa_meas, mask)
    kappa_corr = _pearson_corr(kappa_pred_vals, kappa_meas_vals)
    kappa_rmse = _error_stats(kappa_pred_vals - kappa_meas_vals)["rmse"]
    kappa_pred_stats = _value_stats(kappa_pred_vals)

    w_rmse = float("nan")
    w_pred_stats = _value_stats(_masked_values(w_pred, mask))
    if w_true is not None:
        w_rmse = _error_stats(_masked_values(w_pred - w_true, mask))["rmse"]

    aligned_plane_rmse = float("nan")
    if compute_aligned_plane and w_true is not None and x is not None and y is not None:
        w_aligned_plane, _ = _align_plane(x, y, w_pred, w_true, mask)
        aligned_plane_rmse = _error_stats(_masked_values(w_aligned_plane - w_true, mask))["rmse"]

    return {
        "w_rmse": w_rmse,
        "kappa_rmse": kappa_rmse,
        "corr_kappa": kappa_corr,
        "w_pred_std": w_pred_stats["std"],
        "kappa_pred_std": kappa_pred_stats["std"],
        "w_aligned_plane_rmse": aligned_plane_rmse,
    }


def _coeff_norm(a: torch.Tensor | None) -> float:
    if a is None:
        return float("nan")
    return float(torch.linalg.norm(a.reshape(-1)).item())


def _coeff_diff(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None or b is None:
        return float("nan")
    return float((a.reshape(-1) - b.reshape(-1)).abs().max().item())


def _assign_case_label(
    *,
    w_true_available: bool,
    ls_w: dict[str, float] | None,
    ls_k: dict[str, float] | None,
    ls_b: dict[str, float] | None,
    model_metrics: dict[str, float],
    tol_small_w: float,
    tol_large_w: float,
    tol_high_kappa: float,
    tol_low_corr: float,
    ratio_kappa_rmse: float,
    ratio_w_rmse: float,
    ratio_model_w_rmse: float,
    ratio_model_kappa_rmse: float,
) -> tuple[str, list[str], float]:
    def _nanmin(values: list[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        return float(np.min(arr))

    def _get(metric: dict[str, float] | None, key: str) -> float:
        if metric is None:
            return float("nan")
        return float(metric.get(key, float("nan")))

    w_rmse_w = _get(ls_w, "w_rmse")
    kappa_rmse_w = _get(ls_w, "kappa_rmse")
    corr_kappa_w = _get(ls_w, "corr_kappa")

    w_rmse_k = _get(ls_k, "w_rmse")
    kappa_rmse_k = _get(ls_k, "kappa_rmse")
    w_plane_k = _get(ls_k, "w_aligned_plane_rmse")

    w_rmse_b = _get(ls_b, "w_rmse")
    kappa_rmse_b = _get(ls_b, "kappa_rmse")

    model_w_rmse = float(model_metrics.get("w_rmse", float("nan")))
    model_kappa_rmse = float(model_metrics.get("kappa_rmse", float("nan")))

    reasons: list[str] = []

    if w_true_available and ls_w is not None:
        cond_w_small = np.isfinite(w_rmse_w) and np.isfinite(tol_small_w) and w_rmse_w <= tol_small_w
        cond_kappa_high = (
            np.isfinite(kappa_rmse_w)
            and np.isfinite(tol_high_kappa)
            and kappa_rmse_w >= tol_high_kappa
        )
        cond_corr_low = np.isfinite(corr_kappa_w) and corr_kappa_w <= tol_low_corr
        cond_ratio_kappa = (
            np.isfinite(kappa_rmse_w)
            and np.isfinite(kappa_rmse_k)
            and kappa_rmse_k > 0
            and (kappa_rmse_w / kappa_rmse_k) >= ratio_kappa_rmse
        )
        if cond_w_small and (cond_kappa_high or cond_corr_low or cond_ratio_kappa):
            reasons.append(
                "w_rmse_w small but kappa mismatch (rmse high / corr low / ratio large)"
            )
            score = kappa_rmse_w if np.isfinite(kappa_rmse_w) else 1.0 - corr_kappa_w
            return "CASE_INCONSISTENT", reasons, float(score)

    if w_true_available and ls_k is not None:
        cond_kappa_small = False
        if np.isfinite(kappa_rmse_k) and np.isfinite(tol_high_kappa):
            cond_kappa_small = kappa_rmse_k <= tol_high_kappa
        if (
            np.isfinite(kappa_rmse_w)
            and np.isfinite(kappa_rmse_k)
            and kappa_rmse_w > 0
            and (kappa_rmse_k / kappa_rmse_w) <= (1.0 / max(ratio_kappa_rmse, 1e-12))
        ):
            cond_kappa_small = True

        if np.isfinite(w_plane_k) and np.isfinite(tol_large_w):
            cond_w_large = w_plane_k >= tol_large_w
        else:
            cond_w_large = np.isfinite(w_rmse_k) and np.isfinite(tol_large_w) and w_rmse_k >= tol_large_w
        if (
            np.isfinite(w_rmse_k)
            and np.isfinite(w_rmse_w)
            and w_rmse_w > 0
            and (w_rmse_k / w_rmse_w) >= ratio_w_rmse
        ):
            cond_w_large = True

        if cond_kappa_small and cond_w_large:
            reasons.append("kappa fit is good but w remains large even after plane align")
            score = w_plane_k if np.isfinite(w_plane_k) else w_rmse_k
            return "CASE_UNIDENTIFIABLE", reasons, float(score)

    if w_true_available and ls_w is not None:
        if np.isfinite(w_rmse_w) and np.isfinite(tol_large_w) and w_rmse_w >= tol_large_w:
            reasons.append("LS-w w_rmse is large vs std(w_true)")
            return "CASE_UNDEREXPRESSIVE", reasons, float(w_rmse_w)

    best_ls_w = _nanmin([w_rmse_w, w_rmse_k, w_rmse_b])
    best_ls_k = _nanmin([kappa_rmse_w, kappa_rmse_k, kappa_rmse_b])
    cond_model_w = (
        np.isfinite(model_w_rmse)
        and np.isfinite(best_ls_w)
        and best_ls_w > 0
        and model_w_rmse >= ratio_model_w_rmse * best_ls_w
    )
    cond_model_k = (
        np.isfinite(model_kappa_rmse)
        and np.isfinite(best_ls_k)
        and best_ls_k > 0
        and model_kappa_rmse >= ratio_model_kappa_rmse * best_ls_k
    )
    if cond_model_w or cond_model_k:
        if cond_model_w:
            reasons.append("model w_rmse >> best LS w_rmse")
            score = model_w_rmse / best_ls_w
        else:
            reasons.append("model kappa_rmse >> best LS kappa_rmse")
            score = model_kappa_rmse / best_ls_k
        return "CASE_MODEL_TRAINING_ISSUE", reasons, float(score)

    if not w_true_available:
        reasons.append("w_true missing; LS-w skipped")
        return "CASE_W_MISSING", reasons, float("nan")

    return "CASE_OK", reasons, float("nan")


def _baseline_optimize(
    *,
    a_init: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    tx: torch.Tensor,
    ty: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    Lx: float,
    Ly: float,
    target_name: str,
    steps: int = 300,
    lr: float = 1e-2,
) -> dict[str, Any]:
    a_param = torch.nn.Parameter(a_init.detach().clone())
    optimizer = torch.optim.Adam([a_param], lr=lr)

    if mask is None:
        mask_f = None
    else:
        mask_f = mask.reshape(-1) > 0

    def _loss_for(a_val: torch.Tensor) -> torch.Tensor:
        if target_name == "w":
            pred = w_from_coeff(a_val, x, y, Lx=Lx, Ly=Ly)
        else:
            pred = kappa_t_from_coeff(a_val, x, y, tx, ty, Lx=Lx, Ly=Ly)
        diff = pred.reshape(-1) - target.reshape(-1)
        if mask_f is not None:
            diff = diff[mask_f]
        if diff.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        return torch.mean(diff**2)

    with torch.no_grad():
        init_loss = float(_loss_for(a_param).item())

    for _ in range(int(steps)):
        optimizer.zero_grad()
        loss = _loss_for(a_param)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = float(_loss_for(a_param).item())
        if target_name == "w":
            pred_final = w_from_coeff(a_param, x, y, Lx=Lx, Ly=Ly)
        else:
            pred_final = kappa_t_from_coeff(a_param, x, y, tx, ty, Lx=Lx, Ly=Ly)

    diff = pred_final.reshape(-1) - target.reshape(-1)
    if mask_f is not None:
        diff = diff[mask_f]
    diff_np = diff.detach().cpu().numpy()
    stats = _error_stats(diff_np)
    stats.update(
        {
            "steps": int(steps),
            "lr": float(lr),
            "init_mse": init_loss,
            "final_mse": final_loss,
        }
    )
    return stats


def _compute_topk(
    *,
    dataset: FiberSequenceDataset,
    split_indices: list[int],
    trainer: Trainer,
    topk: int,
) -> tuple[list[int], str]:
    w_scores: list[tuple[float, int]] = []
    kappa_scores: list[tuple[float, int]] = []
    for split_pos, dataset_idx in enumerate(split_indices):
        sample = dataset[dataset_idx]
        batch = fiber_sequence_collate([sample])
        batch = trainer._move_batch(batch)
        with torch.no_grad():
            outputs = trainer._forward(batch)
        x = batch.get("x", batch["X"][..., 0])
        y = batch.get("y", batch["X"][..., 1])
        if outputs.get("w_pred_points") is None:
            w_pred = trainer._compute_w(outputs["a_reshaped"], x, y)
        else:
            w_pred = outputs["w_pred_points"]
        mask = batch.get("mask")
        mask_f = (mask > 0) if mask is not None else None

        kappa_meas = batch.get("kappa_meas", batch["X"][..., 4])
        kappa_pred = outputs["kappa_pred"]
        kappa_diff = (kappa_pred - kappa_meas).reshape(-1)
        if mask_f is not None:
            kappa_diff = kappa_diff[mask_f.reshape(-1)]
        if kappa_diff.numel() > 0:
            rmse = float(torch.sqrt(torch.mean(kappa_diff**2)).item())
            kappa_scores.append((rmse, split_pos))

        w_true = batch.get("w_points")
        if w_true is None:
            continue
        w_diff = (w_pred - w_true).reshape(-1)
        if mask_f is not None:
            w_diff = w_diff[mask_f.reshape(-1)]
        if w_diff.numel() == 0:
            continue
        rmse = float(torch.sqrt(torch.mean(w_diff**2)).item())
        w_scores.append((rmse, split_pos))

    if w_scores:
        w_scores.sort(key=lambda item: item[0], reverse=True)
        return [pos for _, pos in w_scores[: max(1, topk)]], "topk_w_rmse"
    if kappa_scores:
        kappa_scores.sort(key=lambda item: item[0], reverse=True)
        return [pos for _, pos in kappa_scores[: max(1, topk)]], "topk_kappa_rmse"
    return list(range(min(len(split_indices), max(1, topk)))), "split_order_fallback"


def _make_sample_dir(out_dir: Path, split_pos: int, dataset_idx: int) -> Path:
    sample_dir = out_dir / f"sample_{split_pos}_idx{dataset_idx}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    return sample_dir


def _report_hints(stats: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    if not stats.get("w_available", False):
        hints.append("w_true 缺失，仅输出 κ 指标")
    kappa_corr = float(stats.get("kappa", {}).get("corr", float("nan")))
    if np.isfinite(kappa_corr) and kappa_corr <= -0.9:
        hints.append("符号/上下表面顺序可能反")

    kappa_consistency = stats.get("kappa_consistency")
    if isinstance(kappa_consistency, dict):
        corr = float(kappa_consistency.get("corr", float("nan")))
        fit_s = float(kappa_consistency.get("fit_s", float("nan")))
        if (np.isfinite(corr) and corr < 0.2) or (
            np.isfinite(fit_s) and (abs(fit_s) >= 10.0 or abs(fit_s) <= 0.1)
        ):
            hints.append("kappa_meas 与 w_true 物理不一致（corr低或尺度偏差大）")

    w_block = stats.get("w")
    if isinstance(w_block, dict):
        raw_rmse = float(w_block.get("raw", {}).get("rmse", float("nan")))
        aligned_plane_rmse = float(w_block.get("aligned_plane", {}).get("rmse", float("nan")))
        if np.isfinite(raw_rmse) and np.isfinite(aligned_plane_rmse) and raw_rmse > 0:
            if aligned_plane_rmse <= 0.5 * raw_rmse:
                hints.append("低频自由度/边界未锁")

        w_true_std = float(w_block.get("w_true_stats", {}).get("std", float("nan")))
        if np.isfinite(kappa_corr) and kappa_corr >= 0.9 and np.isfinite(w_true_std):
            if np.isfinite(raw_rmse) and raw_rmse > 0.5 * w_true_std:
                hints.append("仅曲率不足以确定 w，需 anchors/边界/更多观测")

    baseline = stats.get("baseline_opt")
    if isinstance(baseline, dict):
        baseline_rmse = float(baseline.get("rmse", float("nan")))
        target_name = baseline.get("target")
        if target_name == "w" and isinstance(w_block, dict):
            model_rmse = float(w_block.get("raw", {}).get("rmse", float("nan")))
            w_true_std = float(w_block.get("w_true_stats", {}).get("std", float("nan")))
            if np.isfinite(model_rmse) and np.isfinite(baseline_rmse):
                if baseline_rmse <= 0.7 * model_rmse:
                    hints.append("模型/表达能力/训练没学到上限")
                elif np.isfinite(w_true_std) and baseline_rmse > 0.5 * w_true_std:
                    hints.append("观测不足或 w_true 定义不一致")
        if target_name == "kappa":
            model_rmse = float(stats.get("kappa", {}).get("rmse", float("nan")))
            if np.isfinite(model_rmse) and np.isfinite(baseline_rmse):
                if baseline_rmse <= 0.7 * model_rmse:
                    hints.append("模型/表达能力/训练没学到上限")
                else:
                    hints.append("观测不足或 w_true 定义不一致")
    sensitivity = stats.get("sensitivity")
    if isinstance(sensitivity, dict):
        a_diff = float(sensitivity.get("a_diff_max", float("nan")))
        w_diff = float(sensitivity.get("w_diff_max", float("nan")))
        kappa_diff = float(sensitivity.get("kappa_diff_max", float("nan")))
        if all(np.isfinite(val) for val in (a_diff, w_diff, kappa_diff)):
            if a_diff <= 1e-4 and w_diff <= 1e-4 and kappa_diff <= 1e-4:
                hints.append("模型几乎没在用 kappa conditioning（可能 LN 抹掉幅值）")

    case_label = stats.get("case_label")
    if isinstance(case_label, str):
        hints.append(f"case={case_label}")
    case_reasons = stats.get("case_reasons")
    if isinstance(case_reasons, list) and case_reasons:
        hints.append(f"case_reason={case_reasons[0]}")
    return hints


def _summary_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _aggregate_summary(stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    def collect(*keys: str) -> list[float]:
        values: list[float] = []
        for stats in stats_list:
            val = _safe_get(stats, *keys)
            if val is None:
                continue
            try:
                val_f = float(val)
            except (TypeError, ValueError):
                continue
            if np.isfinite(val_f):
                values.append(val_f)
        return values

    def ratio(numer_keys: tuple[str, ...], denom_keys: tuple[str, ...]) -> list[float]:
        values: list[float] = []
        for stats in stats_list:
            numer = _safe_get(stats, *numer_keys)
            denom = _safe_get(stats, *denom_keys)
            if numer is None or denom is None:
                continue
            try:
                numer_f = float(numer)
                denom_f = float(denom)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numer_f) and np.isfinite(denom_f) and denom_f != 0:
                values.append(numer_f / denom_f)
        return values

    summary = {
        "num_samples": len(stats_list),
        "num_w_available": int(sum(1 for s in stats_list if s.get("w_available"))),
        "a_pred_norm": _summary_stats(collect("a_pred_norm")),
        "w_pred_std": _summary_stats(collect("w_pred_std")),
        "kappa_pred_std": _summary_stats(collect("kappa_pred_std")),
        "kappa_corr": _summary_stats(collect("kappa", "corr")),
        "kappa_corr_abs": _summary_stats(collect("kappa", "corr_abs")),
        "kappa_corr_sign": _summary_stats(collect("kappa", "corr_sign")),
        "kappa_rmse": _summary_stats(collect("kappa", "rmse")),
        "kappa_mae": _summary_stats(collect("kappa", "mae")),
        "kappa_max_abs": _summary_stats(collect("kappa", "max_abs")),
        "kappa_scale_ratio_std": _summary_stats(collect("kappa", "scale_ratio_std")),
        "kappa_scale_ratio_max": _summary_stats(collect("kappa", "scale_ratio_max")),
        "kappa_consistency_corr": _summary_stats(collect("kappa_consistency", "corr")),
        "kappa_consistency_rmse": _summary_stats(collect("kappa_consistency", "rmse")),
        "kappa_consistency_scale_ratio_std": _summary_stats(
            collect("kappa_consistency", "scale_ratio_std")
        ),
        "kappa_consistency_scale_ratio_max": _summary_stats(
            collect("kappa_consistency", "scale_ratio_max")
        ),
        "kappa_consistency_fit_s": _summary_stats(collect("kappa_consistency", "fit_s")),
        "kappa_consistency_fit_b": _summary_stats(collect("kappa_consistency", "fit_b")),
        "w_raw_rmse": _summary_stats(collect("w", "raw", "rmse")),
        "w_raw_mae": _summary_stats(collect("w", "raw", "mae")),
        "w_raw_max_abs": _summary_stats(collect("w", "raw", "max_abs")),
        "w_aligned_mean_rmse": _summary_stats(collect("w", "aligned_mean", "rmse")),
        "w_aligned_plane_rmse": _summary_stats(collect("w", "aligned_plane", "rmse")),
        "w_aligned_plane_ratio": _summary_stats(
            ratio(("w", "aligned_plane", "rmse"), ("w", "raw", "rmse"))
        ),
        "w_true_std": _summary_stats(collect("w", "w_true_stats", "std")),
        "plane_a": _summary_stats(collect("w", "plane_params", "a")),
        "plane_b": _summary_stats(collect("w", "plane_params", "b")),
        "plane_c": _summary_stats(collect("w", "plane_params", "c")),
        "baseline_rmse": _summary_stats(collect("baseline_opt", "rmse")),
        "baseline_init_mse": _summary_stats(collect("baseline_opt", "init_mse")),
        "baseline_final_mse": _summary_stats(collect("baseline_opt", "final_mse")),
        "baseline_lstsq_w_rmse": _summary_stats(collect("baseline_lstsq", "w_rmse")),
        "baseline_lstsq_kappa_rmse": _summary_stats(collect("baseline_lstsq", "kappa_rmse")),
        "baseline_lstsq_w_pred_std": _summary_stats(collect("baseline_lstsq", "w_pred_std")),
        "baseline_lstsq_kappa_pred_std": _summary_stats(
            collect("baseline_lstsq", "kappa_pred_std")
        ),
        "baseline_w_only_w_rmse": _summary_stats(collect("baseline_w_only", "w_rmse")),
        "baseline_w_only_kappa_rmse": _summary_stats(collect("baseline_w_only", "kappa_rmse")),
        "baseline_kappa_only_kappa_rmse": _summary_stats(
            collect("baseline_kappa_only", "kappa_rmse")
        ),
        "baseline_kappa_only_w_rmse": _summary_stats(collect("baseline_kappa_only", "w_rmse")),
        "baseline_both_w_rmse": _summary_stats(collect("baseline_both", "w_rmse")),
        "baseline_both_kappa_rmse": _summary_stats(collect("baseline_both", "kappa_rmse")),
        "model_w_rmse": _summary_stats(collect("model_metrics", "w_rmse")),
        "model_kappa_rmse": _summary_stats(collect("model_metrics", "kappa_rmse")),
        "case_score": _summary_stats(collect("case_score")),
        "sensitivity_a_diff": _summary_stats(collect("sensitivity", "a_diff_max")),
        "sensitivity_w_diff": _summary_stats(collect("sensitivity", "w_diff_max")),
        "sensitivity_kappa_diff": _summary_stats(collect("sensitivity", "kappa_diff_max")),
    }

    hint_counts: dict[str, int] = {}
    for stats in stats_list:
        for hint in _report_hints(stats):
            hint_counts[hint] = hint_counts.get(hint, 0) + 1
    summary["hint_counts"] = hint_counts
    return summary


def _write_summary_csv(path: Path, stats_list: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "dataset_index",
        "selection_method",
        "num_points",
        "num_valid",
        "w_available",
        "collapse_flag",
        "a_pred_norm",
        "w_pred_std",
        "kappa_pred_std",
        "sensitivity_a_diff",
        "kappa_corr",
        "kappa_rmse",
        "kappa_scale_ratio_std",
        "kappa_scale_ratio_max",
        "kappa_consistency_corr",
        "kappa_consistency_rmse",
        "kappa_consistency_scale_ratio_std",
        "kappa_consistency_scale_ratio_max",
        "kappa_consistency_fit_s",
        "kappa_consistency_fit_b",
        "w_raw_rmse",
        "w_aligned_mean_rmse",
        "w_aligned_plane_rmse",
        "w_aligned_plane_ratio",
        "plane_a",
        "plane_b",
        "plane_c",
        "w_true_std",
        "baseline_target",
        "baseline_rmse",
        "baseline_init_mse",
        "baseline_final_mse",
        "baseline_w_only_w_rmse",
        "baseline_w_only_kappa_rmse",
        "baseline_w_only_corr_kappa",
        "baseline_kappa_only_kappa_rmse",
        "baseline_kappa_only_w_rmse",
        "baseline_kappa_only_w_aligned_plane_rmse",
        "baseline_both_w_rmse",
        "baseline_both_kappa_rmse",
        "model_w_rmse",
        "model_kappa_rmse",
        "model_corr_kappa",
        "case_label",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for stats in stats_list:
            raw_rmse = _safe_get(stats, "w", "raw", "rmse")
            plane_rmse = _safe_get(stats, "w", "aligned_plane", "rmse")
            if raw_rmse is not None and plane_rmse is not None:
                try:
                    raw_f = float(raw_rmse)
                    plane_f = float(plane_rmse)
                    ratio_val = plane_f / raw_f if raw_f != 0 else float("nan")
                except (TypeError, ValueError):
                    ratio_val = float("nan")
            else:
                ratio_val = float("nan")

            row = {
                "sample_id": stats.get("sample_id"),
                "dataset_index": stats.get("dataset_index"),
                "selection_method": stats.get("selection_method"),
                "num_points": stats.get("num_points"),
                "num_valid": stats.get("num_valid"),
                "w_available": stats.get("w_available"),
                "collapse_flag": stats.get("collapse_flag"),
                "a_pred_norm": stats.get("a_pred_norm"),
                "w_pred_std": stats.get("w_pred_std"),
                "kappa_pred_std": stats.get("kappa_pred_std"),
                "sensitivity_a_diff": _safe_get(stats, "sensitivity", "a_diff_max"),
                "kappa_corr": _safe_get(stats, "kappa", "corr"),
                "kappa_rmse": _safe_get(stats, "kappa", "rmse"),
                "kappa_scale_ratio_std": _safe_get(stats, "kappa", "scale_ratio_std"),
                "kappa_scale_ratio_max": _safe_get(stats, "kappa", "scale_ratio_max"),
                "kappa_consistency_corr": _safe_get(stats, "kappa_consistency", "corr"),
                "kappa_consistency_rmse": _safe_get(stats, "kappa_consistency", "rmse"),
                "kappa_consistency_scale_ratio_std": _safe_get(
                    stats, "kappa_consistency", "scale_ratio_std"
                ),
                "kappa_consistency_scale_ratio_max": _safe_get(
                    stats, "kappa_consistency", "scale_ratio_max"
                ),
                "kappa_consistency_fit_s": _safe_get(stats, "kappa_consistency", "fit_s"),
                "kappa_consistency_fit_b": _safe_get(stats, "kappa_consistency", "fit_b"),
                "w_raw_rmse": raw_rmse,
                "w_aligned_mean_rmse": _safe_get(stats, "w", "aligned_mean", "rmse"),
                "w_aligned_plane_rmse": plane_rmse,
                "w_aligned_plane_ratio": ratio_val,
                "plane_a": _safe_get(stats, "w", "plane_params", "a"),
                "plane_b": _safe_get(stats, "w", "plane_params", "b"),
                "plane_c": _safe_get(stats, "w", "plane_params", "c"),
                "w_true_std": _safe_get(stats, "w", "w_true_stats", "std"),
                "baseline_target": _safe_get(stats, "baseline_opt", "target"),
                "baseline_rmse": _safe_get(stats, "baseline_opt", "rmse"),
                "baseline_init_mse": _safe_get(stats, "baseline_opt", "init_mse"),
                "baseline_final_mse": _safe_get(stats, "baseline_opt", "final_mse"),
                "baseline_w_only_w_rmse": _safe_get(stats, "baseline_w_only", "w_rmse"),
                "baseline_w_only_kappa_rmse": _safe_get(stats, "baseline_w_only", "kappa_rmse"),
                "baseline_w_only_corr_kappa": _safe_get(stats, "baseline_w_only", "corr_kappa"),
                "baseline_kappa_only_kappa_rmse": _safe_get(stats, "baseline_kappa_only", "kappa_rmse"),
                "baseline_kappa_only_w_rmse": _safe_get(stats, "baseline_kappa_only", "w_rmse"),
                "baseline_kappa_only_w_aligned_plane_rmse": _safe_get(
                    stats, "baseline_kappa_only", "w_aligned_plane_rmse"
                ),
                "baseline_both_w_rmse": _safe_get(stats, "baseline_both", "w_rmse"),
                "baseline_both_kappa_rmse": _safe_get(stats, "baseline_both", "kappa_rmse"),
                "model_w_rmse": _safe_get(stats, "model_metrics", "w_rmse"),
                "model_kappa_rmse": _safe_get(stats, "model_metrics", "kappa_rmse"),
                "model_corr_kappa": _safe_get(stats, "model_metrics", "corr_kappa"),
                "case_label": stats.get("case_label"),
            }
            writer.writerow(row)


def _run_sensitivity(
    *,
    trainer: Trainer,
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None,
    w_pred_points: torch.Tensor,
) -> dict[str, float]:
    def _forward_scaled(scale: float) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        mod_batch = dict(batch)
        X_mod = batch["X"].clone()
        if "kappa_meas" in batch and "kappa_mean" in batch and "kappa_std" in batch:
            kappa_raw = batch["kappa_meas"] * scale
            kappa_mean = batch["kappa_mean"]
            kappa_std = batch["kappa_std"]
            kappa_in = (kappa_raw - kappa_mean) / (kappa_std + 1e-12)
            X_mod[..., 4] = kappa_in
            mod_batch["kappa_meas"] = kappa_raw
        else:
            X_mod[..., 4] = X_mod[..., 4] * scale
        mod_batch["X"] = X_mod
        with torch.no_grad():
            mod_outputs = trainer._forward(mod_batch)
        w_pred_mod = mod_outputs.get("w_pred_points")
        if w_pred_mod is None:
            w_pred_mod = trainer._compute_w(mod_outputs["a_reshaped"], x, y)
        return mod_outputs, w_pred_mod

    outputs_zero, w_zero = _forward_scaled(0.0)
    outputs_double, w_double = _forward_scaled(2.0)

    a_diff_zero = _max_abs_diff(outputs["a"], outputs_zero["a"])
    a_diff_double = _max_abs_diff(outputs["a"], outputs_double["a"])
    w_diff_zero = _max_abs_diff(w_pred_points, w_zero, mask)
    w_diff_double = _max_abs_diff(w_pred_points, w_double, mask)
    kappa_diff_zero = _max_abs_diff(outputs["kappa_pred"], outputs_zero["kappa_pred"], mask)
    kappa_diff_double = _max_abs_diff(outputs["kappa_pred"], outputs_double["kappa_pred"], mask)

    return {
        "a_diff_zero": a_diff_zero,
        "a_diff_double": a_diff_double,
        "a_diff_max": max(a_diff_zero, a_diff_double),
        "w_diff_zero": w_diff_zero,
        "w_diff_double": w_diff_double,
        "w_diff_max": max(w_diff_zero, w_diff_double),
        "kappa_diff_zero": kappa_diff_zero,
        "kappa_diff_double": kappa_diff_double,
        "kappa_diff_max": max(kappa_diff_zero, kappa_diff_double),
    }


def _diagnose_sample(
    *,
    trainer: Trainer,
    dataset: FiberSequenceDataset,
    split_pos: int,
    dataset_idx: int,
    split: str,
    split_source: str,
    selection_method: str,
    out_dir: Path,
    grid: bool,
    make_plots: bool,
    do_baseline_lstsq: bool,
    do_baseline_opt: bool,
    baseline_target: str,
    do_sensitivity: bool,
    case_params: dict[str, float],
) -> dict[str, Any]:
    sample = dataset[dataset_idx]
    batch = fiber_sequence_collate([sample])
    batch = trainer._move_batch(batch)
    with torch.no_grad():
        outputs = trainer._forward(batch)

    x = batch.get("x", batch["X"][..., 0])
    y = batch.get("y", batch["X"][..., 1])
    tx = batch["X"][..., 2]
    ty = batch["X"][..., 3]
    mask = batch.get("mask")
    kappa_meas = batch.get("kappa_meas", batch["X"][..., 4])
    kappa_pred = outputs["kappa_pred"]

    if outputs.get("w_pred_points") is None:
        w_pred_points = trainer._compute_w(outputs["a_reshaped"], x, y)
    else:
        w_pred_points = outputs["w_pred_points"]

    w_true_points = batch.get("w_points")

    x_np = x[0].detach().cpu().numpy()
    y_np = y[0].detach().cpu().numpy()
    mask_np = mask[0].detach().cpu().numpy() if mask is not None else None
    kappa_meas_np = kappa_meas[0].detach().cpu().numpy()
    kappa_pred_np = kappa_pred[0].detach().cpu().numpy()
    w_pred_np = w_pred_points[0].detach().cpu().numpy()
    w_true_np = w_true_points[0].detach().cpu().numpy() if w_true_points is not None else None

    a_pred_np = outputs["a_reshaped"][0].detach().cpu().numpy()
    a_pred_norm = float(np.linalg.norm(a_pred_np.reshape(-1)))

    kappa_pred_vals, kappa_meas_vals = _masked_pair(kappa_pred_np, kappa_meas_np, mask_np)
    kappa_corr = _pearson_corr(kappa_pred_vals, kappa_meas_vals)
    kappa_corr_abs = _pearson_corr(np.abs(kappa_pred_vals), np.abs(kappa_meas_vals))
    kappa_corr_sign = _pearson_corr(np.sign(kappa_pred_vals), np.sign(kappa_meas_vals))
    kappa_diff = kappa_pred_vals - kappa_meas_vals
    kappa_err = _error_stats(kappa_diff)
    kappa_meas_stats = _value_stats(kappa_meas_vals)
    kappa_pred_stats = _value_stats(kappa_pred_vals)
    ratio_eps = 1e-12
    scale_ratio_std = kappa_pred_stats["std"] / (kappa_meas_stats["std"] + ratio_eps)
    scale_ratio_max = kappa_pred_stats["abs_max"] / (kappa_meas_stats["abs_max"] + ratio_eps)
    w_pred_stats = _value_stats(_masked_values(w_pred_np, mask_np))
    model_metrics = _eval_predictions(
        w_pred=w_pred_np,
        kappa_pred=kappa_pred_np,
        w_true=w_true_np,
        kappa_meas=kappa_meas_np,
        mask=mask_np,
        x=x_np,
        y=y_np,
    )

    stats: dict[str, Any] = {
        "checkpoint": str(trainer.config.get("checkpoint", "")),
        "split": split,
        "split_source": split_source,
        "selection_method": selection_method,
        "sample_id": split_pos,
        "dataset_index": dataset_idx,
        "num_points": int(x_np.size),
        "num_valid": int(kappa_meas_vals.size),
        "a_pred_norm": a_pred_norm,
        "w_pred_std": w_pred_stats["std"],
        "kappa_pred_std": kappa_pred_stats["std"],
        "model_metrics": model_metrics,
        "kappa": {
            "corr": kappa_corr,
            "corr_abs": kappa_corr_abs,
            "corr_sign": kappa_corr_sign,
            **kappa_err,
            "scale_ratio_std": scale_ratio_std,
            "scale_ratio_max": scale_ratio_max,
            "kappa_meas_stats": kappa_meas_stats,
            "kappa_pred_stats": kappa_pred_stats,
        },
        "w_available": bool(w_true_np is not None),
    }

    sample_dir = _make_sample_dir(out_dir, split_pos, dataset_idx)

    if w_true_np is not None:
        w_true_stats = _value_stats(_masked_values(w_true_np, mask_np))
        stats["w_true_std"] = w_true_stats["std"]

        w_diff_raw = w_pred_np - w_true_np
        w_raw_stats = _error_stats(_masked_values(w_diff_raw, mask_np))

        w_aligned_mean, mean_offset = _align_mean(w_pred_np, w_true_np, mask_np)
        w_diff_mean = w_aligned_mean - w_true_np
        w_mean_stats = _error_stats(_masked_values(w_diff_mean, mask_np))

        w_aligned_plane, plane_params = _align_plane(x_np, y_np, w_pred_np, w_true_np, mask_np)
        w_diff_plane = w_aligned_plane - w_true_np
        w_plane_stats = _error_stats(_masked_values(w_diff_plane, mask_np))

        stats["w"] = {
            "raw": w_raw_stats,
            "aligned_mean": w_mean_stats,
            "aligned_plane": w_plane_stats,
            "aligned_mean_offset": mean_offset,
            "plane_params": plane_params,
            "w_true_stats": w_true_stats,
            "w_pred_stats": w_pred_stats,
        }

        kappa_consistency = None
        try:
            x0 = x[0].detach().cpu().to(torch.float64)
            y0 = y[0].detach().cpu().to(torch.float64)
            tx0 = tx[0].detach().cpu().to(torch.float64)
            ty0 = ty[0].detach().cpu().to(torch.float64)
            mask0 = mask[0].detach().cpu() if mask is not None else None
            w_true0 = w_true_points[0].detach().cpu().to(torch.float64)

            M, N = outputs["a_reshaped"][0].shape
            Phi_full = _build_design_matrix_w(
                x0, y0, M, N, Lx=trainer.Lx, Ly=trainer.Ly, dtype=torch.float64
            )
            num_points = int(x0.numel())
            if mask0 is None:
                mask_vec = torch.ones(num_points, dtype=torch.bool)
            else:
                mask_vec = mask0.reshape(-1) > 0
            w_vec = w_true0.reshape(-1)
            finite_w = torch.isfinite(w_vec)
            mask_w = mask_vec & finite_w
            if mask_w.any():
                Phi_w = Phi_full[mask_w]
                b_w = w_vec[mask_w]
                result_w = _solve_lstsq_system(
                    Phi_w,
                    b_w,
                    ridge_lambda=case_params["lstsq_ridge_lambda"],
                    label="w_consistency",
                )
                if result_w.get("success"):
                    a_w_cons = result_w["solution"].reshape(M, N)
                    kappa_true = kappa_t_from_coeff(
                        a_w_cons, x0, y0, tx0, ty0, Lx=trainer.Lx, Ly=trainer.Ly
                    )
                    kappa_true_np = kappa_true.detach().cpu().numpy()
                    kappa_true_vals, kappa_meas_vals2 = _masked_pair(
                        kappa_true_np, kappa_meas_np, mask_np
                    )
                    kappa_corr_cons = _pearson_corr(kappa_true_vals, kappa_meas_vals2)
                    kappa_rmse_cons = _error_stats(kappa_true_vals - kappa_meas_vals2)["rmse"]
                    kappa_true_stats = _value_stats(kappa_true_vals)
                    kappa_meas_stats2 = _value_stats(kappa_meas_vals2)
                    ratio_eps = 1e-12
                    scale_ratio_std = kappa_meas_stats2["std"] / (
                        kappa_true_stats["std"] + ratio_eps
                    )
                    scale_ratio_max = kappa_meas_stats2["abs_max"] / (
                        kappa_true_stats["abs_max"] + ratio_eps
                    )
                    fit_s, fit_b = _fit_scale_bias(kappa_true_vals, kappa_meas_vals2)
                    kappa_consistency = {
                        "corr": kappa_corr_cons,
                        "rmse": kappa_rmse_cons,
                        "scale_ratio_std": scale_ratio_std,
                        "scale_ratio_max": scale_ratio_max,
                        "fit_s": fit_s,
                        "fit_b": fit_b,
                        "num_valid": int(kappa_true_vals.size),
                    }
        except Exception as exc:
            logger.warning("kappa consistency check failed: %s", exc)

        if kappa_consistency is not None:
            stats["kappa_consistency"] = kappa_consistency

        if make_plots and grid:
            grid_mask = np.isfinite(x_np) & np.isfinite(y_np)
            if mask_np is not None:
                grid_mask &= mask_np > 0
            grid_mask &= np.isfinite(w_true_np) & np.isfinite(w_pred_np)
            xv = x_np[grid_mask]
            yv = y_np[grid_mask]
            w_true_v = w_true_np[grid_mask]
            w_pred_v = w_pred_np[grid_mask]
            w_err_v = w_diff_raw[grid_mask]
            w_err_plane_v = w_diff_plane[grid_mask]

            xg, yg, w_true_grid = _grid_from_scatter(xv, yv, w_true_v, grid_n=100)
            _, _, w_pred_grid = _grid_from_scatter(xv, yv, w_pred_v, grid_n=100)
            _, _, w_err_grid = _grid_from_scatter(xv, yv, w_err_v, grid_n=100)
            _, _, w_err_plane_grid = _grid_from_scatter(xv, yv, w_err_plane_v, grid_n=100)

            if w_true_grid.size:
                plot_error_heatmap(
                    w_true_grid, xg, yg, sample_dir / "w_true_heatmap.png", title="w_true"
                )
            if w_pred_grid.size:
                plot_error_heatmap(
                    w_pred_grid, xg, yg, sample_dir / "w_pred_heatmap.png", title="w_pred"
                )
            if w_err_grid.size:
                plot_error_heatmap(
                    w_err_grid, xg, yg, sample_dir / "w_error_raw_heatmap.png", title="w_error_raw"
                )
            if w_err_plane_grid.size:
                plot_error_heatmap(
                    w_err_plane_grid,
                    xg,
                    yg,
                    sample_dir / "w_error_aligned_plane_heatmap.png",
                    title="w_error_aligned_plane",
                )
        elif make_plots:
            valid_mask = np.isfinite(x_np) & np.isfinite(y_np)
            if mask_np is not None:
                valid_mask &= mask_np > 0
            valid_mask &= np.isfinite(w_true_np) & np.isfinite(w_pred_np)
            if np.any(valid_mask):
                xv = x_np[valid_mask]
                yv = y_np[valid_mask]
                _plot_scatter_map(
                    xv,
                    yv,
                    w_true_np[valid_mask],
                    sample_dir / "w_true_scatter.png",
                    title="w_true",
                    cmap="viridis",
                )
                _plot_scatter_map(
                    xv,
                    yv,
                    w_pred_np[valid_mask],
                    sample_dir / "w_pred_scatter.png",
                    title="w_pred",
                    cmap="viridis",
                )
                _plot_scatter_map(
                    xv,
                    yv,
                    w_diff_raw[valid_mask],
                    sample_dir / "w_error_raw_scatter.png",
                    title="w_error_raw",
                    cmap="coolwarm",
                )
                _plot_scatter_map(
                    xv,
                    yv,
                    w_diff_plane[valid_mask],
                    sample_dir / "w_error_aligned_plane_scatter.png",
                    title="w_error_aligned_plane",
                    cmap="coolwarm",
                )

    if do_sensitivity:
        stats["sensitivity"] = _run_sensitivity(
            trainer=trainer,
            batch=batch,
            outputs=outputs,
            x=x,
            y=y,
            mask=mask,
            w_pred_points=w_pred_points,
        )

    baseline_w_only = None
    baseline_kappa_only = None
    baseline_both = None
    a_w = None
    a_k = None
    a_b = None
    if do_baseline_lstsq:
        x0 = x[0].detach().cpu().to(torch.float64)
        y0 = y[0].detach().cpu().to(torch.float64)
        tx0 = tx[0].detach().cpu().to(torch.float64)
        ty0 = ty[0].detach().cpu().to(torch.float64)
        mask0 = mask[0].detach().cpu() if mask is not None else None
        kappa0 = kappa_meas[0].detach().cpu().to(torch.float64)
        w_true0 = w_true_points[0].detach().cpu().to(torch.float64) if w_true_points is not None else None

        M, N = outputs["a_reshaped"][0].shape
        Phi_full = _build_design_matrix_w(
            x0, y0, M, N, Lx=trainer.Lx, Ly=trainer.Ly, dtype=torch.float64
        )
        K_full = _build_design_matrix_kappa(
            x0, y0, tx0, ty0, M, N, Lx=trainer.Lx, Ly=trainer.Ly, dtype=torch.float64
        )

        num_points = int(x0.numel())
        if mask0 is None:
            mask_vec = torch.ones(num_points, dtype=torch.bool)
        else:
            mask_vec = mask0.reshape(-1) > 0

        kappa_vec = kappa0.reshape(-1)
        finite_kappa = torch.isfinite(kappa_vec)
        if w_true0 is not None:
            w_vec = w_true0.reshape(-1)
            finite_w = torch.isfinite(w_vec)
        else:
            w_vec = None
            finite_w = None

        mask_k = mask_vec & finite_kappa
        mask_w = mask_vec & finite_w if w_vec is not None else None
        mask_b = mask_vec & finite_kappa & finite_w if w_vec is not None else None

        if baseline_target in ("w", "both") and w_vec is None:
            logger.warning("LS-w skipped because w_true is missing.")
        if baseline_target == "both" and w_vec is None:
            logger.warning("LS-both skipped because w_true is missing.")

        if baseline_target in ("w", "both") and w_vec is not None:
            Phi_w = Phi_full[mask_w]
            b_w = w_vec[mask_w]
            result_w = _solve_lstsq_system(
                Phi_w,
                b_w,
                ridge_lambda=case_params["lstsq_ridge_lambda"],
                label="w",
            )
            if result_w.get("success"):
                a_w = result_w["solution"].reshape(M, N)
                w_pred_w = w_from_coeff(a_w, x0, y0, Lx=trainer.Lx, Ly=trainer.Ly)
                kappa_pred_w = kappa_t_from_coeff(
                    a_w, x0, y0, tx0, ty0, Lx=trainer.Lx, Ly=trainer.Ly
                )
                metrics_w = _eval_predictions(
                    w_pred=w_pred_w.detach().cpu().numpy(),
                    kappa_pred=kappa_pred_w.detach().cpu().numpy(),
                    w_true=w_true_np,
                    kappa_meas=kappa_meas_np,
                    mask=mask_np,
                    x=x_np,
                    y=y_np,
                )
                meta_w = {k: v for k, v in result_w.items() if k != "solution"}
                baseline_w_only = {"target": "w", **meta_w, **metrics_w}

        if baseline_target in ("kappa", "both"):
            K_k = K_full[mask_k]
            b_k = kappa_vec[mask_k]
            result_k = _solve_lstsq_system(
                K_k,
                b_k,
                ridge_lambda=case_params["lstsq_ridge_lambda"],
                label="kappa",
            )
            if result_k.get("success"):
                a_k = result_k["solution"].reshape(M, N)
                w_pred_k = w_from_coeff(a_k, x0, y0, Lx=trainer.Lx, Ly=trainer.Ly)
                kappa_pred_k = kappa_t_from_coeff(
                    a_k, x0, y0, tx0, ty0, Lx=trainer.Lx, Ly=trainer.Ly
                )
                metrics_k = _eval_predictions(
                    w_pred=w_pred_k.detach().cpu().numpy(),
                    kappa_pred=kappa_pred_k.detach().cpu().numpy(),
                    w_true=w_true_np,
                    kappa_meas=kappa_meas_np,
                    mask=mask_np,
                    x=x_np,
                    y=y_np,
                    compute_aligned_plane=True,
                )
                meta_k = {k: v for k, v in result_k.items() if k != "solution"}
                baseline_kappa_only = {"target": "kappa", **meta_k, **metrics_k}

        if baseline_target == "both" and w_vec is not None:
            Phi_b = Phi_full[mask_b]
            K_b = K_full[mask_b]
            b_w = w_vec[mask_b]
            b_k = kappa_vec[mask_b]
            A_b = torch.cat([Phi_b, K_b], dim=0)
            b_b = torch.cat([b_w, b_k], dim=0)
            result_b = _solve_lstsq_system(
                A_b,
                b_b,
                ridge_lambda=case_params["lstsq_ridge_lambda"],
                label="both",
            )
            if result_b.get("success"):
                a_b = result_b["solution"].reshape(M, N)
                w_pred_b = w_from_coeff(a_b, x0, y0, Lx=trainer.Lx, Ly=trainer.Ly)
                kappa_pred_b = kappa_t_from_coeff(
                    a_b, x0, y0, tx0, ty0, Lx=trainer.Lx, Ly=trainer.Ly
                )
                metrics_b = _eval_predictions(
                    w_pred=w_pred_b.detach().cpu().numpy(),
                    kappa_pred=kappa_pred_b.detach().cpu().numpy(),
                    w_true=w_true_np,
                    kappa_meas=kappa_meas_np,
                    mask=mask_np,
                    x=x_np,
                    y=y_np,
                )
                meta_b = {k: v for k, v in result_b.items() if k != "solution"}
                baseline_both = {"target": "both", **meta_b, **metrics_b}

        if baseline_w_only is not None:
            stats["baseline_w_only"] = baseline_w_only
        if baseline_kappa_only is not None:
            stats["baseline_kappa_only"] = baseline_kappa_only
        if baseline_both is not None:
            stats["baseline_both"] = baseline_both
            stats["baseline_lstsq"] = baseline_both

    a_m = outputs["a_reshaped"][0].detach().cpu().to(torch.float64)
    coeff_norms = {
        "a_w": _coeff_norm(a_w),
        "a_k": _coeff_norm(a_k),
        "a_b": _coeff_norm(a_b),
        "a_m": _coeff_norm(a_m),
    }
    coeff_diffs = {
        "m_minus_w": _coeff_diff(a_m, a_w),
        "m_minus_k": _coeff_diff(a_m, a_k),
        "m_minus_b": _coeff_diff(a_m, a_b),
        "w_minus_k": _coeff_diff(a_w, a_k),
        "w_minus_b": _coeff_diff(a_w, a_b),
        "k_minus_b": _coeff_diff(a_k, a_b),
    }
    stats["coeff_norms"] = coeff_norms
    stats["coeff_diffs"] = coeff_diffs

    w_true_std = stats.get("w_true_std", float("nan"))
    kappa_abs_med = float("nan")
    if kappa_meas_vals.size > 0:
        kappa_abs_med = float(np.median(np.abs(kappa_meas_vals)))
    kappa_scale = kappa_abs_med
    if not np.isfinite(kappa_scale) or kappa_scale <= 0:
        kappa_scale = float(kappa_meas_stats.get("std", float("nan")))
    if not np.isfinite(kappa_scale) or kappa_scale <= 0:
        kappa_scale = float(kappa_meas_stats.get("abs_max", float("nan")))

    tol_small_w = (
        case_params["case_tol_small_w_rmse"] * w_true_std if np.isfinite(w_true_std) else float("nan")
    )
    tol_large_w = (
        case_params["case_tol_large_w_rmse"] * w_true_std if np.isfinite(w_true_std) else float("nan")
    )
    tol_high_kappa = (
        case_params["case_tol_high_kappa_rmse"] * kappa_scale
        if np.isfinite(kappa_scale)
        else float("nan")
    )
    stats["case_thresholds"] = {
        "tol_small_w_rmse": tol_small_w,
        "tol_large_w_rmse": tol_large_w,
        "tol_high_kappa_rmse": tol_high_kappa,
        "tol_low_corr_kappa": case_params["case_tol_low_corr_kappa"],
        "ratio_kappa_rmse": case_params["case_ratio_kappa_rmse"],
        "ratio_w_rmse": case_params["case_ratio_w_rmse"],
        "ratio_model_w_rmse": case_params["case_ratio_model_w_rmse"],
        "ratio_model_kappa_rmse": case_params["case_ratio_model_kappa_rmse"],
    }

    case_label, case_reasons, case_score = _assign_case_label(
        w_true_available=bool(w_true_np is not None),
        ls_w=baseline_w_only,
        ls_k=baseline_kappa_only,
        ls_b=baseline_both,
        model_metrics=model_metrics,
        tol_small_w=tol_small_w,
        tol_large_w=tol_large_w,
        tol_high_kappa=tol_high_kappa,
        tol_low_corr=case_params["case_tol_low_corr_kappa"],
        ratio_kappa_rmse=case_params["case_ratio_kappa_rmse"],
        ratio_w_rmse=case_params["case_ratio_w_rmse"],
        ratio_model_w_rmse=case_params["case_ratio_model_w_rmse"],
        ratio_model_kappa_rmse=case_params["case_ratio_model_kappa_rmse"],
    )
    stats["case_label"] = case_label
    stats["case_reasons"] = case_reasons
    stats["case_score"] = case_score

    if make_plots:
        _plot_kappa_scatter(kappa_meas_vals, kappa_pred_vals, sample_dir / "kappa_scatter.png")

    if do_baseline_opt:
        a_init = outputs["a_reshaped"][0]
        x0 = x[0]
        y0 = y[0]
        tx0 = tx[0]
        ty0 = ty[0]
        mask0 = mask[0] if mask is not None else None
        if baseline_target == "both":
            logger.warning("baseline_opt does not support target=both; skipping baseline optimization.")
        elif baseline_target == "w":
            if w_true_points is None:
                logger.warning("baseline_target=w but w_true missing; skipping baseline optimization.")
            else:
                baseline_stats = _baseline_optimize(
                    a_init=a_init,
                    x=x0,
                    y=y0,
                    tx=tx0,
                    ty=ty0,
                    target=w_true_points[0],
                    mask=mask0,
                    Lx=trainer.Lx,
                    Ly=trainer.Ly,
                    target_name="w",
                )
                baseline_stats["target"] = "w"
                stats["baseline_opt"] = baseline_stats
        else:
            baseline_stats = _baseline_optimize(
                a_init=a_init,
                x=x0,
                y=y0,
                tx=tx0,
                ty=ty0,
                target=kappa_meas[0],
                mask=mask0,
                Lx=trainer.Lx,
                Ly=trainer.Ly,
                target_name="kappa",
            )
            baseline_stats["target"] = "kappa"
            stats["baseline_opt"] = baseline_stats

    stats_path = sample_dir / "stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    logger.info("Saved diagnostics to %s", sample_dir)
    return stats


def _write_collapse_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    fieldnames = [
        "num_samples",
        "w_pred_std_mean",
        "w_pred_std_std",
        "w_pred_std_min",
        "w_pred_std_max",
        "a_pred_norm_mean",
        "a_pred_norm_std",
        "a_pred_norm_min",
        "a_pred_norm_max",
        "kappa_pred_std_mean",
        "kappa_pred_std_std",
        "kappa_pred_std_min",
        "kappa_pred_std_max",
        "corr_w_raw_rmse_w_true_std",
        "w_pred_std_cv",
        "w_true_std_cv",
        "a_pairwise_median_max_abs",
        "collapse_flag",
        "collapse_reason",
    ]
    row = {
        "num_samples": summary.get("num_samples"),
        "w_pred_std_mean": _safe_get(summary, "w_pred_std", "mean"),
        "w_pred_std_std": _safe_get(summary, "w_pred_std", "std"),
        "w_pred_std_min": _safe_get(summary, "w_pred_std", "min"),
        "w_pred_std_max": _safe_get(summary, "w_pred_std", "max"),
        "a_pred_norm_mean": _safe_get(summary, "a_pred_norm", "mean"),
        "a_pred_norm_std": _safe_get(summary, "a_pred_norm", "std"),
        "a_pred_norm_min": _safe_get(summary, "a_pred_norm", "min"),
        "a_pred_norm_max": _safe_get(summary, "a_pred_norm", "max"),
        "kappa_pred_std_mean": _safe_get(summary, "kappa_pred_std", "mean"),
        "kappa_pred_std_std": _safe_get(summary, "kappa_pred_std", "std"),
        "kappa_pred_std_min": _safe_get(summary, "kappa_pred_std", "min"),
        "kappa_pred_std_max": _safe_get(summary, "kappa_pred_std", "max"),
        "corr_w_raw_rmse_w_true_std": summary.get("corr_w_raw_rmse_w_true_std"),
        "w_pred_std_cv": summary.get("w_pred_std_cv"),
        "w_true_std_cv": summary.get("w_true_std_cv"),
        "a_pairwise_median_max_abs": summary.get("a_pairwise_median_max_abs"),
        "collapse_flag": summary.get("collapse_flag"),
        "collapse_reason": ";".join(summary.get("collapse_reasons", [])),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def _run_collapse_detection(
    *,
    trainer: Trainer,
    dataset: FiberSequenceDataset,
    split_indices: list[int],
    split: str,
    collapse_samples: int,
) -> dict[str, Any]:
    sample_positions = list(range(len(split_indices)))
    if collapse_samples > 0:
        sample_positions = sample_positions[: min(len(sample_positions), collapse_samples)]

    w_pred_std_list: list[float] = []
    w_true_std_list: list[float] = []
    w_raw_rmse_list: list[float] = []
    a_pred_norm_list: list[float] = []
    kappa_pred_std_list: list[float] = []
    a_pred_flat_list: list[np.ndarray] = []

    for split_pos in sample_positions:
        dataset_idx = int(split_indices[split_pos])
        sample = dataset[dataset_idx]
        batch = fiber_sequence_collate([sample])
        batch = trainer._move_batch(batch)
        with torch.no_grad():
            outputs = trainer._forward(batch)

        x = batch.get("x", batch["X"][..., 0])
        y = batch.get("y", batch["X"][..., 1])
        tx = batch["X"][..., 2]
        ty = batch["X"][..., 3]
        mask = batch.get("mask")

        if outputs.get("w_pred_points") is None:
            w_pred_points = trainer._compute_w(outputs["a_reshaped"], x, y)
        else:
            w_pred_points = outputs["w_pred_points"]

        mask_np = mask[0].detach().cpu().numpy() if mask is not None else None
        w_pred_np = w_pred_points[0].detach().cpu().numpy()
        w_pred_stats = _value_stats(_masked_values(w_pred_np, mask_np))
        w_pred_std_list.append(w_pred_stats["std"])

        kappa_pred_np = outputs["kappa_pred"][0].detach().cpu().numpy()
        kappa_pred_stats = _value_stats(_masked_values(kappa_pred_np, mask_np))
        kappa_pred_std_list.append(kappa_pred_stats["std"])

        a_pred_np = outputs["a_reshaped"][0].detach().cpu().numpy()
        a_pred_flat = a_pred_np.reshape(-1)
        a_pred_flat_list.append(a_pred_flat)
        a_pred_norm_list.append(float(np.linalg.norm(a_pred_flat)))

        w_true_points = batch.get("w_points")
        if w_true_points is not None:
            w_true_np = w_true_points[0].detach().cpu().numpy()
            w_true_stats = _value_stats(_masked_values(w_true_np, mask_np))
            w_true_std_list.append(w_true_stats["std"])
            w_diff = w_pred_np - w_true_np
            w_raw_rmse = _error_stats(_masked_values(w_diff, mask_np))["rmse"]
            w_raw_rmse_list.append(w_raw_rmse)

    def _cv(values: list[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        mean = float(np.mean(arr))
        if mean == 0:
            return float("nan")
        return float(np.std(arr) / mean)

    def _pairwise_median_max_abs(values: list[np.ndarray]) -> float:
        if len(values) < 2:
            return float("nan")
        diffs: list[float] = []
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                diff = float(np.max(np.abs(values[i] - values[j])))
                if np.isfinite(diff):
                    diffs.append(diff)
        if not diffs:
            return float("nan")
        return float(np.median(diffs))

    def _corr_from_lists(a_list: list[float], b_list: list[float]) -> float:
        a_vals: list[float] = []
        b_vals: list[float] = []
        for a_val, b_val in zip(a_list, b_list):
            if np.isfinite(a_val) and np.isfinite(b_val):
                a_vals.append(a_val)
                b_vals.append(b_val)
        if len(a_vals) < 2:
            return float("nan")
        return _pearson_corr(np.asarray(a_vals), np.asarray(b_vals))

    w_pred_std_cv = _cv(w_pred_std_list)
    w_true_std_cv = _cv(w_true_std_list)
    pairwise_median = _pairwise_median_max_abs(a_pred_flat_list)
    corr_w_rmse_w_true_std = _corr_from_lists(w_raw_rmse_list, w_true_std_list)

    reasons: list[str] = []
    if np.isfinite(w_pred_std_cv) and np.isfinite(w_true_std_cv):
        if w_pred_std_cv < 0.01 and w_true_std_cv > 0.1:
            reasons.append(
                f"w_pred_std_cv={w_pred_std_cv:.3g} < 0.01 AND w_true_std_cv={w_true_std_cv:.3g} > 0.1"
            )
    if np.isfinite(pairwise_median) and pairwise_median < 1e-3:
        reasons.append(f"median_pairwise_max_abs_a={pairwise_median:.3g} < 1e-3")

    summary = {
        "split": split,
        "num_samples": len(sample_positions),
        "w_pred_std": _summary_stats(w_pred_std_list),
        "a_pred_norm": _summary_stats(a_pred_norm_list),
        "kappa_pred_std": _summary_stats(kappa_pred_std_list),
        "corr_w_raw_rmse_w_true_std": corr_w_rmse_w_true_std,
        "w_pred_std_cv": w_pred_std_cv,
        "w_true_std_cv": w_true_std_cv,
        "a_pairwise_median_max_abs": pairwise_median,
        "collapse_flag": bool(reasons),
        "collapse_reasons": reasons,
    }
    return summary


def main() -> int:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint(Path("runs"))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config_path = checkpoint_path.parent / "config.yaml"
    config = load_config(config_path)

    if args.device is not None:
        config["device"] = args.device

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
    if not split_indices:
        raise ValueError(f"No samples available for split '{args.split}'")

    device = resolve_device(config.get("device", "auto"), args.device)
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer, config, device)
    trainer.load_checkpoint(checkpoint_path)
    trainer.model.eval()
    trainer.config["checkpoint"] = str(checkpoint_path)

    if args.all:
        sample_positions = list(range(len(split_indices)))
        selection_method = "all"
    elif args.sample_id is not None:
        if args.sample_id < 0 or args.sample_id >= len(split_indices):
            raise ValueError(
                f"sample_id out of range for split '{args.split}' (size={len(split_indices)})"
            )
        sample_positions = [int(args.sample_id)]
        selection_method = "sample_id"
    else:
        sample_positions, selection_method = _compute_topk(
            dataset=dataset, split_indices=split_indices, trainer=trainer, topk=args.topk
        )

    run_name = checkpoint_path.parent.name
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path("diagnostic_reports") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    case_params = {
        "case_tol_small_w_rmse": float(args.case_tol_small_w_rmse),
        "case_tol_large_w_rmse": float(args.case_tol_large_w_rmse),
        "case_tol_high_kappa_rmse": float(args.case_tol_high_kappa_rmse),
        "case_tol_low_corr_kappa": float(args.case_tol_low_corr_kappa),
        "case_ratio_kappa_rmse": float(args.case_ratio_kappa_rmse),
        "case_ratio_w_rmse": float(args.case_ratio_w_rmse),
        "case_ratio_model_w_rmse": float(args.case_ratio_model_w_rmse),
        "case_ratio_model_kappa_rmse": float(args.case_ratio_model_kappa_rmse),
        "lstsq_ridge_lambda": float(args.lstsq_ridge_lambda),
    }

    summary: list[dict[str, Any]] = []
    for split_pos in sample_positions:
        dataset_idx = int(split_indices[split_pos])
        stats = _diagnose_sample(
            trainer=trainer,
            dataset=dataset,
            split_pos=split_pos,
            dataset_idx=dataset_idx,
            split=args.split,
            split_source=split_source,
            selection_method=selection_method,
            out_dir=out_dir,
            grid=args.grid,
            make_plots=not args.no_plots,
            do_baseline_lstsq=args.do_baseline_lstsq,
            do_baseline_opt=args.do_baseline_opt,
            baseline_target=args.baseline_target,
            do_sensitivity=args.do_sensitivity,
            case_params=case_params,
        )
        summary.append(stats)

    collapse_summary = None
    if args.detect_collapse:
        collapse_summary = _run_collapse_detection(
            trainer=trainer,
            dataset=dataset,
            split_indices=split_indices,
            split=args.split,
            collapse_samples=args.collapse_samples,
        )
        _write_collapse_summary_csv(out_dir / "collapse_summary.csv", collapse_summary)
        collapse_flag = collapse_summary.get("collapse_flag")
        for stats in summary:
            stats["collapse_flag"] = collapse_flag
            stats["collapse_reasons"] = collapse_summary.get("collapse_reasons", [])

    summary_payload = _aggregate_summary(summary)
    summary_payload["split"] = args.split
    summary_payload["selection_method"] = selection_method
    summary_payload["checkpoint"] = str(checkpoint_path)
    if collapse_summary is not None:
        summary_payload["collapse_detection"] = collapse_summary

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    _write_summary_csv(out_dir / "summary_samples.csv", summary)

    if collapse_summary is not None:
        w_pred_stats = collapse_summary.get("w_pred_std", {})
        a_pred_stats = collapse_summary.get("a_pred_norm", {})
        kappa_pred_stats = collapse_summary.get("kappa_pred_std", {})
        corr_val = collapse_summary.get("corr_w_raw_rmse_w_true_std")
        print("Collapse detection summary:")
        print(
            "  w_pred_std mean={mean:.6g} std={std:.6g} min={min:.6g} max={max:.6g}".format(
                mean=w_pred_stats.get("mean", float("nan")),
                std=w_pred_stats.get("std", float("nan")),
                min=w_pred_stats.get("min", float("nan")),
                max=w_pred_stats.get("max", float("nan")),
            )
        )
        print(
            "  a_pred_norm mean={mean:.6g} std={std:.6g} min={min:.6g} max={max:.6g}".format(
                mean=a_pred_stats.get("mean", float("nan")),
                std=a_pred_stats.get("std", float("nan")),
                min=a_pred_stats.get("min", float("nan")),
                max=a_pred_stats.get("max", float("nan")),
            )
        )
        print(
            "  kappa_pred_std mean={mean:.6g} std={std:.6g} min={min:.6g} max={max:.6g}".format(
                mean=kappa_pred_stats.get("mean", float("nan")),
                std=kappa_pred_stats.get("std", float("nan")),
                min=kappa_pred_stats.get("min", float("nan")),
                max=kappa_pred_stats.get("max", float("nan")),
            )
        )
        print(f"  corr(w_raw_rmse, w_true_std)={corr_val}")
        if collapse_summary.get("collapse_flag"):
            reason_msg = "; ".join(collapse_summary.get("collapse_reasons", []))
            print(f"CONDITIONAL_COLLAPSE=TRUE ({reason_msg})")
        else:
            print("CONDITIONAL_COLLAPSE=FALSE")

    print("Diagnostics completed.")
    for stats in summary:
        sample_id = stats.get("sample_id")
        dataset_idx = stats.get("dataset_index")
        hints = _report_hints(stats)
        if not hints:
            hints = ["无明显结论（阈值未触发）"]
        hint_msg = "；".join(hints)
        print(f"[sample_id={sample_id} dataset_idx={dataset_idx}] {hint_msg}")

    case_buckets: dict[str, list[dict[str, Any]]] = {}
    for stats in summary:
        label = stats.get("case_label") or "CASE_UNKNOWN"
        case_buckets.setdefault(str(label), []).append(stats)

    print("Case counts:")
    for label in sorted(case_buckets.keys()):
        print(f"  {label}: {len(case_buckets[label])}")

    print("Top-5 samples per case:")
    for label in sorted(case_buckets.keys()):
        items = case_buckets[label]
        if not items:
            continue

        def _score(item: dict[str, Any]) -> float:
            val = item.get("case_score")
            try:
                val_f = float(val)
            except (TypeError, ValueError):
                return float("-inf")
            return val_f if np.isfinite(val_f) else float("-inf")

        top_items = sorted(items, key=_score, reverse=True)[:5]
        print(f"  {label}:")
        for item in top_items:
            sid = item.get("sample_id")
            did = item.get("dataset_index")
            score = item.get("case_score")
            print(f"    sample_id={sid} dataset_idx={did} score={score}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
