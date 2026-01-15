#!/usr/bin/env python
"""
Agent 8: One-click inference for fiberglass plate reconstruction.

Engineering background:
- Original data are text txt files with fields: x y z strain u v w (space/Tab separated; may have header or not).
- top.txt is the upper-surface fiber data, bot.txt is the lower-surface fiber data (same path).
- After geometry processing we compute tangents (tx, ty) and:
    kappa_t = (strain_bot - strain_top) / h
  where h is plate thickness in meters.
- The trained model (MambaCoeffNet) consumes sequence tokens:
    X = [x_norm, y_norm, tx, ty, kappa_norm], with mask for valid points,
  and outputs 2D basis coefficients a (length K=M*N).
- The basis decoder maps a to w(x,y) grid; the curvature operator computes kappa_pred vs kappa_meas.
- The inference script must be one-click and not rely on notebooks.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from src.io.comsol_txt import load_and_validate
from src.geometry.path import order_path, compute_arclength_and_tangent

try:
    from src.datasets import pair_top_bottom as _pair_top_bottom
except Exception:
    _pair_top_bottom = None

from src.models import MambaCoeffNet
from src.basis.dct2 import w_from_coeff
from src.operators.curvature_projection import kappa_t_from_coeff
from src.utils import setup_logger

logger = setup_logger("infer")
EPS = 1e-12


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-click inference from paired top/bottom txt files.")
    parser.add_argument("--top", type=str, required=True, help="Top surface txt path.")
    parser.add_argument("--bot", type=str, required=True, help="Bottom surface txt path.")
    parser.add_argument("--plate", type=str, required=True, help="plate.yaml path.")
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint.pt path.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: runs/infer/<timestamp>/).",
    )
    parser.add_argument("--device", type=str, default="auto", help="cpu|cuda|auto (default: auto).")
    parser.add_argument(
        "--order_mode",
        type=str,
        default="nn_graph",
        choices=["nn_graph", "as_is"],
        help="Path ordering mode.",
    )
    parser.add_argument(
        "--pair_method",
        type=str,
        default="auto",
        choices=["s", "xy", "auto"],
        help="Pairing method (auto prefers s when available).",
    )
    parser.add_argument("--no_plots", action="store_true", help="Skip plot generation.")
    parser.add_argument(
        "--save_intermediate", action="store_true", help="Save intermediate merged dataframe."
    )
    parser.add_argument("--no_header", action="store_true", help="Force input txt without header.")
    parser.add_argument("--s_tol", type=float, default=None, help="Optional s-distance tolerance.")
    parser.add_argument("--xy_tol", type=float, default=None, help="Optional xy-distance tolerance.")
    return parser.parse_args(argv)


def _resolve_device(value: str | None) -> torch.device:
    device_value = value or "auto"
    if device_value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_value)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a dict, got {type(payload)}")
    return payload


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _grid_from_mesh(Lx: float, Ly: float, mesh_cfg: dict[str, Any] | None) -> tuple[int, int] | None:
    if not mesh_cfg:
        return None
    if "element_size" not in mesh_cfg:
        return None
    elem = float(mesh_cfg["element_size"])
    if elem <= 0:
        return None
    nx = max(2, int(round(Lx / elem)) + 1)
    ny = max(2, int(round(Ly / elem)) + 1)
    return nx, ny


def _load_plate_config(path: Path) -> dict[str, Any]:
    cfg = _load_yaml(path)
    plate_cfg = cfg.get("plate", {})

    h = _first_value(
        plate_cfg.get("h"),
        plate_cfg.get("thickness_h"),
        plate_cfg.get("thickness"),
        cfg.get("h"),
        cfg.get("thickness_h"),
    )
    Lx = _first_value(
        plate_cfg.get("Lx"),
        plate_cfg.get("length"),
        cfg.get("Lx"),
        cfg.get("length"),
    )
    Ly = _first_value(
        plate_cfg.get("Ly"),
        plate_cfg.get("width"),
        cfg.get("Ly"),
        cfg.get("width"),
    )
    if h is None or Lx is None or Ly is None:
        raise ValueError(
            "plate.yaml must define h/Lx/Ly (plate.h|thickness_h, plate.Lx|length, plate.Ly|width)."
        )

    basis_cfg = cfg.get("basis", {}) or {}
    M = basis_cfg.get("M") or basis_cfg.get("m")
    N = basis_cfg.get("N") or basis_cfg.get("n")

    grid_cfg = cfg.get("grid", {}) or {}
    Nx = grid_cfg.get("Nx") or grid_cfg.get("nx")
    Ny = grid_cfg.get("Ny") or grid_cfg.get("ny")
    grid_source = "grid"
    if Nx is None or Ny is None:
        fallback = _grid_from_mesh(float(Lx), float(Ly), cfg.get("mesh", {}))
        if fallback is not None:
            Nx, Ny = fallback
            grid_source = "mesh"

    norm_cfg = cfg.get("norm", {}) or {}

    return {
        "h": float(h),
        "Lx": float(Lx),
        "Ly": float(Ly),
        "M": None if M is None else int(M),
        "N": None if N is None else int(N),
        "Nx": None if Nx is None else int(Nx),
        "Ny": None if Ny is None else int(Ny),
        "norm": norm_cfg,
        "grid_source": grid_source,
    }


def _order_dataframe(df: pd.DataFrame, *, order_mode: str) -> tuple[pd.DataFrame, dict]:
    xy = df[["x", "y"]]
    ordered_xy, ordered_idx, debug = order_path(xy, mode=order_mode)
    df_ord = df.iloc[ordered_idx].copy()
    s, tx, ty = compute_arclength_and_tangent(ordered_xy)
    df_ord["s"] = s
    df_ord["tx"] = tx
    df_ord["ty"] = ty
    return df_ord, debug


def _series_by_s(df_bot: pd.DataFrame, field: str) -> tuple[np.ndarray, np.ndarray] | None:
    if field not in df_bot.columns:
        return None
    s = df_bot["s"].to_numpy(dtype=float)
    v = df_bot[field].to_numpy(dtype=float)
    valid = np.isfinite(s) & np.isfinite(v)
    if not np.any(valid):
        return None
    df_series = pd.DataFrame({"s": s[valid], "v": v[valid]})
    df_series = df_series.groupby("s", as_index=False).mean()
    df_series = df_series.sort_values("s", kind="mergesort")
    return df_series["s"].to_numpy(), df_series["v"].to_numpy()


def _nearest_distance(sorted_points: np.ndarray, targets: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_points, targets)
    left = np.clip(idx - 1, 0, len(sorted_points) - 1)
    right = np.clip(idx, 0, len(sorted_points) - 1)
    left_dist = np.abs(targets - sorted_points[left])
    right_dist = np.abs(sorted_points[right] - targets)
    return np.minimum(left_dist, right_dist)


def _align_by_s(
    df_top: pd.DataFrame, df_bot: pd.DataFrame, field: str, s_tol: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    n_top = len(df_top)
    values = np.zeros(n_top, dtype=float)
    mask = np.zeros(n_top, dtype=bool)

    series = _series_by_s(df_bot, field)
    if series is None:
        return values, mask
    bot_s, bot_vals = series
    if bot_s.size == 0:
        return values, mask

    top_s = df_top["s"].to_numpy(dtype=float)
    valid_top = np.isfinite(top_s)
    if np.any(valid_top):
        values[valid_top] = np.interp(top_s[valid_top], bot_s, bot_vals)

    in_range = np.zeros(n_top, dtype=bool)
    if np.any(valid_top):
        in_range[valid_top] = (top_s[valid_top] >= bot_s[0]) & (top_s[valid_top] <= bot_s[-1])

    mask = valid_top & in_range
    if s_tol is not None:
        nearest = np.full(n_top, np.inf, dtype=float)
        nearest[valid_top] = _nearest_distance(bot_s, top_s[valid_top])
        mask &= nearest <= s_tol

    return values, mask


def _align_by_xy(
    df_top: pd.DataFrame, df_bot: pd.DataFrame, field: str, xy_tol: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    n_top = len(df_top)
    values = np.zeros(n_top, dtype=float)
    mask = np.zeros(n_top, dtype=bool)

    if field not in df_bot.columns:
        return values, mask

    top_x = df_top["x"].to_numpy(dtype=float)
    top_y = df_top["y"].to_numpy(dtype=float)
    valid_top = np.isfinite(top_x) & np.isfinite(top_y)

    bot_x = df_bot["x"].to_numpy(dtype=float)
    bot_y = df_bot["y"].to_numpy(dtype=float)
    bot_val = df_bot[field].to_numpy(dtype=float)
    valid_bot = np.isfinite(bot_x) & np.isfinite(bot_y) & np.isfinite(bot_val)

    if not np.any(valid_bot) or not np.any(valid_top):
        return values, mask

    from scipy.spatial import cKDTree

    bot_xy = np.column_stack([bot_x[valid_bot], bot_y[valid_bot]])
    tree = cKDTree(bot_xy)

    top_xy = np.column_stack([top_x[valid_top], top_y[valid_top]])
    dist, idx = tree.query(top_xy, k=1)

    bot_indices = np.where(valid_bot)[0]
    matched = bot_indices[idx]
    values[valid_top] = bot_val[matched]
    mask[valid_top] = True

    if xy_tol is not None:
        mask[valid_top] &= dist <= xy_tol

    return values, mask


def _fallback_pair_s(
    df_top: pd.DataFrame, df_bot: pd.DataFrame, *, h: float, s_tol: float | None
) -> pd.DataFrame:
    top_x = df_top["x"].to_numpy(dtype=float)
    top_y = df_top["y"].to_numpy(dtype=float)
    top_s = df_top["s"].to_numpy(dtype=float)
    top_tx = df_top["tx"].to_numpy(dtype=float)
    top_ty = df_top["ty"].to_numpy(dtype=float)
    strain_top = df_top["strain"].to_numpy(dtype=float)

    strain_bot, pair_mask = _align_by_s(df_top, df_bot, "strain", s_tol=s_tol)

    finite = (
        np.isfinite(top_x)
        & np.isfinite(top_y)
        & np.isfinite(top_s)
        & np.isfinite(top_tx)
        & np.isfinite(top_ty)
        & np.isfinite(strain_top)
        & np.isfinite(strain_bot)
    )
    mask = pair_mask & finite

    top_x = np.where(np.isfinite(top_x), top_x, 0.0)
    top_y = np.where(np.isfinite(top_y), top_y, 0.0)
    top_s = np.where(np.isfinite(top_s), top_s, 0.0)
    top_tx = np.where(np.isfinite(top_tx), top_tx, 0.0)
    top_ty = np.where(np.isfinite(top_ty), top_ty, 0.0)

    strain_top = np.where(np.isfinite(strain_top), strain_top, 0.0)
    strain_bot = np.where(np.isfinite(strain_bot), strain_bot, 0.0)

    top_x[~mask] = 0.0
    top_y[~mask] = 0.0
    top_s[~mask] = 0.0
    top_tx[~mask] = 0.0
    top_ty[~mask] = 0.0
    strain_top[~mask] = 0.0
    strain_bot[~mask] = 0.0

    strain_avg = 0.5 * (strain_top + strain_bot)
    kappa_t = (strain_bot - strain_top) / float(h)
    kappa_t[~mask] = 0.0

    return pd.DataFrame(
        {
            "x": top_x,
            "y": top_y,
            "s": top_s,
            "tx": top_tx,
            "ty": top_ty,
            "strain_top": strain_top,
            "strain_bot": strain_bot,
            "strain_avg": strain_avg,
            "kappa_t": kappa_t,
            "mask": mask.astype(np.float32),
        }
    )


def _fallback_pair_xy(
    df_top: pd.DataFrame, df_bot: pd.DataFrame, *, h: float, xy_tol: float | None
) -> pd.DataFrame:
    top_x = df_top["x"].to_numpy(dtype=float)
    top_y = df_top["y"].to_numpy(dtype=float)
    top_s = df_top["s"].to_numpy(dtype=float)
    top_tx = df_top["tx"].to_numpy(dtype=float)
    top_ty = df_top["ty"].to_numpy(dtype=float)
    strain_top = df_top["strain"].to_numpy(dtype=float)

    strain_bot, pair_mask = _align_by_xy(df_top, df_bot, "strain", xy_tol=xy_tol)

    finite = (
        np.isfinite(top_x)
        & np.isfinite(top_y)
        & np.isfinite(top_s)
        & np.isfinite(top_tx)
        & np.isfinite(top_ty)
        & np.isfinite(strain_top)
        & np.isfinite(strain_bot)
    )
    mask = pair_mask & finite

    top_x = np.where(np.isfinite(top_x), top_x, 0.0)
    top_y = np.where(np.isfinite(top_y), top_y, 0.0)
    top_s = np.where(np.isfinite(top_s), top_s, 0.0)
    top_tx = np.where(np.isfinite(top_tx), top_tx, 0.0)
    top_ty = np.where(np.isfinite(top_ty), top_ty, 0.0)

    strain_top = np.where(np.isfinite(strain_top), strain_top, 0.0)
    strain_bot = np.where(np.isfinite(strain_bot), strain_bot, 0.0)

    top_x[~mask] = 0.0
    top_y[~mask] = 0.0
    top_s[~mask] = 0.0
    top_tx[~mask] = 0.0
    top_ty[~mask] = 0.0
    strain_top[~mask] = 0.0
    strain_bot[~mask] = 0.0

    strain_avg = 0.5 * (strain_top + strain_bot)
    kappa_t = (strain_bot - strain_top) / float(h)
    kappa_t[~mask] = 0.0

    return pd.DataFrame(
        {
            "x": top_x,
            "y": top_y,
            "s": top_s,
            "tx": top_tx,
            "ty": top_ty,
            "strain_top": strain_top,
            "strain_bot": strain_bot,
            "strain_avg": strain_avg,
            "kappa_t": kappa_t,
            "mask": mask.astype(np.float32),
        }
    )


def _pair_with_fallback(
    df_top: pd.DataFrame,
    df_bot: pd.DataFrame,
    *,
    pair_method: str,
    h: float,
    s_tol: float | None,
    xy_tol: float | None,
) -> tuple[pd.DataFrame, str, bool]:
    method = pair_method
    if method == "auto":
        method = "s" if "s" in df_top.columns and "s" in df_bot.columns else "xy"

    if _pair_top_bottom is not None:
        try:
            result = _pair_top_bottom(
                df_top,
                df_bot,
                method=method,
                h=h,
                s_tol=s_tol,
                xy_tol=xy_tol,
                fill_strategy="zeros",
                return_debug=True,
            )
            if isinstance(result, tuple):
                df_pair, _ = result
            else:
                df_pair = result
            return df_pair, method, False
        except TypeError:
            try:
                df_pair = _pair_top_bottom(
                    df_top,
                    df_bot,
                    method=method,
                    h=h,
                    s_tol=s_tol,
                    xy_tol=xy_tol,
                    fill_strategy="zeros",
                )
                return df_pair, method, False
            except Exception as exc:
                logger.warning("pair_top_bottom failed (%s); using fallback.", exc)
        except Exception as exc:
            logger.warning("pair_top_bottom failed (%s); using fallback.", exc)

    if method == "xy":
        df_pair = _fallback_pair_xy(df_top, df_bot, h=h, xy_tol=xy_tol)
    else:
        df_pair = _fallback_pair_s(df_top, df_bot, h=h, s_tol=s_tol)
    return df_pair, method, True


def _sanitize_merged_df(df: pd.DataFrame) -> pd.DataFrame:
    base_mask = df["mask"].to_numpy(dtype=float) if "mask" in df.columns else np.ones(len(df))
    mask = base_mask > 0

    check_cols = ["x", "y", "s", "tx", "ty", "kappa_t", "strain_top", "strain_bot", "strain_avg"]
    for col in check_cols:
        if col not in df.columns:
            continue
        arr = df[col].to_numpy(dtype=float)
        finite = np.isfinite(arr)
        mask &= finite
        arr = np.where(finite, arr, 0.0)
        df[col] = arr

    for col in ["kappa_t", "strain_top", "strain_bot", "strain_avg"]:
        if col not in df.columns:
            continue
        arr = df[col].to_numpy(dtype=float)
        arr[~mask] = 0.0
        df[col] = arr

    df["mask"] = mask.astype(np.float32)
    return df


def _attach_w(
    merged_df: pd.DataFrame,
    df_top: pd.DataFrame,
    df_bot: pd.DataFrame,
    *,
    method: str,
    s_tol: float | None,
    xy_tol: float | None,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    w_true = None
    w_mask = None
    w_source = None

    if "w" in df_top.columns:
        w_true = df_top["w"].to_numpy(dtype=float)
        w_mask = np.isfinite(w_true)
        w_source = "top"
    elif "w" in df_bot.columns:
        if method == "xy":
            w_true, w_mask = _align_by_xy(df_top, df_bot, "w", xy_tol=xy_tol)
        else:
            w_true, w_mask = _align_by_s(df_top, df_bot, "w", s_tol=s_tol)
        w_source = "bot"

    if w_true is not None:
        w_true = np.where(np.isfinite(w_true), w_true, 0.0)
        merged_df["w"] = w_true

    return w_true, w_mask, w_source


def _rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None) -> float | None:
    if pred.size == 0 or target.size == 0:
        return None
    if mask is None:
        return float(np.sqrt(np.mean((pred - target) ** 2)))
    valid = mask & np.isfinite(pred) & np.isfinite(target)
    if not np.any(valid):
        return None
    return float(np.sqrt(np.mean((pred[valid] - target[valid]) ** 2)))


def _load_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        raise ValueError("Checkpoint must be a dict with model_state or state_dict.")
    return ckpt, state


def _extract_model_cfg(ckpt_cfg: dict[str, Any] | None) -> dict[str, Any]:
    model_cfg = {}
    if isinstance(ckpt_cfg, dict):
        model_cfg = ckpt_cfg.get("model", ckpt_cfg.get("model_cfg", {})) or {}
    return {
        "d_model": int(model_cfg.get("d_model", 128)),
        "n_layers": int(model_cfg.get("n_layers", 4)),
        "dropout": float(model_cfg.get("dropout", 0.0)),
        "encoder_type": str(model_cfg.get("encoder_type", "auto")),
        "cnn_kernel_size": int(model_cfg.get("cnn_kernel_size", 5)),
        "cnn_dilation_base": int(model_cfg.get("cnn_dilation_base", 1)),
        "mlp_hidden": int(model_cfg.get("mlp_hidden", 256)),
    }


def run_infer(args: argparse.Namespace) -> dict[str, Any]:
    plate_cfg = _load_plate_config(Path(args.plate))
    h = float(plate_cfg["h"])
    Lx = float(plate_cfg["Lx"])
    Ly = float(plate_cfg["Ly"])
    if h <= 0:
        raise ValueError("plate.h must be > 0")

    device = _resolve_device(args.device)
    logger.info("Using device: %s", device)

    out_dir = Path(args.out_dir) if args.out_dir else Path("runs/infer") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    has_header = False if args.no_header else None
    df_top, _ = load_and_validate(args.top, has_header=has_header)
    df_bot, _ = load_and_validate(args.bot, has_header=has_header)

    df_top, _ = _order_dataframe(df_top, order_mode=args.order_mode)
    df_bot, _ = _order_dataframe(df_bot, order_mode=args.order_mode)

    df_pair, method_used, pair_fallback = _pair_with_fallback(
        df_top,
        df_bot,
        pair_method=args.pair_method,
        h=h,
        s_tol=args.s_tol,
        xy_tol=args.xy_tol,
    )
    df_pair = _sanitize_merged_df(df_pair)

    required = ["x", "y", "s", "tx", "ty", "kappa_t", "mask"]
    missing = [col for col in required if col not in df_pair.columns]
    if missing:
        raise ValueError(f"Merged dataframe missing required columns: {missing}")

    w_true, w_mask, w_source = _attach_w(
        df_pair,
        df_top,
        df_bot,
        method=method_used,
        s_tol=args.s_tol,
        xy_tol=args.xy_tol,
    )

    mask = df_pair["mask"].to_numpy(dtype=np.float32)
    mask_bool = mask > 0
    x = df_pair["x"].to_numpy(dtype=float)
    y = df_pair["y"].to_numpy(dtype=float)
    s = df_pair["s"].to_numpy(dtype=float)
    tx = df_pair["tx"].to_numpy(dtype=float)
    ty = df_pair["ty"].to_numpy(dtype=float)
    kappa_t = df_pair["kappa_t"].to_numpy(dtype=float)

    norm_cfg = plate_cfg.get("norm", {})
    if not isinstance(norm_cfg, dict):
        norm_cfg = {}

    x_scale = float(norm_cfg.get("x_scale", Lx))
    y_scale = float(norm_cfg.get("y_scale", Ly))
    if x_scale == 0 or y_scale == 0:
        raise ValueError("norm.x_scale and norm.y_scale must be non-zero.")

    if norm_cfg.get("kappa_mean") is not None:
        kappa_mean = float(norm_cfg.get("kappa_mean"))
    else:
        kappa_mean = float(np.mean(kappa_t[mask_bool])) if np.any(mask_bool) else 0.0

    if norm_cfg.get("kappa_std") is not None:
        kappa_std = float(norm_cfg.get("kappa_std"))
    else:
        kappa_std = float(np.std(kappa_t[mask_bool])) if np.any(mask_bool) else 1.0
    kappa_std = max(kappa_std, EPS)

    x_norm = x / x_scale
    y_norm = y / y_scale
    kappa_norm = (kappa_t - kappa_mean) / kappa_std

    X = np.stack([x_norm, y_norm, tx, ty, kappa_norm], axis=-1).astype(np.float32)

    ckpt, state = _load_checkpoint(Path(args.ckpt), device)
    ckpt_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    model_cfg = _extract_model_cfg(ckpt_cfg)

    M = plate_cfg["M"]
    N = plate_cfg["N"]
    if (M is None or N is None) and isinstance(ckpt_cfg, dict):
        basis_cfg = ckpt_cfg.get("basis", {}) or {}
        if M is None:
            M = basis_cfg.get("M")
        if N is None:
            N = basis_cfg.get("N")

    if M is None or N is None:
        raise ValueError("basis M/N missing in plate.yaml and checkpoint config.")
    M = int(M)
    N = int(N)

    Nx = plate_cfg["Nx"]
    Ny = plate_cfg["Ny"]
    if Nx is None or Ny is None:
        raise ValueError("grid Nx/Ny missing in plate.yaml (or mesh.element_size fallback).")
    Nx = int(Nx)
    Ny = int(Ny)

    model = MambaCoeffNet(
        in_features=5,
        out_coeffs=M * N,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        dropout=model_cfg["dropout"],
        encoder_type=model_cfg["encoder_type"],
        cnn_kernel_size=model_cfg["cnn_kernel_size"],
        cnn_dilation_base=model_cfg["cnn_dilation_base"],
        mlp_hidden=model_cfg["mlp_hidden"],
    )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        logger.warning(
            "Checkpoint mismatch: missing=%s unexpected=%s",
            incompatible.missing_keys,
            incompatible.unexpected_keys,
        )
    model.to(device)
    model.eval()

    X_tensor = torch.from_numpy(X).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        a_vec = model(X_tensor, mask_tensor)
    a = a_vec.reshape(1, M, N)

    xg = np.linspace(0.0, Lx, Nx, dtype=np.float32)
    yg = np.linspace(0.0, Ly, Ny, dtype=np.float32)
    Xg, Yg = np.meshgrid(xg, yg)

    with torch.no_grad():
        w_grid_t = w_from_coeff(
            a,
            torch.from_numpy(Xg).to(device),
            torch.from_numpy(Yg).to(device),
            Lx=Lx,
            Ly=Ly,
        )
    w_grid = w_grid_t.squeeze(0).cpu().numpy()

    np.save(out_dir / "w_grid.npy", w_grid)
    pd.DataFrame(w_grid).to_csv(out_dir / "w_grid.csv", index=False, header=False)

    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)
    tx_t = torch.from_numpy(tx).to(device)
    ty_t = torch.from_numpy(ty).to(device)

    with torch.no_grad():
        kappa_pred_t = kappa_t_from_coeff(a, x_t, y_t, tx_t, ty_t, Lx=Lx, Ly=Ly)
    kappa_pred = kappa_pred_t.detach().cpu().numpy()
    if kappa_pred.ndim == 2:
        kappa_pred = kappa_pred[0]

    kappa_rmse = _rmse(kappa_pred, kappa_t, mask_bool)

    w_pred_points = None
    w_rmse_points = None
    if w_true is not None:
        with torch.no_grad():
            w_pred_t = w_from_coeff(a, x_t, y_t, Lx=Lx, Ly=Ly)
        w_pred_points = w_pred_t.detach().cpu().numpy()
        if w_pred_points.ndim == 2:
            w_pred_points = w_pred_points[0]
        mask_w = mask_bool
        if w_mask is not None:
            mask_w = mask_w & w_mask
        w_rmse_points = _rmse(w_pred_points, w_true, mask_w)

    if args.save_intermediate:
        df_pair.to_csv(out_dir / "merged_df.csv", index=False)

    error_heatmap_generated = False
    if not args.no_plots:
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        from src.viz.plotting import (
            plot_surface_3d,
            plot_kappa_curve,
            plot_error_heatmap,
            scatter_to_grid,
        )

        plot_surface_3d(w_grid, xg, yg, fig_dir / "surface_3d.png", title="w(x,y) surface")

        if np.any(mask_bool):
            s_plot = s[mask_bool]
            kappa_meas_plot = kappa_t[mask_bool]
            kappa_pred_plot = kappa_pred[mask_bool]
        else:
            s_plot = s
            kappa_meas_plot = kappa_t
            kappa_pred_plot = kappa_pred
        plot_kappa_curve(
            s_plot,
            kappa_meas_plot,
            kappa_pred_plot,
            fig_dir / "kappa_curve.png",
            title="kappa_t along path",
        )

        if w_true is not None and w_pred_points is not None:
            mask_w = mask_bool
            if w_mask is not None:
                mask_w = mask_w & w_mask
            if np.any(mask_w):
                err_points = w_pred_points - w_true
                err_grid = scatter_to_grid(
                    x[mask_w],
                    y[mask_w],
                    err_points[mask_w],
                    xg,
                    yg,
                    method="linear",
                )
                plot_error_heatmap(
                    err_grid,
                    xg,
                    yg,
                    fig_dir / "error_heatmap.png",
                    title="w_pred - w_true",
                )
                error_heatmap_generated = True

    summary = {
        "inputs": {
            "top": str(args.top),
            "bot": str(args.bot),
            "plate": str(args.plate),
            "ckpt": str(args.ckpt),
        },
        "device": str(device),
        "config": {"h": h, "Lx": Lx, "Ly": Ly, "M": M, "N": N, "Nx": Nx, "Ny": Ny},
        "norm": {
            "x_scale": x_scale,
            "y_scale": y_scale,
            "kappa_mean": kappa_mean,
            "kappa_std": kappa_std,
        },
        "num_points": int(len(df_pair)),
        "num_valid": int(mask_bool.sum()),
        "kappa_rmse": kappa_rmse,
        "w_rmse_points": w_rmse_points,
        "error_heatmap": bool(error_heatmap_generated),
        "pairing": {"method": method_used, "fallback": pair_fallback, "w_source": w_source},
        "model": {
            "d_model": model_cfg["d_model"],
            "n_layers": model_cfg["n_layers"],
            "dropout": model_cfg["dropout"],
            "encoder_type": model_cfg["encoder_type"],
            "cnn_kernel_size": model_cfg["cnn_kernel_size"],
            "cnn_dilation_base": model_cfg["cnn_dilation_base"],
            "mlp_hidden": model_cfg["mlp_hidden"],
            "out_coeffs": M * N,
        },
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_infer(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
