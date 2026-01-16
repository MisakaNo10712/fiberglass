#!/usr/bin/env python
"""
Audit kappa_t consistency and placeholder usage for parquet datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "project"))
sys.path.insert(0, str(ROOT_DIR / "project" / "src"))

from utils.kappa_audit import inspect_kappa_source  # noqa: E402

EPS = 1e-12


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.abs_max = float("nan")

    def update(self, values: np.ndarray) -> None:
        vals = values[np.isfinite(values)]
        if vals.size == 0:
            return
        abs_max = float(np.max(np.abs(vals)))
        if not np.isfinite(self.abs_max):
            self.abs_max = abs_max
        else:
            self.abs_max = max(self.abs_max, abs_max)

        n = int(vals.size)
        batch_mean = float(np.mean(vals))
        batch_M2 = float(np.sum((vals - batch_mean) ** 2))
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean = self.mean + delta * n / total
        self.M2 = self.M2 + batch_M2 + (delta**2) * self.count * n / total
        self.count = total

    def summary(self) -> dict[str, float]:
        if self.count == 0:
            return {"abs_max": float("nan"), "mean": float("nan"), "std": float("nan")}
        std = float(np.sqrt(self.M2 / self.count))
        return {"abs_max": float(self.abs_max), "mean": float(self.mean), "std": std}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit kappa_t consistency")
    parser.add_argument("--parquet", type=str, default=None, help="Path to one parquet sample")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest.json")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing parquet files")
    parser.add_argument("--h", type=float, default=None, help="Plate thickness used in kappa_t = strain/h")
    parser.add_argument("--Lx", type=float, default=None, help="Optional Lx sanity check")
    parser.add_argument("--Ly", type=float, default=None, help="Optional Ly sanity check")
    parser.add_argument("--grid_nx", type=int, default=32, help="Grid resolution for FD sanity")
    parser.add_argument("--grid_ny", type=int, default=32, help="Grid resolution for FD sanity")
    parser.add_argument("--out_json", type=str, default=None, help="Output JSON path")
    return parser.parse_args()


def _resolve_path(base: Path, entry: str) -> Path:
    path = Path(entry)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    candidate = base / path
    if candidate.exists():
        return candidate.resolve()
    for parent in base.parents:
        alt = parent / path
        if alt.exists():
            return alt.resolve()
    return candidate.resolve()


def _load_manifest_paths(manifest_path: Path) -> list[Path]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    files = payload.get("files", [])
    if not isinstance(files, list) or not files:
        raise ValueError("Manifest is missing a non-empty 'files' list.")
    base = manifest_path.parent
    paths: list[Path] = []
    for entry in files:
        if not isinstance(entry, dict):
            continue
        path_value = entry.get("output_file") or entry.get("path") or entry.get("file")
        if path_value is None:
            continue
        paths.append(_resolve_path(base, path_value))
    if not paths:
        raise ValueError("Manifest did not contain any sample paths.")
    return paths


def _resolve_parquet_paths(args: argparse.Namespace) -> list[Path]:
    if args.parquet:
        return [Path(args.parquet)]
    if args.manifest:
        return _load_manifest_paths(Path(args.manifest))
    if args.data_dir:
        root = Path(args.data_dir)
        if not root.exists():
            raise FileNotFoundError(f"Samples directory not found: {root}")
        paths = sorted(root.glob("*.parquet"))
        if not paths:
            raise ValueError(f"No parquet samples found under: {root}")
        return paths
    raise ValueError("Must provide --parquet or --manifest or --data_dir.")


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


def _default_out_json(args: argparse.Namespace) -> Path:
    if args.out_json:
        return Path(args.out_json)
    if args.parquet:
        path = Path(args.parquet)
        return path.with_suffix(".kappa_audit.json")
    if args.manifest:
        path = Path(args.manifest)
        return path.with_name(f"{path.stem}_kappa_audit.json")
    if args.data_dir:
        return Path(args.data_dir) / "kappa_audit.json"
    return Path("kappa_audit.json")


def main() -> int:
    args = parse_args()
    paths = _resolve_parquet_paths(args)

    info = inspect_kappa_source(
        manifest_path=args.manifest,
        samples_dir=args.data_dir,
        parquet_path=args.parquet,
    )

    kappa_stats = RunningStats()
    strain_stats: dict[str, RunningStats] = {}
    w_sample = None
    x_sample = None
    y_sample = None
    tx_sample = None
    ty_sample = None
    mask_sample = None

    for path in paths:
        df = pd.read_parquet(path)
        mask = df["mask"].to_numpy(dtype=bool) if "mask" in df.columns else None

        if "kappa_t" in df.columns:
            values = df["kappa_t"].to_numpy(dtype=float)
            if mask is not None:
                values = values[mask]
            kappa_stats.update(values)

        for col in ("strain", "strain_top", "strain_bot", "strain_avg"):
            if col not in df.columns:
                continue
            if col not in strain_stats:
                strain_stats[col] = RunningStats()
            values = df[col].to_numpy(dtype=float)
            if mask is not None:
                values = values[mask]
            strain_stats[col].update(values)

        if w_sample is None and ("w" in df.columns or "w_true" in df.columns):
            w_col = "w" if "w" in df.columns else "w_true"
            w_sample = df[w_col].to_numpy(dtype=float)
            x_sample = df["x"].to_numpy(dtype=float) if "x" in df.columns else None
            y_sample = df["y"].to_numpy(dtype=float) if "y" in df.columns else None
            tx_sample = df["tx"].to_numpy(dtype=float) if "tx" in df.columns else None
            ty_sample = df["ty"].to_numpy(dtype=float) if "ty" in df.columns else None
            mask_sample = mask

    kappa_summary = kappa_stats.summary()
    strain_summary = {key: stats.summary() for key, stats in strain_stats.items()}

    kappa_fd_summary = None
    kappa_fd_meta = None
    if (
        w_sample is not None
        and x_sample is not None
        and y_sample is not None
        and tx_sample is not None
        and ty_sample is not None
    ):
        valid = np.isfinite(x_sample) & np.isfinite(y_sample) & np.isfinite(w_sample)
        if mask_sample is not None:
            valid &= mask_sample
        x_valid = x_sample[valid]
        y_valid = y_sample[valid]
        if x_valid.size >= 4 and y_valid.size >= 4:
            x_min, x_max = float(np.min(x_valid)), float(np.max(x_valid))
            y_min, y_max = float(np.min(y_valid)), float(np.max(y_valid))
            xg = np.linspace(x_min, x_max, int(args.grid_nx))
            yg = np.linspace(y_min, y_max, int(args.grid_ny))
            w_grid = _grid_nearest(x_valid, y_valid, w_sample[valid], xg, yg)
            tx_grid = _grid_nearest(x_valid, y_valid, tx_sample[valid], xg, yg)
            ty_grid = _grid_nearest(x_valid, y_valid, ty_sample[valid], xg, yg)
            w_xx, w_xy, w_yy, fd_meta = _finite_difference_hessian(w_grid, xg, yg)
            kappa_fd_grid = (tx_grid**2) * w_xx + 2.0 * tx_grid * ty_grid * w_xy + (ty_grid**2) * w_yy
            fd_stats = RunningStats()
            fd_stats.update(kappa_fd_grid.reshape(-1))
            kappa_fd_summary = fd_stats.summary()
            kappa_fd_meta = {
                "grid_nx": int(args.grid_nx),
                "grid_ny": int(args.grid_ny),
                "grid_dx": fd_meta["dx"],
                "grid_dy": fd_meta["dy"],
                "fd_boundary_handling": fd_meta["boundary_handling"],
                "kappa_fd_method": "grid_nearest + central_difference",
            }

    scale_ratio_fd_std = None
    scale_ratio_strain_std = None
    if kappa_fd_summary and np.isfinite(kappa_fd_summary["std"]):
        scale_ratio_fd_std = kappa_summary["std"] / (kappa_fd_summary["std"] + EPS)
    if "strain" in strain_summary and np.isfinite(strain_summary["strain"]["std"]):
        scale_ratio_strain_std = kappa_summary["std"] / (strain_summary["strain"]["std"] + EPS)

    scale_ratio_std = scale_ratio_fd_std if scale_ratio_fd_std is not None else scale_ratio_strain_std

    placeholder = bool(info.get("placeholder_kappa"))
    placeholder_reason = info.get("placeholder_reason")

    suggestions = []
    h_value = args.h if args.h is not None else 0.005

    if placeholder:
        suggestions.append(
            {
                "action": "regen_stage2_with_top_bottom",
                "reason": "kappa_t appears to be placeholder strain/h",
                "command": (
                    "python project/scripts/prepare_stage2.py "
                    "--top <top_dir> --bottom <bottom_dir> "
                    f"--output <output_dir> --h {h_value}"
                ),
            }
        )
        suggestions.append(
            {
                "action": "disable_or_downweight_kappa_loss",
                "reason": "placeholder kappa_t is not physically comparable to t^T H(w) t",
                "command": "set loss.lambda_kappa = 0.0 (or 1e-3) in your config",
            }
        )

    abs_max = kappa_summary.get("abs_max", float("nan"))
    if placeholder and np.isfinite(abs_max) and abs_max < 1e-4:
        suggestions.append(
            {
                "action": "kappa_t_scale_too_small",
                "reason": "kappa_t abs_max < 1e-4 suggests placeholder or unit mismatch",
                "command": "recompute kappa_t with (strain_bot - strain_top) / h",
            }
        )

    if placeholder and scale_ratio_std is not None and np.isfinite(scale_ratio_std) and scale_ratio_std > 1e3:
        suggestions.append(
            {
                "action": "kappa_scale_mismatch",
                "reason": "kappa_t scale differs from FD estimate by >1e3",
                "command": "recompute kappa_t from top/bottom or disable loss_kappa",
            }
        )

    if info.get("has_strain_top") and info.get("has_strain_bot") and placeholder:
        suggestions.append(
            {
                "action": "use_existing_top_bottom_fields",
                "reason": "strain_top/strain_bot columns exist but placeholder is active",
                "command": "ensure prepare_stage2.py is run with --top/--bottom inputs",
            }
        )

    report = {
        "inputs": {
            "parquet": args.parquet,
            "manifest": args.manifest,
            "data_dir": args.data_dir,
            "num_files": len(paths),
            "first_file": str(paths[0]),
        },
        "placeholder_kappa": placeholder,
        "placeholder_reason": placeholder_reason,
        "kappa_stats": kappa_summary,
        "strain_stats": strain_summary,
        "kappa_fd_stats": kappa_fd_summary,
        "kappa_fd_meta": kappa_fd_meta,
        "scale_ratio_std": scale_ratio_std,
        "scale_ratio_fd_std": scale_ratio_fd_std,
        "scale_ratio_strain_std": scale_ratio_strain_std,
        "Lx": args.Lx,
        "Ly": args.Ly,
        "h": args.h,
        "suggestions": suggestions,
        "source_inspection": info,
    }

    out_json = _default_out_json(args)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("USING_PLACEHOLDER_KAPPA =", placeholder, f"(reason={placeholder_reason})")
    print("kappa_t abs_max/mean/std =", kappa_summary)
    if kappa_fd_summary is not None:
        print("kappa_fd abs_max/mean/std =", kappa_fd_summary)
    if scale_ratio_std is not None and np.isfinite(scale_ratio_std):
        print("scale_ratio_std =", scale_ratio_std)
    for suggestion in suggestions:
        print("SUGGEST:", suggestion["action"], "-", suggestion["reason"])
        print("  CMD:", suggestion["command"])
    print(json.dumps(report, indent=2))
    print("Saved JSON:", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
