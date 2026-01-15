"""
Visualization helpers for fiberglass inference.

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

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _meshgrid_from_axes(xg: np.ndarray, yg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xg_arr = np.asarray(xg, dtype=float)
    yg_arr = np.asarray(yg, dtype=float)
    if xg_arr.ndim == 1 and yg_arr.ndim == 1:
        return np.meshgrid(xg_arr, yg_arr)
    return xg_arr, yg_arr


def plot_surface_3d(w_grid, xg, yg, out_path, *, title: str = "") -> None:
    w = np.asarray(w_grid, dtype=float)
    if w.ndim == 3:
        w = np.squeeze(w)
    Xg, Yg = _meshgrid_from_axes(xg, yg)

    fig = plt.figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Xg, Yg, w, linewidth=0, antialiased=True)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("W")
    if title:
        ax.set_title(title)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_kappa_curve(s, kappa_meas, kappa_pred, out_path, *, title: str = "") -> None:
    s_arr = np.asarray(s, dtype=float).ravel()
    meas = np.asarray(kappa_meas, dtype=float).ravel()
    pred = np.asarray(kappa_pred, dtype=float).ravel()
    if s_arr.size != meas.size or pred.size != meas.size:
        raise ValueError("s, kappa_meas, and kappa_pred must have the same length.")

    order = np.argsort(s_arr)
    s_arr = s_arr[order]
    meas = meas[order]
    pred = pred[order]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(s_arr, meas, label="kappa_meas")
    ax.plot(s_arr, pred, label="kappa_pred")
    ax.set_xlabel("s")
    ax.set_ylabel("kappa_t")
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_error_heatmap(err_grid, xg, yg, out_path, *, title: str = "") -> None:
    err = np.asarray(err_grid, dtype=float)
    if err.ndim == 3:
        err = np.squeeze(err)
    err = np.nan_to_num(err)

    Xg, Yg = _meshgrid_from_axes(xg, yg)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    pcm = ax.pcolormesh(Xg, Yg, err, shading="auto")
    fig.colorbar(pcm, ax=ax)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title:
        ax.set_title(title)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def scatter_to_grid(x, y, values, xg, yg, method: str = "linear") -> np.ndarray:
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    v_arr = np.asarray(values, dtype=float).ravel()

    Xg, Yg = _meshgrid_from_axes(xg, yg)
    if x_arr.size == 0 or y_arr.size == 0:
        return np.zeros_like(Xg, dtype=float)

    points = np.column_stack([x_arr, y_arr])

    grid = None
    try:
        from scipy.interpolate import griddata

        grid = griddata(points, v_arr, (Xg, Yg), method=method)
    except Exception:
        grid = None

    if grid is None or not np.isfinite(grid).all():
        try:
            from scipy.interpolate import griddata

            grid_nearest = griddata(points, v_arr, (Xg, Yg), method="nearest")
        except Exception:
            grid_nearest = np.zeros_like(Xg, dtype=float)

        if grid is None:
            grid = grid_nearest
        else:
            grid = np.where(np.isfinite(grid), grid, grid_nearest)

    return grid


__all__ = [
    "plot_surface_3d",
    "plot_kappa_curve",
    "plot_error_heatmap",
    "scatter_to_grid",
]
