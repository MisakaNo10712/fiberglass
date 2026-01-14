"""
上下表面配对与 kappa_t 计算。

工程背景：
我们在薄板上上下表面各布置一根光纤，沿同一条名义路径采样得到两份序列数据
df_top 与 df_bot。每个序列点包含坐标与几何特征（至少 x,y；推荐包含 s,tx,ty），
以及该点的轴向应变 strain。我们希望通过差分消除温漂/膜内同相项，得到沿光纤
切线方向的曲率投影：
  kappa_t = (strain_bot - strain_top) / h
其中 h 是板厚（单位 m）。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

__all__ = ["pair_top_bottom"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    seen: dict[str, str] = {}
    for col in df.columns:
        lowered = col.lower()
        if lowered in seen and seen[lowered] != col:
            raise ValueError(f"Ambiguous columns after normalization: {seen[lowered]} vs {col}")
        seen[lowered] = col
        rename[col] = lowered
    df_copy = df.copy()
    df_copy.columns = [rename[col] for col in df.columns]
    return df_copy


def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _finite_range(values: np.ndarray) -> tuple[float, float] | None:
    finite = np.isfinite(values)
    if not np.any(finite):
        return None
    return float(np.min(values[finite])), float(np.max(values[finite]))


def _nearest_distance(sorted_points: np.ndarray, targets: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_points, targets)
    left = np.clip(idx - 1, 0, len(sorted_points) - 1)
    right = np.clip(idx, 0, len(sorted_points) - 1)
    left_dist = np.abs(targets - sorted_points[left])
    right_dist = np.abs(sorted_points[right] - targets)
    return np.minimum(left_dist, right_dist)


def _fill_nearest(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if not np.any(valid_mask):
        return np.zeros_like(values)
    idx = np.arange(values.shape[0])
    valid_idx = idx[valid_mask]
    valid_values = values[valid_mask]
    pos = np.searchsorted(valid_idx, idx)
    left = np.clip(pos - 1, 0, len(valid_idx) - 1)
    right = np.clip(pos, 0, len(valid_idx) - 1)
    left_dist = np.abs(idx - valid_idx[left])
    right_dist = np.abs(valid_idx[right] - idx)
    choose_right = right_dist < left_dist
    nearest = np.where(choose_right, right, left)
    return valid_values[nearest]


def _bot_series(df_bot: pd.DataFrame, field: str) -> tuple[np.ndarray, np.ndarray] | None:
    s_bot = df_bot["s"].to_numpy(dtype=float)
    values = df_bot[field].to_numpy(dtype=float)
    valid = np.isfinite(s_bot) & np.isfinite(values)
    if not np.any(valid):
        return None
    df_series = pd.DataFrame({"s": s_bot[valid], "v": values[valid]})
    df_series = df_series.groupby("s", as_index=False).mean()
    df_series = df_series.sort_values("s", kind="mergesort")
    return df_series["s"].to_numpy(), df_series["v"].to_numpy()


def _pair_by_s_interpolate(
    df_top: pd.DataFrame,
    df_bot: pd.DataFrame,
    *,
    fields: Iterable[str],
    s_tol: float | None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    n_top = len(df_top)
    s_top = df_top["s"].to_numpy(dtype=float)
    valid_s_top = np.isfinite(s_top)

    outputs: dict[str, np.ndarray] = {field: np.zeros(n_top, dtype=float) for field in fields}
    strain_series = _bot_series(df_bot, "strain")
    if strain_series is None:
        return outputs, np.zeros(n_top, dtype=bool)

    bot_s, bot_strain = strain_series
    if bot_s.size == 0:
        return outputs, np.zeros(n_top, dtype=bool)

    in_range = np.zeros(n_top, dtype=bool)
    if np.any(valid_s_top):
        in_range[valid_s_top] = (s_top[valid_s_top] >= bot_s[0]) & (
            s_top[valid_s_top] <= bot_s[-1]
        )

    nearest_dist = np.full(n_top, np.inf)
    if np.any(valid_s_top):
        nearest_dist[valid_s_top] = _nearest_distance(bot_s, s_top[valid_s_top])

    if s_tol is None:
        tol_mask = np.ones(n_top, dtype=bool)
    else:
        tol_mask = nearest_dist <= s_tol

    mask = valid_s_top & in_range & tol_mask

    if np.any(valid_s_top):
        outputs["strain"][valid_s_top] = np.interp(s_top[valid_s_top], bot_s, bot_strain)

    for field in fields:
        if field == "strain":
            continue
        series = _bot_series(df_bot, field)
        if series is None:
            continue
        s_vals, v_vals = series
        if np.any(valid_s_top):
            outputs[field][valid_s_top] = np.interp(s_top[valid_s_top], s_vals, v_vals)

    return outputs, mask


def _pair_by_xy_kdtree(
    df_top: pd.DataFrame,
    df_bot: pd.DataFrame,
    *,
    fields: Iterable[str],
    xy_tol: float | None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    n_top = len(df_top)
    outputs: dict[str, np.ndarray] = {field: np.zeros(n_top, dtype=float) for field in fields}

    top_x = df_top["x"].to_numpy(dtype=float)
    top_y = df_top["y"].to_numpy(dtype=float)
    valid_top_xy = np.isfinite(top_x) & np.isfinite(top_y)

    bot_x = df_bot["x"].to_numpy(dtype=float)
    bot_y = df_bot["y"].to_numpy(dtype=float)
    valid_bot_xy = np.isfinite(bot_x) & np.isfinite(bot_y)

    if not np.any(valid_bot_xy) or not np.any(valid_top_xy):
        return outputs, np.zeros(n_top, dtype=bool)

    bot_xy = np.column_stack([bot_x[valid_bot_xy], bot_y[valid_bot_xy]])
    tree = cKDTree(bot_xy)

    dist = np.full(n_top, np.inf)
    idx = np.zeros(n_top, dtype=int)

    top_xy = np.column_stack([top_x[valid_top_xy], top_y[valid_top_xy]])
    dist_valid, idx_valid = tree.query(top_xy, k=1)
    dist[valid_top_xy] = dist_valid
    idx[valid_top_xy] = idx_valid

    bot_indices = np.where(valid_bot_xy)[0]
    matched_indices = bot_indices[idx]

    for field in fields:
        values = df_bot[field].to_numpy(dtype=float)
        outputs[field] = values[matched_indices]

    if xy_tol is None:
        tol_mask = np.ones(n_top, dtype=bool)
    else:
        tol_mask = dist <= xy_tol
    mask = valid_top_xy & tol_mask
    return outputs, mask


def pair_top_bottom(
    df_top: pd.DataFrame,
    df_bot: pd.DataFrame,
    *,
    method: str = "s",
    h: float = 1.0,
    s_tol: float | None = None,
    xy_tol: float | None = None,
    fill_strategy: str = "zeros",
    return_debug: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    if h <= 0:
        raise ValueError("h must be > 0.")

    method = method.lower()
    if method not in {"s", "xy"}:
        raise ValueError("method must be 's' or 'xy'.")

    if fill_strategy not in {"zeros", "nearest"}:
        raise ValueError("fill_strategy must be 'zeros' or 'nearest'.")

    df_top = _normalize_columns(df_top)
    df_bot = _normalize_columns(df_bot)

    _require_columns(df_top, ["x", "y", "strain"])
    _require_columns(df_bot, ["x", "y", "strain"])

    n_top = len(df_top)
    n_bot = len(df_bot)

    has_s_top = "s" in df_top.columns
    has_s_bot = "s" in df_bot.columns
    has_tx_ty_top = "tx" in df_top.columns and "ty" in df_top.columns
    has_tx_ty_bot = "tx" in df_bot.columns and "ty" in df_bot.columns

    method_used = method
    fallback_reason: str | None = None
    if method == "s" and not (has_s_top and has_s_bot):
        method_used = "xy"
        fallback_reason = "missing s"

    if has_s_top:
        s_out = df_top["s"].to_numpy(dtype=float)
        s_pseudo_top = False
    else:
        s_out = np.arange(n_top, dtype=float)
        s_pseudo_top = True

    fields = ["strain"]
    use_bot_tx_ty = False
    if not has_tx_ty_top and has_tx_ty_bot:
        fields.extend(["tx", "ty"])
        use_bot_tx_ty = True

    if method_used == "s":
        pair_outputs, pair_mask = _pair_by_s_interpolate(
            df_top, df_bot, fields=fields, s_tol=s_tol
        )
    else:
        pair_outputs, pair_mask = _pair_by_xy_kdtree(
            df_top, df_bot, fields=fields, xy_tol=xy_tol
        )

    strain_bot = pair_outputs["strain"]

    top_x = df_top["x"].to_numpy(dtype=float)
    top_y = df_top["y"].to_numpy(dtype=float)
    strain_top = df_top["strain"].to_numpy(dtype=float)

    if use_bot_tx_ty:
        tx_out = pair_outputs.get("tx", np.zeros(n_top, dtype=float))
        ty_out = pair_outputs.get("ty", np.zeros(n_top, dtype=float))
        tx_ty_source = "bot"
    else:
        if has_tx_ty_top:
            tx_out = df_top["tx"].to_numpy(dtype=float)
            ty_out = df_top["ty"].to_numpy(dtype=float)
            tx_ty_source = "top"
        else:
            tx_out = np.zeros(n_top, dtype=float)
            ty_out = np.zeros(n_top, dtype=float)
            tx_ty_source = "zeros"

    finite_top_xy = np.isfinite(top_x) & np.isfinite(top_y)
    finite_s = np.isfinite(s_out)
    finite_strain_top = np.isfinite(strain_top)
    finite_strain_bot = np.isfinite(strain_bot)

    mask = pair_mask & finite_top_xy & finite_s & finite_strain_top & finite_strain_bot

    x_out = np.where(np.isfinite(top_x), top_x, 0.0)
    y_out = np.where(np.isfinite(top_y), top_y, 0.0)
    s_out = np.where(np.isfinite(s_out), s_out, 0.0)
    tx_out = np.where(np.isfinite(tx_out), tx_out, 0.0)
    ty_out = np.where(np.isfinite(ty_out), ty_out, 0.0)

    strain_top = np.where(np.isfinite(strain_top), strain_top, 0.0)
    strain_bot = np.where(np.isfinite(strain_bot), strain_bot, 0.0)

    strain_avg = 0.5 * (strain_top + strain_bot)
    kappa_t = (strain_bot - strain_top) / h

    if fill_strategy == "zeros":
        if not np.all(mask):
            invalid = ~mask
            strain_bot[invalid] = 0.0
            strain_avg[invalid] = 0.0
            kappa_t[invalid] = 0.0
    else:
        if np.any(mask):
            fill_bot = _fill_nearest(strain_bot, mask)
            fill_avg = _fill_nearest(strain_avg, mask)
            fill_kappa = _fill_nearest(kappa_t, mask)
            invalid = ~mask
            strain_bot[invalid] = fill_bot[invalid]
            strain_avg[invalid] = fill_avg[invalid]
            kappa_t[invalid] = fill_kappa[invalid]
        else:
            strain_bot[:] = 0.0
            strain_avg[:] = 0.0
            kappa_t[:] = 0.0

    invalid_rows = np.zeros(n_top, dtype=bool)
    for arr in (x_out, y_out, s_out, tx_out, ty_out, strain_top, strain_bot, strain_avg, kappa_t):
        invalid = ~np.isfinite(arr)
        if np.any(invalid):
            arr[invalid] = 0.0
            invalid_rows |= invalid
    if np.any(invalid_rows):
        mask[invalid_rows] = False

    mask_out = mask.astype(np.float32)

    df_merged = pd.DataFrame(
        {
            "x": x_out,
            "y": y_out,
            "s": s_out,
            "tx": tx_out,
            "ty": ty_out,
            "strain_top": strain_top,
            "strain_bot": strain_bot,
            "strain_avg": strain_avg,
            "kappa_t": kappa_t,
            "mask": mask_out,
        }
    )

    debug = {
        "method_used": method_used,
        "n_top": n_top,
        "n_bot": n_bot,
        "n_out": n_top,
        "n_failed": int(np.sum(~mask)),
        "has_s_top": has_s_top,
        "has_s_bot": has_s_bot,
        "has_tx_ty_top": has_tx_ty_top,
        "has_tx_ty_bot": has_tx_ty_bot,
        "s_range_top": _finite_range(df_top["s"].to_numpy(dtype=float))
        if has_s_top
        else None,
        "s_range_bot": _finite_range(df_bot["s"].to_numpy(dtype=float))
        if has_s_bot
        else None,
        "xy_tol": xy_tol,
        "s_tol": s_tol,
        "fill_strategy": fill_strategy,
        "fallback_reason": fallback_reason,
        "s_pseudo_top": s_pseudo_top,
        "tx_ty_source": tx_ty_source,
    }

    if return_debug:
        return df_merged, debug
    return df_merged
