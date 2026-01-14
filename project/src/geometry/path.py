"""
Path reconstruction and geometric utilities for fiber data exported from COMSOL.

工程背景（必须保留）:
- 我们有 COMSOL 导出的光纤路径点集（列: x, y, z, strain, u, v, w），需要沿光纤弧长顺序组织成序列以便后续模型训练。
- COMSOL 导出可能出现点顺序不严格、存在多个分段（段与段之间有“大跳点”距离）、以及重复点或极近点（数值噪声）。
- 光纤真实路径是固定的，但当前阶段不依赖用户提供的名义路径模板，先做可用的自动近似排序、弧长与切线计算。
"""

from __future__ import annotations

import warnings
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import KDTree


ArrayLike = Sequence[Sequence[float]] | np.ndarray | pd.DataFrame


def _to_numpy(points: ArrayLike, dims: int, columns: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize various input containers to a numpy array and positional indices.

    Args:
        points: Array-like input (ndarray, DataFrame, or sequence of sequences).
        dims: Number of leading dimensions to keep.
        columns: Column names expected when ``points`` is a DataFrame.

    Returns:
        tuple of (array shape [N, dims], positional indices [N]).
    """
    if isinstance(points, pd.DataFrame):
        missing = [col for col in columns if col not in points.columns]
        if missing:
            raise ValueError(f"Missing required columns {missing} in DataFrame input")
        array = points[list(columns)].to_numpy(dtype=float)
        indices = np.arange(len(points), dtype=int)
    else:
        array = np.asarray(points, dtype=float)
        if array.ndim != 2 or array.shape[1] < dims:
            raise ValueError(f"Expected shape [N,{dims}] or more, got {array.shape}")
        array = array[:, :dims]
        indices = np.arange(array.shape[0], dtype=int)
    return array, indices


def split_into_segments(
    points_xyz: ArrayLike,
    jump_threshold: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Split a 3D point sequence into segments based on large jumps.

    The function walks through the input order, computes pairwise distances of
    consecutive points, and starts a new segment whenever the distance exceeds
    ``jump_threshold``. Short segments (e.g., containing <2 points) are retained
    but marked via warnings to avoid silent drops.

    Args:
        points_xyz: Array-like container with shape [N, 3] or a DataFrame containing
            columns ``x, y, z``.
        jump_threshold: Euclidean distance above which a discontinuity is assumed.

    Returns:
        segments: List of numpy arrays, each with shape [Mi, 3].
        segment_indices: List of numpy arrays, positional indices corresponding to
            the original input order for each segment.
    """
    coords, base_indices = _to_numpy(points_xyz, dims=3, columns=("x", "y", "z"))

    if coords.size == 0:
        return [], []

    segments: list[np.ndarray] = []
    segment_indices: list[np.ndarray] = []
    short_segments: list[int] = []

    start = 0
    deltas = coords[1:] - coords[:-1]
    dists = np.linalg.norm(deltas, axis=1)

    for i, dist in enumerate(dists, start=1):
        if dist > jump_threshold:
            segments.append(coords[start:i])
            segment_indices.append(base_indices[start:i])
            if i - start < 2:
                short_segments.append(len(segments) - 1)
            start = i

    segments.append(coords[start:])
    segment_indices.append(base_indices[start:])
    if coords.shape[0] - start < 2:
        short_segments.append(len(segments) - 1)

    if short_segments:
        warnings.warn(
            f"Short segments detected (length < 2) at indices: {short_segments}",
            RuntimeWarning,
        )

    return segments, segment_indices


def _deduplicate_points(points_xy: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove duplicate or near-duplicate points, preserving first occurrences."""
    kept_points: list[np.ndarray] = []
    kept_indices: list[int] = []
    removed = 0

    for idx, point in enumerate(points_xy):
        if not kept_points:
            kept_points.append(point)
            kept_indices.append(idx)
            continue

        existing = np.vstack(kept_points)
        dists = np.linalg.norm(existing - point, axis=1)
        if np.any(dists < eps):
            removed += 1
            continue

        kept_points.append(point)
        kept_indices.append(idx)

    return np.asarray(kept_points, dtype=float), np.asarray(kept_indices, dtype=int), removed


def _choose_tie_candidate(
    rng: np.random.Generator,
    candidates: np.ndarray,
) -> int:
    """Deterministically choose among tied candidates using RNG for reproducibility."""
    if len(candidates) == 1:
        return int(candidates[0])
    choice = rng.integers(len(candidates))
    return int(candidates[choice])


def order_path(
    points_xy: ArrayLike,
    mode: str = "as_is",
    *,
    k: int = 6,
    dedup_eps: float = 1e-9,
    random_state: int | np.random.Generator = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Order 2D points into an approximate path and deduplicate close points.

    Two modes are supported:
    - ``as_is``: Assume the incoming order already follows the path; only deduplicate.
    - ``nn_graph``: Build an undirected nearest-neighbor graph via KDTree and grow
      a greedy walk that approximates a Hamiltonian path.

    nn_graph 的可复现规则（实现要点）:
    1) 使用 KDTree，为每个点查询 ``k`` 个最近邻（不含自身），构建无向图。
    2) 端点估计: 度数最小的点作为起点；若有多个候选，选择 (x, y) 字典序最小者。
    3) 路径生成: 从起点出发，优先选择“未访问邻居”中距离最近者；若无未访问邻居，则
       在全体未访问点里选距离最近者（记录为跳边）。当距离出现并列时，使用
       ``random_state`` 提供的 RNG 进行可复现的 tie-break。
    4) 允许路径自交，不引入 TSP/ILP 等重量级求解。

    Args:
        points_xy: Array-like container with shape [N, 2] or DataFrame with columns
            ``x, y``.
        mode: ``'as_is'`` or ``'nn_graph'``.
        k: Number of nearest neighbors for graph construction.
        dedup_eps: Distance tolerance below which points are considered duplicates.
        random_state: Seed or Generator used for deterministic tie-breaks.

    Returns:
        ordered_xy: Ordered points after deduplication, shape [M, 2].
        ordered_indices: Indices of the kept points relative to the original input.
        debug: Dictionary with diagnostic information (mode, n_in, n_out,
            dedup_removed, endpoint_strategy, k_used, jump_edges, max_jump, tie_breaks).
    """
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    coords, base_indices = _to_numpy(points_xy, dims=2, columns=("x", "y"))
    n_in = coords.shape[0]

    deduped_xy, dedup_indices, removed = _deduplicate_points(coords, dedup_eps)
    if mode not in {"as_is", "nn_graph"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if mode == "as_is" or deduped_xy.shape[0] <= 1:
        debug = {
            "mode": mode,
            "n_in": n_in,
            "n_out": int(deduped_xy.shape[0]),
            "dedup_removed": int(removed),
            "endpoint_strategy": "input_order" if mode == "as_is" else "min_degree",
            "k_used": 0 if mode == "as_is" else min(k, max(deduped_xy.shape[0] - 1, 0)),
            "jump_edges": 0,
            "max_jump": 0.0,
            "tie_breaks": 0,
        }
        return deduped_xy, base_indices[dedup_indices], debug

    n_points = deduped_xy.shape[0]
    k_used = int(min(k, max(n_points - 1, 1)))
    tree = KDTree(deduped_xy)

    distances, neighbors = tree.query(deduped_xy, k=k_used + 1)
    adjacency: list[dict[int, float]] = [dict() for _ in range(n_points)]

    for i in range(n_points):
        for dist, nbr in zip(distances[i], neighbors[i]):
            if nbr == i:
                continue
            j = int(nbr)
            # ensure symmetric graph with minimal distance
            weight = float(dist)
            adjacency[i][j] = min(adjacency[i].get(j, weight), weight)
            adjacency[j][i] = min(adjacency[j].get(i, weight), weight)

    degrees = np.array([len(adj) for adj in adjacency], dtype=int)
    min_degree = int(degrees.min()) if len(degrees) else 0
    endpoint_candidates = np.where(degrees == min_degree)[0]
    if len(endpoint_candidates) > 1:
        lex_order = np.lexsort((deduped_xy[endpoint_candidates, 1], deduped_xy[endpoint_candidates, 0]))
        endpoint_candidates = endpoint_candidates[lex_order]
    start_idx = int(endpoint_candidates[0])

    path: list[int] = []
    visited = np.zeros(n_points, dtype=bool)
    jump_edges = 0
    max_jump = 0.0
    tie_breaks = 0
    current = start_idx

    while len(path) < n_points:
        path.append(current)
        visited[current] = True
        if len(path) == n_points:
            break

        neighbors_unvisited = [
            (nbr, dist) for nbr, dist in adjacency[current].items() if not visited[nbr]
        ]
        if neighbors_unvisited:
            dists = np.array([dist for _, dist in neighbors_unvisited], dtype=float)
            nb_nodes = np.array([nbr for nbr, _ in neighbors_unvisited], dtype=int)
            min_dist = float(dists.min())
            tied = nb_nodes[np.isclose(dists, min_dist, rtol=1e-9, atol=1e-12)]
        else:
            remaining = np.where(~visited)[0]
            deltas = deduped_xy[remaining] - deduped_xy[current]
            dists = np.linalg.norm(deltas, axis=1)
            min_dist = float(dists.min())
            tied = remaining[np.isclose(dists, min_dist, rtol=1e-9, atol=1e-12)]
            jump_edges += 1
            max_jump = max(max_jump, min_dist)

        if len(tied) > 1:
            tie_breaks += 1
        next_idx = _choose_tie_candidate(rng, tied)
        current = next_idx

    ordered_indices = dedup_indices[np.array(path, dtype=int)]
    ordered_xy = deduped_xy[np.array(path, dtype=int)]
    debug = {
        "mode": mode,
        "n_in": n_in,
        "n_out": int(deduped_xy.shape[0]),
        "dedup_removed": int(removed),
        "endpoint_strategy": "min_degree_lexicographic",
        "k_used": k_used,
        "jump_edges": int(jump_edges),
        "max_jump": float(max_jump),
        "tie_breaks": int(tie_breaks),
        "start_index": int(start_idx),
    }
    return ordered_xy, base_indices[ordered_indices], debug


def _nearest_nonzero_delta(points: np.ndarray, idx: int) -> np.ndarray:
    """Find the nearest non-zero difference vector around ``idx``."""
    n = points.shape[0]
    for offset in range(1, n):
        prev = idx - offset
        if prev >= 0:
            delta_prev = points[idx] - points[prev]
            if np.linalg.norm(delta_prev) > 0:
                return delta_prev
        nxt = idx + offset
        if nxt < n:
            delta_next = points[nxt] - points[idx]
            if np.linalg.norm(delta_next) > 0:
                return delta_next
    return np.zeros(2, dtype=float)


def compute_arclength_and_tangent(xy_ordered: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cumulative arclength and unit tangents for an ordered 2D path.

    Rules:
    1) ``s[0] = 0`` and ``s[i] = s[i-1] + ||p[i] - p[i-1]||``.
    2) Tangents use forward/backward differences at endpoints and central
       differences for interior points.
    3) Tangent vectors are normalized; zero vectors (caused by duplicate points)
       are handled safely by searching the nearest non-zero delta, otherwise set
       to zero while keeping computation deterministic.
    """
    points = np.asarray(xy_ordered, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected shape [M,2], got {points.shape}")

    n = points.shape[0]
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
    if n == 1:
        return np.array([0.0], dtype=float), np.array([0.0], dtype=float), np.array([0.0], dtype=float)

    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    tangents = np.zeros_like(points)
    zero_tangent_count = 0

    for i in range(n):
        if i == 0:
            delta = points[1] - points[0]
        elif i == n - 1:
            delta = points[-1] - points[-2]
        else:
            delta = points[i + 1] - points[i - 1]

        norm = np.linalg.norm(delta)
        if norm == 0:
            delta = _nearest_nonzero_delta(points, i)
            norm = np.linalg.norm(delta)

        if norm == 0:
            zero_tangent_count += 1
            tangents[i] = np.array([0.0, 0.0], dtype=float)
        else:
            tangents[i] = delta / norm

    if zero_tangent_count:
        warnings.warn(
            f"{zero_tangent_count} zero-length tangents encountered; set to 0 vector.",
            RuntimeWarning,
        )

    tx = tangents[:, 0]
    ty = tangents[:, 1]
    return s, tx, ty


def add_path_features(
    df: pd.DataFrame,
    *,
    mode: str = "as_is",
    jump_threshold: float | None = None,
    k: int = 6,
    dedup_eps: float = 1e-9,
    random_state: int | np.random.Generator = 0,
) -> pd.DataFrame:
    """
    Convenience wrapper: order points, compute arclength/tangent, and append features.

    When ``jump_threshold`` is provided, the function first splits the points using
    ``split_into_segments`` (preserving input order), processes each segment
    independently, then concatenates results while maintaining deterministic
    ordering.

    Args:
        df: DataFrame containing at least ``x, y, z`` columns.
        mode: Ordering mode for each segment (``as_is`` or ``nn_graph``).
        jump_threshold: Distance threshold to trigger segmentation; if None, the
            entire set is treated as one segment.
        k: Nearest-neighbor count when ``mode='nn_graph'``.
        dedup_eps: Duplicate tolerance passed to ``order_path``.
        random_state: Seed/Generator for reproducible tie-breaks.

    Returns:
        A new DataFrame ordered along the reconstructed path with additional columns
        ``s, tx, ty``. Input columns are preserved.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("add_path_features expects a pandas DataFrame input")

    if jump_threshold is None:
        ordered_xy, ordered_idx, debug = order_path(
            df[["x", "y"]],
            mode=mode,
            k=k,
            dedup_eps=dedup_eps,
            random_state=random_state,
        )
        s, tx, ty = compute_arclength_and_tangent(ordered_xy)
        df_out = df.iloc[ordered_idx].copy()
        df_out["s"] = s
        df_out["tx"] = tx
        df_out["ty"] = ty
        df_out.attrs["path_debug"] = {"segments": 1, "segment_debug": [debug]}
        return df_out

    segments, seg_indices = split_into_segments(df[["x", "y", "z"]], jump_threshold)
    frames: list[pd.DataFrame] = []
    segment_debug: list[dict] = []
    s_offset = 0.0

    for seg_id, (seg_points, seg_idx) in enumerate(zip(segments, seg_indices)):
        ordered_xy, ordered_rel_idx, dbg = order_path(
            seg_points[:, :2],
            mode=mode,
            k=k,
            dedup_eps=dedup_eps,
            random_state=random_state,
        )
        ordered_orig_idx = seg_idx[ordered_rel_idx]
        s, tx, ty = compute_arclength_and_tangent(ordered_xy)
        s_shifted = s + s_offset

        seg_df = df.iloc[ordered_orig_idx].copy()
        seg_df["s"] = s_shifted
        seg_df["tx"] = tx
        seg_df["ty"] = ty
        frames.append(seg_df)

        segment_debug.append(
            {
                "segment_id": seg_id,
                "n_raw": int(seg_points.shape[0]),
                "n_ordered": int(ordered_xy.shape[0]),
                "order_debug": dbg,
                "s_offset": float(s_offset),
            }
        )

        if s_shifted.size:
            s_offset = float(s_shifted[-1])
            # include inter-segment jump length for cumulative arclength continuity
            if seg_id < len(segments) - 1:
                jump = np.linalg.norm(segments[seg_id + 1][0, :2] - seg_points[-1, :2])
                s_offset += jump

    df_out = pd.concat(frames, ignore_index=True)
    df_out.attrs["path_debug"] = {"segments": len(frames), "segment_debug": segment_debug}
    return df_out


__all__ = [
    "split_into_segments",
    "order_path",
    "compute_arclength_and_tangent",
    "add_path_features",
]
