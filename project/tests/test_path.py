from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.geometry.path import compute_arclength_and_tangent, order_path, split_into_segments


def test_compute_arclength_and_tangent_basic():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=float)

    s, tx, ty = compute_arclength_and_tangent(points)

    assert s[0] == pytest.approx(0.0)
    assert np.all(np.diff(s) >= -1e-9)
    norms = np.linalg.norm(np.stack([tx, ty], axis=1), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    assert tx[0] == pytest.approx(1.0)
    assert ty[-1] == pytest.approx(1.0)


def test_split_into_segments_detects_jump():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [100.0, 100.0, 0.0],
            [101.0, 101.0, 0.0],
        ]
    )
    segments, indices = split_into_segments(pts, jump_threshold=10.0)

    assert len(segments) == 2
    assert segments[0].shape == (2, 3)
    assert segments[1].shape == (2, 3)
    assert np.array_equal(indices[0], np.array([0, 1]))
    assert np.array_equal(indices[1], np.array([2, 3]))


def test_order_path_as_is_dedup_preserves_first_occurrence():
    pts = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],  # duplicate
            [1e-10, 1e-10],  # near-duplicate
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )

    ordered_xy, ordered_indices, debug = order_path(pts, mode="as_is", dedup_eps=1e-6)

    assert ordered_xy.shape[0] == 3
    assert ordered_indices.tolist() == [0, 3, 4]
    assert debug["dedup_removed"] == 2
    assert debug["mode"] == "as_is"


def test_order_path_nn_graph_improves_continuity_on_shuffled_snake():
    # build a simple snake polyline on a grid
    rows, cols = 4, 5
    snake_points = []
    for r in range(rows):
        xs = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in xs:
            snake_points.append([float(c), float(r)])
    snake_points = np.array(snake_points, dtype=float)

    rng = np.random.default_rng(42)
    shuffled = snake_points[rng.permutation(len(snake_points))]

    unordered_mean = np.mean(np.linalg.norm(np.diff(shuffled, axis=0), axis=1))

    ordered_xy, ordered_idx, debug = order_path(
        shuffled, mode="nn_graph", k=4, random_state=123, dedup_eps=1e-9
    )
    s, tx, ty = compute_arclength_and_tangent(ordered_xy)
    ordered_mean = np.mean(np.linalg.norm(np.diff(ordered_xy, axis=0), axis=1))

    assert np.all(np.diff(s) >= -1e-9)
    assert ordered_mean < unordered_mean * 0.8
    assert ordered_xy.shape[0] == shuffled.shape[0]  # no dedup expected
    assert debug["mode"] == "nn_graph"
    assert debug["n_in"] == shuffled.shape[0]
