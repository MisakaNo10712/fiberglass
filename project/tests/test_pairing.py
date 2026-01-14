import numpy as np
import pandas as pd

from src.datasets import pair_top_bottom


def test_pair_by_s_interpolate_basic() -> None:
    top = pd.DataFrame(
        {
            "s": [0.0, 1.0, 2.0, 3.0],
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "tx": [1.0, 1.0, 1.0, 1.0],
            "ty": [0.0, 0.0, 0.0, 0.0],
            "strain": [10.0, 20.0, 30.0, 40.0],
        }
    )
    bot = pd.DataFrame(
        {
            "s": [0.0, 1.0, 2.0, 3.0],
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "tx": [1.0, 1.0, 1.0, 1.0],
            "ty": [0.0, 0.0, 0.0, 0.0],
            "strain": [11.0, 21.0, 31.0, 41.0],
        }
    )
    df = pair_top_bottom(top, bot, method="s", h=2.0)

    assert np.allclose(df["strain_top"].to_numpy(), [10.0, 20.0, 30.0, 40.0])
    assert np.allclose(df["strain_bot"].to_numpy(), [11.0, 21.0, 31.0, 41.0])
    assert np.allclose(df["kappa_t"].to_numpy(), [0.5, 0.5, 0.5, 0.5])
    assert np.allclose(df["mask"].to_numpy(), [1.0, 1.0, 1.0, 1.0])
    assert np.isfinite(df.to_numpy()).all()


def test_pair_by_xy_kdtree_with_tol() -> None:
    top = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "strain": [10.0, 20.0],
        }
    )
    bot = pd.DataFrame(
        {
            "x": [0.0, 100.0],
            "y": [0.0, 100.0],
            "strain": [11.0, 21.0],
        }
    )
    df = pair_top_bottom(top, bot, method="xy", xy_tol=0.5)

    assert df.loc[0, "mask"] == 1.0
    assert df.loc[1, "mask"] == 0.0
    assert df.loc[1, "strain_bot"] == 0.0
    assert df.loc[1, "kappa_t"] == 0.0
    assert np.isfinite(df.to_numpy()).all()


def test_method_s_fallback_to_xy_when_missing_s() -> None:
    top = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "strain": [10.0, 20.0],
        }
    )
    bot = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "strain": [11.0, 21.0],
        }
    )
    df, debug = pair_top_bottom(top, bot, method="s", return_debug=True)

    assert debug["method_used"] == "xy"
    assert "s" in df.columns
    assert np.allclose(df["mask"].to_numpy(), [1.0, 1.0])


def test_invalid_h_raises() -> None:
    top = pd.DataFrame({"x": [0.0], "y": [0.0], "strain": [1.0]})
    bot = pd.DataFrame({"x": [0.0], "y": [0.0], "strain": [1.0]})
    try:
        pair_top_bottom(top, bot, h=0.0)
    except ValueError as exc:
        assert "h must be > 0" in str(exc)
    else:
        raise AssertionError("Expected ValueError for h <= 0.")
