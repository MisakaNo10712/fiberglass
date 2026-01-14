from __future__ import annotations

import importlib.util
import tempfile
from io import StringIO
from pathlib import Path

import numpy as np
import pytest


def _load_comsol_module():
    module_path = Path(__file__).parent.parent / "src" / "io" / "comsol_txt.py"
    spec = importlib.util.spec_from_file_location("fiberglass_comsol_txt", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


comsol_txt = _load_comsol_module()
REQUIRED_COLUMNS = comsol_txt.REQUIRED_COLUMNS
read_comsol_points = comsol_txt.read_comsol_points
validate_comsol_points = comsol_txt.validate_comsol_points
load_and_validate = comsol_txt.load_and_validate


def _write_buffer_to_temp(data: str) -> Path:
    """Persist StringIO content to a temp file and return its path."""
    buffer = StringIO(data)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as handle:
        handle.write(buffer.getvalue())
        temp_path = Path(handle.name)
    return temp_path


HEADERED_DATA = """x y z strain u v w
0.0 0.0 0.0 0.0001 0.010 0.020 0.030
0.1 0.0 0.0 0.0002 0.011 0.021 0.031
0.2 0.0 0.0 0.0003 0.012 0.022 0.032
0.3 0.0 0.0 0.0004 0.013 0.023 0.033
0.4 0.0 0.0 0.0005 0.014 0.024 0.034
0.5 0.0 0.0 0.0006 0.015 0.025 0.035
0.6 0.0 0.0 0.0007 0.016 0.026 0.036
0.7 0.0 0.0 0.0008 0.017 0.027 0.037
0.8 0.0 0.0 0.0009 0.018 0.028 0.038
0.9 0.0 0.0 0.0010 0.019 0.029 0.039
"""

NO_HEADER_DATA = """0.0 0.0 0.0 0.0001 0.010 0.020 0.030
0.1 0.0 0.0 0.0002 0.011 0.021 0.031
0.2 0.0 0.0 0.0003 0.012 0.022 0.032
0.3 0.0 0.0 0.0004 0.013 0.023 0.033
0.4 0.0 0.0 0.0005 0.014 0.024 0.034
0.5 0.0 0.0 0.0006 0.015 0.025 0.035
0.6 0.0 0.0 0.0007 0.016 0.026 0.036
0.7 0.0 0.0 0.0008 0.017 0.027 0.037
0.8 0.0 0.0 0.0009 0.018 0.028 0.038
0.9 0.0 0.0 0.0010 0.019 0.029 0.039
"""

SHORT_DATA = """x y z strain u v w
0.0 0.0 0.0 0.0001 0.01 0.02 0.03
0.1 0.0 0.0 0.0002 0.01 0.02 0.03
0.2 0.0 0.0 0.0003 0.01 0.02 0.03
0.3 0.0 0.0 0.0004 0.01 0.02 0.03
0.4 0.0 0.0 0.0005 0.01 0.02 0.03
"""

NON_FINITE_DATA = """x y z strain u v w
0.0 0.0 0.0 0.0001 0.01 0.02 0.03
0.1 0.0 0.0 0.0002 0.01 0.02 inf
0.2 0.0 0.0 0.0003 0.01 0.02 0.03
0.3 0.0 0.0 0.0004 0.01 0.02 0.03
0.4 0.0 0.0 0.0005 nan 0.02 0.03
0.5 0.0 0.0 0.0006 0.01 0.02 0.03
0.6 0.0 0.0 0.0007 0.01 0.02 0.03
0.7 0.0 0.0 0.0008 0.01 0.02 0.03
0.8 0.0 0.0 0.0009 0.01 0.02 0.03
0.9 0.0 0.0 0.0010 0.01 0.02 0.03
"""


def test_read_with_header_and_validate_report():
    path = _write_buffer_to_temp(HEADERED_DATA)
    try:
        df = read_comsol_points(path, has_header=True)
        assert list(df.columns) == REQUIRED_COLUMNS
        assert all(dtype == np.float64 for dtype in df.dtypes)

        report = validate_comsol_points(df)
        required_keys = {
            "n_rows",
            "n_cols",
            "columns",
            "dtypes",
            "stats",
            "unique_counts",
            "z_is_single_value",
            "z_unique_values_preview",
            "has_negative_w",
        }
        assert required_keys.issubset(report.keys())
    finally:
        path.unlink(missing_ok=True)


def test_read_without_header_and_validate():
    path = _write_buffer_to_temp(NO_HEADER_DATA)
    try:
        df = read_comsol_points(path, has_header=False)
        assert list(df.columns) == REQUIRED_COLUMNS
        assert all(dtype == np.float64 for dtype in df.dtypes)

        report = validate_comsol_points(df)
        assert report["n_rows"] == 10
        assert report["z_is_single_value"] is True
    finally:
        path.unlink(missing_ok=True)


def test_validation_fails_for_short_dataset():
    path = _write_buffer_to_temp(SHORT_DATA)
    try:
        df = read_comsol_points(path, has_header=True)
        with pytest.raises(ValueError):
            validate_comsol_points(df, min_rows=10)
    finally:
        path.unlink(missing_ok=True)


def test_validation_fails_on_nan_or_inf():
    path = _write_buffer_to_temp(NON_FINITE_DATA)
    try:
        df = read_comsol_points(path, has_header=True)
        with pytest.raises(ValueError, match="non-finite|NaN|inf"):
            validate_comsol_points(df)
    finally:
        path.unlink(missing_ok=True)


def test_load_and_validate_sets_file_hint():
    path = _write_buffer_to_temp(HEADERED_DATA)
    try:
        _, report = load_and_validate(path, has_header=True)
        assert report["file_hint"] == str(path)
    finally:
        path.unlink(missing_ok=True)
