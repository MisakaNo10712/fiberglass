"""COMSOL TXT reader and validator.

背景说明（项目需求）:
- COMSOL 会导出纯文本数据文件 (txt)，每行表示一个点，字段顺序固定为 x y z strain u v w。
- 这些数据用于板形状重构工程，z 通常为板厚方向坐标，w 常作为挠度真值用于监督/评估。
- 分隔符可能是空格、多个空格或 Tab，文件可能带表头（大小写不定）也可能没有。

提供的主要接口:
- read_comsol_points: 读取文件并返回规范化的 DataFrame（列名、顺序和 dtype 均强制化）。
- validate_comsol_points: 对 DataFrame 做严格校验并生成可打印的报告。
- load_and_validate: 组合读取与校验，方便脚本直接使用。

设计目标:
- 严格校验，及时抛出可读的 ValueError，避免脏数据进入后续流程。
- 兼容历史接口（detect_header、read_comsol_txt、validate_dataframe、compute_column_stats）
  以便现有脚本和测试继续工作。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["x", "y", "z", "strain", "u", "v", "w"]


class ComsolTxtError(ValueError):
    """Base error for COMSOL txt parsing and validation."""


class ValidationError(ComsolTxtError):
    """Validation error (subclass of ValueError for compatibility)."""


def _normalize_columns(columns: Iterable[str]) -> list[str]:
    """Trim whitespace and lowercase column names for normalization."""
    return [str(col).strip().lower() for col in columns]


def _first_meaningful_line(file_path: Path) -> str | None:
    """Return the first non-empty, non-comment line from a text file."""
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip().lstrip("\ufeff")
            if not line:
                continue
            if line.startswith("#") or line.startswith("%"):
                continue
            return line
    return None


def detect_header(path: str | Path) -> bool:
    """
    Detect whether the file contains a header row.

    Logic: inspect the first non-empty, non-comment line. If every token can be
    parsed as float, we assume there is no header; otherwise header exists.
    """
    file_path = Path(path)
    first_line = _first_meaningful_line(file_path)
    if first_line is None:
        raise ComsolTxtError(f"File is empty or only contains comments: {file_path}")

    for token in first_line.split():
        try:
            float(token)
        except ValueError:
            return True
    return False


def _enforce_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist without extras and are ordered correctly."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    extra = [col for col in df.columns if col not in REQUIRED_COLUMNS]
    if missing:
        raise ComsolTxtError(f"Missing required columns: {missing}")
    if extra:
        raise ComsolTxtError(f"Unexpected extra columns: {extra}")
    if list(df.columns) != REQUIRED_COLUMNS:
        # Enforce canonical order for downstream consumers.
        df = df[REQUIRED_COLUMNS]
    try:
        df = df.astype(np.float64)
    except Exception as exc:  # noqa: BLE001
        raise ComsolTxtError(f"Expected all columns to be float64: {exc}") from exc
    return df


def read_comsol_points(
    path: str | Path,
    *,
    has_header: bool | None = None,
    sep: str | None = None,
) -> pd.DataFrame:
    """
    Read COMSOL-exported txt data into a pandas DataFrame.

    The resulting DataFrame will always have columns exactly
    ['x','y','z','strain','u','v','w'] in that order with dtype float64.

    Args:
        path: Path to the txt file.
        has_header: Whether the file has a header row; None triggers auto-detect.
        sep: Custom separator; None uses whitespace-aware regex (supports space/Tab mix).

    Raises:
        ValueError: If file is missing, columns are mismatched, or data cannot be parsed.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise ComsolTxtError(f"File not found: {file_path}")
    if file_path.is_dir():
        raise ComsolTxtError(f"Expected a file but got directory: {file_path}")

    header_present = has_header if has_header is not None else detect_header(file_path)
    separator = r"\s+" if sep is None else sep

    try:
        df = pd.read_csv(
            file_path,
            sep=separator,
            header=0 if header_present else None,
            names=None if header_present else REQUIRED_COLUMNS,
            dtype=np.float64,
            engine="python",
            skip_blank_lines=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise ComsolTxtError(f"Failed to read COMSOL txt {file_path}: {exc}") from exc

    if header_present:
        df.columns = _normalize_columns(df.columns)

    df = _enforce_required_columns(df)
    df.attrs["file_hint"] = str(file_path)
    return df


def validate_comsol_points(
    df: pd.DataFrame,
    *,
    min_rows: int = 10,
    require_columns: bool = True,
) -> dict:
    """
    Strictly validate a COMSOL points DataFrame and return a diagnostic report.

    Validation rules:
    - Row count must be >= min_rows (default 10).
    - Columns must match REQUIRED_COLUMNS (if require_columns=True).
    - All values must be finite (no NaN/inf/-inf).
    - All dtypes must be float64.

    Raises:
        ValueError: When any rule is violated with a descriptive message.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError("Input must be a pandas.DataFrame")

    n_rows, n_cols = df.shape
    columns = list(df.columns)

    if n_rows < min_rows:
        raise ValidationError(
            f"行数不足 / insufficient rows: expected >= {min_rows}, got {n_rows}"
        )

    columns_to_check = columns
    if require_columns:
        missing = [col for col in REQUIRED_COLUMNS if col not in columns]
        extra = [col for col in columns if col not in REQUIRED_COLUMNS]
        if missing:
            raise ValidationError(f"Missing required columns: {missing}")
        if extra:
            raise ValidationError(f"Unexpected extra columns: {extra}")
        if columns != REQUIRED_COLUMNS:
            raise ValidationError(
                f"Column order mismatch: expected {REQUIRED_COLUMNS}, got {columns}"
            )
        columns_to_check = REQUIRED_COLUMNS

    dtype_mismatch = {
        col: str(df[col].dtype) for col in columns_to_check if df[col].dtype != np.float64
    }
    if dtype_mismatch:
        raise ValidationError(f"Expected float64 dtype but got {dtype_mismatch}")

    non_finite_details: list[str] = []
    for col in columns_to_check:
        series = df[col]
        nan_count = int(series.isna().sum())
        pos_inf = int(np.isposinf(series).sum())
        neg_inf = int(np.isneginf(series).sum())
        if nan_count or pos_inf or neg_inf:
            parts = []
            if nan_count:
                parts.append(f"{nan_count} NaN")
            if pos_inf or neg_inf:
                parts.append(f"{pos_inf + neg_inf} inf")
            non_finite_details.append(f"{col}: {', '.join(parts)}")
    if non_finite_details:
        raise ValidationError(f"Found non-finite values: {', '.join(non_finite_details)}")

    report_columns = columns_to_check
    stats = {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
        }
        for col in report_columns
        if col in df.columns
    }
    unique_counts = {
        col: int(df[col].nunique(dropna=False)) for col in report_columns if col in df.columns
    }

    z_unique_values_preview: list[float] = []
    if "z" in df.columns:
        z_unique_values_preview = [float(v) for v in pd.unique(df["z"])[:5]]

    report = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": columns,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "stats": stats,
        "unique_counts": unique_counts,
        "z_is_single_value": bool(len(set(df["z"])) == 1) if "z" in df.columns else False,
        "z_unique_values_preview": z_unique_values_preview,
        "has_negative_w": bool((df["w"] < 0).any()) if "w" in df.columns else False,
    }

    file_hint = df.attrs.get("file_hint")
    if file_hint:
        report["file_hint"] = file_hint

    return report


def load_and_validate(
    path: str | Path,
    *,
    has_header: bool | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Convenience helper that reads and validates a COMSOL txt file.

    Returns:
        (DataFrame, report)
    """
    df = read_comsol_points(path, has_header=has_header)
    report = validate_comsol_points(df)
    if "file_hint" not in report:
        report["file_hint"] = str(Path(path))
    return df, report


# ----- Backward compatibility helpers (kept for existing scripts/tests) -----

def read_comsol_txt(
    file_path: str | Path,
    has_header: bool | None = None,
    validate: bool | None = None,
    sep: str | None = None,
) -> pd.DataFrame:
    """Legacy alias that optionally validates after reading."""
    df = read_comsol_points(file_path, has_header=has_header, sep=sep)
    if validate is None or validate is True:
        validate_comsol_points(df)
    return df


def validate_dataframe(df: pd.DataFrame, source: str | None = None) -> dict:
    """
    Legacy alias for validate_comsol_points.

    Args:
        df: DataFrame to validate.
        source: Optional text used as file_hint in the report.
    """
    if source:
        df = df.copy()
        df.attrs["file_hint"] = source
    return validate_comsol_points(df)


def compute_column_stats(df: pd.DataFrame) -> dict:
    """Return min/max/mean for required columns present in df."""
    available_columns = [col for col in REQUIRED_COLUMNS if col in df.columns]
    return {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
        }
        for col in available_columns
    }


__all__ = [
    "REQUIRED_COLUMNS",
    "read_comsol_points",
    "validate_comsol_points",
    "load_and_validate",
    "detect_header",
    "read_comsol_txt",
    "validate_dataframe",
    "compute_column_stats",
    "ComsolTxtError",
    "ValidationError",
]
