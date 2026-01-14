"""
IO 模块 - 数据读取与写入

提供 COMSOL 数据文件的读取、校验和转换功能。
"""

from .comsol_txt import (  # noqa: F401
    REQUIRED_COLUMNS,
    compute_column_stats,
    detect_header,
    load_and_validate,
    read_comsol_points,
    read_comsol_txt,
    validate_comsol_points,
    validate_dataframe,
)

__all__ = [
    "REQUIRED_COLUMNS",
    "read_comsol_points",
    "validate_comsol_points",
    "load_and_validate",
    "read_comsol_txt",
    "validate_dataframe",
    "compute_column_stats",
    "detect_header",
]
