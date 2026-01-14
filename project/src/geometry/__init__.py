"""
几何相关工具

提供从 COMSOL 点集重建光纤路径顺序、弧长与切线的功能。
"""

from .path import add_path_features, compute_arclength_and_tangent, order_path, split_into_segments

__all__ = [
    "split_into_segments",
    "order_path",
    "compute_arclength_and_tangent",
    "add_path_features",
]
