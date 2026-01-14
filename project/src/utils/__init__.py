"""
工具模块 - 通用工具函数

提供日志、配置加载等通用功能。
"""

from .logging import setup_logger, get_logger

__all__ = ["setup_logger", "get_logger"]
