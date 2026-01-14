"""
日志模块

提供统一的日志配置和获取功能。

后续扩展点：
- 支持文件日志
- 支持日志轮转
- 支持结构化日志（JSON 格式）
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

# 默认日志格式
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 全局 logger 缓存
_loggers: dict[str, logging.Logger] = {}


def setup_logger(
    name: str = "fiberglass",
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    format_str: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    """
    设置并返回一个 logger。

    Args:
        name: logger 名称
        level: 日志级别
        format_str: 日志格式
        date_format: 日期格式

    Returns:
        logging.Logger: 配置好的 logger
    """
    # 如果已经配置过，直接返回
    if name in _loggers:
        return _loggers[name]

    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # 避免重复添加 handler
    if not logger.handlers:
        # 创建控制台 handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level))

        # 设置格式
        formatter = logging.Formatter(format_str, datefmt=date_format)
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    # 缓存
    _loggers[name] = logger

    return logger


def get_logger(name: str = "fiberglass") -> logging.Logger:
    """
    获取一个 logger。如果不存在，则使用默认配置创建。

    Args:
        name: logger 名称

    Returns:
        logging.Logger: logger 实例
    """
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]
