"""
兼容层: 维持历史导入路径 src.io.*。

当前实现直接从 file_io 包导出同名符号。
"""

from file_io.comsol_txt import *  # noqa: F401,F403
