"""
Smoke Tests - 基础功能测试

确保项目的基本功能可以正常运行。
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import pytest


def _load_module(path: Path, name: str, is_package: bool = False):
    submodule_locations = [] if is_package else None
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=submodule_locations)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
comsol_txt = _load_module(SRC_ROOT / "io" / "comsol_txt.py", "fiberglass_comsol_txt")
utils_mod = _load_module(SRC_ROOT / "utils" / "__init__.py", "fiberglass_utils", is_package=True)
REQUIRED_COLUMNS = comsol_txt.REQUIRED_COLUMNS


class TestImports:
    """测试模块导入"""

    def test_import_io_module(self):
        """测试 io 模块可以导入"""
        assert hasattr(comsol_txt, "read_comsol_txt")
        assert hasattr(comsol_txt, "validate_dataframe")
        assert hasattr(comsol_txt, "REQUIRED_COLUMNS")

    def test_import_utils_module(self):
        """测试 utils 模块可以导入"""
        assert callable(utils_mod.setup_logger)
        assert callable(utils_mod.get_logger)

    def test_required_columns(self):
        """测试必需列定义正确"""
        expected = ["x", "y", "z", "strain", "u", "v", "w"]
        assert REQUIRED_COLUMNS == expected


class TestComsolTxtReader:
    """测试 COMSOL txt 文件读取功能"""

    # 带表头的测试数据
    SAMPLE_DATA_WITH_HEADER = """x y z strain u v w
0.0 0.0 0.0 0.001 0.0001 0.0002 0.0003
0.1 0.0 0.0 0.002 0.0002 0.0003 0.0004
0.2 0.0 0.0 0.003 0.0003 0.0004 0.0005
0.3 0.0 0.0 0.004 0.0004 0.0005 0.0006
0.4 0.0 0.0 0.005 0.0005 0.0006 0.0007
0.5 0.0 0.0 0.006 0.0006 0.0007 0.0008
0.6 0.0 0.0 0.007 0.0007 0.0008 0.0009
0.7 0.0 0.0 0.008 0.0008 0.0009 0.0010
0.8 0.0 0.0 0.009 0.0009 0.0010 0.0011
0.9 0.0 0.0 0.010 0.0010 0.0011 0.0012
1.0 0.0 0.0 0.011 0.0011 0.0012 0.0013
"""

    # 无表头的测试数据
    SAMPLE_DATA_NO_HEADER = """0.0 0.0 0.0 0.001 0.0001 0.0002 0.0003
0.1 0.0 0.0 0.002 0.0002 0.0003 0.0004
0.2 0.0 0.0 0.003 0.0003 0.0004 0.0005
0.3 0.0 0.0 0.004 0.0004 0.0005 0.0006
0.4 0.0 0.0 0.005 0.0005 0.0006 0.0007
0.5 0.0 0.0 0.006 0.0006 0.0007 0.0008
0.6 0.0 0.0 0.007 0.0007 0.0008 0.0009
0.7 0.0 0.0 0.008 0.0008 0.0009 0.0010
0.8 0.0 0.0 0.009 0.0009 0.0010 0.0011
0.9 0.0 0.0 0.010 0.0010 0.0011 0.0012
1.0 0.0 0.0 0.011 0.0011 0.0012 0.0013
"""

    # Tab 分隔的测试数据
    SAMPLE_DATA_TAB_SEPARATED = """x\ty\tz\tstrain\tu\tv\tw
0.0\t0.0\t0.0\t0.001\t0.0001\t0.0002\t0.0003
0.1\t0.0\t0.0\t0.002\t0.0002\t0.0003\t0.0004
0.2\t0.0\t0.0\t0.003\t0.0003\t0.0004\t0.0005
0.3\t0.0\t0.0\t0.004\t0.0004\t0.0005\t0.0006
0.4\t0.0\t0.0\t0.005\t0.0005\t0.0006\t0.0007
0.5\t0.0\t0.0\t0.006\t0.0006\t0.0007\t0.0008
0.6\t0.0\t0.0\t0.007\t0.0007\t0.0008\t0.0009
0.7\t0.0\t0.0\t0.008\t0.0008\t0.0009\t0.0010
0.8\t0.0\t0.0\t0.009\t0.0009\t0.0010\t0.0011
0.9\t0.0\t0.0\t0.010\t0.0010\t0.0011\t0.0012
"""

    def test_read_with_header(self):
        """测试读取带表头的文件"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(self.SAMPLE_DATA_WITH_HEADER)
            temp_path = f.name

        try:
            df = comsol_txt.read_comsol_txt(temp_path)

            # 检查列名
            assert list(df.columns) == REQUIRED_COLUMNS

            # 检查行数
            assert len(df) == 11

            # 检查数据类型
            for col in REQUIRED_COLUMNS:
                assert df[col].dtype == "float64"

        finally:
            Path(temp_path).unlink()

    def test_read_without_header(self):
        """测试读取无表头的文件"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(self.SAMPLE_DATA_NO_HEADER)
            temp_path = f.name

        try:
            df = comsol_txt.read_comsol_txt(temp_path, has_header=False)

            # 检查列名（应该是默认列名）
            assert list(df.columns) == REQUIRED_COLUMNS

            # 检查行数
            assert len(df) == 11

        finally:
            Path(temp_path).unlink()

    def test_read_tab_separated(self):
        """测试读取 Tab 分隔的文件"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(self.SAMPLE_DATA_TAB_SEPARATED)
            temp_path = f.name

        try:
            df = comsol_txt.read_comsol_txt(temp_path)

            # 检查列名
            assert list(df.columns) == REQUIRED_COLUMNS

            # 检查行数
            assert len(df) == 10

        finally:
            Path(temp_path).unlink()

    def test_auto_detect_header(self):
        """测试自动检测表头"""
        # 带表头
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(self.SAMPLE_DATA_WITH_HEADER)
            temp_path = f.name

        try:
            assert comsol_txt.detect_header(temp_path) is True
        finally:
            Path(temp_path).unlink()

        # 无表头
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(self.SAMPLE_DATA_NO_HEADER)
            temp_path = f.name

        try:
            assert comsol_txt.detect_header(temp_path) is False
        finally:
            Path(temp_path).unlink()

    def test_validation_min_rows(self):
        """测试行数校验"""
        # 只有 5 行数据（少于 10 行）
        small_data = """x y z strain u v w
0.0 0.0 0.0 0.001 0.0001 0.0002 0.0003
0.1 0.0 0.0 0.002 0.0002 0.0003 0.0004
0.2 0.0 0.0 0.003 0.0003 0.0004 0.0005
0.3 0.0 0.0 0.004 0.0004 0.0005 0.0006
0.4 0.0 0.0 0.005 0.0005 0.0006 0.0007
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(small_data)
            temp_path = f.name

        try:
            with pytest.raises(comsol_txt.ValidationError, match="行数不足"):
                comsol_txt.read_comsol_txt(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_column_stats(self):
        """测试列统计计算"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(self.SAMPLE_DATA_WITH_HEADER)
            temp_path = f.name

        try:
            df = comsol_txt.read_comsol_txt(temp_path)
            stats = comsol_txt.compute_column_stats(df)

            # 检查 x 列统计
            assert "x" in stats
            assert stats["x"]["min"] == 0.0
            assert stats["x"]["max"] == 1.0
            assert "mean" in stats["x"]

        finally:
            Path(temp_path).unlink()


class TestLogger:
    """测试日志功能"""

    def test_setup_logger(self):
        """测试日志设置"""
        logger = utils_mod.setup_logger("test_logger", level="DEBUG")
        assert logger.name == "test_logger"
        assert logger.level == 10  # DEBUG = 10

    def test_get_logger(self):
        """测试获取日志"""
        logger = utils_mod.get_logger("test_get_logger")
        assert logger is not None
        assert logger.name == "test_get_logger"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
