#!/usr/bin/env python
"""
数据预处理脚本

将 COMSOL 导出的 txt 文件转换为 parquet 格式，并生成 manifest 文件。

用法:
    python scripts/prepare_dataset.py --input data/raw --output data/processed
    python scripts/prepare_dataset.py --input data/raw/sample.txt --output data/processed
    python scripts/prepare_dataset.py --input data/raw --no-header

后续扩展点:
- 支持路径排序
- 支持上下表面配对
- 支持切线计算
- 支持数据增强
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# 添加 src 到路径（支持直接运行脚本）
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from file_io import comsol_txt
from utils import setup_logger

logger = setup_logger("prepare_dataset")


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="将 COMSOL txt 文件转换为 parquet 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理整个目录
  python scripts/prepare_dataset.py --input data/raw --output data/processed

  # 处理单个文件
  python scripts/prepare_dataset.py --input data/raw/sample.txt

  # 指定文件无表头
  python scripts/prepare_dataset.py --input data/raw --no-header

  # 自定义 glob 模式
  python scripts/prepare_dataset.py --input data/raw --pattern "*.dat"
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/raw",
        help="输入路径（文件或目录）。默认: data/raw",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="输出目录。默认: data/processed",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.txt",
        help="当 --input 为目录时的 glob 模式。默认: *.txt",
    )

    parser.add_argument(
        "--no-header",
        action="store_true",
        help="指定输入文件无表头（按列序赋名: x y z strain u v w）",
    )

    return parser.parse_args()


def get_input_files(input_path: Path, pattern: str) -> list[Path]:
    """
    获取输入文件列表。

    Args:
        input_path: 输入路径（文件或目录）
        pattern: glob 模式

    Returns:
        list[Path]: 文件路径列表
    """
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob(pattern))
        return files
    else:
        logger.error(f"输入路径不存在: {input_path}")
        return []


def process_file(
    input_file: Path,
    output_dir: Path,
    has_header: bool | None,
) -> dict | None:
    """
    处理单个文件。

    Args:
        input_file: 输入文件路径
        output_dir: 输出目录
        has_header: 是否有表头（None 表示自动检测）

    Returns:
        dict: 文件处理信息，失败返回 None
    """
    logger.info(f"处理文件: {input_file}")

    try:
        # 检测表头
        detected_header = comsol_txt.detect_header(input_file)
        actual_has_header = has_header if has_header is not None else detected_header

        logger.info(f"  表头检测: {'有表头' if detected_header else '无表头'}")
        if has_header is not None:
            logger.info(f"  使用用户指定: {'有表头' if has_header else '无表头'}")

        # 读取数据
        df = comsol_txt.read_comsol_txt(
            input_file,
            has_header=actual_has_header,
            validate=True,
        )

        logger.info(f"  读取成功: {len(df)} 行, {len(df.columns)} 列")

        # 计算统计信息
        stats = comsol_txt.compute_column_stats(df)

        # 输出文件路径
        output_file = output_dir / f"{input_file.stem}.parquet"

        # 保存为 parquet
        df.to_parquet(output_file, index=False, engine="pyarrow")
        logger.info(f"  保存到: {output_file}")

        return {
            "input_file": str(input_file),
            "output_file": str(output_file),
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "columns": list(df.columns),
            "has_header": actual_has_header,
            "stats": stats,
        }

    except comsol_txt.ComsolReadError as e:
        logger.error(f"  读取错误: {e}")
        return None
    except comsol_txt.ValidationError as e:
        logger.error(f"  校验错误: {e}")
        return None
    except Exception as e:
        logger.error(f"  未知错误: {e}")
        return None


def save_manifest(
    output_dir: Path,
    file_infos: list[dict],
    input_path: str,
    pattern: str,
) -> Path:
    """
    保存 manifest 文件。

    Args:
        output_dir: 输出目录
        file_infos: 文件处理信息列表
        input_path: 输入路径
        pattern: glob 模式

    Returns:
        Path: manifest 文件路径
    """
    manifest = {
        "created_at": datetime.now().isoformat(),
        "input_path": input_path,
        "pattern": pattern,
        "total_files": len(file_infos),
        "total_rows": sum(info["num_rows"] for info in file_infos),
        "files": file_infos,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest_path


def main() -> int:
    """主函数"""
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    logger.info("=" * 60)
    logger.info("COMSOL 数据预处理")
    logger.info("=" * 60)
    logger.info(f"输入路径: {input_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"Glob 模式: {args.pattern}")
    logger.info(f"表头设置: {'无表头' if args.no_header else '自动检测'}")
    logger.info("-" * 60)

    # 获取输入文件
    input_files = get_input_files(input_path, args.pattern)

    if not input_files:
        logger.error("未找到任何输入文件")
        return 1

    logger.info(f"找到 {len(input_files)} 个文件")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理文件
    has_header = False if args.no_header else None
    file_infos = []
    success_count = 0
    fail_count = 0

    for input_file in input_files:
        info = process_file(input_file, output_dir, has_header)
        if info:
            file_infos.append(info)
            success_count += 1
        else:
            fail_count += 1

    logger.info("-" * 60)
    logger.info(f"处理完成: 成功 {success_count}, 失败 {fail_count}")

    # 保存 manifest
    if file_infos:
        manifest_path = save_manifest(
            output_dir,
            file_infos,
            str(input_path),
            args.pattern,
        )
        logger.info(f"Manifest 保存到: {manifest_path}")

    logger.info("=" * 60)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
