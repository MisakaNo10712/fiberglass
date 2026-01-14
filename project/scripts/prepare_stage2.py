#!/usr/bin/env python
"""
Stage-2 dataset preparation: add path features and training labels.

Current behavior:
- add s/tx/ty via geometry.add_path_features
- set kappa_t = strain (placeholder for top/bottom pairing)
- build mask from finite values
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from geometry import add_path_features
from utils import setup_logger

logger = setup_logger("prepare_stage2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-2 prep: add path features and kappa_t/mask",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From stage-1 output directory
  python scripts/prepare_stage2.py --input data/processed --output /data/processed2

  # From stage-1 manifest.json
  python scripts/prepare_stage2.py --input data/processed/manifest.json --output /data/processed2

  # Keep top/bottom interfaces (bottom unused for now)
  python scripts/prepare_stage2.py --top data/processed --bottom data/processed --output /data/processed2
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Stage-1 manifest.json, parquet file, or parquet directory",
    )
    parser.add_argument(
        "--top",
        type=str,
        default=None,
        help="Top input (reserved for future pairing)",
    )
    parser.add_argument(
        "--bottom",
        type=str,
        default=None,
        help="Bottom input (reserved for future pairing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data/processed2",
        help="Output directory for stage-2 parquet + manifest (default: /data/processed2)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.parquet",
        help="Glob pattern when input is a directory (default: *.parquet)",
    )
    parser.add_argument(
        "--order-mode",
        type=str,
        default="as_is",
        choices=["as_is", "nn_graph"],
        help="Path ordering mode for add_path_features",
    )
    parser.add_argument(
        "--jump-threshold",
        type=float,
        default=None,
        help="Distance threshold to split segments (optional)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="k-nearest neighbors for nn_graph ordering",
    )
    parser.add_argument(
        "--dedup-eps",
        type=float,
        default=1e-9,
        help="Duplicate tolerance for path ordering",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic tie-breaks",
    )

    return parser.parse_args()


def _resolve_path(base: Path, entry: str) -> Path:
    path = Path(entry)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (base / path).resolve()


def _load_manifest_paths(manifest_path: Path) -> list[Path]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    files = payload.get("files", [])
    if not isinstance(files, list) or not files:
        raise ValueError("Manifest is missing a non-empty 'files' list.")
    base = manifest_path.parent
    paths: list[Path] = []
    for entry in files:
        if not isinstance(entry, dict):
            continue
        path_value = entry.get("output_file") or entry.get("path") or entry.get("file")
        if path_value is None:
            continue
        paths.append(_resolve_path(base, path_value))
    if not paths:
        raise ValueError("Manifest did not contain any sample paths.")
    return paths


def _load_input_paths(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_dir():
        paths = sorted(input_path.glob(pattern))
        if not paths:
            raise ValueError(f"No parquet samples found under: {input_path}")
        return paths
    if input_path.is_file():
        if input_path.suffix.lower() == ".json":
            return _load_manifest_paths(input_path)
        if input_path.suffix.lower() == ".parquet":
            return [input_path]
        raise ValueError(f"Unsupported input file type: {input_path}")
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _validate_columns(df: pd.DataFrame, *, require_z: bool) -> None:
    required = {"x", "y", "strain"}
    if require_z:
        required.add("z")
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _process_dataframe(
    df: pd.DataFrame,
    *,
    order_mode: str,
    jump_threshold: float | None,
    k: int,
    dedup_eps: float,
    seed: int,
) -> pd.DataFrame:
    df_feat = add_path_features(
        df,
        mode=order_mode,
        jump_threshold=jump_threshold,
        k=k,
        dedup_eps=dedup_eps,
        random_state=seed,
    )

    if "strain" not in df_feat.columns:
        raise ValueError("Missing 'strain' column for kappa_t placeholder.")

    df_feat["kappa_t"] = df_feat["strain"].to_numpy(dtype=np.float32)

    finite = (
        np.isfinite(df_feat["x"].to_numpy())
        & np.isfinite(df_feat["y"].to_numpy())
        & np.isfinite(df_feat["tx"].to_numpy())
        & np.isfinite(df_feat["ty"].to_numpy())
        & np.isfinite(df_feat["kappa_t"].to_numpy())
    )
    df_feat["mask"] = finite.astype(np.float32)
    return df_feat


def _save_manifest(
    output_dir: Path,
    file_infos: list[dict],
    input_path: Path,
    pattern: str,
) -> Path:
    manifest = {
        "created_at": datetime.now().isoformat(),
        "input_path": str(input_path),
        "pattern": pattern,
        "total_files": len(file_infos),
        "total_rows": sum(info["num_rows"] for info in file_infos),
        "note": "kappa_t = strain (placeholder)",
        "files": file_infos,
    }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest_path


def main() -> int:
    args = parse_args()

    if args.input is None and args.top is None:
        raise SystemExit("Provide --input or --top for stage-1 data.")
    if args.input is not None and args.top is not None:
        raise SystemExit("Use only one of --input or --top.")

    input_path = Path(args.input or args.top)
    output_dir = Path(args.output)

    if args.bottom is not None:
        logger.warning("Bottom input provided but unused; using strain as kappa_t.")

    logger.info("Stage-2 input: %s", input_path)
    logger.info("Output directory: %s", output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = _load_input_paths(input_path, args.pattern)
    logger.info("Found %d samples", len(paths))

    file_infos: list[dict] = []
    require_z = args.jump_threshold is not None

    for path in paths:
        logger.info("Processing %s", path)
        df = pd.read_parquet(path)
        _validate_columns(df, require_z=require_z)
        df_out = _process_dataframe(
            df,
            order_mode=args.order_mode,
            jump_threshold=args.jump_threshold,
            k=args.k,
            dedup_eps=args.dedup_eps,
            seed=args.seed,
        )

        output_file = output_dir / path.name
        df_out.to_parquet(output_file, index=False)

        file_infos.append(
            {
                "input_file": str(path),
                "output_file": str(output_file),
                "num_rows": int(len(df_out)),
                "num_cols": int(len(df_out.columns)),
                "columns": list(df_out.columns),
            }
        )

    manifest_path = _save_manifest(output_dir, file_infos, input_path, args.pattern)
    logger.info("Manifest saved to: %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
