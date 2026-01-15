#!/usr/bin/env python
"""
Stage-2 dataset preparation: add path features and training labels.

Current behavior:
- add s/tx/ty via geometry.add_path_features
- if top+bottom provided: compute kappa_t = (strain_bot - strain_top) / h via pairing
- otherwise: set kappa_t = strain / h (placeholder)
- build mask from finite values (optionally include w if present)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root + src to path for direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from datasets import pair_top_bottom
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
    parser.add_argument(
        "--pair-method",
        type=str,
        default="s",
        choices=["s", "xy"],
        help="Pairing method when both top/bottom are provided (default: s)",
    )
    parser.add_argument(
        "--h",
        type=float,
        default=0.005,
        help="Plate thickness for kappa_t = (strain_bot - strain_top) / h (default: 0.005)",
    )
    parser.add_argument(
        "--s-tol",
        type=float,
        default=None,
        help="Max |s_top - s_bot| for s-based pairing (optional)",
    )
    parser.add_argument(
        "--xy-tol",
        type=float,
        default=None,
        help="Max XY distance for nearest-neighbor pairing (optional)",
    )
    parser.add_argument(
        "--fill-strategy",
        type=str,
        default="zeros",
        choices=["zeros", "nearest"],
        help="How to fill missing pairs (default: zeros)",
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


def _pair_key(path: Path) -> str:
    stem = path.stem
    if "_top_" in stem:
        return stem.replace("_top_", "_")
    if "_bottom_" in stem:
        return stem.replace("_bottom_", "_")
    return stem


def _match_pairs(top_paths: list[Path], bottom_paths: list[Path]) -> list[tuple[Path, Path, str]]:
    def _build_map(paths: list[Path], label: str) -> dict[str, Path]:
        mapping: dict[str, Path] = {}
        for path in paths:
            key = _pair_key(path)
            if key in mapping:
                raise ValueError(f"Duplicate {label} key {key}: {mapping[key]} vs {path}")
            mapping[key] = path
        return mapping

    top_map = _build_map(top_paths, "top")
    bottom_map = _build_map(bottom_paths, "bottom")
    top_keys = set(top_map)
    bottom_keys = set(bottom_map)
    missing_bottom = sorted(top_keys - bottom_keys)
    missing_top = sorted(bottom_keys - top_keys)

    if missing_bottom:
        logger.warning("Missing bottom pairs for %d top files.", len(missing_bottom))
    if missing_top:
        logger.warning("Missing top pairs for %d bottom files.", len(missing_top))

    matched_keys = sorted(top_keys & bottom_keys)
    if not matched_keys:
        raise ValueError("No matched top/bottom pairs found.")

    return [(top_map[key], bottom_map[key], key) for key in matched_keys]


def _add_mask(df_feat: pd.DataFrame) -> pd.DataFrame:
    base_mask = None
    if "mask" in df_feat.columns:
        base_mask = df_feat["mask"].to_numpy(dtype=np.float32) > 0
    finite = (
        np.isfinite(df_feat["x"].to_numpy())
        & np.isfinite(df_feat["y"].to_numpy())
        & np.isfinite(df_feat["tx"].to_numpy())
        & np.isfinite(df_feat["ty"].to_numpy())
        & np.isfinite(df_feat["kappa_t"].to_numpy())
    )
    if "w" in df_feat.columns:
        finite &= np.isfinite(df_feat["w"].to_numpy())
    if base_mask is not None:
        finite &= base_mask
    df_feat["mask"] = finite.astype(np.float32)
    return df_feat


def _process_single_dataframe(
    df: pd.DataFrame,
    *,
    order_mode: str,
    jump_threshold: float | None,
    k: int,
    dedup_eps: float,
    seed: int,
    h: float,
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

    df_feat["kappa_t"] = df_feat["strain"].to_numpy(dtype=np.float32) / float(h)

    return _add_mask(df_feat)


def _process_pair_dataframe(
    df_top: pd.DataFrame,
    df_bot: pd.DataFrame,
    *,
    order_mode: str,
    jump_threshold: float | None,
    k: int,
    dedup_eps: float,
    seed: int,
    pair_method: str,
    h: float,
    s_tol: float | None,
    xy_tol: float | None,
    fill_strategy: str,
) -> pd.DataFrame:
    df_top_feat = add_path_features(
        df_top,
        mode=order_mode,
        jump_threshold=jump_threshold,
        k=k,
        dedup_eps=dedup_eps,
        random_state=seed,
    )
    df_bot_feat = add_path_features(
        df_bot,
        mode=order_mode,
        jump_threshold=jump_threshold,
        k=k,
        dedup_eps=dedup_eps,
        random_state=seed,
    )

    df_pair = pair_top_bottom(
        df_top_feat,
        df_bot_feat,
        method=pair_method,
        h=h,
        s_tol=s_tol,
        xy_tol=xy_tol,
        fill_strategy=fill_strategy,
    )

    if "w" in df_top_feat.columns and "w" not in df_pair.columns:
        df_pair["w"] = df_top_feat["w"].to_numpy(dtype=np.float32)
    elif "w" in df_bot_feat.columns and "w" not in df_pair.columns:
        if len(df_bot_feat) != len(df_pair):
            logger.warning("Bottom w length mismatch; skipping w propagation.")
        else:
            df_pair["w"] = df_bot_feat["w"].to_numpy(dtype=np.float32)

    return _add_mask(df_pair)


def _save_manifest(
    output_dir: Path,
    file_infos: list[dict],
    input_path: Path,
    pattern: str,
    note: str,
) -> Path:
    manifest = {
        "created_at": datetime.now().isoformat(),
        "input_path": str(input_path),
        "pattern": pattern,
        "total_files": len(file_infos),
        "total_rows": sum(info["num_rows"] for info in file_infos),
        "note": note,
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
    if args.bottom is not None and args.top is None:
        raise SystemExit("Provide --top when using --bottom.")

    input_path = Path(args.input or args.top)
    output_dir = Path(args.output)

    if args.bottom is None:
        logger.warning("Bottom input not provided; using strain as kappa_t placeholder.")

    logger.info("Stage-2 input: %s", input_path)
    logger.info("Output directory: %s", output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    file_infos: list[dict] = []
    require_z = args.jump_threshold is not None
    note = ""

    if args.bottom is None:
        paths = _load_input_paths(input_path, args.pattern)
        logger.info("Found %d samples", len(paths))
        note = f"kappa_t = strain / h (placeholder, h={args.h})"

        for path in paths:
            logger.info("Processing %s", path)
            df = pd.read_parquet(path)
            _validate_columns(df, require_z=require_z)
            df_out = _process_single_dataframe(
                df,
                order_mode=args.order_mode,
                jump_threshold=args.jump_threshold,
                k=args.k,
                dedup_eps=args.dedup_eps,
                seed=args.seed,
                h=args.h,
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
    else:
        top_paths = _load_input_paths(Path(args.top), args.pattern)
        bottom_paths = _load_input_paths(Path(args.bottom), args.pattern)
        pairs = _match_pairs(top_paths, bottom_paths)
        logger.info("Matched %d top/bottom pairs", len(pairs))
        note = f"kappa_t = (strain_bot - strain_top) / h (h={args.h})"

        for top_path, bottom_path, key in pairs:
            logger.info("Processing %s + %s", top_path, bottom_path)
            df_top = pd.read_parquet(top_path)
            df_bot = pd.read_parquet(bottom_path)
            _validate_columns(df_top, require_z=require_z)
            _validate_columns(df_bot, require_z=require_z)
            df_out = _process_pair_dataframe(
                df_top,
                df_bot,
                order_mode=args.order_mode,
                jump_threshold=args.jump_threshold,
                k=args.k,
                dedup_eps=args.dedup_eps,
                seed=args.seed,
                pair_method=args.pair_method,
                h=args.h,
                s_tol=args.s_tol,
                xy_tol=args.xy_tol,
                fill_strategy=args.fill_strategy,
            )

            output_file = output_dir / f"{key}.parquet"
            df_out.to_parquet(output_file, index=False)

            file_infos.append(
                {
                    "input_top": str(top_path),
                    "input_bottom": str(bottom_path),
                    "output_file": str(output_file),
                    "num_rows": int(len(df_out)),
                    "num_cols": int(len(df_out.columns)),
                    "columns": list(df_out.columns),
                }
            )

    manifest_path = _save_manifest(output_dir, file_infos, input_path, args.pattern, note)
    logger.info("Manifest saved to: %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
