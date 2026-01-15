#!/usr/bin/env python
"""
Compute global kappa/w statistics for the training dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import setup_logger

logger = setup_logger("compute_dataset_stats")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dataset kappa/w stats")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--pattern", type=str, default="*.parquet")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


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


def _load_samples_dir(samples_dir: Path, pattern: str) -> list[Path]:
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    paths = sorted(samples_dir.glob(pattern))
    if not paths:
        raise ValueError(f"No parquet samples found under: {samples_dir}")
    return paths


def _accumulate(values: np.ndarray, accum: dict[str, float], prefix: str) -> None:
    if values.size == 0:
        return
    accum[f"{prefix}_sum"] += float(values.sum(dtype=np.float64))
    accum[f"{prefix}_sumsq"] += float((values**2).sum(dtype=np.float64))
    accum[f"{prefix}_count"] += int(values.size)


def compute_stats(paths: list[Path]) -> dict[str, float]:
    if not paths:
        raise ValueError("No sample paths provided.")

    accum = {
        "kappa_sum": 0.0,
        "kappa_sumsq": 0.0,
        "kappa_count": 0,
        "w_sum": 0.0,
        "w_sumsq": 0.0,
        "w_count": 0,
    }
    missing_w = 0

    for path in paths:
        df = pd.read_parquet(path)
        if "kappa_t" not in df.columns:
            raise ValueError(f"Missing 'kappa_t' column in {path}")
        mask = df["mask"].to_numpy(dtype=np.float64) if "mask" in df.columns else None
        kappa = df["kappa_t"].to_numpy(dtype=np.float64)
        if mask is not None:
            kappa = kappa[mask > 0]
        kappa = kappa[np.isfinite(kappa)]
        _accumulate(kappa, accum, "kappa")

        if "w" in df.columns:
            w_vals = df["w"].to_numpy(dtype=np.float64)
            if mask is not None:
                w_vals = w_vals[mask > 0]
            w_vals = w_vals[np.isfinite(w_vals)]
            _accumulate(w_vals, accum, "w")
        else:
            missing_w += 1

    def _finalize(prefix: str) -> tuple[float, float]:
        count = accum[f"{prefix}_count"]
        if count == 0:
            return 0.0, 1.0
        mean = accum[f"{prefix}_sum"] / count
        var = accum[f"{prefix}_sumsq"] / count - mean**2
        std = float(np.sqrt(max(var, 0.0)))
        return float(mean), std

    kappa_mean, kappa_std = _finalize("kappa")
    w_mean, w_std = _finalize("w")

    return {
        "kappa_mean": kappa_mean,
        "kappa_std": kappa_std,
        "w_mean": w_mean,
        "w_std": w_std,
        "kappa_count": int(accum["kappa_count"]),
        "w_count": int(accum["w_count"]),
        "missing_w_samples": int(missing_w),
    }


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))

    data_cfg = config.get("data", {})
    manifest = args.manifest or data_cfg.get("manifest")
    samples_dir = args.data_dir or data_cfg.get("samples_dir")

    paths: list[Path] = []
    if manifest:
        manifest_path = Path(manifest)
        if manifest_path.suffix.lower() == ".parquet":
            paths = [manifest_path]
        else:
            paths = _load_manifest_paths(manifest_path)
    elif samples_dir:
        paths = _load_samples_dir(Path(samples_dir), args.pattern)
    else:
        raise SystemExit("Provide --manifest or --data_dir, or set them in config.")

    stats = compute_stats(paths)
    output_path = Path(args.output or config.get("stats_path", "data/processed/stats.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    logger.info("Stats saved to %s", output_path)
    logger.info("kappa_std=%.6f, w_std=%.6f", stats["kappa_std"], stats["w_std"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
