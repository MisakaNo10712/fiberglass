"""
Fiber sequence dataset for training coefficient regression models.

Supports:
- Parquet sample files listed in a manifest.json
- A single parquet file path
- In-memory DataFrame list (df_list) for testing and rapid experiments
"""

from __future__ import annotations

import json
from pathlib import Path
import warnings
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

__all__ = ["FiberSequenceDataset", "fiber_sequence_collate"]

REQUIRED_COLUMNS = ["x", "y", "tx", "ty", "kappa_t", "mask"]
OPTIONAL_COLUMNS = ["w"]


def _load_stats(stats_path: Optional[str | Path]) -> dict[str, float]:
    if stats_path is None:
        return {}
    path = Path(stats_path)
    if not path.exists():
        warnings.warn(f"Stats file not found: {path}. Using default std=1.0.")
        return {}
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    elif path.suffix.lower() == ".npz":
        payload = dict(np.load(path))
    else:
        raise ValueError(f"Unsupported stats file type: {path.suffix}")
    stats = {}
    for key in ("kappa_std", "w_std", "kappa_mean", "w_mean"):
        if key in payload:
            stats[key] = float(payload[key])
    return stats


def _resolve_path(base: Path, entry: str) -> Path:
    path = Path(entry)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    candidate = base / path
    if candidate.exists():
        return candidate.resolve()
    return candidate.resolve()


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


def _load_samples_dir(samples_dir: Path) -> list[Path]:
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    paths = sorted(samples_dir.glob("*.parquet"))
    if not paths:
        raise ValueError(f"No parquet samples found under: {samples_dir}")
    return paths


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _df_to_sample(df: pd.DataFrame) -> dict[str, torch.Tensor]:
    _validate_columns(df)
    arrays = {col: df[col].to_numpy(dtype=np.float32) for col in REQUIRED_COLUMNS}
    x = arrays["x"]
    y = arrays["y"]
    tx = arrays["tx"]
    ty = arrays["ty"]
    kappa_t = arrays["kappa_t"]
    X = np.stack([x, y, tx, ty, kappa_t], axis=-1)
    sample = {
        "X": torch.from_numpy(X),
        "mask": torch.from_numpy(arrays["mask"]),
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(y),
    }
    if "w" in df.columns:
        sample["w_points"] = torch.from_numpy(df["w"].to_numpy(dtype=np.float32))
    return sample


class FiberSequenceDataset(Dataset):
    """Dataset returning per-sample fiber sequences with optional deflection labels."""

    def __init__(
        self,
        *,
        manifest_path: Optional[str | Path] = None,
        samples_dir: Optional[str | Path] = None,
        df_list: Optional[Iterable[pd.DataFrame]] = None,
        stats_path: Optional[str | Path] = None,
    ) -> None:
        stats = _load_stats(stats_path)
        self.kappa_std = float(stats.get("kappa_std", 1.0))
        self.w_std = float(stats.get("w_std", 1.0))

        if df_list is not None:
            self._df_list = list(df_list)
            if not self._df_list:
                raise ValueError("df_list is empty.")
            self._paths: list[Path] = []
            self._use_df = True
            return

        self._use_df = False
        self._df_list = []

        if manifest_path is not None:
            manifest_path = Path(manifest_path)
            if manifest_path.suffix == ".parquet":
                self._paths = [manifest_path]
            else:
                self._paths = _load_manifest_paths(manifest_path)
        elif samples_dir is not None:
            self._paths = _load_samples_dir(Path(samples_dir))
        else:
            raise ValueError("Either manifest_path, samples_dir, or df_list must be provided.")

    def __len__(self) -> int:
        return len(self._df_list) if self._use_df else len(self._paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._use_df:
            df = self._df_list[idx]
        else:
            df = pd.read_parquet(self._paths[idx])
        sample = _df_to_sample(df)
        sample["kappa_std"] = self.kappa_std
        sample["w_std"] = self.w_std
        return sample


def fiber_sequence_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length sequences into a batch."""
    if not batch:
        raise ValueError("Empty batch.")
    lengths = [sample["X"].shape[0] for sample in batch]
    max_len = max(lengths)
    batch_size = len(batch)

    dtype = batch[0]["X"].dtype
    device = batch[0]["X"].device

    X = torch.zeros((batch_size, max_len, 5), dtype=dtype, device=device)
    mask = torch.zeros((batch_size, max_len), dtype=dtype, device=device)
    x = torch.zeros((batch_size, max_len), dtype=dtype, device=device)
    y = torch.zeros((batch_size, max_len), dtype=dtype, device=device)

    has_w = any("w_points" in sample for sample in batch)
    if has_w and not all("w_points" in sample for sample in batch):
        raise ValueError("w_points must be present for all samples or none.")
    w_points = torch.zeros((batch_size, max_len), dtype=dtype, device=device) if has_w else None

    for i, sample in enumerate(batch):
        length = sample["X"].shape[0]
        X[i, :length] = sample["X"]
        mask[i, :length] = sample["mask"]
        if "x" in sample:
            x[i, :length] = sample["x"]
        else:
            x[i, :length] = sample["X"][:, 0]
        if "y" in sample:
            y[i, :length] = sample["y"]
        else:
            y[i, :length] = sample["X"][:, 1]
        if has_w and w_points is not None:
            w_points[i, :length] = sample["w_points"]

    output = {"X": X, "mask": mask, "x": x, "y": y}
    if has_w and w_points is not None:
        output["w_points"] = w_points

    if "kappa_std" in batch[0]:
        kappa_std = float(batch[0]["kappa_std"])
        for sample in batch[1:]:
            if "kappa_std" in sample and not np.isclose(float(sample["kappa_std"]), kappa_std):
                raise ValueError("Inconsistent kappa_std values in batch.")
        output["kappa_std"] = kappa_std

    if "w_std" in batch[0]:
        w_std = float(batch[0]["w_std"])
        for sample in batch[1:]:
            if "w_std" in sample and not np.isclose(float(sample["w_std"]), w_std):
                raise ValueError("Inconsistent w_std values in batch.")
        output["w_std"] = w_std

    return output
