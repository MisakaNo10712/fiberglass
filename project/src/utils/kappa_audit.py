"""
Helpers to detect placeholder kappa_t sources from manifests/parquet files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def _resolve_path(base: Path, entry: str) -> Path:
    path = Path(entry)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    candidate = base / path
    if candidate.exists():
        return candidate.resolve()
    for parent in base.parents:
        alt = parent / path
        if alt.exists():
            return alt.resolve()
    return candidate.resolve()


def _load_manifest_note(manifest_path: Path) -> str | None:
    if not manifest_path.exists() or manifest_path.suffix.lower() != ".json":
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        note = payload.get("note")
        return str(note) if note is not None else None
    except Exception:
        return None


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


def _resolve_parquet_paths(
    *,
    manifest_path: str | Path | None = None,
    samples_dir: str | Path | None = None,
    parquet_path: str | Path | None = None,
) -> list[Path]:
    if parquet_path is not None:
        path = Path(parquet_path)
        return [path]
    if manifest_path is not None:
        return _load_manifest_paths(Path(manifest_path))
    if samples_dir is not None:
        root = Path(samples_dir)
        if not root.exists():
            raise FileNotFoundError(f"Samples directory not found: {root}")
        paths = sorted(root.glob("*.parquet"))
        if not paths:
            raise ValueError(f"No parquet samples found under: {root}")
        return paths
    raise ValueError("Need manifest_path, samples_dir, or parquet_path.")


def _note_indicates_placeholder(note: str | None) -> bool:
    if not note:
        return False
    lowered = note.lower()
    return "placeholder" in lowered or "strain / h" in lowered or "strain/h" in lowered


def _columns_placeholder_hint(columns: Iterable[str]) -> bool:
    cols = {str(c) for c in columns}
    if "strain_top" in cols and "strain_bot" in cols:
        return False
    if "strain" in cols and "kappa_t" in cols:
        return True
    return False


def inspect_kappa_source(
    *,
    manifest_path: str | Path | None = None,
    samples_dir: str | Path | None = None,
    parquet_path: str | Path | None = None,
) -> dict[str, object]:
    paths = _resolve_parquet_paths(
        manifest_path=manifest_path, samples_dir=samples_dir, parquet_path=parquet_path
    )
    first_path = paths[0]
    df = pd.read_parquet(first_path)
    columns = list(df.columns)
    note = _load_manifest_note(Path(manifest_path)) if manifest_path is not None else None
    placeholder_from_note = _note_indicates_placeholder(note)
    placeholder_from_cols = _columns_placeholder_hint(columns)
    placeholder = placeholder_from_note or placeholder_from_cols
    reason = None
    if placeholder_from_note:
        reason = "manifest_note"
    elif placeholder_from_cols:
        reason = "missing_strain_top_bot"

    cols = set(columns)
    summary = {
        "manifest_note": note,
        "first_parquet": str(first_path),
        "columns": sorted(columns),
        "has_strain": "strain" in cols,
        "has_strain_top": "strain_top" in cols,
        "has_strain_bot": "strain_bot" in cols,
        "has_kappa_t": "kappa_t" in cols,
        "has_w": "w" in cols,
        "has_tx": "tx" in cols,
        "has_ty": "ty" in cols,
        "has_mask": "mask" in cols,
        "placeholder_kappa": placeholder,
        "placeholder_reason": reason,
        "placeholder_from_note": placeholder_from_note,
        "placeholder_from_columns": placeholder_from_cols,
    }
    return summary

