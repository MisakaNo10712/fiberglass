#!/usr/bin/env python
"""
Wrapper for project/scripts/compute_dataset_stats.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    module_path = root / "project" / "scripts" / "compute_dataset_stats.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing script: {module_path}")
    spec = importlib.util.spec_from_file_location("project_compute_dataset_stats", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
