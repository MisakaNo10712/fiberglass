"""
Compatibility shim for legacy imports of COMSOL txt utilities.

This module keeps the older ``src/io/comsol_txt.py`` path working by delegating
to ``src/file_io/comsol_txt.py`` where the actual implementation lives.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parent.parent
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from file_io import comsol_txt as _comsol_txt

from file_io.comsol_txt import *  # noqa: F401,F403

__all__ = _comsol_txt.__all__
