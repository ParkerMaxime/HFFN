"""Adapter of the CPU M1 compact Pythia leaf to the shared Pythia level.

Puts on ``sys.path`` the shared package ``hffn_pythia`` and the compact leaf, whose runtime is used unchanged."""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
SHARED = ROOT / "runtime" / "common" / "pythia"
LEAF = ROOT / "runtime" / "cpu" / "m1" / "llm" / "inference" / "compact"
BUILD = Path(__file__).resolve().parents[1] / "build"
for _path in (SHARED, LEAF):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from .model import G4_GROUPS, load_pythia, runtime_files  # noqa: E402

__all__ = ["G4_GROUPS", "load_pythia", "runtime_files", "ROOT", "LEAF", "BUILD"]
