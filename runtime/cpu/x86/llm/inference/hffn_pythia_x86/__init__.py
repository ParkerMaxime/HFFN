"""HFFN CPU inference runtime for Pythia on x86_64."""
import sys
from pathlib import Path

# The shared Pythia package lives in the `common/pythia` of the first level above this leaf that has one.
_shared = next(p / "common" / "pythia" for p in Path(__file__).resolve().parents if (p / "common" / "pythia" / "hffn_pythia").is_dir())
if str(_shared) not in sys.path:
    sys.path.insert(0, str(_shared))

from .model import load_pythia  # noqa: E402

__all__ = ["load_pythia"]
