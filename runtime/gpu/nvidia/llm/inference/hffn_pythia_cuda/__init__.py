"""HFFN inference runtime for WANDA-pruned Pythia on CUDA: HFFN kernels for the FFNs, PyTorch for the rest."""
import sys
from pathlib import Path

# The shared Pythia package and the CUDA kernels live in the `common/pythia` and `common/kernels` of the first level
# above this leaf that has each.
for _parts in (("pythia", "hffn_pythia"), ("kernels", "hffn_cuda")):
    _shared = next(p / "common" / _parts[0] for p in Path(__file__).resolve().parents if (p / "common").joinpath(*_parts).is_dir())
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))

from .model import load_pythia  # noqa: E402

__all__ = ["load_pythia"]
