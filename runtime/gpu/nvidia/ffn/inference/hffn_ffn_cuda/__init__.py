"""Isolated HFFN FFN on CUDA: the shared kernels (``hffn_cuda``) on the shared synthetic test layers (``hffn_ffn_layers``)."""
import sys
from pathlib import Path

# The CUDA kernels and the test layers live in the `common/kernels` and `common/ffn` of the first level above this
# leaf that has each.
for _parts in (("kernels", "hffn_cuda"), ("ffn", "hffn_ffn_layers")):
    _shared = next(p / "common" / _parts[0] for p in Path(__file__).resolve().parents if (p / "common").joinpath(*_parts).is_dir())
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))

from .layers import cuda_layer  # noqa: E402

__all__ = ["cuda_layer"]
