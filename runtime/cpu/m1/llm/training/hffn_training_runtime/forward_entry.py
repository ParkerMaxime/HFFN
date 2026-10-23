"""Entry point of the M1 leaf for the shared ``compact_runtime``: the forward of execution ``adaptive`` runs on the
compact G4 inference kernel of the sibling leaf (its dual entry, which stores ``z`` and ``h``; the binding is the one
of that leaf, ``hffn_inference_runtime.dual_entry``)."""
from __future__ import annotations

import sys
from pathlib import Path

# The compact inference leaf (kernel source, build, binding) is `runtime/cpu/m1/llm/inference/compact`.
_INFERENCE_LEAF = next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir()) / "runtime/cpu/m1/llm/inference/compact"
if str(_INFERENCE_LEAF) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_LEAF))

from hffn_inference_runtime.dual_entry import INFERENCE_MIN_N, InferenceKernel, inference_library  # noqa: E402,F401

from .compact_export import G4_GROUPS  # noqa: E402
from .forward_route_check import check_forward_route  # noqa: E402  (the shared copy, joined to this package by its path)

DEFAULT_BUILD_DIR = Path(__file__).resolve().parents[1] / "build_compact"
INFERENCE_THREADS = G4_GROUPS  # the G4 groups of the kernel's own pool; the kernel takes no thread count


def open_inference_forward(layers, routes) -> InferenceKernel:
    """The inference kernel of the forward (the library the inference leaf builds, loaded once), after the structural
    checks of the tables of every layer (once, at load)."""
    kernel = InferenceKernel(inference_library())
    for index, (layer, route) in enumerate(zip(layers, routes)):
        check_forward_route(index, layer, route, INFERENCE_THREADS)
    return kernel
