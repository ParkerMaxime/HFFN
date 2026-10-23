"""The test layers: generated from ``hffn_ffn_layers.synth`` and pinned by a fingerprint, never stored.

``reference/fingerprints.json`` holds, for each case of ``synth.CASES``, the SHA-256 of the layer arrays and of
the benchmark inputs (``fingerprint``). ``load`` regenerates the layer with numpy's PCG64 generator and refuses a
layer whose fingerprint differs from the pinned one, so every check and every benchmark runs on the same layers
and the same inputs, on any machine.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import synth
from .synth import Case, CompactLayer, make_input as _make_input

ROOT = Path(__file__).resolve().parents[1]
FINGERPRINTS = ROOT / "reference" / "fingerprints.json"
FIELDS = ("groups", "up_offsets", "up_idx", "up_val", "b_up", "down_offsets", "down_idx", "down_val", "b_down")
# Batch sizes whose inputs enter the fingerprint.
FINGERPRINT_N = (1, 2, 4, 7, 8, 16, 32, 64, 96, 128)

_verified: set[str] = set()


def names() -> list[str]:
    return sorted(synth.CASES)


def make_input(case: Case, n: int) -> np.ndarray:
    """Deterministic token-major float32 input [N, d_model] of a case (PCG64, seed [case.seed, n])."""
    return _make_input(case, n)


def fingerprint(layer: CompactLayer, case: Case) -> str:
    """Hash of the layer arrays and of the benchmark inputs."""
    digest = hashlib.sha256()
    for name in FIELDS:
        digest.update(getattr(layer, name).tobytes())
    for n in FINGERPRINT_N:
        digest.update(make_input(case, n).tobytes())
    return digest.hexdigest()


def pinned() -> dict[str, str]:
    return json.loads(FINGERPRINTS.read_text())["fingerprints"]


def load(name: str) -> tuple[CompactLayer, Case]:
    """A fresh layer and its case parameters (fingerprint checked once per process)."""
    case = synth.CASES[name]
    layer = synth.make_layer(case)
    if name not in _verified:
        if fingerprint(layer, case) != pinned()[name]:
            raise ValueError(f"{name}: generated layer or inputs differ from reference/fingerprints.json")
        _verified.add(name)
    return layer, case


def main() -> int:
    """Check every generated layer against its pinned fingerprint (``python -m hffn_ffn_layers.layers``, from ``common/ffn``)."""
    for name in names():
        layer, case = load(name)
        print(f"ok   {name}: fingerprint {pinned()[name][:16]}... d_model={layer.d_model} d_ff={layer.d_ff} "
              f"edges up={layer.up_idx.size} down={layer.down_idx.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
