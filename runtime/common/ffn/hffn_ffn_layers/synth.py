"""Seeded synthetic HFFN layers in the canonical compact hidden-major layout.

Nothing is stored: every array is regenerated from ``(case, seed)`` with numpy's PCG64 generator
(``hffn_ffn_layers.layers`` checks the result against pinned fingerprints).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_GROUPS = 4


@dataclass(frozen=True)
class Case:
    name: str
    d_model: int
    d_ff: int
    up_degree: int
    down_degree: int
    irregular: bool
    seed: int


# Homogeneous cases give every hidden neuron the same degree; the irregular case has varying degrees
# (a WANDA-like support) and exercises the exact compact offsets.
CASES = {
    case.name: case
    for case in (
        Case("homog_256_1024", 256, 1024, 26, 26, False, 1001),
        Case("homog_512_2048", 512, 2048, 51, 51, False, 1002),
        Case("irreg_512_2048", 512, 2048, 51, 51, True, 1003),
    )
}


@dataclass(frozen=True)
class CompactLayer:
    d_model: int
    d_ff: int
    groups: np.ndarray        # int32[5], hidden partition of the four G4 groups
    up_offsets: np.ndarray    # int32[d_ff + 1]
    up_idx: np.ndarray        # int16[nnz_up], input feature of each slot
    up_val: np.ndarray        # float32[nnz_up]
    b_up: np.ndarray          # float32[d_ff]
    down_offsets: np.ndarray  # int32[d_ff + 1]
    down_idx: np.ndarray      # int16[nnz_down], output feature of each slot
    down_val: np.ndarray      # float32[nnz_down]
    b_down: np.ndarray        # float32[d_model]


def _degrees(rng: np.random.Generator, d_ff: int, degree: int, d_model: int, irregular: bool) -> np.ndarray:
    if not irregular:
        return np.full(d_ff, degree, dtype=np.int64)
    # Includes zero-degree neurons and heavy neurons, as a pruned support does.
    return rng.integers(0, min(2 * degree, d_model) + 1, size=d_ff, dtype=np.int64)


def _projection(rng: np.random.Generator, degrees: np.ndarray, d_model: int, scale: float):
    offsets = np.zeros(degrees.size + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(degrees)
    idx = np.empty(int(offsets[-1]), dtype=np.int16)
    for hidden, degree in enumerate(degrees):
        chosen = np.sort(rng.choice(d_model, size=int(degree), replace=False))
        idx[offsets[hidden]:offsets[hidden + 1]] = chosen
    val = (rng.standard_normal(idx.size) * scale).astype(np.float32)
    return offsets, idx, val


def make_layer(case: Case) -> CompactLayer:
    rng = np.random.default_rng(case.seed)
    up_deg = _degrees(rng, case.d_ff, case.up_degree, case.d_model, case.irregular)
    down_deg = _degrees(rng, case.d_ff, case.down_degree, case.d_model, case.irregular)
    up_offsets, up_idx, up_val = _projection(rng, up_deg, case.d_model, 1.0 / np.sqrt(case.up_degree))
    down_offsets, down_idx, down_val = _projection(rng, down_deg, case.d_model, 1.0 / np.sqrt(case.down_degree))
    b_up = (rng.standard_normal(case.d_ff) * 0.1).astype(np.float32)
    b_down = (rng.standard_normal(case.d_model) * 0.1).astype(np.float32)
    groups = np.array([(g * case.d_ff) // N_GROUPS for g in range(N_GROUPS + 1)], dtype=np.int32)
    return CompactLayer(case.d_model, case.d_ff, groups, up_offsets, up_idx, up_val, b_up, down_offsets, down_idx, down_val, b_down)


def make_input(case: Case, n: int) -> np.ndarray:
    """Token-major float32 input ``[N, d_model]``."""
    rng = np.random.default_rng([case.seed, n])
    return np.ascontiguousarray(rng.standard_normal((n, case.d_model)).astype(np.float32))
