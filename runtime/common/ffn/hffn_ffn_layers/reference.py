"""Platform-independent float64 reference of the HFFN forward.

Test oracle only: it walks the same hidden-major support as the kernels and
never materialises ``W_up`` or ``W_down``.
"""
from __future__ import annotations

import math

import numpy as np

from .synth import CompactLayer

_erf = np.vectorize(math.erf, otypes=[np.float64])


def gelu_exact(z: np.ndarray) -> np.ndarray:
    return 0.5 * z * (1.0 + _erf(z * 0.7071067811865476))


def forward(layer: CompactLayer, x_nt: np.ndarray) -> np.ndarray:
    """Return ``y[N, d_model]`` in float64."""
    x = x_nt.astype(np.float64)
    n = x.shape[0]
    up_deg = np.diff(layer.up_offsets)
    down_deg = np.diff(layer.down_offsets)
    hidden_ids = np.arange(layer.d_ff)

    z = np.tile(layer.b_up.astype(np.float64), (n, 1))
    up_hidden = np.repeat(hidden_ids, up_deg)
    np.add.at(z.T, up_hidden, (x[:, layer.up_idx] * layer.up_val.astype(np.float64)).T)

    h = gelu_exact(z)

    y = np.tile(layer.b_down.astype(np.float64), (n, 1))
    down_hidden = np.repeat(hidden_ids, down_deg)
    np.add.at(y.T, layer.down_idx.astype(np.int64), (h[:, down_hidden] * layer.down_val.astype(np.float64)).T)
    return y
