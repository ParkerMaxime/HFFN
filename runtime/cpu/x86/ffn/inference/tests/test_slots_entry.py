"""The slots entry gives the bits of the dual entry fed the gathered copy of the down values."""
import ctypes

import numpy as np
import pytest

from hffn import build
from hffn.compact import aligned_empty, dual_route, hidden_partition
from hffn_ffn_layers import layers

THREADS = (1, 4, 8)
N = (2, 8, 13, 32, 61, 96, 97, 128, 200, 512)  # lane tails, the 96-token limit of the blocked paths, and beyond it


def call(lib, layer, threads, x, slots_entry, separate_z=False):
    """One call of the dual entry (gathered copy of down_val) or of the slots entry (canonical down_val and down_slots)."""
    n, h, f = x.shape[0], layer.d_model, layer.d_ff
    route = dual_route(layer, threads)
    slots = np.ascontiguousarray(np.argsort(layer.down_idx.astype(np.int64), kind="stable").astype(np.int32))
    groups = hidden_partition(layer, threads)
    x_t, act, y = aligned_empty(h * n), aligned_empty(f * n), np.empty_like(x)
    z = aligned_empty(f * n) if separate_z else act
    timing = np.zeros(4, np.int64)
    a = lambda array: array.ctypes.data  # noqa: E731
    up = [a(groups), a(layer.b_up), a(layer.up_offsets), a(layer.up_idx), a(layer.up_val)]
    if slots_entry:
        function = lib.hffn_compact_forward_dual_slots_nt_f32
        down = [a(layer.b_down), a(route["offsets"]), a(route["hidden"]), a(layer.down_val), a(slots), a(route["groups"])]
    else:
        function = lib.hffn_compact_forward_dual_nt_f32
        gathered = np.ascontiguousarray(layer.down_val[slots])
        down = [a(layer.b_down), a(route["offsets"]), a(route["hidden"]), a(gathered), a(route["groups"])]
    function.restype = None
    function.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * (1 + len(up) + len(down)) + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 6
    function(threads, a(x), *up, *down, n, h, f, a(x_t), a(y), a(x_t), a(z), a(act), a(timing))
    return y, act.copy(), z.copy()


@pytest.mark.parametrize("name", layers.names())
def test_slots_entry_is_bitwise_the_dual_entry(name):
    """Output and hidden activation, every N and T below, on the build of the leaf (AVX2 or scalar, as HFFN_CXXFLAGS sets it)."""
    layer, case = layers.load(name)
    lib = ctypes.CDLL(str(build.build()))
    for threads in THREADS:
        for n in N:
            x = layers.make_input(case, n)
            y0, act0, _ = call(lib, layer, threads, x, False)
            y1, act1, _ = call(lib, layer, threads, x, True)
            assert np.array_equal(y0.view(np.uint32), y1.view(np.uint32)), f"y: T={threads} N{n}"
            assert np.array_equal(act0.view(np.uint32), act1.view(np.uint32)), f"activation: T={threads} N{n}"


@pytest.mark.parametrize("name", layers.names())
def test_z_store_branch(name):
    """With a z buffer distinct from the activation, z holds the pre-activation and y, activation keep the same bits."""
    layer, case = layers.load(name)
    lib = ctypes.CDLL(str(build.build()))
    for threads in (1, 4):
        for n in (2, 13, 32, 97, 200):
            x = layers.make_input(case, n)
            y0, act0, _ = call(lib, layer, threads, x, True)
            y1, act1, z1 = call(lib, layer, threads, x, True, separate_z=True)
            assert np.array_equal(y0.view(np.uint32), y1.view(np.uint32)), f"y: T={threads} N{n}"
            assert np.array_equal(act0.view(np.uint32), act1.view(np.uint32)), f"activation: T={threads} N{n}"
            reference = np.empty((layer.d_ff, n))
            for r in range(layer.d_ff):
                lo, hi = layer.up_offsets[r], layer.up_offsets[r + 1]
                reference[r] = float(layer.b_up[r]) + x[:, layer.up_idx[lo:hi]].astype(np.float64) @ layer.up_val[lo:hi].astype(np.float64)
            stored = z1.reshape(layer.d_ff, n)
            assert np.allclose(stored, reference, rtol=1e-4, atol=1e-4 * np.abs(reference).max()), f"z: T={threads} N{n}"
