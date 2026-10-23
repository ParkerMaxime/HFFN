"""Thread count T of the compact kernel: parity for every T, T-invariance of the dual route."""
import numpy as np
import pytest

from hffn import checks
from hffn.compact import CompactKernel, default_threads, dual_route, hidden_partition
from hffn_ffn_layers import layers

THREADS = (1, 2, 3, 8)  # T = 4 is covered by test_parity; 3 gives uneven groups
N = (1, 2, 7, 8, 24, 32, 96, 128)


@pytest.mark.parametrize("threads", THREADS)
@pytest.mark.parametrize("name", layers.names())
def test_parity_any_thread_count(name, threads):
    layer, case = layers.load(name)
    kernel = CompactKernel(layer, threads=threads)
    for n in N:
        x = layers.make_input(case, n)
        for route in checks.compact_routes(n):
            error = float(np.abs(kernel.forward(x, route) - checks.oracle(name, n)).max())
            assert error <= checks.ATOL, f"T={threads} {route} N{n}: |y-float64|={error:.3e}"


@pytest.mark.parametrize("name", layers.names())
def test_dual_is_thread_invariant(name):
    """Each hidden value and each output feature is computed by one thread in slot order."""
    layer, case = layers.load(name)
    kernels = [CompactKernel(layer, threads=threads) for threads in (1, 4, 8)]
    for n in (2, 24, 96):
        x = layers.make_input(case, n)
        first, *others = (kernel.forward(x, "dual") for kernel in kernels)
        assert all(np.array_equal(first, other) for other in others), f"N{n}"


@pytest.mark.parametrize("name", layers.names())
def test_token_result_does_not_depend_on_batch(name):
    """AVX2 builds on glibc: every GELU goes through the same 8-lane erf, so the output of a
    token on the dual route depends neither on N nor on its position in the batch."""
    layer, case = layers.load(name)
    kernel = CompactKernel(layer)
    row = layers.make_input(case, 1)
    reference = kernel.forward(np.ascontiguousarray(np.repeat(row, 2, axis=0)), "dual")[0]
    if not np.array_equal(kernel.forward(np.ascontiguousarray(np.repeat(row, 8, axis=0)), "dual")[7], reference):
        pytest.skip("scalar build: batch tails use the scalar erf")
    for n in (3, 7, 9, 13, 31, 40, 100, 200):
        y = kernel.forward(np.ascontiguousarray(np.repeat(row, n, axis=0)), "dual")
        assert all(np.array_equal(y[token], reference) for token in range(n)), f"N{n}"


def test_partitions():
    layer, _ = layers.load("irreg_512_2048")
    assert np.array_equal(hidden_partition(layer, 4), layer.groups)  # the G4 groups of the layer
    for threads in (1, 3, 8):
        for groups in (hidden_partition(layer, threads), dual_route(layer, threads)["groups"]):
            assert groups.size == threads + 1 and groups[0] == 0 and np.all(np.diff(groups) >= 0)
    assert 1 <= default_threads() <= 8


def test_pool_alternating_thread_counts():
    """One process-wide pool: calls on different T, back to back and after the workers slept."""
    import time

    layer, case = layers.load("irreg_512_2048")
    kernels = {t: CompactKernel(layer, threads=t) for t in (2, 8, 3)}
    x = layers.make_input(case, 24)
    expected = kernels[2].forward(x)
    for index in range(300):
        if index % 50 == 0:
            time.sleep(0.002)  # longer than the spin bound: workers go to sleep
        for kernel in kernels.values():
            assert np.array_equal(kernel.forward(x), expected)
