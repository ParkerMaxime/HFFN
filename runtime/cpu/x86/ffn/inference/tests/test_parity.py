import numpy as np
import pytest

from hffn import checks
from hffn.compact import CompactKernel, batch_chunks, recommended_route
from hffn_ffn_layers import layers


@pytest.mark.parametrize("name", layers.names())
def test_kernel_parity(name):
    """Every route and compact N of the main check, within 2e-5 of float64."""
    checks.check(name)


@pytest.mark.parametrize("name", layers.names())
def test_compact_extra_n(name):
    """Generic paths, lane tails and the chunking above 96 tokens (N = 100, 128, 200)."""
    layer, case = layers.load(name)
    kernel = CompactKernel(layer)
    for n in checks.EXTRA_N:
        x = layers.make_input(case, n)
        for route in checks.compact_routes(n) + (None,):
            error = float(np.abs(kernel.forward(x, route) - checks.oracle(name, n)).max())
            assert error <= checks.ATOL, f"{route} N{n}: |y-float64|={error:.3e}"


def test_repeated_calls_are_stable():
    """Shared work buffers: a call at another N in between must not change the result."""
    layer, case = layers.load("irreg_512_2048")
    kernel = CompactKernel(layer)
    x8, x96, x1 = (layers.make_input(case, n) for n in (8, 96, 1))
    first = kernel.forward(x8)
    kernel.forward(x96)
    kernel.forward(x1)
    assert np.array_equal(kernel.forward(x8), first)


def test_routes_and_chunks():
    assert recommended_route(1) == "decode" and recommended_route(2) == "dual"
    assert batch_chunks(200, 96) == [(0, 72), (72, 72), (144, 56)]  # balanced, multiples of 8
    assert batch_chunks(128, 96) == [(0, 64), (64, 64)]
    assert batch_chunks(96, 96) == [(0, 96)]
