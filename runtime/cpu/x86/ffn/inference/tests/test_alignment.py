"""Work buffers start on a cache-line boundary, whatever malloc returns and after regrowth."""
import pytest

from hffn.compact import WORK_ALIGNMENT, CompactKernel, aligned_empty
from hffn_ffn_layers import layers


@pytest.mark.parametrize("size", [1, 7, 4096, 196_608])
def test_aligned_empty(size):
    buffers = [aligned_empty(size) for _ in range(8)]
    assert all(b.size == size and b.ctypes.data % WORK_ALIGNMENT == 0 for b in buffers)


def test_work_buffers_aligned():
    layer, case = layers.load("irreg_512_2048")
    kernel = CompactKernel(layer)
    for n in (1, 8, 32, 96, 200):  # each larger batch reallocates
        kernel.forward(layers.make_input(case, n))
        for route in ("baseline", "dual") if n > 1 else ():
            kernel.forward(layers.make_input(case, n), route)
        assert all(buffer.ctypes.data % WORK_ALIGNMENT == 0 for buffer in kernel._flat.values())
