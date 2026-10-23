import numpy as np
import pytest

from hffn import checks
from hffn_ffn_layers import layers

pytest.importorskip("torch")

from hffn_ffn_layers.dense import DenseTorch  # noqa: E402


@pytest.mark.parametrize("name", layers.names())
def test_torch_dense_reference(name):
    """The benchmarks' dense PyTorch reference computes the same FFN."""
    layer, case = layers.load(name)
    dense = DenseTorch(layer)
    for n in (1, 8, 32, 96, 128):
        assert np.abs(dense.forward(layers.make_input(case, n)) - checks.oracle(name, n)).max() <= checks.ATOL
