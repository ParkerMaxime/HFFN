"""Without the hidden-major down arrays the recommended routes give the same bits; decode does not depend on T."""
import numpy as np
import pytest

from hffn.compact import CompactKernel
from hffn_ffn_layers import layers


@pytest.mark.parametrize("name", layers.names())
def test_recommended_routes_without_hidden_owned_down(name):
    layer, case = layers.load(name)
    full, lean = CompactKernel(layer), CompactKernel(layer, hidden_owned=False)
    assert lean.layer.down_val.size == 0 and lean.layer.down_idx.size == 0
    for n in (1, 2, 7, 8, 32, 100):
        x = layers.make_input(case, n)
        assert np.array_equal(full.forward(x), lean.forward(x))
    assert lean.model_bytes() == full.model_bytes() < full.model_bytes() + layer.down_val.nbytes
    assert full.model_bytes("baseline") == full.model_bytes() - sum(a.nbytes for a in full._route.values()) + layer.down_offsets.nbytes + layer.down_idx.nbytes + layer.down_val.nbytes
    with pytest.raises(ValueError):
        lean.forward(layers.make_input(case, 8), "baseline")


@pytest.mark.parametrize("name", layers.names())
def test_decode_does_not_depend_on_threads(name):
    layer, case = layers.load(name)
    x = layers.make_input(case, 1)
    outputs = [CompactKernel(layer, threads=t).forward(x) for t in (1, 2, 3, 4, 8)]
    assert all(np.array_equal(outputs[0], y) for y in outputs)
