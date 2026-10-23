import pytest

import hffn  # noqa: F401  (puts the shared test layers on sys.path)
from hffn_ffn_layers import layers, synth


def test_layers_match_the_pinned_fingerprints():
    """The generated layers and their benchmark inputs are exactly the pinned ones."""
    assert layers.names() == sorted(layers.pinned())
    for name in layers.names():
        layer, case = layers.load(name)  # also checks the fingerprint
        assert layers.fingerprint(layer, case) == layers.pinned()[name]
        assert case == synth.CASES[name]


def test_a_changed_layer_is_refused(monkeypatch):
    monkeypatch.setattr(layers, "_verified", set())
    monkeypatch.setattr(layers, "pinned", lambda: {name: "0" * 64 for name in synth.CASES})
    with pytest.raises(ValueError):
        layers.load("homog_256_1024")
