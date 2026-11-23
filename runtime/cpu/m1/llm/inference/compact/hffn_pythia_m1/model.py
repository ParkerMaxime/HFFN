"""Pythia with its FFNs executed dense or by the CPU M1 compact runtime: the shared loader, bound to this leaf.

The HFFN arm is built by ``hffn_pythia.model.load_pythia``; its blocks are then rebound to the leaf's own
``HFFNCompactG4InferenceMLP``, which writes into the runtime's reusable output buffer and passes the sequence length
(README, "Open point for the shared level"). Each export file is read once, by the shared ``read_layers`` with
``groups=True``; the leaf builds its layers from those dicts.
"""
from __future__ import annotations

from pathlib import Path

from torch import nn

from hffn_pythia import model as shared
from hffn_pythia.compact_format import HFFNCompactG4Layer
from hffn_inference_runtime import HFFNCompactG4InferenceMLP, HFFNCompactG4InferenceRuntime

from . import BUILD, RUNTIME_LEAF

G4_GROUPS = 4  # fixed by the kernel: the caller owns group 0, three workers own the others
LAYER_FILE = "hffn_training_compact_g4_v2.pt"


def runtime_files() -> list[Path]:
    return [*sorted((RUNTIME_LEAF / "hffn_inference_runtime").glob("*.py")), RUNTIME_LEAF / "native" / "hffn_compact_g4_inference.cpp",
            RUNTIME_LEAF / "torch_extension" / "hffn_compact_inference_torch_binding.cpp", *sorted(Path(__file__).parent.glob("*.py"))]


def layer_of(raw: dict) -> HFFNCompactG4Layer:
    """The leaf's layer from a ``read_layer(groups=True)`` dict: the constructor and ``validate`` of
    ``hffn_pythia.compact_format.load_hffn_compact_g4_layer``, without its second read of the file."""
    layer = HFFNCompactG4Layer(
        raw["d_model"], raw["d_ff"], raw["group_hidden_offsets"],
        raw["up_row_offsets"], raw["up_idx"], raw["up_val"], raw["b_up"],
        raw["down_row_offsets"], raw["down_idx"], raw["down_val"], raw["b_down"],
    )
    layer.validate()
    return layer


def load_pythia(model_dir: str | Path, export_dir: str | Path, ffn: str) -> nn.Module:
    """Pythia in eval mode (float32, CPU) with its FFNs from ``export_dir``, executed dense or by HFFN."""
    built = {}

    def make_runtime(layers: list[dict]) -> HFFNCompactG4InferenceRuntime:
        compact = [layer_of(raw) for raw in layers]
        built["layers"] = compact
        return HFFNCompactG4InferenceRuntime(compact, build_dir=BUILD)  # execution baseline, one whole-block call per layer

    model = shared.load_pythia(model_dir, export_dir, ffn, make_runtime, groups=True)
    if ffn == "hffn":
        runtime = model.hffn_runtime
        for index, (block, layer) in enumerate(zip(model.gpt_neox.layers, built["layers"])):
            block.mlp = HFFNCompactG4InferenceMLP(index, layer, runtime)
    return model
