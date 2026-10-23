"""Pythia with its FFNs executed dense or by the CPU M1 compact runtime: the shared loader, bound to this leaf.

The HFFN arm is built by ``hffn_pythia.model.load_pythia``; its blocks are then rebound to the leaf's own
``HFFNCompactG4InferenceMLP``, which writes into the runtime's reusable output buffer and passes the sequence length
(README, "Open point for the shared level"). Layers are read with the leaf's loader, from the same files as the
shared ``read_layers``, because the shared dicts do not carry ``group_hidden_offsets``.
"""
from __future__ import annotations

from pathlib import Path

from torch import nn

from hffn_pythia import model as shared
from hffn_inference_runtime import HFFNCompactG4InferenceMLP, HFFNCompactG4InferenceRuntime, load_hffn_compact_g4_layer

from . import BUILD, LEAF

G4_GROUPS = 4  # fixed by the kernel: the caller owns group 0, three workers own the others
LAYER_FILE = "hffn_training_compact_g4_v2.pt"


def runtime_files() -> list[Path]:
    return [*sorted((LEAF / "hffn_inference_runtime").glob("*.py")), LEAF / "native" / "hffn_compact_g4_inference.cpp",
            LEAF / "torch_extension" / "hffn_compact_inference_torch_binding.cpp", *sorted(Path(__file__).parent.glob("*.py"))]


def load_pythia(model_dir: str | Path, export_dir: str | Path, ffn: str) -> nn.Module:
    """Pythia in eval mode (float32, CPU) with its FFNs from ``export_dir``, executed dense or by HFFN."""
    built = {}

    def make_runtime(layers: list[dict]) -> HFFNCompactG4InferenceRuntime:
        compact = [load_hffn_compact_g4_layer(Path(export_dir) / f"layer{i}" / LAYER_FILE) for i in range(len(layers))]
        for layer, spec in zip(layers, compact):
            if (layer["d_model"], layer["d_ff"]) != (spec.d_model, spec.d_ff) or not layer["up_val"].equal(spec.up_val):
                raise ValueError("shared reader and leaf loader disagree on the export")
        built["layers"] = compact
        return HFFNCompactG4InferenceRuntime(compact, build_dir=BUILD)  # execution baseline, batch_split32 on

    model = shared.load_pythia(model_dir, export_dir, ffn, make_runtime)
    if ffn == "hffn":
        runtime = model.hffn_runtime
        for index, (block, layer) in enumerate(zip(model.gpt_neox.layers, built["layers"])):
            block.mlp = HFFNCompactG4InferenceMLP(index, layer, runtime)
    return model
