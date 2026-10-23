"""The kernel arrays of a shared synthetic test layer."""
from __future__ import annotations

import torch

from hffn_ffn_layers.synth import CompactLayer


def cuda_layer(layer: CompactLayer):
    """``hffn_cuda.runtime.HFFNLayer`` of a test layer (CPU tensors; the runtime moves them to the device)."""
    from hffn_cuda.runtime import HFFNLayer
    t = torch.from_numpy
    return HFFNLayer.from_hidden_major(layer.d_model, layer.d_ff, t(layer.b_up), t(layer.up_offsets), t(layer.up_idx), t(layer.up_val),
                                       t(layer.b_down), t(layer.down_offsets), t(layer.down_idx), t(layer.down_val))
