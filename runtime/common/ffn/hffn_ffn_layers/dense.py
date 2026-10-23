"""Dense reference of the same FFN, for the benchmarks only (not part of the runtime).

``dense_weights`` materialises ``W_up[d_ff, d_model]`` and ``W_down[d_model, d_ff]`` (zeros
outside the support) with numpy; ``DenseTorch`` computes ``F.linear -> gelu -> F.linear`` with
exact GELU in float32 on PyTorch's CPU thread pool. PyTorch is imported only here.
"""
from __future__ import annotations

import numpy as np

from .synth import CompactLayer


def dense_weights(layer: CompactLayer) -> tuple[np.ndarray, np.ndarray]:
    hidden_up = np.repeat(np.arange(layer.d_ff), np.diff(layer.up_offsets))
    hidden_down = np.repeat(np.arange(layer.d_ff), np.diff(layer.down_offsets))
    w_up = np.zeros((layer.d_ff, layer.d_model), np.float32)
    w_up[hidden_up, layer.up_idx.astype(np.int64)] = layer.up_val
    w_down = np.zeros((layer.d_model, layer.d_ff), np.float32)
    w_down[layer.down_idx.astype(np.int64), hidden_down] = layer.down_val
    return w_up, w_down


class DenseTorch:
    def __init__(self, layer: CompactLayer):
        import torch

        w_up, w_down = dense_weights(layer)
        self.torch = torch
        self.w_up, self.w_down = torch.from_numpy(w_up), torch.from_numpy(w_down)
        self.b_up, self.b_down = torch.from_numpy(layer.b_up.copy()), torch.from_numpy(layer.b_down.copy())

    def forward(self, x_nt: np.ndarray, repeat: int = 1) -> np.ndarray:
        F = self.torch.nn.functional
        x = self.torch.from_numpy(x_nt)
        with self.torch.inference_mode():
            for _ in range(repeat):
                y = F.linear(F.gelu(F.linear(x, self.w_up, self.b_up), approximate="none"), self.w_down, self.b_down)
        return y.numpy()
