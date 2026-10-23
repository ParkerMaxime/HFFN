"""Parity of the HFFN CUDA runtime on the shared synthetic test layers, against the float64 oracle.

For each test layer (``hffn_ffn_layers``, regenerated and checked against its pinned fingerprint) and each N, the
runtime runs with its policy and with each of the nine forced (up, down) kernel pairs (``hffn_cuda.parity``).
Criteria:
- two consecutive calls on the same input give the same bits;
- ``max |y - y_float64| <= TOLERANCE``, the tolerance of the x86 FFN leaf on the same layers, inputs and oracle
  (exact GELU, ``hffn_ffn_layers.reference``).
Printed without threshold: the relative distance, the distance of the float32 dense iso-support FFN on the same
device (``torch.nn.functional.linear``, the control arm), and whether the four sequential pairs give the same bits.

    python -B -m checks.parity
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

import hffn_ffn_cuda
from hffn_cuda.parity import PAIRS, check
from hffn_cuda.runtime import HFFNRuntime
from hffn_ffn_layers import dense, layers, reference

BATCHES = (1, 2, 4, 5, 7, 8, 32, 63, 64, 96, 128, 129, 256, 512)
TOLERANCE = 2e-5


@torch.inference_mode()
def main() -> int:
    device = torch.device("cuda")
    failed = 0
    for name in layers.names():
        layer, case = layers.load(name)
        runtime = HFFNRuntime([hffn_ffn_cuda.cuda_layer(layer)], device)
        w_up, w_down = (torch.from_numpy(w).to(device) for w in dense.dense_weights(layer))
        b_up, b_down = (torch.from_numpy(b.copy()).to(device) for b in (layer.b_up, layer.b_down))
        relative = worst_dense = 0.0
        repeatable = sequential_bits = True
        errors = []
        for n in BATCHES:
            x_np = layers.make_input(case, n)
            expected = torch.from_numpy(reference.forward(layer, x_np)).to(device)
            x = torch.from_numpy(x_np).to(device)
            y_dense = F.linear(F.gelu(F.linear(x, w_up, b_up), approximate="none"), w_down, b_down)
            worst_dense = max(worst_dense, float((y_dense.double() - expected).abs().max()))
            result = check(runtime, 0, x, expected)
            repeatable &= result.repeatable
            sequential_bits &= result.sequential_bits
            relative = max(relative, result.relative)
            errors.append(result.error)
        error = max(errors)
        passed = repeatable and error <= TOLERANCE
        failed += not passed
        print("  " + "  ".join(f"N{n}={e:.2e}" for n, e in zip(BATCHES, errors)), flush=True)
        print(f"{name}: d_model={layer.d_model} d_ff={layer.d_ff} edges up={layer.up_idx.size} down={layer.down_idx.size}, "
              f"{len(PAIRS)} kernel pairs, same bits on repeat: {repeatable}, max|y-float64| {error:.2e} "
              f"(tolerance {TOLERANCE:.0e}) -> {'PASSED' if passed else 'FAILED'}; relative {relative:.2e}, "
              f"dense float32 {worst_dense:.2e}; sequential pairs same bits: {sequential_bits}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
