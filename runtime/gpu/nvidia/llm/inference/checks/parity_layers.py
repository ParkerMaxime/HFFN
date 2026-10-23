"""Parity of the HFFN CUDA runtime on every FFN of one export, layer by layer.

For each layer and each N, the runtime runs with its policy and with each of the nine forced
(up, down) kernel pairs (``hffn_cuda.parity``, shared with the isolated-FFN leaf). Criteria:
- two consecutive calls on the same input give the same bits;
- ``max |y - y_float64| / max(1, |y_float64|) <= GUARD`` against a float64 oracle of the same
  fixed-support FFN (exact GELU). GUARD detects a wrong kernel; it is not an accuracy claim.
Printed without threshold: the absolute distance to the oracle, next to that of the float32
PyTorch dense FFN on the same device, and whether the four sequential pairs (up lane/tile,
down tiled/blocked) give the same bits.

    python -B -m checks.parity_layers --export-dir <export>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from hffn_pythia_cuda.model import layer_from_export
from hffn_pythia_cuda.prepared_runtime import PreparedRuntime
from hffn_cuda.parity import PAIRS, check
from hffn_pythia.export import read_layers
from hffn_pythia.oracle import dense_forward, dense_weights

BATCHES = (1, 2, 4, 5, 32, 63, 64, 128, 129, 256, 512)
GUARD = 1e-3


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", type=Path, required=True)
    args = parser.parse_args()
    count = len(list(args.export_dir.glob("layer*")))
    layers = read_layers(args.export_dir, count)
    runtime = PreparedRuntime([layer_from_export(layer) for layer in layers])
    generator = torch.Generator().manual_seed(0)
    worst = worst_dense = worst_relative = 0.0
    repeatable = sequential_bits = True
    for index, layer in enumerate(layers):
        weights = dense_weights(layer, runtime.device)
        errors = []
        for n in BATCHES:
            x = torch.randn(n, layer["d_model"], generator=generator).to(runtime.device)
            expected = dense_forward(weights, x.double())
            worst_dense = max(worst_dense, float((dense_forward(weights, x).double() - expected).abs().max()))
            result = check(runtime, index, x, expected)
            repeatable &= result.repeatable
            sequential_bits &= result.sequential_bits
            worst_relative = max(worst_relative, result.relative)
            errors.append(result.error)
        worst = max(worst, *errors)
        print(f"layer {index:2d}  max|y-float64| " + "  ".join(f"N{n}={error:.2e}" for n, error in zip(BATCHES, errors)), flush=True)
    passed = repeatable and worst_relative <= GUARD
    print(f"{args.export_dir.parent.name}: {count} layers, {len(PAIRS)} kernel pairs, same bits on repeat: {repeatable}, "
          f"max relative distance {worst_relative:.2e} (guard {GUARD:.0e}) -> {'PASSED' if passed else 'FAILED'}; "
          f"max|y-float64| hffn {worst:.2e}, dense float32 {worst_dense:.2e}; sequential pairs same bits: {sequential_bits}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
