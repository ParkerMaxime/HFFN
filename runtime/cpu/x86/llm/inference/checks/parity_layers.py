"""Parity of the HFFN runtime on every FFN of one export, layer by layer.

Criterion: for each layer and each batch size, the output with T threads has the same bits as
with one thread. The distance to a float64 oracle of the same fixed-support FFN (exact GELU) is
printed, next to that of the float32 PyTorch dense FFN; it measures the float32 accumulation
order and is not a criterion (the token gap of ``checks.parity_model`` is).

    python -B -m checks.parity_layers --export-dir <export>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from hffn_pythia_x86.runtime import HFFNLayer, HFFNRuntime
from hffn_pythia.export import read_layers
from hffn_pythia.oracle import dense_forward

BATCHES = (1, 2, 7, 8, 32, 96, 128, 256)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    count = len(list(args.export_dir.glob("layer*")))
    layers = read_layers(args.export_dir, count)
    compact = [HFFNLayer.from_export(layer) for layer in layers]
    runtime, single = HFFNRuntime(compact, args.threads), HFFNRuntime(compact, 1)
    generator = torch.Generator().manual_seed(0)
    worst = worst_dense = 0.0
    same_bits = True
    for index, layer in enumerate(layers):
        errors = []
        for n in BATCHES:
            x = torch.randn(n, layer["d_model"], generator=generator)
            expected = dense_forward(layer, x.double())
            y = runtime.forward(index, x).clone()
            same_bits &= bool(torch.equal(y, single.forward(index, x)))
            errors.append(float((y.double() - expected).abs().max()))
            worst_dense = max(worst_dense, float((dense_forward(layer, x).double() - expected).abs().max()))
        worst = max(worst, *errors)
        print(f"layer {index:2d}  max|y-float64| " + "  ".join(f"N{n}={error:.2e}" for n, error in zip(BATCHES, errors)), flush=True)
    print(f"{args.export_dir.parent.name}: {count} layers, T={runtime.threads}, bits independent of T: {same_bits} -> {'PASSED' if same_bits else 'FAILED'}; "
          f"max|y-float64| hffn {worst:.2e}, dense float32 {worst_dense:.2e}")
    return 0 if same_bits else 1


if __name__ == "__main__":
    sys.exit(main())
