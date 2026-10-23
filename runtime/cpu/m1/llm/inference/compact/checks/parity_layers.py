"""Per-layer check of the CPU M1 compact runtime on every FFN of one export (README, gate 1).

Criterion: for each layer and each N, two calls on the same input give the same bits (G4 is fixed, so the x86
criterion "same bits for every thread count" becomes determinism). The distance to a float64 oracle of the same
fixed-support FFN (exact GELU) is printed next to that of the float32 PyTorch dense FFN; it is not a criterion.

    python -B -m checks.parity_layers --export-dir <export>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from hffn_pythia_m1 import BUILD
from hffn_pythia_m1.model import LAYER_FILE
from hffn_inference_runtime import HFFNCompactG4InferenceRuntime, load_hffn_compact_g4_layer
from hffn_pythia.export import read_layers
from hffn_pythia.oracle import dense_forward

BATCHES = (1, 2, 7, 8, 32, 96, 128, 256)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", type=Path, required=True)
    args = parser.parse_args()
    count = len(list(args.export_dir.glob("layer*")))
    layers = read_layers(args.export_dir, count)
    runtime = HFFNCompactG4InferenceRuntime([load_hffn_compact_g4_layer(args.export_dir / f"layer{i}" / LAYER_FILE)
                                             for i in range(count)], build_dir=BUILD)
    generator = torch.Generator().manual_seed(0)
    worst = worst_dense = 0.0
    same_bits = True
    with torch.inference_mode():
        for index, layer in enumerate(layers):
            errors = []
            for n in BATCHES:
                x = torch.randn(n, layer["d_model"], generator=generator)
                expected = dense_forward(layer, x.double())
                first = runtime.forward_into(index, x, torch.empty_like(x))
                second = runtime.forward_into(index, x, torch.empty_like(x))
                same_bits &= bool(torch.equal(first, second))
                errors.append(float((first.double() - expected).abs().max()))
                worst_dense = max(worst_dense, float((dense_forward(layer, x).double() - expected).abs().max()))
            worst = max(worst, *errors)
            print(f"layer {index:2d}  max|y-float64| " + "  ".join(f"N{n}={e:.2e}" for n, e in zip(BATCHES, errors)), flush=True)
    print(f"{args.export_dir.parent.name}: {count} layers, deterministic: {same_bits} -> {'PASSED' if same_bits else 'FAILED'}; "
          f"max|y-float64| hffn {worst:.2e}, dense float32 {worst_dense:.2e}")
    return 0 if same_bits else 1


if __name__ == "__main__":
    sys.exit(main())
