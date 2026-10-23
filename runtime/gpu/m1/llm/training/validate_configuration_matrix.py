#!/usr/bin/env python3
"""Validate every canonical Pythia/WANDA training configuration.

The CPU phase checks every layer export.  With ``--gpu-parity`` the first
layer of every shape/density is additionally compared with the dense fixed-
support reference in each selected scheduling regime.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from hffn_metal_llm import load_compact_wanda_layer, select_training_layout
from validate_wanda_layer import validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--models", type=int, nargs="+", default=[70, 160, 410])
    parser.add_argument("--densities", type=int, nargs="+", default=[10, 20, 30, 50, 70, 80])
    parser.add_argument("--n", type=int, nargs="+", default=[1, 8, 256])
    parser.add_argument("--gpu-parity", action="store_true")
    parser.add_argument("--seed", type=int, default=712)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.gpu_parity and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available; run --gpu-parity outside the GPU sandbox")

    rows = []
    for model_size in args.models:
        for density in args.densities:
            export = (
                args.model_root
                / f"pythia{model_size}m_sparse{density}_ffn_only"
                / "export"
            )
            paths = sorted(
                export.glob("layer*/hffn_training_compact_g4_v2.pt"),
                key=lambda path: int(path.parent.name.removeprefix("layer")),
            )
            if not paths:
                raise FileNotFoundError(f"no canonical compact layers under {export}")
            specs = [load_compact_wanda_layer(path) for path in paths]
            shape = {(spec.d_model, spec.d_ff) for spec in specs}
            if len(shape) != 1:
                raise AssertionError(f"inconsistent FFN shapes under {export}: {shape}")
            nnz = {(spec.up_val.numel(), spec.down_val.numel()) for spec in specs}
            layouts = {str(n): select_training_layout(specs[0], n) for n in args.n}
            gpu_rows = []
            if args.gpu_parity:
                for n in args.n:
                    layout = layouts[str(n)]
                    gpu_rows.append(validate(specs[0], n, args.seed + model_size + density, layout))
                torch.mps.synchronize()
            rows.append({
                "model": f"pythia-{model_size}m",
                "density": density,
                "layers": len(specs),
                "shape": {"d_model": specs[0].d_model, "d_ff": specs[0].d_ff},
                "nnz_range": {
                    "up": [min(value[0] for value in nnz), max(value[0] for value in nnz)],
                    "down": [min(value[1] for value in nnz), max(value[1] for value in nnz)],
                },
                "selected_layouts": layouts,
                "gpu_parity": gpu_rows,
            })
            del specs
            gc.collect()
            if args.gpu_parity:
                torch.mps.empty_cache()

    result = {
        "format": "hffn_metal_pythia_configuration_matrix_v1",
        "gpu_parity": args.gpu_parity,
        "n": args.n,
        "configurations": len(rows),
        "rows": rows,
    }
    if len(rows) != len(args.models) * len(args.densities):
        raise AssertionError("configuration matrix is incomplete")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
