#!/usr/bin/env python3
"""Validate every forward-only Metal schedule against dense fixed support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hffn_metal_llm_inference import (
    HFFNMetalInferenceRuntime,
    load_compact_wanda_inference_layer,
)
from hffn_pythia.oracle import dense_forward, dense_weights  # on sys.path once the runtime package is imported


LAYER_KEYS = ("d_ff", "d_model", "up_row_offsets", "up_idx", "up_val", "b_up",
              "down_row_offsets", "down_idx", "down_val", "b_down")


def validate(path: Path, values_n: list[int], layouts: list[str], seed: int) -> dict:
    torch.manual_seed(seed)
    spec = load_compact_wanda_inference_layer(path)
    layer = {key: getattr(spec, key) for key in LAYER_KEYS}
    weights = dense_weights(layer)
    runtime = HFFNMetalInferenceRuntime([spec])
    rows = []
    with torch.inference_mode():
        for n in values_n:
            x = torch.randn(n, spec.d_model, device="mps", dtype=torch.float32)
            reference = dense_forward(weights, x.cpu().double())
            for layout in layouts:
                runtime.set_forced_layout(layout)
                runtime.begin_model_call()
                actual = runtime.forward(0, x)
                runtime.end_model_call()
                delta = (actual.cpu().double() - reference).abs()
                rows.append({
                    "N": n,
                    "layout": layout,
                    "max_abs_error": float(delta.max()),
                    "mean_abs_error": float(delta.mean()),
                    "finite": bool(torch.isfinite(actual).all().cpu()),
                })
    runtime.synchronize()
    return {
        "format": "hffn_metal_wanda_inference_layer_parity_v1",
        "layer": str(path.resolve()),
        "shape": {"d_model": spec.d_model, "d_ff": spec.d_ff},
        "compact_numel": spec.compact_numel,
        "rows": rows,
        "workspace": runtime.workspace_stats(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=Path, required=True)
    parser.add_argument("--n", type=int, nargs="+", default=[1, 8, 32, 128, 256, 512])
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=("token_major", "owner_row_reduce", "feature_persistent", "feature_owner_tile"),
        default=["token_major", "owner_row_reduce", "feature_persistent", "feature_owner_tile"],
    )
    parser.add_argument("--seed", type=int, default=712)
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available; run outside the GPU sandbox")
    if any(n < 1 for n in args.n):
        raise ValueError("N values must be positive")
    result = validate(args.layer, args.n, args.layouts, args.seed)
    failures = [row for row in result["rows"] if not row["finite"] or row["max_abs_error"] > args.atol]
    result["gate"] = {"atol": args.atol, "passed": not failures, "failures": failures}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
