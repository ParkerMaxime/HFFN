#!/usr/bin/env python3
"""Validate every forward-only Metal schedule against dense fixed support."""

from __future__ import annotations

import argparse
import gc
import json
import traceback
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


def judged(path: Path, args) -> dict:
    result = validate(path, args.n, args.layouts, args.seed)
    failures = [row for row in result["rows"] if not row["finite"] or row["max_abs_error"] > args.atol]
    result["gate"] = {"atol": args.atol, "passed": not failures, "failures": failures}
    return result


def validate_export(args) -> int:
    """Every layer of an export in one process: one record and one line `ok|FAIL <export> <layer>` per layer."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for directory in sorted(args.export_dir.glob("layer*"), key=lambda p: int(p.name[len("layer"):])):
        try:
            result = judged(directory / "hffn_training_compact_g4_v2.pt", args)
            (args.output_dir / f"{directory.name}.json").write_text(json.dumps(result, indent=2) + "\n")
            passed = result["gate"]["passed"]
        except Exception:
            traceback.print_exc()
            passed = False
        print(f"{'ok  ' if passed else 'FAIL'} {args.export_dir.parent.name} {directory.name}", flush=True)
        failed = failed or not passed
        gc.collect()
        torch.mps.empty_cache()
    return int(failed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=Path)
    parser.add_argument("--export-dir", type=Path, help="every layer of this export; needs --output-dir")
    parser.add_argument("--output-dir", type=Path)
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
    if args.export_dir is not None:
        raise SystemExit(validate_export(args))
    result = judged(args.layer, args)
    failures = result["gate"]["failures"]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
