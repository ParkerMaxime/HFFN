#!/usr/bin/env python3
"""Fresh-process full-model parity matrix across canonical Pythia/WANDA runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--models", type=int, nargs="+", default=[70, 160, 410])
    parser.add_argument("--densities", type=int, nargs="+", default=[10, 20, 30, 50, 70, 80])
    parser.add_argument("--prompt-length", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=2)
    parser.add_argument("--generation-tokens", type=int, default=2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    script = Path(__file__).resolve().parent / "smoke_pythia_inference.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_size in args.models:
        for density in args.densities:
            model_dir = args.model_root / f"pythia{model_size}m"
            export_dir = args.export_root / f"pythia{model_size}m_sparse{density}_ffn_only" / "export"
            output = args.output_dir / f"pythia{model_size}m_d{density}.json"
            command = [
                sys.executable,
                str(script),
                "--model-dir", str(model_dir),
                "--export-dir", str(export_dir),
                "--prompt-length", str(args.prompt_length),
                "--decode-tokens", str(args.decode_tokens),
                "--generation-tokens", str(args.generation_tokens),
                "--atol", str(args.atol),
                "--output", str(output),
            ]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            payload = json.loads(output.read_text())
            rows.append({
                "model": f"pythia-{model_size}m",
                "density": density,
                "prefill_max_abs_error": payload["prefill_max_abs_error"],
                "decode_max_abs_error": max(payload["decode_max_abs_errors"]),
                "prefill_argmax_equal": payload["prefill_argmax_equal"],
                "generation_equal": payload["generation_equal"],
                "passed": payload["gate"]["passed"],
            })
    failures = [row for row in rows if not row["passed"]]
    result = {
        "format": "hffn_metal_pythia_inference_full_matrix_v1",
        "configurations": len(rows),
        "condition": {
            "prompt_length": args.prompt_length,
            "decode_tokens": args.decode_tokens,
            "generation_tokens": args.generation_tokens,
            "atol": args.atol,
        },
        "rows": rows,
        "gate": {"passed": not failures, "failures": failures},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "configurations": len(rows),
        "max_prefill_error": max(row["prefill_max_abs_error"] for row in rows),
        "max_decode_error": max(row["decode_max_abs_error"] for row in rows),
        "argmax_all": all(row["prefill_argmax_equal"] for row in rows),
        "generation_all": all(row["generation_equal"] for row in rows),
        "gate": result["gate"],
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
