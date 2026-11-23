#!/usr/bin/env python3
"""Reproducible parity gate of the homogeneous HFFN Metal runtime.

This script checks an AdamW trajectory with distinct micro-batches. It does not
measure performance and therefore does not replace the ABBA benchmark protocol.
"""

import json
import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir()) / "runtime/gpu/m1/ffn/training"
BUILD = ROOT / "build"   # binaries of the build_*.sh scripts of this leaf
RUNNER = BUILD / "metal_hffn_train_step_runner"
VALIDATOR = BUILD / "validate_train_step_multistep"
SCRATCH = Path(tempfile.gettempdir()) / f"hffn_homogeneous_runtime_validation_{os.getpid()}"   # per process: concurrent checks never share dumps


def run_case(name: str, i: int, h: int, o: int, n: int, d: int,
             owner_tile: bool) -> dict:
    prefix = SCRATCH / name
    output = SCRATCH / f"{name}.json"
    cmd = [
        str(RUNNER), "--input", str(i), "--hidden", str(h),
        "--output-size", str(o), "--n", str(n), "--degree", str(d),
        "--accum-steps", "2", "--verify-steps", "10", "--dump-prefix",
        str(prefix), "--output", str(output),
    ]
    if owner_tile:
        cmd.append("--owner-tile")
    subprocess.run(cmd, check=True)
    checked = subprocess.run(
        [str(VALIDATOR), str(i), str(h), str(o), str(n), str(d), "10",
         str(prefix), "2"], check=True, text=True, capture_output=True
    )
    values = {}
    for line in checked.stdout.splitlines():
        key, value = line.split()
        values[key] = float(value)
    return {"runner": json.loads(output.read_text()), "validator_max_abs": max(values.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner-tile", action="store_true",
                        help="validate the owner-tile route instead of persistent-N")
    a = parser.parse_args()
    if not RUNNER.exists() or not VALIDATOR.exists():
        raise RuntimeError("compiler: zsh runtime/gpu/m1/ffn/training/build_train_step_runner.sh && zsh runtime/gpu/m1/ffn/training/build_validate_train_step_multistep.sh")
    SCRATCH.mkdir()
    result = {
        "kind": "hffn_metal_homogeneous_runtime_training_parity",
        "microbatches": 2,
        "adamw_steps": 10,
        "cases": {
            "compact16": run_case("compact16", 128, 512, 128, 65, 13,
                                   a.owner_tile),
            "compact32": run_case("compact32", 64, 2049, 64, 65, 13,
                                   a.owner_tile),
        },
    }
    print(json.dumps(result, indent=2))
    shutil.rmtree(SCRATCH)


if __name__ == "__main__":
    main()
