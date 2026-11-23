"""Untimed every-case check of the GPU M1 isolated-FFN training leaf.

Runs the leaf's own validators on the tree as it is and gates them with the bounds those validators already declare.
No time and no memory figure is read: medians printed by the standalone runners are dropped from the records.

Gates (fixed before the first run of this script, 2026-10-20):
  smoke_codec       hffn_route_codec_smoke: exit 0 (compact16 H=2048 D=32, compact32 H=2049 D=32 and H=512 D=51)
  smoke_gelu_exact  hffn_gelu_exact_smoke: exit 0 (finite-difference derivative within 2e-4)
  smoke_gelu_erf    hffn_gelu_erf_accurate_smoke: exit 0 (value within 5e-7, derivative within 3e-7 of exact GELU)
  build             the five build scripts of the leaf: exit 0
  train_step        validate_homogeneous_runtime.py (persistent-N, compact16 and compact32 cases, two micro-batches,
                    ten AdamW steps): exit 0, i.e. validate_train_step_multistep max abs <= 2e-3 on each case
  forward_relu      training forward runner --untimed on a compact16 and a compact32 case: max abs error of z, h and
                    y against the CPU reference <= 1e-5
  backward_runner   standalone backward runner, compact16 and compact32: exit 0 and finite errors (recorded, no bound:
                    this prototype declares none)
  adamw_runner      standalone AdamW runner, --dispatches 1 and 4, five verify steps: exit 0 and finite error
                    (recorded, no bound: this prototype declares none)
  dense_va          validate_pytorch_mps_dense_iso_support_adamw.py (defaults): finite errors, inactive slots 0.0
  export_inprocess  export_gate.py on the ten synthetic exports at N1/8/64/512 with the fault arm: 160 cells, every clean
                    cell inside the float64 band, every fault detected
  dense_tj          validate_learning_trajectory_iso_support.py (defaults, 20 epochs): functional_pass true,
                    inactive 0.0; strict_numerical_pass recorded
Not covered: the owner-tile route (known failing gate), the timed benches (AB, train-step bench, profile).
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path

LEAF = Path(__file__).resolve().parent
ROOT = next(p for p in LEAF.parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
BUILD = LEAF / "build"
PY = sys.executable
BUILD_SCRIPTS = ("build_train_step_runner.sh", "build_training_forward_runner.sh", "build_backward_runner.sh",
                 "build_adamw_runner.sh", "build_validate_train_step_multistep.sh")
SMOKES = (("smoke_codec", "hffn_route_codec_smoke.cpp"), ("smoke_gelu_exact", "hffn_gelu_exact_smoke.cpp"),
          ("smoke_gelu_erf", "hffn_gelu_erf_accurate_smoke.cpp"))
# (name, input, hidden, output, N, degree)
# H=2049 with D=13 still fits compact16 (a known property of the route codec): D=32 forces the compact32 route.
FORWARD_CASES = (("compact16", 37, 101, 29, 65, 7), ("compact32", 64, 2049, 64, 65, 32))
BACKWARD_CASES = (("compact16", 128, 512, 128, 65, 13), ("compact32", 64, 2049, 64, 65, 32))
EXPORT_GRID = ("512_2048_d005", "512_2048_d010", "512_2048_d020", "512_2048", "512_2048_d050", "512_2048_d075",
               "512_2048_d090", "512_2048_d100", "256_1024_d010", "1024_4096_d010")
EXPORT_CELLS = len(EXPORT_GRID) * 4 * 4          # 10 exports x N1/8/64/512 x (clean, index, value, support_write)
PLANNED = len(SMOKES) + 1 + 1 + len(FORWARD_CASES) + len(BACKWARD_CASES) + 2 + 1 + 1 + 1
SOURCES = sorted(p.name for p in LEAF.iterdir() if p.suffix in {".py", ".mm", ".cpp", ".h", ".sh"})


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(values) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


class Check:
    def __init__(self, out: Path):
        self.out = out
        self.rows = []

    def emit(self, row: dict) -> None:
        self.rows.append(row)
        with (self.out / "raw.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
        if not row["passed"]:
            raise AssertionError(f"gate {row['gate']} failed")

    def run(self, cmd, log: str) -> subprocess.CompletedProcess:
        print("+ " + " ".join(map(str, cmd)), flush=True)
        proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=self.out)
        (self.out / log).write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
        return proc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    BUILD.mkdir(exist_ok=True)
    c = Check(out)
    try:
        if "PYTORCH_ENABLE_MPS_FALLBACK" in os.environ:
            raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must stay unset")
        sources = {name: sha(LEAF / name) for name in SOURCES}
        git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        (out / "provenance.json").write_text(json.dumps(dict(
            leaf=str(LEAF.relative_to(ROOT)), git_head=git, mode="untimed every-case check",
            python=PY, sources=sources, fallback=None), indent=2))
        (out / "manifest.json").write_text(json.dumps(dict(status="running", planned=PLANNED, done=0), indent=2))

        for gate, source in SMOKES:
            exe = BUILD / ("check_" + Path(source).stem)
            built = c.run(["clang++", "-std=c++17", "-O2", LEAF / source, "-o", exe], gate + ".build.log")
            proc = c.run([exe], gate + ".log") if built.returncode == 0 else built
            c.emit(dict(gate=gate, build_exit=built.returncode, exit=proc.returncode,
                        stdout=proc.stdout.strip(), passed=built.returncode == 0 and proc.returncode == 0))

        exits = {}
        for name in BUILD_SCRIPTS:
            exits[name] = c.run(["zsh", LEAF / name], "build_" + Path(name).stem + ".log").returncode
        binaries = {p.name: sha(p) for p in sorted(BUILD.iterdir()) if p.is_file() and not p.name.startswith("check_")}
        (out / "binaries.json").write_text(json.dumps(binaries, indent=2))
        c.emit(dict(gate="build", exits=exits, passed=all(v == 0 for v in exits.values())))

        proc = c.run([PY, "-B", LEAF / "validate_homogeneous_runtime.py"], "train_step.log")
        # The runner prints its own JSON before the validator summary: read the summary, the last object.
        marker = '{\n  "kind": "hffn_metal_homogeneous_runtime_training_parity"'
        result = json.loads(proc.stdout[proc.stdout.rindex(marker):]) if proc.returncode == 0 else None
        maxima = {k: v["validator_max_abs"] for k, v in result["cases"].items()} if result else None
        c.emit(dict(gate="train_step", exit=proc.returncode, validator_max_abs=maxima, bound=2e-3,
                    passed=proc.returncode == 0 and maxima is not None and all(v <= 2e-3 for v in maxima.values())))

        for name, i, h, o, n, d in FORWARD_CASES:
            path = out / f"forward_{name}.json"
            proc = c.run([BUILD / "metal_hffn_training_forward_runner", "--input", i, "--hidden", h, "--output-size", o,
                          "--n", n, "--degree", d, "--output", path, "--untimed"], f"forward_{name}.log")
            data = json.loads(path.read_text()) if proc.returncode == 0 and path.exists() else {}
            errors = data.get("max_abs_error", {})
            c.emit(dict(gate="forward_relu", case=name, exit=proc.returncode, route=data.get("route"),
                        max_abs_error=errors, bound=1e-5,
                        passed=proc.returncode == 0 and data.get("route") == name and len(errors) == 3
                        and finite(errors.values()) and max(errors.values()) <= 1e-5))

        for name, i, h, o, n, d in BACKWARD_CASES:
            path = out / f"backward_{name}.json"
            proc = c.run([BUILD / "metal_hffn_backward_runner", "--input", i, "--hidden", h, "--output-size", o,
                          "--n", n, "--degree", d, "--warmup", 0, "--repeats", 1, "--output", path],
                         f"backward_{name}.log")
            data = json.loads(path.read_text()) if proc.returncode == 0 and path.exists() else {}
            errors = data.get("max_abs_error", {})
            c.emit(dict(gate="backward_runner", case=name, exit=proc.returncode, route=data.get("route"),
                        max_abs_error=errors,
                        passed=proc.returncode == 0 and data.get("route") == name and len(errors) == 6
                        and finite(errors.values())))

        for dispatches in (1, 4):
            path = out / f"adamw_d{dispatches}.json"
            proc = c.run([BUILD / "metal_hffn_adamw_runner", "--input", 128, "--hidden", 512, "--output-size", 128,
                          "--degree", 13, "--verify-steps", 5, "--warmup", 0, "--repeats", 1,
                          "--dispatches", dispatches, "--output", path], f"adamw_d{dispatches}.log")
            data = json.loads(path.read_text()) if proc.returncode == 0 and path.exists() else {}
            error = data.get("max_abs_error_vs_cpu")
            c.emit(dict(gate="adamw_runner", dispatches=dispatches, exit=proc.returncode, max_abs_error_vs_cpu=error,
                        passed=proc.returncode == 0 and finite([error])))

        proc = c.run([PY, "-B", LEAF / "validate_pytorch_mps_dense_iso_support_adamw.py"], "dense_va.log")
        data = json.loads(proc.stdout) if proc.returncode == 0 else {}
        errors = data.get("max_abs_error_cpu_vs_mps", {})
        c.emit(dict(gate="dense_va", exit=proc.returncode, max_abs_error_cpu_vs_mps=errors,
                    max_abs_inactive_slot=data.get("max_abs_inactive_slot"),
                    passed=proc.returncode == 0 and len(errors) == 3 and finite(errors.values())
                    and data.get("max_abs_inactive_slot") == 0.0))

        tj_dir = out / "dense_tj"
        proc = c.run([PY, "-B", LEAF / "validate_learning_trajectory_iso_support.py", "--results-dir", tj_dir],
                     "dense_tj.log")
        data = json.loads((tj_dir / "summary.json").read_text()) if proc.returncode == 0 else {}
        c.emit(dict(gate="dense_tj", exit=proc.returncode, epochs=data.get("epochs"),
                    functional_pass=data.get("functional_pass"), strict_numerical_pass=data.get("strict_numerical_pass"),
                    max_abs=data.get("max_abs"), max_abs_inactive=data.get("max_abs_inactive"),
                    post_objective=data.get("post_objective"),
                    passed=proc.returncode == 0 and data.get("functional_pass") is True
                    and data.get("max_abs_inactive") == 0.0))

        gate_dir = out / "export_inprocess"
        proc = c.run([PY, "-B", LEAF / "export_gate.py", "--exports", ",".join(EXPORT_GRID), "--faults", "--out", gate_dir],
                     "export_inprocess.log")
        records = json.loads((gate_dir / "records.json").read_text()) if (gate_dir / "records.json").exists() else []
        c.emit(dict(gate="export_inprocess", exit=proc.returncode, cells=len(records), planned_cells=EXPORT_CELLS,
                    clean_passed=sum(r["passed"] for r in records if r["fault"] is None),
                    faults_detected=sum(r["expected_outcome_met"] for r in records if r["fault"] is not None),
                    peak_rss_mb=max((r["peak_rss_mb"] for r in records), default=None),
                    passed=proc.returncode == 0 and len(records) == EXPORT_CELLS
                    and all(r["expected_outcome_met"] for r in records)))

        if {name: sha(LEAF / name) for name in SOURCES} != sources:
            raise RuntimeError("leaf sources changed during the check")
        status = "passed" if len(c.rows) == PLANNED else "failed"
        (out / "manifest.json").write_text(json.dumps(dict(status=status, planned=PLANNED, done=len(c.rows)), indent=2))
        if status != "passed":
            raise RuntimeError(f"{len(c.rows)} gates of {PLANNED}")
    except Exception:
        traceback.print_exc()
        (out / "manifest.json").write_text(json.dumps(dict(status="failed", planned=PLANNED, done=len(c.rows)),
                                                      indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
