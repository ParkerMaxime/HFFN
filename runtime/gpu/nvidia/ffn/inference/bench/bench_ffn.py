"""Isolated-FFN timing of the Nvidia FFN leaf, HFFN against dense iso-support, a copy on both arms.

Arms in one process per session: `dense`, `dense_copy`, `hffn`, `hffn_copy`. Quantity: host wall time of a window of `repeats`
calls of one arm on one device input, between two `torch.cuda.synchronize()`. Blocks of eight windows with equal time centroid.

    python -B bench_ffn.py --mode smoke|run --out <run folder>
    python -B bench_ffn.py --worker '<job json>'      (one session: all cells in one process, one JSON line per cell)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BENCH = Path(__file__).resolve().parent
REPO = next(p for p in BENCH.parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
FFN_LEAF = BENCH.parent
sys.path.insert(0, str(FFN_LEAF))

import numpy  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import hffn_ffn_cuda  # noqa: E402  (puts the shared kernels and test layers on sys.path)
from hffn_cuda.runtime import HFFNRuntime  # noqa: E402
from hffn_ffn_layers import dense, layers, reference  # noqa: E402

CELLS = {   # mode: ((layer, N), ...)
    "run": (("homog_512_2048", 1), ("homog_512_2048", 4), ("homog_512_2048", 64), ("homog_512_2048", 512),
            ("irreg_512_2048", 4), ("irreg_512_2048", 512)),
    "smoke": (("homog_512_2048", 1), ("homog_512_2048", 64)),
}
GRID = {"run": dict(sessions=3, window_ms=50.0), "smoke": dict(sessions=1, window_ms=20.0)}
BLOCK_ORDERS = (("dense", "hffn", "hffn_copy", "dense_copy", "dense_copy", "hffn_copy", "hffn", "dense"),
                ("dense_copy", "hffn_copy", "hffn", "dense", "dense", "hffn", "hffn_copy", "dense_copy"))
TOLERANCE = 2e-5          # leaf tolerance on max |y - y_float64|
AA_TOLERANCE = 0.03       # every cell-session of each pair must be within it (README criterion)
MAX_REPEATS = 100_000


def gpu_state() -> dict:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,temperature.gpu,clocks_throttle_reasons.active",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().split(", ")
    return dict(zip(("name", "sm_mhz", "temp_c", "throttle"), out))


def sync_time(fn, repeats: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - start) / 1e6   # ms per window


def build_arms(layer, x):
    device = torch.device("cuda")

    def dense_arm():
        w_up, w_down = (torch.from_numpy(w).to(device) for w in dense.dense_weights(layer))
        b_up, b_down = (torch.from_numpy(b.copy()).to(device) for b in (layer.b_up, layer.b_down))
        return lambda: F.linear(F.gelu(F.linear(x, w_up, b_up), approximate="none"), w_down, b_down)

    def hffn_arm():
        runtime = HFFNRuntime([hffn_ffn_cuda.cuda_layer(layer)], device)
        return lambda: runtime.forward(0, x), runtime

    hffn, runtime = hffn_arm()
    hffn_copy, runtime_copy = hffn_arm()
    return {"dense": dense_arm(), "dense_copy": dense_arm(), "hffn": hffn, "hffn_copy": hffn_copy}, (runtime, runtime_copy)


@torch.inference_mode()
def worker(job: dict) -> int:
    mode, window_ms = job["mode"], job["window_ms"]
    print(json.dumps({"event": "session", "session": job["session"], "gpu_before": gpu_state()}), flush=True)
    for name, n in CELLS[mode]:
        layer, case = layers.load(name)                        # checks the pinned fingerprint (gate 1)
        x = torch.from_numpy(layers.make_input(case, n)).to("cuda")
        expected = torch.from_numpy(reference.forward(layer, layers.make_input(case, n))).to("cuda")
        arms, (runtime, runtime_copy) = build_arms(layer, x)
        outputs = {arm: fn().clone() for arm, fn in arms.items()}
        errors = {arm: float((y.double() - expected).abs().max()) for arm, y in outputs.items()}
        bitwise = bool(torch.equal(outputs["hffn"], outputs["hffn_copy"]))
        gates = {"oracle": all(errors[a] <= TOLERANCE for a in errors), "copy_bitwise": bitwise}
        for _ in range(3):                                      # warm-up of every arm (compiles, reserves buffers)
            for fn in arms.values():
                fn()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        once = min(sync_time(arms["dense"], 1) for _ in range(3))
        repeats = min(MAX_REPEATS, max(1, math.ceil(window_ms / once)))
        samples = {arm: [] for arm in arms}
        for block, order in enumerate(BLOCK_ORDERS):
            for arm in order:
                sync_time(arms[arm], repeats)                   # untimed same-arm lead-in
                samples[arm].append(sync_time(arms[arm], repeats) / repeats * 1000.0)   # us per call
        gates["memory_steady"] = torch.cuda.memory_allocated() == allocated
        print(json.dumps({"event": "cell", "session": job["session"], "layer": name, "n": n, "repeats": repeats,
                          "once_ms": once, "samples_us": samples, "errors": errors, "gates": gates,
                          "model_bytes": runtime.layers[0].model_bytes(), "workspace_bytes": runtime.workspace_bytes(),
                          "dense_weight_bytes": 4 * (layer.d_ff * layer.d_model * 2)}), flush=True)
        del arms, runtime, runtime_copy, outputs
        torch.cuda.empty_cache()
    print(json.dumps({"event": "session_end", "session": job["session"], "gpu_after": gpu_state()}), flush=True)
    return 0


def ratio(samples: dict, a: str, b: str) -> float:
    return statistics.median(samples[a]) / statistics.median(samples[b])


def summarise(rows: list[dict]) -> dict:
    cells = [r for r in rows if r["event"] == "cell"]
    aa = {"hffn": [], "dense": []}
    per_cell: dict = {}
    for r in cells:
        s = r["samples_us"]
        aa["hffn"].append(ratio(s, "hffn", "hffn_copy"))
        aa["dense"].append(ratio(s, "dense", "dense_copy"))
        per_cell.setdefault((r["layer"], r["n"]), []).append(
            {"hffn_over_dense": ratio(s, "hffn", "dense"), "hffn_copy_over_dense_copy": ratio(s, "hffn_copy", "dense_copy"),
             "hffn_us": statistics.median(s["hffn"]), "dense_us": statistics.median(s["dense"])})
    planned = len(cells)
    verdict = {}
    for pair, values in aa.items():
        within = sum(abs(v - 1) <= AA_TOLERANCE for v in values)
        worst = max((abs(v - 1) for v in values), default=1.0)
        verdict[pair] = {"within_0.03": within, "of": len(values), "max_abs_dev": worst}
    admissible = planned > 0 and all(v["within_0.03"] == v["of"] for v in verdict.values())
    summary = {"aa": verdict, "admissible": admissible, "cells": []}
    for (layer, n), entries in per_cell.items():
        ratios = [e["hffn_over_dense"] for e in entries]
        row = {"layer": layer, "n": n, "sessions": len(entries)}
        if admissible:
            row["hffn_over_dense"] = {"median": statistics.median(ratios), "min": min(ratios), "max": max(ratios)}
            row["hffn_us_median"] = statistics.median(e["hffn_us"] for e in entries)
            row["dense_us_median"] = statistics.median(e["dense_us"] for e in entries)
        summary["cells"].append(row)
    return summary


def digest(paths) -> dict:
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def orchestrate(mode: str, out: Path) -> int:
    grid = GRID[mode]
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    planned = grid["sessions"] * len(CELLS[mode])
    (out / "config.json").write_text(json.dumps({"mode": mode, "grid": grid, "cells": CELLS[mode], "block_orders": BLOCK_ORDERS}, indent=1))
    import cupy
    kernel_files = sorted((REPO / "runtime/gpu/nvidia/common/kernels/hffn_cuda").glob("*.py"))
    (out / "provenance.json").write_text(json.dumps({
        "gpu": gpu_state(), "torch": torch.__version__, "cuda": torch.version.cuda, "cupy": cupy.__version__, "numpy": numpy.__version__,
        "files": digest([Path(__file__).resolve(), *kernel_files, *sorted((FFN_LEAF / "hffn_ffn_cuda").glob("*.py"))])}, indent=1))
    rows, status = [], "passed"
    with open(out / "raw.jsonl", "w") as raw:
        for session in range(grid["sessions"]):
            job = {"mode": mode, "session": session, "window_ms": grid["window_ms"]}
            proc = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", json.dumps(job)],
                                    stdout=subprocess.PIPE, text=True)
            for line in proc.stdout:
                raw.write(line)
                raw.flush()
                rows.append(json.loads(line))
                print(line.rstrip(), flush=True)
            if proc.wait() != 0:
                status = "failed"
                break
    cells = [r for r in rows if r["event"] == "cell"]
    if any(not all(r["gates"].values()) for r in cells):
        status = "failed"
    done = len(cells)
    if status == "passed" and done != planned:
        status = "partial"
    summary = summarise(rows) if status == "passed" else {"note": "no timing number for a failed or partial run"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "manifest.json").write_text(json.dumps({"status": status, "planned": planned, "done": done, "started": started,
                                                  "ended": datetime.now(timezone.utc).isoformat()}, indent=1))
    print(f"status {status}: {done} of {planned} cell-sessions; summary {json.dumps(summary)[:400]}", flush=True)
    return 0 if status == "passed" else (3 if status == "partial" else 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "run"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        return worker(json.loads(args.worker))
    return orchestrate(args.mode, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
