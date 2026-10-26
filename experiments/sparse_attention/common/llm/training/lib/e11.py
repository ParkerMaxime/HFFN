"""E11 launcher and analysis: `python -m lib.e11 --step-dir e11 --mode check|smoke|run`.

Runs the cells of `step.json` one process at a time (nothing else runs on the machine: the session waits), each in a fresh process
started with OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1, five processes per cell, then computes from the raw slots: per cell `r` (median
and [min-max] over the five processes), `delta` of the run, the outcome, the counted ratio of multiply-adds, steal time and phases.
"""
import argparse
import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

from . import campaign, timing
from .arms import get_arm
from .count import count_arm, layer_macs

TRAIN = Path(__file__).resolve().parents[1]
ENV1 = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def counted_ratio(cell):
    spec = timing.CELLS[cell]
    if spec["level"] == "layer":
        dense, e4a = timing.layer_cfgs(spec["M"], spec["N"])
        return sum(layer_macs(e4a).values()) / sum(layer_macs(dense).values())
    return count_arm(spec["variant"])["inference_macs_per_token"] / count_arm(spec["control"])["inference_macs_per_token"]


def analyse(run_dir, cells, procs):
    per_cell, all_copy = {}, []
    for cell in cells:
        rv, rc, steal, total = [], [], 0, 0
        for i in range(procs):
            d = run_dir / "cells" / cell / f"p{i}"
            res = timing.ratios_from_raw(d / "raw.jsonl")
            rv.append(res["r_variant"])
            rc.append(res["r_copy"])
            all_copy.append(abs(res["r_copy"] - 1.0))
            for w in res["windows"]:
                steal += w["steal_jiffies"] or 0
                total += w["total_jiffies"] or 0
        per_cell[cell] = {"r_per_process": rv, "r_median": st.median(rv), "r_range": [min(rv), max(rv)], "copy_per_process": rc,
                          "steal_fraction": steal / total if total else None, "counted_ratio": counted_ratio(cell)}
    delta = max(all_copy)
    for cell, c in per_cell.items():
        r = c["r_median"]
        c["outcome"] = "faster" if 1 - r > delta else ("slower" if r - 1 > delta else "not resolved")
        c["time_over_count"] = r / c["counted_ratio"]
        pf = run_dir / "cells" / cell / "p0" / "phases.json"
        c["phases_ms_descriptive"] = json.loads(pf.read_text()) if pf.exists() else None
    return {"delta": delta, "cells": per_cell, "processes_per_cell": procs, "dense_execution": "executor"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-dir", required=True)
    ap.add_argument("--mode", required=True, choices=["check", "smoke", "run"])
    a = ap.parse_args()
    step_dir = Path(a.step_dir).resolve()
    spec = json.loads((step_dir / "step.json").read_text())
    run_dir = step_dir / "runs" / f"{a.mode}_{campaign.utc()}"
    run_dir.mkdir(parents=True)
    log = campaign.Log(run_dir / "run.log")
    manifest = {"step": "e11", "mode": a.mode, "started": campaign.utc(), "jobs": {}}
    campaign.write_manifest(run_dir, manifest)

    def finish(code, why=""):
        manifest.update({"ended": campaign.utc(), "exit": code, "why": why})
        campaign.write_manifest(run_dir, manifest)
        (run_dir / "EXIT").write_text(f"{code}\n")
        log(f"EXIT {code} {why}")
        return code

    if not campaign.run_checks(run_dir, log, tuple(spec["checks"])):
        return finish(3, "pre-training checks failed (harness)")
    if a.mode == "check":
        return finish(0)
    gate = subprocess.run([sys.executable, "-m", "lib.executor", "--gate"], cwd=TRAIN, env=ENV1, capture_output=True, text=True)
    g = json.loads([ln for ln in gate.stdout.splitlines() if ln.startswith("{")][-1])
    (run_dir / "gate.json").write_text(json.dumps(g, indent=1))
    log(f"concurrency gate at the start of the run: median ratio {g['median_ratio']:.3f} -> {'PASSED' if g['passed'] else 'FAILED'}")
    if not g["passed"]:
        return finish(0, "concurrency gate failed: E11 not run (a property of the machine or the executor)")
    cells = spec["smoke_cells"] if a.mode == "smoke" else spec["cells"]
    procs = 1 if a.mode == "smoke" else spec["processes"]
    windows = 2 if a.mode == "smoke" else spec["windows"]
    for cell in cells:
        for i in range(procs):
            out = run_dir / "cells" / cell / f"p{i}"
            cmd = [sys.executable, "-m", "lib.timing", "--cell", cell, "--dense-exec", spec["dense_execution"], "--out", str(out),
                   "--process", str(i), "--windows", str(windows)] + (["--phases"] if i == 0 else [])
            p = subprocess.run(cmd, cwd=TRAIN, env=ENV1, capture_output=True, text=True)
            (out.parent / f"p{i}.log").write_text(p.stdout + p.stderr)
            if p.returncode != 0:
                log(f"{cell} process {i} FAILED: {p.stderr[-400:]}")
                return finish(1, f"cell {cell} process {i} failed (harness)")
            log(f"{cell} process {i}: {p.stdout.strip().splitlines()[-1]}")
    res = analyse(run_dir, cells, procs)
    (run_dir / "analysis.json").write_text(json.dumps(res, indent=1))
    L = [f"# E11 analysis (delta of the run = {res['delta']:.4f})", "", "| cell | r median [min–max] | outcome | counted ratio | r / counted | steal fraction |", "|---|---|---|---|---|---|"]
    for c, v in res["cells"].items():
        L.append(f"| {c} | {v['r_median']:.3f} [{v['r_range'][0]:.3f}–{v['r_range'][1]:.3f}] | {v['outcome']} | {v['counted_ratio']:.3f} | "
                 f"{v['time_over_count']:.2f} | {v['steal_fraction']:.4f} |")
    (run_dir / "analysis.md").write_text("\n".join(L) + "\n")
    log("\n".join(L))
    return finish(0)


if __name__ == "__main__":
    sys.exit(main())
