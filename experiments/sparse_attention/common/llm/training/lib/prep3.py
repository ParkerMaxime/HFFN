"""Launcher of the preparation of the third series: `python -m lib.prep3 --step-dir prep3 --mode check|smoke|run` (run = smoke).

Runs, in this order and one after the other: the checks of the three series; the concurrency gate of the executor; the smoke of the
arms of E12-E13 (40 steps of WSD, hash arms in the packed form `gather`); the smoke of E10 on two arms (two fresh processes each);
the smoke of the cells of E11 (one process, 2 windows each, nothing else running), with the two dense executions timed, and the
re-estimate of the total. Writes the run folder with run.log, manifest.json, checks, gate.json, smoke records and sizing.json.
"""
import argparse
import json
import math
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

from . import campaign
from .arms import SERIES_W256, get_arm
from .count import count_arm

TRAIN = Path(__file__).resolve().parents[1]
ENV1 = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def per_call_seconds(raw_path, arm="control"):
    s = r = 0.0
    for line in open(raw_path):
        e = json.loads(line)
        if e.get("arm") == arm and "slot" in e:
            s += e["seconds"]
            r += e["reps"]
    return s / r


def lpt(jobs, workers=4):
    """Wall time of the jobs (seconds) on `workers` workers, longest first."""
    load = [0.0] * workers
    for j in sorted(jobs, reverse=True):
        i = load.index(min(load))
        load[i] += j
    return max(load)


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
    manifest = {"step": "prep3", "mode": a.mode, "started": campaign.utc(), "jobs": {}}
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

    # concurrency gate (a property of the executor on this machine, recorded; a failure means E11 is not run)
    r = subprocess.run([sys.executable, "-m", "lib.executor", "--gate"], cwd=TRAIN, env=ENV1, capture_output=True, text=True)
    gate = json.loads([ln for ln in r.stdout.splitlines() if ln.startswith("{")][-1])
    (run_dir / "gate.json").write_text(json.dumps(gate, indent=1))
    log(f"concurrency gate: median ratio {gate['median_ratio']:.3f} (threshold {gate['threshold']}) -> {'PASSED' if gate['passed'] else 'FAILED'}")

    # smoke of the arms of E12-E13 (and of the hash arms of E8/E9 in the packed form, for sizing E10 and the hash arms in general)
    arms = [x.name for x in SERIES_W256]
    jobs = [{"arm": n, "seed": 0, "steps": 40, "lr": 2e-3, "dir": run_dir / "smoke" / f"{n}_s0", "eval_every": 1000,
             "valid_bytes": 131072, "final_windows": 256, "schedule": "wsd", "tokens_per_step": 8192, "mixing": False,
             "impl": "gather"} for n in arms]
    rcs = [rc for _, rc in campaign.run_jobs(jobs, run_dir, 4, 1, manifest, log, False)]
    if any(rc != 0 for rc in rcs):
        return finish(1, "smoke of the arms failed")
    sizing = {"per_step_s": {}, "eval_s_per_window": {}}
    for n in arms:
        s = json.loads((run_dir / "smoke" / f"{n}_s0" / "summary.json").read_text())
        sizing["per_step_s"][n] = s["train_wall_s"] / 40
        sizing["eval_s_per_window"][n] = s["final_eval_wall_s"] / s["final_eval_windows"]

    # smoke of E10: two arms, two fresh processes each
    for arm, impl in (("E4a_L4", "gather"), ("REF2", "packed")):
        outs = []
        for i in range(2):
            f = run_dir / "smoke_e10" / f"{arm}_{impl}_{i}.json"
            f.parent.mkdir(exist_ok=True)
            subprocess.run([sys.executable, "-m", "lib.memory", "--arm", arm, "--impl", impl, "--out", str(f)], cwd=TRAIN,
                           check=True, capture_output=True)
            outs.append(json.loads(f.read_text()))
        same = outs[0]["saved"] == outs[1]["saved"]
        log(f"E10 smoke {arm} [{impl}]: saved {outs[0]['saved']['saved_total_bytes']} B identical in two processes: {same}")

    # smoke of E11: every cell, one process, 2 windows; the dense control timed under both executions
    if gate["passed"]:
        from . import timing
        dense_cells = [c for c in timing.CELLS if timing.CELLS[c]["control"] in ("dense", "REF2", "REF2_N512")]
        per_call = {}
        for cell in timing.CELLS:
            for dexec in (("executor", "intraop") if cell in dense_cells else ("executor",)):
                out = run_dir / "smoke_e11" / f"{cell}_{dexec}"
                cmd = [sys.executable, "-m", "lib.timing", "--cell", cell, "--dense-exec", dexec, "--out", str(out), "--windows", "2",
                       "--warm-calls", "10", "--warm-seconds", "1"] + (["--phases"] if dexec == "executor" else [])
                p = subprocess.run(cmd, cwd=TRAIN, env=ENV1, capture_output=True, text=True)
                if p.returncode != 0:
                    log(f"E11 smoke {cell} [{dexec}] FAILED: {p.stderr[-400:]}")
                    return finish(1, "E11 smoke failed")
                per_call[f"{cell}/{dexec}"] = per_call_seconds(out / "raw.jsonl")
                log(f"E11 smoke {cell} [{dexec}]: {p.stdout.strip().splitlines()[-1]}")
        # rule (written in the README before this run): the dense execution with the lower geometric mean, over the six dense cells,
        # of the per-call time of the control becomes the control of E11
        gm = {d: math.exp(st.mean(math.log(per_call[f"{c}/{d}"]) for c in dense_cells)) for d in ("executor", "intraop")}
        choice = min(gm, key=gm.get)
        sizing["dense_execution"] = {"geometric_mean_s": gm, "per_call_s": per_call, "choice": choice, "dense_cells": dense_cells}
        log(f"dense control of E11: geometric mean {gm} -> {choice}")

    # re-estimate of the total (wall time; 4 jobs in parallel, 1 thread each; estimated)
    steps, ev = 4000, 9765
    def run_s(n):
        return sizing["per_step_s"][n] * steps + sizing["eval_s_per_window"][n] * ev * 2
    e13 = [run_s(n) for n in arms for _ in range(3)]
    e12_arms = ["REF2_W256", "E4a_L4_W256"]
    e12 = [sizing["per_step_s"][n] * steps + sizing["eval_s_per_window"][n] * ev for n in e12_arms for _ in range(4)]  # 3 rates + 1 extension
    est = {"E13_h": lpt(e13) / 3600, "E12_h": lpt(e12) / 3600, "E10_h": 0.6, "E11_h": 1.0}
    est["total_h"] = sum(est.values())
    sizing["estimate"] = est
    (run_dir / "sizing.json").write_text(json.dumps(sizing, indent=1))
    log("re-estimate (hours): " + json.dumps({k: round(v, 2) for k, v in est.items()}))
    return finish(0)


if __name__ == "__main__":
    sys.exit(main())
