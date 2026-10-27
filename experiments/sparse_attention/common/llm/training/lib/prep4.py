"""Launcher of the preparation of E14: `python -m lib.prep4 --step-dir prep4 --mode check|smoke|run` (run = smoke).

In this order, one after the other and with nothing else running on the machine: the checks of the four suites; the concurrency gate of
15.3; the record of the SDPA kernel each arm runs in prefill and in decoding (operator names of one profiled call, and a descriptive
micro-measurement of the causal flag against an explicit mask on the shapes of the cells); the smoke of every cell in both modes (one
process, 2 windows, short warm-up; it proves nothing); the memory run on two arms (two fresh processes each); the sizing and the
re-estimate of the total from the measured durations. Writes the run folder with run.log, manifest.json, checks, gate.json,
sdpa.json, smoke records and sizing.json.
"""
import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

from . import campaign, e14, timing, timing4

TRAIN = Path(__file__).resolve().parents[1]
ENV1 = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
SDPA_ARMS = ["REF2", "E4a_L4", "IND_L4", "E2h_L4", "REF2_N512", "E4a_L4_N512", "E2h_L4_N512", "REF2_W256", "E4a_L4_W256", "E2h_L4_W256",
             "E4a_L4_W256_h16", "REF2_W256_h16"]


def sdpa_record():
    """In a process started with OMP_NUM_THREADS=1: the SDPA operators seen in one profiled prefill call and one profiled decoded step
    of each arm of SDPA_ARMS, and the time of the causal flag against an explicit boolean mask on the shapes of the cells."""
    import torch
    import torch.nn.functional as F
    from torch.profiler import ProfilerActivity, profile

    from . import decode as D
    from .arms import get_arm
    from .data import Enwik8, balanced_byte_hash
    from .executor import Executor
    from .model import SparseAttnLM
    torch.set_num_threads(1)
    torch.set_num_interop_threads(4)
    torch.set_grad_enabled(False)
    data = Enwik8()
    ht = balanced_byte_hash(data.train_counts(), 4)

    def ops(fn):
        fn()
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            fn()
        return sorted({e.key for e in prof.key_averages() if "scaled_dot" in e.key or "flash" in e.key or "attention" in e.key})

    out = {"operators": {}, "causal_flag_vs_explicit_mask_ms": {}}
    for name in SDPA_ARMS:
        cfg = get_arm(name)
        kind = D.kind_of(cfg)
        tokens, _ = data.eval_windows("valid", cfg.context, 8192 // cfg.context)
        ex = Executor(SparseAttnLM(cfg, 0, hash_table=ht).eval(), kind)
        windows = D.decode_windows(data, cfg.context, 64, 0)
        dec, _m, tok, p = D.build_decoder(cfg, ht, windows)
        out["operators"][name] = {"prefill": ops(lambda: ex.forward(tokens)), "decode": ops(lambda: dec.step(tok, p))}
    for N, dh, hpg, label in ((128, 16, 2, "N 128, 16 windows, 2 heads of 16"), (512, 16, 2, "N 512, 16 windows, 2 heads of 16"),
                              (512, 32, 2, "N 512, 16 windows, 2 heads of 32")):
        B = 8192 // N
        q = torch.randn(B, hpg, N, dh)
        k, v = torch.randn(B, hpg, N, dh), torch.randn(B, hpg, N, dh)
        mask = (torch.arange(N)[None, :] <= torch.arange(N)[:, None])[None, None]

        def med(f, reps=15):
            for _ in range(3):
                f()
            ts = []
            for _ in range(reps):
                t = time.perf_counter()
                f()
                ts.append(time.perf_counter() - t)
            return st.median(ts) * 1000
        out["causal_flag_vs_explicit_mask_ms"][label] = {
            "is_causal": med(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)),
            "explicit_mask": med(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask))}
    return out


def per_call(raw_path, arm):
    s = r = 0.0
    for line in open(raw_path):
        e = json.loads(line)
        if e.get("arm") == arm and "slot" in e:
            s += e["seconds"]
            r += e["reps"]
    return s / r


def estimate_process_seconds(sub, mode, cell, smoke_seconds):
    """Wall of one cell-process of the real run, from its smoke record (raw.jsonl): the build time is what the smoke process took
    beyond its timed windows and its warm-up; the real process adds the warm-up (>= 30 calls and 5 s per arm) and 12 windows of
    slots of at least 0.2 s (6 slots per window)."""
    d = sub / "cells" / cell / "p0"
    t = {a: per_call(d / "raw.jsonl", a) for a in ("control", "copy", "variant")}
    timed = sum(json.loads(ln)["seconds"] for ln in open(d / "raw.jsonl") if '"slot"' in ln)
    warm_smoke = sum(max(1.0, 10 * t[a]) for a in t)
    build = max(0.0, smoke_seconds - timed - warm_smoke)
    warm = sum(max(5.0, 30 * t[a]) for a in t)
    window = 2 * sum(max(0.2, t[a]) for a in t)
    return build + warm + 12 * window


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
    manifest = {"step": "prep4", "mode": a.mode, "started": campaign.utc(), "jobs": {}}
    campaign.write_manifest(run_dir, manifest)

    def finish(code, why=""):
        manifest.update({"ended": campaign.utc(), "exit": code, "why": why})
        campaign.write_manifest(run_dir, manifest)
        (run_dir / "EXIT").write_text(f"{code}\n")
        log(f"EXIT {code} {why}")
        return code

    if not campaign.run_checks(run_dir, log, tuple(spec["checks"])):
        return finish(3, "checks failed (harness)")
    if a.mode == "check":
        return finish(0)

    gate = e14.run_gate(run_dir, log)

    r = subprocess.run([sys.executable, "-c", "import json; from lib.prep4 import sdpa_record; print(json.dumps(sdpa_record()))"], cwd=TRAIN,
                       env=ENV1, capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    if r.returncode != 0 or not lines:
        log("sdpa record FAILED: " + r.stderr[-500:])
        return finish(1, "sdpa record failed (harness)")
    sdpa = json.loads(lines[-1])
    (run_dir / "sdpa.json").write_text(json.dumps(sdpa, indent=1))
    kernels = {k for v in sdpa["operators"].values() for m in v.values() for k in m}
    log(f"SDPA operators seen in the 12 arms (prefill and decoding): {sorted(kernels)}")
    log("causal flag vs explicit mask (ms, descriptive): " + json.dumps(sdpa["causal_flag_vs_explicit_mask_ms"]))

    sizing = {"cell_process_smoke_seconds": {}, "estimated_process_seconds": {}}
    for mode in ("prefill", "decode"):
        sub = run_dir / f"smoke_{mode}"
        sub.mkdir()
        failed, dur = e14.run_timing(sub, log, mode, spec["cells"], 1, 2, quick=True)
        if failed:
            return finish(1, f"smoke {mode}: {failed} failed (harness)")
        sizing["cell_process_smoke_seconds"][mode] = dur
        sizing["estimated_process_seconds"][mode] = {c: estimate_process_seconds(sub, mode, c, dur[f"{c}/p0"]) for c in spec["cells"]}
    outs, failed, m_proc = e14.run_memory(run_dir, log, spec["smoke_memory_arms"], 2)
    if failed:
        return finish(1, f"smoke memory: {failed} failed (harness)")
    (run_dir / "memory_smoke.json").write_text(json.dumps(e14.analyse_memory(outs), indent=1, default=str))

    est = {mode: sum(v for c, v in sizing["estimated_process_seconds"][mode].items()) * spec["processes"] / 3600 for mode in ("prefill", "decode")}
    est["memory"] = len(spec["memory_arms"]) * spec["memory_processes"] * st.mean(m_proc) / 3600
    est["checks_and_gates_per_run"] = 0.1
    est["total_h"] = est["prefill"] + est["decode"] + est["memory"] + 4 * est["checks_and_gates_per_run"]
    sizing["estimate_h"] = est
    sizing["memory_process_seconds_smoked_arms"] = m_proc
    (run_dir / "sizing.json").write_text(json.dumps(sizing, indent=1))
    log("re-estimate (hours, wall; nothing else running): " + json.dumps({k: round(v, 2) for k, v in est.items()}))
    return finish(0)


if __name__ == "__main__":
    sys.exit(main())
