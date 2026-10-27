"""E14 launcher and per-run analysis: `python -m lib.e14 --step-dir e14 --mode check|smoke|prefill|decode|memory`.

* `prefill` / `decode` (one run each, its own folder, EXIT and manifest): the concurrency gate of 15.3, then the cells of `step.json`
  one process at a time (nothing else runs on the machine during a timing run: the session waits), each in a fresh process started
  with OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1, five processes per cell; then, from the raw slots only, per cell `r` (median and
  [min-max] over the processes), `delta` of the run (the largest |copy / anchor - 1| over all its cell-processes), the counted ratio.
* `memory`: the nine arms of the pairs, (a)-(c) of E10 and (e), two fresh processes per arm, the rule of E10.
* `smoke`: every cell in both modes (one process, 2 windows) and the memory run on two arms; sizes the runs (`sizing.json`).
The quality axis is read by `lib/analyze_e14.py` (after both timing runs); nothing is trained here.
"""
import argparse
import json
import os
import statistics as st
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from . import campaign, timing, timing4
from .arms import get_arm
from .count import count, count_arm

TRAIN = Path(__file__).resolve().parents[1]
ENV1 = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
G = 4


def counted_macs(cfg, mode):
    """Inference multiply-adds per token (estimated): prefill = the counter; decode = the same with N_k = N / 2 keys (the cache)."""
    return count(replace(cfg, context=cfg.context // 2) if mode == "decode" else cfg, 0)["inference_macs_per_token"]


def kv_read_values_per_decoded_token(cfg):
    """Keys and values read from the cache by one decoded token over all layers (counted): every head for the dense attention,
    the heads H_g of the token's group for the routed attention; N / 2 positions."""
    heads_values = cfg.d_model // cfg.groups if cfg.attn == "routed" else cfg.d_model
    return cfg.n_layers * 2 * heads_values * (cfg.context // 2)


def analyse_timing(run_dir, cells, procs, mode):
    per_cell, all_copy = {}, []
    for cell in cells:
        v_cfg, c_cfg = timing4.cell_arms(cell)
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
        spec = timing4.CELLS4[cell]
        counts_step = None
        cj = run_dir / "cells" / cell / "p0" / "cell.json"
        if cj.exists():
            counts_step = json.loads(cj.read_text()).get("group_counts_of_the_step")
        cm = counted_macs(v_cfg, mode) / counted_macs(c_cfg, mode)
        entry = {"variant": spec["variant"], "control": spec["control"], "group": spec["group"], "campaign": spec["campaign"],
                 "r_per_process": rv, "r_median": st.median(rv), "r_range": [min(rv), max(rv)], "copy_per_process": rc,
                 "steal_fraction": steal / total if total else None, "counted_ratio": cm, "group_counts_of_the_step": counts_step}
        if mode == "decode":
            entry["counted_kv_read_ratio"] = kv_read_values_per_decoded_token(v_cfg) / kv_read_values_per_decoded_token(c_cfg)
            entry["counted_kv_stored_ratio"] = count_arm(spec["variant"])["kv_cache_values_per_token"] / count_arm(spec["control"])["kv_cache_values_per_token"]
        per_cell[cell] = entry
    delta = max(all_copy)
    for cell, c in per_cell.items():
        r = c["r_median"]
        c["outcome"] = "gain (variant faster)" if 1 - r > delta else ("loss (variant slower)" if r > 1 + delta else "no difference within delta")
        c["time_over_count"] = r / c["counted_ratio"]
    return {"mode": mode, "delta": delta, "cells": per_cell, "processes_per_cell": procs, "dense_execution": "executor",
            "copy_ratios_all": all_copy}


def pair_products(an):
    """p1 x p2, p3 x p4, p5 x p6 (estimated: two cells combined), against the counted ratio of `E4a_L4` / `REF4`."""
    out = {}
    for a, b in (("p1", "p2"), ("p3", "p4"), ("p5", "p6")):
        if a in an["cells"] and b in an["cells"]:
            ca, cb = an["cells"][a], an["cells"][b]
            out[f"{a}x{b}"] = {"product_of_medians": ca["r_median"] * cb["r_median"],
                               "counted_product": ca["counted_ratio"] * cb["counted_ratio"], "class": "estimated"}
    return out


def write_timing_md(run_dir, an):
    L = [f"# E14 {an['mode']}: analysis (delta of the run = {an['delta']:.4f})", "",
         "| cell | variant / anchor | r median [min–max] | outcome | counted ratio | r / counted | steal fraction |",
         "|---|---|---|---|---|---|---|"]
    for c, v in an["cells"].items():
        L.append(f"| {c} | {v['variant']} / {v['control']} | {v['r_median']:.3f} [{v['r_range'][0]:.3f}–{v['r_range'][1]:.3f}] | "
                 f"{v['outcome']} | {v['counted_ratio']:.3f} | {v['time_over_count']:.2f} | {v['steal_fraction']:.4f} |")
    if an.get("pair_products"):
        L += ["", "Products of the pairs (estimated):"] + [f"- {k}: {v['product_of_medians']:.3f} (counted {v['counted_product']:.3f})"
                                                           for k, v in an["pair_products"].items()]
    (run_dir / "analysis.md").write_text("\n".join(L) + "\n")
    return L


def run_gate(run_dir, log):
    gate = subprocess.run([sys.executable, "-m", "lib.executor", "--gate"], cwd=TRAIN, env=ENV1, capture_output=True, text=True)
    g = json.loads([ln for ln in gate.stdout.splitlines() if ln.startswith("{")][-1])
    (run_dir / "gate.json").write_text(json.dumps(g, indent=1))
    log(f"concurrency gate at the start of the run: median ratio {g['median_ratio']:.3f} (threshold {g['threshold']}) -> "
        f"{'PASSED' if g['passed'] else 'FAILED'}")
    return g


def run_timing(run_dir, log, mode, cells, procs, windows, quick=False):
    """Returns None on success or the name of the failing cell-process."""
    import time
    durations = {}
    for cell in cells:
        for i in range(procs):
            out = run_dir / "cells" / cell / f"p{i}"
            cmd = [sys.executable, "-m", "lib.timing4", "--cell", cell, "--mode", mode, "--out", str(out), "--process", str(i),
                   "--windows", str(windows)] + (["--warm-calls", "10", "--warm-seconds", "1"] if quick else [])
            t0 = time.time()
            p = subprocess.run(cmd, cwd=TRAIN, env=ENV1, capture_output=True, text=True)
            durations[f"{cell}/p{i}"] = time.time() - t0
            (out.parent / f"p{i}.log").write_text(p.stdout + p.stderr)
            if p.returncode != 0:
                log(f"{mode} {cell} process {i} FAILED: {p.stderr[-400:]}")
                return f"{cell} process {i}", durations
            log(f"{mode} {cell} process {i}: {p.stdout.strip().splitlines()[-1]}")
    return None, durations


def run_memory(run_dir, log, arms, procs):
    """Returns (outs, failed, seconds per process)."""
    import time
    outs = {a: [] for a in arms}
    seconds = []
    for arm in arms:
        for i in range(procs):
            f = run_dir / "measure" / f"{arm}_p{i}.json"
            f.parent.mkdir(exist_ok=True)
            t0 = time.time()
            p = subprocess.run([sys.executable, "-m", "lib.memory4", "--arm", arm, "--out", str(f)], cwd=TRAIN, env=ENV1,
                               capture_output=True, text=True)
            seconds.append(time.time() - t0)
            if p.returncode != 0:
                log(f"memory {arm} process {i} FAILED: {p.stderr[-400:]}")
                return None, f"{arm} process {i}", seconds
            outs[arm].append(json.loads(f.read_text()))
            log(f"memory {arm} process {i}: {p.stdout.strip().splitlines()[-1]}")
    return outs, None, seconds


def rel(a, b):
    return abs(a - b) / max(a, b, 1)


def analyse_memory(outs):
    rows, failures = {}, []
    for arm, runs in outs.items():
        a, b = runs[0], runs[-1]
        rule = {"a_identical": a["saved"] == b["saved"],
                "b_within_1pct": rel(a["train_step"]["peak_live_bytes"], b["train_step"]["peak_live_bytes"]) <= 0.01,
                "c_within_1pct": rel(a["prefill"]["peak_live_bytes"], b["prefill"]["peak_live_bytes"]) <= 0.01,
                "e_within_1pct": rel(a["decode"]["peak_live_bytes"], b["decode"]["peak_live_bytes"]) <= 0.01}
        if not all(rule.values()):
            failures.append(arm)
        cfg = get_arm(arm)
        cnt = count_arm(arm)
        tokens = a["tokens_per_step"]
        rows[arm] = {"arm": arm, "impl": a["impl"], "tokens": tokens, "a_saved_total_per_token": a["saved"]["saved_total_bytes"] / tokens,
                     "a_saved_token_side_per_token": a["saved"]["saved_token_side_bytes"] / tokens,
                     "a_saved_weight_side_bytes": a["saved"]["saved_weight_side_bytes"],
                     "b_train_peak_per_token": a["train_step"]["peak_live_bytes"] / tokens,
                     "c_prefill_peak_per_token": a["prefill"]["peak_live_bytes"] / tokens,
                     "e_decode_peak_per_decoded_token": a["decode"]["peak_live_bytes"] / a["decode"]["batch"],
                     "e_group_counts": a["decode"]["group_counts"],
                     "d_peak_rss_bytes": [r["peak_rss_bytes"] for r in runs], "rule": rule,
                     "counted_inference_macs": cnt["inference_macs_per_token"], "counted_params_outside_embedding": cnt["params_outside_embedding"],
                     "counted_adamw_state_values": cnt["adamw_state_values_outside_embedding"],
                     "counted_kv_values_stored_per_token": cnt["kv_cache_values_per_token"],
                     "counted_kv_values_read_per_decoded_token": kv_read_values_per_decoded_token(cfg),
                     "counted_training_activation_bytes_per_token": 4 * cnt["training_activation_values_per_token"]}
    ratios = []
    for label, num, den in (("attention alone, 128", "E4a_L4", "E2h_L4"), ("attention alone, 512", "E4a_L4_N512", "E2h_L4_N512"),
                            ("attention alone, width 256", "E4a_L4_W256", "E2h_L4_W256"),
                            ("routing of the FFN alone, 128", "E2h_L4", "REF4"), ("routing of the FFN alone, 512", "E2h_L4_N512", "REF4_N512"),
                            ("routing of the FFN alone, width 256", "E2h_L4_W256", "REF4_W256")):
        if num in rows and den in rows:
            n, d = rows[num], rows[den]
            ratios.append({"label": label, "candidate": num, "control": den,
                           "measured": {k: n[k] / d[k] for k in ("a_saved_token_side_per_token", "b_train_peak_per_token",
                                                                  "c_prefill_peak_per_token", "e_decode_peak_per_decoded_token")},
                           "counted": {"inference_macs": n["counted_inference_macs"] / d["counted_inference_macs"],
                                       "params_and_adamw": n["counted_params_outside_embedding"] / d["counted_params_outside_embedding"],
                                       "kv_stored": n["counted_kv_values_stored_per_token"] / d["counted_kv_values_stored_per_token"],
                                       "kv_read_per_decoded_token": n["counted_kv_values_read_per_decoded_token"] / d["counted_kv_values_read_per_decoded_token"],
                                       "training_activations": n["counted_training_activation_bytes_per_token"] / d["counted_training_activation_bytes_per_token"]}})
    return {"rows": rows, "rule_failures": failures, "ratios": ratios}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-dir", required=True)
    ap.add_argument("--mode", required=True, choices=["check", "smoke", "prefill", "decode", "memory"])
    a = ap.parse_args()
    step_dir = Path(a.step_dir).resolve()
    spec = json.loads((step_dir / "step.json").read_text())
    run_dir = step_dir / "runs" / f"{a.mode}_{campaign.utc()}"
    run_dir.mkdir(parents=True)
    log = campaign.Log(run_dir / "run.log")
    manifest = {"step": spec["step"], "mode": a.mode, "started": campaign.utc(), "jobs": {}}
    campaign.write_manifest(run_dir, manifest)

    def finish(code, why=""):
        manifest.update({"ended": campaign.utc(), "exit": code, "why": why})
        campaign.write_manifest(run_dir, manifest)
        (run_dir / "EXIT").write_text(f"{code}\n")
        log(f"EXIT {code} {why}")
        return code

    if not campaign.run_checks(run_dir, log, tuple(spec["checks"])):
        return finish(3, "pre-run checks failed (harness)")
    if a.mode == "check":
        return finish(0)

    if a.mode in ("smoke", "prefill", "decode"):
        g = run_gate(run_dir, log)
        if not g["passed"]:
            return finish(0, "concurrency gate failed: no timing made (a property of the machine or the executor)")
    if a.mode == "smoke":
        sizing = {"cell_process_seconds": {}, "mode_seconds": {}}
        for mode in ("prefill", "decode"):
            sub = run_dir / mode
            sub.mkdir()
            failed, dur = run_timing(sub, log, mode, spec["cells"], 1, 2, quick=True)
            sizing["cell_process_seconds"][mode] = dur
            if failed:
                return finish(1, f"smoke {mode}: {failed} failed (harness)")
        outs, failed, _ = run_memory(run_dir, log, spec["smoke_memory_arms"], 2)
        if failed:
            return finish(1, f"smoke memory: {failed} failed (harness)")
        (run_dir / "memory_smoke.json").write_text(json.dumps(analyse_memory(outs), indent=1, default=str))
        (run_dir / "sizing.json").write_text(json.dumps(sizing, indent=1))
        return finish(0)
    if a.mode in ("prefill", "decode"):
        procs, windows = spec["processes"], spec["windows"]
        failed, _ = run_timing(run_dir, log, a.mode, spec["cells"], procs, windows)
        if failed:
            return finish(1, f"{a.mode}: {failed} failed (harness)")
        an = analyse_timing(run_dir, spec["cells"], procs, a.mode)
        an["pair_products"] = pair_products(an)
        (run_dir / "analysis.json").write_text(json.dumps(an, indent=1))
        log("\n".join(write_timing_md(run_dir, an)))
        return finish(0)
    # memory
    outs, failed, _ = run_memory(run_dir, log, spec["memory_arms"], spec["memory_processes"])
    if failed:
        return finish(1, f"memory: {failed} failed (harness)")
    res = analyse_memory(outs)
    (run_dir / "analysis.json").write_text(json.dumps(res, indent=1, default=str))
    if res["rule_failures"]:
        return finish(1, "the rule of E10 failed for " + ", ".join(res["rule_failures"]) + " (harness)")
    return finish(0)


if __name__ == "__main__":
    sys.exit(main())
