"""E10 launcher and analysis: `python -m lib.e10 --step-dir e10 --mode check|smoke|run`.

Measures (lib/memory.py) every arm of `step.json` in two fresh processes, applies the frozen rule ((a) identical, (b) and (c) within 1 %)
and writes analysis.json / analysis.md with the readings of PLAN 15.4 (measured beside counted).
"""
import argparse
import concurrent.futures as cf
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

from . import campaign
from .count import count_arm

TRAIN = Path(__file__).resolve().parents[1]


def one(run_dir, cfg, rep):
    f = run_dir / "measure" / f"{cfg['key']}_p{rep}.json"
    f.parent.mkdir(exist_ok=True)
    p = subprocess.run([sys.executable, "-m", "lib.memory", "--arm", cfg["arm"], "--impl", cfg["impl"], "--out", str(f)], cwd=TRAIN,
                       capture_output=True, text=True)
    return cfg["key"], rep, p.returncode, p.stderr[-300:]


def analyse(run_dir, configs):
    rows, failures = {}, []
    for c in configs:
        a, b = (json.loads((run_dir / "measure" / f"{c['key']}_p{i}.json").read_text()) for i in (0, 1))
        rel = lambda u, v: abs(u - v) / max(u, v, 1)  # noqa: E731
        ok_a = a["saved"] == b["saved"]
        ok_b = rel(a["train_step"]["peak_live_bytes"], b["train_step"]["peak_live_bytes"]) <= 0.01
        ok_c = rel(a["prefill"]["peak_live_bytes"], b["prefill"]["peak_live_bytes"]) <= 0.01
        if not (ok_a and ok_b and ok_c):
            failures.append({"key": c["key"], "a_identical": ok_a, "b_within_1pct": ok_b, "c_within_1pct": ok_c})
        counted = count_arm(c["arm"])
        T = a["tokens_per_step"]
        rows[c["key"]] = {
            "arm": c["arm"], "impl": c["impl"], "tokens": T,
            "a_saved_total_per_token": a["saved"]["saved_total_bytes"] / T,
            "a_saved_token_side_per_token": a["saved"]["saved_token_side_bytes"] / T,
            "a_saved_weight_side_bytes": a["saved"]["saved_weight_side_bytes"],
            "b_train_peak_per_token": st.mean(x["train_step"]["peak_live_bytes"] for x in (a, b)) / T,
            "c_prefill_peak_per_token": st.mean(x["prefill"]["peak_live_bytes"] for x in (a, b)) / T,
            "d_peak_rss_bytes": [a["peak_rss_bytes"], b["peak_rss_bytes"]],
            "counted_training_activation_bytes_per_token": 4 * counted["training_activation_values_per_token"],
            "counted_inference_macs": counted["inference_macs_per_token"],
            "counted_params_outside_embedding": counted["params_outside_embedding"],
            "counted_adamw_state_values": counted["adamw_state_values_outside_embedding"],
            "counted_kv_values_per_token": counted["kv_cache_values_per_token"],
            "rule": {"a_identical": ok_a, "b_within_1pct": ok_b, "c_within_1pct": ok_c}}
    return rows, failures


READINGS = [  # (candidate key, control key, label)
    ("E4a_L2", "REF1", "equal counted MACs, 128"), ("E4a_L4", "REF2", "equal counted MACs, 128"),
    ("E4a_L6", "REF3", "equal counted MACs, 128"), ("E4a_L2_N512", "REF1_N512", "equal counted MACs, 512"),
    ("E4a_L4_N512", "REF2_N512", "equal counted MACs, 512"), ("E4a_L6_N512", "REF3_N512", "equal counted MACs, 512"),
    ("E4a_L2_W256", "REF1_W256", "equal counted MACs, width 256"), ("E4a_L4_W256", "REF2_W256", "equal counted MACs, width 256"),
    ("E4a_L4_W256_h16", "REF2_W256_h16", "equal counted MACs, width 256, 16 heads"),
    ("E4a_L2", "REF2", "equal depth, 128"), ("E4a_L4", "REF4", "equal depth, 128"),
    ("E4a_L2_N512", "REF2_N512", "equal depth, 512"), ("E4a_L4_N512", "REF4_N512", "equal depth, 512"),
    ("IND_L4", "E4a_L4", "IND against E4a, equal depth, 128"), ("IND_L4_N512", "E4a_L4_N512", "IND against E4a, equal depth, 512"),
    ("E4a_L4", "E4a_L4:selected", "packed (gather) against selected, 128"),
    ("E4a_L4_N512", "E4a_L4_N512:selected", "packed (gather) against selected, 512")]


def write_analysis(run_dir, configs):
    rows, failures = analyse(run_dir, configs)
    res = {"rows": rows, "rule_failures": failures, "readings": []}
    for cand, ctl, label in READINGS:
        if cand in rows and ctl in rows:
            c, k = rows[cand], rows[ctl]
            res["readings"].append({
                "label": label, "candidate": cand, "control": ctl,
                "measured_a_token_side": c["a_saved_token_side_per_token"] / k["a_saved_token_side_per_token"],
                "measured_b_train_peak": c["b_train_peak_per_token"] / k["b_train_peak_per_token"],
                "measured_c_prefill_peak": c["c_prefill_peak_per_token"] / k["c_prefill_peak_per_token"],
                "counted_training_activations": c["counted_training_activation_bytes_per_token"] / k["counted_training_activation_bytes_per_token"],
                "counted_macs": c["counted_inference_macs"] / k["counted_inference_macs"],
                "counted_params": c["counted_params_outside_embedding"] / k["counted_params_outside_embedding"],
                "counted_kv": c["counted_kv_values_per_token"] / k["counted_kv_values_per_token"]})
    (run_dir / "analysis.json").write_text(json.dumps(res, indent=1))
    L = ["# E10 analysis", "", "Per token (bytes); counted = the counter's training activations × 4 bytes.", "",
         "| config | (a) saved, token-side | (a) saved, weight-side (total) | (b) train peak | (c) prefill peak | counted activations |", "|---|---|---|---|---|---|"]
    for k, r in rows.items():
        L.append(f"| {k} | {r['a_saved_token_side_per_token']:.0f} | {r['a_saved_weight_side_bytes']:,} | {r['b_train_peak_per_token']:.0f} | "
                 f"{r['c_prefill_peak_per_token']:.0f} | {r['counted_training_activation_bytes_per_token']:.0f} |")
    L += ["", "| reading (candidate / control) | measured (a) token-side | (b) | (c) | counted activations | counted MACs | counted params | counted KV |",
          "|---|---|---|---|---|---|---|---|"]
    for r in res["readings"]:
        L.append(f"| {r['label']}: {r['candidate']} / {r['control']} | {r['measured_a_token_side']:.2f} | {r['measured_b_train_peak']:.2f} | "
                 f"{r['measured_c_prefill_peak']:.2f} | {r['counted_training_activations']:.2f} | {r['counted_macs']:.2f} | "
                 f"{r['counted_params']:.2f} | {r['counted_kv']:.2f} |")
    if failures:
        L += ["", "RULE FAILURES (harness): " + json.dumps(failures)]
    (run_dir / "analysis.md").write_text("\n".join(L) + "\n")
    return res


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
    manifest = {"step": "e10", "mode": a.mode, "started": campaign.utc(), "jobs": {}}
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
    configs = spec["smoke_configs"] if a.mode == "smoke" else spec["configs"]
    jobs = [(c, rep) for c in configs for rep in (0, 1)]
    log(f"{len(jobs)} measurement processes ({len(configs)} configurations x 2 fresh processes), {spec.get('workers', 2)} at a time")
    bad = 0
    with cf.ThreadPoolExecutor(max_workers=spec.get("workers", 2)) as ex:
        for key, rep, rc, err in ex.map(lambda j: one(run_dir, *j), jobs):
            log(f"{key} process {rep}: exit {rc}" + (f" {err}" if rc else ""))
            bad += rc != 0
    if bad:
        return finish(1, f"{bad} measurement processes failed")
    res = write_analysis(run_dir, configs)
    if res["rule_failures"]:
        return finish(1, "rule failures (harness): " + json.dumps(res["rule_failures"]))
    return finish(0)


if __name__ == "__main__":
    sys.exit(main())
