"""Readings of E13 (PLAN 15.4) from a run folder: `python -m lib.analyze_w256 <run dir>` -> analysis.json, analysis.md.

Reads the summaries of the `final` stage and `step.json` (`analysis.tau`, `analysis.memory`: the `analysis.json` of the
E10 run, relative to the step folder), the counter (`lib/count.py`), and writes new files only. Every pair is read per
seed, `d_s = bpb(candidate, s) - bpb(control, s)` (positive = candidate worse), with the outcomes of PLAN 13.1 at `tau`
and *dominates* where the candidate costs less counted inference compute than the control (as `lib/analyze_frontier.py`).
The heads-per-group reading is `i_s = d_s(h16 pair) - d_s(h8 pair)`: *favours four heads per group* if `max_s i_s < 0`,
*favours two* if `min_s i_s > 0`, otherwise not resolved, with `|i_s|` against `tau` beside it.
"""
import json
import statistics as st
import sys
from pathlib import Path

from .analyze_frontier import pair
from .count import count_arm

PAIRS_EQUAL_COMPUTE = [("E4a_L2_W256", "REF1_W256"), ("E4a_L4_W256", "REF2_W256")]
PAIRS_EQUAL_DEPTH = [("E4a_L2_W256", "REF2_W256")]
PAIRS_FOUR_HEADS = [("E4a_L4_W256_h16", "REF2_W256_h16")]
PAIR_H8, PAIR_H16 = ("E4a_L4_W256", "REF2_W256"), ("E4a_L4_W256_h16", "REF2_W256_h16")


def heads_per_group(bpb, tau):
    seeds = sorted(set(bpb[PAIR_H8[0]]) & set(bpb[PAIR_H8[1]]) & set(bpb[PAIR_H16[0]]) & set(bpb[PAIR_H16[1]]))
    d8 = {s: bpb[PAIR_H8[0]][s] - bpb[PAIR_H8[1]][s] for s in seeds}
    d16 = {s: bpb[PAIR_H16[0]][s] - bpb[PAIR_H16[1]][s] for s in seeds}
    i = {s: d16[s] - d8[s] for s in seeds}
    v = list(i.values())
    out = "favours four heads per group" if max(v) < 0 else ("favours two heads per group" if min(v) > 0 else "not resolved")
    return {"d_h8_per_seed": d8, "d_h16_per_seed": d16, "i_per_seed": i, "median": st.median(v), "range": [min(v), max(v)],
            "outcome": out, "abs_i_over_tau": {s: abs(x) / tau for s, x in i.items()}, "abs_i_exceeds_tau": {s: abs(x) > tau for s, x in i.items()}}


def main(run_dir):
    run_dir = Path(run_dir).resolve()
    step_dir = run_dir.parents[1]
    spec = json.loads((step_dir / "step.json").read_text())
    tau = spec["analysis"].get("tau", 0.075)
    valid, test, rng = {}, {}, {}
    for d in sorted((run_dir / "final").glob("*_s*")):
        s = json.loads((d / "summary.json").read_text())
        a = s["arm"]
        valid.setdefault(a, {})[s["seed"]] = s["final_valid_bpb"]
        if "final_test" in s:
            test.setdefault(a, {})[s["seed"]] = s["final_test"]["bpb"]
        fv = s["final_valid"]
        if fv.get("bpb_by_range"):
            rng.setdefault(a, {})[s["seed"]] = (fv["range_names"], fv["bpb_by_range"])
    counts = {a: count_arm(a) for a in valid}
    cost = {a: counts[a]["inference_macs_per_token"] for a in counts}
    memory = {}
    mem_path = spec["analysis"].get("memory")
    if mem_path and (step_dir / mem_path).exists():
        memory = json.loads((step_dir / mem_path).read_text())["rows"]
    res = {"tau": tau, "memory_source": mem_path, "arms": {}, "readings": {}}
    for a in valid:
        v = list(valid[a].values())
        e = {"valid_median": st.median(v), "valid_range": [min(v), max(v)], "valid_per_seed": valid[a],
             "counter": {k: counts[a][k] for k in ("inference_macs_per_token", "training_macs_per_token", "params_outside_embedding",
                                                   "adamw_state_values_outside_embedding", "kv_cache_values_per_token",
                                                   "training_activation_values_per_token")}}
        if a in test:
            t = list(test[a].values())
            e["test_median"], e["test_range"], e["test_per_seed"] = st.median(t), [min(t), max(t)], test[a]
        if a in rng:
            names = next(iter(rng[a].values()))[0]
            e["range_names"] = names
            e["bpb_by_range_median"] = [st.median(r[1][i] for r in rng[a].values()) for i in range(len(names))]
        if a in memory:
            m = memory[a]
            e["memory_e10"] = {k: m[k] for k in ("a_saved_token_side_per_token", "b_train_peak_per_token", "c_prefill_peak_per_token")}
        res["arms"][a] = e
    R = res["readings"]
    R["equal_compute_pairs"] = [pair(valid, c, d, tau, cost) for c, d in PAIRS_EQUAL_COMPUTE if c in valid and d in valid]
    R["equal_depth"] = [pair(valid, c, d, tau, cost) for c, d in PAIRS_EQUAL_DEPTH if c in valid and d in valid]
    R["equal_compute_four_heads"] = [pair(valid, c, d, tau, cost) for c, d in PAIRS_FOUR_HEADS if c in valid and d in valid]
    if all(x in valid for x in PAIR_H8 + PAIR_H16):
        R["heads_per_group"] = heads_per_group(valid, tau)
    if test:
        R["test_equal_compute_pairs"] = [pair(test, c, d, tau, cost) for c, d in PAIRS_EQUAL_COMPUTE + PAIRS_FOUR_HEADS
                                         if c in test and d in test]
    (run_dir / "analysis.json").write_text(json.dumps(res, indent=1, default=str))

    L = [f"# Width frontier analysis ({run_dir.name}, tau {tau})", "",
         "| arm | valid median [min–max] | test median [min–max] | inference MACs | training MACs | params | KV values/token | train. act. values/token |",
         "|---|---|---|---|---|---|---|---|"]
    for a in sorted(res["arms"], key=lambda a: (cost[a], a)):
        e = res["arms"][a]
        t = f"{e['test_median']:.4f} [{e['test_range'][0]:.4f}–{e['test_range'][1]:.4f}]" if "test_median" in e else "—"
        c = e["counter"]
        L.append(f"| {a} | {e['valid_median']:.4f} [{e['valid_range'][0]:.4f}–{e['valid_range'][1]:.4f}] | {t} | "
                 f"{c['inference_macs_per_token']:,} | {c['training_macs_per_token']:,} | {c['params_outside_embedding']:,} | "
                 f"{c['kv_cache_values_per_token']:,} | {c['training_activation_values_per_token']:,} |")

    def line(r):
        s = ", ".join(f"s{k} {v:+.4f}" for k, v in r["d_per_seed"].items())
        saved = f" (counted MACs saved {r['compute_saved']:.1%})" if r.get("compute_saved") else ""
        return (f"- {r['candidate']} vs {r['control']}{saved}: d = {s}; median {r['median']:+.4f} [{r['range'][0]:+.4f}, {r['range'][1]:+.4f}] → "
                f"**{r['outcome']}**" + ("; **dominates**" if r.get("dominates") else ""))
    for title, key in (("Equal counted multiply-adds, width (measured)", "equal_compute_pairs"),
                       ("Equal depth, width (measured)", "equal_depth"),
                       ("Equal counted multiply-adds, four heads per group (measured)", "equal_compute_four_heads"),
                       ("Test split, same pairs (beside, selects nothing)", "test_equal_compute_pairs")):
        if key in R:
            L += ["", f"## {title}", ""] + [line(r) for r in R[key]]
    if "heads_per_group" in R:
        h = R["heads_per_group"]
        L += ["", "## Heads per group (measured)", "",
              "i_s = d_s(E4a_L4_W256_h16 − REF2_W256_h16) − d_s(E4a_L4_W256 − REF2_W256)", ""]
        for s in h["i_per_seed"]:
            L.append(f"- seed {s}: d(h8 pair) {h['d_h8_per_seed'][s]:+.4f}, d(h16 pair) {h['d_h16_per_seed'][s]:+.4f}, "
                     f"i = {h['i_per_seed'][s]:+.4f}, |i|/tau = {h['abs_i_over_tau'][s]:.2f}")
        L.append(f"- median {h['median']:+.4f} [{h['range'][0]:+.4f}, {h['range'][1]:+.4f}] → **{h['outcome']}**")
    L += ["", "## Bits per byte by position range (median over seeds)", ""]
    for a in sorted(res["arms"], key=lambda a: (cost[a], a)):
        e = res["arms"][a]
        if "bpb_by_range_median" in e:
            L.append(f"- {a}: " + ", ".join(f"{n}: {v:.4f}" for n, v in zip(e["range_names"], e["bpb_by_range_median"])))
    if memory:
        L += ["", "## Measured memory of E10 for the same arms (bytes per token, fp32, one step of 8,192 bytes)", "",
              "| arm | saved for backward, token side | peak of a training step | peak of a prefill |", "|---|---|---|---|"]
        for a in sorted(res["arms"], key=lambda a: (cost[a], a)):
            if "memory_e10" in res["arms"][a]:
                m = res["arms"][a]["memory_e10"]
                L.append(f"| {a} | {m['a_saved_token_side_per_token']:,.0f} | {m['b_train_peak_per_token']:,.0f} | {m['c_prefill_peak_per_token']:,.0f} |")
    (run_dir / "analysis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
