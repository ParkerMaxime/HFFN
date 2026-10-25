"""Readings of E8 / E9 (PLAN 14.4) from a run folder: `python -m lib.analyze_frontier <run dir>` -> analysis.json, analysis.md.

Reads the summaries of the `final` stage and `step.json` (`analysis.suffix`: "" at N = 128, "_N512" at N = 512), the counter
(`lib/count.py`) and writes new files only. Every reading is paired per seed, `d_s = bpb(candidate, s) - bpb(control, s)`
(positive = candidate worse), with the three outcomes of PLAN 13.1 at `tau` (within: max_s d_s <= tau; beyond: min_s d_s > tau;
otherwise inconclusive) and, where the candidate costs less inference compute than the control, *dominates* (max_s d_s < 0).
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

from .count import count_arm

DENSE = ["REF1", "REF2", "REF3", "REF4"]
PAIRS_EQUAL_COMPUTE = [("E4a_L2", "REF1"), ("E4a_L4", "REF2"), ("E4a_L6", "REF3")]
PAIRS_EQUAL_DEPTH = [("E4a_L2", "REF2"), ("E4a_L4", "REF4")]
PAIRS_MIXING = [("IND_L4", "E4a_L4")]


def outcome(d, tau, cheaper):
    v = list(d.values())
    out = "within bound" if max(v) <= tau else ("beyond bound" if min(v) > tau else "inconclusive (seeds straddle the bound)")
    dom = cheaper and max(v) < 0
    return out, dom


def pair(bpb, cand, ctrl, tau, cost):
    seeds = sorted(set(bpb[cand]) & set(bpb[ctrl]))
    d = {s: bpb[cand][s] - bpb[ctrl][s] for s in seeds}
    cheaper = cost[cand] < cost[ctrl]
    out, dom = outcome(d, tau, cheaper)
    v = list(d.values())
    return {"candidate": cand, "control": ctrl, "d_per_seed": d, "median": st.median(v), "range": [min(v), max(v)],
            "band": max(v) - min(v), "outcome": out, "dominates": dom,
            "inference_macs": {cand: cost[cand], ctrl: cost[ctrl]},
            "compute_saved": 1 - cost[cand] / cost[ctrl] if cheaper else None}


def interpolated(bpb, cand, cost, suffix, tau):
    xs = [(math.log(cost[d]), d) for d in DENSE]
    c = math.log(cost[cand])
    if not (xs[0][0] <= c <= xs[-1][0]):
        return {"candidate": cand, "value": None, "why": "compute outside the range of the dense family"}
    lo = max(x for x in xs if x[0] <= c)
    hi = min(x for x in xs if x[0] >= c)
    w = 0.0 if lo[1] == hi[1] else (c - lo[0]) / (hi[0] - lo[0])
    seeds = sorted(set(bpb[cand]) & set(bpb[lo[1]]) & set(bpb[hi[1]]))
    interp = {s: bpb[lo[1]][s] * (1 - w) + bpb[hi[1]][s] * w for s in seeds}
    d = {s: bpb[cand][s] - interp[s] for s in seeds}
    out, dom = outcome(d, tau, True)
    v = list(d.values())
    return {"candidate": cand, "between": [lo[1], hi[1]], "weight_on_upper": w, "dense_interpolated_per_seed": interp,
            "d_per_seed": d, "median": st.median(v), "range": [min(v), max(v)], "outcome": out, "dominates": dom}


def main(run_dir):
    run_dir = Path(run_dir).resolve()
    spec = json.loads((run_dir.parents[1] / "step.json").read_text())
    suffix, tau = spec["analysis"]["suffix"], spec["analysis"].get("tau", 0.075)
    valid, test, extra = {}, {}, {}
    for d in sorted((run_dir / "final").glob("*_s*")):
        s = json.loads((d / "summary.json").read_text())
        arm = s["arm"][:-len(suffix)] if suffix and s["arm"].endswith(suffix) else s["arm"]
        valid.setdefault(arm, {})[s["seed"]] = s["final_valid_bpb"]
        if "final_test" in s:
            test.setdefault(arm, {})[s["seed"]] = s["final_test"]["bpb"]
        extra.setdefault(arm, {})[s["seed"]] = {k: s["final_valid"].get(k) for k in ("bpb_by_range", "range_names", "range_lengths")}
        if "mixing" in s:
            extra[arm][s["seed"]]["mixing"] = s["mixing"]
    counts = {a: count_arm(a + suffix) for a in valid}
    cost = {a: counts[a]["inference_macs_per_token"] for a in counts}
    res = {"suffix": suffix, "tau": tau, "arms": {}, "readings": {}}
    for a in valid:
        v = list(valid[a].values())
        e = {"valid_median": st.median(v), "valid_range": [min(v), max(v)], "valid_per_seed": valid[a],
             "counter": {k: counts[a][k] for k in ("inference_macs_per_token", "training_macs_per_token", "params_outside_embedding",
                                                   "adamw_state_values_outside_embedding", "kv_cache_values_per_token",
                                                   "training_activation_values_per_token")}}
        if a in test:
            t = list(test[a].values())
            e["test_median"], e["test_range"], e["test_per_seed"] = st.median(t), [min(t), max(t)], test[a]
        if suffix:
            rngs = [x["bpb_by_range"] for x in extra[a].values() if x["bpb_by_range"]]
            if rngs:
                e["range_names"] = next(iter(extra[a].values()))["range_names"]
                e["bpb_by_range_median"] = [st.median(r[i] for r in rngs) for i in range(len(rngs[0]))]
        mx = [x["mixing"] for x in extra[a].values() if "mixing" in x]
        if mx:
            e["mixing_median"] = {k: st.median(m[k] for m in mx) for k in ("delta_q", "delta_kv", "delta_down", "delta_all")}
        res["arms"][a] = e
    R = res["readings"]
    R["equal_compute_pairs"] = [pair(valid, c, d, tau, cost) for c, d in PAIRS_EQUAL_COMPUTE if c in valid and d in valid]
    R["equal_compute_interpolated_estimated"] = [interpolated(valid, c, cost, suffix, tau) for c, _ in PAIRS_EQUAL_COMPUTE if c in valid]
    R["equal_depth"] = [pair(valid, c, d, tau, cost) for c, d in PAIRS_EQUAL_DEPTH if c in valid and d in valid]
    R["mixing_at_depth"] = [pair(valid, c, d, tau, cost) for c, d in PAIRS_MIXING if c in valid and d in valid]
    if test:
        R["test_equal_compute_pairs"] = [pair(test, c, d, tau, cost) for c, d in PAIRS_EQUAL_COMPUTE if c in test and d in test]
    (run_dir / "analysis.json").write_text(json.dumps(res, indent=1, default=str))
    L = [f"# Frontier analysis ({run_dir.name}, suffix '{suffix}', tau {tau})", "",
         "| arm | valid median [min–max] | test median [min–max] | inference MACs | training MACs | params | KV values/token |", "|---|---|---|---|---|---|---|"]
    for a in sorted(res["arms"], key=lambda a: cost[a]):
        e = res["arms"][a]
        t = f"{e['test_median']:.4f} [{e['test_range'][0]:.4f}–{e['test_range'][1]:.4f}]" if "test_median" in e else "—"
        c = e["counter"]
        L.append(f"| {a}{suffix} | {e['valid_median']:.4f} [{e['valid_range'][0]:.4f}–{e['valid_range'][1]:.4f}] | {t} | "
                 f"{c['inference_macs_per_token']:,} | {c['training_macs_per_token']:,} | {c['params_outside_embedding']:,} | {c['kv_cache_values_per_token']:,} |")

    def line(r):
        s = ", ".join(f"s{k} {v:+.4f}" for k, v in r["d_per_seed"].items())
        return (f"- {r['candidate']} vs {r['control']}: d = {s}; median {r['median']:+.4f} [{r['range'][0]:+.4f}, {r['range'][1]:+.4f}] → "
                f"**{r['outcome']}**" + ("; **dominates**" if r.get("dominates") else ""))
    for title, key in (("Equal compute, dominance pairs (measured)", "equal_compute_pairs"),
                       ("Equal depth (measured)", "equal_depth"), ("Mixing at depth (measured)", "mixing_at_depth"),
                       ("Test split, equal-compute pairs (beside, selects nothing)", "test_equal_compute_pairs")):
        if key in R:
            L += ["", f"## {title}", ""] + [line(r) for r in R[key]]
    L += ["", "## Equal compute, interpolated in log C (estimated)", ""]
    for r in R["equal_compute_interpolated_estimated"]:
        L.append(f"- {r['candidate']}: no value ({r['why']})" if r["value" if "value" in r else "d_per_seed"] is None else
                 f"- {r['candidate']} between {r['between'][0]} and {r['between'][1]} (weight {r['weight_on_upper']:.3f} on the upper): "
                 f"d median {r['median']:+.4f} [{r['range'][0]:+.4f}, {r['range'][1]:+.4f}] → **{r['outcome']}**" + ("; **dominates**" if r['dominates'] else ""))
    if suffix:
        L += ["", "## Bits per byte by position range (median over seeds)", ""]
        for a in sorted(res["arms"], key=lambda a: cost[a]):
            e = res["arms"][a]
            if "bpb_by_range_median" in e:
                L.append(f"- {a}{suffix}: " + ", ".join(f"{n}: {v:.4f}" for n, v in zip(e["range_names"], e["bpb_by_range_median"])))
    (run_dir / "analysis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
