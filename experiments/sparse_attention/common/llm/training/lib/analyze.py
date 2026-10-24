"""Re-analysis of a step run folder: `python -m lib.analyze <run dir>` -> analysis.json and analysis.md in that folder.

Reads only the run records (summary.json of each job) and `step.json`; writes new files, never edits a record.

Statistics (frozen in the README of E0, applied by every later step with the bound of its own README):
* per arm: median and [min-max] over the seeds of the final validation bits per byte;
* per comparison: paired difference per seed d_s = bpb(candidate, s) - bpb(control, s) (positive = candidate worse);
  direction "max_degradation" with bound tau: within bound if max_s d_s <= tau; beyond bound if min_s d_s > tau;
  otherwise inconclusive (the seeds straddle the bound). "better in every seed" if max_s d_s < 0.
  direction "min_improvement" (IND): d_s = bpb(control, s) - bpb(candidate, s) (positive = candidate better);
  met if min_s d_s > tau; not met if max_s d_s <= tau; otherwise inconclusive;
* E0 discrimination: Delta_attn = median_s (bpb(NOATT, s) - bpb(E0, s)); band = the larger of the two seed ranges;
  discriminating iff Delta_attn / band >= R_min and d_s > 0 for every seed.
"""
import json
import statistics as st
import sys
from pathlib import Path


def load(run_dir, stage):
    out = {}
    for d in sorted((run_dir / stage).glob("*_s*")):
        f = d / "summary.json"
        if not f.exists() or "_lr" in d.name:
            continue
        s = json.loads(f.read_text())
        out.setdefault(s["arm"], {})[s["seed"]] = {"summary": s, "dir": str(d.relative_to(run_dir.parents[1]))}
    return out


def rng(v):
    return [min(v), max(v)]


def compare(cmp, runs):
    a, b = runs[cmp["candidate"]], runs[cmp["control"]]
    seeds = sorted(set(a) & set(b))
    if cmp.get("direction", "max_degradation") == "max_degradation":
        d = {s: a[s]["summary"]["final_valid_bpb"] - b[s]["summary"]["final_valid_bpb"] for s in seeds}
    else:
        d = {s: b[s]["summary"]["final_valid_bpb"] - a[s]["summary"]["final_valid_bpb"] for s in seeds}
    v = list(d.values())
    tau = cmp["bound"]
    if cmp.get("direction", "max_degradation") == "max_degradation":
        verdict = "within bound" if max(v) <= tau else ("beyond bound" if min(v) > tau else "inconclusive (seeds straddle the bound)")
        if max(v) < 0:
            verdict += "; candidate better than control in every seed"
    else:
        verdict = "met" if min(v) > tau else ("not met" if max(v) <= tau else "inconclusive (seeds straddle the bound)")
    return {"candidate": cmp["candidate"], "control": cmp["control"], "direction": cmp.get("direction", "max_degradation"),
            "bound": tau, "paired_diff_per_seed": d, "median": st.median(v), "range": rng(v), "band": max(v) - min(v),
            "verdict": verdict}


def main(run_dir):
    run_dir = Path(run_dir).resolve()
    spec = json.loads((run_dir.parents[1] / "step.json").read_text())
    stages = spec["analysis"]["stage"]
    runs = {}
    for stage in ([stages] if isinstance(stages, str) else stages):
        runs.update(load(run_dir, stage))
    res = {"step": spec["step"], "run": str(run_dir.relative_to(run_dir.parents[2])), "arms": {}}
    for arm, by_seed in runs.items():
        bpb = {s: v["summary"]["final_valid_bpb"] for s, v in by_seed.items()}
        vals = list(bpb.values())
        cls = [[v["summary"]["final_valid"]["bpb_by_class"][c] for v in by_seed.values()] for c in range(4)]
        mix = {k: [v["summary"]["mixing"][k] for v in by_seed.values()] for k in
               ("base_bpb", "delta_q", "delta_kv", "delta_down", "delta_all")}
        entry = {"seeds": sorted(bpb), "bpb_per_seed": bpb, "median": st.median(vals), "range": rng(vals),
                 "params_outside_embedding": next(iter(by_seed.values()))["summary"]["params_outside_embedding"],
                 "bpb_by_class_median": [st.median(c) for c in cls],
                 "mixing_median": {k: st.median(v) for k, v in mix.items()},
                 "run_dirs": {s: v["dir"] for s, v in by_seed.items()}}
        loads = [v["summary"]["final_valid"].get("router_load") for v in by_seed.values()]
        if loads[0] is not None:
            entry["router_load_median"] = [st.median(l[g] for l in loads) for g in range(len(loads[0]))]
            entry["router_dropped_median"] = st.median(v["summary"]["final_valid"]["router_dropped"] for v in by_seed.values())
        res["arms"][arm] = entry
    res["comparisons"] = [compare(c, runs) for c in spec["analysis"].get("comparisons", [])]
    disc = spec["analysis"].get("discrimination")
    if disc:
        a, b = runs[disc["with"]], runs[disc["reference"]]
        d = {s: a[s]["summary"]["final_valid_bpb"] - b[s]["summary"]["final_valid_bpb"] for s in sorted(set(a) & set(b))}
        band = max(res["arms"][disc["with"]]["range"][1] - res["arms"][disc["with"]]["range"][0],
                   res["arms"][disc["reference"]]["range"][1] - res["arms"][disc["reference"]]["range"][0])
        delta = st.median(d.values())
        ratio = delta / band if band > 0 else float("inf")
        ok = ratio >= disc["r_min"] and all(x > 0 for x in d.values())
        res["discrimination"] = {"delta_attn_per_seed": d, "delta_attn_median": delta, "band": band, "ratio": ratio,
                                 "r_min": disc["r_min"], "discriminating": ok}
    (run_dir / "analysis.json").write_text(json.dumps(res, indent=1))
    lines = [f"# Analysis of {res['run']}", "", "| arm | params | median bpb | [min-max] | by class t mod 4 (median) |", "|---|---|---|---|---|"]
    for arm, e in res["arms"].items():
        lines.append(f"| {arm} | {e['params_outside_embedding']} | {e['median']:.4f} | [{e['range'][0]:.4f}-{e['range'][1]:.4f}] | "
                     + " / ".join(f"{x:.3f}" for x in e["bpb_by_class_median"]) + " |")
    lines += ["", "| arm | mixing: Δ(Q) | Δ(K,V) | Δ(W_down) | Δ(all) | (median Δ bpb on 4096 valid windows) |", "|---|---|---|---|---|---|"]
    for arm, e in res["arms"].items():
        m = e["mixing_median"]
        lines.append(f"| {arm} | {m['delta_q']:+.4f} | {m['delta_kv']:+.4f} | {m['delta_down']:+.4f} | {m['delta_all']:+.4f} | |")
    for c in res["comparisons"]:
        lines += ["", f"**{c['candidate']} vs {c['control']}** ({c['direction']}, bound {c['bound']}): per-seed d = "
                  + ", ".join(f"s{s}: {x:+.4f}" for s, x in c["paired_diff_per_seed"].items())
                  + f"; median {c['median']:+.4f}, range [{c['range'][0]:+.4f}, {c['range'][1]:+.4f}] → **{c['verdict']}**"]
    if "discrimination" in res:
        d = res["discrimination"]
        lines += ["", f"**Discrimination**: Δ_attn = {d['delta_attn_median']:.4f} bpb (per seed "
                  + ", ".join(f"{x:.4f}" for x in d["delta_attn_per_seed"].values())
                  + f"), band = {d['band']:.4f}, ratio = {d['ratio']:.1f} (R_min {d['r_min']}) → "
                  + ("**discriminating**" if d["discriminating"] else "**non-discriminating**")]
    (run_dir / "analysis.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
