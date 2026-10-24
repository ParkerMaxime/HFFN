"""Measured section of a RESULTS.md, generated from analysis.json: `python -m lib.report <run dir>` prints markdown."""
import json
import statistics as st
import sys
from pathlib import Path


def main(run_dir):
    run_dir = Path(run_dir).resolve()
    a = json.loads((run_dir / "analysis.json").read_text())
    man = json.loads((run_dir / "manifest.json").read_text())
    jobs = man["jobs"]
    ok = sum(1 for j in jobs.values() if j["status"] == "passed")
    out = [f"Qualified run: `{run_dir.relative_to(run_dir.parents[2])}/` (EXIT {man['exit']}; {ok}/{len(jobs)} jobs `passed`, "
           f"exit 0, full validation split). Analysis: `analysis.json`, `analysis.md` in that folder.", "",
           "| arm | params | per seed (s0 / s1 / s2 / s3) | median | [min–max] | by class t mod 4 (median) |", "|---|---|---|---|---|---|"]
    for arm, e in a["arms"].items():
        out.append(f"| {arm} | {e['params_outside_embedding']:,} | " + " / ".join(f"{v:.4f}" for v in e["bpb_per_seed"].values())
                   + f" | {e['median']:.4f} | [{e['range'][0]:.4f}–{e['range'][1]:.4f}] | "
                   + " / ".join(f"{x:.3f}" for x in e["bpb_by_class_median"]) + " |")
    for c in a["comparisons"]:
        out += ["", f"**{c['candidate']} against {c['control']}** (bound {c['bound']}, {c['direction']}): paired difference per seed "
                + ", ".join(f"s{s} {x:+.4f}" for s, x in c["paired_diff_per_seed"].items())
                + f"; median {c['median']:+.4f}, range [{c['range'][0]:+.4f}, {c['range'][1]:+.4f}], paired band {c['band']:.4f}. "
                f"**Verdict: {c['verdict']}.**"]
    out += ["", "Mixing diagnostic (descriptive; increase of validation bpb on 4,096 windows when the cross-block part is zeroed, median over seeds):", "",
            "| arm | W_Q | W_K, W_V | W_down | all |", "|---|---|---|---|---|"]
    for arm, e in a["arms"].items():
        m = e["mixing_median"]
        out.append(f"| {arm} | {m['delta_q']:+.4f} | {m['delta_kv']:+.4f} | {m['delta_down']:+.4f} | {m['delta_all']:+.4f} |")
    for arm, e in a["arms"].items():
        if "router_load_median" in e:
            out += ["", f"Router ({arm}, median over seeds, final validation): load per group "
                    + " / ".join(f"{x:.3f}" for x in e["router_load_median"]) + f"; dropped fraction {e['router_dropped_median']:.4f}."]
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
