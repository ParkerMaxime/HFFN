"""Rule of E6 applied to a run folder: `python -m lib.analyze_e6 <run dir>` -> analysis.json, analysis.md (new files)."""
import json
import math
import statistics as st
import sys
from pathlib import Path

from .count import count_arm


def main(run_dir):
    run_dir = Path(run_dir).resolve()
    spec = json.loads((run_dir.parents[1] / "step.json").read_text())
    rule = spec["analysis"]["rule"]
    horizons, ref, tol = rule["horizons"], rule["reference"], rule["tolerance"]
    seeds = sorted({int(d.name.split("_s")[1].split("_")[0]) for d in (run_dir / "branches").glob("*_s*_H*")})
    bpb = {}
    for d in (run_dir / "branches").glob("*_s*_H*"):
        s = json.loads((d / "summary.json").read_text())
        bpb[(s["arm"], s["seed"], int(d.name.split("_H")[1]))] = s["final_valid_bpb"]
    gap = {(s, h): bpb[("E4a_L4", s, h)] - bpb[("REF2", s, h)] for s in seeds for h in horizons}
    dev = {(s, h): abs(gap[(s, h)] - gap[(s, ref)]) for s in seeds for h in horizons if h != ref}
    passed = {h: all(dev[(s, h)] <= tol for s in seeds) for h in horizons if h != ref}
    hstar = next((h for h in sorted(passed) if passed[h]), ref)
    # secondary reading (b): equal training compute, estimated
    tr = {a: count_arm(a)["training_macs_per_token"] for a in ("REF2", "E4a_L4")}
    ratio = tr["REF2"] / tr["E4a_L4"]
    equal = {}
    for s in seeds:
        for h in horizons:
            hc = h * ratio
            if not (horizons[0] <= hc <= horizons[-1]):
                equal[(s, h)] = None
                continue
            lo = max(x for x in horizons if x <= hc)
            hi = min(x for x in horizons if x >= hc)
            w = 0.0 if lo == hi else (math.log(hc) - math.log(lo)) / (math.log(hi) - math.log(lo))
            b = bpb[("E4a_L4", s, lo)] * (1 - w) + bpb[("E4a_L4", s, hi)] * w
            equal[(s, h)] = {"E4a_L4_steps": hc, "E4a_L4_bpb": b, "gap": b - bpb[("REF2", s, h)]}
    res = {"seeds": seeds, "horizons": horizons, "bpb": {f"{a}/s{s}/H{h}": v for (a, s, h), v in sorted(bpb.items())},
           "gap": {f"s{s}/H{h}": v for (s, h), v in gap.items()},
           "deviation_from_H8000": {f"s{s}/H{h}": v for (s, h), v in dev.items()},
           "passes": {str(h): v for h, v in passed.items()}, "tolerance": tol, "H_star": hstar,
           "equal_training_compute_estimated": {f"s{s}/H{h}": v for (s, h), v in equal.items()},
           "training_macs_per_token": tr, "steps_ratio_equal_compute": ratio}
    (run_dir / "analysis.json").write_text(json.dumps(res, indent=1))
    L = ["# E6 analysis", "", "| seed | H | REF2 | E4a_L4 | gap | |gap − gap(8,000)| |", "|---|---|---|---|---|---|"]
    for s in seeds:
        for h in horizons:
            L.append(f"| {s} | {h} | {bpb[('REF2', s, h)]:.4f} | {bpb[('E4a_L4', s, h)]:.4f} | {gap[(s, h)]:+.4f} | "
                     + (f"{dev[(s, h)]:.4f}" if h != ref else "—") + " |")
    L += ["", f"Rule: smallest of {horizons[:-1]} with both deviations ≤ {tol}: " + ", ".join(f"H={h}: {'pass' if v else 'fail'}" for h, v in passed.items())
          + f" → **H\\* = {hstar}**", "", "Equal training compute (estimated; E4a_L4 steps = H × %.4f):" % ratio, ""]
    for s in seeds:
        for h in horizons:
            e = equal[(s, h)]
            L.append(f"- seed {s}, REF2 at {h}: " + ("no value (outside the horizons of E4a_L4)" if e is None else
                     f"E4a_L4 at {e['E4a_L4_steps']:.0f} steps → bpb {e['E4a_L4_bpb']:.4f}, gap {e['gap']:+.4f}"))
    (run_dir / "analysis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
