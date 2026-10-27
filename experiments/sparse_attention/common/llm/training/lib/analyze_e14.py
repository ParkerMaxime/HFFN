"""Readings of E14 (PLAN 17.4 and 17.7): `python -m lib.analyze_e14 --step-dir e14 --prefill RUN --decode RUN [--memory RUN]`.

Reads the per-run analyses of the timing runs (`analysis.json`, written by `lib/e14.py`: the ratio `t = t(arm) / t(anchor)` of every cell
and `delta` of the run), the validation bits per byte per seed already on record (E8 `N = 128`, E9 `N = 512`, E13 `M = 256`), the memory
run, and writes new files only (`runs/analysis_<UTC>/analysis.json`, `analysis.md`). Nothing is trained.

Frontier (frozen in `e14/README.md`). Per campaign (n128, n512, w256), mode and seed `s`: the dense arms of the campaign are the points
`(t_d, bpb_s(d))`, the anchor at `t = 1`. The lower envelope keeps a point unless another is at most as slow and at most as high (one strict);
the frontier is the piecewise-linear interpolation in `log t` between the points of the envelope, flat at the lowest `bpb` beyond the last
point of the envelope, defined for `t` between the fastest and the slowest dense time only (no extrapolation). Gap of a constrained arm:
`g_{c,s} = bpb_s(c) - frontier_s(t_c)`. `g+`: `t_c` multiplied by `1 + delta`, every dense time by `1 - delta`; `g-`: the reverse; `delta` of the run
of the mode. Outcomes: *above the frontier* if `max_s g+ < 0`; *below the frontier beyond tau* if `min_s g- > tau`; otherwise *on the frontier*
(median and [min-max] of `g` against `tau`); *outside the range* if `t_c` is beyond the fastest or the slowest dense time under either reading (then
read as a pair against the nearest dense arm in log time: `d_s = bpb_s(c) - bpb_s(nearest)`, the outcomes of PLAN 13.1 at `tau`, and *dominates* if
`t_c (1 + delta) < t_nearest (1 - delta)` and `max_s d_s < 0`). Answer to decision 16 per mode and campaign: *favourable* if at least one `E4a` arm is
above the frontier, *unfavourable* if every `E4a` arm is below it beyond `tau`, otherwise *not resolved*. `IND_L4` is read the same way, as a control.
Every gap combines a time and a bpb by interpolation: **estimated**; the direct pairs (an arm against its anchor in time and in bpb) are **measured**.
"""
import argparse
import json
import math
import statistics as st
from pathlib import Path

from . import campaign, timing4
from .e14 import pair_products

TAU = 0.075
QUALITY = {  # campaign -> (run folder relative to the training folder, suffix of the arm names in its summaries)
    "n128": ("e8/runs/run_20261025T163405Z", ""),
    "n512": ("e9/runs/run_20261025T220720Z", ""),
    "w256": ("e13/runs/run_20261027T040908Z", ""),
}
DENSE = {"n128": ["REF1", "REF2", "REF3", "REF4"], "n512": ["REF1_N512", "REF2_N512", "REF3_N512", "REF4_N512"],
         "w256": ["REF1_W256", "REF2_W256", "REF2_W256_h16"]}
CONSTRAINED = {"n128": ["E4a_L2", "E4a_L4", "E4a_L6"], "n512": ["E4a_L2_N512", "E4a_L4_N512", "E4a_L6_N512"],
               "w256": ["E4a_L2_W256", "E4a_L4_W256", "E4a_L4_W256_h16"]}
CONTROL = {"n128": ["IND_L4"], "n512": ["IND_L4_N512"], "w256": []}
ANCHOR = {"n128": "REF2", "n512": "REF2_N512", "w256": "REF2_W256"}


def load_bpb(train_dir, campaign_id):
    run = Path(train_dir) / QUALITY[campaign_id][0]
    out = {}
    for d in sorted((run / "final").glob("*_s*")):
        s = json.loads((d / "summary.json").read_text())
        out.setdefault(s["arm"], {})[s["seed"]] = s["final_valid_bpb"]
    return out


def envelope(points):
    """Lower envelope of (t, y): a point is dropped if another is at most as slow and at most as high (one strict)."""
    keep, best = [], math.inf
    for t, y in sorted(points):
        if y < best:
            keep.append((t, y))
            best = y
    return keep


def frontier_at(points, t):
    """Frontier value at `t` (between the fastest and the slowest time of `points`), or None outside that range."""
    ts = [p[0] for p in points]
    if not (min(ts) <= t <= max(ts)):
        return None
    env = envelope(points)
    if t >= env[-1][0]:
        return env[-1][1]
    for (t0, y0), (t1, y1) in zip(env, env[1:]):
        if t0 <= t <= t1:
            w = 0.0 if t1 == t0 else (math.log(t) - math.log(t0)) / (math.log(t1) - math.log(t0))
            return y0 * (1 - w) + y1 * w
    return env[0][1]


def outcome_13_1(d, tau):
    v = list(d.values())
    return "within bound" if max(v) <= tau else ("beyond bound" if min(v) > tau else "inconclusive (seeds straddle the bound)")


def gaps_for(campaign_id, times, bpb, delta, tau):
    """Gap readings of one campaign in one mode. `times`: arm -> t (ratio to the anchor, anchor 1.0); `bpb`: arm -> {seed: bpb}."""
    dense = [d for d in DENSE[campaign_id] if d in times and d in bpb]
    seeds = sorted(set.intersection(*(set(bpb[d]) for d in dense)))
    res = {}
    for c in CONSTRAINED[campaign_id] + CONTROL[campaign_id]:
        if c not in times or c not in bpb:
            continue
        tc = times[c]
        per = {}
        outside = False
        for label, (fc, fd) in (("plus", (1 + delta, 1 - delta)), ("minus", (1 - delta, 1 + delta))):
            g = {}
            for s in seeds:
                pts = [(times[d] * fd, bpb[d][s]) for d in dense]
                f = frontier_at(pts, tc * fc)
                if f is None:
                    outside = True
                    break
                g[s] = bpb[c][s] - f
            per[label] = g
        e = {"arm": c, "t": tc, "bpb_per_seed": {s: bpb[c][s] for s in seeds}, "control": c in CONTROL[campaign_id]}
        if not outside:
            gp, gm = per["plus"], per["minus"]
            gm0 = {s: bpb[c][s] - frontier_at([(times[d], bpb[d][s]) for d in dense], tc) for s in seeds}
            v = list(gm0.values())
            e.update({"g_per_seed": gm0, "g_median": st.median(v), "g_range": [min(v), max(v)],
                      "g_plus_per_seed": gp, "g_minus_per_seed": gm})
            if max(gp.values()) < 0:
                e["outcome"] = "above the frontier"
            elif min(gm.values()) > tau:
                e["outcome"] = "below the frontier beyond tau"
            else:
                e["outcome"] = "on the frontier"
        else:
            nearest = min(dense, key=lambda d: abs(math.log(times[d]) - math.log(tc)))
            d = {s: bpb[c][s] - bpb[nearest][s] for s in seeds}
            v = list(d.values())
            e.update({"outcome": "outside the range", "nearest_dense": nearest, "t_nearest": times[nearest], "time_ratio_to_nearest": tc / times[nearest],
                      "d_per_seed": d, "d_median": st.median(v), "d_range": [min(v), max(v)], "pair_outcome": outcome_13_1(d, tau),
                      "dominates": tc * (1 + delta) < times[nearest] * (1 - delta) and max(v) < 0})
        res[c] = e
    return res


def decision_16(arms):
    e4a = [a for a in arms.values() if not a["control"]]
    if any(a["outcome"] == "above the frontier" for a in e4a):
        return "favourable"
    if e4a and all(a["outcome"] == "below the frontier beyond tau" for a in e4a):
        return "unfavourable"
    return "not resolved"


def times_of(an, camp):
    """arm -> t for a campaign from the cells of a timing analysis (the anchor at 1)."""
    t = {ANCHOR[camp]: 1.0}
    for cell, c in an["cells"].items():
        if c["group"] == "frontier" and c["campaign"] == camp:
            t[c["variant"]] = c["r_median"]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-dir", required=True)
    ap.add_argument("--prefill", required=True)
    ap.add_argument("--decode", required=True)
    ap.add_argument("--memory")
    a = ap.parse_args()
    step_dir = Path(a.step_dir).resolve()
    train = step_dir.parent
    out_dir = step_dir / "runs" / f"analysis_{campaign.utc()}"
    out_dir.mkdir(parents=True)
    res = {"tau": TAU, "inputs": {"prefill": a.prefill, "decode": a.decode, "memory": a.memory, "quality": {k: v[0] for k, v in QUALITY.items()}},
           "modes": {}}
    for mode, rel in (("prefill", a.prefill), ("decode", a.decode)):
        an = json.loads((step_dir / rel / "analysis.json").read_text())
        delta = an["delta"]
        m = {"delta": delta, "campaigns": {}, "pairs": {}, "pair_products": pair_products(an)}
        for camp in ("n128", "n512", "w256"):
            bpb = load_bpb(train, camp)
            times = times_of(an, camp)
            arms = gaps_for(camp, times, bpb, delta, TAU)
            direct = {}
            for arm, t in times.items():
                if arm == ANCHOR[camp] or arm not in bpb or ANCHOR[camp] not in bpb:
                    continue
                d = {s: bpb[arm][s] - bpb[ANCHOR[camp]][s] for s in sorted(set(bpb[arm]) & set(bpb[ANCHOR[camp]]))}
                direct[arm] = {"t_vs_anchor": t, "d_vs_anchor_per_seed": d, "d_median": st.median(d.values()), "class": "measured"}
            m["campaigns"][camp] = {"times": times, "arms": arms, "decision_16": decision_16(arms), "direct_pairs_vs_anchor": direct,
                                    "dense_envelope_note": "per seed, see arms[*].g_per_seed"}
        for pid in timing4.PAIRS:
            c = an["cells"][pid]
            m["pairs"][pid] = {k: c[k] for k in ("variant", "control", "r_median", "r_range", "outcome", "counted_ratio", "time_over_count",
                                                 "steal_fraction", "group_counts_of_the_step")}
            m["pairs"][pid]["isolates"] = timing4.PAIRS[pid][2]
            for k in ("counted_kv_read_ratio", "counted_kv_stored_ratio"):
                if k in c:
                    m["pairs"][pid][k] = c[k]
        res["modes"][mode] = m
    if a.memory:
        res["memory"] = json.loads((step_dir / a.memory / "analysis.json").read_text())
    (out_dir / "analysis.json").write_text(json.dumps(res, indent=1, default=str))

    L = ["# E14 analysis (frontier of quality against measured time, pairs, memory)", "",
         "Gaps are **estimated** (a time and a bits-per-byte combined by interpolation); direct pairs, ratios and memory are **measured**.", ""]
    for mode, m in res["modes"].items():
        L += [f"## {mode} (delta of the run = {m['delta']:.4f})", ""]
        for camp, c in m["campaigns"].items():
            L += [f"### {mode}, campaign {camp}: answer to decision 16 — **{c['decision_16']}**", "",
                  "| arm | t (anchor = 1) | outcome | g median [min–max] (estimated) | g+ per seed | g- per seed |", "|---|---|---|---|---|---|"]
            for arm, e in c["arms"].items():
                if e["outcome"] == "outside the range":
                    L.append(f"| {arm}{' (control)' if e['control'] else ''} | {e['t']:.3f} | outside the range | pair against {e['nearest_dense']} (t {e['t_nearest']:.3f}): "
                             f"d median {e['d_median']:+.4f} [{e['d_range'][0]:+.4f}, {e['d_range'][1]:+.4f}], {e['pair_outcome']}"
                             f"{'; dominates' if e['dominates'] else ''}; time ratio {e['time_ratio_to_nearest']:.3f} | | |")
                else:
                    L.append(f"| {arm}{' (control)' if e['control'] else ''} | {e['t']:.3f} | {e['outcome']} | {e['g_median']:+.4f} "
                             f"[{e['g_range'][0]:+.4f}, {e['g_range'][1]:+.4f}] | "
                             + ", ".join(f"{v:+.4f}" for v in e["g_plus_per_seed"].values()) + " | "
                             + ", ".join(f"{v:+.4f}" for v in e["g_minus_per_seed"].values()) + " |")
            L += ["", "Direct pairs against the anchor (measured): " + "; ".join(
                f"{arm}: t {d['t_vs_anchor']:.3f}, d median {d['d_median']:+.4f}" for arm, d in c["direct_pairs_vs_anchor"].items()), ""]
        L += [f"### {mode}, pairs (measured)", "", "| cell | variant / anchor | what it isolates | r median [min–max] | outcome | counted | r / counted |", "|---|---|---|---|---|---|---|"]
        for pid, p in m["pairs"].items():
            L.append(f"| {pid} | {p['variant']} / {p['control']} | {p['isolates']} | {p['r_median']:.3f} [{p['r_range'][0]:.3f}–{p['r_range'][1]:.3f}] | "
                     f"{p['outcome']} | {p['counted_ratio']:.3f} | {p['time_over_count']:.2f} |")
        L += ["", "Products (estimated): " + "; ".join(f"{k}: {v['product_of_medians']:.3f} (counted {v['counted_product']:.3f})"
                                                       for k, v in m["pair_products"].items()), ""]
    if "memory" in res:
        mem = res["memory"]
        L += ["## Memory (measured; bytes per token, fp32; counted beside)", "", "| reading | (a) saved, token side | (b) train peak | (c) prefill peak | (e) decoded step peak | counted MACs | counted KV read |",
              "|---|---|---|---|---|---|---|"]
        for r in mem["ratios"]:
            m_, c_ = r["measured"], r["counted"]
            L.append(f"| {r['label']} ({r['candidate']} / {r['control']}) | {m_['a_saved_token_side_per_token']:.2f} | {m_['b_train_peak_per_token']:.2f} | "
                     f"{m_['c_prefill_peak_per_token']:.2f} | {m_['e_decode_peak_per_decoded_token']:.2f} | {c_['inference_macs']:.2f} | {c_['kv_read_per_decoded_token']:.2f} |")
    (out_dir / "analysis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
