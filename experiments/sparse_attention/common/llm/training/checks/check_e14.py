#!/usr/bin/env python3
"""Checks of the preparation of E14 (PLAN section 17, CLOUD_E14.md). Exits non-zero on failure.

Covers: the new arms (E2h at three sizes, REF4_W256) and their counts against the formulas of PLAN 17.7, the packed form of E2h against
the selected form (outputs and the gradient of every parameter), the executor kind of E2h against the sequential forward, the decoding
path (the decoded step against the full forward at the same position, for every arm of the three campaigns and of the pairs; the
packed sequential step against the reference; the step under the executor against the sequential step; repeated calls; the filled
part of the cache untouched), the per-group counts of the 64 new tokens, the cells of the timing harness, the analysis of a timing
run on synthetic raw slots, the memory harness for (e) on a synthetic pattern and in two fresh processes. The first three series
stay reproducible: their own check suites are run with this one by the preparation.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import e14, memory, memory4, timing, timing4  # noqa: E402
from lib.arms import ARMS, get_arm  # noqa: E402
from lib.count import count_arm, layer_macs  # noqa: E402
from lib.data import Enwik8, balanced_byte_hash  # noqa: E402
from lib.model import SparseAttnLM  # noqa: E402

TRAIN = Path(__file__).resolve().parents[1]
ENV1 = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
FAILS, RESULTS = [], []
# tolerances (stated in prep4/README.md): float32 rounding of the same computation in another order
TOL_LOGITS, TOL_GRAD = 1e-4, 1e-5

ARMS_ALL = (["REF1", "REF2", "REF3", "REF4", "E4a_L2", "E4a_L4", "E4a_L6", "IND_L4"]
            + [a + "_N512" for a in ("REF1", "REF2", "REF3", "REF4", "E4a_L2", "E4a_L4", "E4a_L6", "IND_L4")]
            + ["REF1_W256", "REF2_W256", "REF2_W256_h16", "E4a_L2_W256", "E4a_L4_W256", "E4a_L4_W256_h16", "REF4_W256",
               "E2h_L4", "E2h_L4_N512", "E2h_L4_W256"])


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def check_arms_and_counts():
    for name, ctx in (("E2h_L4", 128), ("E2h_L4_N512", 512), ("E2h_L4_W256", 512), ("REF4_W256", 512)):
        check(f"arm {name} exists, context {ctx}", name in ARMS and get_arm(name).context == ctx)
    e2h, e4a = get_arm("E2h_L4"), get_arm("E4a_L4")
    check("E2h: dense attention, dense W_O, routed FFN of E4a under the hash route, no router, 4 layers",
          e2h.attn == "dense" and not e2h.wo_blockdiag and e2h.ffn_routed and e2h.ffn_support == "blocks"
          and e2h.routing == "hash" and e2h.n_layers == 4 and e2h.layer_routing == "fixed")
    for a, b, per_layer, label in (("E2h_L4", "E4a_L4", 12288, "M = 128"), ("E2h_L4_W256", "E4a_L4_W256", 49152, "M = 256")):
        pa, pb = count_arm(a)["params_outside_embedding"], count_arm(b)["params_outside_embedding"]
        check(f"{a} has the parameters of {b} plus {per_layer} per layer (W_O), {label}", pa - pb == 4 * per_layer, f"{pa} - {pb} = {pa - pb}")
    # hand count of the multiply-adds of E2h_L4 at M = 128, N = 128: keys, values, queries, scores, weighted values, W_O dense, routed FFN
    M, N, n, k, V, L = 128, 128, 128, 26, 256, 4
    hand = L * (2 * M * M + M * M + N * M + N * M + M * M + 2 * n * k) + V * M
    check("E2h_L4 inference multiply-adds equal the hand count (452,608)", count_arm("E2h_L4")["inference_macs_per_token"] == hand == 452608,
          f"{count_arm('E2h_L4')['inference_macs_per_token']} vs {hand}")
    t = layer_macs(get_arm("E2h_L4"))
    ta = layer_macs(get_arm("REF4"))
    check("E2h layer terms equal the dense layer except the FFN (routed: 2 n k against 2 H k)",
          {x: v for x, v in t.items() if x != "ffn"} == {x: v for x, v in ta.items() if x != "ffn"} and t["ffn"] == 2 * 128 * 26 and ta["ffn"] == 2 * 512 * 26)
    check("KV cache stored: 2 L M in E2h, E4a and REF4 (decision 3)",
          len({count_arm(a)["kv_cache_values_per_token"] for a in ("E2h_L4", "E4a_L4", "REF4")}) == 1)
    kr = {a: e14.kv_read_values_per_decoded_token(get_arm(a)) for a in ("E4a_L4", "E2h_L4", "REF4")}
    check("KV read per decoded token: E4a / E2h = 0.25 at equal depth, E2h / REF4 = 1", kr["E4a_L4"] / kr["E2h_L4"] == 0.25 and kr["E2h_L4"] == kr["REF4"], f"{kr}")


def check_gather_parity():
    data = Enwik8()
    ht = balanced_byte_hash(data.train_counts(), 4)
    for name in ("E2h_L4", "E2h_L4_N512", "E2h_L4_W256"):
        cfg = get_arm(name)
        x, y = data.eval_windows("valid", cfg.context, max(2, 2048 // cfg.context))
        outs = {}
        for impl in ("selected", "gather"):
            m = SparseAttnLM(cfg, 3, hash_table=ht, impl=impl)
            logits, _ = m(x)
            F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)).backward()
            outs[impl] = (logits.detach(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None})
        dl = float((outs["selected"][0] - outs["gather"][0]).abs().max())
        dg = max(float((outs["selected"][1][n] - outs["gather"][1][n]).abs().max()) for n in outs["selected"][1])
        check(f"gather (packed form of E2h) == selected: logits and every parameter gradient, {name}",
              dl <= TOL_LOGITS and dg <= TOL_GRAD and set(outs["selected"][1]) == set(outs["gather"][1]), f"max|logit diff| {dl:.1e}, max|grad diff| {dg:.1e}")


def sub(args):
    r = subprocess.run([sys.executable, *args], cwd=TRAIN, env=ENV1, capture_output=True, text=True)
    last = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    return r, (json.loads(last[-1]) if last else None)


def check_executor_e2h():
    arms = "E2h_L4,E2h_L4_N512,E2h_L4_W256,REF4_W256"
    r, res = sub(["-m", "lib.executor", "--parity", "--arms", arms])
    if r.returncode != 0 or res is None:
        check("executor parity of E2h ran", False, r.stderr[-500:])
        return
    for name, diff in res.items():
        check(f"executor == sequential forward within float32 rounding, {name}", diff <= TOL_LOGITS, f"max|diff| {diff:.1e}")


def check_decode_path():
    r, res = sub(["-m", "lib.decode", "--parity", "--arms", ",".join(ARMS_ALL)])
    if r.returncode != 0 or res is None:
        check("decoding path parity ran", False, r.stderr[-800:])
        return
    for name in ARMS_ALL:
        d = res[name]
        check(f"decoded step == full forward at the same position (reference, packed sequential, executor), {name}",
              d["reference_vs_full_forward"] <= TOL_LOGITS and d["sequential_vs_reference"] <= TOL_LOGITS,
              f"ref vs full {d['reference_vs_full_forward']:.1e}, sequential vs ref {d['sequential_vs_reference']:.1e}")
        check(f"decoded step under the executor == sequential decoded step; repeated call identical; filled cache untouched, {name}",
              d["executor_vs_sequential"] <= TOL_LOGITS and d["repeat"] <= TOL_LOGITS and d["prefix_of_cache_unchanged"],
              f"executor vs sequential {d['executor_vs_sequential']:.1e}, repeat {d['repeat']:.1e}")
    # per-group counts of the 64 new tokens of a process (the hash depends on the byte only: same for every arm of a context)
    from lib import decode as D
    data = Enwik8()
    ht = torch.as_tensor(balanced_byte_hash(data.train_counts(), 4)).long()
    for N in (128, 512):
        toks = D.decode_windows(data, N, 64, 0)
        counts = [int((ht[toks[:, N // 2]] == g).sum()) for g in range(4)]
        check(f"per-group counts of the 64 new tokens at N = {N} recorded (sum 64)", sum(counts) == 64, f"{counts}")
    check("decoding windows: drawn per process, different between processes, equal for equal processes",
          not torch.equal(D.decode_windows(data, 128, 64, 0), D.decode_windows(data, 128, 64, 1))
          and torch.equal(D.decode_windows(data, 128, 64, 0), D.decode_windows(data, 128, 64, 0)))


def check_cells():
    check("cells: 19 frontier cells (7 + 7 + 5) and 6 pair cells",
          sum(c["group"] == "frontier" for c in timing4.CELLS4.values()) == 19 and sum(c["group"] == "pair" for c in timing4.CELLS4.values()) == 6)
    check("every cell: variant and anchor are arms of the same width and context",
          all(timing4.cell_arms(c)[0].d_model == timing4.cell_arms(c)[1].d_model and timing4.cell_arms(c)[0].context == timing4.cell_arms(c)[1].context
              for c in timing4.CELLS4))
    check("pairs p1-p6 as PLAN 17.7 (attention alone: E4a_L4 / E2h_L4; routing of the FFN alone: E2h_L4 / REF4)",
          [timing4.PAIRS[p][:2] for p in ("p1", "p2", "p3", "p4", "p5", "p6")]
          == [("E4a_L4", "E2h_L4"), ("E2h_L4", "REF4"), ("E4a_L4_N512", "E2h_L4_N512"), ("E2h_L4_N512", "REF4_N512"),
              ("E4a_L4_W256", "E2h_L4_W256"), ("E2h_L4_W256", "REF4_W256")])
    # one cell of each kind builds and runs once in both modes (the smoke runs every cell)
    code = ("import torch; torch.set_num_threads(1); torch.set_num_interop_threads(4); torch.set_grad_enabled(False)\n"
            "from lib.timing4 import build_cell4\nimport json\nout={}\n"
            "for cell in ('n128-IND_L4','p5','w256-REF2_W256_h16'):\n"
            "    for mode in ('prefill','decode'):\n"
            "        arms, desc = build_cell4(cell, mode)\n"
            "        for a in arms.values(): a()\n"
            "        out[cell+'/'+mode] = True\n"
            "print(json.dumps(out))\n")
    r, res = sub(["-c", code])
    check("a frontier cell, a pair cell and a width cell build and run once in both modes", r.returncode == 0 and res is not None and len(res) == 6, "" if r.returncode == 0 else r.stderr[-400:])


def check_timing_analysis():
    """analyse_timing on synthetic raw slots with a known ratio (variant 0.7 of the anchor), through the same files as a run."""
    def sleeper(sec):
        def f():
            t = time.perf_counter()
            while time.perf_counter() - t < sec:
                pass
        return f
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        for i in range(3):
            timing.run_windows({"control": sleeper(0.004), "copy": sleeper(0.004), "variant": sleeper(0.0028)},
                               run / "cells" / "p1" / f"p{i}", windows=4, warm_calls=5, warm_seconds=0.1, slot_seconds=0.03, cell="p1", process=i)
        an = e14.analyse_timing(run, ["p1"], 3, "prefill")
    c = an["cells"]["p1"]
    check("analysis of a timing run: median ratio recovers a known 0.7 (within 3 %), the copy 1.0, one delta per run, outcome rule",
          abs(c["r_median"] - 0.7) < 0.021 and abs(an["delta"]) < 0.03 and c["outcome"] == "gain (variant faster)",
          f"r {c['r_median']:.3f}, delta {an['delta']:.3f}, outcome {c['outcome']}")
    check("counted ratios: prefill uses N keys, decode N / 2 keys (E4a_L4 / E2h_L4)",
          e14.counted_macs(get_arm("E4a_L4"), "prefill") == count_arm("E4a_L4")["inference_macs_per_token"]
          and e14.counted_macs(get_arm("E4a_L4"), "decode") < e14.counted_macs(get_arm("E4a_L4"), "prefill"))


def check_memory_decode():
    MiB = 1024 * 1024
    held = {}

    def pattern():
        a = torch.empty(4 * MiB // 4)
        b = torch.empty(2 * MiB // 4)
        del a
        c = torch.empty(3 * MiB // 4)
        held["x"] = (b, c)
        held.clear()

    leftover = [torch.empty(3 * MiB // 4)]  # allocated inside a profiled region and kept alive: shifts the next region's counter
    memory.measure_peak(lambda: leftover.append(torch.empty(MiB // 4)))
    pk = memory4.measure_peak_relative(pattern)
    check("memory (e): the peak of a synthetic pattern equals the hand value (6 MiB) even after a region that left tensors alive",
          pk["peak_live_bytes"] == 6 * MiB and pk["live_at_end_bytes"] == 0, f"{pk}")
    bufs = {"a": torch.empty(MiB // 4)}

    def step():
        bufs["a"] = torch.empty(MiB // 4)
        t = torch.empty(2 * MiB // 4)
        del t

    pk2 = memory4.decode_peak(step, bufs)
    check("memory (e): the working buffers are released before the region and counted inside it (1 MiB buffer + 2 MiB temporary = 3 MiB)",
          pk2["peak_live_bytes"] == 3 * MiB and pk2["live_at_end_bytes"] == MiB, f"{pk2}")
    outs = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(2):
            f = Path(tmp) / f"m{i}.json"
            subprocess.run([sys.executable, "-m", "lib.memory4", "--arm", "E2h_L4", "--out", str(f)], cwd=TRAIN, env=ENV1, check=True, capture_output=True)
            outs.append(json.loads(f.read_text()))
    a, b = outs
    rel = lambda u, v: abs(u - v) / max(u, 1)  # noqa: E731
    check("memory harness of E14 on E2h_L4 in two fresh processes: (a) identical, (b), (c), (e) within 1 %",
          a["saved"] == b["saved"] and rel(a["train_step"]["peak_live_bytes"], b["train_step"]["peak_live_bytes"]) <= 0.01
          and rel(a["prefill"]["peak_live_bytes"], b["prefill"]["peak_live_bytes"]) <= 0.01
          and rel(a["decode"]["peak_live_bytes"], b["decode"]["peak_live_bytes"]) <= 0.01,
          f"(e) {a['decode']['peak_live_bytes']} / {b['decode']['peak_live_bytes']}, (c) {a['prefill']['peak_live_bytes']} / {b['prefill']['peak_live_bytes']}")


def main():
    t0 = time.time()
    torch.manual_seed(0)
    for fn in (check_arms_and_counts, check_gather_parity, check_executor_e2h, check_decode_path, check_cells, check_timing_analysis,
               check_memory_decode):
        print(f"--- {fn.__name__}", flush=True)
        fn()
    out = {"checks": RESULTS, "failed": FAILS, "wall_s": time.time() - t0}
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(f"{len(RESULTS) - len(FAILS)}/{len(RESULTS)} checks passed", flush=True)
    if FAILS:
        print("FAILED:", *FAILS, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
