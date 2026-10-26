#!/usr/bin/env python3
"""Checks of the preparation of the third series (E10-E13), PLAN section 15 and CLOUD_E10_E13.md. Exits non-zero on failure.

Covers: the packed form under the hash route against the selected form (outputs and the gradient of every parameter), the arms of
width 256 and their counts against PLAN 15.5, the memory harness on synthetic modules known by hand and in two fresh processes, the
executor against the sequential forward, the timing harness on synthetic arms with a known ratio, and the reproducibility of the
second series (the selected form stays the default under the hash route). The concurrency gate is run and recorded by the
preparation step (a property of the machine, not a check).
"""
import glob
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
from lib import memory, timing, train  # noqa: E402
from lib.arms import SERIES_W256, get_arm  # noqa: E402
from lib.count import count_arm  # noqa: E402
from lib.data import Enwik8, balanced_byte_hash  # noqa: E402
from lib.model import SparseAttnLM  # noqa: E402

TRAIN = Path(__file__).resolve().parents[1]
FAILS, RESULTS = [], []
# tolerances (stated in prep3/README.md): float32 rounding of the same computation in another order
TOL_LOGITS, TOL_GRAD = 1e-4, 1e-5


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def check_gather_parity():
    data = Enwik8()
    ht = balanced_byte_hash(data.train_counts(), 4)
    for name in ("E4a_L2", "E4a_L4", "IND_L4", "E4a_L2_N512", "E4a_L4_N512", "IND_L4_N512", "E4a_L4_W256", "E4a_L4_W256_h16"):
        cfg = get_arm(name)
        x, y = data.eval_windows("valid", cfg.context, max(2, 2048 // cfg.context))
        outs = {}
        for impl in ("selected", "gather"):
            m = SparseAttnLM(cfg, 3, hash_table=ht, impl=impl)
            logits, _ = m(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            loss.backward()
            outs[impl] = (logits.detach(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None})
        dl = float((outs["selected"][0] - outs["gather"][0]).abs().max())
        dg = max(float((outs["selected"][1][n] - outs["gather"][1][n]).abs().max()) for n in outs["selected"][1])
        same_params = set(outs["selected"][1]) == set(outs["gather"][1])
        check(f"gather (packed hash form) == selected: logits and every parameter gradient, {name}",
              dl <= TOL_LOGITS and dg <= TOL_GRAD and same_params, f"max|logit diff| {dl:.1e}, max|grad diff| {dg:.1e}")


def check_w256_counts():
    plan = {  # PLAN 15.5: inference MACs, parameters outside the embedding, KV values, training activations
        "E4a_L2_W256": (552960, 642816, 1024, 3328), "REF1_W256": (696320, 370688, 512, 6656),
        "E4a_L4_W256": (1040384, 1285376, 2048, 6656), "E4a_L4_W256_h16": (1040384, 1285376, 2048, 10752),
        "REF2_W256": (1327104, 741120, 1024, 13312), "REF2_W256_h16": (1327104, 741120, 1024, 21504)}
    for name, exp in plan.items():
        c = count_arm(name)
        got = (c["inference_macs_per_token"], c["params_outside_embedding"], c["kv_cache_values_per_token"],
               c["training_activation_values_per_token"])
        check(f"counter {name} equal to the table of PLAN 15.5", got == exp, f"{got}")
    check("the two head layouts of a family have equal parameters and multiply-adds",
          all(count_arm(a)["params_outside_embedding"] == count_arm(b)["params_outside_embedding"]
              and count_arm(a)["inference_macs_per_token"] == count_arm(b)["inference_macs_per_token"]
              for a, b in (("REF2_W256", "REF2_W256_h16"), ("E4a_L4_W256", "E4a_L4_W256_h16"))))
    check("arms of width 256: 8 heads of 32 and 16 heads of 16, N = 512, H = 1024, degree 52",
          all(a.d_model == 256 and a.hidden == 1024 and a.degree == 52 and a.context == 512 for a in SERIES_W256)
          and get_arm("REF2_W256").d_head == 32 and get_arm("REF2_W256_h16").d_head == 16)
    # the supports of the constrained arm at width: degree 52, inside S_g
    m = SparseAttnLM(get_arm("E4a_L4_W256"), 0, hash_table=torch.arange(256) % 4)
    cfg = m.cfg
    ok = True
    for layer in m.layers:
        um, dm = layer.mask_w_up, layer.mask_w_down
        ok &= bool((um.sum(1) == 52).all()) and bool((dm.sum(0) == 52).all())
        for g in range(4):
            r = um[g * cfg.n:(g + 1) * cfg.n]
            ok &= bool(r[:, :g * cfg.m].sum() == 0 and r[:, (g + 1) * cfg.m:].sum() == 0)
    check("width 256: degree 52 per neuron and up_idx of R_g inside S_g", ok)


def check_memory_harness():
    # (a) saved tensors: x0 (data), e = x0 @ E, h = e @ W1, a = relu(h), out = a @ W2 -> saved: x0, e, W1, a, W2
    n, d, k = 64, 32, 16
    E, W1, W2 = (torch.nn.Parameter(torch.randn(k, d)), torch.nn.Parameter(torch.randn(d, d)), torch.nn.Parameter(torch.randn(d, 4)))
    x0 = torch.randn(n, k)

    class Syn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.E, self.W1, self.W2 = E, W1, W2

    syn = Syn()

    def fwd():
        return (F.relu((x0 @ E) @ W1) @ W2).sum()

    res = memory.measure_saved(syn, fwd, extra_weight_storages={x0.untyped_storage().data_ptr()} if False else ())
    f4 = 4
    expected_total = f4 * (n * k + n * d + d * d + n * d + d * 4)   # x0, e, W1, a, W2 (a counted once)
    expected_weight = f4 * (d * d + d * 4)                            # W1, W2
    check("memory (a): saved bytes of a synthetic module equal the hand count (distinct storages, relu result shared)",
          res["saved_total_bytes"] == expected_total and res["saved_weight_side_bytes"] == expected_weight,
          f"{res['saved_total_bytes']} vs {expected_total}; weight-side {res['saved_weight_side_bytes']} vs {expected_weight}")
    # (b)/(c) peak live bytes: allocate A, allocate B, free A, allocate C, free all -> peak = max(A + B, B + C)
    MiB = 1024 * 1024
    held = {}

    def alloc_pattern():
        a = torch.empty(4 * MiB // 4)
        b = torch.empty(2 * MiB // 4)
        del a
        c = torch.empty(3 * MiB // 4)
        held["x"] = (b, c)
        del b, c
        held.clear()

    pk = memory.measure_peak(alloc_pattern)
    check("memory (b)/(c): peak live bytes of a synthetic allocation pattern equal the hand value (A + B = 6 MiB)",
          pk["peak_live_bytes"] == 6 * MiB and pk["live_at_end_bytes"] == 0, f"{pk}")
    # two fresh processes give identical (a) and (b), (c) within 1 %
    outs = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(2):
            f = Path(tmp) / f"m{i}.json"
            subprocess.run([sys.executable, "-m", "lib.memory", "--arm", "E4a_L2", "--impl", "gather", "--out", str(f)],
                           cwd=TRAIN, check=True, capture_output=True)
            outs.append(json.loads(f.read_text()))
    a, b = outs
    rel = lambda u, v: abs(u - v) / max(u, 1)  # noqa: E731
    check("memory harness: two fresh processes give identical (a) and (b), (c) within 1 %",
          a["saved"] == b["saved"] and rel(a["train_step"]["peak_live_bytes"], b["train_step"]["peak_live_bytes"]) <= 0.01
          and rel(a["prefill"]["peak_live_bytes"], b["prefill"]["peak_live_bytes"]) <= 0.01,
          f"saved {a['saved']['saved_total_bytes']} / {b['saved']['saved_total_bytes']}, train peak {a['train_step']['peak_live_bytes']} / "
          f"{b['train_step']['peak_live_bytes']}, prefill {a['prefill']['peak_live_bytes']} / {b['prefill']['peak_live_bytes']}")


def check_executor_parity():
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-m", "lib.executor", "--parity"], cwd=TRAIN, env=env, capture_output=True, text=True)
    last = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    if r.returncode != 0 or not last:
        check("executor parity ran", False, r.stderr[-500:])
        return
    res = json.loads(last[-1])
    for name, diff in res.items():
        check(f"executor == sequential forward within float32 rounding, {name}", diff <= TOL_LOGITS, f"max|diff| {diff:.1e}")


def check_timing_harness():
    """Synthetic arms: the variant sleeps 1.5 times the control, the copy as the control. The ratios come from raw.jsonl only."""
    def sleeper(sec):
        def f():
            t = time.perf_counter()
            while time.perf_counter() - t < sec:
                pass
        return f
    with tempfile.TemporaryDirectory() as tmp:
        arms = {"control": sleeper(0.004), "copy": sleeper(0.004), "variant": sleeper(0.006)}
        timing.run_windows(arms, tmp, windows=4, warm_calls=5, warm_seconds=0.1, slot_seconds=0.05, cell="syn", process=0)
        lines = [json.loads(ln) for ln in open(Path(tmp) / "raw.jsonl")]
        slots = [ln for ln in lines if "slot" in ln]
        order_ok = all([s["arm"] for s in slots if s["window"] == w] == list(timing.ORDER) for w in range(4))
        res = timing.ratios_from_raw(Path(tmp) / "raw.jsonl")
        reps_ok = all(s["seconds"] >= 0.05 * 0.9 for s in slots)
        stat_ok = sum(1 for ln in lines if "stat_delta_jiffies" in ln) == 4 and (Path(tmp) / "meta.json").exists()
        meta = json.loads((Path(tmp) / "meta.json").read_text())
    check("timing harness: 12-slot protocol per window is control, copy, variant, variant, copy, control; every slot lasts >= 0.2 s (0.05 s here)",
          order_ok and reps_ok)
    check("timing harness: ratios from raw.jsonl only recover a known variant/control = 1.5 (within 3 %) and copy/control = 1 (within 3 %)",
          abs(res["r_variant"] - 1.5) < 0.045 and abs(res["r_copy"] - 1.0) < 0.03,
          f"r_variant {res['r_variant']:.3f}, r_copy {res['r_copy']:.3f}")
    check("timing harness: steal/user time per window, lscpu, torch configuration and load average recorded",
          stat_ok and "lscpu" in meta and "torch_config" in meta and "loadavg_at_start" in meta)


def check_second_series_reproducible():
    ref = sorted(glob.glob(str(TRAIN / "e8" / "runs" / "smoke_*" / "smoke" / "E4a_L4_s0" / "summary.json")))
    if not ref:
        check("second series: recorded smoke of E4a_L4 available", False)
        return
    rec = json.loads(Path(ref[0]).read_text())
    with tempfile.TemporaryDirectory() as tmp:
        rc = train.main(["--arm", "E4a_L4", "--seed", "0", "--steps", "40", "--lr", "0.002", "--out", tmp, "--threads", "1",
                         "--eval-every", "20", "--valid-windows", "256", "--final-windows", "512", "--schedule", "wsd",
                         "--tokens-per-step", "8192", "--valid-bytes", "524288"])
        new = json.loads((Path(tmp) / "summary.json").read_text())
    check("second series reproducible: the default form under the hash route (selected) gives the bits of the E8 smoke of E4a_L4",
          rc == 0 and new["final_valid_bpb"] == rec["final_valid_bpb"], f"{new['final_valid_bpb']!r} vs {rec['final_valid_bpb']!r}")


def main():
    t0 = time.time()
    torch.manual_seed(0)
    for fn in (check_w256_counts, check_gather_parity, check_memory_harness, check_executor_parity, check_timing_harness,
               check_second_series_reproducible):
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
