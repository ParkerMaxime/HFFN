"""Timing harness of E11 (PLAN 15.3, the timing rule; AGENTS.md section 7).

One *cell-process* = one fresh process for one cell. In it three arms are built independently: the control, the copy (a second
instance of the control with the same weights) and the variant. Each arm is warmed up (at least `warm_calls` calls and
`warm_seconds` seconds), then timed in `windows` windows. A window is one block `control, copy, variant, variant, copy, control`;
each slot repeats its arm's call so that the slot lasts at least `slot_seconds`. Per window the ratios variant / control and copy /
control are formed on the per-call times of the two slots of each arm; per process, the median over windows. The steal and user
time (`/proc/stat`) of every window, the load average at the start, `lscpu` and the torch configuration are recorded; nothing is
excluded after the fact. `raw.jsonl` has one line per slot (its seconds are raw measurements; they are never reported as results);
the ratios are computed from it only (`ratios_from_raw`).

Cells (variant / control), 8,192 tokens per call, forward only, without gradient. The parallel executor is `lib/executor.py`
(TorchScript fork/wait, 4 workers, one intra-op thread each); the process must start with OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1
(a thread started by the runtime would otherwise open its own team of OpenMP threads).
"""
import argparse
import json
import math
import os
import platform
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ORDER = ("control", "copy", "variant", "variant", "copy", "control")

CELLS = {
    "a1": {"level": "layer", "M": 128, "N": 128, "variant": "e4a", "control": "dense"},
    "a2": {"level": "layer", "M": 128, "N": 512, "variant": "e4a", "control": "dense"},
    "a3": {"level": "layer", "M": 512, "N": 512, "variant": "e4a", "control": "dense"},
    "a4": {"level": "layer", "M": 512, "N": 2048, "variant": "e4a", "control": "dense"},
    "b1": {"level": "model", "variant": "E4a_L4", "control": "REF2"},
    "b2": {"level": "model", "variant": "E4a_L4_N512", "control": "REF2_N512"},
    "c1": {"level": "model", "variant": "IND_L4", "control": "E4a_L4"},
    "c2": {"level": "model", "variant": "IND_L4_N512", "control": "E4a_L4_N512"},
}


# --------------------------------------------------------------------------- measurement of the machine
def read_stat():
    """Jiffies of the aggregate cpu line of /proc/stat: user, nice, system, idle, iowait, irq, softirq, steal."""
    with open("/proc/stat") as f:
        parts = f.readline().split()[1:9]
    return [int(v) for v in parts]


def machine_meta():
    import torch
    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    return {"lscpu": lscpu, "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
            "torch_config": torch.__config__.show(), "torch_parallel_info": torch.__config__.parallel_info(),
            "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "MALLOC_ARENA_MAX")},
            "loadavg_at_start": open("/proc/loadavg").read().strip(), "cpu_count": os.cpu_count()}


# --------------------------------------------------------------------------- the windows
def run_windows(arms, out_dir, windows=12, warm_calls=30, warm_seconds=5.0, slot_seconds=0.2, cell="x", process=0):
    """`arms` = {"control": f, "copy": f, "variant": f} callables with no argument. Writes raw.jsonl and meta.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = machine_meta()
    meta.update({"cell": cell, "process": process, "windows": windows, "warm_calls": warm_calls, "warm_seconds": warm_seconds,
                 "slot_seconds": slot_seconds, "order": list(ORDER)})
    reps = {}
    for name in ("control", "copy", "variant"):  # each arm is warmed up before its first timed call
        fn = arms[name]
        t0 = time.perf_counter()
        calls, last = 0, []
        while calls < warm_calls or time.perf_counter() - t0 < warm_seconds:
            t = time.perf_counter()
            fn()
            last.append(time.perf_counter() - t)
            calls += 1
        per_call = st.median(last[-10:])
        reps[name] = max(1, math.ceil(slot_seconds / per_call))
        meta[f"warm_calls_done_{name}"] = calls
    meta["reps"] = reps
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    with open(out / "raw.jsonl", "w", buffering=1) as raw:
        for w in range(windows):
            s0 = read_stat()
            for slot, name in enumerate(ORDER):
                fn, r = arms[name], reps[name]
                t = time.perf_counter()
                for _ in range(r):
                    fn()
                dt = time.perf_counter() - t
                raw.write(json.dumps({"cell": cell, "process": process, "window": w, "slot": slot, "arm": name, "reps": r,
                                      "seconds": dt}) + "\n")
            s1 = read_stat()
            raw.write(json.dumps({"cell": cell, "process": process, "window": w, "stat_delta_jiffies":
                                  dict(zip(("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal"),
                                           [b - a for a, b in zip(s0, s1)]))}) + "\n")
    return reps


def ratios_from_raw(raw_path):
    """Per window and per process, from the raw slots only: r_variant = t(variant) / t(control), r_copy = t(copy) / t(control),
    with t(arm) the per-call time on the sums over the two slots of the arm."""
    slots, stats = {}, {}
    for line in open(raw_path):
        e = json.loads(line)
        if "slot" in e:
            slots.setdefault(e["window"], {}).setdefault(e["arm"], []).append(e)
        else:
            stats[e["window"]] = e["stat_delta_jiffies"]
    per_window = []
    for w in sorted(slots):
        t = {a: sum(s["seconds"] for s in v) / sum(s["reps"] for s in v) for a, v in slots[w].items()}
        per_window.append({"window": w, "r_variant": t["variant"] / t["control"], "r_copy": t["copy"] / t["control"],
                           "steal_jiffies": stats.get(w, {}).get("steal"), "user_jiffies": stats.get(w, {}).get("user"),
                           "total_jiffies": sum(stats.get(w, {}).values()) if stats.get(w) else None})
    return {"windows": per_window, "r_variant": st.median(x["r_variant"] for x in per_window),
            "r_copy": st.median(x["r_copy"] for x in per_window)}


# --------------------------------------------------------------------------- the cells
def _hash_table(groups=4):
    from .data import Enwik8, balanced_byte_hash
    return balanced_byte_hash(Enwik8().train_counts(), groups), Enwik8()


def layer_cfgs(M, N):
    from .arms import ArmConfig
    base = dict(d_model=M, hidden=4 * M, degree=13 * M // 64, context=N, n_layers=1)
    dense = ArmConfig("cell_dense", **base)
    e4a = ArmConfig("cell_e4a", ffn_support="blocks", ffn_routed=True, wo_blockdiag=True, attn="routed", routing="hash", **base)
    return dense, e4a


def _kind(cfg):
    if cfg.attn == "dense":
        return "dense"
    return "ind" if cfg.qkv_blockdiag else "e4a"


def build_arm(cfg, kind, level, tokens, ht, dense_exec, seed=0):
    """Returns a callable with no argument for one arm (a fresh instance: weights from `seed`, own buffers)."""
    import torch
    from .executor import Executor
    from .model import SparseAttnLM
    model = SparseAttnLM(cfg, seed, hash_table=ht, impl="packed").eval()
    if kind == "dense" and dense_exec == "intraop":
        def call_model():
            torch.set_num_threads(4)
            try:
                model(tokens)
            finally:
                torch.set_num_threads(1)
        if level == "model":
            return call_model
        x = model.emb(tokens).detach()
        layer = model.layers[0]

        def call_layer():
            torch.set_num_threads(4)
            try:
                layer(x, tokens, model.cos[:tokens.shape[1]], model.sin[:tokens.shape[1]], model.hash_table, "packed")
            finally:
                torch.set_num_threads(1)
        return call_layer
    ex = Executor(model, kind)
    if level == "model":
        return lambda: ex.forward(tokens)
    x, xi = ex.embed(tokens)
    return lambda: ex.layers_forward(x, xi, ex.plan(tokens) if kind != "dense" else None)


def build_cell(cell, dense_exec="executor", seed=0):
    """Three independent callables (control, copy, variant) for a cell, and a description."""
    import torch
    from .arms import get_arm
    spec = CELLS[cell]
    ht, data = _hash_table()
    if spec["level"] == "layer":
        dense, e4a = layer_cfgs(spec["M"], spec["N"])
        c_cfg, v_cfg = dense, e4a
        N = spec["N"]
    else:
        c_cfg, v_cfg = get_arm(spec["control"]), get_arm(spec["variant"])
        N = c_cfg.context
    x, _ = data.eval_windows("valid", N, 8192 // N)
    tokens = x
    arms = {
        "control": build_arm(c_cfg, _kind(c_cfg), spec["level"], tokens, ht, dense_exec, seed),
        "copy": build_arm(c_cfg, _kind(c_cfg), spec["level"], tokens, ht, dense_exec, seed),
        "variant": build_arm(v_cfg, _kind(v_cfg), spec["level"], tokens, ht, dense_exec, seed),
    }
    return arms, {"cell": cell, **spec, "control_kind": _kind(c_cfg), "variant_kind": _kind(v_cfg), "dense_exec": dense_exec,
                  "tokens": int(tokens.numel()), "context": N}


def phase_pass(cell, dense_exec="executor", reps=15):
    """Untimed descriptive pass: the time of the phases of the variant's first layer, worker by worker, single-threaded,
    and the time of the whole fork-join in place. Barrier + launch overhead = join time - max over workers of their phases."""
    import torch
    from . import executor as E
    from .arms import get_arm
    spec = CELLS[cell]
    ht, data = _hash_table()
    if spec["level"] == "layer":
        dense, e4a = layer_cfgs(spec["M"], spec["N"])
        v_cfg, N = e4a, spec["N"]
    else:
        v_cfg = get_arm(spec["variant"])
        N = v_cfg.context
    tokens, _ = data.eval_windows("valid", N, 8192 // N)
    from .model import SparseAttnLM
    kind = _kind(v_cfg)
    ex = E.Executor(SparseAttnLM(v_cfg, 0, hash_table=ht, impl="packed").eval(), kind)
    x, xi = ex.embed(tokens)
    G, cfg, w = ex.G, ex.cfg, ex.layers[0]
    hpg, dh, m, n = cfg.n_heads // G, cfg.d_head, cfg.m, cfg.n
    cos, sin = ex.cos[:N], ex.sin[:N]
    final = len(ex.layers) == 1

    def med(f):
        for _ in range(3):
            f()
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            f()
            ts.append(time.perf_counter() - t)
        return st.median(ts) * 1000

    out = {"cell": cell, "variant_kind": kind, "unit": "ms (descriptive, single call, first layer)"}
    with torch.no_grad():
        if kind == "dense":
            a = torch.empty_like(x)
            x_out, xi_out = torch.empty_like(x), torch.empty_like(x)
            out["heads_ms"] = [med(lambda g=g: E.dense_heads_worker(xi, a, w["wq"], w["wk"], w["wv"], cos, sin, g, hpg, dh, m)) for g in range(G)]
            out["tail_ms"] = [med(lambda g=g: E.dense_tail_worker(a, x, x_out, xi_out, w["wo"], w["w_up"], w["b_up"], w["w_down"],
                                                                    w["g2"], w["b_down"], w["g1"], g, G, final)) for g in range(G)]
            out["join_heads_ms"] = med(lambda: E.run_dense_heads(xi, a, w["wq"], w["wk"], w["wv"], cos, sin, G, hpg, dh, m))
            out["join_tail_ms"] = med(lambda: E.run_dense_tail(a, x, x_out, xi_out, w["wo"], w["w_up"], w["b_up"], w["w_down"],
                                                                 w["g2"], w["b_down"], w["g1"], G, final))
            out["wait_ms"] = out["join_heads_ms"] + out["join_tail_ms"] - max(out["heads_ms"]) - max(out["tail_ms"])
        else:
            plan = ex.plan(tokens)
            ind = kind == "ind"
            x_out, xi_out = torch.empty_like(x), torch.empty_like(x)
            kv = {g: E.kv_phase(xi, w["wk"], w["wv"], cos, sin, g, hpg, dh, m, ind) for g in range(G)}
            out["kv_ms"] = [med(lambda g=g: E.kv_phase(xi, w["wk"], w["wv"], cos, sin, g, hpg, dh, m, ind)) for g in range(G)]
            out["routed_ms"] = [med(lambda g=g: E.routed_phase(xi, x, x_out, xi_out, kv[g][0], kv[g][1], w["wq"], w["wo"], w["w_up"],
                                                                w["b_up"], w["w_down"], w["g2"], w["b_down"], w["g1"], cos, sin,
                                                                plan[g][0], plan[g][1], plan[g][2], g, hpg, dh, m, n, ind,
                                                                cfg.ffn_support == "ind", final)) for g in range(G)]
            out["tokens_per_group"] = [int(p[1].sum()) for p in plan]
            out["join_ms"] = med(lambda: E.run_constrained(xi, x, x_out, xi_out, w["wq"], w["wk"], w["wv"], w["wo"], w["w_up"],
                                                            w["b_up"], w["w_down"], w["g2"], w["b_down"], w["g1"], cos, sin,
                                                            [p[0] for p in plan], [p[1] for p in plan], [p[2] for p in plan],
                                                            hpg, dh, m, n, ind, cfg.ffn_support == "ind", final))
            out["plan_ms"] = med(lambda: ex.plan(tokens))
            out["wait_ms"] = out["join_ms"] - max(k + r for k, r in zip(out["kv_ms"], out["routed_ms"]))
    return out


# --------------------------------------------------------------------------- one cell-process
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True)
    p.add_argument("--dense-exec", default="executor", choices=["executor", "intraop"])
    p.add_argument("--out", required=True)
    p.add_argument("--process", type=int, default=0)
    p.add_argument("--windows", type=int, default=12)
    p.add_argument("--warm-calls", type=int, default=30)
    p.add_argument("--warm-seconds", type=float, default=5.0)
    p.add_argument("--slot-seconds", type=float, default=0.2)
    p.add_argument("--phases", action="store_true")
    a = p.parse_args(argv)
    import torch
    if os.environ.get("OMP_NUM_THREADS") != "1" or os.environ.get("MKL_NUM_THREADS") != "1":
        raise SystemExit("start the process with OMP_NUM_THREADS=1 MKL_NUM_THREADS=1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(4)
    torch.set_grad_enabled(False)
    arms, desc = build_cell(a.cell, a.dense_exec)
    out = Path(a.out)
    run_windows(arms, out, a.windows, a.warm_calls, a.warm_seconds, a.slot_seconds, a.cell, a.process)
    (out / "cell.json").write_text(json.dumps(desc, indent=1))
    res = ratios_from_raw(out / "raw.jsonl")
    (out / "ratios.json").write_text(json.dumps(res, indent=1))
    if a.phases:
        (out / "phases.json").write_text(json.dumps(phase_pass(a.cell, a.dense_exec), indent=1))
    print(json.dumps({"cell": a.cell, "process": a.process, "r_variant": res["r_variant"], "r_copy": res["r_copy"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
