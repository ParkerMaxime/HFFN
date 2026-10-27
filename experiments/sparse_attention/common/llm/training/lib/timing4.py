"""Timing cells of E14 (PLAN 17.3 and 17.7): the frontier cells and the pairs, in two modes, by the timing rule of E11.

A cell = (variant, anchor), `t = t(variant) / t(anchor)` in the same process, the copy of the anchor in the same process, the
windows and the ratios of `lib/timing.py` (`run_windows`, `ratios_from_raw`: nothing of the rule changes).
* mode `prefill`: one forward over whole windows, 8,192 tokens per call (executor of E11, `lib/executor.py`);
* mode `decode`: one decoded step of B = 64 sequences, one new token each, with a cache of N / 2 positions (`lib/decode.py`).

Cells: the frontier cells of PLAN 17.3 (anchor `REF2` at M = 128 for N = 128 and N = 512, `REF2_W256` at M = 256) and the pairs p1-p6
of PLAN 17.7 (the attention alone: `E4a_L4` / `E2h_L4`; the routing of the FFN alone: `E2h_L4` / `REF4`). Weights are random (time
does not depend on them); inputs are bytes of the valid split, since the route depends on them. The process must start with
OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from . import timing

B_DECODE = 64

FRONTIER = {
    "n128": {"anchor": "REF2", "suffix": "", "variants": ["REF1", "REF3", "REF4", "E4a_L2", "E4a_L4", "E4a_L6", "IND_L4"]},
    "n512": {"anchor": "REF2_N512", "suffix": "_N512", "variants": ["REF1", "REF3", "REF4", "E4a_L2", "E4a_L4", "E4a_L6", "IND_L4"]},
    "w256": {"anchor": "REF2_W256", "suffix": "", "variants": ["REF1_W256", "REF2_W256_h16", "E4a_L2_W256", "E4a_L4_W256",
                                                              "E4a_L4_W256_h16"]},
}
PAIRS = {
    "p1": ("E4a_L4", "E2h_L4", "the attention alone"),
    "p2": ("E2h_L4", "REF4", "the routing of the FFN alone"),
    "p3": ("E4a_L4_N512", "E2h_L4_N512", "the attention alone"),
    "p4": ("E2h_L4_N512", "REF4_N512", "the routing of the FFN alone"),
    "p5": ("E4a_L4_W256", "E2h_L4_W256", "the attention alone, at width"),
    "p6": ("E2h_L4_W256", "REF4_W256", "the routing of the FFN alone, at width"),
}


def _cells():
    cells = {}
    for camp, spec in FRONTIER.items():
        for v in spec["variants"]:
            name = v + (spec["suffix"] if not v.endswith("W256") and not v.endswith("h16") else "")
            cells[f"{camp}-{v}"] = {"group": "frontier", "campaign": camp, "variant": name, "control": spec["anchor"]}
    for pid, (v, c, what) in PAIRS.items():
        cells[pid] = {"group": "pair", "campaign": pid, "variant": v, "control": c, "isolates": what}
    return cells


CELLS4 = _cells()
CAMPAIGN_OF_PAIR = {"p1": "n128", "p2": "n128", "p3": "n512", "p4": "n512", "p5": "w256", "p6": "w256"}


def cell_arms(cell):
    from .arms import get_arm
    spec = CELLS4[cell]
    return get_arm(spec["variant"]), get_arm(spec["control"])


def build_cell4(cell, mode, process=0, seed=0):
    """Three independent callables (control, copy, variant) and a description. Fresh instances: weights from `seed`, own buffers."""
    from . import decode as D
    from .data import Enwik8, balanced_byte_hash
    v_cfg, c_cfg = cell_arms(cell)
    assert v_cfg.context == c_cfg.context and v_cfg.d_model == c_cfg.d_model
    N = c_cfg.context
    data = Enwik8()
    ht = balanced_byte_hash(data.train_counts(), c_cfg.groups)
    desc = {"cell": cell, **CELLS4[cell], "mode": mode, "context": N, "d_model": c_cfg.d_model,
            "variant_kind": D.kind_of(v_cfg), "control_kind": D.kind_of(c_cfg)}
    arms = {}
    if mode == "prefill":
        tokens, _ = data.eval_windows("valid", N, 8192 // N)
        desc["tokens_per_call"] = int(tokens.numel())
        for name, cfg in (("control", c_cfg), ("copy", c_cfg), ("variant", v_cfg)):
            arms[name] = timing.build_arm(cfg, D.kind_of(cfg), "model", tokens, ht, "executor", seed)
    else:
        windows = D.decode_windows(data, N, B_DECODE, process)
        desc.update({"batch": B_DECODE, "cache_positions": N // 2, "position": N // 2})
        counts = {}
        for name, cfg in (("control", c_cfg), ("copy", c_cfg), ("variant", v_cfg)):
            dec, _model, tok, p = D.build_decoder(cfg, ht, windows, seed)
            arms[name] = (lambda dec=dec, tok=tok, p=p: dec.step(tok, p))
            dec.step(tok, p)
            counts[name] = dec.last_counts
        desc["group_counts_of_the_step"] = counts
    return arms, desc


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True)
    p.add_argument("--mode", required=True, choices=["prefill", "decode"])
    p.add_argument("--out", required=True)
    p.add_argument("--process", type=int, default=0)
    p.add_argument("--windows", type=int, default=12)
    p.add_argument("--warm-calls", type=int, default=30)
    p.add_argument("--warm-seconds", type=float, default=5.0)
    p.add_argument("--slot-seconds", type=float, default=0.2)
    a = p.parse_args(argv)
    import torch
    if os.environ.get("OMP_NUM_THREADS") != "1" or os.environ.get("MKL_NUM_THREADS") != "1":
        raise SystemExit("start the process with OMP_NUM_THREADS=1 MKL_NUM_THREADS=1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(4)
    torch.set_grad_enabled(False)
    arms, desc = build_cell4(a.cell, a.mode, a.process)
    out = Path(a.out)
    reps = timing.run_windows(arms, out, a.windows, a.warm_calls, a.warm_seconds, a.slot_seconds, a.cell, a.process)
    desc["reps"] = reps
    (out / "cell.json").write_text(json.dumps(desc, indent=1))
    res = timing.ratios_from_raw(out / "raw.jsonl")
    (out / "ratios.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"cell": a.cell, "mode": a.mode, "process": a.process, "r_variant": res["r_variant"], "r_copy": res["r_copy"]}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
