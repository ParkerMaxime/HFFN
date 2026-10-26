"""Memory harness of E10 (PLAN 15.3). Measures, for one arm and one training step of 8,192 bytes (float32):

(a) bytes of the tensors saved for the backward, by distinct storage (`torch.autograd.graph.saved_tensors_hooks`), split
    into weight-side (storages of parameters, buffers and of the masked weights `w * mask` that `Layer.eff` builds: a cost
    of the control representation, reported apart) and token-side (everything else: the activations);
(b) the peak of live tensor bytes allocated during the forward and the backward (memory events of `torch.profiler`:
    "Total Allocated" is relative to the start of the profiled region, so tensors alive before it, the parameters, the
    masks and the batch, are not in it; the masked weights, the activations and the gradients are);
(c) the same peak for one forward without gradient (prefill of the same bytes);
(d) the peak resident memory of the process (descriptive only: allocator caching, masked weights).

Weights, AdamW states and the KV cache are not measured here: they are counted (`lib/count.py`). Deterministic quantities:
the CLI is meant to be run in two fresh processes per arm; (a) must be identical, (b) and (c) within 1 %.
"""
import argparse
import json
import os
import resource
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from .arms import get_arm
from .data import BatchStream, Enwik8, balanced_byte_hash
from .model import Layer, SparseAttnLM


def _storage_key(t):
    s = t.untyped_storage()
    return s.data_ptr(), s.nbytes()


class SavedTensorAccount:
    """Counts the distinct storages saved for the backward and classifies them as weight-side or token-side."""

    def __init__(self, weight_storages=()):
        self.seen = {}
        self.weight = set(weight_storages)

    def pack(self, t):
        try:
            key = _storage_key(t)
        except Exception:  # a tensor without storage (e.g. a fake) saves nothing
            return t
        self.seen[key] = True
        return t

    @staticmethod
    def unpack(t):
        return t

    def totals(self):
        weight = sum(n for (p, n) in self.seen if p in self.weight)
        total = sum(n for (_p, n) in self.seen)
        return {"saved_total_bytes": total, "saved_weight_side_bytes": weight, "saved_token_side_bytes": total - weight,
                "saved_distinct_storages": len(self.seen)}


def model_weight_storages(model):
    out = {p.untyped_storage().data_ptr() for p in model.parameters()}
    out |= {b.untyped_storage().data_ptr() for b in model.buffers()}
    return out


def measure_saved(model, forward, extra_weight_storages=()):
    """(a). `forward()` runs the forward with gradients; no backward is needed to know what is saved."""
    acc = SavedTensorAccount(model_weight_storages(model) | set(extra_weight_storages))
    # the masked weights `w * mask` are built by Layer.eff at every forward: record their storages as weight-side
    orig = Layer.eff

    def eff(self, name):
        w = orig(self, name)
        acc.weight.add(w.untyped_storage().data_ptr())
        return w
    Layer.eff = eff
    try:
        with torch.autograd.graph.saved_tensors_hooks(acc.pack, acc.unpack):
            forward()
    finally:
        Layer.eff = orig
    return acc.totals()


def measure_peak(run):
    """(b) / (c): peak of live tensor bytes allocated inside `run()`, from the memory events of the profiler."""
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=False) as prof:
        run()
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        events = json.load(open(path))["traceEvents"]
    finally:
        os.unlink(path)
    peak, last = 0, 0
    for e in events:
        if e.get("name") == "[memory]" and e["args"].get("Device Type") == 0:
            cur = e["args"]["Total Allocated"]
            peak, last = max(peak, cur), cur
    return {"peak_live_bytes": peak, "live_at_end_bytes": last}


def measure_arm(arm: str, impl: str, seed: int = 0, tokens_per_step: int = 8192):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    cfg = get_arm(arm)
    data = Enwik8()
    ht = balanced_byte_hash(data.train_counts(), cfg.groups) if cfg.routing == "hash" else None
    model = SparseAttnLM(cfg, seed, hash_table=ht, impl=impl)
    batch = tokens_per_step // cfg.context
    x, y = BatchStream(data, seed, batch, cfg.context).next()  # real bytes of the train split: the route depends on them

    def fwd_loss():
        logits, _ = model(x)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

    def fwd_bwd():
        model.zero_grad(set_to_none=True)
        fwd_loss().backward()

    out = {"arm": arm, "impl": impl, "seed": seed, "tokens_per_step": batch * cfg.context, "context": cfg.context}
    # (c) first, in a clean allocator state, then (a), then (b)
    with torch.no_grad():
        model(x)  # warm-up of lazy initialisations (rope slices, kernels), not measured
    with torch.no_grad():
        out["prefill"] = measure_peak(lambda: model(x))
    out["saved"] = measure_saved(model, fwd_loss)
    model.zero_grad(set_to_none=True)
    fwd_bwd()
    model.zero_grad(set_to_none=True)
    out["train_step"] = measure_peak(fwd_bwd)
    out["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    out["per_token"] = {
        "saved_total": out["saved"]["saved_total_bytes"] / out["tokens_per_step"],
        "saved_token_side": out["saved"]["saved_token_side_bytes"] / out["tokens_per_step"],
        "train_peak": out["train_step"]["peak_live_bytes"] / out["tokens_per_step"],
        "prefill_peak": out["prefill"]["peak_live_bytes"] / out["tokens_per_step"],
    }
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--impl", default="packed", choices=["packed", "selected", "gather"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    res = measure_arm(a.arm, a.impl, a.seed)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["per_token"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
