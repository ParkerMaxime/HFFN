"""Memory harness of E14 (PLAN 17.7): the quantities (a)-(c) of E10 (`lib/memory.py`, unchanged) and (e), the peak of live tensor
bytes of one decoded step (B = 64 sequences, a cache of N / 2 positions), by the method of (c) (memory events of the profiler,
relative to the start of the profiled region). The caches and the model are alive before the region and are not in the peak (the KV
cache is counted, `lib/count.py`); the working buffers of the step are allocated inside the region. The decoded step is the
sequential packed form (`Decoder.step(parallel=False)`, one thread): the same workers as the executor, one after the other.

Run as `python -m lib.memory4 --arm A --out f.json` in fresh processes (two per arm); (a) must be identical, (b), (c), (e) within 1 %.
"""
import argparse
import json
import resource
import sys
from pathlib import Path

import torch

from . import decode as D
from . import memory
from .arms import get_arm
from .data import Enwik8, balanced_byte_hash

B_DECODE = 64


def measure_peak_relative(run):
    """The method of `memory.measure_peak`, with the baseline of the profiler's counter removed. "Total Allocated" is a running total
    that is not reset between two profiled regions of one process (a region that leaves tensors alive, or frees tensors allocated
    in an earlier region, shifts the next one: observed, 3.29 MB after (a)-(c)); the baseline is read at the first event of the
    region (`Total Allocated` minus `Bytes`). Equal to `measure_peak` in a clean process (checked on the synthetic pattern)."""
    import json as _json
    import os
    import tempfile

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=False) as prof:
        run()
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        events = [e for e in _json.load(open(path))["traceEvents"] if e.get("name") == "[memory]" and e["args"].get("Device Type") == 0]
    finally:
        os.unlink(path)
    if not events:
        return {"peak_live_bytes": 0, "live_at_end_bytes": 0, "baseline_bytes": 0}
    first = events[0]["args"]
    base = first["Total Allocated"] - first["Bytes"]
    cur = [e["args"]["Total Allocated"] - base for e in events]
    return {"peak_live_bytes": max(cur), "live_at_end_bytes": cur[-1], "baseline_bytes": base}


def decode_peak(step, bufs):
    """(e): the buffers of the step are released before the region (so that its first call allocates them inside it), then
    `step()` is profiled."""
    bufs.clear()
    return measure_peak_relative(step)


def measure_decode(arm, seed=0):
    cfg = get_arm(arm)
    data = Enwik8()
    ht = balanced_byte_hash(data.train_counts(), cfg.groups)
    windows = D.decode_windows(data, cfg.context, B_DECODE, 0)
    dec, _model, tok, p = D.build_decoder(cfg, ht, windows, seed)
    with torch.no_grad():
        for _ in range(8):  # warm-up: lazy initialisations and the profiling runs of the scripted workers, not measured
            dec.step(tok, p, parallel=False)
        res = decode_peak(lambda: dec.step(tok, p, parallel=False), dec.bufs)
    res.update({"batch": B_DECODE, "cache_positions": p, "group_counts": dec.last_counts,
                "cache_bytes_per_token_counted": 2 * cfg.n_layers * cfg.d_model * 4})
    return res


def measure(arm, seed=0):
    cfg = get_arm(arm)
    impl = "gather" if cfg.routing == "hash" else "packed"
    out = memory.measure_arm(arm, impl, seed)          # (a), (b), (c), (d) of E10
    out["decode"] = measure_decode(arm, seed)           # (e)
    out["per_token"]["decode_peak_per_decoded_token"] = out["decode"]["peak_live_bytes"] / B_DECODE
    out["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    res = measure(a.arm, a.seed)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["per_token"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
