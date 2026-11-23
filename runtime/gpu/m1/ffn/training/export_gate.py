"""Every-export in-process gate of the retained native context.

For one export and one N: `Context` created from the export (shared storage, K = 1), one step on seeded x / gy, then
the context's views against a float64 computation of the same step, banded by the dense float32 control. The float64
reference and the dense control build dense matrices from the support: they are controls of this gate, never an HFFN
path. Untimed: only the peak resident size and the wall time of a cell are recorded, as sizing figures.
"""
import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
LEAF = HERE
sys.path.insert(0, str(LEAF))
from export_input import checked, read_export          # noqa: E402
from native_context import Context, prepare_exact_sources   # noqa: E402

LIBRARY = LEAF / "build/libhffn_train.dylib"
# Band constants, written before the first smoke (README, Gates). Not changed after any result.
BAND, FLOOR, CEIL = 10.0, 4 * 2.0 ** -23, 1e-4
GROUPS = ("up", "down", "bup", "bdown")
# Hyperparameters as the Metal build stores them (float32), native_train_context.mm:92.
LR, B1, B2, EPS, WD = (float(np.float32(v)) for v in (1e-3, .9, .999, 1e-8, .01))
STEP_SIZE = float(np.float32(LR / (1 - B1)))
SCALE = float(np.float32(np.sqrt(1 - B2)))
SEED = 2710
KEYS = ("up_ptr", "down_ptr", "up_idx", "down_idx", "up_val", "down_val", "b_up", "b_down")


def digest(payload):
    h = hashlib.sha256()
    for k in KEYS:
        h.update(payload[k].contiguous().numpy().tobytes())
    return h.hexdigest()


def rows_of(ptr):
    return torch.repeat_interleave(torch.arange(ptr.numel() - 1), ptr.long().diff())


def reference(payload, x, gy, dtype):
    """Dense step by autograd in `dtype`: outputs and compact gradients."""
    i, h, o, _, _ = checked(payload)
    ru, rd = rows_of(payload["up_ptr"]), rows_of(payload["down_ptr"])
    iu, idn = payload["up_idx"].long(), payload["down_idx"].long()
    wu = torch.zeros(h, i, dtype=dtype); wu[ru, iu] = payload["up_val"].to(dtype); wu.requires_grad_()
    wd = torch.zeros(h, o, dtype=dtype); wd[rd, idn] = payload["down_val"].to(dtype); wd.requires_grad_()
    bu = payload["b_up"].to(dtype).clone().requires_grad_()
    bd = payload["b_down"].to(dtype).clone().requires_grad_()
    xx = x.to(dtype).clone().requires_grad_()
    y = F.gelu(xx @ wu.T + bu) @ wd + bd
    y.backward(gy.to(dtype))
    return {"y": y.detach(), "gx": xx.grad, "up": wu.grad[ru, iu], "down": wd.grad[rd, idn], "bup": bu.grad, "bdown": bd.grad}


def adam(p0, g, dtype):
    """First AdamW step from zero moments; float32 arithmetic for the dense control, float64 for the reference."""
    c = (lambda v: np.float32(v)) if dtype == np.float32 else (lambda v: np.float64(v))
    p0, g = p0.astype(dtype), g.astype(dtype)
    m, v = (c(1) - c(B1)) * g, (c(1) - c(B2)) * g * g
    return p0 * (c(1) - c(LR) * c(WD)) - c(STEP_SIZE) * m / (np.sqrt(v) / c(SCALE) + c(EPS)), m, v


def dists(out, ref64, p0, moments_m, weights, moments_v):
    """Distances of a candidate (outputs, gradients, updated weights, moments) to the float64 reference."""
    d = {}
    for k in ("y", "gx"):
        d[k] = float(np.abs(out[k].astype(np.float64) - ref64[k].numpy()).max() / np.abs(ref64[k].numpy()).max())
    for g in GROUPS:
        gr = ref64[g].numpy()
        d["g_" + g] = float(np.abs(moments_m[g].astype(np.float64) / (1 - B1) - gr).max() / np.abs(gr).max())
        rp = adam(p0[g], gr, np.float64)[0]
        cond = np.abs(gr) >= .01 * np.abs(gr).max()
        assert cond.any(), g
        err = np.abs(weights[g].astype(np.float64) - rp)[cond]
        d["w_" + g] = float((err / np.maximum.reduce([np.abs(rp)[cond], np.abs(p0[g].astype(np.float64))[cond],
                                                      np.full(cond.sum(), LR)])).max())
        d["v_" + g] = float(np.abs(moments_v[g].astype(np.float64) - (1 - B2) * gr * gr).max() / ((1 - B2) * (gr * gr).max()))
    return d


def f32(buffer):
    return np.frombuffer(buffer, np.float32)


def run_cell(export, n, fault=None):
    started = time.time()
    payload = read_export(export)
    before = digest(payload)
    i, h, o, ue, de = checked(payload)
    g = torch.Generator().manual_seed(SEED + n)
    x, gy = torch.randn(n, i, generator=g), torch.randn(n, o, generator=g)
    ref64, dense32 = reference(payload, x, gy, torch.float64), reference(payload, x, gy, torch.float32)
    p0 = {k: payload[v].detach().numpy().copy() for k, v in zip(GROUPS, ("up_val", "down_val", "b_up", "b_down"))}
    built = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in payload.items()}
    if fault == "index":      # one admitted up index moved to a feature outside the row of connections
        row = set(built["up_idx"][:int(built["up_ptr"][1])].tolist())
        free = next((j for j in range(i) if j not in row), None)
        if free is not None:
            built["up_idx"][0] = free
        else:                 # a full row (density 1.0) has no free feature: swap two indices, values stay in place
            built["up_idx"][0], built["up_idx"][1] = int(built["up_idx"][1]), int(built["up_idx"][0])
    elif fault == "value":    # one admitted down value scaled
        built["down_val"][0] *= 1.5
    built_before = digest(built)    # the support actually handed to the context (after a construction fault, if any)
    # The native context is feature-major: x[i * N + n], gy[o * N + n], and y / gx the same (native_train_context.mm:597, :757).
    xt, gyt = x.T.contiguous().numpy().tobytes(), gy.T.contiguous().numpy().tobytes()
    args = ["--n", n, "--accum-steps", 1, "--shared-readonly"]
    guards = {}
    with Context(LIBRARY, prepare_exact_sources(built, args), args, support=built) as c:
        pre = {k: c.snapshot(k) for k in GROUPS}
        guards["admitted_values_equal_export"] = all(pre[k] == p0[k].tobytes() for k in GROUPS) if fault is None else True
        guards["moments_zero"] = all(not f32(c.snapshot(p + k)).any() for p in ("m_", "v_") for k in GROUPS)
        c.write_input("x", xt); c.write_input("gy", gyt)
        if fault == "support_write":    # the support handed to the context is written while it is held: the guard must see it
            built["up_idx"][0] = (int(built["up_idx"][0]) + 1) % i
        c.step()
        snap = {k: c.snapshot(k) for k in ("y", "gx", "z", "h", "gz", *GROUPS, *(p + k for p in ("m_", "v_", "g_") for k in GROUPS))}
        counts = (ue, de, h, o)
        guards["extents"] = all(f32(snap[k]).size == m for k, m in zip(GROUPS, counts)) and f32(snap["y"]).size == n * o
        guards["finite"] = all(np.isfinite(f32(v)).all() for v in snap.values())
        guards["gradient_cleared"] = all(not f32(snap["g_" + k]).any() for k in GROUPS)
        c.reset(b"".join(pre[k] for k in GROUPS))
        c.write_input("x", xt); c.write_input("gy", gyt)
        c.step()
        again = {k: c.snapshot(k) for k in ("y", "gx", *GROUPS, *(p + k for p in ("m_", "v_") for k in GROUPS))}
        guards["repeat_bitwise"] = all(snap[k] == again[k] for k in again)
    guards["support_unchanged"] = digest(payload) == before and digest(built) == built_before
    out = {k: f32(snap[k]).reshape(-1, n).T for k in ("y", "gx")}
    hff = dists(out, ref64, p0, {k: f32(snap["m_" + k]) for k in GROUPS},
                {k: f32(snap[k]) for k in GROUPS}, {k: f32(snap["v_" + k]) for k in GROUPS})
    dw, dm, dv = {}, {}, {}
    for k in GROUPS:
        dw[k], dm[k], dv[k] = adam(p0[k], dense32[k].numpy(), np.float32)
    den = dists({k: dense32[k].numpy() for k in ("y", "gx")}, ref64, p0, {k: dm[k] for k in GROUPS},
                dw, {k: dv[k] for k in GROUPS})
    groups, passed, admissible = {}, True, True
    for k, d in hff.items():
        band = min(max(BAND * den[k], FLOOR), CEIL)
        groups[k] = {"hffn": d, "dense": den[k], "band": band, "passed": bool(d <= band)}
        admissible = admissible and den[k] <= CEIL
        passed = passed and d <= band
    passed = passed and all(guards.values())
    return {"export": Path(export).parent.name, "n": n, "fault": fault, "passed": bool(passed and admissible),
            "admissible": bool(admissible), "guards": guards, "groups": groups,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, "wall_s": time.time() - started}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--exports", required=True, help="comma separated folders of $HFFN_ASSETS/homogeneous/models")
    a.add_argument("--n", default="1,8,64,512")
    a.add_argument("--faults", action="store_true")
    a.add_argument("--out", required=True)
    a = a.parse_args()
    if not LIBRARY.exists():
        sys.exit("missing " + str(LIBRARY) + " (build_train_step_runner.sh of the leaf)")
    if not os.environ.get("HFFN_ASSETS"):
        sys.exit("HFFN_ASSETS is not set: point it at the folder that holds homogeneous/models")
    assets = Path(os.environ["HFFN_ASSETS"]) / "homogeneous/models"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    records, bad = [], 0
    for folder in a.exports.split(","):
        for n in map(int, a.n.split(",")):
            for fault in (None, *(("index", "value", "support_write") if a.faults else ())):
                r = run_cell(assets / folder / "export", n, fault)
                # A fault cell must fail the gate; the support_write cell must fail exactly the support guard.
                ok = r["passed"] if fault is None else (not r["guards"]["support_unchanged"] if fault == "support_write"
                                                       else not r["passed"])
                r["expected_outcome_met"] = ok
                bad += not ok
                records.append(r)
                print(f"{folder} N={n} fault={fault} passed={r['passed']} expected_met={ok} rss={r['peak_rss_mb']:.0f}MB "
                      f"wall={r['wall_s']:.1f}s", flush=True)
                (out / "records.json").write_text(json.dumps(records, indent=1, allow_nan=False))
    print(f"cells={len(records)} unmet={bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
