"""Gates of `check.py` on the real FFN layers of the 70M and 410M exports (gate 5 sets the exit, the other gates are listed apart).

Reuses `gates_for` of the leaf's `check.py` unchanged; only the source of the layer differs: an export layer
(`hffn_pythia.export.read_layer`, hidden-major) is wrapped as the `CompactLayer` the gates expect.
Layers: `--layer` (default the first, a middle and the last), at the N of `--n`. Input seed: PCG64([index + 1000 + seed offset, n]), printed in each JSON line; the seed varies the input stream only, not the export weights.
Usage (VM): python -B check_exports.py [--density 50 70] [--models pythia410m] [--layer -1] [--n 2100]    (HFFN_ASSETS: the assets folder)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

LEAF = Path(__file__).resolve().parent   # the folder of this file: its check.py and train_step.py are the ones checked
sys.path.insert(0, str(LEAF))
import check  # noqa: E402  (the leaf's check.py: puts the shared kernels and test layers on sys.path)
from hffn_ffn_layers import layers  # noqa: E402

REPO = check.REPO
sys.path.insert(0, str(REPO / "runtime/common/pythia"))
from hffn_pythia.export import FORMAT, read_layer  # noqa: E402
ASSETS = Path(os.environ["HFFN_ASSETS"]) if "HFFN_ASSETS" in os.environ else None
if ASSETS is None and "HFFN_EXPORTS" not in os.environ:
    raise SystemExit("HFFN_ASSETS (or HFFN_EXPORTS) must name the assets folder (pythia/exports)")
EXPORTS = Path(os.environ["HFFN_EXPORTS"]) if "HFFN_EXPORTS" in os.environ else ASSETS / "pythia/exports"
MODELS = ("pythia70m", "pythia410m")
NS = (128, 2100)


def timed(owner, name: str) -> None:
    """Wrap `owner.name` so that each call prints its start and its duration (timestamped progress of a long case)."""
    fn = getattr(owner, name)

    def wrapper(*a, **k):
        t0 = time.time()
        print(f"{time.strftime('%H:%M:%S', time.gmtime())}Z phase {name} start", flush=True)
        out = fn(*a, **k)
        print(f"{time.strftime('%H:%M:%S', time.gmtime())}Z phase {name} done {time.time() - t0:.1f} s", flush=True)
        return out
    setattr(owner, name, wrapper)


_EDGES: dict = {}


def oracle_gpu(layer, x, gy, params=None, device="cuda"):
    """`train_step.oracle` unchanged in its equations (direct float64 equations on the compact support), run on the GPU.

    Run on the GPU because the CPU oracle is minutes per call at N = 2100 on a 410M layer. Inputs and result stay CPU
    tensors, so that the gates (masks, indexing) are untouched. `index_add_` on CUDA accumulates float64 with atomics: the
    order differs from the CPU oracle at the 1e-16 level, far below the float32 distances that the gates measure."""
    f64, ts = torch.float64, check.ts
    p = params or {"up_val": torch.from_numpy(layer.up_val), "b_up": torch.from_numpy(layer.b_up),
                   "down_val": torch.from_numpy(layer.down_val), "b_down": torch.from_numpy(layer.b_down)}
    p = {k: v.to(device=device, dtype=f64) for k, v in p.items()}
    if id(layer) not in _EDGES or _EDGES[id(layer)][0] is not layer:   # the layer is kept alive so that its id is not reused (session J crash)
        _EDGES[id(layer)] = (layer, tuple(t.to(device) for t in ts.hidden_major_edges(layer)))
    up_hidden, up_input, down_hidden, down_output = _EDGES[id(layer)][1]
    n = x.shape[0]
    out = {"y": torch.empty(n, layer.d_model, dtype=f64, device=device), "gz": torch.empty(layer.d_ff, n, dtype=f64, device=device),
           "gx": torch.empty(n, layer.d_model, dtype=f64, device=device),
           "g_up": torch.zeros_like(p["up_val"]), "g_down": torch.zeros_like(p["down_val"]),
           "gb_up": torch.zeros(layer.d_ff, dtype=f64, device=device), "gb_down": gy.to(device=device, dtype=f64).sum(0)}
    for lo in range(0, n, ts.ORACLE_TOKENS):
        hi = min(n, lo + ts.ORACLE_TOKENS)
        xc, gc = x[lo:hi].to(device=device, dtype=f64), gy[lo:hi].to(device=device, dtype=f64)
        z = p["b_up"].expand(hi - lo, -1).clone().index_add_(1, up_hidden, xc[:, up_input] * p["up_val"])
        h = ts._gelu(z)
        out["y"][lo:hi] = p["b_down"].expand(hi - lo, -1).clone().index_add_(1, down_output, h[:, down_hidden] * p["down_val"])
        out["g_down"] += (gc[:, down_output] * h[:, down_hidden]).sum(0)
        dh = torch.zeros(hi - lo, layer.d_ff, dtype=f64, device=device).index_add_(1, down_hidden, gc[:, down_output] * p["down_val"])
        gz = dh * ts._dgelu(z)
        out["gz"][:, lo:hi] = gz.T
        out["gb_up"] += gz.sum(0)
        out["g_up"] += (gz[:, up_hidden] * xc[:, up_input]).sum(0)
        out["gx"][lo:hi] = torch.zeros(hi - lo, layer.d_model, dtype=f64, device=device).index_add_(1, up_input, gz[:, up_hidden] * p["up_val"])
    return {k: v.cpu() for k, v in out.items()}


def equivalence(n: int = 300) -> bool:
    """Step 1 of the session: `oracle_gpu` against the CPU `train_step.oracle` on the synthetic test layer (max abs difference per group)."""
    ts = check.ts
    layer, case = layers.load(check.LAYER)
    x = torch.from_numpy(layers.make_input(case, n))
    gy = ts.make_gy(case, n, 0)
    a, b = ts.oracle(layer, x, gy), oracle_gpu(layer, x, gy)
    diff = {k: float((a[k] - b[k]).abs().max()) for k in ts.GROUPS}
    scale = {k: float(a[k].abs().max()) for k in ts.GROUPS}
    ok = all(diff[k] <= 1e-12 * max(scale[k], 1.0) for k in ts.GROUPS)
    print(json.dumps({"oracle_equivalence": {"layer": check.LAYER, "n": n, "max_abs_diff": diff, "max_abs_value": scale, "pass": ok}}), flush=True)
    return ok


def compact(raw: dict) -> SimpleNamespace:
    """Export layer -> the attributes of `CompactLayer` (numpy views; `groups` is unused by the training gates)."""
    a = lambda k: raw[k].numpy()
    return SimpleNamespace(d_model=raw["d_model"], d_ff=raw["d_ff"], groups=None, b_up=a("b_up"), b_down=a("b_down"),
                           up_offsets=a("up_row_offsets"), up_idx=a("up_idx"), up_val=a("up_val"),
                           down_offsets=a("down_row_offsets"), down_idx=a("down_idx"), down_val=a("down_val"))


def make_input_for(case, n: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64([case.seed, n]))
    return rng.standard_normal((n, case.d_model)).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--density", nargs="*", default=["50", "70"], help="sparseNN exports to cover (ffn_only)")
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--layer", type=int, nargs="*", help="layer indices (default: first, middle, last)")
    parser.add_argument("--n", type=int, nargs="*", default=list(NS))
    parser.add_argument("--seed-offset", type=int, default=0, help="added to the case seed (criterion 1 reads layer 0 under two seeds)")
    parser.add_argument("--equivalence", action="store_true", help="only compare the GPU oracle with the CPU one on the test layer")
    args = parser.parse_args()
    if args.equivalence:
        return 0 if equivalence() else 1
    check.ts.oracle = oracle_gpu if os.environ.get("HFFN_ORACLE", "gpu") == "gpu" else check.ts.oracle
    gate5 = check.ts.gate5_local   # local form of this leaf's train_step.py

    def printed(*a, **k):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = gate5(*a, **k)
        torch.cuda.synchronize()
        print(json.dumps({"gate5_local": out, "memory_allocated_before": before, "memory_allocated_after": torch.cuda.memory_allocated(),
                          "memory_peak_during": torch.cuda.max_memory_allocated()}), flush=True)
        return out
    check.ts.gate5_local = printed
    for name in ("oracle", "dense_full_step", "gate2", "gate3", "gate5_local", "restore"):
        timed(check.ts, name)
    timed(check, "state_bits")
    layers.make_input = lambda case, n: make_input_for(case, n)   # real layers: no pinned synthetic input
    ok, other_failed = True, []
    for model in args.models:
        for dens in args.density:
            export = EXPORTS / f"{model}_sparse{dens}_ffn_only/export"
            count = len(list(export.glob("layer*")))
            for index in (args.layer if args.layer else sorted({0, count // 2, count - 1})):
                index = index % count   # -1 = last layer
                raw = read_layer(export / f"layer{index}" / f"{FORMAT}.pt")
                layer, case = compact(raw), SimpleNamespace(seed=index + 1000 + args.seed_offset, d_model=raw["d_model"])
                for n in args.n:
                    gates = check.gates_for(layer, case, n)
                    ok &= gates["update_local_5_steps"]   # criterion 1 reads gate 5; any other gate that fails is reported on its own, not folded into the exit
                    other_failed += [f'{export.parent.name} L{index} N{n} {k}' for k, v in gates.items() if k != "update_local_5_steps" and not v]
                    print(json.dumps({"export": export.parent.name, "layer": index, "n": n, "pcg64_seed": [case.seed, n], "oracle": os.environ.get("HFFN_ORACLE", "gpu"), "gates": gates}), flush=True)
    print(json.dumps({"other_gates_failed": other_failed}), flush=True)
    print("check_exports: PASSED" if ok else "check_exports: FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
