"""Verification of the CUDA FFN training step on a CUDA machine, without timing (run with `check.sh`).

For each (layer, N) of `CASES`: the gates of the step as it stands in this leaf.
  1 inference identity: `forward` without `z` is bitwise the inference runtime's, policy and the nine forced pairs;
  2 gradients and activations per group against the float64 oracle, within 10x the dense float32 control's distance;
  3 AdamW update against its float64 closed form, on the conditioned entries;
  4 same bits twice from the same state, supports and offsets unchanged;
  5 five steps, each update against the float64 closed form from the build's own pre-update state and gradient, plus five
    decay-only steps (local form, tolerance 1e-5); the cumulative own-gradient replay is printed as a record, not a gate.
Then the guards of the partial workspace: no buffer growth in a step after `reserve`, a refusal beyond 32-bit indexing.
N covers one tile (1, 33, 100, 128), a one-token last tile (257), partial last tiles (300, 2100) and exact tiles (512, 1024).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
FFN_INFERENCE = REPO / "runtime/gpu/nvidia/ffn/inference"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FFN_INFERENCE))

import torch  # noqa: E402

import hffn_ffn_cuda  # noqa: E402  (the leaf entry: puts the shared kernels and the test layers on sys.path)
from hffn_cuda.parity import PAIRS  # noqa: E402
from hffn_cuda.runtime import HFFNRuntime  # noqa: E402
from hffn_ffn_layers import layers  # noqa: E402

import train_step as ts  # noqa: E402
from train_runtime import HFFNTrainRuntime  # noqa: E402

LAYER = "homog_256_1024"
CASES = tuple((LAYER, n) for n in (1, 33, 100, 128, 257, 300, 512, 1024, 2100))
GATE1_N = (1, 2, 4, 5, 63, 64, 128, 129, 512)


def bits(*tensors) -> list:
    return [t.detach().contiguous().view(torch.int32).cpu() for t in tensors]


def same_bits(a: list, b: list) -> bool:
    return all(torch.equal(u, v) for u, v in zip(a, b))


def state_bits(train, n: int) -> list:
    grads = train.gradients(0)
    out = [train.activations(n)["gx"].clone()]
    out += [grads[k].clone() for k in sorted(grads)]
    out += [t.clone() for k, v in sorted(train.moments(0).items()) for t in v]
    out += [v.clone() for k, v in sorted(train.parameters(0).items())]
    return bits(*out)


def gates_for(layer, case, n: int) -> dict:
    device = torch.device("cuda")
    x = torch.from_numpy(layers.make_input(case, n)).to(device)
    gy = ts.make_gy(case, n, 0).to(device)
    baseline = HFFNRuntime([hffn_ffn_cuda.cuda_layer(layer)], device)
    train = HFFNTrainRuntime(HFFNRuntime([hffn_ffn_cuda.cuda_layer(layer)], device), **ts.HYPER, max_tokens=max(n, 2048))
    g1 = True
    with torch.no_grad():
        for m in GATE1_N:
            xm = torch.from_numpy(layers.make_input(case, m)).to(device)
            for pair in ((None, None), *PAIRS):
                baseline.force(*pair)
                train.runtime.force(*pair)
                g1 &= bool(torch.equal(baseline.forward(0, xm).clone(), train.forward(0, xm).clone()))
        baseline.force(None, None)
        train.runtime.force(None, None)
    train.reserve(n)
    x_cpu, gy_cpu = x.cpu(), gy.cpu()
    ref = ts.oracle(layer, x_cpu, gy_cpu)
    control, _ = ts.dense_full_step(layer, x, gy)
    before = {k: v.double() for k, v in ts.hidden_major_params(layer).items()}
    initial = ts.snapshot(train)
    growths = train.growths
    train.step(0, x, gy)
    torch.cuda.synchronize()
    no_growth = train.growths == growths
    groups = ts.hffn_groups(train, layer, n)
    r2 = ts.gate2(groups, control, ref)
    masks = ts.conditioned_masks(ref, control)
    r3 = ts.gate3(train, layer, groups, masks, before)
    first = state_bits(train, n)
    supports = {k: v.clone() for k, v in train.supports(0).items()}
    offsets = (train.runtime._tensors[0]["up_offsets"].clone(), train.runtime._tensors[0]["out_offsets"].clone())
    ts.restore(train, initial)
    train.step(0, x, gy)
    torch.cuda.synchronize()
    g4 = same_bits(first, state_bits(train, n))
    g4 &= all(torch.equal(supports[k], v) for k, v in train.supports(0).items())
    g4 &= torch.equal(offsets[0], train.runtime._tensors[0]["up_offsets"]) and torch.equal(offsets[1], train.runtime._tensors[0]["out_offsets"])
    ts.restore(train, initial)
    r5 = ts.gate5_local(train, layer, case, x, masks)
    ok = {"inference_identity": g1, "layer_float64": r2["pass"], "adamw_closed_form": r3["pass"], "repeatable": bool(g4),
          "update_local_5_steps": r5["pass"], "no_growth_in_step": no_growth}
    del baseline, train
    torch.cuda.empty_cache()
    return ok


def guards(layer) -> dict:
    """The refusal of a workspace beyond 32-bit indexing, and growth accounted only by an explicit larger `reserve`."""
    device = torch.device("cuda")
    train = HFFNTrainRuntime(HFFNRuntime([hffn_ffn_cuda.cuda_layer(layer)], device), **ts.HYPER)
    stride = max(int(c[3]) for c in train._columns)
    try:
        train._reserve_partials(2 ** 32 // stride * 256 + 256)
        refused = False
    except ValueError:
        refused = True
    before = train.growths
    train.reserve(train._tokens)             # same N: nothing may grow
    unchanged = train.growths == before
    train.reserve(train._tokens + 1)
    counted = train.growths > before
    return {"refuses_32bit_overflow": refused, "same_n_no_growth": unchanged, "larger_n_counted": counted}


def main() -> int:
    gate5 = ts.gate5_local

    def printed(*a, **k):   # the record of gate 5 (local errors, decay pass, cumulative replay) on its own line
        out = gate5(*a, **k)
        print(json.dumps({"gate5_local": out}), flush=True)
        return out
    ts.gate5_local = printed
    ok = True
    for name, n in CASES:
        layer, case = layers.load(name)
        gates = gates_for(layer, case, n)
        ok &= all(gates.values())
        print(json.dumps({"layer": name, "n": n, "gates": gates}), flush=True)
    layer, _ = layers.load(LAYER)
    g = guards(layer)
    ok &= all(g.values())
    print(json.dumps({"guards": g}), flush=True)
    print("check: PASSED" if ok else "check: FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
