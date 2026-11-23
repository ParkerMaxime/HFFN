"""Parity-only S1 qualification against independent isolated CUDA runtimes.

No runtime or kernel equations are copied. Gate snapshots are outside the
dedicated allocation-stability span. --smoke reduces only the same-N grid;
mixed-N, noncontiguous inputs, alias/guard checks and the large-N memory check
remain. --out is a new run folder, never a file to overwrite.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import traceback

HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
LEAF = HERE.parent
FULL_N = (1, 4, 33, 64, 129, 257, 2100)
CAPACITY = 2100
LAYER_NAMES = ("homog_512_2048", "irreg_512_2048", "irreg_512_2048_zero_input0")
PARAMETERS = ("up_val", "b_up", "out_val", "b_down")


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def update_manifest(path: Path, value) -> None:
    """Publish progress atomically during this new run; completed records stay put."""
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".manifest-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def provenance(base: str | None) -> dict:
    paths = [HERE / "context_check.py", LEAF / "README.md", LEAF / "launch.sh"]
    paths += [REPO / "runtime/common/pythia/hffn_pythia/wikitext2_test_tokens.json"]
    paths += sorted((LEAF / "hffn_pythia_cuda_training").glob("*.py"))
    paths += [REPO / name for name in (
        "runtime/gpu/nvidia/ffn/training/train_runtime.py",
        "runtime/gpu/nvidia/ffn/training/kernels_train.py",
        "runtime/gpu/nvidia/ffn/training/train_step.py",
        "runtime/gpu/nvidia/common/kernels/hffn_cuda/runtime.py",
        "runtime/gpu/nvidia/common/kernels/hffn_cuda/kernels.py",
        "runtime/gpu/nvidia/ffn/inference/hffn_ffn_cuda/__init__.py",
        "runtime/gpu/nvidia/ffn/inference/hffn_ffn_cuda/layers.py",
        "runtime/common/ffn/hffn_ffn_layers/layers.py",
        "runtime/common/ffn/hffn_ffn_layers/synth.py",
        "runtime/common/ffn/hffn_ffn_layers/dense.py",
        "runtime/common/ffn/reference/fingerprints.json",
        "runtime/common/pythia/hffn_pythia/model.py",
        "runtime/common/pythia/hffn_pythia/export.py",
    )]
    # Bind every imported Python source in the reached leaf/shared packages,
    # including the campaign driver and the model gate's helper dependencies.
    for folder in (LEAF, REPO / "runtime/gpu/nvidia/ffn/training",
                   REPO / "runtime/gpu/nvidia/ffn/inference/hffn_ffn_cuda",
                   REPO / "runtime/gpu/nvidia/common/kernels/hffn_cuda",
                   REPO / "runtime/common/ffn", REPO / "runtime/common/pythia/hffn_pythia"):
        paths += [path for path in folder.rglob("*.py")
                  if not {"build", "runs", "__pycache__"}.intersection(path.relative_to(folder).parts)]
    hashes = {path.relative_to(REPO).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False)
    head = git.stdout.strip() if git.returncode == 0 else None
    chosen_base = base or os.environ.get("HFFN_BASE_COMMIT") or head
    if not chosen_base:
        raise RuntimeError("record the source base with --base or HFFN_BASE_COMMIT when git HEAD is unavailable")
    digest = hashlib.sha256("".join(f"{name} {sha}\n" for name, sha in sorted(hashes.items())).encode()).hexdigest()
    return {"base": chosen_base, "git_head": head, "source_hashes": hashes, "tree_sha256": digest,
            "python": platform.python_version(), "platform": platform.platform()}


def import_runtime() -> None:
    # Imports are inside the recorded run, so missing CUDA dependencies become
    # a failed manifest and reliable EXIT rather than an unrecorded import exit.
    for path in (LEAF, REPO / "runtime/gpu/nvidia/ffn/inference",
                 REPO / "runtime/gpu/nvidia/ffn/training",
                 REPO / "runtime/common/ffn"):
        sys.path.insert(0, str(path))
    global torch, np, cupy, layers, cuda_layer, HFFNRuntime, HFFNTrainRuntime, ContextTrainRuntime, ts
    import torch
    import numpy as np
    import cupy
    from hffn_ffn_layers import layers
    from hffn_ffn_cuda import cuda_layer
    from hffn_cuda.runtime import HFFNRuntime
    from train_runtime import HFFNTrainRuntime
    from hffn_pythia_cuda_training import ContextTrainRuntime
    import train_step as ts
    if not torch.cuda.is_available():
        raise RuntimeError("this qualification requires CUDA; it must not run on the Mac")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def same_bits(a, b) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype == torch.float32:
        return bool(torch.equal(a.detach().contiguous().view(torch.int32), b.detach().contiguous().view(torch.int32)))
    return bool(torch.equal(a, b))


def finite(tensors) -> bool:
    return all(bool(torch.isfinite(t).all()) for t in tensors)


def state_snapshot(runtime) -> dict:
    return {"parameters": [{name: tensor.clone() for name, tensor in runtime.parameters(i).items()} for i in range(3)],
            "moments": [{name: tuple(t.clone() for t in pair) for name, pair in runtime.moments(i).items()} for i in range(3)],
            "steps": list(runtime.steps)}


def state_unchanged(runtime, snapshot: dict) -> bool:
    return (list(runtime.steps) == snapshot["steps"]
            and all(same_bits(t, snapshot["parameters"][i][name])
                    for i in range(3) for name, t in runtime.parameters(i).items())
            and all(same_bits(t, snapshot["moments"][i][name][k])
                    for i in range(3) for name, pair in runtime.moments(i).items() for k, t in enumerate(pair)))


def supports_snapshot(runtime) -> list[dict]:
    return [{**{name: t.clone() for name, t in runtime.supports(i).items()},
             **{name: runtime.runtime._tensors[i][name].clone()
                for name in ("up_offsets", "up_idx", "out_offsets", "out_hidden")}} for i in range(3)]


def supports_unchanged(runtime, snapshot: list[dict]) -> bool:
    return all(same_bits(t, snapshot[i][name]) for i in range(3)
               for name, t in {**runtime.supports(i),
                               **{name: runtime.runtime._tensors[i][name]
                                  for name in ("up_offsets", "up_idx", "out_offsets", "out_hidden")}}.items())


def make_layers():
    homog, homog_case = layers.load("homog_512_2048")
    irregular, irregular_case = layers.load("irreg_512_2048")
    keep = irregular.up_idx != 0
    owners = np.repeat(np.arange(irregular.d_ff), np.diff(irregular.up_offsets))
    degrees = np.bincount(owners[keep], minlength=irregular.d_ff)
    offsets = np.zeros(irregular.d_ff + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(degrees)
    zero = replace(irregular, up_offsets=offsets, up_idx=irregular.up_idx[keep].copy(),
                   up_val=irregular.up_val[keep].copy())
    require(not np.any(zero.up_idx == 0), "derived layer still has feature-0 incoming edges")
    require(int(offsets[-1]) == zero.up_idx.size == zero.up_val.size, "derived offsets are not exact")
    require(np.all(np.diff(offsets) == degrees), "derived offsets do not preserve each hidden owner")
    fixtures = (homog, irregular, zero)
    cases = (homog_case, irregular_case, irregular_case)
    metadata = {"fixture_fingerprints": {name: layers.fingerprint(layer, case)
                                        for name, layer, case in zip(LAYER_NAMES, fixtures, cases)},
                "pinned": {name: layers.pinned()[name] for name in LAYER_NAMES[:2]},
                "removed_feature0_slots": int((~keep).sum()),
                "gates": {"pinned_homogeneous": layers.fingerprint(homog, homog_case) == layers.pinned()[homog_case.name],
                          "pinned_irregular": layers.fingerprint(irregular, irregular_case) == layers.pinned()[irregular_case.name],
                          "derived_exact_offsets": True, "zero_input0_support": True}}
    return fixtures, cases, metadata


def make_runtimes(fixtures):
    train = HFFNTrainRuntime(HFFNRuntime([cuda_layer(layer) for layer in fixtures], "cuda"),
                             **ts.HYPER, max_tokens=CAPACITY)
    runtime = ContextTrainRuntime(train, max_tokens=CAPACITY)
    controls = [HFFNTrainRuntime(HFFNRuntime([cuda_layer(layer)], "cuda"),
                                **ts.HYPER, max_tokens=CAPACITY) for layer in fixtures]
    for control in controls:
        control.reserve(CAPACITY)
    return runtime, controls


def make_inputs(cases, sizes, noncontiguous: bool = False):
    xs, gys = [], []
    for i, (case, n) in enumerate(zip(cases, sizes)):
        x = torch.from_numpy(layers.make_input(case, n)).to("cuda")
        gy = ts.make_gy(case, n, i).to("cuda")
        if noncontiguous:
            require(n == 4, "noncontiguous fixture requires four tokens")
            x, gy = x.view(2, 2, case.d_model).transpose(0, 1), gy.view(2, 2, case.d_model).transpose(0, 1)
            require(not x.is_contiguous() and not gy.is_contiguous(), "noncontiguous test accidentally became contiguous")
        xs.append(x)
        gys.append(gy)
    return xs, gys


def same_weight_repeatability(runtime, cases) -> dict:
    """Repeat all three forwards on the same values, with no intervening update.

    The unconsumed contexts are discarded only after their outputs have been
    observed and cloned. This establishes forward repeatability; the isolated
    controls in complete_cycle separately establish bitwise gradients/updates.
    """
    xs, _gys = make_inputs(cases, (129, 129, 129))
    before = state_snapshot(runtime)
    supports = supports_snapshot(runtime)
    workspace = runtime.workspace_stats()
    first = None
    for _ in range(2):
        contexts, outputs = [], []
        for i in range(3):
            y, cid = runtime.forward_with_context(i, xs[i])
            contexts.append(cid)
            outputs.append(y.clone())
            require(state_unchanged(runtime, before), "same-weight repeat forward changed optimizer state")
        require(finite(outputs), "same-weight repeat forward is not finite")
        if first is None:
            first = outputs
        else:
            require(all(same_bits(a, b) for a, b in zip(first, outputs)), "same-weight forward repeat is not bitwise")
        for cid in contexts:
            runtime.discard_context(cid)
        require(state_unchanged(runtime, before), "discarding repeat contexts changed optimizer state")
        require(supports_unchanged(runtime, supports), "repeat forward changed fixed supports")
    require(runtime.workspace_stats() == workspace, "same-weight repeat changed workspace/context state")
    return {"sizes": [129, 129, 129], "scope": "same-weight FFN forward only", "repeats": 2,
            "gates": {"forward_bits_repeatable": True, "parameters_moments_unchanged": True,
                      "fixed_support": True, "discarded_after_observation": True, "workspace_unchanged": True}}


def retained_activations(runtime, contexts):
    return [{name: tensor.clone() for name, tensor in runtime.activations(cid).items() if name != "gx"}
            for cid in contexts]


def check_aliases(runtime, contexts, snapshots, returned_y, returned_gx, gx_snapshots) -> None:
    for i, cid in enumerate(contexts):
        actual = runtime.activations(cid)
        require(all(same_bits(actual[name], tensor) for name, tensor in snapshots[i].items()),
                f"layer {i} activation buffer overwritten by another forward/backward")
        require(same_bits(returned_y[i], snapshots[i]["y"]), f"returned y of layer {i} was overwritten")
    for i, tensor in returned_gx.items():
        require(same_bits(tensor, gx_snapshots[i]), f"returned gx of layer {i} was overwritten")


def positive_zero_feature(gx) -> bool:
    return bool(torch.equal(gx[:, 0].contiguous().view(torch.int32), torch.zeros(gx.shape[0], dtype=torch.int32, device=gx.device)))


def complete_cycle(runtime, controls, xs, gys) -> dict:
    before = state_snapshot(runtime)
    supports = supports_snapshot(runtime)
    inference_outputs = []
    # All inference calls precede retained training contexts, so the cycle's
    # default-stream/call-count guard captures their completed call history.
    for i in range(3):
        inference_outputs.append(runtime.forward(i, xs[i]).clone())
        require(state_unchanged(runtime, before), "inference changed a weight, moment or AdamW step")
    contexts, returned_y, snapshots = [], [], []
    returned_gx, gx_snapshots = {}, {}
    for i in range(3):
        y, cid = runtime.forward_with_context(i, xs[i])
        contexts.append(cid)
        returned_y.append(y)
        snapshots.append(retained_activations(runtime, [cid])[0])
        require(same_bits(y, inference_outputs[i]), f"layer {i} with-z forward differs from its S1 inference output")
        control_x = xs[i].contiguous().view(-1, runtime.d_model)
        control_y = controls[i].forward(0, control_x, controls[i]._z)
        require(same_bits(y, control_y), f"layer {i} forward differs from isolated control")
        require(finite(snapshots[-1].values()), f"layer {i} activation is not finite")
        check_aliases(runtime, contexts, snapshots, returned_y, returned_gx, gx_snapshots)
        require(state_unchanged(runtime, before), "forward changed a weight, moment or AdamW step")
    for i in (2, 1, 0):
        gx = runtime.backward_with_context(contexts[i], i, gys[i])
        returned_gx[i], gx_snapshots[i] = gx, gx.clone()
        control_gy = gys[i].contiguous().view(-1, runtime.d_model)
        control_gx = controls[i].backward(0, control_gy)
        require(same_bits(gx, control_gx), f"layer {i} gx differs from isolated control")
        require(all(same_bits(runtime.gradients(i)[name], controls[i].gradients(0)[name]) for name in PARAMETERS),
                f"layer {i} compact gradients differ from isolated control")
        require(finite([gx, *runtime.gradients(i).values()]), f"layer {i} gradient is not finite")
        if i == 2:
            require(positive_zero_feature(gx), "zero-degree input gx is not positive-zero bits")
        check_aliases(runtime, contexts, snapshots, returned_y, returned_gx, gx_snapshots)
        require(state_unchanged(runtime, before), "backward updated a weight, moment or AdamW step")
    require(runtime.workspace_stats()["awaiting_backward"] == 0, "the complete backward left a pending context")
    runtime.adamw_all()
    for i, control in enumerate(controls):
        control.adamw(0)
        require(all(same_bits(runtime.parameters(i)[name], control.parameters(0)[name]) for name in PARAMETERS),
                f"layer {i} AdamW parameters differ from isolated control")
        require(all(same_bits(t, control.moments(0)[name][k])
                    for name, pair in runtime.moments(i).items() for k, t in enumerate(pair)),
                f"layer {i} AdamW moments differ from isolated control")
        require(runtime.steps[i] == control.steps[0], f"layer {i} AdamW step differs")
        require(any(not same_bits(t, before["parameters"][i][name]) for name, t in runtime.parameters(i).items()),
                f"layer {i} update did not change any canonical device value")
        require(finite([*runtime.parameters(i).values(), *(t for pair in runtime.moments(i).values() for t in pair)]),
                f"layer {i} updated state is not finite")
        # Views may remain held by the caller after the update. They must not
        # change until the next forward of this same layer.
        require(same_bits(returned_y[i], snapshots[i]["y"]), f"AdamW overwrote y of layer {i}")
        require(same_bits(returned_gx[i], gx_snapshots[i]), f"AdamW overwrote gx of layer {i}")
    require(supports_unchanged(runtime, supports), "an update changed the fixed support")
    require(runtime.workspace_stats()["live"] == 0, "AdamW did not release consumed contexts")
    return {"gates": {"inference_identity": True, "bitwise_y": True, "bitwise_gx": True, "bitwise_compact_gradients": True,
                      "retained_activation_aliases": True, "retained_y_gx_aliases": True,
                      "no_early_update": True, "bitwise_adamw": True, "fixed_support": True,
                      "finite_state": True, "zero_degree_gx_positive_zero": True},
            "output_snapshots": [snapshot["y"] for snapshot in snapshots]}


def two_cycles(runtime, controls, cases, sizes, noncontiguous: bool = False) -> dict:
    xs, gys = make_inputs(cases, sizes, noncontiguous)
    first = complete_cycle(runtime, controls, xs, gys)
    second = complete_cycle(runtime, controls, xs, gys)
    require(all(not same_bits(a, b) for a, b in zip(first["output_snapshots"], second["output_snapshots"])),
            "second forward did not read the changed device weights")
    return {"sizes": list(sizes), "cycles": 2, "noncontiguous_3d": noncontiguous,
            "gates": {**first["gates"], **second["gates"], "second_cycle_reads_updated_device_weights": True}}


def guard(runtime, name, call) -> dict:
    before = state_snapshot(runtime)
    workspace = runtime.workspace_stats()
    refused = False
    try:
        call()
    except (RuntimeError, ValueError, IndexError):
        refused = True
    require(refused, f"{name} was not refused")
    require(state_unchanged(runtime, before), f"{name} changed parameters, moments or AdamW steps")
    require(runtime.workspace_stats() == workspace, f"{name} changed context/workspace state")
    return {"gates": {"refused": True, "state_unchanged": True, "workspace_unchanged": True}}


def bind_failure(runtime, cases) -> dict:
    class InjectedFailure(RuntimeError):
        pass

    xs, gys = make_inputs(cases, (4, 4, 4))
    base, train = runtime.runtime, runtime.train
    before = state_snapshot(runtime)
    names_base = ("_x_t", "_act", "_y", "_work")
    names_train = ("_z", "_gx", "_work", "_saved")
    original_base = tuple(getattr(base, name) for name in names_base)
    original_train = tuple(getattr(train, name) for name in names_train)
    absent = object()
    previous_method = train.__dict__.get("forward", absent)

    def raise_inside_forward(*args, **kwargs):
        raise InjectedFailure("deliberate failure inside the rebound isolated forward")

    train.forward = raise_inside_forward
    refused = False
    try:
        runtime.forward_with_context(0, xs[0])
    except InjectedFailure:
        refused = True
    finally:
        if previous_method is absent:
            del train.forward
        else:
            train.forward = previous_method
    require(refused, "injected forward exception did not propagate")
    require(all(getattr(base, name) is value for name, value in zip(names_base, original_base)),
            "inference buffers were not restored after a forward exception")
    require(all(getattr(train, name) is value for name, value in zip(names_train, original_train)),
            "training buffers were not restored after a forward exception")
    require(not runtime._binding and runtime.workspace_stats()["live"] == 0, "failed forward retained a context")
    require(state_unchanged(runtime, before), "failed forward changed parameters or moments")
    y, cid = runtime.forward_with_context(0, xs[0])
    runtime.backward_with_context(cid, 0, gys[0])
    runtime.adamw_all()
    require(finite([y, *runtime.parameters(0).values()]), "wrapper was not usable after the injected failure")
    return {"gates": {"injected_failure": True, "bindings_restored": True, "state_unchanged": True,
                      "usable_after_failure": True}}


def bare_cycle(runtime, xs, gys) -> None:
    """No snapshots, bit comparisons or newly created device inputs in this span."""
    runtime.zero_grad()
    contexts = []
    for i in range(3):
        _y, cid = runtime.forward_with_context(i, xs[i])
        contexts.append(cid)
    for i in (2, 1, 0):
        _gx = runtime.backward_with_context(contexts[i], i, gys[i])
    runtime.adamw_all()


def memory_stability(runtime, cases) -> dict:
    xs, gys = make_inputs(cases, (CAPACITY, CAPACITY, CAPACITY))
    bare_cycle(runtime, xs, gys)
    torch.cuda.synchronize()
    allocated_before = int(torch.cuda.memory_allocated(runtime.device))
    stats_before = runtime.workspace_stats()
    for _ in range(3):
        bare_cycle(runtime, xs, gys)
    torch.cuda.synchronize()
    allocated_after = int(torch.cuda.memory_allocated(runtime.device))
    stats_after = runtime.workspace_stats()
    require(allocated_after == allocated_before, "warmed complete cycles grew torch.cuda.memory_allocated")
    require(stats_before == stats_after, "warmed complete cycles grew context or isolated workspaces")
    return {"sizes": [CAPACITY] * 3, "warmed_cycles": 1, "checked_cycles": 3,
            "allocated_before_bytes": allocated_before, "allocated_after_bytes": allocated_after,
            "workspace_before": stats_before, "workspace_after": stats_after,
            "gates": {"memory_allocated_unchanged": True, "context_growth_unchanged": True, "train_growth_unchanged": True}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new folder holding the qualification records")
    parser.add_argument("--smoke", action="store_true", help="same-N 1/129 only; other essential cases remain")
    parser.add_argument("--base", help="base commit when the VM checkout lacks git metadata")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reserved = ("config.json", "provenance.json", "raw.jsonl", "manifest.json", "summary.json", "EXIT")
    if any((args.out / name).exists() for name in reserved):
        print("Refusing to overwrite existing run records", flush=True)
        return 2
    grid = (1, 129) if args.smoke else FULL_N
    guard_names = ("invalid_forward_index", "invalid_backward_index", "invalid_context", "wrong_context_layer",
                   "wrong_backward_n", "premature_adamw", "premature_preflight", "reserve_live",
                   "double_live_forward", "repeated_backward", "consumed_forward", "reserve_consumed", "stale_context")
    planned = ["fixtures", "same_weight_repeatability", *(f"same_n_{n}" for n in grid), "mixed_n", "noncontiguous_3d",
               *guard_names, "bind_exception_restore", "warmed_memory_stability"]
    config = {"mode": "smoke" if args.smoke else "qualification", "timing": False, "dtype": "float32",
              "device": "cuda:0", "shape": [512, 2048], "layers": list(LAYER_NAMES), "same_n": list(grid),
              "capacity": CAPACITY, "mixed_n": [1, 129, 2100], "planned_ids": planned,
              "control": "three independent current HFFNTrainRuntime instances", "hyper_source": "train_step.HYPER",
              "zero_degree_derivation": "remove all input feature-0 slots and rebuild exact hidden offsets"}
    write_json(args.out / "config.json", config)
    rows, done, error = [], [], None
    device = None
    raw = (args.out / "raw.jsonl").open("x", encoding="utf-8")

    def progress(status="running", *, phase="checks", active_case=None, exit_code=None, failure=None):
        value = {"status": status, "exit_code": exit_code, "planned": len(planned), "done": len(done),
                 "planned_ids": planned, "done_ids": list(done), "full_coverage": done == planned,
                 "missing": [name for name in planned if name not in done], "mode": config["mode"],
                 "phase": phase, "active_case": active_case, "attempted": len(rows), "error": failure}
        update_manifest(args.out / "manifest.json", value)
        return value

    progress(phase="provenance")

    def record(name, operation):
        progress(active_case=name)
        print(f"context check: {name}", flush=True)
        try:
            result = operation()
            gates = result["gates"]
            require(bool(gates) and all(gates.values()), f"{name} has a failing gate")
            row = {"case": name, "status": "passed", **result}
        except Exception as exc:
            row = {"case": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            raw.write(json.dumps(row, allow_nan=False) + "\n")
            raw.flush()
            progress("failed", active_case=name, exit_code=1, failure=row["error"])
            print(json.dumps(row, allow_nan=False), flush=True)
            raise
        rows.append(row)
        done.append(name)
        raw.write(json.dumps(row, allow_nan=False) + "\n")
        raw.flush()
        progress()
        print(json.dumps(row, allow_nan=False), flush=True)

    try:
        write_json(args.out / "provenance.json", provenance(args.base))
        progress(phase="imports")
        print("Loading CUDA dependencies and pinned synthetic fixtures", flush=True)
        import_runtime()
        device = {"name": torch.cuda.get_device_name(0), "torch": torch.__version__, "cupy": cupy.__version__,
                  "cuda": torch.version.cuda, "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory)}
        print(json.dumps({"device": device}, allow_nan=False), flush=True)
        fixtures, cases, metadata = make_layers()
        record("fixtures", lambda: metadata)
        progress(phase="reserve_runtimes")
        print("Constructing S1 and three isolated controls; reserving 2100 tokens", flush=True)
        runtime, controls = make_runtimes(fixtures)
        record("same_weight_repeatability", lambda: same_weight_repeatability(runtime, cases))
        for n in grid:
            record(f"same_n_{n}", lambda n=n: two_cycles(runtime, controls, cases, (n, n, n)))
        record("mixed_n", lambda: two_cycles(runtime, controls, cases, (1, 129, 2100)))
        record("noncontiguous_3d", lambda: two_cycles(runtime, controls, cases, (4, 4, 4), True))
        xs, gys = make_inputs(cases, (4, 4, 4))
        _y, cid = runtime.forward_with_context(0, xs[0])
        calls = {
            "invalid_forward_index": lambda: runtime.forward_with_context(-1, xs[0]),
            "invalid_backward_index": lambda: runtime.backward_with_context(cid, 3, gys[0]),
            "invalid_context": lambda: runtime.backward_with_context(cid + 100000, 0, gys[0]),
            "wrong_context_layer": lambda: runtime.backward_with_context(cid, 1, gys[0]),
            "wrong_backward_n": lambda: runtime.backward_with_context(cid, 0, gys[0][:1]),
            "premature_adamw": runtime.adamw_all,
            "premature_preflight": runtime.ensure_can_step,
            "reserve_live": lambda: runtime.reserve(CAPACITY),
            "double_live_forward": lambda: runtime.forward_with_context(0, xs[0]),
        }
        for name, call in calls.items():
            record(name, lambda name=name, call=call: guard(runtime, name, call))
        runtime.backward_with_context(cid, 0, gys[0])
        consumed_calls = {
            "repeated_backward": lambda: runtime.backward_with_context(cid, 0, gys[0]),
            "consumed_forward": lambda: runtime.forward_with_context(0, xs[0]),
            "reserve_consumed": lambda: runtime.reserve(CAPACITY),
        }
        for name, call in consumed_calls.items():
            record(name, lambda name=name, call=call: guard(runtime, name, call))
        runtime.adamw_all()
        record("stale_context", lambda: guard(runtime, "stale_context", lambda: runtime.backward_with_context(cid, 0, gys[0])))
        record("bind_exception_restore", lambda: bind_failure(runtime, cases))
        # Alias comparison snapshots have fallen out of scope before this
        # dedicated span. Its inputs are created before the allocation reading.
        record("warmed_memory_stability", lambda: memory_stability(runtime, cases))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        raw.close()
    full_coverage = done == planned
    passed = error is None and full_coverage and all(row["status"] == "passed" for row in rows)
    exit_code = 0 if passed else 1
    manifest = progress("passed" if passed else "failed", phase="complete", exit_code=exit_code, failure=error)
    write_json(args.out / "summary.json", {"passed": passed, "status": manifest["status"], "mode": config["mode"],
                                          "timing": False, "device": device, "coverage": manifest, "checks": rows})
    with (args.out / "EXIT").open("x", encoding="utf-8") as stream:
        stream.write(f"{exit_code}\n")
    print(f"context check: {'PASSED' if passed else 'FAILED'} ({len(done)}/{len(planned)}, EXIT {exit_code})", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
