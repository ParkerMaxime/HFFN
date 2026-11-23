"""Functional model qualification of the leaf `llm/training` on one targeted export, without timing.

The dense64 reference, initial-only dense32 probe, HFFN model and fresh dense32 iso-support update control are constructed
sequentially. The probe is reduced to scalar distances, conditioning masks
and exact initial-gradient/loss fingerprints; the fresh control must reproduce those fingerprints before its first update.

Every optimiser update of the five-step horizon of `hffn` and `dense32` is judged against the float64 closed form from the build's own pre-update state (`update_gate.UpdateGate`, snapshots read around
each fused `optimizer.step()`); two decay-only arms `hffn_decay` and `dense32_decay` (zero loss gradient); on the 70M export
only, four build-side fault arms (a constructor hyper-parameter of the HFFN optimiser changed) and the detection of the eight
reference-side faults on the recorded real states; the cumulative form D is still recorded, with its `exceeded` flag, and no
longer gates `update_pass`.

    python -B checks/model_check.py --mode smoke --export pythia70m_sparse10_ffn_only --out <new folder>
    python -B checks/model_check.py --mode run --export pythia410m_sparse80_ffn_only --out <new folder>

Both modes use the functional batch (1 x 4 tokens) and five actual updates.
Smoke is not qualification evidence. The whole-model time bar is unmeasured.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import struct
import sys
import traceback

HERE = Path(__file__).resolve().parent
LEAF = HERE.parent
REPO = next(parent for parent in HERE.parents if (parent / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
BATCH, SEQ_LEN, STEPS = 1, 4, 5
TARGETS = ("pythia70m_sparse10_ffn_only", "pythia410m_sparse80_ffn_only")
KINDS = ("up_val", "down_val", "b_up", "b_down")
GROUPS = ("g_up", "g_down", "gb_up", "gb_down")
JUDGED_ARMS = ("hffn", "dense32")
BUILD_FAULT_NAMES = ("weight_decay_zero", "eps_times_1000", "beta2_0.99", "lr_times_1.0001")    # 70M stage only
PEAK_BOUND = 14 * 2**30
DEVICE = "cuda"
TOKENS = REPO / "runtime/common/pythia/hffn_pythia/wikitext2_test_tokens.json"

sys.path.insert(0, str(LEAF / "checks"))
sys.path.insert(0, str(LEAF))
sys.path.insert(0, str(HERE))
import context_check as context_helpers  # noqa: E402 (stdlib-only until import_runtime is called)


def extra_sources(tokens: Path) -> list:
    """Files bound in `provenance.json` on top of the leaf's own list (`context_check.provenance`); `launch.sh check` verifies
    that each exists, so a bound path cannot silently stop being hashed."""
    return [HERE / "model_check.py", HERE / "update_gate.py", HERE / "test_update_gate.py",
            LEAF / "launch.sh", LEAF / "README.md", LEAF / "check.sh", tokens,
            REPO / "runtime/common/pythia/hffn_pythia/training_gate.py",
            REPO / "runtime/gpu/nvidia/ffn/training/check_exports.py"]


def digest_of(hashes: dict) -> str:
    return hashlib.sha256("".join(f"{name} {sha}\n" for name, sha in sorted(hashes.items())).encode()).hexdigest()


def extra_hashes(tokens: Path) -> dict:
    return {(path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in extra_sources(tokens)}


def bound_hashes(tokens: Path, base: str) -> dict:
    """Hash of every file `provenance.json` binds (the leaf's list plus the model checker's), read from the disk now."""
    return {**context_helpers.provenance(base)["source_hashes"], **extra_hashes(tokens)}


def compare_bound(start: dict, hashes: dict) -> list:
    """Pure: the differences between the bound files hashed at the start of a run and those on disk now, empty when the tree
    did not move during the run."""
    problems = [f"missing: {name}" for name in sorted(set(start) - set(hashes))]
    problems += [f"added: {name}" for name in sorted(set(hashes) - set(start))]
    problems += [f"changed: {name}" for name in sorted(set(start) & set(hashes)) if start[name] != hashes[name]]
    return problems


def planned_ids(fault_stage: bool) -> list:
    """Every stage that records a row `{"id": ...}` and stores `results[id]`; the manifest counts exactly these."""
    return ["dense64", "dense32_probe", "hffn", "dense32"] + [decay_id(name) for name in JUDGED_ARMS] + (
        [fault_id(name) for name in BUILD_FAULT_NAMES] if fault_stage else []) + ["model_gate"]


def decay_id(arm: str) -> str:
    return f"{arm}_decay"


def fault_id(name: str) -> str:
    return f"hffn_build_fault_{name}"


_ORDER: dict = {}


def order(layer):
    """`ts.output_major_order(layer)` (a stable argsort of `down_idx`), computed once per layer: the gate, the snapshots and the
    form D ask for it at every step. The layer is kept in the cache so that its `id` cannot be reused while the entry lives."""
    hit = _ORDER.get(id(layer))
    if hit is None or hit[0] is not layer:
        hit = _ORDER[id(layer)] = (layer, ts.output_major_order(layer))
    return hit[1]


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(item) for item in value]
    return value


def write_json(path: Path, value) -> None:
    context_helpers.write_json(path, finite_json(value))


def replace_json(path: Path, value) -> None:
    """Atomic progress replacement while this run owns its records; completed runs are never reopened."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(finite_json(value), stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def import_runtime() -> None:
    context_helpers.import_runtime()
    global torch, cupy, ts, tg, ug, shared_model, export, s1_load, s1_optimizer, compact_adapter
    torch, cupy, ts = context_helpers.torch, context_helpers.cupy, context_helpers.ts
    import update_gate as ug       # needs train_step on sys.path, which context_helpers.import_runtime has just arranged
    from hffn_pythia import model as shared_model, export, training_gate as tg
    from hffn_pythia_cuda_training import load_pythia as s1_load, create_optimizer as s1_optimizer
    from check_exports import compact as compact_adapter
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)


def cpu_evaluation(evaluation):
    loss, ffn, other = evaluation
    return loss, [tuple(t.detach().cpu() for t in row) for row in ffn], {name: t.detach().cpu() for name, t in other.items()}


def evaluation_fingerprints(evaluation) -> dict:
    """Exact loss bits and every initial gradient, without retaining tensor payloads."""
    def fingerprint(tensor):
        array = tensor.detach().contiguous().numpy()
        return {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
                "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}
    return {"loss_float64_bits": struct.pack(">d", evaluation[0]).hex(),
            "ffn": [{name: fingerprint(tensor) for name, tensor in zip(KINDS, row)}
                    for row in evaluation[1]],
            "non_ffn": {name: fingerprint(tensor) for name, tensor in sorted(evaluation[2].items())}}


def host_progress(record, arm, phase):
    """Untimed lifetime markers; current RSS is available on the Linux VM."""
    rss = None
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1])
                break
    row = {"event": "host_lifetime", "arm": arm, "phase": phase,
           "host_rss_kib": rss, "host_peak_rss_kib_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           "cuda_allocated_bytes": torch.cuda.memory_allocated()}
    record(row)
    return row


def trim_host_heap(record, arm):
    """Linux-only checker integration: return unused libc pages after released snapshots."""
    before = host_progress(record, arm, "before_host_heap_trim")
    result = {"event": "host_heap_trim", "arm": arm, "platform": sys.platform,
              "attempted": sys.platform.startswith("linux"), "available": False, "return_status": None,
              "scope": "checker CPU heap after gc/reference release and before replay; no numerical gate"}
    if result["attempted"]:
        import ctypes
        try:
            trim = ctypes.CDLL(None).malloc_trim
            trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
            result["available"] = True
            result["return_status"] = int(trim(0))
        except (AttributeError, OSError) as exc:
            result["unavailable_reason"] = f"{type(exc).__name__}: {exc}"
    else:
        result["unavailable_reason"] = "Linux-only operation; libc is not loaded on this platform"
    after = host_progress(record, arm, "after_host_heap_trim")
    result.update({"host_rss_before_kib": before["host_rss_kib"], "host_rss_after_kib": after["host_rss_kib"]})
    record(result)


def prepare_fixtures(export_dir: Path, count: int):
    """Shared export reader and support helpers; the canonical order is hidden-major."""
    fixtures, supports, fingerprints = [], [], {}
    for index in range(count):
        path = export_dir / f"layer{index}" / f"{export.FORMAT}.pt"
        raw = export.read_layer(path)
        layer = compact_adapter(raw)
        fixtures.append(layer)
        up = export.hidden_of_slot(raw, "up") * raw["d_model"] + raw["up_idx"].long()
        down = raw["down_idx"].long() * raw["d_ff"] + export.hidden_of_slot(raw, "down")
        supports.append((up, down))
        fingerprints[f"layer{index}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return fixtures, supports, fingerprints


def model_gradient_coverage(model, evaluation) -> dict:
    expected = {name: parameter.shape for name, parameter in model.named_parameters() if ".mlp." not in name}
    found = evaluation[2]
    return {"pass": set(found) == set(expected) and all(found[name].shape == shape for name, shape in expected.items() if name in found),
            "expected_non_ffn_tensors": len(expected), "observed_non_ffn_tensors": len(found),
            "missing": sorted(set(expected) - set(found)), "unexpected": sorted(set(found) - set(expected)),
            "shape_mismatch": sorted(name for name in set(found) & set(expected) if found[name].shape != expected[name]),
            "ffn_layers": len(evaluation[1])}


def all_finite(evaluation) -> bool:
    return math.isfinite(evaluation[0]) and all(bool(torch.isfinite(t).all()) for row in evaluation[1] for t in row) \
        and all(bool(torch.isfinite(t).all()) for t in evaluation[2].values())


def hffn_evaluation(model, ids, fixtures):
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    runtime = model.hffn_runtime
    ffn = []
    for index, layer in enumerate(fixtures):
        gradients = runtime.gradients(index)
        permutation = order(layer).to(DEVICE)
        hidden_down = torch.empty_like(gradients["out_val"])
        hidden_down[permutation] = gradients["out_val"]
        ffn.append((gradients["up_val"].clone(), hidden_down, gradients["b_up"].clone(), gradients["b_down"].clone()))
    other = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
             if ".mlp." not in name and parameter.grad is not None}
    return float(loss), ffn, other


class KeepGradProxy:
    """Retain dense-control gradients while delegating all collection to the shared helper."""
    def __init__(self, model):
        self.model = model

    def zero_grad(self, set_to_none=True):
        return self.model.zero_grad(set_to_none=False)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


def attention_context(backend):
    if backend == "math":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.MATH)
    return nullcontext()


def first_evaluation(name, evaluate, record):
    """Record actual first-step attention operators; never a timed window."""
    flags = {key: getattr(torch.backends.cuda, key + "_sdp_enabled")()
             for key in ("math", "flash", "mem_efficient", "cudnn")}
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        result = evaluate()
    operators = sorted({event.key for event in profile.key_averages()
                        if "attention" in event.key or "scaled_dot" in event.key})
    record({"event": "attention_trace", "arm": name, "enabled_backends": flags,
            "operators": operators, "scope": "first functional evaluation; no profiler durations read"})
    if flags == {"math": True, "flash": False, "mem_efficient": False, "cudnn": False}:
        if "aten::_scaled_dot_product_attention_math" not in operators:
            raise AssertionError("Math policy did not produce the declared attention operator")
    del profile
    return result


class DenseOptimizers:
    """Test control only: identical PyTorch non-FFN AdamW and a separately masked dense FFN AdamW."""
    def __init__(self, model, supports) -> None:
        named = list(model.named_parameters())
        self.other = torch.optim.AdamW([parameter for name, parameter in named if ".mlp." not in name], fused=True, **ts.HYPER)
        self.ffn = torch.optim.AdamW([parameter for name, parameter in named if ".mlp." in name], fused=True, **ts.HYPER)
        self.masks = {}
        for block, (up, down) in zip(model.gpt_neox.layers, supports):
            for weight, indices in ((block.mlp.dense_h_to_4h.weight, up), (block.mlp.dense_4h_to_h.weight, down)):
                mask = torch.zeros_like(weight, dtype=torch.bool)
                mask.reshape(-1)[indices.to("cuda")] = True
                self.masks[weight] = mask

    def zero_grad(self) -> None:
        self.other.zero_grad(set_to_none=False)
        self.ffn.zero_grad(set_to_none=False)

    def step(self) -> None:
        with torch.no_grad():
            for parameter, mask in self.masks.items():
                if parameter.grad is None:
                    raise RuntimeError("dense control is missing an FFN gradient")
                parameter.grad.mul_(mask)       # no update re-mask: an off-support write must fail the check
            self.other.step()
            self.ffn.step()

    def inactive_check(self, *, gradients: bool) -> dict:
        failures = []
        for index, (parameter, mask) in enumerate(self.masks.items()):
            checks = {"parameter": parameter}
            if gradients:
                checks["gradient"] = parameter.grad
            for key, value in self.ffn.state.get(parameter, {}).items():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    checks[key] = value
            for name, value in checks.items():
                if value is None or bool(torch.count_nonzero(value.masked_select(~mask))):
                    failures.append({"matrix": index, "tensor": name})
        return {"pass": not failures, "failures": failures, "matrices": len(self.masks)}


class LayerView:
    """Adapter for the unchanged isolated-layer gate3, on a selected layer after a full model update."""
    def __init__(self, model, index, layer, supports, optimizer=None):
        self.model, self.index, self.layer, self.supports, self.optimizer = model, index, layer, supports, optimizer

    def _dense_tensors(self):
        mlp = self.model.gpt_neox.layers[self.index].mlp
        up, down = self.supports[self.index]
        permutation = order(self.layer)
        return {"up_val": (mlp.dense_h_to_4h.weight, up), "b_up": (mlp.dense_h_to_4h.bias, None),
                "out_val": (mlp.dense_4h_to_h.weight, down[permutation]), "b_down": (mlp.dense_4h_to_h.bias, None)}

    def parameters(self, unused):
        if self.optimizer is None:
            return self.model.hffn_runtime.parameters(self.index)
        return {name: parameter.detach() if flat is None else parameter.detach().reshape(-1).index_select(0, flat.to(DEVICE))
                for name, (parameter, flat) in self._dense_tensors().items()}

    def moments(self, unused):
        if self.optimizer is None:
            return self.model.hffn_runtime.moments(self.index)
        return {name: tuple(self.optimizer.ffn.state[parameter][key] if flat is None else
                           self.optimizer.ffn.state[parameter][key].reshape(-1).index_select(0, flat.to(DEVICE))
                           for key in ("exp_avg", "exp_avg_sq")) for name, (parameter, flat) in self._dense_tensors().items()}


def initial_replay(fixtures):
    state = []
    for layer in fixtures:
        start = ts.hidden_major_params(layer)  # immutable FP32 views of the existing fixture arrays
        values = {name: tensor.double().clone() for name, tensor in start.items()}
        state.append({"p": values, "m": {name: torch.zeros_like(tensor) for name, tensor in values.items()},
                      "v": {name: torch.zeros_like(tensor) for name, tensor in values.items()},
                      "start": start})
    return state


def advance_replay(state, ffn, step: int) -> None:
    for row, gradients in zip(state, ffn):
        for name, gradient in zip(KINDS, gradients):
            row["p"][name], row["m"][name], row["v"][name] = ts.adamw_closed_form(
                row["p"][name], gradient.double().cpu(), row["m"][name], row["v"][name], step, **ts.HYPER)


def form_d(model, fixtures, supports, optimizer, state, masks) -> dict:
    """Cumulative own-gradient replay (form D): relative to max(|reference|, |initial|), fed by the actual model gradients, bound
    `ts.GATE5_OWN_GRADIENT_RELATIVE` at step 5. Recorded with `exceeded` (the former bound) as a limit; it no longer gates (rule
    of the optimiser update level)."""
    rows, finite, passed = [], True, True
    for index, (layer, expected, mask) in enumerate(zip(fixtures, state, masks)):
        values = LayerView(model, index, layer, supports, optimizer).parameters(0)
        permutation = order(layer)
        for group, name in (("g_up", "up_val"), ("g_down", "down_val")):
            value = values["out_val" if name == "down_val" else name].detach().double().cpu()
            if name == "down_val":
                hidden = torch.empty_like(value)
                hidden[permutation] = value
                value = hidden
            conditioned = int(mask[group].sum())
            scale = torch.maximum(expected["p"][name].abs(), expected["start"][name].double().abs()).clamp_min(1e-30)
            error = float(((value - expected["p"][name]).abs() / scale)[mask[group]].max()) if conditioned else float("inf")
            ok = conditioned > 0 and error <= ts.GATE5_OWN_GRADIENT_RELATIVE
            rows.append({"layer": index, "parameter": name, "conditioned": conditioned, "relative_error": error,
                         "tolerance": ts.GATE5_OWN_GRADIENT_RELATIVE, "pass": ok})
            passed &= ok
        finite &= all(bool(torch.isfinite(tensor).all()) for tensor in values.values())
    return {"pass": passed and finite, "exceeded": not passed, "finite": finite, "steps": STEPS, "groups": rows,
            "gating": False,
            "scope": "FFN weight values; recorded limit of the cumulative own-gradient replay, not a gate; per-update judgement in `update_gate`"}


def layer_snapshot(name, model, index, layer, supports, optimizer) -> dict:
    """Float32 device copies `{runtime_name: (p, m, v)}` of the compact state of one layer, in the build's own order (output-major
    `out_val`). The dense32 control is read at its active support, through `LayerView`; before its first step the fused AdamW holds no
    state and the moments are exact zeros (README, Update gate)."""
    if name == "hffn":
        runtime = model.hffn_runtime
        values, moments = runtime.parameters(index), runtime.moments(index)
        return {n: (values[n].detach().clone(), moments[n][0].detach().clone(), moments[n][1].detach().clone())
                for n in ug.RUNTIME_NAME.values()}
    out = {}
    for n, (parameter, flat) in LayerView(model, index, layer, supports, optimizer)._dense_tensors().items():
        def active(tensor):
            return tensor.detach().clone() if flat is None else tensor.detach().reshape(-1).index_select(0, flat.to(DEVICE))
        state = optimizer.ffn.state.get(parameter)
        p = active(parameter)
        out[n] = (p, active(state["exp_avg"]), active(state["exp_avg_sq"])) if state else (p, torch.zeros_like(p), torch.zeros_like(p))
    return out


def layer_counters(name, model, index, layer, supports, optimizer) -> list:
    """The build's optimiser step counter(s) of one layer: HFFN `runtime.steps[index]`, dense32 the `step` of each FFN parameter."""
    if name == "hffn":
        return [model.hffn_runtime.steps[index]]
    return [int(optimizer.ffn.state[parameter]["step"]) if parameter in optimizer.ffn.state else 0
            for parameter, _ in LayerView(model, index, layer, supports, optimizer)._dense_tensors().values()]


def model_snapshots(name, model, fixtures, supports, optimizer) -> list:
    return [layer_snapshot(name, model, index, layer, supports, optimizer) for index, layer in enumerate(fixtures)]


def judge_step(gate, name, model, step, pre, fixtures, supports, optimizer, gradients, masks) -> dict:
    """Compare the update `step` of every layer; the post-state is read layer by layer and released at once."""
    worst = {"value": 0.0, "m": 0.0, "v": 0.0}
    for index, layer in enumerate(fixtures):
        post = layer_snapshot(name, model, index, layer, supports, optimizer)
        row = gate.update(index, step, pre[index], post, dict(zip(GROUPS, gradients[index])), masks[index], order(layer),
                          layer_counters(name, model, index, layer, supports, optimizer))
        for parameter in ("up_val", "down_val"):
            for key, value in row.get(parameter, {}).items():
                worst[key] = ts._worse(worst[key], value)
        pre[index] = None
        del post
    return worst


def gradient_identity(name, model, fixtures, supports, read) -> dict:
    """Bitwise: the gradient the update gate was fed (`read`, the host copies `[(g_up, g_down, gb_up, gb_down)]` taken at the
    evaluation) is what the fused step is about to consume. HFFN: `runtime.gradients(i)` with `out_val` brought back to hidden-major;
    dense32: `.grad` at the active support and the bias gradients. Called just before `optimizer.step()`, which re-masks the dense
    gradients without changing an active entry."""
    mismatches = []
    for index, layer in enumerate(fixtures):
        if name == "hffn":
            live = model.hffn_runtime.gradients(index)
            hidden_down = torch.empty_like(live["out_val"])
            hidden_down[order(layer).to(DEVICE)] = live["out_val"]
            current = (live["up_val"], hidden_down, live["b_up"], live["b_down"])
        else:
            block = model.gpt_neox.layers[index].mlp
            up_flat, down_flat = supports[index]
            current = (block.dense_h_to_4h.weight.grad.reshape(-1).index_select(0, up_flat.to(DEVICE)),
                       block.dense_4h_to_h.weight.grad.reshape(-1).index_select(0, down_flat.to(DEVICE)),
                       block.dense_h_to_4h.bias.grad, block.dense_4h_to_h.bias.grad)
        for group, live_tensor, seen in zip(GROUPS, current, read[index]):
            if live_tensor.dtype != seen.dtype or live_tensor.shape != seen.shape or not torch.equal(live_tensor.cpu(), seen):
                mismatches.append({"layer": index, "group": group})
    return {"pass": not mismatches, "layers": len(fixtures), "mismatches": mismatches}


def merge_identity(rows) -> dict:
    return {"pass": bool(rows) and all(row["pass"] for row in rows), "steps": len(rows),
            "mismatches": [{"step": step, **item} for step, row in enumerate(rows, 1) for item in row["mismatches"]]}


def model_layers(model) -> int:
    """The layer count the gates are held to: read from the model, never from the fixtures or from what a gate saw."""
    return len(model.gpt_neox.layers)


def fixture_masks(reference, control):
    return [ts.conditioned_masks(dict(zip(GROUPS, ref)), dict(zip(GROUPS, got)))
            for ref, got in zip(reference[1], control[1])]


def finite_optimizer(model, optimizer, name) -> bool:
    bases = (optimizer.base,) if name == "hffn" else (optimizer.other, optimizer.ffn)
    finite = all(bool(torch.isfinite(value).all()) for base in bases for state in base.state.values()
                 for value in state.values() if torch.is_tensor(value))
    if name == "hffn":
        finite &= all(bool(torch.isfinite(value).all()) for index in range(len(model.gpt_neox.layers))
                      for pair in model.hffn_runtime.moments(index).values() for value in pair)
    return finite


def finite_parameters(model, name) -> bool:
    finite = all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())
    if name == "hffn":
        finite &= all(bool(torch.isfinite(value).all()) for index in range(len(model.gpt_neox.layers))
                      for value in model.hffn_runtime.parameters(index).values())
    return finite


CANONICAL_SUPPORT = ("up_offsets", "up_idx", "out_offsets", "out_hidden")


def support_identity(runtime, canonical_before, transposed_before) -> bool:
    """Compare the actual CUDA tensors read by kernels, including transposed backward support."""
    def same(before, current):
        return before.dtype == current.dtype and before.shape == current.shape and torch.equal(before, current.detach().cpu())
    return all(same(before[key], runtime.runtime._tensors[index][key]) for index, before in enumerate(canonical_before)
               for key in CANONICAL_SUPPORT) and all(
        set(before) == set(runtime.supports(index)) and all(same(value, runtime.supports(index)[key]) for key, value in before.items())
        for index, before in enumerate(transposed_before))


def arm_run(name, model_dir, export_dir, ids_cpu, fixtures, supports, reference, masks, record, expected_probe=None, faults=()):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ids = ids_cpu.to("cuda")
    if name == "hffn":
        model = s1_load(model_dir, export_dir, "cuda", max_tokens=BATCH * SEQ_LEN).eval().requires_grad_(True)
        optimizer = s1_optimizer(model, fused=True, **ts.HYPER)
    else:
        model = shared_model.load_pythia(model_dir, export_dir, "dense", None, "cpu")
        model = model.to(device="cuda", dtype=torch.float64 if name == "dense64" else torch.float32).eval().requires_grad_(True)
        optimizer = None if name == "dense64" else DenseOptimizers(model, supports)
    model.config.use_cache = False
    helper_model = None if name == "hffn" else KeepGradProxy(model)
    torch.cuda.synchronize()
    allocated_after_loader = torch.cuda.memory_allocated()
    record({"event": "loader", "arm": name, "allocated_after_loader": allocated_after_loader})
    output_checks = []
    def finite_output(_module, _inputs, output):
        finite = bool(torch.isfinite(output.logits).all()) and bool(torch.isfinite(output.loss))
        output_checks.append(finite)
        if not finite:
            raise AssertionError("non-finite model logits or loss")
    model.register_forward_hook(finite_output)
    device_supports = [(up.to("cuda"), down.to("cuda")) for up, down in supports]
    if name == "hffn":
        runtime = model.hffn_runtime
        for index, ((up, down), layer) in enumerate(zip(supports, fixtures)):
            got_up, got_down = runtime.flat_support(index)
            if not torch.equal(got_up, up) or not torch.equal(got_down, down[order(layer)]):
                raise AssertionError("S1 active gradient order differs from the declared canonical support")
        workspace_before = runtime.workspace_stats()
        support_before = [{key: value.detach().cpu().clone() for key, value in runtime.supports(index).items()}
                          for index in range(len(fixtures))]
        canonical_before = [{key: runtime.runtime._tensors[index][key].detach().cpu().clone() for key in CANONICAL_SUPPORT}
                            for index in range(len(fixtures))]
    else:
        workspace_before, support_before, canonical_before = None, None, None
    # The initial-only dense32 probe is released before HFFN. HFFN releases the
    # dense64 reference after its first comparison, before allocating replay.
    # The fresh dense32 update arm uses probe scalars/masks only. Later steps
    # transfer active FFN gradients to CPU, with non-FFN coverage checked on GPU.
    replay = None
    evaluation_cpu, warm_allocated, update_rows, inactive_rows = None, None, [], []
    initial_distance, initial_details, initial_model_gate = None, None, None
    initial_fingerprints, initial_identity = None, None
    optimizer_updates = 0
    update_gate = ug.UpdateGate(name, DEVICE, faults) if name in ("hffn", "dense32") else None    # not `gate`: tg.judge binds that
    step_errors, identity_rows = [], []
    steps = 1 if name in ("dense64", "dense32_probe") else STEPS
    for step in range(1, steps + 1):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=False) if name == "hffn" else optimizer.zero_grad()
        def evaluate():
            return (hffn_evaluation(model, ids, fixtures) if name == "hffn"
                    else tg.dense_gradients(helper_model, ids, device_supports))
        evaluation = first_evaluation(name, evaluate, record) if step == 1 else evaluate()
        coverage = model_gradient_coverage(model, evaluation)
        if len(evaluation[1]) != len(fixtures) or not coverage["pass"] or not all_finite(evaluation):
            raise AssertionError(f"{name}: incomplete or non-finite model gradient evaluation")
        current = cpu_evaluation(evaluation) if step == 1 else (
            evaluation[0], [tuple(t.detach().cpu() for t in row) for row in evaluation[1]], {})
        del evaluation
        if step == 1:
            record({"id": f"{name}_gradients", "arm": name, "loss": current[0], "coverage": coverage,
                    "active_ffn_shapes": [[list(t.shape) for t in row] for row in current[1]], "finite": True})
            if name == "dense64":
                evaluation_cpu = current
            else:
                if name == "dense32":
                    if expected_probe is None:
                        raise AssertionError("the dense32 update arm requires the frozen initial-only probe")
                    host_progress(record, name, "before_initial_identity")
                    initial_fingerprints = evaluation_fingerprints(current)
                    initial_identity = initial_fingerprints == expected_probe["initial_fingerprints"]
                    record({"id": "dense32_initial_identity", "pass": initial_identity,
                            "scope": "exact loss bits and all four FFN gradient groups in every layer plus every non-FFN gradient",
                            "fingerprints": initial_fingerprints})
                    if not initial_identity:
                        raise AssertionError("fresh dense32 initial loss or gradients differ from the initial-only probe")
                    initial_distance = expected_probe["to_float64"]
                    initial_details = expected_probe["per_tensor_to_float64"]
                    host_progress(record, name, "after_initial_identity")
                else:
                    reference_evaluation = reference["evaluation"]
                    host_progress(record, name, "before_distance")
                    initial_distance = tg.distance(current, reference_evaluation)
                    initial_details = distance_details(current, reference_evaluation)
                    host_progress(record, name, "after_distance")
                if name == "dense32_probe":
                    host_progress(record, name, "before_conditioning_and_fingerprints")
                    masks = fixture_masks(reference_evaluation, current)
                    reference["control_distance"] = initial_distance
                    initial_fingerprints = evaluation_fingerprints(current)
                    del reference_evaluation, current
                    gc.collect()
                    host_progress(record, name, "after_conditioning_and_release")
                    break                    # this probe performs no optimizer update
                elif name == "hffn":
                    distances = {"dense32": reference["control_distance"], "hffn_s1": initial_distance}
                    gate, parity_pass = tg.judge(distances, distances["dense32"], ("s1",))
                    initial_model_gate = {"pass": parity_pass, "gate": gate, "to_float64": distances}
                    record({"id": "hffn_initial_model_gate", **initial_model_gate})
                    reference.clear()
                    del reference_evaluation
                    host_progress(record, name, "after_reference_release")
                current = current[0], current[1], {}    # no retained non-FFN gradient snapshots
                gc.collect()
                trim_host_heap(record, name)
                host_progress(record, name, "before_replay_allocation")
                replay = initial_replay(fixtures)
                host_progress(record, name, "after_replay_allocation")
        if optimizer is None:
            break
        host_progress(record, name, f"before_replay_step_{step}")
        advance_replay(replay, current[1], step)
        host_progress(record, name, f"after_replay_step_{step}")
        # The step is fused over every layer: the pre-update state of all of them exists before the one call. Re-read at each step
        # (the previous post-state was released, so that the allocation read below still sees no growth).
        pre = model_snapshots(name, model, fixtures, supports, optimizer)
        snapshot_device_bytes = sum(t.numel() * t.element_size() for state in pre for triple in state.values() for t in triple)
        identity_rows.append(gradient_identity(name, model, fixtures, supports, current[1]))
        optimizer.step()
        optimizer_updates += 1
        torch.cuda.synchronize()
        step_errors.append({"step": step, **judge_step(update_gate, name, model, step, pre, fixtures, supports, optimizer, current[1], masks)})
        record({"event": "update_gate_step", "arm": name, "snapshot_device_bytes": snapshot_device_bytes,
                "peak_allocated_so_far": torch.cuda.max_memory_allocated(), **step_errors[-1]})
        del pre
        if name == "dense32":
            inactive = optimizer.inactive_check(gradients=True)
            inactive_rows.append(inactive)
            if not inactive["pass"]:
                raise AssertionError("dense iso-support control updated an inactive FFN slot")
        if step == 1:
            for index, (layer, gradients, mask) in enumerate(zip(fixtures, current[1], masks)):
                view = LayerView(model, index, layer, supports, None if name == "hffn" else optimizer)
                update_rows.append({"layer": index, **ts.gate3(view, layer, dict(zip(GROUPS, gradients)), mask,
                                                            ts.hidden_major_params(layer))})
        optimizer.zero_grad(set_to_none=False) if name == "hffn" else optimizer.zero_grad()
        del current
        gc.collect()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        if step == 1:
            warm_allocated = allocated
        supports_unchanged = support_identity(model.hffn_runtime, canonical_before, support_before) if name == "hffn" else True
        record({"id": f"{name}_step_{step}", "arm": name, "step": step, "finite_parameters":
                finite_parameters(model, name),
                "finite_optimizer_state": finite_optimizer(model, optimizer, name),
                "supports_unchanged": supports_unchanged,
                "allocated_after_complete_step": allocated, "warm_allocated": warm_allocated,
                "host_peak_rss_kib_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "no_growth_after_warm_step": allocated == warm_allocated})
        if not supports_unchanged:
            raise AssertionError("actual CUDA canonical or transposed support changed during the model update")
    result = {"arm": name, "coverage": coverage, "first_step_update": update_rows,
              "allocated_after_loader": allocated_after_loader,
              "optimizer_updates": optimizer_updates,
              "finite_model_outputs": all(output_checks), "forwards_observed": len(output_checks)}
    if name != "dense64":
        result["to_float64"] = initial_distance
        result["per_tensor_to_float64"] = initial_details
        if initial_model_gate is not None:
            result["initial_model_gate"] = initial_model_gate
        if initial_fingerprints is not None:
            result["initial_fingerprints"] = initial_fingerprints
        if initial_identity is not None:
            result["initial_identity_pass"] = initial_identity
    if name == "dense32_probe":
        result["scope"] = "initial-only control probe; distances/masks/fingerprints, no judged update arm"
    elif name != "dense64":
        result["form_d"] = form_d(model, fixtures, supports, None if name == "hffn" else optimizer, replay, masks)
        result["update_gate"] = update_gate.result(model_layers(model), STEPS)
        result["update_gate_steps"] = step_errors
        result["gradient_identity"] = merge_identity(identity_rows)
        result["update_pass"] = (all(row["pass"] for row in update_rows) and result["update_gate"]["pass"] and result["form_d"]["finite"]
                                 and result["gradient_identity"]["pass"] and result["gradient_identity"]["steps"] == STEPS)
        if name == "dense32":
            result["inactive_control"] = {"pass": all(row["pass"] for row in inactive_rows), "steps": len(inactive_rows)}
        else:
            runtime = model.hffn_runtime
            result["workspace_before"] = workspace_before
            result["workspace_after"] = runtime.workspace_stats()
            result["compact_and_context_bytes"] = runtime.nbytes()
            result["supports_unchanged"] = support_identity(runtime, canonical_before, support_before)
            result["support_identity_scope"] = {"canonical_cuda_tensors": CANONICAL_SUPPORT,
                                                "transposed_cuda_tensors": sorted(support_before[0]),
                                                "checked_after_each_step": True}
            result["workspace_stable"] = all(runtime.workspace_stats()[key] == workspace_before[key]
                                             for key in ("allocated", "capacity", "growths", "train_growths")) \
                and runtime.workspace_stats()["live"] == 0
    torch.cuda.synchronize()
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    result["peak_bound_bytes"] = PEAK_BOUND
    result["peak_within_bound"] = result["peak_allocated_bytes"] <= PEAK_BOUND
    result["peak_scope"] = "model loading, gradients, optimizer steps and functional gate temporaries; sequential arms"
    result["host_peak_rss_kib_process"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record({"id": name, **result})
    del model, helper_model, optimizer, replay, device_supports, ids
    gc.collect()
    torch.cuda.empty_cache()
    return evaluation_cpu, masks, result


def build_model(name, model_dir, export_dir, supports, hyper):
    """A fresh build of one judged arm, as `arm_run` builds it; `hyper` reaches the HFFN optimiser only (a build-side fault)."""
    if name == "hffn":
        model = s1_load(model_dir, export_dir, "cuda", max_tokens=BATCH * SEQ_LEN).eval().requires_grad_(True)
        optimizer = s1_optimizer(model, fused=True, **hyper)
    else:
        model = shared_model.load_pythia(model_dir, export_dir, "dense", None, "cpu")
        model = model.to(device="cuda", dtype=torch.float32).eval().requires_grad_(True)
        optimizer = DenseOptimizers(model, supports)
    model.config.use_cache = False
    return model, optimizer


def ffn_gradient_max(name, model, optimizer, fixtures) -> float:
    """Largest absolute FFN gradient the next step will consume (compact groups for HFFN, every dense FFN gradient otherwise)."""
    if name == "hffn":
        return max(float(t.abs().max()) for index in range(len(fixtures)) for t in model.hffn_runtime.gradients(index).values())
    return max(float(p.grad.abs().max()) for p in optimizer.ffn.param_groups[0]["params"])


def decay_arm(name, model_dir, export_dir, ids, fixtures, supports, masks, faults=(), hyper=None) -> dict:
    """Pass B on a fresh build: five fused steps on a zero loss gradient (`loss * 0`), judged by `update_gate.DecayGate`."""
    model, optimizer = build_model(name, model_dir, export_dir, supports, hyper or ts.HYPER)
    gate = ug.DecayGate(name, DEVICE, faults, STEPS)
    pre = model_snapshots(name, model, fixtures, supports, optimizer)
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=False) if name == "hffn" else optimizer.zero_grad()
        (model(input_ids=ids, labels=ids).loss * 0).backward()
        gate.gradient(ffn_gradient_max(name, model, optimizer, fixtures))
        optimizer.step()
        torch.cuda.synchronize()
    for index, layer in enumerate(fixtures):
        post = layer_snapshot(name, model, index, layer, supports, optimizer)
        gate.layer(index, pre[index], post, masks[index], order(layer),
                   layer_counters(name, model, index, layer, supports, optimizer))
        pre[index] = None
        del post
    result = gate.result(model_layers(model))
    del model, optimizer, pre, gate
    gc.collect()
    torch.cuda.empty_cache()
    return result


def live_arm(name, model_dir, export_dir, ids, fixtures, supports, masks, hyper) -> dict:
    """Pass A on a fresh build with real gradients and no replay or reference (the build-side fault arms), nominal gate."""
    model, optimizer = build_model(name, model_dir, export_dir, supports, hyper)
    gate = ug.UpdateGate(name, DEVICE)
    identity_rows = []
    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=False) if name == "hffn" else optimizer.zero_grad()
        evaluation = hffn_evaluation(model, ids, fixtures)
        gradients = [tuple(t.detach().cpu() for t in row) for row in evaluation[1]]
        del evaluation
        pre = model_snapshots(name, model, fixtures, supports, optimizer)
        identity_rows.append(gradient_identity(name, model, fixtures, supports, gradients))
        optimizer.step()
        torch.cuda.synchronize()
        judge_step(gate, name, model, step, pre, fixtures, supports, optimizer, gradients, masks)
        del pre
    result = gate.result(model_layers(model), STEPS)
    result["gradient_identity"] = merge_identity(identity_rows)
    del model, optimizer, gate
    gc.collect()
    torch.cuda.empty_cache()
    return result


def decay_run(name, arm_id, model_dir, export_dir, ids_cpu, fixtures, supports, masks, record, faults=()):
    """Planned stage `hffn_decay` / `dense32_decay`."""
    gc.collect()
    trim_host_heap(record, arm_id)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result = decay_arm(name, model_dir, export_dir, ids_cpu.to("cuda"), fixtures, supports, masks, faults)
    result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  peak_bound_bytes=PEAK_BOUND, host_peak_rss_kib_process=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result["peak_within_bound"] = result["peak_allocated_bytes"] <= PEAK_BOUND
    record({"id": arm_id, **result})
    return result


def fault_run(arm_id, change, model_dir, export_dir, ids_cpu, fixtures, supports, masks, record):
    """Planned stage `hffn_build_fault_*`: the HFFN optimiser is built with a changed hyper-parameter, the gate keeps the nominal
    ones. Detected on a judged group when that group fails pass A or the decay-only pass (README, Update gate)."""
    gc.collect()
    trim_host_heap(record, arm_id)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ids, hyper = ids_cpu.to("cuda"), {**ts.HYPER, **change}
    pass_a = live_arm("hffn", model_dir, export_dir, ids, fixtures, supports, masks, hyper)
    decay = decay_arm("hffn", model_dir, export_dir, ids, fixtures, supports, masks, (), hyper)
    detected = ug.build_fault_detected(pass_a, decay)
    result = {"arm": "hffn", "build_fault": change, "detection": detected, "pass_a_group_pass": pass_a["group_pass"],
              "decay_group_pass": decay["group_pass"], "gradient_identity": pass_a["gradient_identity"],
              "layers_complete": pass_a["layers_complete"] and decay["layers_complete"],
              "decay_gradient_max": decay["gradient_max"], "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_bound_bytes": PEAK_BOUND}
    result["peak_within_bound"] = result["peak_allocated_bytes"] <= PEAK_BOUND
    record({"id": arm_id, **result})
    return result


def build_faults(hyper: dict) -> dict:
    """The four changed constructor hyper-parameters of the HFFN optimiser (README, Update gate), keyed by planned-id suffix."""
    return {"weight_decay_zero": {"weight_decay": 0.0}, "eps_times_1000": {"eps": hyper["eps"] * 1000},
            "beta2_0.99": {"betas": (hyper["betas"][0], 0.99)}, "lr_times_1.0001": {"lr": hyper["lr"] * 1.0001}}


def distance_details(evaluation, reference):
    if len(evaluation[1]) != len(reference[1]) or set(evaluation[2]) != set(reference[2]):
        raise AssertionError("gradient names or layer coverage differ between the model arms")
    rows = []
    for index, (got, ref) in enumerate(zip(evaluation[1], reference[1])):
        if len(got) != 4 or len(ref) != 4:
            raise AssertionError("a model arm is missing an active FFN gradient group")
        for name, a, b in zip(KINDS, got, ref):
            if a.shape != b.shape:
                raise AssertionError("active FFN gradient shape mismatch")
            rows.append({"family": "ffn", "layer": index, "tensor": name, "distance": tg.gap(a, b)})
    for name, ref in reference[2].items():
        got = evaluation[2][name]
        if got.shape != ref.shape:
            raise AssertionError("non-FFN gradient shape mismatch")
        rows.append({"family": "embedding" if "embed" in name else "attention" if ".attention." in name else
                     "norm" if "layernorm" in name or "layer_norm" in name else "other",
                     "tensor": name, "distance": tg.gap(got, ref)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "run"), required=True)
    parser.add_argument("--attention-backend", choices=("default", "math"), default="math",
                        help="S2 qualification requires the user-approved Math policy; default is diagnostic smoke only")
    parser.add_argument("--export", choices=TARGETS, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--tokens", type=Path, default=TOKENS)
    parser.add_argument("--base")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assets = Path(os.environ["HFFN_ASSETS"])
    model_dir = (args.model_dir or assets / "pythia/models/base" / args.export.split("_")[0]).resolve()
    export_dir = (args.export_dir or Path(os.environ.get("HFFN_EXPORTS", assets / "pythia/exports")) / args.export / "export").resolve()
    args.tokens = args.tokens.resolve()
    reserved = ("config.json", "provenance.json", "raw.jsonl", "manifest.json", "summary.json", "EXIT")
    if any((args.out / name).exists() for name in reserved):
        raise RuntimeError("the output folder contains existing run records; none is overwritten")
    args.out.mkdir(parents=True, exist_ok=True)
    config = {"mode": args.mode, "export": args.export, "model_dir": str(model_dir), "export_dir": str(export_dir),
              "batch": BATCH, "seq_len": SEQ_LEN, "steps": STEPS, "tokens": str(args.tokens), "fused_adamw": True,
              "attention_backend": args.attention_backend, "dense_helper_set_to_none": False,
              "attention_backend_scope": "S2 approved Math qualification; same setting for dense64, dense32_probe, HFFN and dense32",
              "eval_mode_with_gradients": True, "use_cache": False, "tf32": False, "timing": "unmeasured",
              "claim": "functional qualification only; no default promotion", "memory_peak_bound_bytes": PEAK_BOUND,
              "host_gradient_lifetime": "initial-only dense32 probe reduced to scalar distances, per-tensor diagnostics, conditioning masks "
                                        "and exact loss/all-gradient fingerprints, then released; dense64 released after S1 first comparison "
                                        "before replay; fresh dense32 five-update arm starts after S1 release and verifies probe identity "
                                        "before its first update; later steps copy FFN gradients only",
              "initial_identity_scope": "loss Python-float IEEE754 bits; dtype, shape and SHA256 of all four FFN gradient groups "
                                        "in every layer and every non-FFN gradient, without cast",
              "judged_update_arms": {"hffn": STEPS, "dense32": STEPS}, "probe_optimizer_updates": 0,
              "host_heap_release": "Linux-only ctypes malloc_trim(0) after gc/reference release and before replay; RSS/return status diagnostic only",
              "replay_storage": "CPU float64 p/m/v own-gradient replay; immutable start aliases existing fixture FP32 arrays, "
                                "promoted to float64 per tensor for the unchanged form-D scale; allocated after initial reduction/reference release"}
    rows, results, error, source = [], {}, None, None
    fault_stage = args.export == TARGETS[0]      # detection is shown on the real 70M stage only (README, Update gate)
    planned = planned_ids(fault_stage)
    config["planned_stages"] = planned
    config["update_gate"] = {"gate": "update_gate", "steps": STEPS, "faults_injected": fault_stage,
                             "build_fault_arms": list(BUILD_FAULT_NAMES) if fault_stage else [],
                             "judged": "value, m, v per update on the conditioned entries of up_val and down_val; decay-only pass; "
                                       "form D recorded, not gating"}
    manifest = {"status": "running", "mode": args.mode, "planned": len(planned), "done": 0,
                "planned_ids": planned, "done_ids": [], "full_coverage": False, "phase": "setup"}
    write_json(args.out / "config.json", config)
    replace_json(args.out / "manifest.json", manifest)
    with (args.out / "raw.jsonl").open("x", encoding="utf-8") as raw:
        def record(row):
            row = finite_json(row)
            rows.append(row)
            raw.write(json.dumps(row, allow_nan=False) + "\n")
            raw.flush()
            print(json.dumps(row, allow_nan=False), flush=True)
            observed = [name for name in planned if any(item.get("id") == name for item in rows)]
            manifest.update({"done": len(observed), "done_ids": observed, "phase": row.get("phase", row.get("id", row.get("event"))),
                             "full_coverage": observed == planned})
            replace_json(args.out / "manifest.json", manifest)
        try:
            if args.mode == "run" and args.attention_backend != "math":
                raise ValueError("S2 qualification requires the approved Math SDPA policy; default is smoke-only")
            source = context_helpers.provenance(args.base)
            source["attention_backend"] = args.attention_backend
            source["attention_backend_scope"] = config["attention_backend_scope"]
            source["dense_helper_set_to_none"] = False
            source["host_gradient_lifetime"] = config["host_gradient_lifetime"]
            source["replay_storage"] = config["replay_storage"]
            source["host_heap_release"] = config["host_heap_release"]
            source["planned_stages"] = planned
            source["initial_identity_scope"] = config["initial_identity_scope"]
            source["judged_update_arms"] = config["judged_update_arms"]
            source["probe_optimizer_updates"] = 0
            source["source_hashes"].update(extra_hashes(args.tokens))
            source["tree_sha256"] = digest_of(source["source_hashes"])
            source["bound_tree"] = {"tree_sha256": source["tree_sha256"], "problems_at_end": None}
            write_json(args.out / "provenance.json", source)
            import_runtime()
            config["adamw_hyperparameters"] = ts.HYPER
            replace_json(args.out / "config.json", config)
            source.update({"torch": torch.__version__, "cuda": torch.version.cuda, "cupy": cupy.__version__,
                           "gpu": torch.cuda.get_device_name(0), "hyperparameters": ts.HYPER,
                           "conditioning": ts.GATE3_CONDITIONING, "gate3_relative": ts.GATE3_RELATIVE,
                           "form_d_relative": ts.GATE5_OWN_GRADIENT_RELATIVE})
            model_config = json.loads((model_dir / "config.json").read_text())
            fixtures, supports, fingerprints = prepare_fixtures(export_dir, model_config["num_hidden_layers"])
            source["export_layer_sha256"] = fingerprints
            for name in ("config.json", "model.safetensors"):
                path = model_dir / name
                stat = path.stat()
                source.setdefault("checkpoint_files", {})[name] = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            ids_cpu = torch.tensor(json.loads(args.tokens.read_text())["tokens"][:BATCH * SEQ_LEN]).reshape(BATCH, SEQ_LEN)
            source["token_ids_sha256"] = hashlib.sha256(ids_cpu.numpy().tobytes()).hexdigest()
            replace_json(args.out / "provenance.json", source)
            record({"event": "setup", "export": args.export, "layers": len(fixtures), "batch": BATCH,
                    "seq_len": SEQ_LEN, "steps": STEPS, "hyperparameters": ts.HYPER, "base": source["base"]})
            with attention_context(args.attention_backend):
                reference_evaluation, _, results["dense64"] = arm_run("dense64", model_dir, export_dir, ids_cpu, fixtures, supports, None, None, record)
                reference = {"evaluation": reference_evaluation}
                del reference_evaluation
                _, masks, results["dense32_probe"] = arm_run("dense32_probe", model_dir, export_dir, ids_cpu, fixtures, supports, reference, None, record)
                host_progress(record, "dense32_probe", "after_model_release")
                faults = ug.FAULT_NAMES if fault_stage else ()
                _, _, results["hffn"] = arm_run("hffn", model_dir, export_dir, ids_cpu, fixtures, supports, reference, masks, record,
                                                faults=faults)
                if reference:
                    raise AssertionError("dense64 evaluation was retained after the HFFN first comparison")
                host_progress(record, "hffn", "after_model_release")
                _, _, results["dense32"] = arm_run("dense32", model_dir, export_dir, ids_cpu, fixtures, supports, None, masks, record,
                                                  expected_probe=results["dense32_probe"], faults=faults)
                for name in JUDGED_ARMS:
                    results[decay_id(name)] = decay_run(name, decay_id(name), model_dir, export_dir, ids_cpu, fixtures, supports, masks,
                                                         record, faults)
                if fault_stage:
                    for name, change in build_faults(ts.HYPER).items():
                        results[fault_id(name)] = fault_run(fault_id(name), change, model_dir, export_dir,
                                                                        ids_cpu, fixtures, supports, masks, record)
                    for name in JUDGED_ARMS:
                        results[name]["fault_detection"] = ug.detection(results[name]["update_gate"], results[decay_id(name)])
                        record({"event": "fault_detection", "arm": name, **results[name]["fault_detection"]})
            results["model_gate"] = {**results["hffn"]["initial_model_gate"],
                                     "per_tensor": {"dense32": results["dense32"]["per_tensor_to_float64"],
                                                    "hffn": results["hffn"]["per_tensor_to_float64"]},
                                     "criterion_scope": "shared aggregate FFN and non-FFN groups; per-tensor rows are diagnostics"}
            record({"id": "model_gate", **results["model_gate"]})
            for relative, digest in source["source_hashes"].items():
                path = Path(relative) if Path(relative).is_absolute() else REPO / relative
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise AssertionError("source or token file changed during model qualification")
            late = compare_bound(source["source_hashes"], bound_hashes(args.tokens, source["base"]))
            source["bound_tree"]["problems_at_end"] = late
            if late:
                raise AssertionError(f"the bound tree changed during the run: {late}")
            for name, initial in source["checkpoint_files"].items():
                stat = (model_dir / name).stat()
                if stat.st_size != initial["bytes"] or stat.st_mtime_ns != initial["mtime_ns"]:
                    raise AssertionError("checkpoint source changed during model qualification")
            for name, digest in fingerprints.items():
                if hashlib.sha256((export_dir / name / f"{export.FORMAT}.pt").read_bytes()).hexdigest() != digest:
                    raise AssertionError("compact export changed during model qualification")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            record({"event": "failure", "error": error})
            traceback.print_exc()
    done = [name for name in planned if name in results]
    full = done == planned
    traces = [row for row in rows if row.get("event") == "attention_trace"]
    trace_arms = {row["arm"] for row in traces}
    math_policy = (len(traces) == 4 and trace_arms == {"dense64", "dense32_probe", "hffn", "dense32"}
                   and all(row["enabled_backends"] == {"math": True, "flash": False, "mem_efficient": False, "cudnn": False}
                           and "aten::_scaled_dot_product_attention_math" in row["operators"] for row in traces))
    added = [name for name in planned if name not in ("dense64", "dense32_probe", "hffn", "dense32", "model_gate")]
    detection_arms = ("hffn", "dense32")
    checks = {"full_coverage": full, "model_parity": results.get("model_gate", {}).get("pass", False),
              "attention_policy": math_policy if args.attention_backend == "math" else len(traces) == 4,
              "peak_memory": full and all(results[name]["peak_within_bound"]
                                          for name in ("dense64", "dense32_probe", "hffn", "dense32") + tuple(added)),
              "updates": full and all(results[name]["update_pass"] for name in ("dense32", "hffn")),
              "decay_only": full and all(results[decay_id(name)]["pass"] for name in JUDGED_ARMS),
              "fault_detection": (not fault_stage) or (full and all(results[name]["fault_detection"]["pass"] for name in detection_arms)),
              "build_fault_detection": (not fault_stage) or (full and all(results[fault_id(name)]["detection"]["pass"]
                                                                          for name in BUILD_FAULT_NAMES)),
              "gradient_identity": full and all(results[name]["gradient_identity"]["pass"] for name in JUDGED_ARMS)
                                   and all(results[fault_id(name)]["gradient_identity"]["pass"] for name in BUILD_FAULT_NAMES if fault_stage),
              "layer_coverage": full and all(results[name]["update_gate"]["layers_complete"] and results[decay_id(name)]["layers_complete"]
                                             for name in JUDGED_ARMS)
                                and all(results[fault_id(name)]["layers_complete"] for name in BUILD_FAULT_NAMES if fault_stage),
              "source_identity": source is not None and source.get("bound_tree", {}).get("problems_at_end") == [],
              "initial_control_identity": results.get("dense32", {}).get("initial_identity_pass", False),
              "actual_update_count": full and all(results[name]["optimizer_updates"] == STEPS for name in ("dense32", "hffn"))
                                     and results["dense32_probe"]["optimizer_updates"] == 0,
              "inactive_control": results.get("dense32", {}).get("inactive_control", {}).get("pass", False),
              "support_identity": results.get("hffn", {}).get("supports_unchanged", False),
              "workspace_stability": results.get("hffn", {}).get("workspace_stable", False),
              "finite_parameters_and_states": all(row.get("finite_parameters", True) and row.get("finite_optimizer_state", True) for row in rows),
              "memory_no_growth": all(row.get("no_growth_after_warm_step", True) for row in rows)}
    passed = error is None and all(checks.values())
    if not (args.out / "config.json").exists():
        write_json(args.out / "config.json", config)
    if source is not None:
        source["tree_sha256"] = digest_of(source["source_hashes"])
        replace_json(args.out / "provenance.json", source)
    else:
        replace_json(args.out / "provenance.json", {"error": "source provenance unavailable"})
    write_json(args.out / "summary.json", {"passed": passed, "checks": checks, "error": error,
                                           "functional_only": True, "time_gate": "unmeasured; no promotion verdict",
                                           "host_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if passed else None,
                                           "arm_results": results if passed else "failed/partial run: diagnostics stay in raw.jsonl"})
    replace_json(args.out / "manifest.json", {"status": "passed" if passed else "failed", "mode": args.mode,
                                            "planned": len(planned), "done": len(done), "planned_ids": planned,
                                            "done_ids": done, "full_coverage": full, "checks": checks, "error": error})
    code = 0 if passed else 1
    with (args.out / "EXIT").open("x", encoding="utf-8") as stream:
        stream.write(f"{code}\n")
    print(f"model check: {'PASSED' if passed else 'FAILED'} ({len(done)}/{len(planned)}, EXIT {code})", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
