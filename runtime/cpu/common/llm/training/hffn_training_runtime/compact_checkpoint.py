"""Cold compact LLM runtime snapshots; no forward, gradient or update implementation.

Called only at completed update boundaries, never from the training hot path.
The runtime snapshot excludes ordinary PyTorch non-FFN state; the whole-optimizer snapshot at the end adds the
non-FFN AdamW state (never the model state).
"""
from __future__ import annotations

import torch

FORMAT = "hffn_compact_g4_training_runtime_v1"
MAX_RESUMABLE_STEP = (1 << 63) - 2  # reserve the next native int64 update increment
SUPPORT = ("group_hidden_offsets", "up_row_offsets", "up_idx", "down_row_offsets", "down_idx")
VALUES = ("up_val", "down_val", "b_up", "b_down")
MOMENTS = ("um", "uv", "dm", "dv", "bum", "buv", "bdm", "bdv")
SECOND = frozenset(("uv", "dv", "buv", "bdv"))
TOP = frozenset(("format", "layers", "adam", "steps", "adamw_all_calls", "dual_route_min_n"))


def _keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f"invalid compact checkpoint {name} fields")


def _integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"invalid compact checkpoint {name}")
    return value


def _boundary(runtime):
    if runtime._contexts or any(runtime._has):
        raise RuntimeError("compact checkpoint requires a completed update boundary")
    if any(bool(torch.count_nonzero(t)) for a in runtime._accum for t in a.values()):
        raise RuntimeError("compact checkpoint refuses unfinished gradients")


def _tensor(source, target, name, *, second=False):
    if (not isinstance(source, torch.Tensor) or source.device.type != "cpu"
            or source.layout != torch.strided or not source.is_contiguous()
            or source.dtype != target.dtype or source.shape != target.shape):
        raise ValueError(f"invalid compact checkpoint tensor {name}")
    if source.is_floating_point() and not bool(torch.isfinite(source).all()):
        raise ValueError(f"nonfinite compact checkpoint tensor {name}")
    if second and bool((source < 0).any()):
        raise ValueError(f"negative compact checkpoint second moment {name}")
    return source.detach().clone()


def _targets(runtime):
    """Refuse overlapping mutable destinations before a potentially partial restore."""
    count = len(runtime.layers)
    if any(len(getattr(runtime, name)) != count for name in ("_moments", "_has", "_accum", "_steps")):
        raise ValueError("inconsistent compact checkpoint destination layer state")
    mutable, support = [], []
    for layer, moments in zip(runtime.layers, runtime._moments):
        layer.validate()
        for name, value_name in zip(MOMENTS, ("up_val", "up_val", "down_val", "down_val", "b_up", "b_up", "b_down", "b_down")):
            tensor, value = getattr(moments, name), getattr(layer, value_name)
            if tensor.dtype != torch.float32 or tensor.shape != value.shape:
                raise ValueError("invalid compact checkpoint destination moment")
        mutable.extend(getattr(layer, n) for n in VALUES)
        mutable.extend(getattr(moments, n) for n in MOMENTS)
        support.extend(getattr(layer, n) for n in SUPPORT)
    for tensor in mutable + support:
        if tensor.device.type != "cpu" or tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ValueError("invalid compact checkpoint destination storage")
    if any(torch.is_inference(t) or t.is_neg() or t.is_conj() for t in mutable):
        raise ValueError("compact checkpoint requires ordinary mutable training tensors")
    intervals = [(t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in mutable]
    readonly = [(t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in support]
    overlap = lambda a, b: a[0] < a[1] and b[0] < b[1] and max(a[0], b[0]) < min(a[1], b[1])
    if any(overlap(a, b) for i, a in enumerate(intervals) for b in intervals[i + 1:] + readonly):
        raise ValueError("overlapping compact checkpoint destinations")


def _stage(runtime, payload):
    _keys(payload, TOP, "top-level")
    if type(payload["format"]) is not str or payload["format"] != FORMAT:
        raise ValueError("unsupported compact runtime checkpoint format")
    route = _integer(payload["dual_route_min_n"], "dual_route_min_n", 1)
    if route != runtime.dual_route_min_n:
        raise ValueError("compact checkpoint route policy mismatch")
    count = len(runtime.layers)
    for name in ("layers", "adam", "steps"):
        if type(payload[name]) is not list or len(payload[name]) != count:
            raise ValueError(f"invalid compact checkpoint {name} count")
    steps = [_integer(x, "step", maximum=MAX_RESUMABLE_STEP) for x in payload["steps"]]
    calls = _integer(payload["adamw_all_calls"], "adamw_all_calls")
    if any(step > calls for step in steps):
        raise ValueError("compact checkpoint step exceeds update calls")
    layers, adam = [], []
    for index, (layer, moments, source, source_moments) in enumerate(zip(
            runtime.layers, runtime._moments, payload["layers"], payload["adam"])):
        _keys(source, (*SUPPORT, *VALUES, "d_model", "d_ff"), f"layer {index}")
        _keys(source_moments, MOMENTS, f"moments {index}")
        for name in ("d_model", "d_ff"):
            if _integer(source[name], name, 1) != getattr(layer, name):
                raise ValueError("compact checkpoint dimension mismatch")
        staged = {"d_model": layer.d_model, "d_ff": layer.d_ff}
        for name in SUPPORT:
            staged[name] = _tensor(source[name], getattr(layer, name), name)
            if not torch.equal(staged[name], getattr(layer, name)):
                raise ValueError(f"compact checkpoint support mismatch: {name}")
        for name in VALUES:
            staged[name] = _tensor(source[name], getattr(layer, name), name)
        layers.append(staged)
        adam.append({name: _tensor(source_moments[name], getattr(moments, name), name,
                                   second=name in SECOND) for name in MOMENTS})
    return {"format": FORMAT, "layers": layers, "adam": adam, "steps": steps,
            "adamw_all_calls": calls, "dual_route_min_n": route}


def state_dict(runtime):
    """Return an independently owned compact snapshot at a completed update boundary."""
    _boundary(runtime)
    _targets(runtime)
    payload = {"format": FORMAT, "dual_route_min_n": runtime.dual_route_min_n,
               "steps": list(runtime._steps), "adamw_all_calls": runtime._adamw_all_calls,
               "layers": [{"d_model": layer.d_model, "d_ff": layer.d_ff,
                           **{name: getattr(layer, name) for name in (*SUPPORT, *VALUES)}}
                          for layer in runtime.layers],
               "adam": [{name: getattr(moments, name) for name in MOMENTS}
                        for moments in runtime._moments]}
    return _stage(runtime, payload)


def load_state_dict(runtime, payload):
    """Validate/stage the whole payload, then copy into existing parameter/moment owners."""
    _boundary(runtime)
    _targets(runtime)
    staged = _stage(runtime, payload)
    with torch.no_grad():
        for layer, source, moments, source_moments in zip(
                runtime.layers, staged["layers"], runtime._moments, staged["adam"]):
            for name in VALUES:
                getattr(layer, name).copy_(source[name])
            for name in MOMENTS:
                getattr(moments, name).copy_(source_moments[name])
    runtime._steps = staged["steps"]
    runtime._adamw_all_calls = staged["adamw_all_calls"]


# Whole optimizer (HFFNCompactG4Optimizer): the non-FFN torch AdamW, lr/decay and the compact runtime above.
OPTIMIZER_FORMAT = "hffn_compact_g4_training_optimizer_v1"
OPTIMIZER_TOP = frozenset(("format", "lr", "weight_decay", "base", "hffn_runtime"))
BASE_STATE = frozenset(("step", "exp_avg", "exp_avg_sq"))


def _rate(value, name):
    if type(value) is not float or not 0.0 <= value < float("inf"):
        raise ValueError(f"invalid compact optimizer checkpoint {name}")
    return value


def _step_dtype():
    """The dtype torch AdamW (not fused, not capturable) gives a new `step` in this process."""
    return torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32


def _stage_base(base, payload, lr, weight_decay, calls):
    """Validate a torch AdamW state_dict against this optimizer's own groups; return an independently owned copy."""
    _keys(payload, ("state", "param_groups"), "base")
    groups, saved = base.param_groups, payload["param_groups"]
    if type(saved) is not list or len(saved) != len(groups):
        raise ValueError("invalid compact optimizer checkpoint base groups")
    params, staged_groups = [], []
    for group, source in zip(groups, saved):
        if type(source) is not dict or set(source) != set(group):
            raise ValueError("invalid compact optimizer checkpoint base group fields")
        ids = source["params"]
        if (type(ids) is not list or any(type(i) is not int for i in ids)
                or ids != list(range(len(params), len(params) + len(group["params"])))):
            raise ValueError("invalid compact optimizer checkpoint base parameter ids")
        for key, value in source.items():
            expected = {"lr": lr, "weight_decay": weight_decay}.get(key, group[key])
            if key != "params" and (type(value) is not type(expected) or value != expected):
                raise ValueError(f"compact optimizer checkpoint base hyperparameter mismatch: {key}")
        params.extend(group["params"])
        staged_groups.append({**source, "params": list(ids)})
    state = payload["state"]
    if type(state) is not dict:
        raise ValueError("invalid compact optimizer checkpoint base state")
    staged_state = {}
    for key, entry in state.items():
        if type(key) is not int or not 0 <= key < len(params):
            raise ValueError("invalid compact optimizer checkpoint base state id")
        _keys(entry, BASE_STATE, "base state")
        step = entry["step"]
        if (not isinstance(step, torch.Tensor) or step.dim() != 0 or step.device.type != "cpu"
                or step.dtype != _step_dtype() or not bool(torch.isfinite(step))
                or float(step) < 1 or float(step) != int(float(step))):
            raise ValueError("invalid compact optimizer checkpoint base step")
        if float(step) > calls:
            raise ValueError("compact optimizer checkpoint base step exceeds update calls")
        staged_state[key] = {"step": step.detach().clone(),
                             "exp_avg": _tensor(entry["exp_avg"], params[key], "exp_avg"),
                             "exp_avg_sq": _tensor(entry["exp_avg_sq"], params[key], "exp_avg_sq", second=True)}
    return {"state": staged_state, "param_groups": staged_groups}


def optimizer_state_dict(optimizer):
    """Return an independently owned whole-optimizer snapshot at a completed update boundary."""
    hffn = state_dict(optimizer.runtime)
    lr, weight_decay = _rate(optimizer.lr, "lr"), _rate(optimizer.weight_decay, "weight_decay")
    base = _stage_base(optimizer.base, optimizer.base.state_dict(), lr, weight_decay, hffn["adamw_all_calls"])
    return {"format": OPTIMIZER_FORMAT, "lr": lr, "weight_decay": weight_decay, "base": base, "hffn_runtime": hffn}


def load_optimizer_state_dict(optimizer, payload):
    """Validate and stage the top level, the compact runtime and the non-FFN AdamW before the first write.

    The writes then load the staged copies: torch keeps the tensors of a loaded state, so the destination owns them;
    the runtime restore re-validates a payload that already passed the same checks on the same unchanged destination.
    """
    runtime = optimizer.runtime
    _keys(payload, OPTIMIZER_TOP, "optimizer top-level")
    if type(payload["format"]) is not str or payload["format"] != OPTIMIZER_FORMAT:
        raise ValueError("unsupported compact optimizer checkpoint format")
    lr, weight_decay = _rate(payload["lr"], "lr"), _rate(payload["weight_decay"], "weight_decay")
    _boundary(runtime)
    _targets(runtime)
    hffn = _stage(runtime, payload["hffn_runtime"])
    base = _stage_base(optimizer.base, payload["base"], lr, weight_decay, hffn["adamw_all_calls"])
    optimizer.base.load_state_dict(base)
    load_state_dict(runtime, hffn)
    optimizer.lr, optimizer.weight_decay = lr, weight_decay
