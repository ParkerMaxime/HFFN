"""Per-update float64 closed-form gate of the Nvidia LLM training checker .

Model-level counterpart of `train_step.gate5_local` (float64 closed form of the compact AdamW update). The model arms have one fused optimiser step for every layer, so the gate works on
snapshots of the build's compact state read before and after each step instead of driving `train.step` on one layer.

Conventions. A snapshot of one layer is `{runtime_name: (p, m, v)}` of float32 tensors in the build's own order (`up_val`,
`b_up`, `out_val` output-major, `b_down`); gradients are hidden-major float32 groups `g_up, g_down, gb_up, gb_down` (the order of
`ts.PARAM_OF`). Every comparison runs in float64 on `device`, in hidden-major slot order.

Pass A (`UpdateGate`): each update k of the horizon is compared with the closed form started from the build's actual pre-update
parameters and moments, fed its actual gradient and the nominal step number, on the conditioned entries of the layer's mask: the value
relative to max(|reference|, |previous|, lr), `m` relative to max(|m_ref|, beta1 |m_prev|, (1 - beta1) |g|), `v` plainly relative, all
within `TOLERANCE`. Biases are compared and recorded, not judged. Pass B (`DecayGate`): the zero-gradient decay-only chain.

Hyper-parameter representation, as the build stores them (`REPRESENTATION`): the HFFN kernel holds eight float32 scalars
(`float32_scalars`); the dense control (PyTorch fused AdamW) is given `lr, beta1, beta2, weight_decay, eps` as doubles
(`fused_adam_utils.cuh`) and derives its bias corrections and step in float32, so its reference is `ts.adamw_closed_form` at the
nominal doubles, the float32 derivation being a recorded limit (about 1e-7). Fault injection is on the reference side
(as in the FFN training gate): the build's gradient and pre-update state are its own, so a fault in the reference is
equivalent for the update mechanics.
"""
from __future__ import annotations

import math
import struct

import torch

import train_step as ts

HYPER = ts.HYPER
TOLERANCE = ts.GATE5_OWN_GRADIENT_RELATIVE
STEPS = 5
JUDGED = (("g_up", "up_val"), ("g_down", "down_val"))
PARAMETERS = tuple(ts.PARAM_OF.values())                       # up_val, b_up, down_val, b_down
GROUP_OF = {parameter: group for group, parameter in ts.PARAM_OF.items()}
RUNTIME_NAME = {"up_val": "up_val", "b_up": "b_up", "down_val": "out_val", "b_down": "b_down"}
REPRESENTATION = {"hffn": "float32 scalars of the compact kernel (eight, rounded once)",
                  "dense32": "closed form at the nominal doubles: PyTorch's fused AdamW takes lr, betas, eps and weight decay as "
                             "doubles, then derives the bias corrections and the step in float32 (header read; the difference, about "
                             "1e-7, is below TOLERANCE and is not a gated quantity; the VM's torch version is recorded in provenance)"}
FAULT_NAMES = ("skip_moment_update_step3", "v_bias_correction_uses_k_minus_1", "step_counter_frozen_at_1", "weight_decay_omitted",
               "beta2_0.99_from_step_2", "eps_times_1000", "lr_times_1.0001", "stale_gradient_from_step_2")


def f32(value: float) -> float:
    """Round a double to float32 once (nearest even) and widen it back, as the kernel's scalar arguments."""
    return struct.unpack("f", struct.pack("f", value))[0]


def float32_scalars(step: int, hyper: dict = HYPER) -> dict:
    """The eight scalars `HFFNTrainRuntime.adamw` passes to the compact kernel (`train_runtime.py`), float32-rounded."""
    lr, (b1, b2), eps, wd = hyper["lr"], hyper["betas"], hyper["eps"], hyper["weight_decay"]
    return {"lr_decay": f32(lr * wd), "beta1": f32(b1), "one_minus_beta1": f32(1.0 - b1), "beta2": f32(b2),
            "one_minus_beta2": f32(1.0 - b2), "step_size": f32(lr / (1.0 - b1 ** step)),
            "bias2_sqrt": f32(math.sqrt(1.0 - b2 ** step)), "eps": f32(eps)}


def adamw(p, g, m, v, step, lr, betas, eps, weight_decay, *, scalars, step_m=None, step_v=None, skip_moments=False, decay=True):
    """One AdamW update in float64 in the form of the compact kernel, with the fault switches (separate counters of the two bias
    corrections, no moment update, no decay). `scalars` rounds the eight derived constants to float32 as the kernel's arguments;
    without it the constants are the double values."""
    b1, b2 = betas
    r = f32 if scalars else (lambda x: x)
    sm, sv = step if step_m is None else step_m, step if step_v is None else step_v
    lr_decay = r(lr * weight_decay) if decay else 0.0
    step_size, bias2_sqrt = r(lr / (1.0 - b1 ** sm)), r(math.sqrt(1.0 - b2 ** sv))
    if not skip_moments:
        m = r(b1) * m + r(1.0 - b1) * g
        v = r(b2) * v + r(1.0 - b2) * g * g
    p = p - lr_decay * p
    return p - step_size * m / (torch.sqrt(v) / bias2_sqrt + r(eps)), m, v


def adamw_closed_form_scalars(p, g, m, v, step: int, hyper: dict = HYPER):
    """The closed form of the HFFN arm: the kernel's update evaluated in float64 with its float32 scalars."""
    return adamw(p, g, m, v, step, scalars=True, **hyper)


def base_form(arm: str):
    """The closed form an arm is judged against, in its declared representation (`form(k, p, g, m, v, g_prev) -> (p, m, v)`)."""
    if arm == "hffn":
        return lambda k, p, g, m, v, gp: adamw_closed_form_scalars(p, g, m, v, k)
    return lambda k, p, g, m, v, gp: ts.adamw_closed_form(p, g, m, v, k, **HYPER)


def nominal_form():
    return lambda k, p, g, m, v, gp: ts.adamw_closed_form(p, g, m, v, k, **HYPER)


def fault_form(name: str, scalars: bool):
    """the faults of the FFN training gate, in either representation."""
    h = HYPER
    if name == "skip_moment_update_step3":
        return lambda k, p, g, m, v, gp: adamw(p, g, m, v, k, **h, scalars=scalars, skip_moments=(k == 3))
    if name == "v_bias_correction_uses_k_minus_1":
        return lambda k, p, g, m, v, gp: adamw(p, g, m, v, k, **h, scalars=scalars, step_v=max(k - 1, 1))
    if name == "step_counter_frozen_at_1":
        return lambda k, p, g, m, v, gp: adamw(p, g, m, v, 1, **h, scalars=scalars)
    if name == "weight_decay_omitted":
        return lambda k, p, g, m, v, gp: adamw(p, g, m, v, k, **h, scalars=scalars, decay=False)
    if name == "beta2_0.99_from_step_2":
        return lambda k, p, g, m, v, gp: adamw(p, g, m, v, k, **{**h, "betas": (0.9, 0.99 if k >= 2 else 0.999)}, scalars=scalars)
    if name == "eps_times_1000":
        return lambda k, p, g, m, v, gp: adamw(p, g, m, v, k, **{**h, "eps": 1000 * h["eps"]}, scalars=scalars)
    if name == "lr_times_1.0001":
        return lambda k, p, g, m, v, gp: adamw(p, g, m, v, k, **{**h, "lr": h["lr"] * 1.0001}, scalars=scalars)
    if name == "stale_gradient_from_step_2":
        return lambda k, p, g, m, v, gp: adamw(p, gp if (k >= 2 and gp is not None) else g, m, v, k, **h, scalars=scalars)
    raise KeyError(name)


def hidden_major(tensor: torch.Tensor, runtime_name: str, perm: torch.Tensor, device) -> torch.Tensor:
    """A float32 tensor of the build, as float64 in hidden-major slot order (the output-major `out_val` is permuted back)."""
    t = tensor.to(device).double()
    if runtime_name == "out_val":
        back = torch.empty_like(t)
        back[perm.to(device)] = t
        return back
    return t


def _errors(got, ref, pre, grad, mask) -> dict:
    p_got, m_got, v_got = got
    p_ref, m_ref, v_ref = ref
    p_pre, m_pre, _ = pre
    b1 = HYPER["betas"][0]
    return {"value": ts._rel(p_got, p_ref, torch.maximum(p_ref.abs(), p_pre.abs()).clamp_min(HYPER["lr"]), mask),
            "m": ts._rel(m_got, m_ref, torch.maximum(torch.maximum(m_ref.abs(), b1 * m_pre.abs()), (1 - b1) * grad.abs()), mask),
            "v": ts._rel(v_got, v_ref, v_ref.abs(), mask)}


def _merge(old: dict | None, new: dict) -> dict:
    return dict(new) if old is None else {key: ts._worse(old[key], new[key]) for key in new}


class UpdateGate:
    """Pass A over the steps of one arm. `forms` names: `gate` (judged), `nominal_recorded` (hffn only, recorded), `fault:<name>`."""

    def __init__(self, arm: str, device, faults: tuple = ()) -> None:
        self.arm, self.device = arm, torch.device(device)
        self.forms = {"gate": base_form(arm)}
        if arm == "hffn":
            self.forms["nominal_recorded"] = nominal_form()
        for name in faults:
            self.forms[f"fault:{name}"] = fault_form(name, arm == "hffn")
        self.keep_previous = "fault:stale_gradient_from_step_2" in self.forms
        self.worst: dict = {}
        self.conditioned: dict = {}
        self.counters: dict = {}
        self.steps_seen: dict = {}
        self.finite = True
        self._previous: dict = {}

    def update(self, layer: int, k: int, pre: dict, post: dict, grads: dict, masks: dict, perm: torch.Tensor, counters: list) -> dict:
        """Compare the update k of one layer; returns the gate-form errors of this step for the progress record."""
        row = {"layer": layer, "k": k}
        self.counters.setdefault(layer, {})[k] = [int(c) for c in counters]
        self.steps_seen.setdefault(layer, set()).add(k)
        for parameter in PARAMETERS:
            name = RUNTIME_NAME[parameter]
            self.finite &= all(bool(torch.isfinite(t).all()) for t in post[name])
            p_pre, m_pre, v_pre = (hidden_major(t, name, perm, self.device) for t in pre[name])
            got = tuple(hidden_major(t, name, perm, self.device) for t in post[name])
            grad = grads[GROUP_OF[parameter]].to(self.device).double()
            mask = masks[GROUP_OF[parameter]].to(self.device)
            self.conditioned[(layer, parameter)] = int(mask.sum())
            previous = self._previous.get((layer, parameter))
            if mask.any():
                for form_name, form in self.forms.items():
                    ref = form(k, p_pre, grad, m_pre, v_pre, previous)
                    errors = _errors(got, ref, (p_pre, m_pre, v_pre), grad, mask)
                    key = (form_name, layer, parameter)
                    self.worst[key] = _merge(self.worst.get(key), errors)
                    if form_name == "gate":
                        row[parameter] = errors
            if self.keep_previous:
                self._previous[(layer, parameter)] = grad
        return row

    def _group(self, form_name: str, parameter: str, layers) -> float:
        values = [ts._worse(*self.worst[(form_name, layer, parameter)].values()) for layer in layers
                  if (form_name, layer, parameter) in self.worst]
        return ts._worse(*values) if values else float("inf")

    def result(self, expected_layers: int, steps: int = STEPS) -> dict:
        """`expected_layers` is the layer count of the model, read by the caller from the model, not from what the gate saw: a layer the
        driver skipped would otherwise never be judged (README, Update gate, guard on the layer coverage)."""
        layers = sorted(self.steps_seen)
        layers_complete = expected_layers > 0 and layers == list(range(expected_layers))
        rows, group_ok = [], {parameter: True for _, parameter in JUDGED}
        for layer in layers:
            for _, parameter in JUDGED:
                count = self.conditioned.get((layer, parameter), 0)
                errors = self.worst.get(("gate", layer, parameter), {"value": float("inf"), "m": float("inf"), "v": float("inf")})
                worst = ts._worse(*errors.values()) if count else float("inf")
                ok = count > 0 and worst <= TOLERANCE
                group_ok[parameter] &= ok
                rows.append({"layer": layer, "parameter": parameter, "conditioned": count, **errors, "tolerance": TOLERANCE, "pass": ok})
        counters_ok = bool(layers) and all(self.counters[layer].get(k) is not None and set(self.counters[layer][k]) == {k}
                                           for layer in layers for k in range(1, steps + 1))
        steps_complete = bool(layers) and all(self.steps_seen[layer] == set(range(1, steps + 1)) for layer in layers)
        out = {"arm": self.arm, "tolerance": TOLERANCE, "steps": steps, "layers": len(layers), "expected_layers": expected_layers,
               "layers_complete": layers_complete, "finite": self.finite,
               "counters_ok": counters_ok, "steps_complete": steps_complete, "groups": rows, "group_pass": group_ok,
               "representation": REPRESENTATION[self.arm],
               "biases": {p: {key: ts._worse(*[self.worst[("gate", layer, p)][key] for layer in layers if ("gate", layer, p) in self.worst] or [0.0])
                              for key in ("value", "m", "v")} for p in ("b_up", "b_down")},
               "conditioned_min": {p: min([self.conditioned.get((layer, p), 0) for layer in layers] or [0]) for _, p in JUDGED}}
        out["pass"] = all(group_ok.values()) and self.finite and counters_ok and steps_complete and layers_complete
        if "nominal_recorded" in self.forms:
            out["nominal_recorded"] = {p: self._group("nominal_recorded", p, layers) for _, p in JUDGED}
        out["fault_worst"] = {name.split(":", 1)[1]: {p: self._group(name, p, layers) for _, p in JUDGED}
                              for name in self.forms if name.startswith("fault:")}
        return out


class DecayGate:
    """Pass B of one arm: five steps on a zero loss gradient. The gradients must be exactly zero, the moments stay exactly zero, the
    counters read `steps`, and the parameters after the last step equal the float64 decay chain."""

    def __init__(self, arm: str, device, faults: tuple = (), steps: int = STEPS) -> None:
        self.arm, self.device, self.steps = arm, torch.device(device), steps
        self.forms = {"gate": base_form(arm)}
        for name in faults:
            self.forms[f"fault:{name}"] = fault_form(name, arm == "hffn")
        self.gradient_max = 0.0
        self.worst: dict = {}
        self.conditioned: dict = {}
        self.layers: set = set()
        self.counters_ok = True
        self.moments_zero = True
        self.finite = True

    def gradient(self, max_abs: float) -> None:
        self.gradient_max = ts._worse(self.gradient_max, max_abs)

    def layer(self, layer: int, pre: dict, post: dict, masks: dict, perm: torch.Tensor, counters: list) -> None:
        self.layers.add(layer)
        self.counters_ok &= all(int(c) == self.steps for c in counters)
        for parameter in PARAMETERS:
            name = RUNTIME_NAME[parameter]
            self.finite &= all(bool(torch.isfinite(t).all()) for t in post[name])
            self.moments_zero &= all(int(torch.count_nonzero(t)) == 0 for t in post[name][1:])
            if parameter not in dict(JUDGED).values():
                continue
            p_pre = hidden_major(pre[name][0], name, perm, self.device)
            got = hidden_major(post[name][0], name, perm, self.device)
            mask = masks[GROUP_OF[parameter]].to(self.device)
            self.conditioned[(layer, parameter)] = int(mask.sum())
            if not mask.any():
                continue
            zero = torch.zeros_like(p_pre)
            for form_name, form in self.forms.items():
                ref = p_pre
                for k in range(1, self.steps + 1):
                    ref, _, _ = form(k, ref, zero, zero, zero, zero)
                self.worst[(form_name, layer, parameter)] = ts._rel(got, ref, torch.maximum(ref.abs(), p_pre.abs()), mask)

    def _group(self, form_name: str, parameter: str) -> float:
        values = [self.worst[(form_name, layer, parameter)] for layer in sorted(self.layers) if (form_name, layer, parameter) in self.worst]
        return ts._worse(*values) if values else float("inf")

    def result(self, expected_layers: int) -> dict:
        layers_complete = expected_layers > 0 and sorted(self.layers) == list(range(expected_layers))
        rows, group_ok = [], {parameter: True for _, parameter in JUDGED}
        for layer in sorted(self.layers):
            for _, parameter in JUDGED:
                count = self.conditioned.get((layer, parameter), 0)
                error = self.worst.get(("gate", layer, parameter), float("inf"))
                ok = count > 0 and error <= TOLERANCE
                group_ok[parameter] &= ok
                rows.append({"layer": layer, "parameter": parameter, "conditioned": count, "decay": error, "tolerance": TOLERANCE, "pass": ok})
        zero_gradient = self.gradient_max == 0.0     # a decay-only pass must stay a decay-only pass: a non-zero gradient fails
        out = {"arm": self.arm, "steps": self.steps, "layers": len(self.layers), "expected_layers": expected_layers,
               "layers_complete": layers_complete, "gradient_max": self.gradient_max,
               "zero_gradient": zero_gradient, "counters_ok": self.counters_ok, "moments_exactly_zero": self.moments_zero,
               "finite": self.finite, "groups": rows, "group_pass": group_ok}
        out["pass"] = (all(group_ok.values()) and layers_complete and zero_gradient and self.counters_ok and self.moments_zero
                       and self.finite)
        out["fault_worst"] = {name.split(":", 1)[1]: {p: self._group(name, p) for _, p in JUDGED}
                              for name in self.forms if name.startswith("fault:")}
        return out


def detection(pass_a: dict, decay: dict) -> dict:
    """Detection of the reference-side faults of one arm: each fault must exceed `TOLERANCE` on both judged groups, at pass A (any
    step) or at the decay-only pass; the margin is the largest error read divided by `TOLERANCE`. `none` must pass both."""
    faults, ok = {}, True
    for name, worst_a in pass_a["fault_worst"].items():
        row = {}
        for _, parameter in JUDGED:
            a, b = worst_a[parameter], decay["fault_worst"][name][parameter]
            worst = ts._worse(a, b)
            row[parameter] = {"pass_a": a, "decay": b, "margin": worst / TOLERANCE, "detected": not (worst <= TOLERANCE)}
            ok &= row[parameter]["detected"]
        faults[name] = row
    return {"pass": ok and bool(faults) and pass_a["pass"] and decay["pass"], "none_passes": pass_a["pass"] and decay["pass"],
            "faults": faults, "tolerance": TOLERANCE}


def build_fault_detected(pass_a: dict, decay: dict) -> dict:
    """A build-side fault (changed constructor hyper-parameter of the build) is detected on a judged group when that group fails
    pass A or the decay-only pass; the margin is read from the larger of the two errors."""
    rows = {}
    for _, parameter in JUDGED:
        rows[parameter] = {"detected": (not pass_a["group_pass"][parameter]) or (not decay["group_pass"][parameter])}
    return {"pass": all(row["detected"] for row in rows.values()), "groups": rows}
