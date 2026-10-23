"""Dense control step, float64 oracle and gates of the CUDA FFN training step (used by `check.py`).

Nothing here is part of the runtime.

Conventions: compact gradients of the HFFN arm are `g_up` in hidden-major slot order (as `up_val`), `g_down` in the
output-major edge order of the runtime (as `out_val`), `gb_up`, `gb_down`. The oracle works in hidden-major slot order and
is mapped to the runtime order with the stable argsort of `down_idx` (the order `HFFNLayer.from_hidden_major` uses).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from hffn_ffn_layers import dense, layers

HYPER = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
GROUPS = ("y", "gb_down", "g_down", "gz", "g_up", "gb_up", "gx")
GATE2_FACTOR = 10.0          # per-group distance to float64 within this factor of the dense float32 control's distance
GATE2_FLOOR = 1e-7           # relative floor of the control distance (a control at 0 distance would give a zero tolerance)
GATE3_CONDITIONING = 100.0   # entries whose float64 |gradient| >= this factor times the dense float32 group distance
GATE3_RELATIVE = 1e-5
GATE5_OWN_GRADIENT_RELATIVE = 1e-5    # gate 5, fixed before any run
ORACLE_TOKENS = 64           # token chunk of the float64 oracle


def make_gy(case, n: int, salt: int) -> torch.Tensor:
    """Deterministic upstream gradient [N, d_model] (float32 CPU), PCG64 seeded by the case, N and the step."""
    import numpy as np
    rng = np.random.Generator(np.random.PCG64([case.seed, n, 7919, salt]))
    return torch.from_numpy(rng.standard_normal((n, case.d_model)).astype(np.float32) * 0.1)


def hidden_major_edges(layer):
    t = torch.from_numpy
    up_hidden = torch.repeat_interleave(torch.arange(layer.d_ff), t(layer.up_offsets).diff().long())
    down_hidden = torch.repeat_interleave(torch.arange(layer.d_ff), t(layer.down_offsets).diff().long())
    return up_hidden, t(layer.up_idx).long(), down_hidden, t(layer.down_idx).long()


def output_major_order(layer) -> torch.Tensor:
    """Permutation: output-major edge position -> hidden-major slot (stable argsort of `down_idx`)."""
    return torch.argsort(torch.from_numpy(layer.down_idx).long(), stable=True)


# -- float64 oracle -----------------------------------------------------------------------------------------------
def _gelu(z):
    return 0.5 * z * (1.0 + torch.erf(z * 0.7071067811865476))


def _dgelu(z):
    return 0.5 * (1.0 + torch.erf(z * 0.7071067811865476)) + z * torch.exp(-0.5 * z * z) * 0.3989422804014327


def oracle(layer, x: torch.Tensor, gy: torch.Tensor, params: dict | None = None) -> dict:
    """Direct equations in float64 on the compact hidden-major support (CPU). Returns every group, hidden-major slot order.

    ``params`` (hidden-major float64 `up_val`, `b_up`, `down_val`, `b_down`) overrides the layer's values (multi-step replay)."""
    f64 = torch.float64
    p = params or {"up_val": torch.from_numpy(layer.up_val).to(f64), "b_up": torch.from_numpy(layer.b_up).to(f64),
                   "down_val": torch.from_numpy(layer.down_val).to(f64), "b_down": torch.from_numpy(layer.b_down).to(f64)}
    up_hidden, up_input, down_hidden, down_output = hidden_major_edges(layer)
    n = x.shape[0]
    out = {"y": torch.empty(n, layer.d_model, dtype=f64), "gz": torch.empty(layer.d_ff, n, dtype=f64),
           "gx": torch.empty(n, layer.d_model, dtype=f64),
           "g_up": torch.zeros_like(p["up_val"]), "g_down": torch.zeros_like(p["down_val"]),
           "gb_up": torch.zeros(layer.d_ff, dtype=f64), "gb_down": gy.to(f64).sum(0)}
    for lo in range(0, n, ORACLE_TOKENS):
        hi = min(n, lo + ORACLE_TOKENS)
        xc, gc = x[lo:hi].to(f64), gy[lo:hi].to(f64)
        z = p["b_up"].expand(hi - lo, -1).clone().index_add_(1, up_hidden, xc[:, up_input] * p["up_val"])
        h = _gelu(z)
        out["y"][lo:hi] = p["b_down"].expand(hi - lo, -1).clone().index_add_(1, down_output, h[:, down_hidden] * p["down_val"])
        out["g_down"] += (gc[:, down_output] * h[:, down_hidden]).sum(0)
        dh = torch.zeros(hi - lo, layer.d_ff, dtype=f64).index_add_(1, down_hidden, gc[:, down_output] * p["down_val"])
        gz = dh * _dgelu(z)
        out["gz"][:, lo:hi] = gz.T
        out["gb_up"] += gz.sum(0)
        out["g_up"] += (gz[:, up_hidden] * xc[:, up_input]).sum(0)
        out["gx"][lo:hi] = torch.zeros(hi - lo, layer.d_model, dtype=f64).index_add_(1, up_input, gz[:, up_hidden] * p["up_val"])
    return out


def adamw_closed_form(p, g, m, v, step: int, lr, betas, eps, weight_decay):
    """One AdamW update in float64 (decoupled decay, bias correction), the form of `torch.optim.AdamW`."""
    b1, b2 = betas
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    p = p * (1 - lr * weight_decay) - lr * (m / (1 - b1 ** step)) / (torch.sqrt(v / (1 - b2 ** step)) + eps)
    return p, m, v


# -- dense float32 control ----------------------------------------------------------------------------------------
class DenseArm:
    """PyTorch float32 iso-support step : masked gradient, fused AdamW, no update re-mask."""

    def __init__(self, layer, device="cuda") -> None:
        w_up, w_down = (torch.from_numpy(w).to(device) for w in dense.dense_weights(layer))
        up_hidden, up_input, down_hidden, down_output = (i.to(device) for i in hidden_major_edges(layer))
        self.mask_up = torch.zeros_like(w_up)   # structural support: a stored zero value stays in it
        self.mask_up[up_hidden, up_input] = 1.0
        self.mask_down = torch.zeros_like(w_down)
        self.mask_down[down_output, down_hidden] = 1.0
        self.index = (up_hidden, up_input, down_hidden, down_output)
        self.params = [w_up.requires_grad_(), torch.from_numpy(layer.b_up.copy()).to(device).requires_grad_(),
                       w_down.requires_grad_(), torch.from_numpy(layer.b_down.copy()).to(device).requires_grad_()]
        self.optimiser = torch.optim.AdamW(self.params, fused=True, **HYPER)
        self.masks = (self.mask_up, None, self.mask_down, None)

    def step(self, x: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
        w_up, b_up, w_down, b_down = self.params
        z = F.linear(x, w_up, b_up)
        h = F.gelu(z, approximate="none")
        y = F.linear(h, w_down, b_down)
        y.backward(gy)
        for p, mask in zip(self.params, self.masks):
            if mask is not None:
                p.grad.mul_(mask)
        self.optimiser.step()
        self.optimiser.zero_grad(set_to_none=False)
        return y

    def compact_gradients(self) -> dict:
        w_up, b_up, w_down, b_down = self.params
        up_hidden, up_input, down_hidden, down_output = self.index
        return {"g_up": w_up.grad[up_hidden, up_input], "gb_up": b_up.grad, "g_down": w_down.grad[down_output, down_hidden],
                "gb_down": b_down.grad}

    def compact_values(self) -> dict:
        w_up, b_up, w_down, b_down = self.params
        up_hidden, up_input, down_hidden, down_output = self.index
        return {"up_val": w_up.detach()[up_hidden, up_input], "b_up": b_up.detach(),
                "down_val": w_down.detach()[down_output, down_hidden], "b_down": b_down.detach()}

    def dense_bytes(self) -> int:
        """Weights + gradients + both AdamW moments of the control (float32)."""
        state = sum(t.numel() * t.element_size() for s in self.optimiser.state.values() for t in s.values() if torch.is_tensor(t) and t.dim() > 0)
        return state + sum(2 * p.numel() * p.element_size() for p in self.params)


def dense_full_step(layer, x, gy, device="cuda") -> tuple[dict, DenseArm]:
    """One dense float32 step with autograd; returns every group (hidden-major slot order) before the update.

    Used for the control's distance to float64 (gate 2/3). The activation gradient `gz` and `gx` come from autograd."""
    arm = DenseArm(layer, device)
    w_up, b_up, w_down, b_down = arm.params
    xg = x.detach().clone().requires_grad_()
    z = F.linear(xg, w_up, b_up)
    z.retain_grad()
    y = F.linear(F.gelu(z, approximate="none"), w_down, b_down)
    y.backward(gy)
    up_hidden, up_input, down_hidden, down_output = arm.index
    groups = {"y": y.detach(), "gz": z.grad.T.contiguous(), "gx": xg.grad, "g_up": w_up.grad[up_hidden, up_input],
              "gb_up": b_up.grad, "g_down": w_down.grad[down_output, down_hidden], "gb_down": b_down.grad}
    return groups, DenseArm(layer, device)


# -- gates --------------------------------------------------------------------------------------------------------
def _dist(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double().cpu() - b.double().cpu()).abs().max())


def hffn_groups(train, layer, n: int) -> dict:
    """The seven groups of the last step of the HFFN arm, hidden-major slot order (g_down mapped back from output-major)."""
    perm = output_major_order(layer).to(train.runtime.device)
    grads, acts = train.gradients(0), train.activations(n)
    g_down = torch.empty_like(grads["out_val"])
    g_down[perm] = grads["out_val"]
    return {"y": train.runtime._y[:n], "gb_down": grads["b_down"], "g_down": g_down, "gz": acts["gz"], "g_up": grads["up_val"],
            "gb_up": grads["b_up"], "gx": acts["gx"]}


def gate2(hffn: dict, control: dict, ref: dict) -> dict:
    """Per group: max-abs distance to float64 of the HFFN arm within GATE2_FACTOR x that of the dense float32 control."""
    rows, ok = {}, True
    for name in GROUPS:
        scale = float(ref[name].abs().max())
        d_h, d_c = _dist(hffn[name], ref[name]), _dist(control[name], ref[name])
        tolerance = GATE2_FACTOR * max(d_c, GATE2_FLOOR * scale)
        rows[name] = {"hffn": d_h, "control": d_c, "tolerance": tolerance, "relative_hffn": d_h / scale if scale else 0.0,
                      "pass": d_h <= tolerance}
        ok &= rows[name]["pass"]
    return {"pass": ok, "groups": rows}


def conditioned_masks(ref: dict, control: dict) -> dict:
    """Entries of each gradient group whose float64 |g| >= GATE3_CONDITIONING x the dense float32 group distance."""
    masks = {}
    for name in ("g_up", "gb_up", "g_down", "gb_down"):
        bound = GATE3_CONDITIONING * _dist(control[name], ref[name])
        masks[name] = ref[name].abs() >= bound
    return masks


PARAM_OF = {"g_up": "up_val", "gb_up": "b_up", "g_down": "down_val", "gb_down": "b_down"}


def gate3(train, layer, groups: dict, masks: dict, before: dict) -> dict:
    """One AdamW step against the float64 closed form on conditioned entries (relative GATE3_RELATIVE).

    The closed form is fed the build's own float32 gradient of the step (`groups`, hidden-major), so that the gate
    judges the update and not the gradient already judged by gate 2.

    The updated value is compared relative to max(|updated|, |previous|) (the magnitude the operation works at); the
    moments plainly relative. Entries below the conditioning threshold are counted, not compared."""
    perm = output_major_order(layer)
    after = train.parameters(0)
    moments = train.moments(0)
    rows, ok = {}, True
    for g, p in PARAM_OF.items():
        runtime_name = "out_val" if p == "down_val" else p
        p0 = before[p].double()
        g_build = groups[g].double().cpu()
        new_p, new_m, new_v = adamw_closed_form(p0, g_build, torch.zeros_like(p0), torch.zeros_like(p0), 1, **HYPER)
        got_p, (got_m, got_v) = after[runtime_name].double().cpu(), tuple(t.double().cpu() for t in moments[runtime_name])
        if p == "down_val":   # output-major -> hidden-major
            back = torch.empty_like(got_p)
            back[perm] = got_p
            got_p = back
            m_back, v_back = torch.empty_like(got_m), torch.empty_like(got_v)
            m_back[perm], v_back[perm] = got_m, got_v
            got_m, got_v = m_back, v_back
        mask = masks[g]
        def rel(a, b, ref_scale):
            return float(((a - b).abs() / ref_scale.clamp_min(1e-30))[mask].max()) if mask.any() else 0.0
        errors = {"value": rel(got_p, new_p, torch.maximum(new_p.abs(), p0.abs())), "m": rel(got_m, new_m, new_m.abs()),
                  "v": rel(got_v, new_v, new_v.abs())}
        rows[g] = {**errors, "conditioned": int(mask.sum()), "total": mask.numel(), "pass": max(errors.values()) <= GATE3_RELATIVE}
        ok &= rows[g]["pass"]
    return {"pass": ok, "groups": rows}


def snapshot(train) -> dict:
    return {"params": {k: v.clone() for k, v in train.parameters(0).items()},
            "moments": {k: tuple(t.clone() for t in v) for k, v in train.moments(0).items()}, "steps": train.steps[0]}


def restore(train, state: dict) -> None:
    for k, v in train.parameters(0).items():
        v.copy_(state["params"][k])
    for k, v in train.moments(0).items():
        for t, s in zip(v, state["moments"][k]):
            t.copy_(s)
    train.steps[0] = state["steps"]


def hidden_major_params(layer) -> dict:
    return {"up_val": torch.from_numpy(layer.up_val), "b_up": torch.from_numpy(layer.b_up),
            "down_val": torch.from_numpy(layer.down_val), "b_down": torch.from_numpy(layer.b_down)}


JUDGED = (("g_up", "up_val"), ("g_down", "down_val"))


def build_state(train, layer, perm: torch.Tensor) -> dict:
    """The build's float32 parameter and AdamW moments of layer 0, per parameter, as float64 CPU tensors in hidden-major slot order."""
    params, moments = train.parameters(0), train.moments(0)
    out = {}
    for p in PARAM_OF.values():
        runtime_name = "out_val" if p == "down_val" else p
        tensors = [t.double().cpu() for t in (params[runtime_name], *moments[runtime_name])]
        if p == "down_val":   # output-major -> hidden-major
            for i, t in enumerate(tensors):
                back = torch.empty_like(t)
                back[perm] = t
                tensors[i] = back
        out[p] = tuple(tensors)
    return out


def nominal_update(name: str, step: int, p, g, m, v, g_prev):
    """The float64 closed form of one update at the nominal hyper-parameters (`name` and `g_prev` serve the fault injections)."""
    return adamw_closed_form(p, g, m, v, step, **HYPER)


def _rel(a, b, scale, mask) -> float:
    return float(((a - b).abs() / scale.clamp_min(1e-30))[mask].max())


def _worse(*values: float) -> float:
    """Largest error, NaN-propagating (Python's max drops a NaN that is not first)."""
    return float("nan") if any(v != v for v in values) else max(values)


def gate5_local(train, layer, case, x: torch.Tensor, masks: dict, steps: int = 5, closed_form=nominal_update) -> dict:
    """Gate 5, local update form.

    Pass A: `steps` consecutive steps on the upstream gradients of `make_gy(case, n, k)`. Each update is compared with the
    float64 closed form started from the build's actual pre-update float32 parameters and moments, fed the build's actual
    gradient of that step, its step number (the build's counter, which must equal the nominal one) and the nominal
    hyper-parameters; on the conditioned entries of `masks`, at every step: the value relative to max(|reference|,
    |previous|, lr) (lr nominal, the unit of one update) and the moments as gate 3 (m relative to
    max(|m_ref|, beta1 |m_prev|, (1 - beta1) |g|), v plainly relative), all within GATE5_OWN_GRADIENT_RELATIVE. Biases
    are compared and recorded, not judged.
    Pass B (decay only): from the same initial state, `steps` steps on a zero upstream gradient; the build's parameters
    after the last step against the float64 chain fed the build's gradients (zero by the equations), relative to
    max(|reference|, |initial|). With zero moments the update term vanishes, so this pass isolates the decoupled decay,
    whose omission sits at the per-step bound in pass A (lr x weight_decay = 1e-5); a non-zero gradient read in it fails.
    The cumulative own-gradient replay of pass A (read at the last step on the same masks, relative to
    max(|reference|, |initial|)) is recorded with the bound of the per-step gate as a limit; it does not gate. A group whose mask is
    empty fails closed; every parameter and moment must stay finite. The build is left in the state it had on entry."""
    n = x.shape[0]
    perm = output_major_order(layer)
    initial = snapshot(train)
    k0 = int(train.steps[0])
    start = build_state(train, layer, perm)
    b1 = HYPER["betas"][0]
    local = {p: {"value": 0.0, "m": 0.0, "v": 0.0} for p in PARAM_OF.values()}
    replay = {p: s for p, s in start.items()}
    previous: dict = {}
    counters, finite = [], True
    for k in range(1, steps + 1):
        pre = build_state(train, layer, perm)
        train.step(0, x, make_gy(case, n, k).to(x.device))
        counters.append(int(train.steps[0]) - k0)
        groups = hffn_groups(train, layer, n)
        post = build_state(train, layer, perm)
        finite &= all(bool(torch.isfinite(t).all()) for s in post.values() for t in s)
        for g, p in PARAM_OF.items():
            grad = groups[g].double().cpu()
            p_pre, m_pre, v_pre = pre[p]
            ref_p, ref_m, ref_v = closed_form(p, k, p_pre, grad, m_pre, v_pre, previous.get(p))
            replay[p] = closed_form(p, k, *replay[p][:1], grad, *replay[p][1:], previous.get(p))
            previous[p] = grad
            mask = masks.get(g, torch.ones_like(grad, dtype=torch.bool))
            if not mask.any():
                continue
            got_p, got_m, got_v = post[p]
            errors = {"value": _rel(got_p, ref_p, torch.maximum(ref_p.abs(), p_pre.abs()).clamp_min(HYPER["lr"]), mask),
                      "m": _rel(got_m, ref_m, torch.maximum(torch.maximum(ref_m.abs(), b1 * m_pre.abs()), (1 - b1) * grad.abs()), mask),
                      "v": _rel(got_v, ref_v, ref_v.abs(), mask)}
            local[p] = {e: _worse(local[p][e], errors[e]) for e in errors}
    end = build_state(train, layer, perm)
    restore(train, initial)
    zero = torch.zeros(n, layer.d_model, device=x.device)
    decay = {p: s for p, s in start.items()}
    previous, decay_gradient = {}, 0.0
    for k in range(1, steps + 1):
        train.step(0, x, zero)
        groups = hffn_groups(train, layer, n)
        for g, p in PARAM_OF.items():
            grad = groups[g].double().cpu()
            decay_gradient = _worse(decay_gradient, float(grad.abs().max()))
            decay[p] = closed_form(p, k, *decay[p][:1], grad, *decay[p][1:], previous.get(p))
            previous[p] = grad
    after_decay = build_state(train, layer, perm)
    finite &= all(bool(torch.isfinite(t).all()) for s in after_decay.values() for t in s)
    restore(train, initial)
    rows, record, ok = {}, {}, True
    for g, p in JUDGED:
        mask = masks[g]
        conditioned = int(mask.sum())
        decay_err = _rel(after_decay[p][0], decay[p][0], torch.maximum(decay[p][0].abs(), start[p][0].abs()), mask) if conditioned else float("inf")
        worst = _worse(*local[p].values()) if conditioned else float("inf")
        rows[p] = {"local": local[p], "decay": decay_err, "conditioned": conditioned, "tolerance": GATE5_OWN_GRADIENT_RELATIVE,
                   "pass": conditioned > 0 and worst <= GATE5_OWN_GRADIENT_RELATIVE and decay_err <= GATE5_OWN_GRADIENT_RELATIVE}
        ok &= rows[p]["pass"]
        cumulative = _rel(end[p][0], replay[p][0], torch.maximum(replay[p][0].abs(), start[p][0].abs()), mask) if conditioned else float("inf")
        record[p] = {"relative": cumulative, "bound": GATE5_OWN_GRADIENT_RELATIVE, "exceeded": cumulative > GATE5_OWN_GRADIENT_RELATIVE}
    counted = counters == list(range(1, steps + 1))
    zero_gradients = decay_gradient == 0.0   # pass B must stay a decay-only check, not a cumulative replay
    return {"pass": ok and finite and counted and zero_gradients, "finite": finite, "step_counter": counters, "decay_gradient_max": decay_gradient,
            "groups": rows, "biases": {p: local[p] for p in ("b_up", "b_down")}, "replay_record": record}
