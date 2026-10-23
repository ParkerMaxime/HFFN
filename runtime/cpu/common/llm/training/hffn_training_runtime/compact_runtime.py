"""Compact hidden-major HFFN training runtime with fixed G4 backward and AdamW.

Every compact FFN forward is dispatched through the sibling inference leaf. The training extension owns only compact
backward routes and AdamW; compact support, gradients and optimizer state remain hidden-major.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .compact_export import G4_GROUPS, HFFNCompactG4Layer, HFFNDualRoute, build_dual_route
from .compact_torch_extension import load_hffn_compact_torch_extension
# The entry point of the leaf (one file per hardware, in the leaf's portion of the package): its default build folder,
# the thread count and the smallest block of its inference forward, and the inference kernel it provides for the
# forward, or none.
from .forward_entry import (DEFAULT_BUILD_DIR, INFERENCE_MIN_N as _INFERENCE_MIN_N, INFERENCE_THREADS as _INFERENCE_THREADS,
                            open_inference_forward)


@dataclass
class _State:
    context_id: int; layer_idx: int; n: int; consumed: bool; dual_route: bool
    x_t: torch.Tensor; y_nt: torch.Tensor; y_t: torch.Tensor; z_t: torch.Tensor; h_t: torch.Tensor
    gy_t: torch.Tensor; gx_nt: torch.Tensor; gx_t: torch.Tensor; gx_local: torch.Tensor; gz_t: torch.Tensor
    gup: torch.Tensor; gdown: torch.Tensor; gbup: torch.Tensor; gbdown: torch.Tensor
    bt: torch.Tensor; ln2_mean: torch.Tensor; ln2_invstd: torch.Tensor
    grad_ln2_nt: torch.Tensor; grad_residual_nt: torch.Tensor; grad_attn_nt: torch.Tensor
    grad_gamma: torch.Tensor; grad_beta: torch.Tensor; pft: torch.Tensor; pbt: torch.Tensor; dft: torch.Tensor; dbt: torch.Tensor


@dataclass
class _Moments:
    um: torch.Tensor; uv: torch.Tensor; dm: torch.Tensor; dv: torch.Tensor
    bum: torch.Tensor; buv: torch.Tensor; bdm: torch.Tensor; bdv: torch.Tensor


class HFFNCompactG4TrainingRuntime:
    """HFFN contract: compact support state, dense reusable activation state."""

    def __init__(self, layers: list[HFFNCompactG4Layer], *, build_dir: Path | None = None, dual_route_min_n: int = 32) -> None:
        if not layers:
            raise ValueError("compact runtime requires layers")
        for layer in layers:
            layer.validate()
        if dual_route_min_n < 1:
            raise ValueError("dual_route_min_n must be positive")
        self.layers = tuple(layers)
        self.dual_route_min_n = int(dual_route_min_n)
        self._routes: tuple[HFFNDualRoute, ...] = tuple(build_dual_route(x) for x in self.layers)
        self.ops = load_hffn_compact_torch_extension(build_dir or DEFAULT_BUILD_DIR)
        self._inference = open_inference_forward(self.layers, self._routes)
        self._contexts: dict[int, _State] = {}
        self._free: dict[tuple[int, int], list[_State]] = {}
        self._next = 1
        self._workspace_allocations = 0
        self._accum = [{"up": torch.zeros_like(x.up_val), "down": torch.zeros_like(x.down_val), "bup": torch.zeros_like(x.b_up), "bdown": torch.zeros_like(x.b_down)} for x in self.layers]
        self._has = [False] * len(self.layers)
        self._steps = [0] * len(self.layers)
        self._adamw_all_calls = 0
        self._moments = tuple(_Moments(torch.zeros_like(x.up_val), torch.zeros_like(x.up_val), torch.zeros_like(x.down_val), torch.zeros_like(x.down_val), torch.zeros_like(x.b_up), torch.zeros_like(x.b_up), torch.zeros_like(x.b_down), torch.zeros_like(x.b_down)) for x in self.layers)
        self.reset_timing()

    def reset_timing(self) -> None:
        self._timing_ns = {"forward": 0, "backward": 0, "post_forward": 0, "post_backward": 0}

    def timing_ms(self) -> dict[str, float]:
        return {name: value / 1e6 for name, value in self._timing_ns.items()}

    def workspace_stats(self) -> dict[str, int]:
        return {"allocated": self._workspace_allocations, "live": len(self._contexts), "free": sum(len(v) for v in self._free.values())}

    def _new_state(self, layer: int, n: int) -> _State:
        spec = self.layers[layer]
        self._workspace_allocations += 1
        gx_local = torch.empty(G4_GROUPS, spec.d_model, n)
        state = _State(0, layer, n, False, False,
            torch.empty(spec.d_model, n), torch.empty(n, spec.d_model), torch.empty(spec.d_model, n), torch.empty(spec.d_ff, n), torch.empty(spec.d_ff, n),
            torch.empty(spec.d_model, n), torch.empty(n, spec.d_model), torch.empty(spec.d_model, n), gx_local, torch.empty(spec.d_ff, n),
            torch.empty_like(spec.up_val), torch.empty_like(spec.down_val), torch.empty_like(spec.b_up), torch.empty_like(spec.b_down),
            torch.empty(3, dtype=torch.int64), torch.empty(n), torch.empty(n),
            torch.empty(n, spec.d_model), torch.empty(n, spec.d_model), torch.empty(n, spec.d_model), torch.empty(spec.d_model), torch.empty(spec.d_model), torch.empty(4, dtype=torch.int64), torch.empty(5, dtype=torch.int64), torch.empty(4, dtype=torch.int64), torch.empty(4, dtype=torch.int64))
        # The inference kernel is run on 64-byte aligned work buffers (as in the FFN training leaf). Checked once,
        # where the context is allocated; never in the forward.
        for name, value in vars(state).items():
            if isinstance(value, torch.Tensor) and value.numel() and value.data_ptr() % 64:
                raise RuntimeError(f"work buffer {name} of layer {layer} (N = {n}) is not 64-byte aligned")
        return state

    def _state(self, layer: int, n: int) -> _State:
        key = (layer, n)
        bucket = self._free.get(key)
        state = bucket.pop() if bucket else self._new_state(layer, n)
        state.context_id, state.layer_idx, state.n, state.consumed = self._next, layer, n, False
        state.dual_route = n >= self.dual_route_min_n
        self._next += 1
        self._contexts[state.context_id] = state
        return state

    def _release(self, state: _State) -> None:
        self._free.setdefault((state.layer_idx, state.n), []).append(state)

    @staticmethod
    def _check(x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cpu" or x.dtype != torch.float32:
            raise ValueError("compact HFFN requires contiguous CPU float32 tensors")
        return x.contiguous()

    def _record(self, name: str, timing: torch.Tensor, *, has_total: bool = False) -> None:
        self._timing_ns[name] += int(timing[-1].item() if has_total else timing.sum().item())

    def forward_with_context(self, layer: int, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        x = self._check(x); spec = self.layers[layer]
        if x.dim() != 2 or x.size(1) != spec.d_model:
            raise ValueError("expected [N,H]")
        n = int(x.size(0))
        if n < _INFERENCE_MIN_N:
            raise ValueError(f"inference forward needs N >= {_INFERENCE_MIN_N}, the smallest block of this leaf's inference kernel")
        state = self._state(layer, n)
        route = self._routes[layer]
        self._inference.forward_dual_slots_nt(_INFERENCE_THREADS, x.data_ptr(), spec.group_hidden_offsets.data_ptr(), spec.b_up.data_ptr(), spec.up_row_offsets.data_ptr(), spec.up_idx.data_ptr(), spec.up_val.data_ptr(),
            spec.b_down.data_ptr(), route.down_by_output_offsets.data_ptr(), route.down_slot_hidden.data_ptr(), spec.down_val.data_ptr(), route.down_slot_indices.data_ptr(), route.output_group_offsets.data_ptr(),
            n, spec.d_model, spec.d_ff, state.x_t.data_ptr(), state.y_nt.data_ptr(), state.y_t.data_ptr(), state.z_t.data_ptr(), state.h_t.data_ptr(), state.dft.data_ptr())
        self._record("forward", state.dft)
        return state.y_nt, state.context_id

    def discard_context(self, cid: int) -> None:
        """Release a context only once the caller has copied its output."""
        state = self._contexts.pop(int(cid), None)
        if state is not None:
            self._release(state)

    def _accumulate(self, layer: int, state: _State) -> None:
        accum = self._accum[layer]
        if self._has[layer]:
            for name, grad in (("up", state.gup), ("down", state.gdown), ("bup", state.gbup), ("bdown", state.gbdown)):
                accum[name].add_(grad)
        else:
            for name, grad in (("up", state.gup), ("down", state.gdown), ("bup", state.gbup), ("bdown", state.gbdown)):
                accum[name].copy_(grad)
            self._has[layer] = True

    def backward_with_context(self, cid: int, layer: int, gy: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self._contexts.get(int(cid)); spec = self.layers[layer]
        if state is None or state.layer_idx != layer or state.consumed:
            raise RuntimeError("invalid compact HFFN context")
        gy = self._check(gy)
        if not state.dual_route:
            self.ops.backward_nt(gy, state.x_t, state.z_t, state.h_t, spec.group_hidden_offsets, spec.up_row_offsets, spec.up_idx, spec.up_val, spec.down_row_offsets, spec.down_idx, spec.down_val, state.gy_t, state.gx_nt, state.gx_t, state.gx_local, state.gz_t, state.gup, state.gdown, state.gbup, state.gbdown, state.bt)
            self._record("backward", state.bt)
        else:
            route = self._routes[layer]
            self.ops.backward_dual_nt(gy, state.x_t, state.z_t, state.h_t, spec.group_hidden_offsets, spec.up_row_offsets, spec.up_idx, spec.up_val, spec.down_row_offsets, spec.down_idx, spec.down_val, route.up_by_input_offsets, route.up_slot_indices, route.up_slot_hidden, route.input_group_offsets, state.gy_t, state.gx_nt, state.gx_t, state.gz_t, state.gup, state.gdown, state.gbup, state.gbdown, state.dbt)
            self._record("backward", state.dbt)
        self._accumulate(layer, state); state.consumed = True
        return {"grad_x": state.gx_nt, "grad_up": state.gup, "grad_down": state.gdown, "grad_b_up": state.gbup, "grad_b_down": state.gbdown}

    def zero_grad(self) -> None:
        for accum in self._accum:
            for value in accum.values():
                value.zero_()
        self._has = [False] * len(self.layers)

    def adamw_all(self, *, lr: float, weight_decay: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        live = [cid for cid, state in self._contexts.items() if not state.consumed]
        if live:
            raise RuntimeError(f"AdamW with live compact contexts: {live}")
        steps = []
        for index in range(len(self.layers)):
            if self._has[index]: self._steps[index] += 1
            steps.append(self._steps[index])
        self.ops.adamw_all([x.up_val for x in self.layers], [a["up"] for a in self._accum], [m.um for m in self._moments], [m.uv for m in self._moments], [x.down_val for x in self.layers], [a["down"] for a in self._accum], [m.dm for m in self._moments], [m.dv for m in self._moments], [x.b_up for x in self.layers], [a["bup"] for a in self._accum], [m.bum for m in self._moments], [m.buv for m in self._moments], [x.b_down for x in self.layers], [a["bdown"] for a in self._accum], [m.bdm for m in self._moments], [m.bdv for m in self._moments], steps, lr, beta1, beta2, eps, weight_decay)
        for state in self._contexts.values(): self._release(state)
        self._contexts.clear(); self._adamw_all_calls += 1; self.zero_grad()

    def state_dict(self) -> dict:
        """Cold, independently owned compact optimizer-boundary snapshot."""
        from .compact_checkpoint import state_dict
        return state_dict(self)

    def load_state_dict(self, payload: dict) -> None:
        """Cold in-place restore; support, ownership and pending-context guards apply."""
        from .compact_checkpoint import load_state_dict
        load_state_dict(self, payload)

    # Single-layer API: thin adapters on the context API above, no other execution path.
    # An implicit forward is keyed by (layer, N) and consumed by the next backward of the same layer and N; several live
    # forwards of one layer and N use the context API. An entry whose context is no longer live (discarded, or consumed
    # through the context API) is replaced by the next implicit forward. The table is created on first use, so the members
    # above keep their attributes and allocations.
    def _implicit_contexts(self) -> dict[tuple[int, int], int]:
        return self.__dict__.setdefault("_implicit", {})

    def _single_layer(self, layer: int) -> None:
        if len(self.layers) != 1 or layer != 0:
            raise RuntimeError("Use adamw_all for a multi-layer HFFN runtime.")

    def forward(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("expected [N,H]")
        key, implicit = (int(layer), int(x.size(0))), self._implicit_contexts()
        pending = self._contexts.get(implicit.get(key, 0))
        if pending is not None and not pending.consumed:
            raise RuntimeError(f"an implicit forward of layer {key[0]} with N = {key[1]} awaits its backward")
        y, cid = self.forward_with_context(layer, x)
        implicit[key] = cid
        return y

    def backward(self, layer: int, gy: torch.Tensor) -> dict[str, torch.Tensor]:
        if gy.dim() != 2 or gy.size(1) != self.layers[layer].d_model:
            raise ValueError("expected [N,H]")
        key, implicit = (int(layer), int(gy.size(0))), self._implicit_contexts()
        if key not in implicit:
            raise RuntimeError("matching HFFN forward must precede backward")
        grads = self.backward_with_context(implicit[key], layer, gy)
        del implicit[key]
        return grads

    def adamw(self, layer: int, *, lr: float, weight_decay: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self._single_layer(layer)
        self.adamw_all(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2, eps=eps)

    def mse_step(self, layer: int, x: torch.Tensor, target: torch.Tensor, *, lr: float, weight_decay: float) -> float:
        """One forward, mean-squared-error backward and AdamW update of a one-layer runtime; checks precede any work."""
        self._single_layer(layer)
        target = self._check(target)
        if x.dim() != 2 or tuple(target.shape) != (int(x.size(0)), self.layers[layer].d_model):
            raise ValueError("mse_step target must be [N,H] like the input")
        if any(not state.consumed for state in self._contexts.values()):
            raise RuntimeError("mse_step with live compact contexts")
        y = self.forward(layer, x); diff = y - target
        self.backward(layer, diff * (2.0 / diff.numel())); self.adamw(layer, lr=lr, weight_decay=weight_decay)
        return float(diff.square().mean().item())
