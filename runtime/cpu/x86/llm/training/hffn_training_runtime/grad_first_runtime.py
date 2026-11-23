"""First-write variant of the compact-G4 training runtime (x86 LLM-training leaf).

The first backward of an update writes its gradients straight into `_accum`; the four gradient scratch tensors of a state
are placeholders until a second microbatch of the same update needs them. The native backward zero-fills `gup`, `gdown`,
`gbup` and assigns `gbdown` before it accumulates (`hffn_compact_g4_training.cpp`), so writing into the zeroed `_accum` is the
same as the `copy_` of the scratch. One microbatch per update never allocates the scratch.
"""
from __future__ import annotations

import torch

from hffn_pythia.compact_format import G4_GROUPS
from hffn_training_runtime.compact_runtime import HFFNCompactG4TrainingRuntime, _State


class HFFNCompactG4TrainingRuntimeFirstWrite(HFFNCompactG4TrainingRuntime):
    def _new_state(self, layer: int, n: int) -> _State:
        spec = self.layers[layer]
        self._workspace_allocations += 1
        gx_local = torch.empty(G4_GROUPS, spec.d_model, n)
        state = _State(0, layer, n, False, False,
            torch.empty(spec.d_model, n), torch.empty(n, spec.d_model), torch.empty(spec.d_model, n), torch.empty(spec.d_ff, n), torch.empty(spec.d_ff, n),
            torch.empty(spec.d_model, n), torch.empty(n, spec.d_model), torch.empty(spec.d_model, n), gx_local, torch.empty(spec.d_ff, n),
            torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0),
            torch.empty(3, dtype=torch.int64), torch.empty(n), torch.empty(n),
            torch.empty(n, spec.d_model), torch.empty(n, spec.d_model), torch.empty(n, spec.d_model), torch.empty(spec.d_model), torch.empty(spec.d_model), torch.empty(4, dtype=torch.int64), torch.empty(5, dtype=torch.int64), torch.empty(4, dtype=torch.int64), torch.empty(4, dtype=torch.int64))
        # Same check as the base runtime, on the buffers allocated here (the placeholders are empty and skipped).
        for name, value in vars(state).items():
            if isinstance(value, torch.Tensor) and value.numel() and value.data_ptr() % 64:
                raise RuntimeError(f"work buffer {name} of layer {layer} (N = {n}) is not 64-byte aligned")
        return state

    def _scratch(self, layer: int, state: _State) -> None:
        """Gradient scratch of a state, allocated once when a second microbatch first needs it."""
        if state.gup.numel():
            return
        spec = self.layers[layer]
        state.gup, state.gdown = torch.empty_like(spec.up_val), torch.empty_like(spec.down_val)
        state.gbup, state.gbdown = torch.empty_like(spec.b_up), torch.empty_like(spec.b_down)
        for name in ("gup", "gdown", "gbup", "gbdown"):
            if getattr(state, name).data_ptr() % 64:
                raise RuntimeError(f"gradient scratch {name} of layer {layer} (N = {state.n}) is not 64-byte aligned")

    def backward_with_context(self, cid: int, layer: int, gy: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self._contexts.get(int(cid)); spec = self.layers[layer]
        if state is None or state.layer_idx != layer or state.consumed:
            raise RuntimeError("invalid compact HFFN context")
        gy = self._check(gy)
        first = not self._has[layer]
        if first:
            accum = self._accum[layer]
            up, down, bup, bdown = accum["up"], accum["down"], accum["bup"], accum["bdown"]
        else:
            self._scratch(layer, state)
            up, down, bup, bdown = state.gup, state.gdown, state.gbup, state.gbdown
        if not state.dual_route:
            self.ops.backward_nt(gy, state.x_t, state.z_t, state.h_t, spec.group_hidden_offsets, spec.up_row_offsets, spec.up_idx, spec.up_val, spec.down_row_offsets, spec.down_idx, spec.down_val, state.gy_t, state.gx_nt, state.gx_t, state.gx_local, state.gz_t, up, down, bup, bdown, state.bt)
            self._record("backward", state.bt)
        else:
            route = self._routes[layer]
            self.ops.backward_dual_nt(gy, state.x_t, state.z_t, state.h_t, spec.group_hidden_offsets, spec.up_row_offsets, spec.up_idx, spec.up_val, spec.down_row_offsets, spec.down_idx, spec.down_val, route.up_by_input_offsets, route.up_slot_indices, route.up_slot_hidden, route.input_group_offsets, state.gy_t, state.gx_nt, state.gx_t, state.gz_t, up, down, bup, bdown, state.dbt)
            self._record("backward", state.dbt)
        if first:
            self._has[layer] = True
        else:
            self._accumulate(layer, state)
        state.consumed = True
        return {"grad_x": state.gx_nt, "grad_up": up, "grad_down": down, "grad_b_up": bup, "grad_b_down": bdown}
