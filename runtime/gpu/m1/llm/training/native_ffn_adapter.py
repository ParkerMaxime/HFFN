"""Native backward of one HFFN FFN layer.

The native backward is the route of the leaf, with no
opt-in flag and no second route; the host-side bridging costs live in this adapter only. The forward stays the Torch route
of the leaf (``ExactOffsetMetalFFN.forward``); the backward of the block of N tokens is taken from the owner's native
staged context of ``gpu/m1/ffn/training`` (``stage_begin``, ``stage_forward`` recompute, ``stage_backward``), created
over the module's ``flat`` pool. It also holds the compact AdamW: one dedicated optimizer context per layer
(``prepare_optimizer``, ``optimizer_step``, ``export_moments``, ``restore_moments``), the owner's optimizer-only step.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import torch

TORCH_VERSION = "2.10.0"
if torch.__version__ != TORCH_VERSION:
    raise RuntimeError(f"native_ffn_adapter requires torch {TORCH_VERSION} (internal header), found {torch.__version__}")

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
OWNER = ROOT / "runtime/gpu/m1/ffn/training"
if str(OWNER) not in sys.path:
    sys.path.append(str(OWNER))  # appended, not prepended: native_context imports export_input by bare name

import native_context  # noqa: E402
from hffn_metal_llm.model import HFFNMetalMLP  # noqa: E402

BRIDGE_PATH = Path(__file__).resolve().parent / "build" / "native_ffn_bridge.so"


def _load_bridge():
    loader = importlib.machinery.ExtensionFileLoader("native_ffn_bridge", str(BRIDGE_PATH))
    spec = importlib.util.spec_from_file_location("native_ffn_bridge", BRIDGE_PATH, loader=loader)
    if spec is None or not BRIDGE_PATH.is_file():
        raise ImportError(f"bridge not built: {BRIDGE_PATH} (run build_bridge.sh)")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    sys.modules["native_ffn_bridge"] = module
    return module


bridge = _load_bridge()

STORAGE_PRIVATE = int(bridge.STORAGE_PRIVATE)
POOL_ALIGNMENT = 256  # bytes; F11
# Native configuration of the 2c cells: owner_tile and flat_adamw_pool keep their
# owner defaults; "--n" is prepended per context.
NATIVE_ARGUMENTS = ("--accum-steps", "1", "--shared-readonly", "--cache-gelu-grad")
PARAMETER_VIEWS = ("up", "down", "bup", "bdown")
# The dedicated optimizer context of a layer: N = 1, owner policy of the leaf.
# "--optimizer-only": the owner allocates only the flat AdamW state, no backward buffers.
OPTIMIZER_ARGUMENTS = ("--n", "1", "--accum-steps", "1", "--shared-readonly", "--adamw-llm-policy", "--optimizer-only")
MOMENT_VIEWS = (("m_up", "m_down", "m_bup", "m_bdown"), ("v_up", "v_down", "v_bup", "v_bdown"))
# Declared test switch of the optimizer gates only: None outside them.
OPTIMIZER_FAULTS = frozenset({"F_scalar", "F_stale", "F_skip_blit", "F_keep_grad"})
GRADIENT_VIEWS = ("g_up", "g_down", "g_bup", "g_bdown")
# Declared test switch, for the gates only: an adapter's ``test_fault`` is None outside them.
TEST_FAULTS = frozenset({"F2_clear_before_blit", "F3_skip_clear", "F4_skip_version", "F5_alias_staging",
                         "F9_alloc_after_barrier", "F10_scale"})
# F6 acts on the support conversion, at construction (``fault_prepare``), not per call.
PREPARE_FAULTS = frozenset({"F6_permute_slots"})


def f6_target_row(up_ptr, up_idx, down_ptr, require_down: bool = True) -> int:
    """First up row with two distinct slots (and, by default, a down degree >= 1): the row the F6 test fault permutes."""
    for r in range(up_ptr.numel() - 1):
        a = int(up_ptr[r])
        if int(up_ptr[r + 1]) - a >= 2 and int(up_idx[a]) != int(up_idx[a + 1]):
            if not require_down or int(down_ptr[r + 1]) - int(down_ptr[r]) >= 1:
                return r
    raise RuntimeError("F6: no up row with two distinct slots" + (" and a down degree >= 1" if require_down else ""))


def support_payload(spec, fault: str | None = None) -> dict:
    """The ``.pt`` spec of one layer as the owner's exact ``Support`` payload, identity order.

    Rows are hidden neurons; slot ``e`` of row ``r`` is ``up_val[up_row_offsets[r] + e]`` (resp. down), exactly the
    order of the module's ``flat = cat(up_val, down_val, b_up, b_down)``. Nothing is sorted or regrouped, and the
    owner's loader (``native_exact_support.h``) keeps that order. Indices are widened from int16 to int32."""
    tensors = {
        "up_ptr": spec.up_row_offsets, "up_idx": spec.up_idx, "up_val": spec.up_val,
        "down_ptr": spec.down_row_offsets, "down_idx": spec.down_idx, "down_val": spec.down_val,
        "b_up": spec.b_up, "b_down": spec.b_down,
    }
    payload = {"format": "hidden_ffn",
               "metadata": {"d_model": int(spec.d_model), "d_ff": int(spec.d_ff), "activation": "gelu"}}
    for name, value in tensors.items():
        if value.device.type != "cpu" or value.dim() != 1:
            raise ValueError(f"{name}: expected a CPU vector")
        if name.endswith(("_ptr", "_idx")):
            value = value.to(torch.int32)
        elif value.dtype != torch.float32:
            raise ValueError(f"{name}: expected float32")
        payload[name] = value.detach().contiguous()
    if fault is not None:
        if fault not in PREPARE_FAULTS:
            raise ValueError(f"unknown prepare fault {fault!r}")
        # F6 (test only): swap the column indices of the first two slots of the target up row, leaving the values (read from
        # the module's flat) in place: a permuted slot order. Target: the first up
        # row with two distinct slots AND a down degree >= 1 (a hidden neuron with no outgoing connection reaches neither y
        # nor a gradient, so a swap there is null by construction).
        row = f6_target_row(payload["up_ptr"], payload["up_idx"], payload["down_ptr"])
        ptr, idx = payload["up_ptr"], payload["up_idx"].clone()
        a = int(ptr[row])
        idx[a], idx[a + 1] = idx[a + 1].clone(), idx[a].clone()
        payload["up_idx"] = idx
    return payload


class _Context:
    """The native context of one (layer, N), its persistent staging tensors and its blit queue."""

    __slots__ = ("n", "native", "queue", "x_stage", "gy_stage", "gx_stage")

    def __init__(self, n: int) -> None:
        self.n = n
        self.native = self.queue = self.x_stage = self.gy_stage = self.gx_stage = None

    def release(self) -> None:
        # The native context borrows the staging buffers: it is closed first, the staging tensors are dropped last.
        torch.mps.synchronize()  # cold path: no torch kernel still reads a staging tensor (audit R6)
        try:
            if self.native is not None:
                self.native.close()
        finally:
            self.native = None
            try:
                if self.queue is not None:
                    self.queue.close()
            finally:
                self.queue = None
                self.x_stage = self.gy_stage = self.gx_stage = None


class _OptimizerContext:
    """The dedicated native optimizer context of one layer and its blit queue (never discarded by a backward error)."""

    __slots__ = ("native", "queue")

    def __init__(self) -> None:
        self.native = self.queue = None

    def release(self) -> None:
        torch.mps.synchronize()
        try:
            if self.native is not None:
                self.native.close()
        finally:
            self.native = None
            if self.queue is not None:
                self.queue.close()
            self.queue = None


class NativeFFNAdapter:
    """Native backward of one layer: one resident native context, for the last N prepared, created by ``prepare(n)`` only."""

    def __init__(self, layer_idx: int, spec, flat: torch.Tensor, library, bundle=None, *, fault_prepare: str | None = None) -> None:
        if bundle is not None and not callable(bundle):
            raise TypeError("bundle: a callable (payload, arguments) -> bundle directory, or None for the owner's "
                            "prepare_exact_sources (the owner's generated sources depend on --n)")
        self.layer_idx = int(layer_idx)
        self.spec = spec
        self.flat = flat
        self.library = Path(library)
        self._bundle = bundle if bundle is not None else native_context.prepare_exact_sources
        self.payload = support_payload(spec, fault_prepare)
        self.d_model = int(spec.d_model)
        self.counts = (int(spec.up_val.numel()), int(spec.down_val.numel()), int(spec.d_ff), int(spec.d_model))
        self.pool = sum(self.counts)
        self._contexts: dict[int, _Context] = {}
        self._flat_key: tuple[int, int] | None = None
        self.counters = {"native_calls": 0, "forward_calls": 0, "contexts_created": 0, "contexts_discarded": 0, "lazy_prepares": 0}
        self.test_fault: str | None = None
        self.test_scale: float | None = None  # F10: the factor 1 + e
        self._optimizer: _OptimizerContext | None = None
        self.optimizer_fault: str | None = None
        self._previous_scalars: tuple[float, ...] | None = None

    # --- binding (cold, outside any timed window) -------------------------------------------------------------
    def _bind_flat(self) -> tuple[int, int]:
        flat = self.flat
        if (not isinstance(flat, torch.Tensor) or flat.device.type != "mps" or flat.dtype != torch.float32
                or flat.dim() != 1 or not flat.is_contiguous() or flat.numel() != self.pool):
            raise ValueError(f"flat: expected a contiguous MPS float32 vector of {self.pool} entries")
        handle, offset, mode, _ = bridge.buffer_info(flat)
        if mode != STORAGE_PRIVATE:
            raise RuntimeError(f"flat: storage mode {mode}, expected Private ({STORAGE_PRIVATE})")
        if offset % POOL_ALIGNMENT:
            raise ValueError(f"flat: pool byte offset {offset} is not a multiple of {POOL_ALIGNMENT}")
        if handle != flat.untyped_storage().data_ptr():
            raise RuntimeError("flat: bridge handle differs from the storage pointer")
        key = (flat.data_ptr(), flat.storage_offset())
        if self._flat_key is None:
            self._flat_key = key
        elif key != self._flat_key:
            raise RuntimeError("flat storage changed after binding")
        return handle, offset

    @staticmethod
    def _bind_stage(tensor: torch.Tensor, name: str) -> tuple[int, int]:
        handle, offset, mode, _ = bridge.buffer_info(tensor)
        if mode != STORAGE_PRIVATE:
            raise RuntimeError(f"{name}: storage mode {mode}, expected Private ({STORAGE_PRIVATE})")
        return handle, offset

    def _check_parameter_views(self, native, handle: int, offset: int) -> None:
        """The four parameter parts of the native context are the module's flat, in its order and at its offsets."""
        start = 0
        for name, count in zip(PARAMETER_VIEWS, self.counts):
            view = native.view(name)
            if view.metal_buffer != handle or view.byte_offset != offset + 4 * start or view.elements != count:
                raise RuntimeError(f"native parameter view {name} is not flat[{start}:{start + count}]")
            start += count

    def prepare(self, n: int) -> None:
        """Creates the native context of N, its staging tensors and its blit queue, and binds them once.

        One context is resident per layer: preparing an N first discards this layer's
        contexts of every other N. A context holds no state across calls (its gradients are cleared at step 7 of
        ``backward``; the moments live in the optimizer context, never discarded), so an evicted N is prepared again at
        its next use, a declared bridging cost counted by ``lazy_prepares`` and ``contexts_discarded``."""
        n = int(n)
        if n < 1:
            raise ValueError("N must be positive")
        if n in self._contexts:
            return
        if torch.__version__ != TORCH_VERSION:
            raise RuntimeError(f"torch {TORCH_VERSION} required, found {torch.__version__}")
        for other in [key for key in self._contexts if key != n]:
            self._discard(other)
        handle, offset = self._bind_flat()
        arguments = ("--n", str(n), *NATIVE_ARGUMENTS)
        bundle = self._bundle(self.payload, arguments)
        context = _Context(n)
        try:
            context.x_stage = torch.empty(self.d_model, n, device="mps", dtype=torch.float32)
            context.gy_stage = torch.empty(self.d_model, n, device="mps", dtype=torch.float32)
            context.gx_stage = torch.empty(self.d_model, n, device="mps", dtype=torch.float32)
            x = self._bind_stage(context.x_stage, "x_stage")
            gy = self._bind_stage(context.gy_stage, "gy_stage")
            gx = self._bind_stage(context.gx_stage, "gx_stage")
            context.native = native_context.Context(self.library, bundle, arguments, support=self.payload,
                                                    parameters=(handle, offset, self.pool))
            self._check_parameter_views(context.native, handle, offset)
            context.native.bind_activations(x=[x], gy=[gy], gx=gx)
            context.queue = bridge.BlitQueue()
        except BaseException:
            context.release()
            raise
        self._contexts[n] = context
        self.counters["contexts_created"] += 1

    # --- compact AdamW ----------------------------------------------
    def prepare_optimizer(self) -> None:
        """Cold: the dedicated optimizer context of this layer over the module's flat pool, caller-managed step."""
        if self._optimizer is not None:
            return
        handle, offset = self._bind_flat()
        arguments = OPTIMIZER_ARGUMENTS
        bundle = self._bundle(self.payload, arguments)
        optimizer = _OptimizerContext()
        try:
            optimizer.native = native_context.Context(self.library, bundle, arguments, support=self.payload,
                                                      parameters=(handle, offset, self.pool))
            self._check_parameter_views(optimizer.native, handle, offset)
            optimizer.native.enable_caller_managed_step()
            optimizer.queue = bridge.BlitQueue()
        except BaseException:
            optimizer.release()
            raise
        self._optimizer = optimizer

    @property
    def optimizer_updates(self) -> int:
        return int(self._optimizer.native.completed_updates) if self._optimizer is not None else 0

    def optimizer_step(self, scalars: tuple[float, ...], grad: torch.Tensor) -> None:
        """One update of the compact pool from ``grad`` (``flat.grad``). ``scalars`` are the
        seven float32 values of the owner's update (lr, beta1, beta2, eps, weight decay, step_size, beta2_scale)."""
        self.prepare_optimizer()
        optimizer = self._optimizer
        native = optimizer.native
        fault = self.optimizer_fault
        if fault is not None and fault not in OPTIMIZER_FAULTS:
            raise ValueError(f"unknown optimizer fault {fault!r}")
        if (not isinstance(grad, torch.Tensor) or grad.device.type != "mps" or grad.dtype != torch.float32
                or grad.dim() != 1 or not grad.is_contiguous() or grad.numel() != self.pool):
            raise ValueError(f"grad: expected a contiguous MPS float32 vector of {self.pool} entries")
        if (self.flat.data_ptr(), self.flat.storage_offset()) != self._flat_key:
            raise RuntimeError(f"layer {self.layer_idx}: flat storage changed after binding")
        values = tuple(float(v) for v in scalars)
        if fault == "F_scalar":  # test only: step_size one float32 ulp up
            import numpy
            values = values[:5] + (float(numpy.nextafter(numpy.float32(values[5]), numpy.float32(numpy.inf))),) + values[6:]
        if fault == "F_stale" and self._previous_scalars is not None:
            values = self._previous_scalars  # test only: the setter keeps the scalars of the previous step
        self._previous_scalars = values
        torch.mps.synchronize()  # the Torch queue is drained: the backward's writes to grad and the previous forward are done
        native.stage_optimizer_begin()
        native.set_update_scalars(values)
        destinations = []
        for name, count in zip(GRADIENT_VIEWS, self.counts):
            view = native.stage_view(name)
            if view.elements != count:
                raise RuntimeError(f"native gradient view {name}: {view.elements} entries, expected {count}")
            destinations.append((view.metal_buffer, view.byte_offset, view.elements))
        if fault != "F_skip_blit":
            optimizer.queue.copy_in(grad, destinations)
        native.stage_update()
        bridge.increment_version(self.flat)
        if fault != "F_keep_grad":
            grad.zero_()

    def export_moments(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The two moments in flat order (CPU float32), read from the host-accessible views of the optimizer context."""
        self.prepare_optimizer()
        native = self._optimizer.native
        result = []
        for names in MOMENT_VIEWS:
            parts = [torch.frombuffer(bytearray(native.snapshot(name)), dtype=torch.float32) for name in names]
            result.append(torch.cat(parts))
        return result[0], result[1]

    def restore_moments(self, first: torch.Tensor, second: torch.Tensor, completed: int) -> None:
        """Cold import of the moments (flat order) and of the counter into the optimizer context."""
        self.prepare_optimizer()
        first = first.detach().to(device="cpu", dtype=torch.float32).contiguous()
        second = second.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if first.numel() != self.pool or second.numel() != self.pool:
            raise ValueError(f"moments: expected {self.pool} entries")
        self._optimizer.native.restore_optimizer(first.numpy().tobytes(), second.numpy().tobytes(), int(completed))

    def close(self) -> None:
        if getattr(self, "_optimizer", None) is not None:
            optimizer, self._optimizer = self._optimizer, None
            optimizer.release()
        while self._contexts:
            _, context = self._contexts.popitem()
            context.release()

    def __del__(self) -> None:
        if getattr(self, "_contexts", None) or getattr(self, "_optimizer", None) is not None:
            self.close()

    # --- per call ---------------------------------------------------------------------------------------------
    def check_call(self, n: int, flat: torch.Tensor) -> _Context:
        """Prepares an N not yet seen (lazy preparation: a declared bridging cost; the owner refuses N >= 8192
        and the error is raised as is) and raises on a flat storage replaced after binding (F8)."""
        context = self._contexts.get(n)
        if context is None:
            self.prepare(n)
            self.counters["lazy_prepares"] += 1
            context = self._contexts[n]
        if (flat.data_ptr(), flat.storage_offset()) != self._flat_key:
            raise RuntimeError(f"layer {self.layer_idx}: flat storage changed after binding")
        return context

    def _discard(self, n: int) -> None:
        context = self._contexts.pop(n, None)
        self.counters["contexts_discarded"] += 1
        if context is not None:
            context.release()

    def backward(self, n: int, x: torch.Tensor, gy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Steps 1 to 7 of the native backward, in that order."""
        context = self.check_call(n, self.flat)
        if gy.device.type != "mps" or gy.dtype != torch.float32 or gy.shape != x.shape:
            raise ValueError(f"grad_y: expected MPS float32 of shape {tuple(x.shape)}")
        d = self.d_model
        lead = tuple(x.shape[:-1])
        fault = self.test_fault
        if fault is not None and fault not in TEST_FAULTS:
            raise ValueError(f"unknown test fault {fault!r}")
        # Step 1: the outputs, before the barrier of step 3.
        if fault != "F9_alloc_after_barrier":
            if fault == "F5_alias_staging":  # test only: grad_x aliases the x staging tensor (caught by the F5 check)
                grad_x = context.x_stage.view(-1).view(x.shape)
            else:
                grad_x = torch.empty(x.shape, device="mps", dtype=torch.float32)
            grad_flat = torch.empty(self.pool, device="mps", dtype=torch.float32)
        # Step 2: x and gy into the persistent Private staging tensors, [d, N] (torch queue).
        context.x_stage.view(d, *lead).copy_(x.movedim(-1, 0))
        context.gy_stage.view(d, *lead).copy_(gy.movedim(-1, 0))
        # Step 3: the barrier; also drains the asynchronous HFFNMetalAdamW of the previous step. F1 replaces this
        # reference by a no-op.
        torch.mps.synchronize()
        if fault == "F9_alloc_after_barrier":
            recycled = torch.empty(self.pool, device="mps", dtype=torch.float32)
            recycled.fill_(1.0)
            del recycled
            grad_x = torch.empty(x.shape, device="mps", dtype=torch.float32)
            grad_flat = torch.empty(self.pool, device="mps", dtype=torch.float32)
        # F5: the outputs alias neither a staging tensor nor the pool.
        bound = {t.untyped_storage().data_ptr()
                 for t in (context.x_stage, context.gy_stage, context.gx_stage, self.flat)}
        if (grad_x.untyped_storage().data_ptr() in bound or grad_flat.untyped_storage().data_ptr() in bound
                or grad_x.untyped_storage().data_ptr() == grad_flat.untyped_storage().data_ptr()):
            raise AssertionError("grad_x or grad_flat aliases a bound buffer")
        native = context.native
        try:
            # Step 4: native begin, forward recompute, backward (two blocking waits inside the owner).
            native.stage_begin()
            native.stage_forward(0)
            native.stage_backward(0)
            self.counters["native_calls"] += 1
            # Step 5: gx out of its staging tensor, back to [N, d] (torch queue).
            grad_x.copy_(context.gx_stage.view(d, *lead).movedim(0, -1))
            # Step 6: the four compact gradient views into grad_flat, one blit on the adapter queue, blocking wait,
            # then the version bump the native write bypasses.
            sources = []
            for name, count in zip(GRADIENT_VIEWS, self.counts):
                view = native.stage_view(name)
                if view.elements != count:
                    raise RuntimeError(f"native gradient view {name}: {view.elements} entries, expected {count}")
                sources.append((view.metal_buffer, view.byte_offset, view.elements))
            version = grad_flat._version
            if fault == "F2_clear_before_blit":
                context.queue.copy(grad_flat, sources, True)
                native.stage_clear_gradients()
                context.queue.release()
            else:
                context.queue.copy(grad_flat, sources, False)
            if fault != "F4_skip_version":
                bridge.increment_version(grad_flat)
            if grad_flat._version == version:
                raise AssertionError("grad_flat version did not move across the native write")
            if fault == "F10_scale":
                grad_x.mul_(self.test_scale)
                grad_flat.mul_(self.test_scale)
            # Step 7: only after the blit wait, the host memset of the native gradient pool (context back to ready).
            if fault not in ("F2_clear_before_blit", "F3_skip_clear"):
                native.stage_clear_gradients()
        except BaseException:
            self._discard(n)
            raise
        return grad_x, grad_flat


class NativeFFNFunction(torch.autograd.Function):
    """Forward: the Torch forward of the leaf. Backward: ``NativeFFNAdapter.backward``."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, flat: torch.Tensor, module: HFFNMetalMLP, adapter: NativeFFNAdapter):
        shape = tuple(x.shape)
        n = int(x.numel() // shape[-1])
        adapter.check_call(n, flat)
        y, _ = module.runtime_for_n(n).forward(x.reshape(-1, shape[-1]))
        adapter.counters["forward_calls"] += 1
        ctx.save_for_backward(x)
        ctx.key = (adapter, n)
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        (x,) = ctx.saved_tensors
        adapter, n = ctx.key
        grad_x, grad_flat = adapter.backward(n, x, grad_y)
        return grad_x, grad_flat, None, None


class NativeBackwardMLP(HFFNMetalMLP):
    """``HFFNMetalMLP`` whose training backward is native. The grad-disabled forward is the parent's."""

    def __init__(self, layer_idx: int, spec, library, bundle=None) -> None:
        super().__init__(layer_idx, spec)
        self.adapter = NativeFFNAdapter(layer_idx, spec, self.flat, library, bundle)

    def prepare(self, n: int) -> None:
        self.adapter.prepare(n)

    def close(self) -> None:
        self.adapter.close()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return super().forward(x)
        return NativeFFNFunction.apply(x, self.flat, self, self.adapter)
