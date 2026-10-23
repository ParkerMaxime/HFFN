"""One training step of one FFN layer: forward, backward, AdamW, on persistent buffers.

- Forward: dual route of the inference kernel of the hardware (``kernels.py`` of the leaf) on the G4 hidden partition of
  the export, with ``z`` stored (a ``z`` buffer distinct from the activation), reading the canonical down values through
  ``down_slots``: no output-order copy, no refresh after AdamW. The smallest N is the one the entry of the leaf declares
  (``INFERENCE_MIN_N``: 2 on x86, whose decode path does not store ``z``; 1 on M1).
- Backward and AdamW: the training kernel, policy ``adaptive`` of ``compact_runtime.py`` (hidden-owned below
  ``DUAL_MIN_N`` tokens, dual route from it).
- Token count: the buffers are reserved once for ``n`` tokens; ``resize(m)`` selects ``n_min <= m <= n`` tokens on the
  same buffers (views of their first ``m`` columns), explicitly and outside the calls. ``n_min`` defaults to ``n``: a
  step of fixed N reserves exactly what it uses.
Every buffer is a torch CPU tensor (64-byte aligned allocations), allocated once.
"""
from __future__ import annotations

import ctypes
import math
from numbers import Real
from pathlib import Path

import torch

from .kernels import _F, _I, _L, _P, INFERENCE_MIN_N, InferenceKernel, _bind  # the entry point of the leaf

from .layer import GROUPS, Layer

DUAL_MIN_N = 32
LR, BETA1, BETA2, EPS, WD = 1e-4, 0.9, 0.999, 1e-8, 0.01


def _validate_adamw_config(lr: Real, weight_decay: Real) -> tuple[float, float]:
    """Validate cold configuration; retain nominal floats for the existing ctypes binding."""
    values = []
    for name, value in (("lr", lr), ("weight_decay", weight_decay)):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real numeric scalar")
        try:
            nominal = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{name} must be finite and nonnegative") from exc
        if not math.isfinite(nominal) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        native = ctypes.c_float(nominal).value
        if not math.isfinite(native) or (value > 0 and native == 0):
            raise ValueError(f"{name} must remain finite and nonzero when positive in native float32")
        values.append(nominal)
    return tuple(values)


class TrainingKernel:
    """Backward and AdamW of the G4 training kernel."""

    def __init__(self, path: Path):
        lib = ctypes.CDLL(str(path))
        self.backward_nt = _bind(lib, "hffn_compact_g4_backward_nt_f32", [_P] * 11 + [_I] * 3 + [_P] * 10)
        self.backward_weights_nt = _bind(lib, "hffn_compact_g4_backward_weights_nt_f32", [_P] * 11 + [_I] * 3 + [_P] * 7)
        self.backward_dual_nt = _bind(lib, "hffn_compact_g4_backward_dual_nt_f32", [_P] * 15 + [_I] * 3 + [_P] * 9)
        self.adamw = _bind(lib, "hffn_compact_g4_adamw_f32", ([_P] * 4 + [_I]) * 4 + [_F] * 5 + [_L])


def _p(tensor: torch.Tensor) -> int:
    assert tensor.is_contiguous()
    return tensor.data_ptr()


class Step:
    def __init__(self, layer: Layer, n: int, training: TrainingKernel, inference: InferenceKernel, n_min: int | None = None,
                 *, lr: Real = LR, weight_decay: Real = WD):
        config = _validate_adamw_config(lr, weight_decay)
        n_min = n if n_min is None else n_min
        if n_min < INFERENCE_MIN_N:
            raise ValueError(f"n >= {INFERENCE_MIN_N} is needed: below it the inference entry of this leaf does not store z")
        if n_min > n:
            raise ValueError(f"n_min {n_min} > n {n}")
        self._adamw_config = config
        self.layer, self.n_max, self.n_min = layer.clone(), n, n_min
        self.training, self.inference = training, inference
        s, h, f = self.layer, layer.d_model, layer.d_ff
        empty = lambda *shape: torch.zeros(*shape, dtype=torch.float32)  # noqa: E731
        # Work buffers, flat, reserved for n tokens: (name, rows, layout [N, rows] if True else [rows, N]).
        self._reserved = {name: (empty(rows * n), rows, nt) for name, rows, nt in (
            ("x_t", h, False), ("y_t", h, False), ("y_nt", h, True), ("z", f, False), ("act", f, False), ("gy_t", h, False),
            ("gx_nt", h, True), ("gx_t", h, False), ("gz", f, False))}
        # The hidden-owned backward (below DUAL_MIN_N tokens) reduces the input gradient through one buffer per group.
        local = min(n, DUAL_MIN_N - 1) if n_min < DUAL_MIN_N else 0
        self._gx_local = empty(GROUPS * h * local)
        up, down = s.up_val.numel(), s.down_val.numel()
        self.g_up, self.g_down, self.g_b_up, self.g_b_down = empty(up), empty(down), empty(f), empty(h)
        self.moments = [empty(up), empty(up), empty(down), empty(down), empty(f), empty(f), empty(h), empty(h)]
        self.timing = torch.zeros(8, dtype=torch.int64)
        self.steps = 0
        self.resize(n)
        for name, tensor in vars(self).items():
            if isinstance(tensor, torch.Tensor) and tensor.numel() and tensor.data_ptr() % 64:
                raise RuntimeError(f"work buffer {name} is not 64-byte aligned")

    @property
    def lr(self) -> float:
        """Constructor-selected nominal learning rate; native conversion remains float32."""
        return self._adamw_config[0]

    @property
    def weight_decay(self) -> float:
        """Constructor-selected nominal weight decay; zero is supported."""
        return self._adamw_config[1]

    def resize(self, m: int) -> None:
        """Select m tokens on the reserved buffers: views, no allocation of work memory, no copy."""
        if not self.n_min <= m <= self.n_max:
            raise ValueError(f"{m} tokens outside the reservation [{self.n_min}, {self.n_max}]")
        self.n = m
        for name, (flat, rows, nt) in self._reserved.items():
            view = flat[: rows * m]
            setattr(self, name, view.view(m, rows) if nt else view.view(rows, m))
        self.backward_dual = m >= DUAL_MIN_N
        self.gx_local = self._gx_local[: GROUPS * self.layer.d_model * m].view(GROUPS, self.layer.d_model, m) \
            if not self.backward_dual else self._gx_local[:0]

    def forward(self, x_nt: torch.Tensor) -> torch.Tensor:
        s, n, h, f = self.layer, self.n, self.layer.d_model, self.layer.d_ff
        self.inference.forward_dual_slots_nt(GROUPS, _p(x_nt), _p(s.groups), _p(s.b_up), _p(s.up_offsets), _p(s.up_idx), _p(s.up_val),
                                             _p(s.b_down), _p(s.down_by_output), _p(s.down_hidden), _p(s.down_val), _p(s.down_slots),
                                             _p(s.output_groups), n, h, f, _p(self.x_t), _p(self.y_nt), _p(self.y_t), _p(self.z),
                                             _p(self.act), _p(self.timing))
        return self.y_nt

    def backward(self, gy_nt: torch.Tensor, *, input_grad: bool = True) -> None:
        """Compute parameter gradients; input_grad=False leaves all input-gradient buffers untouched.

        After a weights-only call, results()['grad_x'] is unspecified. The
        default keeps the existing adaptive full-backward route.
        """
        s, n, h, f = self.layer, self.n, self.layer.d_model, self.layer.d_ff
        common = [_p(gy_nt), _p(self.x_t), _p(self.z), _p(self.act), _p(s.groups), _p(s.up_offsets), _p(s.up_idx), _p(s.up_val),
                  _p(s.down_offsets), _p(s.down_idx), _p(s.down_val)]
        grads = [_p(self.gz), _p(self.g_up), _p(self.g_down), _p(self.g_b_up), _p(self.g_b_down), _p(self.timing[4:])]
        if not input_grad:
            self.training.backward_weights_nt(*common, n, h, f, _p(self.gy_t), *grads)
        elif self.backward_dual:
            self.training.backward_dual_nt(*common, _p(s.up_by_input), _p(s.up_slots), _p(s.up_hidden), _p(s.input_groups), n, h, f,
                                           _p(self.gy_t), _p(self.gx_nt), _p(self.gx_t), *grads)
        else:
            self.training.backward_nt(*common, n, h, f, _p(self.gy_t), _p(self.gx_nt), _p(self.gx_t), _p(self.gx_local), *grads)

    def update(self) -> None:
        s, m = self.layer, self.moments
        self.steps += 1
        self.training.adamw(_p(s.up_val), _p(self.g_up), _p(m[0]), _p(m[1]), s.up_val.numel(),
                            _p(s.down_val), _p(self.g_down), _p(m[2]), _p(m[3]), s.down_val.numel(),
                            _p(s.b_up), _p(self.g_b_up), _p(m[4]), _p(m[5]), s.d_ff,
                            _p(s.b_down), _p(self.g_b_down), _p(m[6]), _p(m[7]), s.d_model,
                            self._adamw_config[0], BETA1, BETA2, EPS, self._adamw_config[1], self.steps)

    def step(self, x_nt: torch.Tensor, gy_nt: torch.Tensor) -> None:
        self.forward(x_nt)
        self.backward(gy_nt)
        self.update()

    def results(self) -> dict[str, torch.Tensor]:
        """Forward and backward results of the last step, [N, ...] for tokens (copies)."""
        return {"y": self.y_nt.clone(), "z": self.z.T.clone(), "h": self.act.T.clone(), "grad_x": self.gx_nt.clone(),
                "grad_up": self.g_up.clone(), "grad_down": self.g_down.clone(), "grad_b_up": self.g_b_up.clone(),
                "grad_b_down": self.g_b_down.clone()}
