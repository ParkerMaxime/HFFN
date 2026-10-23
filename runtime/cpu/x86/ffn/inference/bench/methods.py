"""The methods compared in ``vs_dense.py`` and ``process_footprint.py``: same entry point
(float32 x[N, d] -> y[N, d]), model bytes, work buffers held between calls.

- ``hffn_compact``   the runtime: x86 compact kernel on --threads threads, recommended route
                     (decode N = 1, dual N >= 2), without the hidden-major down arrays
- ``dense_eager``    F.linear -> F.gelu -> F.linear (allocates h and y on each call)
- ``dense_prealloc`` addmm(out=) into a preallocated h, in-place GELU, pre-transposed weights
                     (the dense counterpart of HFFN's reused work buffers)
- ``dense_compile``  torch.compile (Inductor, frozen weights), one static graph per (layer, N)
- ``dense_numpy``    numpy/OpenBLAS + scipy erf, no PyTorch (leanest dense deployment)
"""
from __future__ import annotations

import numpy as np

import common
import hffn  # noqa: F401  (puts the shared test layers on sys.path)
from hffn_ffn_layers.dense import dense_weights


class HffnCompact:
    name = "hffn_compact"

    def __init__(self, layer):
        # Recommended routes only: the hidden-major down arrays are neither kept nor counted.
        self.kernel = common.compact_kernel(layer, hidden_owned=False)
        self.model_bytes = self.kernel.model_bytes()

    def supports(self, n):
        return True

    def forward(self, x):
        return self.kernel.forward(x)

    def held_bytes(self):
        return self.kernel.workspace_bytes()


class DenseEager:
    name = "dense_eager"

    def __init__(self, layer):
        import torch
        self.torch, self.F = torch, torch.nn.functional
        w_up, w_down = dense_weights(layer)
        self.w_up, self.w_down = torch.from_numpy(w_up), torch.from_numpy(w_down)
        self.b_up, self.b_down = torch.from_numpy(layer.b_up.copy()), torch.from_numpy(layer.b_down.copy())
        self.model_bytes = sum(t.nbytes for t in (self.w_up, self.w_down, self.b_up, self.b_down))

    def supports(self, n):
        return True

    def forward(self, x):
        F = self.F
        with self.torch.inference_mode():
            return F.linear(F.gelu(F.linear(self.torch.from_numpy(x), self.w_up, self.b_up)), self.w_down, self.b_down).numpy()

    def held_bytes(self):
        return 0


class DensePrealloc(DenseEager):
    name = "dense_prealloc"

    def __init__(self, layer):
        super().__init__(layer)
        self.w_up_t, self.w_down_t = self.w_up.t().contiguous(), self.w_down.t().contiguous()  # [d, F], [F, d]
        del self.w_up, self.w_down
        self.model_bytes = sum(t.nbytes for t in (self.w_up_t, self.w_down_t, self.b_up, self.b_down))
        self.h = self.torch.empty(0)

    def forward(self, x):
        torch, n, f = self.torch, x.shape[0], self.w_up_t.shape[1]
        if self.h.numel() < n * f:
            self.h = torch.empty(n * f)
        h = self.h[: n * f].view(n, f)
        with torch.inference_mode():
            torch.addmm(self.b_up, torch.from_numpy(x), self.w_up_t, out=h)
            torch._C._nn.gelu_(h)
            y = torch.empty(n, self.w_down_t.shape[1])
            torch.addmm(self.b_down, h, self.w_down_t, out=y)
        return y.numpy()

    def held_bytes(self):
        return self.h.nbytes


class DenseCompile(DenseEager):
    name = "dense_compile"

    def __init__(self, layer):
        import torch._dynamo.config as dynamo
        import torch._inductor.config as inductor
        inductor.freezing = True
        inductor.compile_threads = 1
        dynamo.recompile_limit = 256              # one static graph per (layer, N): the default 8
        dynamo.accumulated_recompile_limit = 4096  # would silently fall back to eager
        super().__init__(layer)
        torch, nn = self.torch, self.torch.nn

        class Module(nn.Module):
            def __init__(self, w_up, b_up, w_down, b_down):
                super().__init__()
                self.up, self.down = nn.Linear(w_up.shape[1], w_up.shape[0]), nn.Linear(w_down.shape[1], w_down.shape[0])
                with torch.no_grad():
                    self.up.weight.copy_(w_up); self.up.bias.copy_(b_up); self.down.weight.copy_(w_down); self.down.bias.copy_(b_down)

            def forward(self, x):
                return self.down(torch.nn.functional.gelu(self.up(x)))

        self.module = Module(self.w_up, self.b_up, self.w_down, self.b_down).eval()
        del self.w_up, self.w_down
        self.model_bytes = sum(p.nbytes for p in self.module.parameters())
        self.compiled = torch.compile(self.module, dynamic=False)

    def forward(self, x):
        with self.torch.no_grad():
            return self.compiled(self.torch.from_numpy(x)).numpy()


class DenseNumpy:
    name = "dense_numpy"
    # OpenBLAS workers busy-wait ~0.1 s after a call: the next block must not overlap them.
    settle_s = 0.3

    def __init__(self, layer):
        from scipy.special import erf
        self.erf = erf
        w_up, w_down = dense_weights(layer)
        self.w_up_t, self.w_down_t = np.ascontiguousarray(w_up.T), np.ascontiguousarray(w_down.T)
        self.b_up, self.b_down = layer.b_up.copy(), layer.b_down.copy()
        self.model_bytes = sum(a.nbytes for a in (self.w_up_t, self.w_down_t, self.b_up, self.b_down))
        self.h, self.t = np.empty(0, np.float32), np.empty(0, np.float32)

    def supports(self, n):
        return True

    def forward(self, x):
        n, f = x.shape[0], self.w_up_t.shape[1]
        if self.h.size < n * f:
            self.h, self.t = np.empty(n * f, np.float32), np.empty(n * f, np.float32)
        h, t = self.h[: n * f].reshape(n, f), self.t[: n * f].reshape(n, f)
        np.matmul(x, self.w_up_t, out=h)
        h += self.b_up
        np.multiply(h, np.float32(0.7071067811865476), out=t)   # gelu(h) = 0.5 h (1 + erf(h / sqrt 2))
        self.erf(t, out=t)
        t += 1.0
        h *= t
        h *= 0.5
        y = np.matmul(h, self.w_down_t)
        y += self.b_down
        return y

    def held_bytes(self):
        return self.h.nbytes + self.t.nbytes


METHODS = {cls.name: cls for cls in (HffnCompact, DenseEager, DensePrealloc, DenseCompile, DenseNumpy)}
