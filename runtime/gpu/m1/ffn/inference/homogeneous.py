"""Isolated sparse FFN on the GPU M1 FFN kernel (`native_transformer/hffn` of the LLM inference leaf).

One FFN (per hidden unit its up offsets, indices and values, its bias, and its down connections) is presented to
the kernel as one layer with one bank and one route. Weights, indices and offsets stay compact; offsets are int32,
indices int16, or int32 when a width reaches 32 768, cast once at load. The input and the output may have
different widths.
"""
import sys
from pathlib import Path

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'runtime'/'common'/'pythia'/'hffn_pythia').is_dir())
LLM_LEAF = ROOT/'runtime/gpu/m1/llm/inference'
for path in (LLM_LEAF/'pythia_torch_attention', LLM_LEAF):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from native_transformer.hffn.bank_store import INDEX_LIMIT, device_bank, validate    # noqa: E402
from hffn_metal_llm_inference.model import NativeHFFNInference          # noqa: E402

EXPORT = 'hidden_ffn_format.pt'


class OneBankStore:
    """The bank store of the FFN kernel over one FFN: one layer, one bank, route 0 runs bank 0.

    `ffn`: `up_ptr`, `up_idx`, `up_val` and `down_ptr`, `down_idx`, `down_val` grouped by hidden unit, `b_up`
    [d_ff], `b_down` [d_out]; `d_in` is the width of the input."""

    def __init__(self, ffn, d_in, activation='gelu'):
        d_ff, d_out = ffn['b_up'].numel(), ffn['b_down'].numel()
        self.layers, self.pairs, self.banks_per_layer = 1, ((),), 1
        self.d_model, self.d_out, self.d_ff = int(d_in), d_out, d_ff      # d_model: the kernel's name for the input width
        self.activation = activation
        self.index_dtype = torch.int16 if max(self.d_model, d_out, d_ff) < INDEX_LIMIT else torch.int32
        self.widths = [[d_ff]]
        self.output_biases = (ffn['b_down'].float(),)
        self._raw = {'d_model': self.d_model, 'd_out': d_out, 'd_ff': d_ff, 'b_up': ffn['b_up'].float(),
                     'up_row_offsets': ffn['up_ptr'].int(), 'up_idx': ffn['up_idx'].to(self.index_dtype),
                     'up_val': ffn['up_val'].float(), 'down_row_offsets': ffn['down_ptr'].int(),
                     'down_idx': ffn['down_idx'].to(self.index_dtype), 'down_val': ffn['down_val'].float()}

    def load(self, layer, bank):
        if (layer, bank) != (0, 0):
            raise KeyError('an isolated FFN has one layer and one bank')
        validate(self._raw, self.d_model, self.index_dtype)
        return device_bank(self._raw)


class HomogeneousExportStore(OneBankStore):
    """`OneBankStore` over an exported model of `$HFFN_ASSETS/homogeneous/models` (same input and output width)."""

    def __init__(self, export_dir, activation=None):
        data = torch.load(Path(export_dir)/EXPORT, map_location='cpu', weights_only=True)
        # the given activation, else the one the export declares, else GELU (the older exports declare none)
        super().__init__(data, data['b_down'].numel(), activation or data['metadata'].get('activation', 'gelu'))


class FFNInference:
    """y = W_down activation(W_up x + b_up) + b_down for x: MPS FP32 [N, d_in]; y: [N, d_out].

    One call runs the whole block of N tokens; `workspace_limit` bounds the dense buffers it may hold."""

    def __init__(self, store, workspace_limit=None):
        self.store = store
        self.runtime = NativeHFFNInference(store, workspace_limit)    # bytes; None: the kernel's default (64 MiB)

    def forward(self, x):
        n, d_in = x.shape
        shaped = x.view(n, 1, d_in) if n <= 4 else x.view(1, n, d_in)    # decode or prefill shape of the kernel
        return self.runtime.forward(0, shaped).view(n, self.store.d_out)

    def workspace_stats(self):
        return self.runtime.workspace_stats()

    def synchronize(self):
        self.runtime.synchronize()


class HomogeneousFFNInference(FFNInference):
    """`FFNInference` over an exported model. The activation is the one the export declares (`gelu` for every
    model of the repository), or the one given."""

    def __init__(self, export_dir, activation=None, workspace_limit=None):
        super().__init__(HomogeneousExportStore(export_dir, activation), workspace_limit)


def synthetic_ffn(d_in, d_ff, d_out, density, seed, bias=True):
    """A deterministic random sparse FFN with an irregular support, for tests: the compact FFN (`OneBankStore`
    input) and its dense fixed-support matrices (a control, never used by the kernel).

    Every hidden unit and every output keeps at least one connection."""
    generator = torch.Generator().manual_seed(seed)
    rand = lambda *shape: torch.rand(*shape, generator=generator)
    up_mask, down_mask = rand(d_in, d_ff) < density, rand(d_ff, d_out) < density
    up_mask[torch.randint(d_in, (d_ff,), generator=generator), torch.arange(d_ff)] = True
    down_mask[torch.arange(d_ff), torch.randint(d_out, (d_ff,), generator=generator)] = True
    down_mask[torch.randint(d_ff, (d_out,), generator=generator), torch.arange(d_out)] = True
    w_up = torch.where(up_mask, (rand(d_in, d_ff)-0.5)*2/d_in**0.5, torch.zeros(()))
    w_down = torch.where(down_mask, (rand(d_ff, d_out)-0.5)*2/d_ff**0.5, torch.zeros(()))
    b_up = (rand(d_ff)-0.5)*0.2 if bias else torch.zeros(d_ff)
    b_down = (rand(d_out)-0.5)*0.2 if bias else torch.zeros(d_out)
    up_rows = up_mask.T.nonzero(as_tuple=True)[1]        # input indices, grouped by hidden unit
    down_cols = down_mask.nonzero(as_tuple=True)[1]      # output indices, grouped by hidden unit
    pointer = lambda counts: torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0))).int()
    ffn = {'up_ptr': pointer(up_mask.sum(0)), 'up_idx': up_rows, 'up_val': w_up.T[up_mask.T],
           'down_ptr': pointer(down_mask.sum(1)), 'down_idx': down_cols, 'down_val': w_down[down_mask],
           'b_up': b_up, 'b_down': b_down}
    dense = {'W_up_dense': w_up, 'W_down_dense': w_down, 'b_up': b_up, 'b_down': b_down}
    return ffn, dense
