"""Hidden-major compact banks (`e1b_hidden_banks_v1`): validation and device placement.

Each bank file holds, per hidden unit, its active up connections (offsets,
feature indices, values), its up bias, and its active down connections. On the
device a bank keeps the hidden-major up support and the same down values in
output-owner order (a stable permutation, not a dense reconstruction).

Indices are 16-bit, and d_model, d_out and every bank width then stay below 32768; a store whose FFN is wider
asks for 32-bit indices (`index_dtype`). A bank file may give `d_out`, the width of the output, when it differs
from `d_model`, the width of the input.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

FORMAT = 'e1b_hidden_banks_v1'
INDEX_LIMIT = 32768        # int16 feature, output and hidden indices
WIDE_LIMIT = 2**31         # int32 indices


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def validate(raw, d_model, index_dtype=torch.int16):
    d_out = raw.get('d_out', d_model)
    limit = {torch.int16: INDEX_LIMIT, torch.int32: WIDE_LIMIT}[index_dtype]
    if raw['d_model'] != d_model or not all(0 < size < limit for size in (d_model, d_out, raw['d_ff'])):
        raise ValueError('unsupported bank dimensions')
    for side, limit in (('up', d_model), ('down', d_out)):
        offsets, index, values = raw[side+'_row_offsets'], raw[side+'_idx'], raw[side+'_val']
        if offsets.dtype != torch.int32 or offsets.shape != (raw['d_ff']+1,):
            raise ValueError('invalid hidden-major offsets')
        if index.dtype != index_dtype or values.dtype != torch.float32 or index.shape != values.shape:
            raise ValueError('invalid compact indices/values')
        if int(offsets[0]) != 0 or int(offsets[-1]) != len(values) or (offsets[1:] < offsets[:-1]).any():
            raise ValueError('invalid support extent')
        if len(index) and not 0 <= int(index.min()) <= int(index.max()) < limit:
            raise ValueError('invalid feature index')
        if not torch.isfinite(values).all():
            raise ValueError('non-finite weights')
    bias = raw['b_up']
    if bias.dtype != torch.float32 or bias.shape != (raw['d_ff'],) or not torch.isfinite(bias).all():
        raise ValueError('invalid hidden bias')


@dataclass(frozen=True)
class Bank:
    width: int
    up_offsets: torch.Tensor     # int32 [width+1]
    up_index: torch.Tensor       # int16 or int32 [up edges]
    up_weights: torch.Tensor     # f32 [width + up edges]: hidden bias, then values
    down_offsets: torch.Tensor   # int32 [d_out+1], output-owner order
    down_hidden: torch.Tensor    # int16 or int32 [down edges], hidden unit of each edge
    down_values: torch.Tensor    # f32 [down edges]


def device_bank(raw):
    """Device placement of one validated bank: hidden-major up, down values in output-owner order."""
    d_out = raw.get('d_out', raw['d_model'])     # the down side is owned by the outputs
    degree = (raw['down_row_offsets'][1:]-raw['down_row_offsets'][:-1]).long()
    hidden = torch.repeat_interleave(torch.arange(raw['d_ff']), degree)
    order = torch.argsort(raw['down_idx'].long(), stable=True)
    offsets = torch.zeros(d_out+1, dtype=torch.int32)
    offsets[1:] = torch.cumsum(torch.bincount(raw['down_idx'].long(), minlength=d_out), 0)
    return Bank(width=raw['d_ff'],
                up_offsets=raw['up_row_offsets'].to('mps'), up_index=raw['up_idx'].to('mps'),
                up_weights=torch.cat((raw['b_up'], raw['up_val'])).to('mps'),
                down_offsets=offsets.to('mps'), down_hidden=hidden[order].to(raw['up_idx'].dtype).to('mps'),
                down_values=raw['down_val'][order].to('mps'))


class BankStore:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        manifest = json.loads((self.directory/'manifest.json').read_text())
        if manifest['format'] != FORMAT:
            raise ValueError('unsupported bank package')
        self.d_model = manifest['model_dim']
        self.layers, self.pairs = manifest['layers'], tuple(map(tuple, manifest['pairs']))
        self.banks_per_layer = len(manifest['banks'])//self.layers
        expected = [(l, b) for l in range(self.layers) for b in range(self.banks_per_layer)]
        if [(e['layer'], e['bank']) for e in manifest['banks']] != expected:
            raise ValueError('invalid bank inventory')
        self.manifest = manifest
        self.widths = [[manifest['banks'][l*self.banks_per_layer+b]['width']
                        for b in range(self.banks_per_layer)] for l in range(self.layers)]

    def _read(self, entry):
        path = (self.directory/entry['file']).resolve()
        if path.parent != self.directory or sha256(path) != entry['sha256']:
            raise ValueError('invalid package path or checksum: '+entry['file'])
        return torch.load(path, map_location='cpu', weights_only=True)

    def backbone(self):
        return self._read(self.manifest['backbone'])

    def load(self, layer, bank):
        entry = self.manifest['banks'][layer*self.banks_per_layer+bank]
        raw = self._read(entry)
        validate(raw, self.d_model)
        if (raw['d_ff'], raw['up_val'].numel(), raw['down_val'].numel()) != (
                entry['width'], entry['up_edges'], entry['down_edges']):
            raise ValueError('bank metadata mismatch')
        return device_bank(raw)
