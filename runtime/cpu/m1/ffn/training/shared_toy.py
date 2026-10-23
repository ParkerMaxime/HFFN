"""Cold toy compatibility facade on the shared inference-owned training module.

Forward/backward are inherited unchanged. Dense masked weights belong to the
reference; construction only extracts their active values once.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch

from .hffn_ffn_training.checkpoint import (
    MAX_DIMENSION, export_checkpoint, from_checkpoint, from_legacy_state_dict,
)
from .hffn_ffn_training.layer import Layer
from .hffn_ffn_training.module import HFFNTrainFFN
from .hffn_ffn_training.step import LR, WD, _validate_adamw_config
from .models import SparseMaskedReferenceFFN, build_hidden_major


class HFFNToyFFN(HFFNTrainFFN):
    """Toy dimensions/support aliases; one shared Step reserved before calls."""

    def __init__(self, source, n, *, kernels=None, lr=LR, weight_decay=WD):
        lr, weight_decay = _validate_adamw_config(lr, weight_decay)
        if isinstance(source, SparseMaskedReferenceFFN):
            d_model, d_ff = source.up.in_features, source.up.out_features
            if not (0 < d_model <= MAX_DIMENSION and 0 < d_ff <= MAX_DIMENSION):
                raise ValueError('v1 requires 1 <= d_model,d_ff <= 32767')
            for mask in (source.up.mask, source.down.mask):
                if mask.device.type != 'cpu' or mask.layout != torch.strided:
                    raise TypeError('reference masks must be strided CPU tensors')
                for row in mask:
                    low, high = torch.aminmax(row)
                    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or float(low) < 0:
                        raise ValueError('reference masks must be finite and nonnegative')
            state = build_hidden_major(source.up.weight.detach(), source.up.mask.detach(),
                                       source.down.weight.detach(), source.down.mask.detach())
            for side, mask in (('up', source.up.mask), ('down', source.down.mask)):
                owner = torch.repeat_interleave(torch.arange(d_ff), state[side + '_ptr'].diff())
                index = state[side + '_idx']
                factors = mask[owner, index] if side == 'up' else mask[index, owner]
                state[side + '_val'] = (state[side + '_val'] * factors).float()
            state.update(b_up=source.up.bias.detach().float(), b_down=source.down.bias.detach().float())
            source = from_legacy_state_dict(state, name='toy')
        if not isinstance(source, Layer):
            raise TypeError('expected a masked reference or compact Layer')
        if type(source.d_model) is not int or type(source.d_ff) is not int or not \
                (0 < source.d_model <= MAX_DIMENSION and 0 < source.d_ff <= MAX_DIMENSION):
            raise ValueError('v1 requires 1 <= d_model,d_ff <= 32767')
        super().__init__(source, n, n_min=1, kernels=kernels, lr=lr, weight_decay=weight_decay)

    @property
    def d_model(self): return self.step.layer.d_model

    @property
    def d_ff(self): return self.step.layer.d_ff

    @property
    def up_ptr(self): return self.step.layer.up_offsets

    @property
    def up_idx(self): return self.step.layer.up_idx

    @property
    def down_ptr(self): return self.step.layer.down_offsets

    @property
    def down_idx(self): return self.step.layer.down_idx

    def clone_reserved(self, n):
        """Cold parity clone preserving actual compact support and G4 groups."""
        return type(self)(from_checkpoint(export_checkpoint(self), name='toy-parity'), n,
                          kernels=(self.step.training, self.step.inference),
                          lr=self.step.lr, weight_decay=self.step.weight_decay)


def load_toy_checkpoint(payload, n, *, kernels=None):
    """Load an old/new complete toy payload; optimizer state stays separate."""
    if not isinstance(payload, Mapping) or not {'state_dict', 'architecture', 'config', 'extra'} <= set(payload):
        raise ValueError('expected a complete toy checkpoint')
    architecture = payload['architecture']
    if not isinstance(architecture, Mapping) or architecture.get('activation') != 'gelu_exact' or \
            architecture.get('layout') != 'hidden_major_compact':
        raise ValueError('unsupported toy activation/layout')
    if 'hffn_runtime' in payload:
        metadata = payload['hffn_runtime']
        if not isinstance(metadata, Mapping) or 'state_dict' in metadata:
            raise ValueError('invalid toy runtime metadata')
        layer = from_checkpoint({**metadata, 'state_dict': payload['state_dict']}, name='toy-checkpoint')
    else:
        layer = from_legacy_state_dict(payload['state_dict'], name='legacy-toy-checkpoint')
    if type(architecture.get('d_model')) is not int or type(architecture.get('d_ff')) is not int or \
            (architecture['d_model'], architecture['d_ff']) != (layer.d_model, layer.d_ff):
        raise ValueError('toy architecture dimensions disagree with its state')
    config = payload['config']
    if not isinstance(config, Mapping) or type(config.get('d_model')) is not int or type(config.get('d_ff')) is not int or \
            (config['d_model'], config['d_ff']) != (layer.d_model, layer.d_ff):
        raise ValueError('toy config dimensions disagree with its state')
    return HFFNToyFFN(layer, n, kernels=kernels)
