"""Cold, compact checkpoint interoperability; never called by forward/backward.

``export_checkpoint(module)`` preserves support and actual G4 partition.
``from_checkpoint(payload)`` and ``from_legacy_state_dict(state)`` return a
fresh Layer for HFFNTrainFFN. ``copy_parameters(module, payload_or_legacy)``
loads values into existing storage only when the complete support matches.
The ordinary module state_dict remains parameter-only; use this explicit API
when a checkpoint must reconstruct its support. Optimizer state is separate.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch

from .layer import GROUPS, Layer, _reverse, balanced_groups

FORMAT = 'hffn_ffn_training_checkpoint_v1'
MAX_DIMENSION = 32767
PARAMETERS = ('up_val', 'down_val', 'b_up', 'b_down')
SUPPORT = {'up_ptr': 'up_offsets', 'up_idx': 'up_idx',
           'down_ptr': 'down_offsets', 'down_idx': 'down_idx'}
STATE_KEYS = frozenset((*PARAMETERS, *SUPPORT))
INTEGERS = (torch.int16, torch.int32, torch.int64)


def _tensor(value, key, dtype=None):
    if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.layout != torch.strided:
        raise TypeError(f'{key}: expected a strided CPU tensor')
    if value.ndim != 1:
        raise ValueError(f'{key}: expected one dimension')
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f'{key}: expected {dtype}')
    return value


def _state(raw):
    if not isinstance(raw, Mapping) or set(raw) != STATE_KEYS:
        raise ValueError('legacy state must contain exactly four parameters and four support arrays')
    for key in PARAMETERS:
        value = _tensor(raw[key], key, torch.float32)
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'{key}: nonfinite values')
    d_model, d_ff = raw['b_down'].numel(), raw['b_up'].numel()
    if not (0 < d_model <= MAX_DIMENSION and 0 < d_ff <= MAX_DIMENSION):
        raise ValueError('v1 requires 1 <= d_model,d_ff <= 32767')
    for side in ('up', 'down'):
        offsets = _tensor(raw[side + '_ptr'], side + '_ptr')
        index = _tensor(raw[side + '_idx'], side + '_idx')
        if offsets.dtype not in INTEGERS or index.dtype not in INTEGERS:
            raise TypeError('support arrays must have int16/int32/int64 dtype')
        if offsets.numel() != d_ff + 1 or index.numel() != raw[side + '_val'].numel():
            raise ValueError(side + ': support/parameter shape mismatch')
        # Every range check precedes the first narrowing conversion.
        if int(offsets.min()) < 0 or int(offsets.max()) > torch.iinfo(torch.int32).max:
            raise ValueError(side + ': offsets exceed the int32 domain')
        if int(offsets[0]) != 0 or int(offsets[-1]) != index.numel() or bool((offsets[1:] < offsets[:-1]).any()):
            raise ValueError(side + ': invalid offsets/endpoints')
        if index.numel() and (int(index.min()) < 0 or int(index.max()) >= d_model):
            raise ValueError(side + ': index outside the feature domain')
    return raw, d_model, d_ff


def _groups(value, d_ff):
    groups = _tensor(value, 'groups')
    if groups.dtype not in INTEGERS or groups.numel() != GROUPS + 1:
        raise ValueError('groups: expected five integer boundaries')
    if int(groups[0]) != 0 or int(groups[-1]) != d_ff or bool((groups[1:] < groups[:-1]).any()):
        raise ValueError('groups: invalid G4 partition')
    return groups


def _payload(payload):
    if not isinstance(payload, Mapping) or set(payload) != {'format', 'd_model', 'd_ff', 'groups', 'state_dict'}:
        raise ValueError('invalid checkpoint key set')
    if payload['format'] != FORMAT:
        raise ValueError('unsupported checkpoint format')
    raw, d_model, d_ff = _state(payload['state_dict'])
    if type(payload['d_model']) is not int or type(payload['d_ff']) is not int or \
            (payload['d_model'], payload['d_ff']) != (d_model, d_ff):
        raise ValueError('checkpoint dimensions disagree with its state')
    return raw, d_model, d_ff, _groups(payload['groups'], d_ff)


def _owned(value, dtype):
    return value.detach().contiguous().to(dtype=dtype, copy=True)


def _layer(raw, d_model, d_ff, groups, name):
    up_offsets = _owned(raw['up_ptr'], torch.int32)
    down_offsets = _owned(raw['down_ptr'], torch.int32)
    up_idx = _owned(raw['up_idx'], torch.int16)
    down_idx = _owned(raw['down_idx'], torch.int16)
    if groups is None:
        groups = balanced_groups(up_offsets.diff().long() + down_offsets.diff().long())
    else:
        groups = _owned(groups, torch.int32)
    down = _reverse(down_offsets, down_idx, d_ff, d_model)
    up = _reverse(up_offsets, up_idx, d_ff, d_model)
    return Layer(name, d_model, d_ff, groups, up_offsets, up_idx,
                 _owned(raw['up_val'], torch.float32), _owned(raw['b_up'], torch.float32),
                 down_offsets, down_idx, _owned(raw['down_val'], torch.float32),
                 _owned(raw['b_down'], torch.float32), *down, *up)


def from_legacy_state_dict(state, *, name='legacy-checkpoint') -> Layer:
    """Reconstruct legacy eight-tensor state, with a deterministic G4 partition."""
    raw, d_model, d_ff = _state(state)
    return _layer(raw, d_model, d_ff, None, name)


def from_checkpoint(payload, *, name='checkpoint') -> Layer:
    """Reconstruct a versioned state, preserving its actual reduction partition."""
    raw, d_model, d_ff, groups = _payload(payload)
    return _layer(raw, d_model, d_ff, groups, name)


def export_checkpoint(module):
    """Owned CPU tensors, including support; no live parameter/storage aliases."""
    module.check_storage()
    layer = module.step.layer
    original = {key: getattr(module, key) for key in PARAMETERS}
    original.update({key: getattr(layer, field) for key, field in SUPPORT.items()})
    _, d_model, d_ff = _state(original)
    if type(layer.d_model) is not int or type(layer.d_ff) is not int or \
            (layer.d_model, layer.d_ff) != (d_model, d_ff):
        raise ValueError('module dimensions disagree with its state')
    _groups(layer.groups, d_ff)
    state = {key: _owned(original[key], torch.float32) for key in PARAMETERS}
    state.update({key: _owned(original[key], torch.int64) for key in SUPPORT})
    payload = dict(format=FORMAT, d_model=layer.d_model, d_ff=layer.d_ff,
                   groups=_owned(layer.groups, torch.int32), state_dict=state)
    _payload(payload)
    return payload


def copy_parameters(module, checkpoint_or_legacy):
    """Copy validated values only. Metadata never changes, Parameters never rebind.

    A different support/partition must be loaded by constructing a fresh module.
    Optimizer moments are neither loaded nor reset by this parameter-only API.
    Parameter version counters advance so a pending backward fails closed.
    """
    module.check_storage()
    if isinstance(checkpoint_or_legacy, Mapping) and set(checkpoint_or_legacy) == STATE_KEYS:
        raw, d_model, d_ff = _state(checkpoint_or_legacy)
        groups = None
    else:
        raw, d_model, d_ff, groups = _payload(checkpoint_or_legacy)
    layer = module.step.layer
    if (d_model, d_ff) != (layer.d_model, layer.d_ff):
        raise ValueError('different dimensions: construct a fresh module')
    for key, field in SUPPORT.items():
        if not torch.equal(raw[key], getattr(layer, field)):
            raise ValueError('different support: construct a fresh module')
    if groups is not None and not torch.equal(groups, layer.groups):
        raise ValueError('different G4 partition: construct a fresh module')
    # Sources may be views of the destination (including a parameter swap).
    # Finish every allocation/read before writing any possibly overlapping value.
    staged = {key: _owned(raw[key], torch.float32) for key in PARAMETERS}
    with torch.no_grad():
        for key in PARAMETERS:
            getattr(module, key).copy_(staged[key])
