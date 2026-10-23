"""Cold import of a padded legacy training checkpoint into the compact G4 runtime; no forward, gradient or update.

The padded legacy format stores the hidden neurons of a layer in the rows of [4, M, width] tensors: group g holds
n_hidden_group[g] of them in rows 0 .. n_hidden_group[g] - 1, hidden neurons numbered group after group (for the canonical
partition, hidden h at (h // M, h % M)), active slots first (degree tensors). Its runtime payload holds values and moments
only; the support lives in the padded export or in the model state (buffers). An import maps every active slot onto the
compact slot of the same hidden neuron, in the same order, ignores padded slots and rows, and loads the result through the
validated compact loaders (`compact_checkpoint`): nothing is written before the whole payload passed. Called at a completed
update boundary, never from the hot path.
"""
from __future__ import annotations

import torch

from . import compact_checkpoint as cc
from .padded_export import G4_GROUPS

RUNTIME_FORMAT = "hffn_padded_training_runtime_v2"
OPTIMIZER_FORMAT = "hffn_training_optimizer_v1"
RUNTIME_TOP = frozenset(("format", "steps", "adamw_all_calls", "layers", "adam"))
OPTIMIZER_TOP = frozenset(("format", "lr", "weight_decay", "base", "hffn_runtime"))
SUPPORT = ("n_hidden_group", "up_idx_pad", "up_deg", "down_idx_pad", "down_deg")
VALUES = ("up_val_pad", "down_val_pad", "b_up_pad", "b_down")
MOMENTS = ("up_m", "up_v", "down_m", "down_v", "b_up_m", "b_up_v", "b_down_m", "b_down_v")
# compact name <- (padded name, the side whose active slots it follows: "up", "down", "b_up" or "b_down")
_MAP = {"up_val": ("up_val_pad", "up"), "down_val": ("down_val_pad", "down"), "b_up": ("b_up_pad", "b_up"),
        "b_down": ("b_down", "b_down"), "um": ("up_m", "up"), "uv": ("up_v", "up"), "dm": ("down_m", "down"),
        "dv": ("down_v", "down"), "bum": ("b_up_m", "b_up"), "buv": ("b_up_v", "b_up"), "bdm": ("b_down_m", "b_down"),
        "bdv": ("b_down_v", "b_down")}


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"invalid padded checkpoint {name} fields")


def _tensor(value, dtype, dim, name):
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.layout != torch.strided
            or value.dtype != dtype or value.dim() != dim):
        raise ValueError(f"invalid padded checkpoint tensor {name}")
    return value


def _support(source, index):
    """The padded support of one layer, from a padded export layer or from the model-state buffers."""
    if isinstance(source, dict):
        _keys(source, SUPPORT, f"layer {index} support")
        return source
    try:
        return {name: getattr(source, name) for name in SUPPORT}
    except AttributeError:
        raise ValueError(f"invalid padded checkpoint layer {index} support") from None


def _layout(support, layer, index):
    """Active-slot masks of one padded layer, once its support is the compact layer's, hidden by hidden, in slot order."""
    support = _support(support, index)
    groups = _tensor(support["n_hidden_group"], torch.int32, 1, "n_hidden_group")
    up_deg, down_deg = (_tensor(support[n], torch.int32, 2, n) for n in ("up_deg", "down_deg"))
    up_idx, down_idx = (_tensor(support[n], torch.int16, 3, n) for n in ("up_idx_pad", "down_idx_pad"))
    if (groups.shape != (G4_GROUPS,) or up_deg.shape[0] != G4_GROUPS or down_deg.shape != up_deg.shape
            or up_idx.shape[:2] != up_deg.shape or down_idx.shape[:2] != up_deg.shape):
        raise ValueError(f"invalid padded checkpoint layer {index} shapes")
    rows, d_ff = up_deg.shape[1], layer.d_ff
    if bool((groups < 0).any()) or bool((groups > rows).any()) or int(groups.sum()) != d_ff:
        raise ValueError(f"invalid padded checkpoint layer {index} hidden partition")
    hidden = torch.arange(rows).view(1, -1) < groups.view(-1, 1)  # [4, M]: the rows that hold a hidden neuron, group-major
    layout = {"hidden": hidden}
    for side, degrees, indices in (("up", up_deg, up_idx), ("down", down_deg, down_idx)):
        offsets = getattr(layer, f"{side}_row_offsets")
        degree, width = degrees[hidden], indices.shape[2]
        if not torch.equal(degree, offsets[1:] - offsets[:-1]) or (d_ff and int(degree.max()) > width):
            raise ValueError(f"padded checkpoint layer {index} support mismatch: {side} degrees")
        mask = torch.arange(width).view(1, -1) < degree.view(-1, 1)
        if not torch.equal(indices[hidden][mask], getattr(layer, f"{side}_idx")):
            raise ValueError(f"padded checkpoint layer {index} support mismatch: {side} indices")
        layout[side] = (mask, tuple(indices.shape))
    return layout


def _compact(tensors, names, layout):
    """Compact tensors of padded ones: active slots (weights), the hidden rows (b_up) or the tensor itself (b_down)."""
    out = {}
    for name in names:
        source, side = _MAP[name]
        if side in ("up", "down"):
            mask, shape = layout[side]
            value = _tensor(tensors[source], torch.float32, 3, source)
            if tuple(value.shape) != shape:
                raise ValueError(f"invalid padded checkpoint tensor {source}")
            out[name] = value[layout["hidden"]][mask]
        elif side == "b_up":
            value = _tensor(tensors[source], torch.float32, 2, source)
            if value.shape != layout["hidden"].shape:
                raise ValueError(f"invalid padded checkpoint tensor {source}")
            out[name] = value[layout["hidden"]]
        else:
            out[name] = _tensor(tensors[source], torch.float32, 1, source).contiguous()
    return out


def _runtime_payload(runtime, payload, layouts):
    """The compact runtime payload of a padded runtime payload; `layouts` holds the checked layout of each layer."""
    _keys(payload, RUNTIME_TOP, "runtime top-level")
    if type(payload["format"]) is not str or payload["format"] != RUNTIME_FORMAT:
        raise ValueError("unsupported padded runtime checkpoint format")
    for name in ("layers", "adam", "steps"):
        if type(payload[name]) is not list or len(payload[name]) != len(runtime.layers):
            raise ValueError(f"invalid padded checkpoint {name} count")
    layers, adam = [], []
    for index, (layer, layout, values, moments) in enumerate(zip(runtime.layers, layouts, payload["layers"], payload["adam"])):
        _keys(values, VALUES, f"layer {index}")
        _keys(moments, MOMENTS, f"moments {index}")
        layers.append({"d_model": layer.d_model, "d_ff": layer.d_ff, **{n: getattr(layer, n) for n in cc.SUPPORT},
                       **_compact(values, cc.VALUES, layout)})
        adam.append(_compact(moments, cc.MOMENTS, layout))
    return {"format": cc.FORMAT, "dual_route_min_n": runtime.dual_route_min_n, "steps": payload["steps"],
            "adamw_all_calls": payload["adamw_all_calls"], "layers": layers, "adam": adam}


def _layouts(runtime, supports):
    if not isinstance(supports, (list, tuple)) or len(supports) != len(runtime.layers):
        raise ValueError("invalid padded checkpoint support count")
    return [_layout(support, layer, index) for index, (layer, support) in enumerate(zip(runtime.layers, supports))]


def import_padded_runtime_state(runtime, payload, supports):
    """Load a padded runtime payload (`hffn_padded_training_runtime_v2`) into a compact runtime of the same support.

    `supports` gives one padded layer per runtime layer (a `load_hffn_padded_training_layer` result or a dict of its five
    support tensors). The compact loader validates and stages the whole converted payload before its first write.
    """
    cc.load_state_dict(runtime, _runtime_payload(runtime, payload, _layouts(runtime, supports)))


def import_padded_training_checkpoint(model, optimizer, model_state, optimizer_state):
    """Load a padded whole-model checkpoint (`model.state_dict()`, `hffn_training_optimizer_v1`) into a compact model.

    `model` holds the compact MLPs of `optimizer.runtime`, built from an export of the same support. The non-FFN model
    tensors are validated and staged, the padded FFN values of the model state must equal those of the runtime payload
    bit for bit, and the compact optimizer loader validates the rest before its first write; the non-FFN model tensors
    are copied last.
    """
    runtime = optimizer.runtime
    blocks = list(getattr(getattr(model, "gpt_neox", None), "layers", ()))
    if len(blocks) != len(runtime.layers) or any(
            getattr(block.mlp, name, None) is not getattr(layer, name)
            for block, layer in zip(blocks, runtime.layers) for name in cc.VALUES):
        raise ValueError("padded checkpoint import needs the compact model of this optimizer")
    _keys(optimizer_state, OPTIMIZER_TOP, "optimizer top-level")
    if type(optimizer_state["format"]) is not str or optimizer_state["format"] != OPTIMIZER_FORMAT:
        raise ValueError("unsupported padded optimizer checkpoint format")
    destination = model.state_dict()
    prefixes = [f"gpt_neox.layers.{i}.mlp." for i in range(len(blocks))]
    compact_ffn = {p + n for p in prefixes for n in (*cc.SUPPORT, *cc.VALUES)}
    if not compact_ffn <= set(destination) or any(k.startswith(p) for k in set(destination) - compact_ffn for p in prefixes):
        raise ValueError("unexpected compact model state")
    dense = [k for k in destination if k not in compact_ffn]
    if not isinstance(model_state, dict) or set(model_state) != {*dense, *(p + n for p in prefixes for n in (*SUPPORT, *VALUES))}:
        raise ValueError("padded checkpoint model state keys mismatch")
    staged = {}
    for key in dense:
        source, target = model_state[key], destination[key]
        if (not isinstance(source, torch.Tensor) or source.device.type != "cpu" or source.layout != torch.strided
                or source.dtype != target.dtype or source.shape != target.shape):
            raise ValueError(f"invalid padded checkpoint model tensor {key}")
        staged[key] = source.detach().clone()
    layouts = _layouts(runtime, [{n: model_state[p + n] for n in SUPPORT} for p in prefixes])
    hffn = _runtime_payload(runtime, optimizer_state["hffn_runtime"], layouts)
    for index, (layout, prefix) in enumerate(zip(layouts, prefixes)):
        mine = _compact({n: model_state[prefix + n] for n in VALUES}, cc.VALUES, layout)
        for name in cc.VALUES:
            theirs = hffn["layers"][index][name]
            if mine[name].shape != theirs.shape or not torch.equal(mine[name].view(torch.int32), theirs.view(torch.int32)):
                raise ValueError(f"padded checkpoint layer {index} model and runtime values differ: {name}")
    cc.load_optimizer_state_dict(optimizer, {"format": cc.OPTIMIZER_FORMAT, "lr": optimizer_state["lr"],
                                             "weight_decay": optimizer_state["weight_decay"],
                                             "base": optimizer_state["base"], "hffn_runtime": hffn})
    with torch.no_grad():
        for key, value in staged.items():
            destination[key].copy_(value)
