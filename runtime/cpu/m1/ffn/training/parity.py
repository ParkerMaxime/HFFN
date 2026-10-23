from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from runtime.cpu.m1.ffn.training.models import HFFNCompactFFN, SparseMaskedReferenceFFN


def build_optimizer(name: str, params, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def _clone_hffn(hffn, reference, n):
    from runtime.cpu.m1.ffn.training.shared_toy import HFFNToyFFN
    if isinstance(hffn, HFFNToyFFN):
        return hffn.clone_reserved(max(n, 1))
    state = {k: v.detach().clone() for k, v in hffn.state_dict().items()}
    clone = HFFNCompactFFN(reference)
    clone.load_state_dict(state)
    return clone


def active_dense_to_compact(weight_tensor: torch.Tensor, ptr: torch.Tensor, idx: torch.Tensor, axis: str) -> torch.Tensor:
    pieces: List[torch.Tensor] = []
    hidden_count = int(ptr.numel() - 1)
    for hidden in range(hidden_count):
        start = int(ptr[hidden].item())
        end = int(ptr[hidden + 1].item())
        if end <= start:
            continue
        active_idx = idx[start:end].long()
        if axis == "up":
            pieces.append(weight_tensor[hidden, active_idx])
        elif axis == "down":
            pieces.append(weight_tensor[active_idx, hidden])
        else:
            raise ValueError(f"Unsupported axis: {axis}")
    if not pieces:
        return torch.empty(0, dtype=weight_tensor.dtype)
    return torch.cat(pieces, dim=0)


def parity_snapshot(
    reference: SparseMaskedReferenceFFN,
    hffn: HFFNCompactFFN,
    x: torch.Tensor,
    loss_target: torch.Tensor,
) -> Dict[str, float]:
    ref_state = {k: v.detach().clone() for k, v in reference.state_dict().items()}

    ref_clone = SparseMaskedReferenceFFN(reference.up.in_features, reference.up.out_features, 1.0, torch.Generator())
    ref_clone.load_state_dict(ref_state)
    hffn_clone = _clone_hffn(hffn, ref_clone, x.shape[0])

    x_ref = x.detach().clone().requires_grad_(True)
    x_hffn = x.detach().clone().requires_grad_(True)
    target_ref = loss_target.detach().clone()
    target_hffn = loss_target.detach().clone()

    ref_out = ref_clone(x_ref)
    hffn_out = hffn_clone(x_hffn)
    ref_loss = F.mse_loss(ref_out, target_ref)
    hffn_loss = F.mse_loss(hffn_out, target_hffn)
    ref_loss.backward()
    hffn_loss.backward()

    ref_up_grad = active_dense_to_compact(ref_clone.up.weight.grad, hffn_clone.up_ptr, hffn_clone.up_idx, "up")
    ref_down_grad = active_dense_to_compact(ref_clone.down.weight.grad, hffn_clone.down_ptr, hffn_clone.down_idx, "down")

    return {
        "forward_max_abs": float((ref_out - hffn_out).abs().max().item()),
        "loss_abs": float(abs(ref_loss.item() - hffn_loss.item())),
        "grad_x_max_abs": float((x_ref.grad - x_hffn.grad).abs().max().item()),
        "grad_up_val_max_abs": float((ref_up_grad - hffn_clone.up_val.grad).abs().max().item()),
        "grad_down_val_max_abs": float((ref_down_grad - hffn_clone.down_val.grad).abs().max().item()),
        "grad_b_up_max_abs": float((ref_clone.up.bias.grad - hffn_clone.b_up.grad).abs().max().item()),
        "grad_b_down_max_abs": float((ref_clone.down.bias.grad - hffn_clone.b_down.grad).abs().max().item()),
    }


def optimizer_step_parity(
    reference: SparseMaskedReferenceFFN,
    hffn: HFFNCompactFFN,
    x: torch.Tensor,
    target: torch.Tensor,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
) -> Dict[str, float]:
    ref_state = {k: v.detach().clone() for k, v in reference.state_dict().items()}

    ref_clone = SparseMaskedReferenceFFN(reference.up.in_features, reference.up.out_features, 1.0, torch.Generator())
    ref_clone.load_state_dict(ref_state)
    hffn_clone = _clone_hffn(hffn, ref_clone, x.shape[0])

    ref_opt = build_optimizer(optimizer_name, ref_clone.parameters(), lr, weight_decay)
    hffn_opt = build_optimizer(optimizer_name, hffn_clone.parameters(), lr, weight_decay)

    ref_opt.zero_grad(set_to_none=True)
    hffn_opt.zero_grad(set_to_none=True)

    ref_loss = F.mse_loss(ref_clone(x), target)
    hffn_loss = F.mse_loss(hffn_clone(x), target)
    ref_loss.backward()
    hffn_loss.backward()
    ref_opt.step()
    ref_clone.apply_masks_()
    hffn_opt.step()

    ref_up = active_dense_to_compact(ref_clone.up.weight.detach(), hffn_clone.up_ptr, hffn_clone.up_idx, "up")
    ref_down = active_dense_to_compact(ref_clone.down.weight.detach(), hffn_clone.down_ptr, hffn_clone.down_idx, "down")

    return {
        "step_up_val_max_abs": float((ref_up - hffn_clone.up_val.detach()).abs().max().item()),
        "step_down_val_max_abs": float((ref_down - hffn_clone.down_val.detach()).abs().max().item()),
        "step_b_up_max_abs": float((ref_clone.up.bias.detach() - hffn_clone.b_up.detach()).abs().max().item()),
        "step_b_down_max_abs": float((ref_clone.down.bias.detach() - hffn_clone.b_down.detach()).abs().max().item()),
        "step_loss_abs": float(abs(ref_loss.item() - hffn_loss.item())),
    }
