from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from runtime.cpu.m1.ffn.training.models import HFFNCompactFFN, SparseMaskedReferenceFFN


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_rel_l2_num = 0.0
    total_rel_l2_den = 0.0
    n = x.shape[0]

    for start in range(0, n, batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        pred = model(xb)
        loss = F.mse_loss(pred, yb, reduction="sum")
        total_loss += float(loss.item())
        diff = pred - yb
        total_rel_l2_num += float((diff * diff).sum().item())
        total_rel_l2_den += float((yb * yb).sum().item())

    mse = total_loss / n
    rel_l2 = math.sqrt(total_rel_l2_num / max(total_rel_l2_den, 1e-30))
    return {"mse": mse, "relative_l2": rel_l2}


def train_one_epoch(
    reference: SparseMaskedReferenceFFN,
    hffn: HFFNCompactFFN,
    ref_opt: torch.optim.Optimizer,
    hffn_opt: torch.optim.Optimizer,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    batch_size: int,
    shuffle_generator: torch.Generator,
) -> Dict[str, float]:
    reference.train()
    hffn.train()
    n = x_train.shape[0]
    perm = torch.randperm(n, generator=shuffle_generator)
    total_ref_loss = 0.0
    total_hffn_loss = 0.0
    total_step_gap = 0.0
    total_samples = 0

    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        xb = x_train[idx]
        yb = y_train[idx]

        ref_opt.zero_grad(set_to_none=True)
        hffn_opt.zero_grad(set_to_none=True)

        ref_pred = reference(xb)
        hffn_pred = hffn(xb)
        ref_loss = F.mse_loss(ref_pred, yb)
        hffn_loss = F.mse_loss(hffn_pred, yb)

        ref_loss.backward()
        hffn_loss.backward()
        ref_opt.step()
        reference.apply_masks_()
        hffn_opt.step()

        total_ref_loss += float(ref_loss.item()) * xb.shape[0]
        total_hffn_loss += float(hffn_loss.item()) * xb.shape[0]
        total_step_gap += float((ref_pred.detach() - hffn_pred.detach()).abs().max().item())
        total_samples += int(xb.shape[0])

    step_count = max(n // batch_size + int(n % batch_size > 0), 1)
    return {
        "reference_train_mse": total_ref_loss / n,
        "hffn_train_mse": total_hffn_loss / n,
        "mean_step_forward_gap": total_step_gap / step_count,
        "train_samples": float(total_samples),
    }
