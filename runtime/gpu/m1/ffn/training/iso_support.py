"""Mask of the fixed-support dense control of the iso-support scripts of this leaf. The projection is not here: it is
``FixedSupportAdamW`` of ``common/pythia/hffn_pythia/dense_control.py``, the one definition of the dense control.

The control keeps dense parameters, gradients and AdamW moments; the support is enforced by one projection, of the
gradient before the step (``FixedSupportAdamW.step``). Nothing else needs it: an inactive entry starts at zero, its masked gradient keeps both
moments at zero, so AdamW leaves it at zero (decoupled decay of zero and an update of zero over ``eps``). A parameter
that does not start on its support is the caller's error, not repaired here. Biases are dense and not masked.
"""
import torch


def synthetic_support_masks(up: torch.Tensor, down: torch.Tensor, degree: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The 0/1 masks of the synthetic support (``degree`` slots per hidden neuron), shaped and placed as ``up`` and ``down``."""
    hidden, input_size = up.shape
    output_size = down.shape[0]
    up_mask = torch.zeros_like(up)
    down_mask = torch.zeros_like(down)
    rows = torch.arange(hidden, device=up.device)[:, None]
    slots = torch.arange(degree, device=up.device)[None, :]
    up_cols = (rows * 19 + slots * 7) % input_size
    down_rows = (rows * 17 + slots * 5) % output_size
    up_mask.scatter_(1, up_cols, 1.0)
    down_mask[down_rows, rows.expand(-1, degree)] = 1.0
    return up_mask, down_mask
