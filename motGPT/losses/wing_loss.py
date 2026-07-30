"""
Wing Loss (Feng et al., CVPR 2018)

A loss function where gradient magnitude INCREASES as error approaches 0,
unlike MSE whose gradient vanishes near zero.

    L(e) = w * ln(1 + |e| / eps)    if |e| < w
         = |e| - C                    otherwise

where C = w - w * ln(1 + w / eps) ensures continuity at |e| = w.

Gradient magnitude:
    |e| < w  :  w / (eps + |e|)   →  at e=0, grad = w/eps (maximum)
    |e| >= w :  1                  (L1 behavior, constant)
"""

import torch
import torch.nn.functional as F


def wing_loss(pred: torch.Tensor, target: torch.Tensor,
              w: float = 10.0, eps: float = 2.0) -> torch.Tensor:
    """Element-wise Wing Loss, returns mean-reduced scalar."""
    diff = pred - target
    abs_diff = diff.abs()
    C = w - w * torch.log(torch.tensor(1.0 + w / eps, device=pred.device))
    loss = torch.where(
        abs_diff < w,
        w * torch.log(1.0 + abs_diff / eps),
        abs_diff - C,
    )
    return loss.mean()


def masked_wing_loss(pred: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor,
                     w: float = 10.0, eps: float = 2.0) -> torch.Tensor:
    """
    Wing Loss with padding mask support.

    Args:
        pred:   (B, T, D) or (B, T)
        target: same shape as pred
        mask:   broadcastable to pred (e.g. (B, T, 1) or (B, T, D))
        w, eps: Wing Loss hyperparameters

    Returns:
        Scalar loss averaged over valid (masked) elements.
    """
    diff = pred - target
    abs_diff = diff.abs()
    C = w - w * torch.log(torch.tensor(1.0 + w / eps, device=pred.device))
    elem_loss = torch.where(
        abs_diff < w,
        w * torch.log(1.0 + abs_diff / eps),
        abs_diff - C,
    )
    masked_loss = elem_loss * mask
    # Normalize by the number of valid elements, not just valid time steps.
    # When mask has shape (B, T, 1) and pred has shape (B, T, D), divide by
    # B*T*D-valid elements so the scalar stays on a per-element scale.
    n_valid = mask.expand_as(pred).sum().clamp(min=1e-8)
    return masked_loss.sum() / n_valid


def sample_masked_wing_loss(pred: torch.Tensor, target: torch.Tensor,
                            mask: torch.Tensor,
                            w: float = 10.0, eps: float = 2.0) -> torch.Tensor:
    """Wing Loss with per-sample reduction, then batch averaging.

    Matches Stage 2's ``_compute_motion_losses`` convention: each sample's
    loss is averaged over its own valid elements, and the final scalar is
    the mean over samples with at least one valid frame. This gives every
    sample equal weight regardless of motion length, as opposed to
    ``masked_wing_loss`` which weights long sequences more heavily.

    Args:
        pred:   (B, T, D) or (B, T)
        target: same shape as pred
        mask:   broadcastable to pred (e.g. (B, T) or (B, T, 1) or (B, T, D))
        w, eps: Wing Loss hyperparameters

    Returns:
        Scalar loss: mean across valid samples of per-sample element means.
    """
    diff = pred - target
    abs_diff = diff.abs()
    C = w - w * torch.log(torch.tensor(1.0 + w / eps, device=pred.device))
    elem_loss = torch.where(
        abs_diff < w,
        w * torch.log(1.0 + abs_diff / eps),
        abs_diff - C,
    )
    mask_b = mask.to(elem_loss.dtype).expand_as(elem_loss)
    # Per-sample normalized loss: sum over (T, D) / valid elements per sample.
    per_sample_numer = (elem_loss * mask_b).flatten(1).sum(dim=1)
    per_sample_denom = mask_b.flatten(1).sum(dim=1).clamp(min=1e-8)
    per_sample_loss = per_sample_numer / per_sample_denom
    # Average over samples that actually had any valid element.
    has_valid = (mask_b.flatten(1).sum(dim=1) > 0).to(elem_loss.dtype)
    n_valid_samples = has_valid.sum().clamp(min=1e-8)
    return (per_sample_loss * has_valid).sum() / n_valid_samples
