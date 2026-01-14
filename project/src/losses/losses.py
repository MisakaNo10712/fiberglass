"""
Loss functions for curvature-consistent coefficient training.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

__all__ = ["huber_loss", "hf_l2_loss", "optional_bc_loss", "total_loss"]


def huber_loss(pred: Tensor, target: Tensor, mask: Optional[Tensor] = None, delta: float = 1.0) -> Tensor:
    """Masked Huber loss returning a scalar mean."""
    diff = pred - target
    abs_diff = diff.abs()
    loss = torch.where(abs_diff <= delta, 0.5 * diff**2, delta * (abs_diff - 0.5 * delta))
    if mask is None:
        return loss.mean()
    mask_f = mask.to(device=loss.device, dtype=loss.dtype)
    loss = loss * mask_f
    denom = mask_f.sum().clamp_min(1.0)
    return loss.sum() / denom


def hf_l2_loss(a: Tensor, hf_W: Tensor) -> Tensor:
    """High-frequency weighted L2 penalty."""
    if a.ndim == 2:
        m, n = hf_W.shape
        if a.shape[1] != m * n:
            raise ValueError(f"Expected K={m*n}, got {a.shape[1]}")
        coeff = a.reshape(a.shape[0], m, n)
    elif a.ndim == 3:
        coeff = a
    else:
        raise ValueError(f"`a` must have shape [B,K] or [B,M,N], got {tuple(a.shape)}")

    hf = hf_W.to(device=coeff.device, dtype=coeff.dtype)
    loss = (coeff**2 * hf).sum(dim=(-2, -1))
    return loss.mean()


def optional_bc_loss(w_pred_on_boundary: Tensor, mask_boundary: Optional[Tensor] = None) -> Tensor:
    """Boundary condition loss; defaults to mean-squared on provided boundary points."""
    if mask_boundary is None:
        return (w_pred_on_boundary**2).mean()
    mask_f = mask_boundary.to(device=w_pred_on_boundary.device, dtype=w_pred_on_boundary.dtype)
    loss = (w_pred_on_boundary**2) * mask_f
    denom = mask_f.sum().clamp_min(1.0)
    return loss.sum() / denom


def total_loss(
    kappa_pred: Tensor,
    kappa_meas: Tensor,
    a: Tensor,
    hf_W: Tensor,
    *,
    mask: Optional[Tensor] = None,
    w_pred_points: Optional[Tensor] = None,
    w_true_points: Optional[Tensor] = None,
    mask_w: Optional[Tensor] = None,
    bc_pred: Optional[Tensor] = None,
    bc_mask: Optional[Tensor] = None,
    lambda_hf: float = 1.0,
    lambda_w: float = 0.0,
    lambda_bc: float = 0.0,
    huber_delta: float = 1.0,
) -> dict[str, Tensor]:
    """Compute total loss and return a dict of components."""
    loss_kappa = huber_loss(kappa_pred, kappa_meas, mask=mask, delta=huber_delta)
    loss_hf = hf_l2_loss(a, hf_W)

    loss = loss_kappa + lambda_hf * loss_hf
    output = {
        "loss": loss,
        "loss_kappa": loss_kappa,
        "loss_hf": loss_hf,
    }

    if w_pred_points is not None and w_true_points is not None:
        w_mask = mask_w if mask_w is not None else mask
        loss_w = huber_loss(w_pred_points, w_true_points, mask=w_mask, delta=huber_delta)
        output["loss_w"] = loss_w
        loss = loss + lambda_w * loss_w

    if bc_pred is not None and lambda_bc > 0:
        loss_bc = optional_bc_loss(bc_pred, mask_boundary=bc_mask)
        output["loss_bc"] = loss_bc
        loss = loss + lambda_bc * loss_bc

    output["loss"] = loss
    return output
