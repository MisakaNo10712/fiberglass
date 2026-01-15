"""
Loss functions for curvature-consistent coefficient training.
"""

from __future__ import annotations

from typing import Optional
import logging

import torch
from torch import Tensor

__all__ = ["huber_loss", "hf_l2_loss", "optional_bc_loss", "total_loss"]

logger = logging.getLogger(__name__)


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


def _as_scalar_tensor(value: float | Tensor, ref: Tensor) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=ref.device, dtype=ref.dtype)
    return torch.tensor(float(value), device=ref.device, dtype=ref.dtype)


def _assert_positive(name: str, value: float | Tensor) -> None:
    if isinstance(value, Tensor):
        if not torch.all(value > 0):
            raise ValueError(f"{name} must be > 0")
    else:
        if float(value) <= 0:
            raise ValueError(f"{name} must be > 0")


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
    kappa_std: float | Tensor | None = None,
    w_std: float | Tensor | None = None,
    lambda_kappa: float = 1.0,
    lambda_hf: float = 1.0,
    lambda_w: float = 0.0,
    lambda_bc: float = 0.0,
    huber_delta: float = 1.0,
    step: Optional[int] = None,
    log_every_n: Optional[int] = None,
) -> dict[str, Tensor]:
    """Compute total loss and return a dict of components."""
    if kappa_std is None or w_std is None:
        raise ValueError("kappa_std and w_std must be provided in batch.")
    _assert_positive("kappa_std", kappa_std)
    _assert_positive("w_std", w_std)

    kappa_std_t = _as_scalar_tensor(kappa_std, kappa_pred).clamp_min(1e-8)
    kappa_pred_hat = kappa_pred / kappa_std_t
    kappa_meas_hat = kappa_meas / kappa_std_t
    raw_loss_kappa = huber_loss(kappa_pred_hat, kappa_meas_hat, mask=mask, delta=huber_delta)
    raw_loss_hf = hf_l2_loss(a, hf_W)

    weighted_loss_kappa = raw_loss_kappa * lambda_kappa
    weighted_loss_hf = raw_loss_hf * lambda_hf

    loss = weighted_loss_kappa + weighted_loss_hf
    output = {
        "loss": loss,
        "loss_kappa": raw_loss_kappa,
        "loss_hf": raw_loss_hf,
        "raw_loss_kappa": raw_loss_kappa,
        "raw_loss_hf": raw_loss_hf,
        "weighted_loss_kappa": weighted_loss_kappa,
        "weighted_loss_hf": weighted_loss_hf,
    }

    if w_pred_points is not None and w_true_points is not None:
        w_mask = mask_w if mask_w is not None else mask
        w_std_t = _as_scalar_tensor(w_std, w_pred_points).clamp_min(1e-8)
        w_pred_hat = w_pred_points / w_std_t
        w_true_hat = w_true_points / w_std_t
        raw_loss_w = huber_loss(w_pred_hat, w_true_hat, mask=w_mask, delta=huber_delta)
        weighted_loss_w = raw_loss_w * lambda_w
        output["loss_w"] = raw_loss_w
        output["raw_loss_w"] = raw_loss_w
        output["weighted_loss_w"] = weighted_loss_w
        loss = loss + weighted_loss_w

    if bc_pred is not None and lambda_bc > 0:
        loss_bc = optional_bc_loss(bc_pred, mask_boundary=bc_mask)
        output["loss_bc"] = loss_bc
        loss = loss + lambda_bc * loss_bc

    if log_every_n is not None and log_every_n > 0 and step is not None:
        if step % log_every_n == 0:
            w_numel = int(w_pred_points.numel()) if w_pred_points is not None else 0
            logger.info(
                "loss stats: kappa_std=%.6g, w_std=%.6g, lambdas(kappa=%.4g,w=%.4g,hf=%.4g), numel(kappa=%d,w=%d)",
                float(kappa_std),
                float(w_std),
                float(lambda_kappa),
                float(lambda_w),
                float(lambda_hf),
                int(kappa_pred.numel()),
                w_numel,
            )

    output["loss"] = loss
    return output
