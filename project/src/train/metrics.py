"""
Training metrics for coefficient regression tasks.
"""

from __future__ import annotations

from typing import Optional

import math
import torch
from torch import Tensor

__all__ = ["rmse", "boundary_error", "compute_metrics"]


def rmse(pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> float:
    """Compute RMSE with optional mask."""
    diff = pred - target
    if mask is None:
        mse = (diff**2).mean()
    else:
        mask_f = mask.to(device=diff.device, dtype=diff.dtype)
        mse = ((diff**2) * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    return float(torch.sqrt(mse).item())


def boundary_error(
    w_pred_points: Optional[Tensor],
    y: Optional[Tensor],
    *,
    Ly: float,
    mask: Optional[Tensor] = None,
    tol: float = 1e-4,
) -> float:
    """Boundary error on y=Ly; returns NaN if boundary points are unavailable."""
    if w_pred_points is None or y is None:
        return float("nan")
    boundary = (y - Ly).abs() <= tol
    if mask is not None:
        boundary = boundary & (mask > 0)
    if not boundary.any():
        return float("nan")
    values = w_pred_points[boundary]
    if values.numel() == 0:
        return float("nan")
    return float(torch.sqrt((values**2).mean()).item())


def compute_metrics(batch: dict, outputs: dict, config: dict) -> dict[str, float]:
    """Compute kappa RMSE, optional w RMSE, and boundary error."""
    mask = batch.get("mask")
    kappa_meas = batch.get("kappa_meas", batch["X"][..., 4])
    kappa_pred = outputs["kappa_pred"]
    metrics = {"rmse_kappa": rmse(kappa_pred, kappa_meas, mask=mask)}

    if "w_points" in batch and outputs.get("w_pred_points") is not None:
        metrics["rmse_w"] = rmse(outputs["w_pred_points"], batch["w_points"], mask=mask)

    Ly = float(config.get("basis", {}).get("Ly", math.nan))
    metrics["boundary_error"] = boundary_error(
        outputs.get("w_pred_points"),
        batch.get("y"),
        Ly=Ly,
        mask=mask,
    )
    return metrics
