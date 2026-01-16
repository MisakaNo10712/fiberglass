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
EPS = 1e-12


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


def _masked_mean(values: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is None:
        return values.mean(dim=-1)
    mask_f = mask.to(device=values.device, dtype=values.dtype)
    denom = mask_f.sum(dim=-1).clamp_min(1.0)
    return (values * mask_f).sum(dim=-1) / denom


def _align_mean(
    w_pred: Tensor, w_true: Tensor, mask: Optional[Tensor]
) -> tuple[Tensor, Tensor]:
    diff = w_true - w_pred
    diff_flat = diff.reshape(diff.shape[0], -1)
    mask_flat = mask.reshape(mask.shape[0], -1) if mask is not None else None
    offset = _masked_mean(diff_flat, mask_flat)
    view_shape = (diff.shape[0],) + (1,) * (w_pred.ndim - 1)
    w_aligned = w_pred + offset.view(view_shape)
    return w_aligned, offset


def _align_plane(
    x: Tensor,
    y: Tensor,
    w_pred: Tensor,
    w_true: Tensor,
    mask: Optional[Tensor],
) -> tuple[Tensor, Tensor]:
    if x is None or y is None:
        return w_pred, torch.full((w_pred.shape[0], 3), float("nan"), device=w_pred.device)
    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)
    diff = (w_true - w_pred).reshape(w_pred.shape[0], -1)
    mask_flat = mask.reshape(mask.shape[0], -1) if mask is not None else None

    coeffs = []
    aligned_list = []
    for idx in range(diff.shape[0]):
        xi = x_flat[idx]
        yi = y_flat[idx]
        di = diff[idx]
        valid = torch.isfinite(xi) & torch.isfinite(yi) & torch.isfinite(di)
        if mask_flat is not None:
            valid = valid & (mask_flat[idx] > 0)
        if valid.sum() < 3:
            coeffs.append(
                torch.tensor([float("nan"), float("nan"), float("nan")], device=w_pred.device)
            )
            aligned_list.append(w_pred[idx].reshape(-1))
            continue
        A = torch.stack([xi[valid], yi[valid], torch.ones_like(xi[valid])], dim=1)
        sol = torch.linalg.lstsq(A, di[valid]).solution
        plane = sol[0] * xi + sol[1] * yi + sol[2]
        aligned_list.append(w_pred[idx].reshape(-1) + plane)
        coeffs.append(sol)

    aligned = torch.stack(aligned_list, dim=0).reshape(w_pred.shape)
    coeffs_t = torch.stack(coeffs, dim=0)
    return aligned, coeffs_t


def _select_anchor(
    values: Tensor, mask: Optional[Tensor], anchor_index: int
) -> tuple[Tensor, Tensor]:
    values_flat = values.reshape(values.shape[0], -1)
    mask_flat = mask.reshape(mask.shape[0], -1) if mask is not None else None
    anchors = []
    valid_flags = []
    for idx in range(values_flat.shape[0]):
        if mask_flat is None:
            pos = min(max(anchor_index, 0), values_flat.shape[1] - 1)
            anchors.append(values_flat[idx, pos])
            valid_flags.append(True)
            continue
        valid_idx = torch.nonzero(mask_flat[idx] > 0, as_tuple=False).reshape(-1)
        if valid_idx.numel() == 0:
            anchors.append(torch.tensor(float("nan"), device=values.device, dtype=values.dtype))
            valid_flags.append(False)
            continue
        if 0 <= anchor_index < int(valid_idx.numel()):
            pos = valid_idx[int(anchor_index)]
        else:
            pos = valid_idx[0]
        anchors.append(values_flat[idx, pos])
        valid_flags.append(True)
    return torch.stack(anchors, dim=0), torch.tensor(valid_flags, device=values.device)


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
    w_align_x: Optional[Tensor] = None,
    w_align_y: Optional[Tensor] = None,
    kappa_mean: float | Tensor | None = None,
    kappa_std: float | Tensor | None = None,
    w_mean: float | Tensor | None = None,
    w_std: float | Tensor | None = None,
    lambda_kappa: float = 1.0,
    lambda_hf: float = 1.0,
    lambda_w: float = 0.0,
    lambda_bc: float = 0.0,
    lambda_delta_a: float = 0.0,
    a_delta: Optional[Tensor] = None,
    w_align_mode: str = "none",
    anchor_mode: str = "none",
    lambda_anchor: float = 0.0,
    anchor_index: int = 0,
    huber_delta: float = 1.0,
    step: Optional[int] = None,
    log_every_n: Optional[int] = None,
) -> dict[str, Tensor]:
    """Compute total loss and return a dict of components."""
    if kappa_std is None or w_std is None or kappa_mean is None or w_mean is None:
        raise ValueError("kappa_mean/kappa_std and w_mean/w_std must be provided in batch.")
    _assert_positive("kappa_std", kappa_std)
    _assert_positive("w_std", w_std)

    kappa_std_t = _as_scalar_tensor(kappa_std, kappa_pred).clamp_min(EPS)
    kappa_mean_t = _as_scalar_tensor(kappa_mean, kappa_pred)
    kappa_pred_hat = (kappa_pred - kappa_mean_t) / (kappa_std_t + EPS)
    kappa_meas_hat = (kappa_meas - kappa_mean_t) / (kappa_std_t + EPS)
    raw_loss_kappa = huber_loss(kappa_pred_hat, kappa_meas_hat, mask=mask, delta=huber_delta)
    raw_loss_hf = hf_l2_loss(a, hf_W)

    weighted_loss_kappa = raw_loss_kappa * lambda_kappa
    weighted_loss_hf = raw_loss_hf * lambda_hf
    raw_loss_w = torch.tensor(0.0, device=weighted_loss_kappa.device, dtype=weighted_loss_kappa.dtype)
    weighted_loss_w = raw_loss_w

    loss = weighted_loss_kappa + weighted_loss_hf
    output = {
        "loss": loss,
        "loss_kappa": raw_loss_kappa,
        "loss_hf": raw_loss_hf,
        "raw_loss_kappa": raw_loss_kappa,
        "raw_loss_hf": raw_loss_hf,
        "weighted_loss_kappa": weighted_loss_kappa,
        "weighted_loss_hf": weighted_loss_hf,
        "raw_loss_w": raw_loss_w,
        "weighted_loss_w": weighted_loss_w,
    }

    if w_pred_points is not None and w_true_points is not None:
        w_pred_for_loss = w_pred_points
        plane_coeffs = None
        mean_offset = None

        if w_align_mode == "mean":
            w_pred_for_loss, mean_offset = _align_mean(w_pred_points, w_true_points, mask_w)
        elif w_align_mode == "plane":
            w_pred_for_loss, plane_coeffs = _align_plane(
                w_align_x, w_align_y, w_pred_points, w_true_points, mask_w
            )

        w_mask = mask_w if mask_w is not None else mask
        w_std_t = _as_scalar_tensor(w_std, w_pred_points).clamp_min(EPS)
        w_mean_t = _as_scalar_tensor(w_mean, w_pred_points)
        w_pred_hat = (w_pred_for_loss - w_mean_t) / (w_std_t + EPS)
        w_true_hat = (w_true_points - w_mean_t) / (w_std_t + EPS)
        raw_loss_w = huber_loss(w_pred_hat, w_true_hat, mask=w_mask, delta=huber_delta)
        weighted_loss_w = raw_loss_w * lambda_w
        output["loss_w"] = raw_loss_w
        output["raw_loss_w"] = raw_loss_w
        output["weighted_loss_w"] = weighted_loss_w
        loss = loss + weighted_loss_w

        if mean_offset is not None:
            output["w_align_mean_offset"] = mean_offset.mean()
        if plane_coeffs is not None:
            output["w_plane_a"] = plane_coeffs[:, 0].mean()
            output["w_plane_b"] = plane_coeffs[:, 1].mean()
            output["w_plane_c"] = plane_coeffs[:, 2].mean()
    else:
        output["loss_w"] = raw_loss_w

    anchor_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
    if (
        w_pred_points is not None
        and w_true_points is not None
        and lambda_anchor > 0
        and anchor_mode != "none"
    ):
        w_mask = mask_w if mask_w is not None else mask
        w_std_t = _as_scalar_tensor(w_std, w_pred_points).clamp_min(EPS)
        if anchor_mode == "mean":
            diff = (w_pred_points - w_true_points).reshape(w_pred_points.shape[0], -1)
            mask_flat = w_mask.reshape(w_mask.shape[0], -1) if w_mask is not None else None
            diff_mean = _masked_mean(diff, mask_flat)
            anchor_loss = torch.mean((diff_mean / (w_std_t + EPS)) ** 2)
        elif anchor_mode == "point":
            anchor_pred, valid_pred = _select_anchor(w_pred_points, w_mask, anchor_index)
            anchor_true, valid_true = _select_anchor(w_true_points, w_mask, anchor_index)
            valid = valid_pred & valid_true & torch.isfinite(anchor_pred) & torch.isfinite(anchor_true)
            if valid.any():
                diff = (anchor_pred[valid] - anchor_true[valid]) / (w_std_t + EPS)
                anchor_loss = torch.mean(diff**2)

        loss = loss + lambda_anchor * anchor_loss
        output["loss_anchor"] = anchor_loss
        output["weighted_loss_anchor"] = lambda_anchor * anchor_loss

    delta_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
    if a_delta is not None and lambda_delta_a > 0:
        delta_loss = torch.mean(a_delta**2)
        loss = loss + lambda_delta_a * delta_loss
        output["loss_delta_a"] = delta_loss
        output["weighted_loss_delta_a"] = lambda_delta_a * delta_loss

    if bc_pred is not None and lambda_bc > 0:
        loss_bc = optional_bc_loss(bc_pred, mask_boundary=bc_mask)
        output["loss_bc"] = loss_bc
        loss = loss + lambda_bc * loss_bc

    if log_every_n is not None and log_every_n > 0 and step is not None:
        if step % log_every_n == 0:
            total_weighted = (
                weighted_loss_kappa
                + weighted_loss_w
                + weighted_loss_hf
                + output.get("weighted_loss_anchor", torch.tensor(0.0, device=loss.device))
                + output.get("weighted_loss_delta_a", torch.tensor(0.0, device=loss.device))
            ).clamp_min(EPS)
            ratio_kappa = float((weighted_loss_kappa / total_weighted).item())
            ratio_w = float((weighted_loss_w / total_weighted).item())
            ratio_hf = float((weighted_loss_hf / total_weighted).item())
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
            logger.info(
                "loss ratios (weighted): kappa=%.3f w=%.3f hf=%.3f",
                ratio_kappa,
                ratio_w,
                ratio_hf,
            )

    output["loss"] = loss
    return output
