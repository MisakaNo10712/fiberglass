"""
Curvature projection operator κ_t = tᵀ H(w) t over a 2D cosine basis expansion.

工程背景:
- 我们要从光纤测得的双面应变差分得到 κ_t(s)，并反演板的挠度 w(x,y)。
- 为了让欠定问题可解，我们用低维二维基函数展开：w(x,y) = Σ a[m,n] φ_m(x) ψ_n(y)。
- 需要实现可微算子，通过损失 L_kappa = || κ_pred - κ_meas ||，其中 κ_pred = tᵀ H(w) t。
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.basis.dct2 import hessian_from_coeff

__all__ = ["kappa_t_from_coeff", "solve_a_from_kappa_lstsq"]


def kappa_t_from_coeff(a, x, y, tx, ty, *, Lx: float, Ly: float) -> Tensor:
    """
    Compute curvature projection κ_t at arbitrary sample points.

    Args:
        a: Modal coefficients [M, N] or [B, M, N].
        x, y: Sample coordinates, broadcastable to a common spatial shape.
        tx, ty: Tangent components (unit-length upstream), broadcastable with ``x``/``y``.
        Lx, Ly: Domain lengths.

    Returns:
        Tensor shaped like ``w_xx``: [*spatial] for non-batch input or [B, *spatial]
        when coefficients include a batch dimension.
    """
    w_xx, w_xy, w_yy = hessian_from_coeff(a, x, y, Lx=Lx, Ly=Ly)

    tx_t = torch.as_tensor(tx, device=w_xx.device, dtype=w_xx.dtype)
    ty_t = torch.as_tensor(ty, device=w_xx.device, dtype=w_xx.dtype)
    tx_b, ty_b = torch.broadcast_tensors(tx_t, ty_t)

    kappa = (tx_b**2) * w_xx + 2 * tx_b * ty_b * w_xy + (ty_b**2) * w_yy
    return kappa


def solve_a_from_kappa_lstsq(
    batch_x: Tensor,
    batch_y: Tensor,
    batch_tx: Tensor,
    batch_ty: Tensor,
    batch_kappa: Tensor,
    M: int,
    N: int,
    Lx: float,
    Ly: float,
    ridge: float = 1e-6,
    mask: Tensor | None = None,
) -> Tensor:
    """
    Solve for coefficients a that best fit kappa via linear least squares.

    Args:
        batch_x/y/tx/ty/kappa: Tensors shaped [B, L] or [L].
        M, N: Basis dimensions.
        Lx, Ly: Domain lengths.
        ridge: Tikhonov ridge for underdetermined systems.
        mask: Optional mask [B, L] or [L] where >0 indicates valid points.

    Returns:
        Tensor of shape [B, M, N] (or [M, N] if input was unbatched).
    """

    x_t = torch.as_tensor(batch_x)
    y_t = torch.as_tensor(batch_y)
    tx_t = torch.as_tensor(batch_tx)
    ty_t = torch.as_tensor(batch_ty)
    kappa_t = torch.as_tensor(batch_kappa)

    added_batch = False
    if x_t.ndim == 1:
        x_t = x_t.unsqueeze(0)
        y_t = y_t.unsqueeze(0)
        tx_t = tx_t.unsqueeze(0)
        ty_t = ty_t.unsqueeze(0)
        kappa_t = kappa_t.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)
        added_batch = True

    device = x_t.device
    dtype = x_t.dtype if x_t.dtype in (torch.float32, torch.float64) else torch.float32
    basis = torch.eye(M * N, device=device, dtype=dtype).reshape(M * N, M, N)

    outputs: list[Tensor] = []
    for idx in range(x_t.shape[0]):
        x_i = x_t[idx].to(dtype=dtype).reshape(-1)
        y_i = y_t[idx].to(dtype=dtype).reshape(-1)
        tx_i = tx_t[idx].to(dtype=dtype).reshape(-1)
        ty_i = ty_t[idx].to(dtype=dtype).reshape(-1)
        kappa_i = kappa_t[idx].to(dtype=dtype).reshape(-1)

        valid = (
            torch.isfinite(x_i)
            & torch.isfinite(y_i)
            & torch.isfinite(tx_i)
            & torch.isfinite(ty_i)
            & torch.isfinite(kappa_i)
        )
        if mask is not None:
            mask_i = mask[idx].reshape(-1) > 0
            valid = valid & mask_i
        if not torch.any(valid):
            outputs.append(torch.zeros((M, N), device=device, dtype=dtype))
            continue

        kappa_basis = kappa_t_from_coeff(basis, x_i, y_i, tx_i, ty_i, Lx=Lx, Ly=Ly)
        K = kappa_basis.transpose(0, 1)
        K = K[valid]
        b = kappa_i[valid]

        if K.shape[0] < K.shape[1]:
            AtA = K.T @ K
            Atb = K.T @ b
            reg = ridge * torch.eye(K.shape[1], device=device, dtype=dtype)
            sol = torch.linalg.solve(AtA + reg, Atb)
        else:
            sol = torch.linalg.lstsq(K, b).solution
            if torch.isnan(sol).any() or torch.isinf(sol).any():
                AtA = K.T @ K
                Atb = K.T @ b
                reg = ridge * torch.eye(K.shape[1], device=device, dtype=dtype)
                sol = torch.linalg.solve(AtA + reg, Atb)

        outputs.append(sol.reshape(M, N))

    out = torch.stack(outputs, dim=0)
    return out.squeeze(0) if added_batch else out
