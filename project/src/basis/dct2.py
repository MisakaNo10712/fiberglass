"""
Cosine DCT-like basis utilities for 2D plate deflection reconstruction.

工程背景:
- 我们要从光纤测得的双面应变差分得到 κ_t(s)，并反演板的挠度 w(x,y)。
- 为了让欠定问题可解，我们用低维二维基函数展开：w(x,y) = Σ a[m,n] φ_m(x) ψ_n(y)。
- 需要实现可微算子，以便在训练时通过损失 L_kappa = || κ_pred - κ_meas ||，其中 κ_pred = tᵀ H(w) t。
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


__all__ = [
    "make_mode_frequencies",
    "w_from_coeff",
    "hessian_from_coeff",
    "hf_weights",
]


def make_mode_frequencies(
    M: int,
    N: int,
    Lx: float,
    Ly: float,
    device=None,
    dtype=None,
) -> tuple[Tensor, Tensor]:
    """
    Construct modal spatial frequencies for the cosine basis.

    Args:
        M: Number of modes along x.
        N: Number of modes along y.
        Lx: Domain length in x (must be > 0).
        Ly: Domain length in y (must be > 0).
        device: Optional torch device for the returned tensors.
        dtype: Optional dtype for the returned tensors.

    Returns:
        kx: Tensor with shape [M], where kx[m] = m * pi / Lx.
        ky: Tensor with shape [N], where ky[n] = n * pi / Ly.
    """
    if Lx <= 0 or Ly <= 0:
        raise ValueError("Lx and Ly must be positive.")

    factor_x = torch.pi / float(Lx)
    factor_y = torch.pi / float(Ly)
    kx = torch.arange(M, device=device, dtype=dtype) * factor_x
    ky = torch.arange(N, device=device, dtype=dtype) * factor_y
    return kx, ky


def _prepare_inputs(
    a: Tensor,
    x: Tensor,
    y: Tensor,
    Lx: float,
    Ly: float,
) -> Tuple[Tensor, bool, Tensor, Tensor, tuple[int, ...], Tensor, Tensor]:
    """Normalize inputs, broadcast x/y, and emit modal frequencies."""
    if a.ndim == 2:
        coeff = a.unsqueeze(0)
        has_batch = False
    elif a.ndim == 3:
        coeff = a
        has_batch = True
    else:
        raise ValueError(f"`a` must have shape [M,N] or [B,M,N], got {tuple(a.shape)}")

    device, dtype = coeff.device, coeff.dtype
    x_t = torch.as_tensor(x, device=device, dtype=dtype)
    y_t = torch.as_tensor(y, device=device, dtype=dtype)
    x_b, y_b = torch.broadcast_tensors(x_t, y_t)

    x_flat = x_b.reshape(-1)
    y_flat = y_b.reshape(-1)
    kx, ky = make_mode_frequencies(coeff.shape[1], coeff.shape[2], Lx, Ly, device=device, dtype=dtype)
    return coeff, has_batch, x_flat, y_flat, x_b.shape, kx, ky


def _reshape_output(values: Tensor, has_batch: bool, shape: tuple[int, ...]) -> Tensor:
    """Reshape [B, P] result back to broadcasted spatial shape and drop batch if absent."""
    reshaped = values.reshape(values.shape[0], *shape)
    return reshaped if has_batch else reshaped.squeeze(0)


def w_from_coeff(a, x, y, *, Lx: float, Ly: float) -> Tensor:
    """
    Evaluate deflection field w(x, y) from modal coefficients.

    Broadcasting rules:
    - ``a`` may be [M, N] (no batch) or [B, M, N] (batched).
    - ``x`` and ``y`` are broadcast against each other; resulting spatial shape is
      preserved in the output.
    - If ``a`` is 2D (no batch), the returned tensor omits the batch dimension.
      Otherwise the leading dimension is ``B``.
    """
    coeff, has_batch, x_flat, y_flat, out_shape, kx, ky = _prepare_inputs(
        torch.as_tensor(a), x, y, Lx, Ly
    )

    phi_x = torch.cos(x_flat[:, None] * kx[None, :])  # [P, M]
    psi_y = torch.cos(y_flat[:, None] * ky[None, :])  # [P, N]

    w_flat = torch.einsum("bmn,pm,pn->bp", coeff, phi_x, psi_y)
    return _reshape_output(w_flat, has_batch, out_shape)


def hessian_from_coeff(a, x, y, *, Lx: float, Ly: float) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute analytic Hessian components (w_xx, w_xy, w_yy) from modal coefficients.

    Derivatives follow the closed-form expressions:
        d²φ_m/dx² = -kx_m² * φ_m(x)
        dφ_m/dx = -kx_m * sin(kx_m * x)
    with analogous terms for ψ_n(y).
    """
    coeff, has_batch, x_flat, y_flat, out_shape, kx, ky = _prepare_inputs(
        torch.as_tensor(a), x, y, Lx, Ly
    )

    phi_x = torch.cos(x_flat[:, None] * kx[None, :])  # [P, M]
    psi_y = torch.cos(y_flat[:, None] * ky[None, :])  # [P, N]

    sin_x = torch.sin(x_flat[:, None] * kx[None, :])
    sin_y = torch.sin(y_flat[:, None] * ky[None, :])

    dphi_x = -(kx[None, :] * sin_x)  # dφ/dx, shape [P, M]
    dpsi_y = -(ky[None, :] * sin_y)  # dψ/dy, shape [P, N]

    kx_sq = kx**2
    ky_sq = ky**2
    d2phi_x = -(kx_sq[None, :] * phi_x)  # d²φ/dx²
    d2psi_y = -(ky_sq[None, :] * psi_y)  # d²ψ/dy²

    w_xx_flat = torch.einsum("bmn,pm,pn->bp", coeff, d2phi_x, psi_y)
    w_xy_flat = torch.einsum("bmn,pm,pn->bp", coeff, dphi_x, dpsi_y)
    w_yy_flat = torch.einsum("bmn,pm,pn->bp", coeff, phi_x, d2psi_y)

    w_xx = _reshape_output(w_xx_flat, has_batch, out_shape)
    w_xy = _reshape_output(w_xy_flat, has_batch, out_shape)
    w_yy = _reshape_output(w_yy_flat, has_batch, out_shape)
    return w_xx, w_xy, w_yy


def hf_weights(
    M: int,
    N: int,
    *,
    alpha: float = 2.0,
    eps: float = 1e-12,
    device=None,
    dtype=None,
) -> Tensor:
    """
    High-frequency penalty weights W[m, n] = ((m^2 + n^2) + eps)^(alpha / 2).

    The weights grow with spatial frequency; W[0,0] is minimal to avoid over-penalizing
    the DC component.
    """
    m = torch.arange(M, device=device, dtype=dtype)
    n = torch.arange(N, device=device, dtype=dtype)
    m2 = m[:, None] ** 2
    n2 = n[None, :] ** 2
    return (m2 + n2 + eps) ** (alpha / 2)
