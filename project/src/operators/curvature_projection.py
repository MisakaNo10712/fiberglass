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

__all__ = ["kappa_t_from_coeff"]


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
