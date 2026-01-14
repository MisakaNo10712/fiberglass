from __future__ import annotations

import torch

from src.basis.dct2 import hessian_from_coeff, w_from_coeff
from src.operators.curvature_projection import kappa_t_from_coeff


def test_w_xx_matches_finite_difference():
    torch.manual_seed(0)
    M, N = 6, 5
    Lx = Ly = 1.0

    a = torch.randn(M, N, dtype=torch.float64)
    rng = torch.Generator().manual_seed(1)
    x = torch.rand(20, generator=rng, dtype=torch.float64) * 0.9 + 0.05
    y = torch.rand(20, generator=rng, dtype=torch.float64) * 0.9 + 0.05
    delta = 1e-5

    w_xx, _, _ = hessian_from_coeff(a, x, y, Lx=Lx, Ly=Ly)
    w_plus = w_from_coeff(a, x + delta, y, Lx=Lx, Ly=Ly)
    w = w_from_coeff(a, x, y, Lx=Lx, Ly=Ly)
    w_minus = w_from_coeff(a, x - delta, y, Lx=Lx, Ly=Ly)

    w_xx_fd = (w_plus - 2 * w + w_minus) / (delta**2)

    max_err = (w_xx_fd - w_xx).abs().max()
    # Central differences incur O(delta^2) truncation error; 1e-4 comfortably covers
    # high-frequency modes up to the tested order in float64.
    assert max_err.item() < 1e-4


def test_kappa_shape_and_dtype():
    torch.manual_seed(0)
    M, N, P = 4, 3, 7
    x = torch.rand(P)
    y = torch.rand(P)
    tx = torch.ones(P)
    ty = torch.zeros(P)

    a = torch.randn(M, N, dtype=torch.float32)
    kappa = kappa_t_from_coeff(a, x, y, tx, ty, Lx=1.0, Ly=1.0)
    assert kappa.shape == (P,)
    assert kappa.dtype == torch.float32

    a_batch = torch.randn(2, M, N, dtype=torch.float32)
    kappa_batch = kappa_t_from_coeff(a_batch, x, y, tx, ty, Lx=1.0, Ly=1.0)
    assert kappa_batch.shape == (2, P)
    assert kappa_batch.dtype == torch.float32


def test_autograd_grad_exists():
    a = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    x = torch.linspace(0.05, 0.95, steps=5, dtype=torch.float64)
    y = torch.linspace(0.05, 0.95, steps=5, dtype=torch.float64)
    tx = torch.ones_like(x)
    ty = torch.zeros_like(y)

    kappa = kappa_t_from_coeff(a, x, y, tx, ty, Lx=1.0, Ly=1.0)
    loss = kappa.mean()
    loss.backward()

    assert a.grad is not None
    assert a.grad.shape == a.shape
