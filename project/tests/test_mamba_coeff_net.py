from __future__ import annotations

import torch

from src.models import MambaCoeffNet


def test_output_shape_and_dtype_cpu():
    torch.manual_seed(42)
    batch, seq_len, feat, coeffs = 2, 64, 5, 30
    X = torch.randn(batch, seq_len, feat, dtype=torch.float32)
    model = MambaCoeffNet(in_features=feat, out_coeffs=coeffs, d_model=128, n_layers=2)

    out = model(X, mask=None)

    assert out.shape == (batch, coeffs)
    assert out.dtype == X.dtype


def test_masked_pooling_effect_without_encoder_mixing():
    torch.manual_seed(0)
    batch, seq_len, feat = 1, 10, 3
    big_value = 1000.0

    # First half zero, second half very large values that should be masked out.
    X = torch.zeros(batch, seq_len, feat, dtype=torch.float32)
    X[:, seq_len // 2 :, :] = big_value

    mask = torch.ones(batch, seq_len, dtype=torch.float32)
    mask[:, seq_len // 2 :] = 0.0

    model = MambaCoeffNet(
        in_features=feat,
        out_coeffs=4,
        d_model=16,
        n_layers=0,  # identity encoder; no token mixing
        encoder_type="cnn",
    )

    out_masked = model(X, mask=mask)
    X_zeroed = X.clone()
    X_zeroed[:, seq_len // 2 :, :] = 0.0
    out_zero_padded = model(X_zeroed, mask=torch.ones_like(mask))

    assert torch.allclose(out_masked, out_zero_padded, atol=1e-5, rtol=1e-5)

    # Without masking, the giant values should change the output noticeably.
    out_unmasked = model(X, mask=torch.ones_like(mask))
    assert not torch.allclose(out_unmasked, out_masked)


def test_forward_runs_with_auto_encoder():
    torch.manual_seed(123)
    batch, seq_len, feat, coeffs = 3, 32, 7, 11
    X = torch.randn(batch, seq_len, feat, dtype=torch.float32)
    rand_mask = (torch.rand(batch, seq_len) > 0.3).float()
    # Guarantee at least one valid token per batch
    rand_mask[:, 0] = 1.0

    model = MambaCoeffNet(
        in_features=feat,
        out_coeffs=coeffs,
        d_model=32,
        n_layers=1,
        encoder_type="auto",
    )

    out = model(X, mask=rand_mask)

    assert out.shape == (batch, coeffs)
