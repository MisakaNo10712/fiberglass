"""
Agent 6: 主模型（Mamba Encoder + 系数 Head）。

工程背景:
- 我们从光纤路径采样得到序列 token：每个点包含坐标/切线/曲率等特征，形成 X ∈ R^{B×L×F}。
- 需要一个序列编码器将 X 编码为全局向量，再回归低维系数 a ∈ R^{B×K}，用于后续通过 2D 基函数解码 w(x,y) 并计算 κ_t。
- 本任务只实现“序列到系数”的网络，不实现损失与训练循环。

该实现优先使用 mamba-ssm（若可用且输入在 CUDA 上），否则自动回退到无额外依赖的 1D CNN 编码器。
"""

from __future__ import annotations

from typing import Optional
import warnings

import torch
from torch import Tensor, nn

try:
    from mamba_ssm import Mamba  # type: ignore

    _MAMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    Mamba = None
    _MAMBA_AVAILABLE = False

__all__ = ["MambaCoeffNet"]


def _masked_mean_pooling(hidden: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Masked mean pooling over the sequence length dimension."""
    if mask is None:
        mask_f = torch.ones(hidden.shape[:2], device=hidden.device, dtype=hidden.dtype)
    else:
        mask_f = mask.to(device=hidden.device, dtype=hidden.dtype)

    masked_hidden = hidden * mask_f.unsqueeze(-1)
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    return masked_hidden.sum(dim=1) / denom


def _masked_attention_pooling(hidden: Tensor, mask: Optional[Tensor], proj: nn.Linear) -> Tensor:
    """Masked attention pooling with a learnable projection."""
    scores = proj(hidden).squeeze(-1)
    if mask is None:
        mask_f = torch.ones_like(scores, dtype=hidden.dtype, device=hidden.device)
    else:
        mask_f = mask.to(device=hidden.device, dtype=hidden.dtype)
    scores = scores.masked_fill(mask_f <= 0, -1e9)
    weights = torch.softmax(scores, dim=1)
    return (hidden * weights.unsqueeze(-1)).sum(dim=1)


class ConvResidualBlock(nn.Module):
    """Lightweight residual 1D CNN block."""

    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.norm = nn.GroupNorm(1, d_model)
        self.conv1 = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, padding=padding, dilation=dilation
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, L, d]
        y = x.transpose(1, 2)  # [B, d, L]
        y = self.norm(y)
        y = self.conv1(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)
        return x + y


class MambaCoeffNet(nn.Module):
    """
    Sequence-to-coefficient network with optional Mamba encoder and CNN fallback.

    Args:
        in_features: Input feature dimension F.
        d_model: Hidden dimension for the encoder.
        n_layers: Number of encoder layers.
        dropout: Dropout probability applied after embedding and inside blocks.
        out_coeffs: Output coefficient dimension K.
        encoder_type: "auto" (prefer Mamba when available), "mamba", or "cnn".
        cnn_kernel_size: Kernel size for CNN fallback.
        cnn_dilation_base: Base dilation factor for CNN layers.
        mlp_hidden: Hidden dimension of the prediction head.
        embedding_norm: Whether to apply LayerNorm after the input projection.
        pooling: Pooling mode ("mean" or "attn").
        kappa_scale_learnable: Whether to expose a learnable scalar for kappa loss scaling.
        kappa_scale_init: Initial value for the kappa scale parameter.
    """

    def __init__(
        self,
        *,
        in_features: int,
        d_model: int = 128,
        n_layers: int = 4,
        dropout: float = 0.0,
        out_coeffs: int,
        encoder_type: str = "auto",
        cnn_kernel_size: int = 5,
        cnn_dilation_base: int = 1,
        mlp_hidden: int = 256,
        embedding_norm: bool = True,
        pooling: str = "mean",
        kappa_scale_learnable: bool = False,
        kappa_scale_init: float = 1.0,
    ) -> None:
        super().__init__()

        if encoder_type not in {"auto", "mamba", "cnn"}:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        self.encoder_type = encoder_type
        self._warned_cpu_fallback = False
        self.pooling = pooling

        if pooling not in {"mean", "attn"}:
            raise ValueError(f"Unsupported pooling: {pooling}")

        want_mamba = encoder_type in {"auto", "mamba"} and _MAMBA_AVAILABLE
        if encoder_type == "mamba" and not _MAMBA_AVAILABLE:
            warnings.warn("mamba-ssm not available; falling back to CNN encoder.", RuntimeWarning)
        self.d_model = d_model

        # Mask invalid tokens before embedding to mirror hard deletion; linear layer is bias-free
        # so zeroed tokens stay zero.
        embedding_layers: list[nn.Module] = [nn.Linear(in_features, d_model, bias=False)]
        if embedding_norm:
            embedding_layers.append(nn.LayerNorm(d_model))
        embedding_layers.append(nn.Dropout(dropout))
        self.embedding = nn.Sequential(*embedding_layers)

        self.mamba_layers = (
            nn.ModuleList([Mamba(d_model=d_model) for _ in range(n_layers)]) if want_mamba else None
        )
        self.cnn_layers = nn.ModuleList(
            [
                ConvResidualBlock(
                    d_model=d_model,
                    kernel_size=cnn_kernel_size,
                    dilation=max(1, cnn_dilation_base ** i),
                    dropout=dropout,
                )
                for i in range(n_layers)
            ]
        )

        self.pool_attn = nn.Linear(d_model, 1) if pooling == "attn" else None

        self.head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, out_coeffs),
        )

        self.kappa_scale_log = (
            nn.Parameter(torch.log(torch.tensor(float(kappa_scale_init))))
            if kappa_scale_learnable
            else None
        )

    def get_kappa_scale(self) -> Tensor | None:
        if self.kappa_scale_log is None:
            return None
        return self.kappa_scale_log.exp()

    def forward(self, X: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            X: Input tensor of shape [B, L, F].
            mask: Optional mask of shape [B, L]; 1 for valid tokens, 0 for padding.

        Returns:
            Tensor of shape [B, K] representing basis coefficients.
        """

        if mask is None:
            mask_f = torch.ones(X.shape[:2], device=X.device, dtype=X.dtype)
        else:
            mask_f = mask.to(device=X.device, dtype=X.dtype)

        x_masked = X * mask_f.unsqueeze(-1)
        hidden = self.embedding(x_masked)

        # Zero-out invalid positions before encoding to minimize leakage through convolutions.
        hidden = hidden * mask_f.unsqueeze(-1)

        use_mamba = self.encoder_type in {"auto", "mamba"} and self.mamba_layers is not None
        if use_mamba and not hidden.is_cuda:
            if not self._warned_cpu_fallback:
                warnings.warn(
                    "Mamba encoder requires CUDA in this environment; falling back to CNN.",
                    RuntimeWarning,
                )
                self._warned_cpu_fallback = True
            use_mamba = False

        if use_mamba:
            for layer in self.mamba_layers:
                hidden = layer(hidden)
        else:
            for layer in self.cnn_layers:
                hidden = layer(hidden)

        if self.pooling == "attn":
            pooled = _masked_attention_pooling(hidden, mask_f, self.pool_attn)
        else:
            pooled = _masked_mean_pooling(hidden, mask_f)
        return self.head(pooled)
