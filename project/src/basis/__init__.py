"""
二维余弦基函数工具集。

工程背景:
- 我们要从光纤测得的双面应变差分得到 κ_t(s)，并反演板的挠度 w(x,y)。
- 为了让欠定问题可解，我们用低维二维基函数展开，并需要可微算子支持 κ_pred = tᵀ H(w) t。
"""

from .dct2 import hf_weights, hessian_from_coeff, make_mode_frequencies, w_from_coeff

__all__ = [
    "make_mode_frequencies",
    "w_from_coeff",
    "hessian_from_coeff",
    "hf_weights",
]
