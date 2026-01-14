"""
可微物理算子集合，目前提供沿切线方向的曲率投影 κ_t = tᵀ H(w) t。

工程背景:
- 我们要从光纤测得的双面应变差分得到 κ_t(s)，并反演板的挠度 w(x,y)。
- 为了让欠定问题可解，我们用低维二维基函数展开，并需要可微算子支持 κ_pred = tᵀ H(w) t。
"""

from .curvature_projection import kappa_t_from_coeff

__all__ = ["kappa_t_from_coeff"]
