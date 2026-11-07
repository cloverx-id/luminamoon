"""
LuminaMoon Neural Network Architecture Components (Ra-Family).

Rational, tanh-free and square-root-free neural components:
- RaNorm / L1Norm: Zero square-root, single-reciprocal token normalization.
- RaBatchNorm2d / L1BatchNorm2d: Mean Absolute Deviation (MAD) 2D batch normalization.
- RaTanh / RationalTanh & ra_tanh / rational_tanh: Rational hyperbolic tangent with non-vanishing tail.
- RaGELU / RationalGELU & ra_gelu / rational_gelu: Rational GELU formulation for vision and language models.
- PosRaGLU / RationalGLU: Strictly positive rational gate MLP block without activation explosion.
- RaGLU & TriGLU: Rational gated linear units.
- RaAttention / RationalAttention & ra_attention / flash_ra_attention: Fused tiled multi-head attention using rational softmax without exponential SFU bottlenecks or O(N^2) HBM materialization.
- ra_softmax / rational_softmax: Cayley-Padé rational softmax with compact support and closed-form autograd.
- FusedLinearCrossEntropy / RaLinearCrossEntropy & fused_linear_cross_entropy / ra_linear_cross_entropy: Fused Linear + Cross-Entropy loss with online LogSumExp and closed-form analytical backward without [B * N, V] logit materialization.
"""

from __future__ import annotations

from ._compat import (
    BaseModule,
    is_numpy_available,
    is_rust_simd_available,
    is_torch_available,
    is_triton_available,
)
from .norm import (
    RaBatchNorm2d,
    RaNorm,
)
from .activation import (
    PosRaGLU,
    RaGELU,
    RaGLU,
    RaTanh,
    TriGLU,
    pos_ra_gate,
    pos_ra_glu,
    ra_gelu,
    ra_tanh,
)
from .attention import (
    RaAttention,
    RaAttn,
    flash_ra_attention,
    ra_attention,
    ra_softmax,
)
from .loss import (
    FusedLinearCrossEntropy,
    fused_linear_cross_entropy,
)

# Academic Aliases
L1Norm = RaNorm
L1LayerNorm = RaNorm
L1BatchNorm2d = RaBatchNorm2d

RationalTanh = RaTanh
rational_tanh = ra_tanh
RationalGELU = RaGELU
rational_gelu = ra_gelu

RationalGLU = PosRaGLU
BoundedRationalGLU = PosRaGLU

RationalAttention = RaAttention
PadeAttention = RaAttention
rational_attention = ra_attention
pade_attention = ra_attention
rational_softmax = ra_softmax
pade_softmax = ra_softmax

RaLinearCrossEntropy = FusedLinearCrossEntropy
LinearCrossEntropy = FusedLinearCrossEntropy
ra_linear_cross_entropy = fused_linear_cross_entropy
linear_cross_entropy = fused_linear_cross_entropy

__all__ = [
    # Compat & Environment
    "BaseModule",
    "is_torch_available",
    "is_numpy_available",
    "is_triton_available",
    "is_rust_simd_available",
    # Normalization
    "RaNorm",
    "RaBatchNorm2d",
    # Activation & Gated Units
    "ra_tanh",
    "RaTanh",
    "ra_gelu",
    "RaGELU",
    "pos_ra_gate",
    "pos_ra_glu",
    "PosRaGLU",
    "RaGLU",
    "TriGLU",
    # Attention & Softmax
    "ra_softmax",
    "ra_attention",
    "flash_ra_attention",
    "RaAttention",
    "RaAttn",
    # Fused Linear + Cross-Entropy
    "fused_linear_cross_entropy",
    "FusedLinearCrossEntropy",
    "ra_linear_cross_entropy",
    "RaLinearCrossEntropy",
    "linear_cross_entropy",
    "LinearCrossEntropy",
    # Academic Aliases
    "L1Norm",
    "L1LayerNorm",
    "L1BatchNorm2d",
    "RationalTanh",
    "rational_tanh",
    "RationalGELU",
    "rational_gelu",
    "RationalGLU",
    "BoundedRationalGLU",
    "RationalAttention",
    "PadeAttention",
    "rational_attention",
    "pade_attention",
    "rational_softmax",
    "pade_softmax",
]
