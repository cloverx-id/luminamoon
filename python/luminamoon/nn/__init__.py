"""
LuminaMoon Neural Network Architecture Components (Ra-Family).

Rational, tanh-free and square-root-free neural components:
- RaNorm: Zero square-root, single-reciprocal token normalization.
- RaBatchNorm2d: Mean Absolute Deviation (MAD) 2D batch normalization.
- RaTanh & ra_tanh: Rational hyperbolic tangent with non-vanishing tail.
- RaGELU & ra_gelu: Rational GELU formulation for vision and language models.
- PosRaGLU: Strictly positive rational gate MLP block without activation explosion.
- RaGLU & TriGLU: Rational gated linear units.
- RaAttention & RaAttn: Multi-head attention using rational softmax without exponential SFU bottlenecks.
- ra_softmax: Cayley-Padé rational softmax with compact support.
"""

from __future__ import annotations

from ._compat import (
    BaseModule,
    is_numpy_available,
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
    ra_softmax,
)

__all__ = [
    # Compat & Environment
    "BaseModule",
    "is_torch_available",
    "is_numpy_available",
    "is_triton_available",
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
    "RaAttention",
    "RaAttn",
]
