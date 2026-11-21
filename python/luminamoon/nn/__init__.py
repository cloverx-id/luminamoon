"""LuminaMoon Neural Network Architecture Components (Ra-Family)."""

from __future__ import annotations

from . import activation, attention, loss, moe, norm, softmax
from ._compat import (
    BaseModule,
    is_cubecl_available,
    is_numpy_available,
    is_rust_simd_available,
    is_torch_available,
    is_triton_available,
)
from .activation import (
    RaGELU,
    RaGLU,
    RaSwiGLU,
    RaTanh,
    pos_ra_gate,
    ra_gelu,
    ra_glu,
    ra_swiglu,
    ra_tanh,
)
from .attention import RaAttention, ra_attention
from .loss import LinearCrossEntropy, linear_cross_entropy
from .moe import RaMoE, permute_tokens_by_expert
from .norm import RaBatchNorm2d, RaNorm
from .softmax import ra_softmax

__all__ = [
    # Submodules
    "activation",
    "attention",
    "loss",
    "moe",
    "norm",
    "softmax",
    # Environment
    "BaseModule",
    "is_torch_available",
    "is_numpy_available",
    "is_triton_available",
    "is_rust_simd_available",
    "is_cubecl_available",
    # Normalization
    "RaNorm",
    "RaBatchNorm2d",
    # Activation & Gated Units
    "ra_tanh",
    "RaTanh",
    "ra_gelu",
    "RaGELU",
    "pos_ra_gate",
    "ra_swiglu",
    "RaSwiGLU",
    "ra_glu",
    "RaGLU",
    # Softmax & Attention
    "ra_softmax",
    "ra_attention",
    "RaAttention",
    # Loss
    "linear_cross_entropy",
    "LinearCrossEntropy",
    # Mixture-of-Experts
    "RaMoE",
    "permute_tokens_by_expert",
]
