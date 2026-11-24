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

_REMOVED_NN_MIGRATION: dict[str, str] = {
    "PosRaGLU": "Removed in v2.0.0. Use `RaSwiGLU` (general FFN) or `RaGLU` (hard-bounded FFN) instead.",
    "pos_ra_glu": "Removed in v2.0.0. Use `ra_swiglu` or `ra_glu` instead.",
    "TriGLU": "Removed in v2.0.0. Use `RaSwiGLU` instead.",
    "RaEMA": "Removed in v2.0.0. Use `RaAttention(qk_norm=True, is_causal=True)` instead.",
    "ra_ema": "Removed in v2.0.0. Use `ra_attention(..., is_causal=True)` instead.",
    "RationalMoE": "Renamed in v2.0.0. Use `RaMoE` instead.",
    "HybridRationalMoE": "Renamed in v2.0.0. Use `RaMoE(..., num_shared_experts=...)` instead.",
    "FusedLinearCrossEntropy": "Renamed in v2.0.0. Use `LinearCrossEntropy` instead.",
    "flash_ra_attention": "Renamed in v2.0.0. Use `ra_attention` instead.",
    "L1Norm": "Renamed in v2.0.0. Use `RaNorm` instead.",
    "L1LayerNorm": "Renamed in v2.0.0. Use `RaNorm` instead.",
    "L1BatchNorm2d": "Renamed in v2.0.0. Use `RaBatchNorm2d` instead.",
    "RationalTanh": "Renamed in v2.0.0. Use `RaTanh` instead.",
    "RationalGELU": "Renamed in v2.0.0. Use `RaGELU` instead.",
}


def __getattr__(name: str):
    hint = _REMOVED_NN_MIGRATION.get(name)
    if hint is not None:
        raise AttributeError(f"luminamoon.nn.{name}: {hint}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
