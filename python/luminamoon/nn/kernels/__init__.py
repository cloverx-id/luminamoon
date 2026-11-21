from __future__ import annotations

from ._base import HAS_TRITON
from .attention_bwd import triton_flash_ra_attention_backward
from .attention_fwd import triton_flash_ra_attention
from .ema import triton_fused_ra_ema_bwd, triton_fused_ra_ema_fwd
from .loss_ce_chunk import triton_fused_ce_chunk_bwd_inplace, triton_fused_ce_chunk_fwd
from .loss_linear_ce import (
    triton_fused_linear_cross_entropy,
    triton_fused_linear_cross_entropy_backward,
)
from .norm_act import (
    triton_pos_ra_glu,
    triton_pos_ra_glu_backward,
    triton_ra_norm,
    triton_ra_norm_backward,
)
from .softmax import triton_ra_softmax, triton_ra_softmax_backward

__all__ = [
    "HAS_TRITON",
    "triton_ra_norm",
    "triton_ra_norm_backward",
    "triton_pos_ra_glu",
    "triton_pos_ra_glu_backward",
    "triton_ra_softmax",
    "triton_ra_softmax_backward",
    "triton_flash_ra_attention",
    "triton_flash_ra_attention_backward",
    "triton_fused_linear_cross_entropy",
    "triton_fused_linear_cross_entropy_backward",
    "triton_fused_ce_chunk_fwd",
    "triton_fused_ce_chunk_bwd_inplace",
    "triton_fused_ra_ema_fwd",
    "triton_fused_ra_ema_bwd",
]
