"""Triton Flash-RaAttention fused distance-bias and hub-masking autograd wrappers."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ._compat import _HAS_TORCH, _HAS_TRITON, _is_torch_compiling, torch
from .kernels import triton_flash_ra_attention, triton_flash_ra_attention_backward

_TRITON_ATTN_MIN_SEQ: int = 64

if _HAS_TORCH:

    class _FlashRaBiasAttentionFunctionImpl(torch.autograd.Function):
        """Triton Rational Attention autograd function with distance decay and hub masking computed in SRAM."""

        @staticmethod
        def forward(
            ctx: Any,
            q: Any,
            k: Any,
            v: Any,
            bias_slopes: Optional[Any],
            hub_penalty: Optional[Any],
            bias_mode: Optional[str],
            hub_size: int,
            q_offset: int,
            check_hub_q: bool,
            is_causal: bool,
            sm_scale: float,
            ra_scale: float,
        ) -> Any:
            with torch.no_grad():
                triton_res = triton_flash_ra_attention(
                    q,
                    k,
                    v,
                    is_causal=is_causal,
                    sm_scale=sm_scale,
                    ra_scale=ra_scale,
                    bias_slopes=bias_slopes,
                    hub_penalty=hub_penalty,
                    bias_mode=bias_mode,
                    hub_size=hub_size,
                    q_offset=q_offset,
                    check_hub_q=check_hub_q,
                )
                if triton_res is None:
                    raise RuntimeError("Triton fused bias Rational Attention forward kernel could not be launched.")
                out, m, l, m_idx = triton_res
                ctx.save_for_backward(q, k, v, out, m, l, m_idx, bias_slopes, hub_penalty)
                ctx.bias_mode = bias_mode
                ctx.hub_size = int(hub_size)
                ctx.q_offset = int(q_offset)
                ctx.check_hub_q = bool(check_hub_q)
                ctx.is_causal = bool(is_causal)
                ctx.sm_scale = float(sm_scale)
                ctx.ra_scale = float(ra_scale)
                return out

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Tuple[Any, ...]:  # type: ignore[override]
            with torch.no_grad():
                q, k, v, out, m, l, m_idx, bias_slopes, hub_penalty = ctx.saved_tensors
                triton_bwd = triton_flash_ra_attention_backward(
                    grad_out,
                    q,
                    k,
                    v,
                    out,
                    m,
                    l,
                    m_idx,
                    is_causal=ctx.is_causal,
                    sm_scale=ctx.sm_scale,
                    ra_scale=ctx.ra_scale,
                    bias_slopes=bias_slopes,
                    hub_penalty=hub_penalty,
                    bias_mode=ctx.bias_mode,
                    hub_size=ctx.hub_size,
                    q_offset=ctx.q_offset,
                    check_hub_q=ctx.check_hub_q,
                )
                if triton_bwd is None:
                    raise RuntimeError("Triton fused bias Rational Attention backward kernel could not be launched.")
                dq, dk, dv = triton_bwd
                return (dq, dk, dv, None, None, None, None, None, None, None, None, None)

    _FlashRaBiasAttentionFunction: Any = _FlashRaBiasAttentionFunctionImpl
else:
    _FlashRaBiasAttentionFunction = None


def can_use_triton_ra_fused_bias(
    q: Any,
    k: Any,
    v: Any,
    distance_bias: Optional[Any],
    *,
    hub_size: int = 0,
) -> bool:
    """Return True when CUDA Triton Flash-RaAttention can compute distance decay and hub masking in SRAM."""
    if not (
        _HAS_TORCH
        and _HAS_TRITON
        and _FlashRaBiasAttentionFunction is not None
        and isinstance(q, torch.Tensor)
        and isinstance(k, torch.Tensor)
        and isinstance(v, torch.Tensor)
        and q.is_cuda
        and k.device == q.device
        and v.device == q.device
        and q.ndim == 4
        and k.ndim == 4
        and v.ndim == 4
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and k.dtype == q.dtype
        and v.dtype == q.dtype
    ):
        return False
    if _is_torch_compiling() and not (hasattr(torch, "library") and hasattr(torch.library, "custom_op")):
        return False
    batch, h_q, n_q, d_head = q.shape
    h_kv = k.shape[1]
    n_k = k.shape[2]
    if not (
        batch > 0
        and h_q > 0
        and h_kv > 0
        and k.shape[0] == batch
        and v.shape[0] == batch
        and v.shape[1] == h_kv
        and v.shape[2] == n_k
        and (h_q % h_kv == 0)
        and n_q > 0
        and n_k > 0
        and d_head > 0
        and d_head <= 256
        and k.shape[-1] == d_head
        and v.shape[-1] == d_head
        and (n_q >= _TRITON_ATTN_MIN_SEQ or n_k >= 256)
    ):
        return False
    if distance_bias is not None:
        if bool(getattr(distance_bias, "learnable", False)):
            return False
        slopes = getattr(distance_bias, "slopes", None)
        if not (isinstance(slopes, torch.Tensor) and slopes.numel() == h_q):
            return False
        if int(hub_size) > 0:
            hub_pen = getattr(distance_bias, "hub_penalty", None)
            if not (isinstance(hub_pen, torch.Tensor) and hub_pen.numel() == h_q):
                return False
    return True


def ra_attention_fused_bias(
    q: Any,
    k: Any,
    v: Any,
    *,
    distance_bias: Optional[Any] = None,
    hub_size: int = 0,
    q_offset: int = 0,
    check_hub_q: bool = False,
    is_causal: bool = True,
    scale: Optional[float] = None,
    ra_scale: float = 0.0625,
) -> Any:
    """Run Triton Rational Attention with distance decay and hub masking computed in SRAM."""
    sm_scale = float(scale) if scale is not None else (1.0 / (q.shape[-1] ** 0.5))
    slopes = (
        distance_bias.slopes.reshape(-1).to(device=q.device, dtype=torch.float32)
        if distance_bias is not None
        else None
    )
    hub_pen = (
        distance_bias.hub_penalty.reshape(-1).to(device=q.device, dtype=torch.float32)
        if (distance_bias is not None and hub_size > 0)
        else None
    )
    bias_mode = str(distance_bias.mode) if distance_bias is not None else None
    if _is_torch_compiling():
        from ._custom_ops_bias import _HAS_CUSTOM_OP, compiled_ra_attention_bias

        if _HAS_CUSTOM_OP:
            return compiled_ra_attention_bias(
                q,
                k,
                v,
                bias_slopes=slopes,
                hub_penalty=hub_pen,
                bias_mode=bias_mode,
                hub_size=int(hub_size),
                q_offset=int(q_offset),
                check_hub_q=bool(check_hub_q),
                is_causal=bool(is_causal),
                sm_scale=sm_scale,
                ra_scale=float(ra_scale),
            )
    return _FlashRaBiasAttentionFunction.apply(
        q,
        k,
        v,
        slopes,
        hub_pen,
        bias_mode,
        int(hub_size),
        int(q_offset),
        bool(check_hub_q),
        bool(is_causal),
        sm_scale,
        float(ra_scale),
    )


__all__ = [
    "_FlashRaBiasAttentionFunction",
    "can_use_triton_ra_fused_bias",
    "ra_attention_fused_bias",
]
