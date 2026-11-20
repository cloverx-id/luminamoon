from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple

from ._compat import _HAS_TORCH, _HAS_TRITON, _is_torch_compiling, torch
from .kernels import triton_flash_ra_attention, triton_flash_ra_attention_backward
from .softmax import _MAX_CHUNK_ELEMENTS, _ra_softmax_backward_no_grad, _ra_softmax_forward_no_grad

_CUDA_FAST_ELEMENTS: int = 134_217_728
_CUDA_FAST_MAX_SEQ: int = 2048
_WARNED_TRITON_ATTN_FALLBACK: bool = False


def _gqa_matmul_qk(q: Any, k: Any) -> Any:
    h_q, h_kv = q.shape[1], k.shape[1]
    k_match = k.to(q.dtype) if hasattr(k, "dtype") and k.dtype != q.dtype else k
    if h_q == h_kv:
        return q @ k_match.transpose(-2, -1)
    B, _, N_Q, D = q.shape
    g = h_q // h_kv
    return (q.reshape(B, h_kv, g * N_Q, D) @ k_match.transpose(-2, -1)).reshape(B, h_q, N_Q, k.shape[2])


def _gqa_matmul_pv(p: Any, v: Any) -> Any:
    h_q, h_kv = p.shape[1], v.shape[1]
    v_match = v.to(p.dtype) if hasattr(v, "dtype") and v.dtype != p.dtype else v
    if h_q == h_kv:
        return p @ v_match
    B, _, N_Q, N_K = p.shape
    g = h_q // h_kv
    return (p.reshape(B, h_kv, g * N_Q, N_K) @ v_match).reshape(B, h_q, N_Q, v.shape[-1])


def _gqa_matmul_do_vt(do: Any, v: Any) -> Any:
    h_q, h_kv = do.shape[1], v.shape[1]
    v_match = v.to(do.dtype) if hasattr(v, "dtype") and v.dtype != do.dtype else v
    if h_q == h_kv:
        return do @ v_match.transpose(-2, -1)
    B, _, N_Q, D_V = do.shape
    g = h_q // h_kv
    return (do.reshape(B, h_kv, g * N_Q, D_V) @ v_match.transpose(-2, -1)).reshape(B, h_q, N_Q, v.shape[2])


def _gqa_matmul_pt_do(p: Any, do: Any, h_kv: int) -> Any:
    h_q = p.shape[1]
    do_match = do.to(p.dtype) if hasattr(do, "dtype") and do.dtype != p.dtype else do
    if h_q == h_kv:
        return p.transpose(-2, -1) @ do_match
    B, _, N_Q, N_K = p.shape
    g = h_q // h_kv
    return p.reshape(B, h_kv, g * N_Q, N_K).transpose(-2, -1) @ do_match.reshape(B, h_kv, g * N_Q, do.shape[-1])


def _gqa_matmul_ds_k(ds: Any, k: Any) -> Any:
    h_q, h_kv = ds.shape[1], k.shape[1]
    k_match = k.to(ds.dtype) if hasattr(k, "dtype") and k.dtype != ds.dtype else k
    if h_q == h_kv:
        return ds @ k_match
    B, _, N_Q, N_K = ds.shape
    g = h_q // h_kv
    return (ds.reshape(B, h_kv, g * N_Q, N_K) @ k_match).reshape(B, h_q, N_Q, k.shape[-1])


def _gqa_matmul_dst_q(ds: Any, q: Any, h_kv: int) -> Any:
    h_q = ds.shape[1]
    q_match = q.to(ds.dtype) if hasattr(q, "dtype") and q.dtype != ds.dtype else q
    if h_q == h_kv:
        return ds.transpose(-2, -1) @ q_match
    B, _, N_Q, N_K = ds.shape
    g = h_q // h_kv
    return ds.reshape(B, h_kv, g * N_Q, N_K).transpose(-2, -1) @ q_match.reshape(B, h_kv, g * N_Q, q.shape[-1])


def _slice_attn_mask(attn_mask: Any, q_start: int, q_end: int, k_limit: int) -> Any:
    if attn_mask.ndim == 2:
        m_view = attn_mask.unsqueeze(0).unsqueeze(0)
    elif attn_mask.ndim == 3:
        m_view = attn_mask.unsqueeze(1)
    else:
        m_view = attn_mask
    q_sl = slice(q_start, q_end) if m_view.shape[-2] > 1 else slice(0, 1)
    k_sl = slice(0, k_limit) if m_view.shape[-1] > 1 else slice(0, 1)
    return m_view[:, :, q_sl, k_sl]


def _accum_attn_mask_grad(
    d_mask: Any,
    ds_direct: Any,
    q_start: int,
    q_end: int,
    k_limit: int,
) -> None:
    if d_mask.ndim == 2:
        d_view = d_mask.unsqueeze(0).unsqueeze(0)
    elif d_mask.ndim == 3:
        d_view = d_mask.unsqueeze(1)
    else:
        d_view = d_mask
    reduce_dims = tuple(ax for ax in range(4) if d_view.shape[ax] == 1)
    grad_slice = ds_direct.sum(dim=reduce_dims, keepdim=True) if reduce_dims else ds_direct
    q_sl = slice(q_start, q_end) if d_view.shape[-2] > 1 else slice(0, 1)
    k_sl = slice(0, k_limit) if d_view.shape[-1] > 1 else slice(0, 1)
    d_view[:, :, q_sl, k_sl] += grad_slice


def _ra_attn_weights_fwd(scores: Any, ra_scale: float) -> Tuple[Any, Any, Any, Any]:
    finite = torch.isfinite(scores)
    m_val, m_idx = scores.max(dim=-1, keepdim=True)
    delta = torch.where(finite, torch.where(scores == m_val, 0.0, m_val - scores), float("inf"))
    u = delta * ra_scale
    num = (1.0 - u).clamp_(min=0.0)
    den = 1.0 + u
    q_rat = torch.where(finite, num.div_(den), torch.zeros_like(u))
    q_rat.square_()
    q_rat.square_()
    phi = q_rat.square_()
    l_val = phi.sum(dim=-1, keepdim=True).clamp(min=1e-38)
    return phi.div_(l_val), m_val, l_val, m_idx


def _ra_attn_scores_bwd(
    scores: Any,
    m_val: Any,
    l_val: Any,
    m_idx: Any,
    do_gemm: Any,
    v_gemm: Any,
    d_val: Any,
    ra_scale: float,
    work_dtype: Any,
    gemm_dtype: Any,
) -> Tuple[Any, Any]:
    finite = torch.isfinite(scores)
    delta = torch.where(finite, torch.where(scores == m_val, 0.0, m_val - scores), float("inf"))
    u = delta * ra_scale
    active = finite & (u < 1.0)
    safe_u = torch.where(active, u, torch.zeros_like(u))

    q_rat = (1.0 - safe_u).clamp(min=0.0) / (1.0 + safe_u)
    q2 = q_rat * q_rat
    q4 = q2 * q2
    p = torch.where(active, (q4 * q4) / l_val, torch.zeros_like(u))

    dp = _gqa_matmul_do_vt(do_gemm, v_gemm).to(work_dtype)
    w = (16.0 * ra_scale) / (1.0 - safe_u * safe_u)
    ds_direct = torch.where(active, p * (dp - d_val) * w, torch.zeros_like(p))
    sum_ds = ds_direct.sum(dim=-1, keepdim=True)
    ds_direct.scatter_add_(-1, m_idx, -sum_ds)
    return p.to(gemm_dtype), ds_direct


_FlashRaAttentionFunction: Any = None

if _HAS_TORCH:

    class _FlashRaAttentionFunctionImpl(torch.autograd.Function):
        """Fused Tiled Rational Attention (Flash-RaAttention) autograd function."""

        @staticmethod
        def forward(
            ctx: Any,
            q: Any,
            k: Any,
            v: Any,
            attn_mask: Optional[Any],
            is_causal: bool,
            sm_scale: float,
            ra_scale: float,
            q_chunk_size: Optional[int],
        ) -> Any:
            global _WARNED_TRITON_ATTN_FALLBACK
            with torch.no_grad():
                B, H, N_Q, _ = q.shape
                N_K = k.shape[2]
                D_V = v.shape[-1]
                orig_dtype = q.dtype

                if (
                    q.is_cuda
                    and attn_mask is None
                    and q_chunk_size is None
                    and orig_dtype in (torch.float16, torch.bfloat16, torch.float32)
                    and N_Q <= _CUDA_FAST_MAX_SEQ
                    and N_K <= _CUDA_FAST_MAX_SEQ
                    and (B * H * N_Q * N_K <= _CUDA_FAST_ELEMENTS)
                ):
                    s_full = _gqa_matmul_qk(q, k) * sm_scale
                    if is_causal:
                        q_idx = torch.arange(N_Q, device=q.device).unsqueeze(1)
                        k_idx = torch.arange(N_K, device=q.device).unsqueeze(0)
                        s_full = s_full.masked_fill(q_idx < k_idx, float("-inf"))
                    p_full = _ra_softmax_forward_no_grad(s_full, dim=-1, scale=ra_scale)
                    out = _gqa_matmul_pv(p_full.to(orig_dtype), v)
                    ctx.save_for_backward(q, k, v)
                    ctx.mode = "cuda_fast"
                    ctx.used_triton = False
                    ctx.is_causal = is_causal
                    ctx.sm_scale = sm_scale
                    ctx.ra_scale = ra_scale
                    ctx.q_chunk_size = None
                    return out

                if _HAS_TRITON and q.is_cuda and attn_mask is None:
                    triton_res = triton_flash_ra_attention(
                        q, k, v, is_causal=is_causal, sm_scale=sm_scale, ra_scale=ra_scale
                    )
                    if triton_res is not None:
                        out, m, l, m_idx = triton_res
                        ctx.save_for_backward(q, k, v, out, m, l, m_idx)
                        ctx.mode = "triton_sram"
                        ctx.used_triton = True
                        ctx.is_causal = is_causal
                        ctx.sm_scale = sm_scale
                        ctx.ra_scale = ra_scale
                        ctx.q_chunk_size = q_chunk_size
                        return out

                if (
                    q.is_cuda
                    and attn_mask is None
                    and not _WARNED_TRITON_ATTN_FALLBACK
                    and not _is_torch_compiling()
                ):
                    _WARNED_TRITON_ATTN_FALLBACK = True
                    warnings.warn(
                        "LuminaMoon RaAttention: Triton SRAM kernel is unavailable in this environment; "
                        "using dynamic fast-path / query-chunked PyTorch execution.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

                work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
                if q_chunk_size is not None and int(q_chunk_size) > 0:
                    chunk_sz = max(1, min(int(q_chunk_size), N_Q))
                else:
                    budget = _CUDA_FAST_ELEMENTS if q.is_cuda else _MAX_CHUNK_ELEMENTS
                    chunk_sz = min(N_Q, max(16, budget // max(1, B * H * N_K)))

                if chunk_sz >= N_Q:
                    s_full = _gqa_matmul_qk(q, k).to(work_dtype) * sm_scale
                    if is_causal:
                        q_idx = torch.arange(N_Q, device=q.device).unsqueeze(1)
                        k_idx = torch.arange(N_K, device=q.device).unsqueeze(0)
                        s_full = s_full.masked_fill(q_idx < k_idx, float("-inf"))
                    if attn_mask is not None:
                        s_full = s_full + _slice_attn_mask(attn_mask, 0, N_Q, N_K).to(work_dtype)
                    p_full, m_all, l_all, m_idx_all = _ra_attn_weights_fwd(s_full, ra_scale)
                    out = _gqa_matmul_pv(p_full.to(orig_dtype), v)
                else:
                    out = torch.empty((B, H, N_Q, D_V), device=q.device, dtype=orig_dtype)
                    m_all = torch.empty((B, H, N_Q, 1), device=q.device, dtype=work_dtype)
                    l_all = torch.empty((B, H, N_Q, 1), device=q.device, dtype=work_dtype)
                    m_idx_all = torch.empty((B, H, N_Q, 1), device=q.device, dtype=torch.int64)

                    for q_start in range(0, N_Q, chunk_sz):
                        q_end = min(N_Q, q_start + chunk_sz)
                        k_limit = min(N_K, q_end) if is_causal else N_K
                        q_chunk = q[:, :, q_start:q_end, :]
                        k_sub = k[:, :, :k_limit, :]
                        v_sub = v[:, :, :k_limit, :]

                        s_chunk = _gqa_matmul_qk(q_chunk, k_sub).to(work_dtype) * sm_scale
                        if is_causal:
                            q_idx = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)
                            k_idx = torch.arange(0, k_limit, device=q.device).unsqueeze(0)
                            s_chunk = s_chunk.masked_fill(q_idx < k_idx, float("-inf"))
                        if attn_mask is not None:
                            s_chunk = s_chunk + _slice_attn_mask(attn_mask, q_start, q_end, k_limit).to(work_dtype)

                        p_chunk, m_chunk, l_chunk, m_idx_chunk = _ra_attn_weights_fwd(s_chunk, ra_scale)
                        out[:, :, q_start:q_end, :] = _gqa_matmul_pv(p_chunk.to(orig_dtype), v_sub)
                        m_all[:, :, q_start:q_end, :] = m_chunk
                        l_all[:, :, q_start:q_end, :] = l_chunk
                        m_idx_all[:, :, q_start:q_end, :] = m_idx_chunk

                if attn_mask is not None:
                    ctx.save_for_backward(q, k, v, out, m_all, l_all, m_idx_all, attn_mask)
                    ctx.has_mask = True
                else:
                    ctx.save_for_backward(q, k, v, out, m_all, l_all, m_idx_all)
                    ctx.has_mask = False

                ctx.mode = "chunked"
                ctx.used_triton = False
                ctx.is_causal = is_causal
                ctx.sm_scale = sm_scale
                ctx.ra_scale = ra_scale
                ctx.q_chunk_size = chunk_sz
                return out

        @staticmethod
        def backward(  # type: ignore[override]
            ctx: Any,
            grad_out: Any,
        ) -> Tuple[Any, Any, Any, Optional[Any], None, None, None, None]:
            with torch.no_grad():
                mode = getattr(ctx, "mode", "triton_sram" if ctx.used_triton else "chunked")

                if mode == "cuda_fast":
                    q, k, v = ctx.saved_tensors
                    H_kv, N_Q, N_K, orig_dtype = k.shape[1], q.shape[2], k.shape[2], q.dtype
                    s_full = _gqa_matmul_qk(q, k) * ctx.sm_scale
                    if ctx.is_causal:
                        q_idx = torch.arange(N_Q, device=q.device).unsqueeze(1)
                        k_idx = torch.arange(N_K, device=q.device).unsqueeze(0)
                        s_full = s_full.masked_fill(q_idx < k_idx, float("-inf"))
                    p_full = _ra_softmax_forward_no_grad(s_full, dim=-1, scale=ctx.ra_scale)
                    do_orig = grad_out.to(orig_dtype)
                    dv = _gqa_matmul_pt_do(p_full.to(orig_dtype), do_orig, H_kv)
                    dp_full = _gqa_matmul_do_vt(do_orig, v)
                    ds_full = _ra_softmax_backward_no_grad(s_full, p_full, dp_full, dim=-1, scale=ctx.ra_scale)
                    ds_scaled = (ds_full * ctx.sm_scale).to(orig_dtype)
                    return _gqa_matmul_ds_k(ds_scaled, k), _gqa_matmul_dst_q(ds_scaled, q, H_kv), dv, None, None, None, None, None

                if ctx.used_triton:
                    q, k, v, out, m, l, m_idx = ctx.saved_tensors
                    triton_bwd = triton_flash_ra_attention_backward(
                        grad_out, q, k, v, out, m, l, m_idx,
                        is_causal=ctx.is_causal, sm_scale=ctx.sm_scale, ra_scale=ctx.ra_scale,
                    )
                    if triton_bwd is not None:
                        dq, dk, dv = triton_bwd
                        return dq, dk, dv, None, None, None, None, None

                if getattr(ctx, "has_mask", False):
                    q, k, v, out, m_all, l_all, m_idx_all, attn_mask = ctx.saved_tensors
                else:
                    q, k, v, out, m_all, l_all, m_idx_all = ctx.saved_tensors
                    attn_mask = None

                B, H, N_Q, _ = q.shape
                H_kv, N_K, orig_dtype = k.shape[1], k.shape[2], q.dtype
                work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
                gemm_dtype = orig_dtype if (q.is_cuda and orig_dtype in (torch.float16, torch.bfloat16)) else work_dtype

                if m_all.ndim == 3:
                    m_all = m_all.unsqueeze(-1).to(work_dtype)
                    l_all = l_all.unsqueeze(-1).to(work_dtype)
                    m_idx_all = m_idx_all.unsqueeze(-1).to(torch.int64)

                d_all = (grad_out.to(work_dtype) * out.to(work_dtype)).sum(dim=-1, keepdim=True)
                d_mask = (
                    torch.zeros_like(attn_mask, dtype=work_dtype)
                    if (attn_mask is not None and attn_mask.requires_grad)
                    else None
                )

                raw_chunk = ctx.q_chunk_size
                if raw_chunk is not None and int(raw_chunk) > 0:
                    chunk_sz = max(1, min(int(raw_chunk), N_Q))
                else:
                    budget = _CUDA_FAST_ELEMENTS if q.is_cuda else _MAX_CHUNK_ELEMENTS
                    chunk_sz = min(N_Q, max(16, budget // max(1, B * H * N_K)))

                is_causal, sm_scale, ra_scale = ctx.is_causal, ctx.sm_scale, ctx.ra_scale
                q_gemm, k_gemm, v_gemm, do_gemm = (
                    q.to(gemm_dtype), k.to(gemm_dtype), v.to(gemm_dtype), grad_out.to(gemm_dtype)
                )

                if chunk_sz >= N_Q:
                    s_full = _gqa_matmul_qk(q_gemm, k_gemm).to(work_dtype) * sm_scale
                    if is_causal:
                        q_idx = torch.arange(N_Q, device=q.device).unsqueeze(1)
                        k_idx = torch.arange(N_K, device=q.device).unsqueeze(0)
                        s_full = s_full.masked_fill(q_idx < k_idx, float("-inf"))
                    if attn_mask is not None:
                        s_full = s_full + _slice_attn_mask(attn_mask, 0, N_Q, N_K).to(work_dtype)

                    p_gemm, ds_direct = _ra_attn_scores_bwd(
                        s_full, m_all, l_all, m_idx_all, do_gemm, v_gemm, d_all, ra_scale, work_dtype, gemm_dtype
                    )
                    dv = _gqa_matmul_pt_do(p_gemm, do_gemm, H_kv)
                    if d_mask is not None:
                        _accum_attn_mask_grad(d_mask, ds_direct, 0, N_Q, N_K)
                    ds_scaled = (ds_direct * sm_scale).to(gemm_dtype)
                    dq = _gqa_matmul_ds_k(ds_scaled, k_gemm)
                    dk = _gqa_matmul_dst_q(ds_scaled, q_gemm, H_kv)
                else:
                    dq = torch.zeros_like(q, dtype=work_dtype)
                    dk = torch.zeros_like(k, dtype=work_dtype)
                    dv = torch.zeros_like(v, dtype=work_dtype)
                    for q_start in range(0, N_Q, chunk_sz):
                        q_end = min(N_Q, q_start + chunk_sz)
                        k_limit = min(N_K, q_end) if is_causal else N_K
                        q_chunk = q_gemm[:, :, q_start:q_end, :]
                        do_chunk = do_gemm[:, :, q_start:q_end, :]
                        k_sub, v_sub = k_gemm[:, :, :k_limit, :], v_gemm[:, :, :k_limit, :]

                        s_chunk = _gqa_matmul_qk(q_chunk, k_sub).to(work_dtype) * sm_scale
                        if is_causal:
                            q_idx = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)
                            k_idx = torch.arange(0, k_limit, device=q.device).unsqueeze(0)
                            s_chunk = s_chunk.masked_fill(q_idx < k_idx, float("-inf"))
                        if attn_mask is not None:
                            s_chunk = s_chunk + _slice_attn_mask(attn_mask, q_start, q_end, k_limit).to(work_dtype)

                        p_gemm, ds_direct = _ra_attn_scores_bwd(
                            s_chunk,
                            m_all[:, :, q_start:q_end, :],
                            l_all[:, :, q_start:q_end, :],
                            m_idx_all[:, :, q_start:q_end, :],
                            do_chunk, v_sub,
                            d_all[:, :, q_start:q_end, :],
                            ra_scale, work_dtype, gemm_dtype,
                        )
                        dv[:, :, :k_limit, :] += _gqa_matmul_pt_do(p_gemm, do_chunk, H_kv).to(work_dtype)
                        if d_mask is not None:
                            _accum_attn_mask_grad(d_mask, ds_direct, q_start, q_end, k_limit)
                        ds_scaled = (ds_direct * sm_scale).to(gemm_dtype)
                        dq[:, :, q_start:q_end, :] = _gqa_matmul_ds_k(ds_scaled, k_sub).to(work_dtype)
                        dk[:, :, :k_limit, :] += _gqa_matmul_dst_q(ds_scaled, q_chunk, H_kv).to(work_dtype)

                return (
                    dq.to(orig_dtype),
                    dk.to(orig_dtype),
                    dv.to(orig_dtype),
                    d_mask.to(attn_mask.dtype) if (d_mask is not None and attn_mask is not None) else None,
                    None, None, None, None,
                )

    _FlashRaAttentionFunction = _FlashRaAttentionFunctionImpl
