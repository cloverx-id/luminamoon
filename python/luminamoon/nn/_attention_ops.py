from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple

from ._attention_gqa import (
    _accum_attn_mask_grad,
    _get_causal_mask,
    _gqa_matmul_do_vt,
    _gqa_matmul_ds_k,
    _gqa_matmul_dst_q,
    _gqa_matmul_pt_do,
    _gqa_matmul_pv,
    _gqa_matmul_qk,
    _slice_attn_mask,
)
from ._compat import _HAS_RUST_SIMD, _HAS_TORCH, _HAS_TRITON, _is_torch_compiling, _rs_engine, torch
from .kernels import triton_flash_ra_attention, triton_flash_ra_attention_backward
from .softmax import _MAX_CHUNK_ELEMENTS, _ra_softmax_backward_no_grad, _ra_softmax_forward_no_grad

_CUDA_FAST_ELEMENTS: int = 134_217_728
_CUDA_FAST_CACHE_ELEMENTS: int = 16_777_216
_CUDA_FAST_MAX_SEQ: int = 2048
_TRITON_ATTN_MIN_SEQ: int = 64
_WARNED_TRITON_ATTN_FALLBACK: bool = False


def _should_prefer_triton_attn(
    B: int,
    H: int,
    N_Q: int,
    N_K: int,
    q_chunk_size: Optional[int],
) -> bool:
    return bool(
        q_chunk_size is not None
        or N_Q >= _TRITON_ATTN_MIN_SEQ
        or N_K > _CUDA_FAST_MAX_SEQ
        or (B * H * N_Q * N_K > _CUDA_FAST_ELEMENTS)
    )


def _can_use_rust_attn_simd(tensor: Any, attn_mask: Optional[Any], fn_name: str) -> bool:
    return bool(
        _HAS_RUST_SIMD
        and attn_mask is None
        and not _is_torch_compiling()
        and tensor.device.type == "cpu"
        and tensor.dtype == torch.float32
        and hasattr(_rs_engine, fn_name)
    )


def _ra_attn_weights_fwd(
    scores: Any,
    ra_scale: float,
    *,
    q_start: int = 0,
    is_causal: bool = False,
    use_simd: bool = False,
) -> Tuple[Any, Any, Any, Any]:
    if use_simd:
        s_c = scores if scores.is_contiguous() else scores.contiguous()
        B, H, n_q, cols = s_c.shape
        rows = B * H * n_q
        m_val = torch.empty((B, H, n_q, 1), device=s_c.device, dtype=torch.float32)
        l_val = torch.empty((B, H, n_q, 1), device=s_c.device, dtype=torch.float32)
        m_idx = torch.empty((B, H, n_q, 1), device=s_c.device, dtype=torch.int64)
        # In-place scores -> p avoids allocating an extra [B, H, n_q, cols] buffer.
        _rs_engine.ra_attn_weights_fwd_f32_pyo3(
            s_c, s_c, m_val, l_val, m_idx, rows, cols, n_q, q_start, is_causal, ra_scale
        )
        return s_c, m_val, l_val, m_idx

    if is_causal:
        scores = scores.masked_fill(_get_causal_mask(q_start, q_start + scores.shape[-2], scores.shape[-1], scores.device), float("-inf"))
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
    *,
    q_start: int = 0,
    is_causal: bool = False,
    use_simd: bool = False,
) -> Tuple[Any, Any]:
    dp = _gqa_matmul_do_vt(do_gemm, v_gemm).to(work_dtype)
    if use_simd:
        s_c = scores if scores.is_contiguous() else scores.contiguous()
        dp_c = dp if dp.is_contiguous() else dp.contiguous()
        if dp_c.data_ptr() == s_c.data_ptr():
            dp_c = dp_c.clone()
        m_c = m_val if m_val.is_contiguous() else m_val.contiguous()
        l_c = l_val if l_val.is_contiguous() else l_val.contiguous()
        idx_c = m_idx if m_idx.is_contiguous() else m_idx.contiguous()
        d_c = d_val if d_val.is_contiguous() else d_val.contiguous()
        B, H, n_q, cols = s_c.shape
        rows = B * H * n_q
        # In-place scores -> p and dp -> ds eliminates two [B, H, n_q, cols] allocations.
        _rs_engine.ra_attn_scores_bwd_f32_pyo3(
            s_c, dp_c, m_c, l_c, idx_c, d_c, s_c, dp_c, rows, cols, n_q, q_start, is_causal, ra_scale
        )
        return s_c.to(gemm_dtype), dp_c

    if is_causal:
        scores = scores.masked_fill(_get_causal_mask(q_start, q_start + scores.shape[-2], scores.shape[-1], scores.device), float("-inf"))
    finite = torch.isfinite(scores)
    delta = torch.where(finite, torch.where(scores == m_val, 0.0, m_val - scores), float("inf"))
    u = delta * ra_scale
    active = finite & (u < 1.0)
    safe_u = torch.where(active, u, torch.zeros_like(u))

    one_plus_u = 1.0 + safe_u
    q_rat = torch.where(active, (1.0 - safe_u) / one_plus_u, torch.zeros_like(safe_u))
    q2 = q_rat * q_rat
    q4 = q2 * q2
    q7 = q_rat * q2 * q4
    p = torch.where(active, (q4 * q4) / l_val, torch.zeros_like(u))

    # Factored form avoids dividing by (1 - u^2) near the u -> 1^- boundary.
    pw = ((16.0 * ra_scale) / l_val) * q7 / (one_plus_u * one_plus_u)
    ds_direct = torch.where(active, pw * (dp - d_val), torch.zeros_like(p))
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

                # Prefer 2-pass Triton SRAM for sequence workloads (N_Q >= 64) to avoid O(N^2) HBM materialization.
                if (
                    _HAS_TRITON
                    and q.is_cuda
                    and attn_mask is None
                    and _should_prefer_triton_attn(B, H, N_Q, N_K, q_chunk_size)
                ):
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
                    and q_chunk_size is None
                    and orig_dtype in (torch.float16, torch.bfloat16, torch.float32)
                    and N_Q <= _CUDA_FAST_MAX_SEQ
                    and N_K <= _CUDA_FAST_MAX_SEQ
                    and (B * H * N_Q * N_K <= _CUDA_FAST_ELEMENTS)
                ):
                    q_scaled = q * sm_scale
                    s_full = _gqa_matmul_qk(q_scaled, k)
                    if is_causal:
                        s_full = s_full.masked_fill(_get_causal_mask(0, N_Q, N_K, q.device), float("-inf"))
                    p_full = _ra_softmax_forward_no_grad(s_full, dim=-1, scale=ra_scale)
                    out = _gqa_matmul_pv(p_full.to(orig_dtype), v)
                    if B * H * N_Q * N_K <= _CUDA_FAST_CACHE_ELEMENTS:
                        ctx.save_for_backward(q_scaled, k, v, s_full, p_full)
                        ctx.mode = "cuda_fast_cached"
                    else:
                        ctx.save_for_backward(q_scaled, k, v)
                        ctx.mode = "cuda_fast"
                    ctx.used_triton = False
                    ctx.is_causal = is_causal
                    ctx.sm_scale = sm_scale
                    ctx.ra_scale = ra_scale
                    ctx.q_chunk_size = None
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
                elif is_causal and not q.is_cuda and N_Q >= 384:
                    # Causal tiling skips upper-triangular GEMM FLOPs and stays L2-cache friendly.
                    chunk_sz = 128
                else:
                    budget = _CUDA_FAST_ELEMENTS if q.is_cuda else _MAX_CHUNK_ELEMENTS
                    chunk_sz = min(N_Q, max(16, budget // max(1, B * H * N_K)))

                use_simd_fwd = _can_use_rust_attn_simd(q, attn_mask, "ra_attn_weights_fwd_f32_pyo3")
                # Hoist sm_scale onto Q ([B, H, N_Q, D]) instead of scaling [B, H, N_Q, N_K] scores.
                q_scaled = q.to(work_dtype) * sm_scale if orig_dtype in (torch.float16, torch.bfloat16) else q * sm_scale
                q_for_qk = q_scaled.to(orig_dtype) if orig_dtype in (torch.float16, torch.bfloat16) else q_scaled
                if chunk_sz >= N_Q:
                    s_full = _gqa_matmul_qk(q_for_qk, k).to(work_dtype)
                    if attn_mask is not None:
                        s_full = s_full + _slice_attn_mask(attn_mask, 0, N_Q, N_K).to(work_dtype)
                    p_full, m_all, l_all, m_idx_all = _ra_attn_weights_fwd(
                        s_full, ra_scale, q_start=0, is_causal=is_causal, use_simd=use_simd_fwd
                    )
                    out = _gqa_matmul_pv(p_full.to(orig_dtype), v)
                else:
                    out = torch.empty((B, H, N_Q, D_V), device=q.device, dtype=orig_dtype)
                    m_all = torch.empty((B, H, N_Q, 1), device=q.device, dtype=work_dtype)
                    l_all = torch.empty((B, H, N_Q, 1), device=q.device, dtype=work_dtype)
                    m_idx_all = torch.empty((B, H, N_Q, 1), device=q.device, dtype=torch.int64)

                    for q_start in range(0, N_Q, chunk_sz):
                        q_end = min(N_Q, q_start + chunk_sz)
                        k_limit = min(N_K, q_end) if is_causal else N_K
                        q_chunk = q_for_qk[:, :, q_start:q_end, :]
                        k_sub, v_sub = k[:, :, :k_limit, :], v[:, :, :k_limit, :]
                        s_chunk = _gqa_matmul_qk(q_chunk, k_sub).to(work_dtype)
                        if attn_mask is not None:
                            s_chunk = s_chunk + _slice_attn_mask(attn_mask, q_start, q_end, k_limit).to(work_dtype)
                        p_chunk, m_chunk, l_chunk, m_idx_chunk = _ra_attn_weights_fwd(
                            s_chunk, ra_scale, q_start=q_start, is_causal=is_causal, use_simd=use_simd_fwd
                        )
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

                if mode in ("cuda_fast", "cuda_fast_cached"):
                    if mode == "cuda_fast_cached":
                        q_scaled, k, v, s_full, p_full = ctx.saved_tensors
                        H_kv, orig_dtype = k.shape[1], q_scaled.dtype
                    else:
                        q_scaled, k, v = ctx.saved_tensors
                        H_kv, N_Q, N_K, orig_dtype = k.shape[1], q_scaled.shape[2], k.shape[2], q_scaled.dtype
                        s_full = _gqa_matmul_qk(q_scaled, k)
                        if ctx.is_causal:
                            s_full = s_full.masked_fill(_get_causal_mask(0, N_Q, N_K, q_scaled.device), float("-inf"))
                        p_full = _ra_softmax_forward_no_grad(s_full, dim=-1, scale=ctx.ra_scale)
                    do_orig = grad_out.to(orig_dtype)
                    dv = _gqa_matmul_pt_do(p_full.to(orig_dtype), do_orig, H_kv)
                    dp_full = _gqa_matmul_do_vt(do_orig, v)
                    ds_full = _ra_softmax_backward_no_grad(s_full, p_full, dp_full, dim=-1, scale=ctx.ra_scale).to(orig_dtype)
                    dq = _gqa_matmul_ds_k(ds_full, k).mul_(ctx.sm_scale)
                    dk = _gqa_matmul_dst_q(ds_full, q_scaled, H_kv)
                    return dq, dk, dv, None, None, None, None, None

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
                q_scaled = (q.to(work_dtype) * sm_scale).to(gemm_dtype) if gemm_dtype != work_dtype else q_gemm * sm_scale

                use_simd_bwd = _can_use_rust_attn_simd(q, attn_mask, "ra_attn_scores_bwd_f32_pyo3")
                if chunk_sz >= N_Q:
                    s_full = _gqa_matmul_qk(q_scaled, k_gemm).to(work_dtype)
                    if attn_mask is not None:
                        s_full = s_full + _slice_attn_mask(attn_mask, 0, N_Q, N_K).to(work_dtype)
                    p_gemm, ds_direct = _ra_attn_scores_bwd(
                        s_full, m_all, l_all, m_idx_all, do_gemm, v_gemm, d_all, ra_scale, work_dtype, gemm_dtype,
                        q_start=0, is_causal=is_causal, use_simd=use_simd_bwd,
                    )
                    dv = _gqa_matmul_pt_do(p_gemm, do_gemm, H_kv)
                    if d_mask is not None:
                        _accum_attn_mask_grad(d_mask, ds_direct, 0, N_Q, N_K)
                    ds_gemm = ds_direct.to(gemm_dtype) if ds_direct.dtype != gemm_dtype else ds_direct
                    dq = _gqa_matmul_ds_k(ds_gemm, k_gemm).mul_(sm_scale)
                    dk = _gqa_matmul_dst_q(ds_gemm, q_scaled, H_kv)
                else:
                    dq = torch.empty_like(q, dtype=work_dtype)
                    dk = torch.zeros_like(k, dtype=work_dtype)
                    dv = torch.zeros_like(v, dtype=work_dtype)
                    for q_start in range(0, N_Q, chunk_sz):
                        q_end = min(N_Q, q_start + chunk_sz)
                        k_limit = min(N_K, q_end) if is_causal else N_K
                        q_chunk = q_scaled[:, :, q_start:q_end, :]
                        do_chunk = do_gemm[:, :, q_start:q_end, :]
                        k_sub, v_sub = k_gemm[:, :, :k_limit, :], v_gemm[:, :, :k_limit, :]
                        s_chunk = _gqa_matmul_qk(q_chunk, k_sub).to(work_dtype)
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
                            q_start=q_start, is_causal=is_causal, use_simd=use_simd_bwd,
                        )
                        dv[:, :, :k_limit, :].add_(_gqa_matmul_pt_do(p_gemm, do_chunk, H_kv).to(work_dtype))
                        if d_mask is not None:
                            _accum_attn_mask_grad(d_mask, ds_direct, q_start, q_end, k_limit)
                        ds_gemm = ds_direct.to(gemm_dtype) if ds_direct.dtype != gemm_dtype else ds_direct
                        dq[:, :, q_start:q_end, :] = _gqa_matmul_ds_k(ds_gemm, k_sub).to(work_dtype)
                        dk[:, :, :k_limit, :].add_(_gqa_matmul_dst_q(ds_gemm, q_chunk, H_kv).to(work_dtype))
                    dq.mul_(sm_scale)

                return (
                    dq.to(orig_dtype),
                    dk.to(orig_dtype),
                    dv.to(orig_dtype),
                    d_mask.to(attn_mask.dtype) if (d_mask is not None and attn_mask is not None) else None,
                    None, None, None, None,
                )

    _FlashRaAttentionFunction = _FlashRaAttentionFunctionImpl
