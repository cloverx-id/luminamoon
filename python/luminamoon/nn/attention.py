from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Tuple, Union

from ._compat import (
    BaseModule,
    _HAS_NUMPY,
    _HAS_RUST_SIMD,
    _HAS_TORCH,
    _HAS_TRITON,
    _is_torch_compiling,
    _rs_engine,
    np,
    torch,
    nn,
)
from .activation import pos_ra_gate
from .norm import RaNorm
from .triton_kernels import (
    triton_flash_ra_attention,
    triton_flash_ra_attention_backward,
    triton_fused_ra_ema_bwd,
    triton_fused_ra_ema_fwd,
    triton_ra_softmax,
    triton_ra_softmax_backward,
)

# Maximum elements per temporary chunk buffer (~16 MB in FP32) to prevent CPU / L2 spikes
_MAX_CHUNK_ELEMENTS: int = 4_194_304
# CUDA TensorCore cuBLAS + 1D Triton ra_softmax fast-path budget (e.g. B=4, H=8, T=2048 -> 134M elements)
_CUDA_FAST_ELEMENTS: int = 134_217_728
_CUDA_FAST_MAX_SEQ: int = 2048
_WARNED_TRITON_ATTN_FALLBACK: bool = False


def _gqa_matmul_qk(q: Any, k: Any) -> Any:
    """
    Computes Q @ K^T -> (B, H_q, N_Q, N_K) supporting Grouped-Query Attention (H_q % H_kv == 0)
    via zero-copy 4D batched GEMM views without `repeat_interleave`.
    """
    h_q = q.shape[1]
    h_kv = k.shape[1]
    k_match = k.to(q.dtype) if hasattr(k, "dtype") and k.dtype != q.dtype else k
    if h_q == h_kv:
        return q @ k_match.transpose(-2, -1)
    B, _, N_Q, D = q.shape
    N_K = k.shape[2]
    g = h_q // h_kv
    q_grouped = q.reshape(B, h_kv, g * N_Q, D)
    scores_grouped = q_grouped @ k_match.transpose(-2, -1)
    return scores_grouped.reshape(B, h_q, N_Q, N_K)


def _gqa_matmul_pv(p: Any, v: Any) -> Any:
    """
    Computes P @ V -> (B, H_q, N_Q, D_V) supporting Grouped-Query Attention (H_q % H_kv == 0)
    via zero-copy 4D batched GEMM views without `repeat_interleave`.
    """
    h_q = p.shape[1]
    h_kv = v.shape[1]
    v_match = v.to(p.dtype) if hasattr(v, "dtype") and v.dtype != p.dtype else v
    if h_q == h_kv:
        return p @ v_match
    B, _, N_Q, N_K = p.shape
    D_V = v.shape[-1]
    g = h_q // h_kv
    p_grouped = p.reshape(B, h_kv, g * N_Q, N_K)
    out_grouped = p_grouped @ v_match
    return out_grouped.reshape(B, h_q, N_Q, D_V)


def _gqa_matmul_do_vt(do: Any, v: Any) -> Any:
    """
    Computes dO @ V^T -> (B, H_q, N_Q, N_K) supporting Grouped-Query Attention (H_q % H_kv == 0).
    """
    h_q = do.shape[1]
    h_kv = v.shape[1]
    v_match = v.to(do.dtype) if hasattr(v, "dtype") and v.dtype != do.dtype else v
    if h_q == h_kv:
        return do @ v_match.transpose(-2, -1)
    B, _, N_Q, D_V = do.shape
    N_K = v.shape[2]
    g = h_q // h_kv
    do_grouped = do.reshape(B, h_kv, g * N_Q, D_V)
    dp_grouped = do_grouped @ v_match.transpose(-2, -1)
    return dp_grouped.reshape(B, h_q, N_Q, N_K)


def _gqa_matmul_pt_do(p: Any, do: Any, h_kv: int) -> Any:
    """
    Computes dV = sum_{g} P_g^T @ dO_g -> (B, H_kv, N_K, D_V) in a single batched GEMM
    by contracting across the joint (G * N_Q) dimension.
    """
    h_q = p.shape[1]
    do_match = do.to(p.dtype) if hasattr(do, "dtype") and do.dtype != p.dtype else do
    if h_q == h_kv:
        return p.transpose(-2, -1) @ do_match
    B, _, N_Q, N_K = p.shape
    D_V = do.shape[-1]
    g = h_q // h_kv
    p_grouped = p.reshape(B, h_kv, g * N_Q, N_K)
    do_grouped = do_match.reshape(B, h_kv, g * N_Q, D_V)
    return p_grouped.transpose(-2, -1) @ do_grouped


def _gqa_matmul_ds_k(ds: Any, k: Any) -> Any:
    """
    Computes dQ = dS @ K -> (B, H_q, N_Q, D) supporting Grouped-Query Attention (H_q % H_kv == 0).
    """
    h_q = ds.shape[1]
    h_kv = k.shape[1]
    k_match = k.to(ds.dtype) if hasattr(k, "dtype") and k.dtype != ds.dtype else k
    if h_q == h_kv:
        return ds @ k_match
    B, _, N_Q, N_K = ds.shape
    D = k.shape[-1]
    g = h_q // h_kv
    ds_grouped = ds.reshape(B, h_kv, g * N_Q, N_K)
    dq_grouped = ds_grouped @ k_match
    return dq_grouped.reshape(B, h_q, N_Q, D)


def _gqa_matmul_dst_q(ds: Any, q: Any, h_kv: int) -> Any:
    """
    Computes dK = sum_{g} dS_g^T @ Q_g -> (B, H_kv, N_K, D) in a single batched GEMM
    by contracting across the joint (G * N_Q) dimension.
    """
    h_q = ds.shape[1]
    q_match = q.to(ds.dtype) if hasattr(q, "dtype") and q.dtype != ds.dtype else q
    if h_q == h_kv:
        return ds.transpose(-2, -1) @ q_match
    B, _, N_Q, N_K = ds.shape
    D = q.shape[-1]
    g = h_q // h_kv
    ds_grouped = ds.reshape(B, h_kv, g * N_Q, N_K)
    q_grouped = q_match.reshape(B, h_kv, g * N_Q, D)
    return ds_grouped.transpose(-2, -1) @ q_grouped


def _ra_softmax_chunk_fwd(s_chunk: Any, dim: int, scale: float) -> Any:
    """Computes Cayley-Padé rational softmax on a single chunk without building an autograd graph."""
    finite = torch.isfinite(s_chunk)
    max_s = s_chunk.max(dim=dim, keepdim=True)[0]
    delta = torch.where(finite, torch.where(s_chunk == max_s, 0.0, max_s - s_chunk), float("inf"))
    u = delta * scale
    u.mul_(-1.0).add_(1.0).clamp_(min=0.0)  # u now holds num = clamp(1 - delta*scale, min=0)
    den = delta.mul_(scale).add_(1.0)
    q = torch.where(finite, u.div_(den), torch.zeros_like(u))
    q.square_()
    q.square_()
    phi = q.square_()
    sum_phi = phi.sum(dim=dim, keepdim=True).clamp(min=1e-38)
    return phi.div_(sum_phi)


def _ra_softmax_chunk_bwd(s_chunk: Any, p_chunk: Any, g_chunk: Any, dim: int, scale: float) -> Any:
    """
    Computes exact closed-form backward gradient of Cayley-Padé rational softmax:
        d(phi)/du = -16 / (1 - u^2) * phi   for 0 <= u < 1
        dS_direct = p * (g - sum(g * p)) * (16 * scale / (1 - u^2)) * I(u < 1)
        dS = dS_direct - I(j == argmax) * sum(dS_direct)
    Requires zero intermediate power buffers (q, q2, q4, phi).
    """
    orig_dtype = s_chunk.dtype
    s_f32 = s_chunk.to(torch.float32) if orig_dtype in (torch.float16, torch.bfloat16) else s_chunk
    p_f32 = p_chunk.to(torch.float32) if orig_dtype in (torch.float16, torch.bfloat16) else p_chunk
    g_f32 = g_chunk.to(torch.float32) if orig_dtype in (torch.float16, torch.bfloat16) else g_chunk

    finite = torch.isfinite(s_f32)
    max_s, max_idx = s_f32.max(dim=dim, keepdim=True)
    delta = torch.where(finite, torch.where(s_f32 == max_s, 0.0, max_s - s_f32), float("inf"))
    u = delta * scale
    active = finite & (u < 1.0)
    safe_u = torch.where(active, u, torch.zeros_like(u))

    sum_gp = (g_f32 * p_f32).sum(dim=dim, keepdim=True)
    w = (16.0 * scale) / (1.0 - safe_u * safe_u)
    ds_direct = torch.where(active, p_f32 * (g_f32 - sum_gp) * w, torch.zeros_like(p_f32))

    sum_ds = ds_direct.sum(dim=dim, keepdim=True)
    ds_direct.scatter_add_(dim, max_idx, -sum_ds)
    return ds_direct.to(orig_dtype) if orig_dtype in (torch.float16, torch.bfloat16) else ds_direct


def _slice_attn_mask(attn_mask: Any, q_start: int, q_end: int, k_limit: int) -> Any:
    """Slices a 2D, 3D, or 4D broadcastable attention mask for query chunk [q_start:q_end] and keys [:k_limit]."""
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
    """Accumulates chunk score gradients into a 2D, 3D, or 4D broadcastable attention mask gradient tensor."""
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


def _ra_softmax_forward_no_grad(scores: Any, dim: int, scale: float) -> Any:
    """Memory-bounded forward execution of ra_softmax outside autograd."""
    ndim = scores.ndim
    norm_dim = dim if dim >= 0 else ndim + dim

    # 1a. Triton GPU fast-path
    if (
        _HAS_TRITON
        and scores.is_cuda
        and norm_dim == ndim - 1
        and scores.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        triton_p = triton_ra_softmax(scores, scale=scale)
        if triton_p is not None:
            return triton_p

    # 1b. Rust CPU SIMD fast-path
    if (
        _HAS_RUST_SIMD
        and not _is_torch_compiling()
        and not scores.is_cuda
        and scores.dtype == torch.float32
        and scores.is_contiguous()
        and norm_dim == ndim - 1
    ):
        out = torch.empty_like(scores)
        cols = scores.shape[-1]
        rows = scores.numel() // cols
        _rs_engine.ra_softmax_f32_pyo3(scores, out, rows, cols, scale)
        return out

    # 1c. Row-chunked in-place PyTorch path (prevents multi-GB intermediate allocations)
    if norm_dim == ndim - 1 and scores.numel() > _MAX_CHUNK_ELEMENTS:
        cols = scores.shape[-1]
        rows = scores.numel() // cols
        flat_in = scores.reshape(rows, cols)
        flat_out = torch.empty_like(flat_in)
        chunk_rows = max(1, _MAX_CHUNK_ELEMENTS // max(1, cols))
        for r0 in range(0, rows, chunk_rows):
            r1 = min(rows, r0 + chunk_rows)
            flat_out[r0:r1] = _ra_softmax_chunk_fwd(flat_in[r0:r1], dim=-1, scale=scale)
        return flat_out.reshape(scores.shape)

    return _ra_softmax_chunk_fwd(scores, dim=norm_dim, scale=scale)


def _ra_softmax_backward_no_grad(scores: Any, out: Any, grad_out: Any, dim: int, scale: float) -> Any:
    """Memory-bounded backward execution of ra_softmax using closed-form analytical derivative."""
    ndim = scores.ndim
    norm_dim = dim if dim >= 0 else ndim + dim

    # 1a. Triton GPU fast-path
    if (
        _HAS_TRITON
        and scores.is_cuda
        and norm_dim == ndim - 1
        and scores.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        triton_ds = triton_ra_softmax_backward(scores, out, grad_out, scale=scale)
        if triton_ds is not None:
            return triton_ds

    # 1b. Row-chunked closed-form PyTorch path
    if norm_dim == ndim - 1 and scores.numel() > _MAX_CHUNK_ELEMENTS:
        cols = scores.shape[-1]
        rows = scores.numel() // cols
        flat_s = scores.reshape(rows, cols)
        flat_p = out.reshape(rows, cols)
        flat_g = grad_out.reshape(rows, cols)
        flat_ds = torch.empty_like(flat_s)
        chunk_rows = max(1, _MAX_CHUNK_ELEMENTS // max(1, cols))
        for r0 in range(0, rows, chunk_rows):
            r1 = min(rows, r0 + chunk_rows)
            flat_ds[r0:r1] = _ra_softmax_chunk_bwd(
                flat_s[r0:r1],
                flat_p[r0:r1],
                flat_g[r0:r1],
                dim=-1,
                scale=scale,
            )
        return flat_ds.reshape(scores.shape)

    return _ra_softmax_chunk_bwd(scores, out, grad_out, dim=norm_dim, scale=scale)


if _HAS_TORCH:
    class _RaSoftmaxFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, scores: Any, dim: int, scale: float) -> Any:
            with torch.no_grad():
                out = _ra_softmax_forward_no_grad(scores, dim=dim, scale=scale)
            ctx.save_for_backward(scores, out)
            ctx.dim = dim
            ctx.scale = scale
            return out

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Tuple[Any, None, None]:
            scores, out = ctx.saved_tensors
            with torch.no_grad():
                grad_scores = _ra_softmax_backward_no_grad(
                    scores,
                    out,
                    grad_out,
                    dim=ctx.dim,
                    scale=ctx.scale,
                )
            return grad_scores, None, None

    class _FlashRaAttentionFunction(torch.autograd.Function):
        """
        Fused Tiled Rational Attention (Flash-RaAttention) Autograd Function.

        Supports both Multi-Head Attention (H_q == H_kv) and Grouped-Query Attention (H_q % H_kv == 0):
          - Tier 1A (CUDA TensorCore cuBLAS + 1D Triton ra_softmax Fast-Path): When N_Q, N_K <= 2048
            and B * H_q * N_Q * N_K <= 134M elements without persistent O(N^2) autograd tensors.
          - Tier 1B (CUDA 2D Triton SRAM Tiled Kernel): Executes 2-pass SRAM tiling with Tensor Core
            tl.dot MMA, compact-support block skipping, and 1-pass dQ backward for long contexts.
          - Tier 2 (Universal Fallback): Executes dynamic fast-path or query-chunked L2 tiling with
            closed-form analytical backward recomputation and native-dtype CUDA GEMMs.
        """

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
                H_kv = k.shape[1]
                N_K = k.shape[2]
                D_V = v.shape[-1]
                orig_dtype = q.dtype

                # Tier 1A: CUDA TensorCore cuBLAS + 1D Triton ra_softmax Fast-Path for short/medium contexts
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

                # Tier 1B: 2D Triton SRAM Tiled Kernel (long contexts or explicit SRAM tiling on CUDA)
                if _HAS_TRITON and q.is_cuda and attn_mask is None:
                    triton_res = triton_flash_ra_attention(
                        q,
                        k,
                        v,
                        is_causal=is_causal,
                        sm_scale=sm_scale,
                        ra_scale=ra_scale,
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

                # Tier 2: Dynamic Fast-Path & Query-Chunked Execution
                work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype

                if q_chunk_size is not None and int(q_chunk_size) > 0:
                    chunk_sz = max(1, min(int(q_chunk_size), N_Q))
                else:
                    budget = _CUDA_FAST_ELEMENTS if q.is_cuda else _MAX_CHUNK_ELEMENTS
                    max_q_rows = max(16, budget // max(1, B * H * N_K))
                    chunk_sz = min(N_Q, max_q_rows)

                # Fast-Path: Single-shot vectorized GEMM when attention matrix fits within budget
                if chunk_sz >= N_Q:
                    s_full = _gqa_matmul_qk(q, k).to(work_dtype) * sm_scale
                    if is_causal:
                        q_idx = torch.arange(N_Q, device=q.device).unsqueeze(1)
                        k_idx = torch.arange(N_K, device=q.device).unsqueeze(0)
                        s_full = s_full.masked_fill(q_idx < k_idx, float("-inf"))
                    if attn_mask is not None:
                        m_slice = _slice_attn_mask(attn_mask, 0, N_Q, N_K)
                        s_full = s_full + m_slice.to(work_dtype)

                    finite = torch.isfinite(s_full)
                    m_all, m_idx_all = s_full.max(dim=-1, keepdim=True)
                    delta = torch.where(finite, torch.where(s_full == m_all, 0.0, m_all - s_full), float("inf"))
                    u = delta * ra_scale
                    num = (1.0 - u).clamp_(min=0.0)
                    den = 1.0 + u
                    q_rat = torch.where(finite, num.div_(den), torch.zeros_like(u))
                    q_rat.square_()
                    q_rat.square_()
                    phi = q_rat.square_()
                    l_all = phi.sum(dim=-1, keepdim=True).clamp(min=1e-38)
                    p_full = phi.div_(l_all)
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
                            m_slice = _slice_attn_mask(attn_mask, q_start, q_end, k_limit)
                            s_chunk = s_chunk + m_slice.to(work_dtype)

                        finite = torch.isfinite(s_chunk)
                        m_chunk, m_idx_chunk = s_chunk.max(dim=-1, keepdim=True)
                        delta = torch.where(finite, torch.where(s_chunk == m_chunk, 0.0, m_chunk - s_chunk), float("inf"))
                        u = delta * ra_scale

                        num = (1.0 - u).clamp_(min=0.0)
                        den = 1.0 + u
                        q_rat = torch.where(finite, num.div_(den), torch.zeros_like(u))
                        q_rat.square_()
                        q_rat.square_()
                        phi = q_rat.square_()

                        l_chunk = phi.sum(dim=-1, keepdim=True).clamp(min=1e-38)
                        p_chunk = phi.div_(l_chunk)

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
        def backward(
            ctx: Any,
            grad_out: Any,
        ) -> Tuple[Any, Any, Any, Optional[Any], None, None, None, None]:
            with torch.no_grad():
                mode = getattr(ctx, "mode", "triton_sram" if ctx.used_triton else "chunked")

                if mode == "cuda_fast":
                    q, k, v = ctx.saved_tensors
                    H_kv = k.shape[1]
                    N_Q = q.shape[2]
                    N_K = k.shape[2]
                    orig_dtype = q.dtype

                    s_full = _gqa_matmul_qk(q, k) * ctx.sm_scale
                    if ctx.is_causal:
                        q_idx = torch.arange(N_Q, device=q.device).unsqueeze(1)
                        k_idx = torch.arange(N_K, device=q.device).unsqueeze(0)
                        s_full = s_full.masked_fill(q_idx < k_idx, float("-inf"))

                    p_full = _ra_softmax_forward_no_grad(s_full, dim=-1, scale=ctx.ra_scale)
                    do_orig = grad_out.to(orig_dtype)
                    dv = _gqa_matmul_pt_do(p_full.to(orig_dtype), do_orig, H_kv)
                    dp_full = _gqa_matmul_do_vt(do_orig, v)
                    ds_full = _ra_softmax_backward_no_grad(
                        s_full,
                        p_full,
                        dp_full,
                        dim=-1,
                        scale=ctx.ra_scale,
                    )
                    ds_scaled = (ds_full * ctx.sm_scale).to(orig_dtype)
                    dq = _gqa_matmul_ds_k(ds_scaled, k)
                    dk = _gqa_matmul_dst_q(ds_scaled, q, H_kv)
                    return dq, dk, dv, None, None, None, None, None

                if ctx.used_triton:
                    q, k, v, out, m, l, m_idx = ctx.saved_tensors
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
                H_kv = k.shape[1]
                N_K = k.shape[2]
                orig_dtype = q.dtype
                work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
                gemm_dtype = orig_dtype if (q.is_cuda and orig_dtype in (torch.float16, torch.bfloat16)) else work_dtype

                # If fallback triggered from a Triton forward tensor shape (B, H, N_Q), unsqueeze trailing dim
                if m_all.ndim == 3:
                    m_all = m_all.unsqueeze(-1).to(work_dtype)
                    l_all = l_all.unsqueeze(-1).to(work_dtype)
                    m_idx_all = m_idx_all.unsqueeze(-1).to(torch.int64)

                do_work = grad_out.to(work_dtype)
                out_work = out.to(work_dtype)
                d_all = (do_work * out_work).sum(dim=-1, keepdim=True)

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
                    max_q_rows = max(16, budget // max(1, B * H * N_K))
                    chunk_sz = min(N_Q, max_q_rows)

                is_causal = ctx.is_causal
                sm_scale = ctx.sm_scale
                ra_scale = ctx.ra_scale

                q_gemm = q.to(gemm_dtype)
                k_gemm = k.to(gemm_dtype)
                v_gemm = v.to(gemm_dtype)
                do_gemm = grad_out.to(gemm_dtype)

                # Fast-Path: Single-shot vectorized backward when attention matrix fits within budget
                if chunk_sz >= N_Q:
                    s_full = _gqa_matmul_qk(q_gemm, k_gemm).to(work_dtype) * sm_scale
                    if is_causal:
                        q_idx = torch.arange(N_Q, device=q.device).unsqueeze(1)
                        k_idx = torch.arange(N_K, device=q.device).unsqueeze(0)
                        s_full = s_full.masked_fill(q_idx < k_idx, float("-inf"))
                    if attn_mask is not None:
                        m_slice = _slice_attn_mask(attn_mask, 0, N_Q, N_K)
                        s_full = s_full + m_slice.to(work_dtype)

                    finite = torch.isfinite(s_full)
                    delta = torch.where(finite, torch.where(s_full == m_all, 0.0, m_all - s_full), float("inf"))
                    u = delta * ra_scale
                    active = finite & (u < 1.0)
                    safe_u = torch.where(active, u, torch.zeros_like(u))

                    num = (1.0 - safe_u).clamp(min=0.0)
                    den = 1.0 + safe_u
                    q_rat = num / den
                    q2 = q_rat * q_rat
                    q4 = q2 * q2
                    p_full = torch.where(active, (q4 * q4) / l_all, torch.zeros_like(u))

                    dv = _gqa_matmul_pt_do(p_full.to(gemm_dtype), do_gemm, H_kv)
                    dp_full = _gqa_matmul_do_vt(do_gemm, v_gemm).to(work_dtype)
                    w = (16.0 * ra_scale) / (1.0 - safe_u * safe_u)
                    ds_direct = torch.where(active, p_full * (dp_full - d_all) * w, torch.zeros_like(p_full))

                    sum_ds = ds_direct.sum(dim=-1, keepdim=True)
                    ds_direct.scatter_add_(-1, m_idx_all, -sum_ds)

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
                        k_sub = k_gemm[:, :, :k_limit, :]
                        v_sub = v_gemm[:, :, :k_limit, :]

                        m_chunk = m_all[:, :, q_start:q_end, :]
                        l_chunk = l_all[:, :, q_start:q_end, :]
                        m_idx_chunk = m_idx_all[:, :, q_start:q_end, :]
                        d_chunk = d_all[:, :, q_start:q_end, :]

                        s_chunk = _gqa_matmul_qk(q_chunk, k_sub).to(work_dtype) * sm_scale

                        if is_causal:
                            q_idx = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)
                            k_idx = torch.arange(0, k_limit, device=q.device).unsqueeze(0)
                            s_chunk = s_chunk.masked_fill(q_idx < k_idx, float("-inf"))

                        if attn_mask is not None:
                            m_slice = _slice_attn_mask(attn_mask, q_start, q_end, k_limit)
                            s_chunk = s_chunk + m_slice.to(work_dtype)

                        finite = torch.isfinite(s_chunk)
                        delta = torch.where(finite, torch.where(s_chunk == m_chunk, 0.0, m_chunk - s_chunk), float("inf"))
                        u = delta * ra_scale
                        active = finite & (u < 1.0)
                        safe_u = torch.where(active, u, torch.zeros_like(u))

                        num = (1.0 - safe_u).clamp(min=0.0)
                        den = 1.0 + safe_u
                        q_rat = num / den
                        q2 = q_rat * q_rat
                        q4 = q2 * q2
                        p_chunk = torch.where(active, (q4 * q4) / l_chunk, torch.zeros_like(u))

                        dv[:, :, :k_limit, :] += _gqa_matmul_pt_do(p_chunk.to(gemm_dtype), do_chunk, H_kv).to(work_dtype)

                        dp_chunk = _gqa_matmul_do_vt(do_chunk, v_sub).to(work_dtype)
                        w = (16.0 * ra_scale) / (1.0 - safe_u * safe_u)
                        ds_direct = torch.where(active, p_chunk * (dp_chunk - d_chunk) * w, torch.zeros_like(p_chunk))

                        sum_ds = ds_direct.sum(dim=-1, keepdim=True)
                        ds_direct.scatter_add_(-1, m_idx_chunk, -sum_ds)

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
                    None,
                    None,
                    None,
                    None,
                )


def ra_softmax(scores: Any, dim: int = -1, scale: float = 0.0625) -> Any:
    """
    Cayley-Padé Rational Softmax with Compact Support.

    Replaces exponential-based softmax with a rational kernel:
        M_i = max_m S_{im}
        Delta_{ij} = M_i - S_{ij} >= 0
        u_{ij} = Delta_{ij} * scale        (default scale = 1/16 = 0.0625)
        phi_{Ra}(Delta) = (max(0, 1 - u) / (1 + u))^8
        RaSoftmax(S)_{ij} = phi_{Ra}(Delta_{ij}) / sum_k phi_{Ra}(Delta_{ik})

    Mathematical & Hardware Properties:
      1. Order-2 Match with Exponential: Matches Maclaurin series of exp(-Delta)
         up to second derivative: phi(Delta) = 1 - Delta + 0.5 * Delta^2 - O(Delta^3).
      2. Exact Compact Support: For Delta >= 16, u >= 1 -> phi(Delta) == 0.0 exactly.
         Completely eliminates attention dilution and background noise accumulation
         in long contexts (N -> inf).
      3. Pole-Free & Zero-Epsilon: Because the max token has Delta = 0 -> phi(0) = 1.0,
         the denominator sum_k phi >= 1.0 is strictly guaranteed without requiring
         any epsilon safety floor.
      4. Closed-Form Autograd & Fused Triton Kernel: Both forward and backward passes
         execute without storing intermediate power buffers (q, q2, q4, phi) in HBM,
         evaluating the analytical derivative d(phi)/du = -16 / (1 - u^2) * phi in SRAM.
    """
    scale = float(scale)

    # 1. PyTorch Tensor Path
    if _HAS_TORCH and isinstance(scores, torch.Tensor):
        if scores.requires_grad and torch.is_grad_enabled():
            return _RaSoftmaxFunction.apply(scores, int(dim), scale)
        with torch.no_grad():
            return _ra_softmax_forward_no_grad(scores, dim=int(dim), scale=scale)

    # 2. NumPy ndarray Path
    if _HAS_NUMPY and isinstance(scores, np.ndarray):
        # 2a. Rust CPU SIMD fast-path
        if (
            _HAS_RUST_SIMD
        and not _is_torch_compiling()
            and scores.dtype == np.float32
            and scores.flags.c_contiguous
            and dim in (-1, scores.ndim - 1)
        ):
            out = np.empty_like(scores)
            cols = scores.shape[-1]
            rows = scores.size // cols
            _rs_engine.ra_softmax_f32_pyo3(scores, out, rows, cols, scale)
            return out

        # 2b. Standard NumPy path
        finite = np.isfinite(scores)
        max_scores = np.max(scores, axis=dim, keepdims=True)
        delta = np.where(finite, np.where(scores == max_scores, 0.0, max_scores - scores), np.inf)
        u = delta * scale
        num = np.maximum(1.0 - u, 0.0)
        den = 1.0 + u
        q = np.where(finite, num / den, 0.0)
        q2 = q * q
        q4 = q2 * q2
        phi = q4 * q4
        sum_phi = np.maximum(np.sum(phi, axis=dim, keepdims=True), 1e-38)
        return phi / sum_phi

    raise TypeError(f"ra_softmax expects torch.Tensor or numpy.ndarray, got {type(scores)}")


def _reduce_broadcast_grad(grad_expanded: Any, target_shape: Tuple[int, ...]) -> Any:
    """Reduces an expanded gradient tensor back to `target_shape` after broadcast expansion."""
    if grad_expanded.shape == target_shape:
        return grad_expanded
    g = grad_expanded
    ndim_diff = g.ndim - len(target_shape)
    if ndim_diff > 0:
        g = g.sum(dim=tuple(range(ndim_diff)))
    reduce_axes = tuple(i for i, (gs, ts) in enumerate(zip(g.shape, target_shape)) if ts == 1 and gs > 1)
    if reduce_axes:
        g = g.sum(dim=reduce_axes, keepdim=True)
    return g.reshape(target_shape)


if _HAS_TORCH:
    class _FusedRaEMAFunction(torch.autograd.Function):
        """
        Fused Rational Exponential Moving Average (Fused-RaEMA) Autograd Function.

        Computes the causal recurrence:
            h_t = decay_t * h_{t-1} + (1 - decay_t) * x_t
        and its exact reverse-time adjoint gradient:
            delta_t = grad_out_t + decay_{t+1} * delta_{t+1}
            dx_t = (1 - decay_t) * delta_t
            ddecay_t = (h_{t-1} - x_t) * delta_t
            dh0 = decay_1 * delta_1
        in a single Triton SRAM kernel launch on CUDA or a zero-graph-overhead scan on CPU.
        """

        @staticmethod
        def forward(
            ctx: Any,
            x: Any,
            decay: Any,
            h0: Optional[Any],
        ) -> Tuple[Any, Any]:
            with torch.no_grad():
                orig_shape = x.shape
                orig_dtype = x.dtype
                T = orig_shape[-2]
                D = orig_shape[-1]
                B_flat = x.numel() // max(1, T * D)
                state_shape = orig_shape[:-2] + (D,)

                decay_exp = decay.expand(orig_shape)
                x_3d = x.reshape(B_flat, T, D)
                decay_3d = decay_exp.reshape(B_flat, T, D)

                if h0 is not None:
                    h0_exp = h0.expand(state_shape)
                    h0_2d = h0_exp.reshape(B_flat, D)
                else:
                    h0_2d = None

                # Tier 1: Single-launch Triton SRAM recurrence kernel on CUDA
                if (
                    _HAS_TRITON
                    and x.is_cuda
                    and orig_dtype in (torch.float16, torch.bfloat16, torch.float32)
                ):
                    triton_res = triton_fused_ra_ema_fwd(
                        x_3d,
                        decay_3d.to(orig_dtype),
                        h0_2d.to(orig_dtype) if h0_2d is not None else None,
                    )
                    if triton_res is not None:
                        out_3d, h_last_2d = triton_res
                        if h0 is not None:
                            ctx.save_for_backward(x, decay, h0, out_3d)
                            ctx.has_h0 = True
                        else:
                            ctx.save_for_backward(x, decay, out_3d)
                            ctx.has_h0 = False
                        ctx.orig_shape = orig_shape
                        ctx.state_shape = state_shape
                        ctx.decay_shape = decay.shape
                        ctx.h0_shape = h0.shape if h0 is not None else None
                        return out_3d.reshape(orig_shape), h_last_2d.reshape(state_shape)

                # Tier 2: Zero-graph-overhead FP32/FP64 scan
                work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
                x_w = x_3d.to(work_dtype)
                a_w = decay_3d.to(work_dtype)
                out_w = torch.empty_like(x_w)
                h = (
                    h0_2d.to(work_dtype).clone()
                    if h0_2d is not None
                    else torch.zeros((B_flat, D), device=x.device, dtype=work_dtype)
                )

                for t in range(T):
                    a_t = a_w[:, t, :]
                    x_t = x_w[:, t, :]
                    h = a_t * h + (1.0 - a_t) * x_t
                    out_w[:, t, :] = h

                out_3d = out_w.to(orig_dtype)
                h_last_2d = h.to(orig_dtype)

                if h0 is not None:
                    ctx.save_for_backward(x, decay, h0, out_3d)
                    ctx.has_h0 = True
                else:
                    ctx.save_for_backward(x, decay, out_3d)
                    ctx.has_h0 = False
                ctx.orig_shape = orig_shape
                ctx.state_shape = state_shape
                ctx.decay_shape = decay.shape
                ctx.h0_shape = h0.shape if h0 is not None else None
                return out_3d.reshape(orig_shape), h_last_2d.reshape(state_shape)

        @staticmethod
        def backward(
            ctx: Any,
            grad_out: Any,
            grad_h_last: Any,
        ) -> Tuple[Any, Any, Optional[Any]]:
            with torch.no_grad():
                if ctx.has_h0:
                    x, decay, h0, out_3d = ctx.saved_tensors
                else:
                    x, decay, out_3d = ctx.saved_tensors
                    h0 = None

                orig_shape = ctx.orig_shape
                state_shape = ctx.state_shape
                orig_dtype = x.dtype
                T = orig_shape[-2]
                D = orig_shape[-1]
                B_flat = x.numel() // max(1, T * D)

                x_3d = x.reshape(B_flat, T, D)
                decay_3d = decay.expand(orig_shape).reshape(B_flat, T, D)
                h0_2d = h0.expand(state_shape).reshape(B_flat, D) if h0 is not None else None

                go_3d = (
                    grad_out.reshape(B_flat, T, D)
                    if grad_out is not None
                    else torch.zeros((B_flat, T, D), device=x.device, dtype=orig_dtype)
                )
                dh_last_2d = (
                    grad_h_last.reshape(B_flat, D)
                    if grad_h_last is not None
                    else None
                )
                need_dh0 = bool(ctx.has_h0 and ctx.needs_input_grad[2])

                # Tier 1: Single-launch Triton SRAM reverse-time adjoint kernel on CUDA
                if (
                    _HAS_TRITON
                    and x.is_cuda
                    and orig_dtype in (torch.float16, torch.bfloat16, torch.float32)
                ):
                    triton_bwd = triton_fused_ra_ema_bwd(
                        go_3d.to(orig_dtype),
                        dh_last_2d.to(orig_dtype) if dh_last_2d is not None else None,
                        x_3d,
                        decay_3d.to(orig_dtype),
                        h0_2d.to(orig_dtype) if h0_2d is not None else None,
                        out_3d,
                        need_dh0=need_dh0,
                    )
                    if triton_bwd is not None:
                        dx_3d, da_3d, dh0_2d = triton_bwd
                        dx = dx_3d.reshape(orig_shape).to(orig_dtype)
                        da = _reduce_broadcast_grad(da_3d.reshape(orig_shape), ctx.decay_shape).to(decay.dtype)
                        dh0 = (
                            _reduce_broadcast_grad(dh0_2d.reshape(state_shape), ctx.h0_shape).to(h0.dtype)
                            if (dh0_2d is not None and h0 is not None and ctx.h0_shape is not None)
                            else None
                        )
                        return dx, da, dh0

                # Tier 2: Exact reverse-time adjoint recurrence
                work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
                x_w = x_3d.to(work_dtype)
                a_w = decay_3d.to(work_dtype)
                out_w = out_3d.to(work_dtype)
                go_w = go_3d.to(work_dtype)
                h0_w = h0_2d.to(work_dtype) if h0_2d is not None else None

                dx_w = torch.empty_like(x_w)
                da_w = torch.empty_like(a_w)
                delta = (
                    dh_last_2d.to(work_dtype).clone()
                    if dh_last_2d is not None
                    else torch.zeros((B_flat, D), device=x.device, dtype=work_dtype)
                )

                for t in range(T - 1, -1, -1):
                    delta = delta + go_w[:, t, :]
                    a_t = a_w[:, t, :]
                    x_t = x_w[:, t, :]
                    if t > 0:
                        h_prev = out_w[:, t - 1, :]
                    elif h0_w is not None:
                        h_prev = h0_w
                    else:
                        h_prev = torch.zeros_like(x_t)

                    dx_w[:, t, :] = (1.0 - a_t) * delta
                    da_w[:, t, :] = (h_prev - x_t) * delta
                    delta = a_t * delta

                dx = dx_w.reshape(orig_shape).to(orig_dtype)
                da = _reduce_broadcast_grad(da_w.reshape(orig_shape), ctx.decay_shape).to(decay.dtype)
                dh0 = (
                    _reduce_broadcast_grad(delta.reshape(state_shape), ctx.h0_shape).to(h0.dtype)
                    if (need_dh0 and h0 is not None and ctx.h0_shape is not None)
                    else None
                )
                return dx, da, dh0


def fused_ra_ema(
    x: Any,
    decay: Any,
    h0: Optional[Any] = None,
    return_state: bool = False,
) -> Any:
    """
    Fused Rational Exponential Moving Average (Fused-RaEMA).

    Computes the causal O(N) sequence recurrence:
        h_t = decay_t * h_{t-1} + (1 - decay_t) * x_t
    for `t = 0, ..., T - 1` along `dim = -2`, keeping long-range temporal context alive
    beyond the compact-support horizon of rational attention without O(N^2) cost.

    Args:
        x: Input sequence tensor of shape `(..., T, D)`.
        decay: Rational decay gate in `(0, 1)`, broadcastable to `x.shape`
            (e.g. `(D,)`, `(1, 1, D)`, or `(B, T, D)`).
        h0: Optional initial recurrent state of shape `(..., D)` for streaming / chunked inference.
        return_state: If `True`, returns `(out, h_last)`; otherwise returns `out`.
    """
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        if x.ndim < 2:
            raise ValueError(f"fused_ra_ema expects at least a 2D (T, D) tensor, got {x.ndim}D")
        if not isinstance(decay, torch.Tensor):
            decay = torch.as_tensor(decay, device=x.device, dtype=x.dtype)
        if _is_torch_compiling():
            if decay is x:
                decay = decay.clone()
            if h0 is not None and (h0 is x or h0 is decay):
                h0 = h0.clone()
        out, h_last = _FusedRaEMAFunction.apply(x, decay, h0)
        return (out, h_last) if return_state else out

    if _HAS_NUMPY and isinstance(x, np.ndarray):
        if x.ndim < 2:
            raise ValueError(f"fused_ra_ema expects at least a 2D (T, D) array, got {x.ndim}D")
        orig_shape = x.shape
        T = orig_shape[-2]
        D = orig_shape[-1]
        B_flat = x.size // max(1, T * D)
        state_shape = orig_shape[:-2] + (D,)

        a_arr = np.broadcast_to(np.asarray(decay, dtype=x.dtype), orig_shape).reshape(B_flat, T, D)
        x_3d = x.reshape(B_flat, T, D)
        out_3d = np.empty_like(x_3d)
        if h0 is not None:
            h = np.broadcast_to(np.asarray(h0, dtype=x.dtype), state_shape).reshape(B_flat, D).copy()
        else:
            h = np.zeros((B_flat, D), dtype=x.dtype)

        for t in range(T):
            a_t = a_arr[:, t, :]
            h = a_t * h + (1.0 - a_t) * x_3d[:, t, :]
            out_3d[:, t, :] = h

        out = out_3d.reshape(orig_shape)
        h_last = h.reshape(state_shape)
        return (out, h_last) if return_state else out

    raise TypeError(f"fused_ra_ema expects torch.Tensor or numpy.ndarray, got {type(x)}")


class RaEMA(BaseModule):
    decay_proj: Any
    decay_logit: Any
    residual_logit: Any
    """
    Fused Rational Exponential Moving Average Layer (RaEMA / RationalEMA / FusedRaEMA).

    Provides an O(N) causal sequence mixer with algebraic rational decay gates
    `alpha_t = PosRaGate(z_t) in (0, 1)` and optional residual blending
    `y_t = gamma * h_t + (1 - gamma) * x_t`. Ideal as a pre-attention long-horizon
    memory backbone alongside compact-support `RaAttention`.
    """

    def __init__(
        self,
        dim: int,
        input_dependent_decay: bool = True,
        use_residual_gate: bool = True,
        init_decay_logit: float = 2.0,
        b: float = 1.0,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        self.input_dependent_decay = bool(input_dependent_decay)
        self.use_residual_gate = bool(use_residual_gate)
        self.init_decay_logit = float(init_decay_logit)
        self.b = float(b)

        if _HAS_TORCH:
            if self.input_dependent_decay:
                self.decay_proj = nn.Linear(self.dim, self.dim, bias=True, device=device, dtype=dtype)
                with torch.no_grad():
                    nn.init.zeros_(self.decay_proj.weight)
                    nn.init.constant_(self.decay_proj.bias, self.init_decay_logit)
                self.decay_logit = None
            else:
                self.decay_proj = None
                self.decay_logit = nn.Parameter(
                    torch.full((self.dim,), self.init_decay_logit, device=device, dtype=dtype)
                )

            if self.use_residual_gate:
                # Initialize residual logit to -1.0 -> PosRaGate(-1.0) = 0.25 (gentle 25% EMA injection at init)
                self.residual_logit = nn.Parameter(
                    torch.full((self.dim,), -1.0, device=device, dtype=dtype)
                )
            else:
                self.residual_logit = None
        else:
            self.decay_proj = None
            self.decay_logit = (
                np.full((self.dim,), self.init_decay_logit, dtype=np.float32)
                if _HAS_NUMPY
                else None
            )
            self.residual_logit = (
                np.full((self.dim,), -1.0, dtype=np.float32)
                if (_HAS_NUMPY and self.use_residual_gate)
                else None
            )

    def forward(
        self,
        x: Any,
        h0: Optional[Any] = None,
        return_state: bool = False,
    ) -> Any:
        if _HAS_TORCH and isinstance(x, torch.Tensor):
            if self.input_dependent_decay and self.decay_proj is not None:
                decay = pos_ra_gate(self.decay_proj(x), b=self.b)
            else:
                decay = pos_ra_gate(self.decay_logit, b=self.b)

            ema_res = fused_ra_ema(x, decay, h0=h0, return_state=return_state)
            if return_state:
                h_seq, h_last = ema_res
            else:
                h_seq = ema_res
                h_last = None

            if self.use_residual_gate and self.residual_logit is not None:
                gamma = pos_ra_gate(self.residual_logit, b=self.b).to(x.dtype)
                out = gamma * h_seq + (1.0 - gamma) * x
            else:
                out = h_seq

            return (out, h_last) if return_state else out

        if _HAS_NUMPY and isinstance(x, np.ndarray):
            logit = (
                self.decay_logit
                if isinstance(self.decay_logit, np.ndarray)
                else np.full((self.dim,), self.init_decay_logit, dtype=x.dtype)
            )
            decay = pos_ra_gate(logit, b=self.b)
            ema_res = fused_ra_ema(x, decay, h0=h0, return_state=return_state)
            if return_state:
                h_seq, h_last = ema_res
            else:
                h_seq = ema_res
                h_last = None

            if self.use_residual_gate:
                r_logit = (
                    self.residual_logit
                    if isinstance(self.residual_logit, np.ndarray)
                    else np.full((self.dim,), -1.0, dtype=x.dtype)
                )
                gamma = pos_ra_gate(r_logit, b=self.b)
                out = gamma * h_seq + (1.0 - gamma) * x
            else:
                out = h_seq
            return (out, h_last) if return_state else out

        raise TypeError(f"RaEMA expects torch.Tensor or numpy.ndarray, got {type(x)}")

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"input_dependent_decay={self.input_dependent_decay}, "
            f"use_residual_gate={self.use_residual_gate}"
        )


def ra_attention(
    q: Any,
    k: Any,
    v: Any,
    attn_mask: Optional[Any] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    ra_scale: float = 0.0625,
    q_chunk_size: Optional[int] = None,
) -> Any:
    """
    Fused Tiled Rational Scaled Dot-Product Attention (Flash-RaAttention).

    Supports both Multi-Head Attention (`H_q == H_kv`) and Grouped-Query Attention (`H_q % H_kv == 0`)
    directly from `(Q, K, V) -> O` without `repeat_interleave`:
      - On CUDA for short/medium contexts (`N <= 2048`): Uses cuBLAS TensorCore GEMM + fused 1D Triton
        `ra_softmax` with recomputed backward (zero persistent `O(N^2)` buffers across layers).
      - On CUDA for long contexts (`N > 2048`): Uses 2D Triton SRAM tiling with TensorCore `tl.dot`,
        compact-support block skipping, and 1-pass `dQ` backward.
      - On CPU / fallback: Uses dynamic fast-path single-shot GEMM when within memory budget
        and automatically partitions into bounded query chunks otherwise.
    """
    if _HAS_TORCH and isinstance(q, torch.Tensor):
        squeeze_head = False
        if q.ndim == 3:
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            squeeze_head = True
        elif q.ndim != 4:
            raise ValueError(f"ra_attention expects 3D (B, N, D) or 4D (B, H, N, D) tensors, got {q.ndim}D")

        h_q = q.shape[1]
        h_kv = k.shape[1]
        if h_kv <= 0 or (h_q % h_kv != 0) or v.shape[1] != h_kv:
            raise ValueError(
                f"ra_attention requires H_q ({h_q}) divisible by H_kv ({h_kv}) and k.shape[1] == v.shape[1] ({v.shape[1]})"
            )

        D = q.shape[-1]
        sm_scale = float(scale) if scale is not None else (1.0 / math.sqrt(D))
        ra_scale = float(ra_scale)

        if dropout_p <= 0.0:
            if _is_torch_compiling():
                if k is q:
                    k = k.clone()
                if v is q or v is k:
                    v = v.clone()
                if attn_mask is not None and (attn_mask is q or attn_mask is k or attn_mask is v):
                    attn_mask = attn_mask.clone()
            out = _FlashRaAttentionFunction.apply(
                q,
                k,
                v,
                attn_mask,
                bool(is_causal),
                sm_scale,
                ra_scale,
                int(q_chunk_size) if q_chunk_size is not None else None,
            )
        else:
            N_Q = q.shape[-2]
            N_K = k.shape[-2]
            scores = _gqa_matmul_qk(q, k) * sm_scale
            if is_causal:
                q_idx = torch.arange(N_Q, device=q.device).unsqueeze(1)
                k_idx = torch.arange(N_K, device=q.device).unsqueeze(0)
                scores = scores.masked_fill(q_idx < k_idx, float("-inf"))
            if attn_mask is not None:
                m_add = attn_mask.unsqueeze(1) if attn_mask.ndim == 3 else attn_mask
                scores = scores + m_add
            weights = ra_softmax(scores, dim=-1, scale=ra_scale)
            weights = torch.nn.functional.dropout(weights, p=float(dropout_p), training=True)
            out = _gqa_matmul_pv(weights, v)

        return out.squeeze(1) if squeeze_head else out

    if _HAS_NUMPY and isinstance(q, np.ndarray):
        D = q.shape[-1]
        sm_scale = float(scale) if scale is not None else (1.0 / math.sqrt(D))
        k_np = k
        v_np = v
        if q.ndim == 4 and k_np.ndim == 4 and q.shape[1] != k_np.shape[1]:
            if k_np.shape[1] <= 0 or (q.shape[1] % k_np.shape[1] != 0) or v_np.shape[1] != k_np.shape[1]:
                raise ValueError(
                    f"ra_attention requires H_q ({q.shape[1]}) divisible by H_kv ({k_np.shape[1]})"
                )
            g = q.shape[1] // k_np.shape[1]
            k_np = np.repeat(k_np, g, axis=1)
            v_np = np.repeat(v_np, g, axis=1)
        scores = np.matmul(q, np.swapaxes(k_np, -2, -1)) * sm_scale
        if is_causal:
            N_Q, N_K = q.shape[-2], k_np.shape[-2]
            mask = np.triu(np.full((N_Q, N_K), -np.inf, dtype=scores.dtype), k=1)
            scores = scores + mask
        if attn_mask is not None:
            m_add = np.expand_dims(attn_mask, axis=1) if (scores.ndim == 4 and attn_mask.ndim == 3) else attn_mask
            scores = scores + m_add
        weights = ra_softmax(scores, dim=-1, scale=ra_scale)
        return np.matmul(weights, v_np)

    raise TypeError(f"ra_attention expects torch.Tensor or numpy.ndarray, got {type(q)}")


class RaAttention(BaseModule):
    q_proj: Any
    k_proj: Any
    v_proj: Any
    out_proj: Any
    q_norm: Any
    k_norm: Any
    ema: Any
    attn_drop: Any
    proj_drop: Any
    """
    Rational Multi-Head & Grouped-Query Attention (RaAttention / RaAttn / RaGQA).

    Uses fused tiled rational Cayley-Padé attention (`ra_attention` / Flash-RaAttention)
    with compact support, native Grouped-Query Attention (`num_kv_heads`), and optional
    pre-attention Fused Rational EMA (`use_ema=True`) for long-horizon sequence memory.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        causal: bool = False,
        scale: Optional[float] = None,
        ra_scale: float = 0.0625,
        q_chunk_size: Optional[int] = None,
        use_ema: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads})")

        if isinstance(num_kv_heads, bool):
            qkv_bias = bool(num_kv_heads)
            self.num_kv_heads = self.num_heads
        elif num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = int(num_kv_heads)

        if self.num_kv_heads <= 0 or (self.num_heads % self.num_kv_heads != 0):
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
            )

        self.head_dim = self.dim // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(self.head_dim)
        self.ra_scale = float(ra_scale)
        self.q_chunk_size = int(q_chunk_size) if q_chunk_size is not None else None
        self.causal = bool(causal)
        self.attn_drop_p = float(attn_drop)
        self.use_ema = bool(use_ema)

        if _HAS_TORCH:
            self.ema = RaEMA(self.dim, device=device, dtype=dtype) if self.use_ema else None
            self.q_proj = nn.Linear(self.dim, self.dim, bias=qkv_bias, device=device, dtype=dtype)
            self.k_proj = nn.Linear(self.dim, self.kv_dim, bias=qkv_bias, device=device, dtype=dtype)
            self.v_proj = nn.Linear(self.dim, self.kv_dim, bias=qkv_bias, device=device, dtype=dtype)
            self.out_proj = nn.Linear(self.dim, self.dim, bias=qkv_bias, device=device, dtype=dtype)

            if qk_norm:
                self.q_norm = RaNorm(self.head_dim, device=device, dtype=dtype)
                self.k_norm = RaNorm(self.head_dim, device=device, dtype=dtype)
            else:
                self.q_norm = None
                self.k_norm = None

            self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0.0 else nn.Identity()
            self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()
        else:
            self.ema = None
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.out_proj = None
            self.q_norm = None
            self.k_norm = None
            self.attn_drop = None
            self.proj_drop = None

    def forward(
        self,
        x: Any,
        attn_mask: Optional[Any] = None,
    ) -> Any:
        if not (_HAS_TORCH and isinstance(x, torch.Tensor)):
            raise TypeError(f"RaAttention forward currently requires a torch.Tensor, got {type(x)}")

        B, N, C = x.shape
        if self.ema is not None:
            x = self.ema(x)

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q).to(x.dtype)
        if self.k_norm is not None:
            k = self.k_norm(k).to(x.dtype)

        drop_p = self.attn_drop_p if self.training else 0.0
        out = ra_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=drop_p,
            is_causal=self.causal,
            scale=self.scale,
            ra_scale=self.ra_scale,
            q_chunk_size=self.q_chunk_size,
        )

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"scale={self.scale:.4f}, "
            f"causal={self.causal}, "
            f"use_ema={self.use_ema}"
        )


# Direct aliases for RaAttention & RaEMA
RaAttn = RaAttention
RaGQA = RaAttention
RationalGQA = RaAttention
flash_ra_attention = ra_attention
ra_ema = fused_ra_ema
RationalEMA = RaEMA
FusedRaEMA = RaEMA
