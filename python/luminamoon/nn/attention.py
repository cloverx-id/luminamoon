from __future__ import annotations

import math
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
from .norm import RaNorm
from .triton_kernels import (
    triton_flash_ra_attention,
    triton_flash_ra_attention_backward,
    triton_ra_softmax,
    triton_ra_softmax_backward,
)

# Maximum elements per temporary chunk buffer (~16 MB in FP32) to prevent HBM spikes
_MAX_CHUNK_ELEMENTS: int = 4_194_304


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
    if _HAS_TRITON and scores.is_cuda and norm_dim == ndim - 1:
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
    if _HAS_TRITON and scores.is_cuda and norm_dim == ndim - 1:
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

        Eliminates [B, H, N_Q, N_K] attention matrix materialization in HBM:
          - Tier 1 (CUDA + Triton): Executes 2-pass SRAM tiling with compact-support block skipping.
          - Tier 2 (Universal Fallback): Executes Query-chunked L2 tiling with closed-form
            analytical backward recomputation, keeping peak working memory O(B * H * C_Q * N_K)
            and autograd saved state strictly O(B * H * N_Q * D).
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
            q_chunk_size: int,
        ) -> Any:
            with torch.no_grad():
                # Tier 1: Triton SRAM Tiled Kernel (when on CUDA and no arbitrary external mask tensor)
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
                        ctx.used_triton = True
                        ctx.is_causal = is_causal
                        ctx.sm_scale = sm_scale
                        ctx.ra_scale = ra_scale
                        ctx.q_chunk_size = q_chunk_size
                        return out

                # Tier 2: Query-Chunked Tiled Execution (Zero full [B, H, N_Q, N_K] materialization)
                B, H, N_Q, _ = q.shape
                N_K = k.shape[2]
                D_V = v.shape[-1]
                orig_dtype = q.dtype
                work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype

                out = torch.empty((B, H, N_Q, D_V), device=q.device, dtype=orig_dtype)
                m_all = torch.empty((B, H, N_Q, 1), device=q.device, dtype=work_dtype)
                l_all = torch.empty((B, H, N_Q, 1), device=q.device, dtype=work_dtype)
                m_idx_all = torch.empty((B, H, N_Q, 1), device=q.device, dtype=torch.int64)

                chunk_sz = max(16, min(int(q_chunk_size), N_Q))
                k_t = k.transpose(-2, -1)

                for q_start in range(0, N_Q, chunk_sz):
                    q_end = min(N_Q, q_start + chunk_sz)
                    k_limit = min(N_K, q_end) if is_causal else N_K

                    q_chunk = q[:, :, q_start:q_end, :]
                    k_sub_t = k_t[:, :, :, :k_limit]
                    v_sub = v[:, :, :k_limit, :]

                    s_chunk = (q_chunk @ k_sub_t).to(work_dtype) * sm_scale

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

                    out[:, :, q_start:q_end, :] = p_chunk.to(orig_dtype) @ v_sub
                    m_all[:, :, q_start:q_end, :] = m_chunk
                    l_all[:, :, q_start:q_end, :] = l_chunk
                    m_idx_all[:, :, q_start:q_end, :] = m_idx_chunk

                if attn_mask is not None:
                    ctx.save_for_backward(q, k, v, out, m_all, l_all, m_idx_all, attn_mask)
                    ctx.has_mask = True
                else:
                    ctx.save_for_backward(q, k, v, out, m_all, l_all, m_idx_all)
                    ctx.has_mask = False

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
                N_K = k.shape[2]
                orig_dtype = q.dtype
                work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype

                # If fallback triggered from a Triton forward tensor shape (B, H, N_Q), unsqueeze trailing dim
                if m_all.ndim == 3:
                    m_all = m_all.unsqueeze(-1).to(work_dtype)
                    l_all = l_all.unsqueeze(-1).to(work_dtype)
                    m_idx_all = m_idx_all.unsqueeze(-1).to(torch.int64)

                do_work = grad_out.to(work_dtype)
                out_work = out.to(work_dtype)
                d_all = (do_work * out_work).sum(dim=-1, keepdim=True)

                dq = torch.zeros_like(q, dtype=work_dtype)
                dk = torch.zeros_like(k, dtype=work_dtype)
                dv = torch.zeros_like(v, dtype=work_dtype)
                d_mask = (
                    torch.zeros_like(attn_mask, dtype=work_dtype)
                    if (attn_mask is not None and attn_mask.requires_grad)
                    else None
                )

                chunk_sz = max(16, min(int(ctx.q_chunk_size), N_Q))
                is_causal = ctx.is_causal
                sm_scale = ctx.sm_scale
                ra_scale = ctx.ra_scale

                q_work = q.to(work_dtype)
                k_work = k.to(work_dtype)
                v_work = v.to(work_dtype)
                k_t = k_work.transpose(-2, -1)

                for q_start in range(0, N_Q, chunk_sz):
                    q_end = min(N_Q, q_start + chunk_sz)
                    k_limit = min(N_K, q_end) if is_causal else N_K

                    q_chunk = q_work[:, :, q_start:q_end, :]
                    do_chunk = do_work[:, :, q_start:q_end, :]
                    k_sub = k_work[:, :, :k_limit, :]
                    k_sub_t = k_t[:, :, :, :k_limit]
                    v_sub = v_work[:, :, :k_limit, :]

                    m_chunk = m_all[:, :, q_start:q_end, :]
                    l_chunk = l_all[:, :, q_start:q_end, :]
                    m_idx_chunk = m_idx_all[:, :, q_start:q_end, :]
                    d_chunk = d_all[:, :, q_start:q_end, :]

                    s_chunk = (q_chunk @ k_sub_t) * sm_scale

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

                    dv[:, :, :k_limit, :] += p_chunk.transpose(-2, -1) @ do_chunk

                    dp_chunk = do_chunk @ v_sub.transpose(-2, -1)
                    w = (16.0 * ra_scale) / (1.0 - safe_u * safe_u)
                    ds_direct = torch.where(active, p_chunk * (dp_chunk - d_chunk) * w, torch.zeros_like(p_chunk))

                    sum_ds = ds_direct.sum(dim=-1, keepdim=True)
                    ds_direct.scatter_add_(-1, m_idx_chunk, -sum_ds)

                    if d_mask is not None:
                        _accum_attn_mask_grad(d_mask, ds_direct, q_start, q_end, k_limit)

                    ds_scaled = ds_direct * sm_scale
                    dq[:, :, q_start:q_end, :] = ds_scaled @ k_sub
                    dk[:, :, :k_limit, :] += ds_scaled.transpose(-2, -1) @ q_chunk

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


def ra_attention(
    q: Any,
    k: Any,
    v: Any,
    attn_mask: Optional[Any] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    ra_scale: float = 0.0625,
    q_chunk_size: int = 128,
) -> Any:
    """
    Fused Tiled Rational Scaled Dot-Product Attention (Flash-RaAttention).

    Computes rational Cayley-Padé attention directly from (Q, K, V) -> O without
    materializing the [B, H, N_Q, N_K] attention matrix in HBM:
      - On CUDA with Triton: Uses 2-pass SRAM tiling with compact-support block skipping.
      - On CPU / fallback: Uses Query-chunked L2 tiling with closed-form backward recomputation.
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
                int(q_chunk_size),
            )
        else:
            N_Q = q.shape[-2]
            N_K = k.shape[-2]
            scores = (q @ k.transpose(-2, -1)) * sm_scale
            if is_causal:
                q_idx = torch.arange(N_Q, device=q.device).unsqueeze(1)
                k_idx = torch.arange(N_K, device=q.device).unsqueeze(0)
                scores = scores.masked_fill(q_idx < k_idx, float("-inf"))
            if attn_mask is not None:
                m_add = attn_mask.unsqueeze(1) if attn_mask.ndim == 3 else attn_mask
                scores = scores + m_add
            weights = ra_softmax(scores, dim=-1, scale=ra_scale)
            weights = torch.nn.functional.dropout(weights, p=float(dropout_p), training=True)
            out = weights @ v

        return out.squeeze(1) if squeeze_head else out

    if _HAS_NUMPY and isinstance(q, np.ndarray):
        D = q.shape[-1]
        sm_scale = float(scale) if scale is not None else (1.0 / math.sqrt(D))
        scores = np.matmul(q, np.swapaxes(k, -2, -1)) * sm_scale
        if is_causal:
            N_Q, N_K = q.shape[-2], k.shape[-2]
            mask = np.triu(np.full((N_Q, N_K), -np.inf, dtype=scores.dtype), k=1)
            scores = scores + mask
        if attn_mask is not None:
            m_add = np.expand_dims(attn_mask, axis=1) if (scores.ndim == 4 and attn_mask.ndim == 3) else attn_mask
            scores = scores + m_add
        weights = ra_softmax(scores, dim=-1, scale=ra_scale)
        return np.matmul(weights, v)

    raise TypeError(f"ra_attention expects torch.Tensor or numpy.ndarray, got {type(q)}")


class RaAttention(BaseModule):
    q_proj: Any
    k_proj: Any
    v_proj: Any
    out_proj: Any
    q_norm: Any
    k_norm: Any
    attn_drop: Any
    proj_drop: Any
    """
    Rational Multi-Head Attention (RaAttention / RaAttn).

    Uses fused tiled rational Cayley-Padé attention (ra_attention / Flash-RaAttention)
    with compact support. Eliminates exponential instructions, prevents attention
    dilution on long sequence lengths, and avoids materializing the O(N^2) attention
    matrix in GPU HBM.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        causal: bool = False,
        scale: Optional[float] = None,
        ra_scale: float = 0.0625,
        q_chunk_size: int = 128,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads})")

        self.head_dim = self.dim // self.num_heads
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(self.head_dim)
        self.ra_scale = float(ra_scale)
        self.q_chunk_size = int(q_chunk_size)
        self.causal = bool(causal)
        self.attn_drop_p = float(attn_drop)

        if _HAS_TORCH:
            self.q_proj = nn.Linear(self.dim, self.dim, bias=qkv_bias, device=device, dtype=dtype)
            self.k_proj = nn.Linear(self.dim, self.dim, bias=qkv_bias, device=device, dtype=dtype)
            self.v_proj = nn.Linear(self.dim, self.dim, bias=qkv_bias, device=device, dtype=dtype)
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

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

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
            f"head_dim={self.head_dim}, "
            f"scale={self.scale:.4f}, "
            f"causal={self.causal}"
        )


# Direct alias for RaAttention
RaAttn = RaAttention
flash_ra_attention = ra_attention
