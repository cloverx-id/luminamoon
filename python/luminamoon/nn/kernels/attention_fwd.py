from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional, Tuple

from ._base import HAS_TRITON, _HAS_TORCH, tl, torch, triton

if TYPE_CHECKING:
    _flash_ra_attn_fwd_kernel: Any = None
elif HAS_TRITON:
    @triton.jit
    def _flash_ra_attn_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        m_ptr,
        l_ptr,
        m_idx_ptr,
        stride_bh_q,
        stride_bh_k,
        stride_bh_v,
        stride_bh_o,
        stride_bh_stat,
        N_Q,
        N_K,
        sm_scale: tl.constexpr,
        ra_scale: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        bh_q = tl.program_id(1)
        bh_kv = bh_q // GROUP_SIZE

        q_base = q_ptr + bh_q * stride_bh_q
        k_base = k_ptr + bh_kv * stride_bh_k
        v_base = v_ptr + bh_kv * stride_bh_v
        o_base = out_ptr + bh_q * stride_bh_o
        stat_offset = bh_q * stride_bh_stat

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < N_Q
        mask_d = offs_d < HEAD_DIM

        q_ptrs = q_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
        q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

        hi = tl.minimum(N_K, (start_m + 1) * BLOCK_M) if IS_CAUSAL else N_K

        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        m_idx_i = tl.full([BLOCK_M], N_K, dtype=tl.int32)

        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N_K
            k_ptrs = k_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :]
            k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

            s = tl.dot(q, tl.trans(k)) * sm_scale
            valid = mask_m[:, None] & mask_n[None, :]
            if IS_CAUSAL:
                if start_n + BLOCK_N > start_m * BLOCK_M:
                    valid = valid & (offs_m[:, None] >= offs_n[None, :])
            s = tl.where(valid, s, -float("inf"))

            tile_max = tl.max(s, axis=1)
            new_max = tl.maximum(m_i, tile_max)

            finite = valid & (s > -float("inf"))
            is_tile_max = (s == new_max[:, None]) & finite
            tile_first_col = tl.min(tl.where(is_tile_max, offs_n[None, :], N_K), axis=1)

            m_idx_i = tl.where(new_max > m_i, tile_first_col, tl.minimum(m_idx_i, tile_first_col))
            m_i = new_max

        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N_K
            k_ptrs = k_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :]
            k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

            s = tl.dot(q, tl.trans(k)) * sm_scale
            valid = mask_m[:, None] & mask_n[None, :]
            if IS_CAUSAL:
                if start_n + BLOCK_N > start_m * BLOCK_M:
                    valid = valid & (offs_m[:, None] >= offs_n[None, :])
            s = tl.where(valid, s, -float("inf"))

            finite = valid & (s > -float("inf"))
            delta = tl.where(finite, tl.where(s == m_i[:, None], 0.0, m_i[:, None] - s), float("inf"))
            u = delta * ra_scale

            tile_min_u = tl.min(tl.min(u, axis=1), axis=0)
            if tile_min_u < 1.0:
                active = finite & (u < 1.0)
                safe_u = tl.where(active, u, 0.0)
                q_rat = tl.maximum(1.0 - safe_u, 0.0) / (1.0 + safe_u)
                q2 = q_rat * q_rat
                q4 = q2 * q2
                phi = tl.where(active, q4 * q4, 0.0)

                l_i += tl.sum(phi, axis=1)

                v_ptrs = v_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :]
                v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
                acc += tl.dot(phi.to(v.dtype), v)

        safe_l = tl.where(l_i > 0.0, l_i, 1.0)
        out = acc / safe_l[:, None]

        o_ptrs = o_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
        tl.store(o_ptrs, out, mask=mask_m[:, None] & mask_d[None, :])
        tl.store(m_ptr + stat_offset + offs_m, m_i, mask=mask_m)
        tl.store(l_ptr + stat_offset + offs_m, safe_l, mask=mask_m)
        tl.store(m_idx_ptr + stat_offset + offs_m, m_idx_i, mask=mask_m)
else:
    _flash_ra_attn_fwd_kernel = None


def triton_flash_ra_attention(
    q: Any,
    k: Any,
    v: Any,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    ra_scale: float = 0.0625,
) -> Optional[Tuple[Any, Any, Any, Any]]:
    """Dispatches fused tiled Rational Attention forward pass to Triton SRAM kernel."""
    if not (
        HAS_TRITON
        and _HAS_TORCH
        and isinstance(q, torch.Tensor)
        and q.is_cuda
        and q.ndim == 4
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return None

    B, H, N_Q, D = q.shape
    H_KV = k.shape[1]
    N_K = k.shape[2]
    if (
        D > 256
        or k.shape[-1] != D
        or v.shape[-1] != D
        or v.shape[1] != H_KV
        or H_KV <= 0
        or (H % H_KV != 0)
    ):
        return None

    GROUP_SIZE = H // H_KV
    BLOCK_D = max(16, triton.next_power_of_2(D))
    BLOCK_M = 64 if N_Q >= 64 else max(16, triton.next_power_of_2(N_Q))
    BLOCK_N = 64 if N_K >= 64 else max(16, triton.next_power_of_2(N_K))

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()

    out = torch.empty_like(q_c)
    bh_q = B * H
    m = torch.empty((B, H, N_Q), device=q.device, dtype=torch.float32)
    l = torch.empty((B, H, N_Q), device=q.device, dtype=torch.float32)
    m_idx = torch.empty((B, H, N_Q), device=q.device, dtype=torch.int32)

    grid = (triton.cdiv(N_Q, BLOCK_M), bh_q)
    _flash_ra_attn_fwd_kernel[grid](
        q_c,
        k_c,
        v_c,
        out,
        m,
        l,
        m_idx,
        N_Q * D,
        N_K * D,
        N_K * D,
        N_Q * D,
        N_Q,
        N_Q,
        N_K,
        sm_scale=float(sm_scale),
        ra_scale=float(ra_scale),
        IS_CAUSAL=bool(is_causal),
        HEAD_DIM=D,
        BLOCK_D=BLOCK_D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUP_SIZE=GROUP_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return out, m, l, m_idx
