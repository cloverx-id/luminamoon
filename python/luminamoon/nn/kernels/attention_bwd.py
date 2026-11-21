from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional, Tuple

from ._base import HAS_TRITON, _HAS_TORCH, tl, torch, triton

if TYPE_CHECKING:
    _flash_ra_attn_bwd_dq_kernel: Any = None
    _flash_ra_attn_bwd_dkdv_kernel: Any = None
elif HAS_TRITON:
    @triton.jit
    def _flash_ra_attn_bwd_dq_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        do_ptr,
        dq_ptr,
        m_ptr,
        l_ptr,
        m_idx_ptr,
        d_ptr,
        c_ptr,
        stride_bh_q,
        stride_bh_k,
        stride_bh_v,
        stride_bh_do,
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
        do_base = do_ptr + bh_q * stride_bh_do
        dq_base = dq_ptr + bh_q * stride_bh_q
        stat_offset = bh_q * stride_bh_stat

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < N_Q
        mask_d = offs_d < HEAD_DIM

        q = tl.load(
            q_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        )
        do = tl.load(
            do_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        )
        m_i = tl.load(m_ptr + stat_offset + offs_m, mask=mask_m, other=-float("inf")).to(tl.float32)
        l_i = tl.load(l_ptr + stat_offset + offs_m, mask=mask_m, other=1.0).to(tl.float32)
        m_idx_i = tl.load(m_idx_ptr + stat_offset + offs_m, mask=mask_m, other=N_K).to(tl.int32)
        d_i = tl.load(d_ptr + stat_offset + offs_m, mask=mask_m, other=0.0).to(tl.float32)

        hi = tl.minimum(N_K, (start_m + 1) * BLOCK_M) if IS_CAUSAL else N_K
        c_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
        dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N_K
            k = tl.load(
                k_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            )

            s = tl.dot(q, tl.trans(k)) * sm_scale
            valid = mask_m[:, None] & mask_n[None, :]
            if IS_CAUSAL:
                valid = valid & (offs_m[:, None] >= offs_n[None, :])
            s = tl.where(valid, s, -float("inf"))

            finite = valid & (s > -float("inf"))
            delta = tl.where(finite, tl.where(s == m_i[:, None], 0.0, m_i[:, None] - s), float("inf"))
            u = delta * ra_scale
            tile_min_u = tl.min(tl.min(u, axis=1), axis=0)
            if tile_min_u < 1.0:
                active = finite & (u < 1.0)
                safe_u = tl.where(active, u, 0.0)
                one_plus_u = 1.0 + safe_u
                q_rat = tl.where(active, (1.0 - safe_u) / one_plus_u, 0.0)
                q2 = q_rat * q_rat
                q4 = q2 * q2
                q7 = q_rat * q2 * q4
                p = tl.where(active, (q4 * q4) / l_i[:, None], 0.0)

                v = tl.load(
                    v_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=mask_n[:, None] & mask_d[None, :],
                    other=0.0,
                )
                dp = tl.dot(do, tl.trans(v))
                pw = ((16.0 * ra_scale) / l_i[:, None]) * q7 / (one_plus_u * one_plus_u)
                ds_direct = tl.where(active, pw * (dp - d_i[:, None]), 0.0)
                c_acc += tl.sum(ds_direct, axis=1)
                dq_acc += tl.dot(ds_direct.to(k.dtype), k)

        tl.store(c_ptr + stat_offset + offs_m, c_acc, mask=mask_m)

        has_argmax = mask_m & (m_idx_i >= 0) & (m_idx_i < N_K)
        safe_m_idx = tl.where(has_argmax, m_idx_i, 0)
        k_argmax = tl.load(
            k_base + safe_m_idx[:, None] * HEAD_DIM + offs_d[None, :],
            mask=has_argmax[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        dq_out = (dq_acc - c_acc[:, None] * k_argmax) * sm_scale

        tl.store(
            dq_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
            dq_out,
            mask=mask_m[:, None] & mask_d[None, :],
        )

    @triton.jit
    def _flash_ra_attn_bwd_dkdv_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        do_ptr,
        dk_ptr,
        dv_ptr,
        m_ptr,
        l_ptr,
        m_idx_ptr,
        d_ptr,
        c_ptr,
        stride_bh_q,
        stride_bh_k,
        stride_bh_v,
        stride_bh_do,
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
        start_n = tl.program_id(0)
        bh_kv = tl.program_id(1)

        k_base = k_ptr + bh_kv * stride_bh_k
        v_base = v_ptr + bh_kv * stride_bh_v
        dk_base = dk_ptr + bh_kv * stride_bh_k
        dv_base = dv_ptr + bh_kv * stride_bh_v

        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        mask_n = offs_n < N_K
        mask_d = offs_d < HEAD_DIM

        k = tl.load(
            k_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )
        v = tl.load(
            v_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )

        dk_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

        lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M if IS_CAUSAL else 0

        for g in range(0, GROUP_SIZE):
            bh_q = bh_kv * GROUP_SIZE + g
            q_base = q_ptr + bh_q * stride_bh_q
            do_base = do_ptr + bh_q * stride_bh_do
            stat_offset = bh_q * stride_bh_stat

            for start_m in range(lo, N_Q, BLOCK_M):
                offs_m = start_m + tl.arange(0, BLOCK_M)
                mask_m = offs_m < N_Q

                q = tl.load(
                    q_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=mask_m[:, None] & mask_d[None, :],
                    other=0.0,
                )
                m_i = tl.load(m_ptr + stat_offset + offs_m, mask=mask_m, other=-float("inf")).to(tl.float32)

                s = tl.dot(q, tl.trans(k)) * sm_scale
                valid = mask_m[:, None] & mask_n[None, :]
                if IS_CAUSAL:
                    valid = valid & (offs_m[:, None] >= offs_n[None, :])
                s = tl.where(valid, s, -float("inf"))

                finite = valid & (s > -float("inf"))
                delta = tl.where(finite, tl.where(s == m_i[:, None], 0.0, m_i[:, None] - s), float("inf"))
                u = delta * ra_scale
                tile_min_u = tl.min(tl.min(u, axis=1), axis=0)
                if tile_min_u < 1.0:
                    do = tl.load(
                        do_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                        mask=mask_m[:, None] & mask_d[None, :],
                        other=0.0,
                    )
                    l_i = tl.load(l_ptr + stat_offset + offs_m, mask=mask_m, other=1.0).to(tl.float32)
                    m_idx_i = tl.load(m_idx_ptr + stat_offset + offs_m, mask=mask_m, other=N_K).to(tl.int32)
                    d_i = tl.load(d_ptr + stat_offset + offs_m, mask=mask_m, other=0.0).to(tl.float32)
                    c_i = tl.load(c_ptr + stat_offset + offs_m, mask=mask_m, other=0.0).to(tl.float32)

                    active = finite & (u < 1.0)
                    safe_u = tl.where(active, u, 0.0)
                    one_plus_u = 1.0 + safe_u
                    q_rat = tl.where(active, (1.0 - safe_u) / one_plus_u, 0.0)
                    q2 = q_rat * q_rat
                    q4 = q2 * q2
                    q7 = q_rat * q2 * q4
                    p = tl.where(active, (q4 * q4) / l_i[:, None], 0.0)

                    dv_acc += tl.dot(tl.trans(p.to(do.dtype)), do)

                    dp = tl.dot(do, tl.trans(v))
                    pw = ((16.0 * ra_scale) / l_i[:, None]) * q7 / (one_plus_u * one_plus_u)
                    ds_direct = tl.where(active, pw * (dp - d_i[:, None]), 0.0)

                    is_first_max = finite & (offs_n[None, :] == m_idx_i[:, None])
                    ds = (ds_direct - tl.where(is_first_max, c_i[:, None], 0.0)) * sm_scale

                    dk_acc += tl.dot(tl.trans(ds.to(q.dtype)), q)

        tl.store(
            dk_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
            dk_acc,
            mask=mask_n[:, None] & mask_d[None, :],
        )
        tl.store(
            dv_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
            dv_acc,
            mask=mask_n[:, None] & mask_d[None, :],
        )
else:
    _flash_ra_attn_bwd_dq_kernel = None
    _flash_ra_attn_bwd_dkdv_kernel = None


def triton_flash_ra_attention_backward(
    grad_out: Any,
    q: Any,
    k: Any,
    v: Any,
    out: Any,
    m: Any,
    l: Any,
    m_idx: Any,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    ra_scale: float = 0.0625,
) -> Optional[Tuple[Any, Any, Any]]:
    """Dispatches fused tiled Rational Attention backward pass to Triton SRAM kernels."""
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
    do_c = grad_out.contiguous()
    out_c = out.contiguous()

    d_vec = (do_c.to(torch.float32) * out_c.to(torch.float32)).sum(dim=-1).contiguous()
    c_vec = torch.empty_like(d_vec)

    dq = torch.empty_like(q_c)
    dk = torch.empty_like(k_c)
    dv = torch.empty_like(v_c)

    shared_kw = dict(
        sm_scale=float(sm_scale),
        ra_scale=float(ra_scale),
        IS_CAUSAL=bool(is_causal),
        HEAD_DIM=D,
        BLOCK_D=BLOCK_D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUP_SIZE=GROUP_SIZE,
    )
    _flash_ra_attn_bwd_dq_kernel[(triton.cdiv(N_Q, BLOCK_M), B * H)](
        q_c, k_c, v_c, do_c, dq, m, l, m_idx, d_vec, c_vec,
        N_Q * D, N_K * D, N_K * D, N_Q * D, N_Q, N_Q, N_K,
        **shared_kw,
    )
    _flash_ra_attn_bwd_dkdv_kernel[(triton.cdiv(N_K, BLOCK_N), B * H_KV)](
        q_c, k_c, v_c, do_c, dk, dv, m, l, m_idx, d_vec, c_vec,
        N_Q * D, N_K * D, N_K * D, N_Q * D, N_Q, N_Q, N_K,
        **shared_kw,
    )
    return dq, dk, dv
