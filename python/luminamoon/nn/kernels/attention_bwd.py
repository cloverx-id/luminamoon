from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional, Tuple

from ._base import HAS_FDIV_APPROX, HAS_TRITON, _HAS_TORCH, tl, torch, triton

if TYPE_CHECKING:
    _flash_ra_attn_bwd_preprocess_kernel: Any = None
    _flash_ra_attn_bwd_dq_kernel: Any = None
    _flash_ra_attn_bwd_dkdv_kernel: Any = None
elif HAS_TRITON:
    _ATTN_BWD_DQ_CONFIGS = [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    ]
    _ATTN_BWD_DKDV_CONFIGS = [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    ]

    @triton.jit
    def _flash_ra_attn_bwd_preprocess_kernel(
        do_ptr,
        out_ptr,
        d_ptr,
        stride_dob,
        stride_doh,
        stride_don,
        stride_ob,
        stride_oh,
        stride_on,
        H_Q,
        N_Q,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        EVEN_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        bh_q = tl.program_id(0)
        start_m = tl.program_id(1)
        b = (bh_q // H_Q).to(tl.int64)
        h_q = (bh_q % H_Q).to(tl.int64)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < N_Q
        load_mask = mask_m[:, None] if EVEN_D else (mask_m[:, None] & (offs_d[None, :] < HEAD_DIM))

        do_base = do_ptr + b * stride_dob + h_q * stride_doh
        o_base = out_ptr + b * stride_ob + h_q * stride_oh
        do = tl.load(do_base + offs_m[:, None] * stride_don + offs_d[None, :], mask=load_mask, other=0.0).to(tl.float32)
        out = tl.load(o_base + offs_m[:, None] * stride_on + offs_d[None, :], mask=load_mask, other=0.0).to(tl.float32)
        d_i = tl.sum(do * out, axis=1)
        tl.store(d_ptr + bh_q.to(tl.int64) * N_Q + offs_m, d_i, mask=mask_m)

    @triton.jit
    def _ra_bwd_dq_step(
        q,
        do,
        k_base,
        v_base,
        stride_kn,
        stride_vn,
        offs_m,
        offs_d,
        mask_m,
        mask_d,
        start_n,
        N_K,
        qk_scale,
        m_scaled,
        coeff_i,
        d_i,
        c_acc,
        dq_acc,
        BLOCK_N: tl.constexpr,
        EVEN_D: tl.constexpr,
        MASK_N: tl.constexpr,
        MASK_CAUSAL: tl.constexpr,
        USE_APPROX_DIV: tl.constexpr,
    ):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :]
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :]
        if MASK_N:
            mask_n = offs_n < N_K
            kv_mask = mask_n[:, None] if EVEN_D else (mask_n[:, None] & mask_d[None, :])
            k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
            v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
        elif EVEN_D:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=mask_d[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=mask_d[None, :], other=0.0)

        u = tl.maximum(m_scaled[:, None] - tl.dot(q, tl.trans(k)) * qk_scale, 0.0)
        if MASK_N or MASK_CAUSAL:
            valid = mask_m[:, None] & (offs_n[None, :] < N_K)
            if MASK_CAUSAL:
                valid = valid & (offs_m[:, None] >= offs_n[None, :])
            active = valid & (u < 1.0)
        else:
            active = u < 1.0

        safe_u = tl.where(active, u, 0.0)
        if USE_APPROX_DIV:
            inv_1pu = tl.fdiv(1.0, 1.0 + safe_u, approx=True)
        else:
            inv_1pu = tl.fdiv(1.0, 1.0 + safe_u)
        q_rat = (1.0 - safe_u) * inv_1pu
        q2 = q_rat * q_rat
        q4 = q2 * q2
        q7 = q_rat * q2 * q4

        dp = tl.dot(do, tl.trans(v))
        pw = (coeff_i[:, None] * q7) * (inv_1pu * inv_1pu)
        ds_direct = tl.where(active, pw * (dp - d_i[:, None]), 0.0)
        c_acc += tl.sum(ds_direct, axis=1)
        dq_acc += tl.dot(ds_direct.to(k.dtype), k)
        return c_acc, dq_acc

    @triton.jit
    def _is_block_active(
        am_base,
        m64_idx,
        n32_idx,
        num_m64,
        num_n32,
        BM_CELLS: tl.constexpr,
        BN_CELLS: tl.constexpr,
    ):
        r0 = am_base + m64_idx.to(tl.int64) * num_n32 + n32_idx
        v = tl.load(r0).to(tl.int32)
        if BN_CELLS >= 2:
            v = v | tl.load(r0 + 1, mask=(n32_idx + 1) < num_n32, other=0).to(tl.int32)
        if BN_CELLS >= 4:
            v = v | tl.load(r0 + 2, mask=(n32_idx + 2) < num_n32, other=0).to(tl.int32)
            v = v | tl.load(r0 + 3, mask=(n32_idx + 3) < num_n32, other=0).to(tl.int32)
        if BM_CELLS >= 2:
            m1 = (m64_idx + 1) < num_m64
            r1 = am_base + (m64_idx + 1).to(tl.int64) * num_n32 + n32_idx
            v = v | tl.load(r1, mask=m1, other=0).to(tl.int32)
            if BN_CELLS >= 2:
                v = v | tl.load(r1 + 1, mask=m1 & ((n32_idx + 1) < num_n32), other=0).to(tl.int32)
        return v != 0

    @triton.autotune(
        configs=_ATTN_BWD_DQ_CONFIGS,
        key=["N_Q", "N_K", "HEAD_DIM", "IS_CAUSAL", "USE_BLOCK_MASK"],
    )
    @triton.jit
    def _flash_ra_attn_bwd_dq_kernel(
        q_ptr, k_ptr, v_ptr, do_ptr, dq_ptr,
        m_ptr, l_ptr, m_idx_ptr, d_ptr, c_ptr, active_mask_ptr,
        stride_qb, stride_qh, stride_qn,
        stride_kb, stride_kh, stride_kn,
        stride_vb, stride_vh, stride_vn,
        stride_dob, stride_doh, stride_don,
        H_Q, N_Q, N_K, num_m64, num_n32,
        sm_scale: tl.constexpr,
        ra_scale: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        EVEN_D: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        USE_APPROX_DIV: tl.constexpr,
        USE_BLOCK_MASK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        bh_q = tl.program_id(0)
        pid_m = tl.program_id(1)
        start_m = (tl.num_programs(1) - 1 - pid_m) if IS_CAUSAL else pid_m

        b = (bh_q // H_Q).to(tl.int64)
        h_q = (bh_q % H_Q).to(tl.int64)
        h_kv = h_q // GROUP_SIZE

        q_base = q_ptr + b * stride_qb + h_q * stride_qh
        k_base = k_ptr + b * stride_kb + h_kv * stride_kh
        v_base = v_ptr + b * stride_vb + h_kv * stride_vh
        do_base = do_ptr + b * stride_dob + h_q * stride_doh
        dq_base = dq_ptr + bh_q.to(tl.int64) * (N_Q * HEAD_DIM)
        stat_offset = bh_q.to(tl.int64) * N_Q
        am_base = active_mask_ptr + bh_q.to(tl.int64) * (num_m64.to(tl.int64) * num_n32.to(tl.int64))
        m64_idx = start_m * (BLOCK_M // 64)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < N_Q
        mask_d = offs_d < HEAD_DIM
        q_mask = mask_m[:, None] if EVEN_D else (mask_m[:, None] & mask_d[None, :])

        q = tl.load(q_base + offs_m[:, None] * stride_qn + offs_d[None, :], mask=q_mask, other=0.0)
        do = tl.load(do_base + offs_m[:, None] * stride_don + offs_d[None, :], mask=q_mask, other=0.0)
        m_i = tl.load(m_ptr + stat_offset + offs_m, mask=mask_m, other=-float("inf")).to(tl.float32)
        l_i = tl.load(l_ptr + stat_offset + offs_m, mask=mask_m, other=1.0).to(tl.float32)
        coeff_i = (16.0 * ra_scale) / l_i
        m_idx_i = tl.load(m_idx_ptr + stat_offset + offs_m, mask=mask_m, other=N_K).to(tl.int32)
        d_i = tl.load(d_ptr + stat_offset + offs_m, mask=mask_m, other=0.0).to(tl.float32)

        qk_scale = sm_scale * ra_scale
        m_scaled = tl.where(m_i > -float("inf"), m_i * ra_scale, float("inf"))

        hi = tl.minimum(N_K, (start_m + 1) * BLOCK_M) if IS_CAUSAL else N_K
        n_k_full = (N_K // BLOCK_N) * BLOCK_N
        if IS_CAUSAL:
            diag_start = (start_m * BLOCK_M // BLOCK_N) * BLOCK_N
            hi_full = tl.minimum(n_k_full, diag_start)
        else:
            hi_full = n_k_full

        c_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
        dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for start_n in range(0, hi_full, BLOCK_N):
            if (not USE_BLOCK_MASK) or _is_block_active(
                am_base, m64_idx, start_n // 32, num_m64, num_n32, BLOCK_M // 64, BLOCK_N // 32
            ):
                c_acc, dq_acc = _ra_bwd_dq_step(
                    q, do, k_base, v_base, stride_kn, stride_vn, offs_m, offs_d, mask_m, mask_d,
                    start_n, N_K, qk_scale, m_scaled, coeff_i, d_i, c_acc, dq_acc,
                    BLOCK_N=BLOCK_N, EVEN_D=EVEN_D, MASK_N=False, MASK_CAUSAL=False,
                    USE_APPROX_DIV=USE_APPROX_DIV,
                )
        for start_n in range(hi_full, hi, BLOCK_N):
            if (not USE_BLOCK_MASK) or _is_block_active(
                am_base, m64_idx, start_n // 32, num_m64, num_n32, BLOCK_M // 64, BLOCK_N // 32
            ):
                c_acc, dq_acc = _ra_bwd_dq_step(
                    q, do, k_base, v_base, stride_kn, stride_vn, offs_m, offs_d, mask_m, mask_d,
                    start_n, N_K, qk_scale, m_scaled, coeff_i, d_i, c_acc, dq_acc,
                    BLOCK_N=BLOCK_N, EVEN_D=EVEN_D, MASK_N=True, MASK_CAUSAL=IS_CAUSAL,
                    USE_APPROX_DIV=USE_APPROX_DIV,
                )

        tl.store(c_ptr + stat_offset + offs_m, c_acc, mask=mask_m)

        has_argmax = mask_m & (m_idx_i >= 0) & (m_idx_i < N_K)
        safe_m_idx = tl.where(has_argmax, m_idx_i, 0)
        argmax_mask = has_argmax[:, None] if EVEN_D else (has_argmax[:, None] & mask_d[None, :])
        k_argmax = tl.load(
            k_base + safe_m_idx[:, None] * stride_kn + offs_d[None, :],
            mask=argmax_mask,
            other=0.0,
        ).to(tl.float32)
        dq_out = (dq_acc - c_acc[:, None] * k_argmax) * sm_scale
        tl.store(dq_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :], dq_out, mask=q_mask)

    @triton.jit
    def _ra_bwd_dkdv_step(
        k,
        v,
        q_base,
        do_base,
        stride_qn,
        stride_don,
        m_ptr,
        l_ptr,
        m_idx_ptr,
        d_ptr,
        c_ptr,
        stat_offset,
        offs_n,
        offs_d,
        mask_n,
        mask_d,
        start_m,
        N_Q,
        N_K,
        sm_scale,
        ra_scale,
        qk_scale,
        dk_acc,
        dv_acc,
        BLOCK_M: tl.constexpr,
        EVEN_D: tl.constexpr,
        MASK_M: tl.constexpr,
        MASK_CAUSAL: tl.constexpr,
        USE_APPROX_DIV: tl.constexpr,
    ):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        q_ptrs = q_base + offs_m[:, None] * stride_qn + offs_d[None, :]
        do_ptrs = do_base + offs_m[:, None] * stride_don + offs_d[None, :]
        if MASK_M:
            mask_m = offs_m < N_Q
            qm_mask = mask_m[:, None] if EVEN_D else (mask_m[:, None] & mask_d[None, :])
            q = tl.load(q_ptrs, mask=qm_mask, other=0.0)
            do = tl.load(do_ptrs, mask=qm_mask, other=0.0)
            m_i = tl.load(m_ptr + stat_offset + offs_m, mask=mask_m, other=-float("inf")).to(tl.float32)
            l_i = tl.load(l_ptr + stat_offset + offs_m, mask=mask_m, other=1.0).to(tl.float32)
            m_idx_i = tl.load(m_idx_ptr + stat_offset + offs_m, mask=mask_m, other=N_K).to(tl.int32)
            d_i = tl.load(d_ptr + stat_offset + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            c_i = tl.load(c_ptr + stat_offset + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        else:
            if EVEN_D:
                q = tl.load(q_ptrs)
                do = tl.load(do_ptrs)
            else:
                q = tl.load(q_ptrs, mask=mask_d[None, :], other=0.0)
                do = tl.load(do_ptrs, mask=mask_d[None, :], other=0.0)
            m_i = tl.load(m_ptr + stat_offset + offs_m).to(tl.float32)
            l_i = tl.load(l_ptr + stat_offset + offs_m).to(tl.float32)
            m_idx_i = tl.load(m_idx_ptr + stat_offset + offs_m).to(tl.int32)
            d_i = tl.load(d_ptr + stat_offset + offs_m).to(tl.float32)
            c_i = tl.load(c_ptr + stat_offset + offs_m).to(tl.float32)

        inv_l_i = 1.0 / l_i
        coeff_i = (16.0 * ra_scale) * inv_l_i
        m_scaled = tl.where(m_i > -float("inf"), m_i * ra_scale, float("inf"))
        u = tl.maximum(m_scaled[:, None] - tl.dot(q, tl.trans(k)) * qk_scale, 0.0)

        if MASK_M or MASK_CAUSAL:
            valid = (offs_m[:, None] < N_Q) & mask_n[None, :]
            if MASK_CAUSAL:
                valid = valid & (offs_m[:, None] >= offs_n[None, :])
            active = valid & (u < 1.0)
            is_first_max = valid & (offs_n[None, :] == m_idx_i[:, None])
        else:
            active = u < 1.0
            is_first_max = offs_n[None, :] == m_idx_i[:, None]

        safe_u = tl.where(active, u, 0.0)
        if USE_APPROX_DIV:
            inv_1pu = tl.fdiv(1.0, 1.0 + safe_u, approx=True)
        else:
            inv_1pu = tl.fdiv(1.0, 1.0 + safe_u)
        q_rat = (1.0 - safe_u) * inv_1pu
        q2 = q_rat * q_rat
        q4 = q2 * q2
        q7 = q_rat * q2 * q4
        p = tl.where(active, (q4 * q4) * inv_l_i[:, None], 0.0)

        dv_acc += tl.dot(tl.trans(p.to(do.dtype)), do)
        dp = tl.dot(do, tl.trans(v))
        pw = (coeff_i[:, None] * q7) * (inv_1pu * inv_1pu)
        ds_direct = tl.where(active, pw * (dp - d_i[:, None]), 0.0)
        ds = (ds_direct - tl.where(is_first_max, c_i[:, None], 0.0)) * sm_scale
        dk_acc += tl.dot(tl.trans(ds.to(q.dtype)), q)
        return dk_acc, dv_acc

    @triton.autotune(
        configs=_ATTN_BWD_DKDV_CONFIGS,
        key=["N_Q", "N_K", "HEAD_DIM", "IS_CAUSAL", "USE_BLOCK_MASK"],
    )
    @triton.jit
    def _flash_ra_attn_bwd_dkdv_kernel(
        q_ptr, k_ptr, v_ptr, do_ptr, dk_ptr, dv_ptr,
        m_ptr, l_ptr, m_idx_ptr, d_ptr, c_ptr, active_mask_ptr,
        stride_qb, stride_qh, stride_qn,
        stride_kb, stride_kh, stride_kn,
        stride_vb, stride_vh, stride_vn,
        stride_dob, stride_doh, stride_don,
        H_Q, N_Q, N_K, num_m64, num_n32,
        sm_scale: tl.constexpr,
        ra_scale: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        EVEN_D: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        USE_APPROX_DIV: tl.constexpr,
        USE_BLOCK_MASK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # Parallelize across all B * H_Q heads so GQA never serializes query groups inside one CTA.
        bh_q = tl.program_id(0)
        start_n = tl.program_id(1)
        b = (bh_q // H_Q).to(tl.int64)
        h_q = (bh_q % H_Q).to(tl.int64)
        h_kv = h_q // GROUP_SIZE

        q_base = q_ptr + b * stride_qb + h_q * stride_qh
        do_base = do_ptr + b * stride_dob + h_q * stride_doh
        k_base = k_ptr + b * stride_kb + h_kv * stride_kh
        v_base = v_ptr + b * stride_vb + h_kv * stride_vh
        dk_base = dk_ptr + bh_q.to(tl.int64) * (N_K * HEAD_DIM)
        dv_base = dv_ptr + bh_q.to(tl.int64) * (N_K * HEAD_DIM)
        stat_offset = bh_q.to(tl.int64) * N_Q
        am_base = active_mask_ptr + bh_q.to(tl.int64) * (num_m64.to(tl.int64) * num_n32.to(tl.int64))
        n32_base = start_n * (BLOCK_N // 32)

        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        mask_n = offs_n < N_K
        mask_d = offs_d < HEAD_DIM
        kv_mask = mask_n[:, None] if EVEN_D else (mask_n[:, None] & mask_d[None, :])

        k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :], mask=kv_mask, other=0.0)
        v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :], mask=kv_mask, other=0.0)

        dk_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

        qk_scale = sm_scale * ra_scale
        lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M if IS_CAUSAL else 0
        bulk_start = (((start_n + 1) * BLOCK_N + BLOCK_M - 1) // BLOCK_M) * BLOCK_M if IS_CAUSAL else 0
        bulk_end = (N_Q // BLOCK_M) * BLOCK_M
        diag_end = tl.minimum(N_Q, bulk_start)
        tail_start = tl.maximum(bulk_start, bulk_end)

        if IS_CAUSAL:
            for start_m in range(lo, diag_end, BLOCK_M):
                if (not USE_BLOCK_MASK) or _is_block_active(
                    am_base, start_m // 64, n32_base, num_m64, num_n32, 1, BLOCK_N // 32
                ):
                    dk_acc, dv_acc = _ra_bwd_dkdv_step(
                        k, v, q_base, do_base, stride_qn, stride_don,
                        m_ptr, l_ptr, m_idx_ptr, d_ptr, c_ptr, stat_offset,
                        offs_n, offs_d, mask_n, mask_d, start_m, N_Q, N_K,
                        sm_scale, ra_scale, qk_scale, dk_acc, dv_acc,
                        BLOCK_M=BLOCK_M, EVEN_D=EVEN_D, MASK_M=True, MASK_CAUSAL=True,
                        USE_APPROX_DIV=USE_APPROX_DIV,
                    )
        for start_m in range(bulk_start, bulk_end, BLOCK_M):
            if (not USE_BLOCK_MASK) or _is_block_active(
                am_base, start_m // 64, n32_base, num_m64, num_n32, 1, BLOCK_N // 32
            ):
                dk_acc, dv_acc = _ra_bwd_dkdv_step(
                    k, v, q_base, do_base, stride_qn, stride_don,
                    m_ptr, l_ptr, m_idx_ptr, d_ptr, c_ptr, stat_offset,
                    offs_n, offs_d, mask_n, mask_d, start_m, N_Q, N_K,
                    sm_scale, ra_scale, qk_scale, dk_acc, dv_acc,
                    BLOCK_M=BLOCK_M, EVEN_D=EVEN_D, MASK_M=False, MASK_CAUSAL=False,
                    USE_APPROX_DIV=USE_APPROX_DIV,
                )
        for start_m in range(tail_start, N_Q, BLOCK_M):
            if (not USE_BLOCK_MASK) or _is_block_active(
                am_base, start_m // 64, n32_base, num_m64, num_n32, 1, BLOCK_N // 32
            ):
                dk_acc, dv_acc = _ra_bwd_dkdv_step(
                    k, v, q_base, do_base, stride_qn, stride_don,
                    m_ptr, l_ptr, m_idx_ptr, d_ptr, c_ptr, stat_offset,
                    offs_n, offs_d, mask_n, mask_d, start_m, N_Q, N_K,
                    sm_scale, ra_scale, qk_scale, dk_acc, dv_acc,
                    BLOCK_M=BLOCK_M, EVEN_D=EVEN_D, MASK_M=True, MASK_CAUSAL=False,
                    USE_APPROX_DIV=USE_APPROX_DIV,
                )

        tl.store(dk_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :], dk_acc, mask=kv_mask)
        tl.store(dv_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :], dv_acc, mask=kv_mask)
else:
    _flash_ra_attn_bwd_preprocess_kernel = None
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
    active_mask: Optional[Any] = None,
) -> Optional[Tuple[Any, Any, Any]]:
    """Dispatches fused tiled Rational Attention backward pass to Triton SRAM kernels."""
    if not (
        HAS_TRITON
        and _HAS_TORCH
        and isinstance(q, torch.Tensor)
        and q.is_cuda
        and q.ndim == 4
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and k.dtype == q.dtype
        and v.dtype == q.dtype
    ):
        return None

    B, H, N_Q, D = q.shape
    H_KV = k.shape[1]
    N_K = k.shape[2]
    if (
        B == 0
        or N_Q == 0
        or N_K == 0
        or D > 256
        or k.shape[-1] != D
        or v.shape[-1] != D
        or v.shape[1] != H_KV
        or H_KV <= 0
        or (H % H_KV != 0)
    ):
        return None

    GROUP_SIZE = H // H_KV
    BLOCK_D = max(16, triton.next_power_of_2(D))
    EVEN_D = D == BLOCK_D

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    q_c = q if q.stride(-1) == 1 else q.contiguous()
    k_c = k if k.stride(-1) == 1 else k.contiguous()
    v_c = v if v.stride(-1) == 1 else v.contiguous()
    do_in = grad_out if grad_out.dtype == q_c.dtype else grad_out.to(q_c.dtype)
    do_c = do_in if do_in.stride(-1) == 1 else do_in.contiguous()
    out_c = out if out.stride(-1) == 1 else out.contiguous()
    m_c = m if m.is_contiguous() else m.contiguous()
    l_c = l if l.is_contiguous() else l.contiguous()
    m_idx_c = m_idx if m_idx.is_contiguous() else m_idx.contiguous()

    num_m64 = triton.cdiv(N_Q, 64)
    num_n32 = triton.cdiv(N_K, 32)
    use_block_mask = bool(
        active_mask is not None
        and isinstance(active_mask, torch.Tensor)
        and active_mask.is_cuda
        and active_mask.dtype == torch.uint8
        and active_mask.shape == (B * H, num_m64, num_n32)
    )
    am_c = (
        (active_mask if active_mask.is_contiguous() else active_mask.contiguous())
        if use_block_mask
        else torch.empty((1,), device=q_c.device, dtype=torch.uint8)
    )

    d_vec = torch.empty((B, H, N_Q), device=q_c.device, dtype=torch.float32)
    c_vec = torch.empty_like(d_vec)
    _flash_ra_attn_bwd_preprocess_kernel[(B * H, triton.cdiv(N_Q, 64))](
        do_c, out_c, d_vec,
        do_c.stride(0), do_c.stride(1), do_c.stride(2),
        out_c.stride(0), out_c.stride(1), out_c.stride(2),
        H, N_Q, HEAD_DIM=D, BLOCK_D=BLOCK_D, EVEN_D=EVEN_D, BLOCK_M=64, num_warps=4,
    )

    dq = torch.empty(q_c.shape, device=q_c.device, dtype=q_c.dtype)
    dk_buf = torch.empty((B, H, N_K, D), device=k_c.device, dtype=k_c.dtype)
    dv_buf = torch.empty((B, H, N_K, D), device=v_c.device, dtype=v_c.dtype)

    strides = (
        q_c.stride(0), q_c.stride(1), q_c.stride(2),
        k_c.stride(0), k_c.stride(1), k_c.stride(2),
        v_c.stride(0), v_c.stride(1), v_c.stride(2),
        do_c.stride(0), do_c.stride(1), do_c.stride(2),
    )
    shared_kw = dict(
        sm_scale=float(sm_scale),
        ra_scale=float(ra_scale),
        IS_CAUSAL=bool(is_causal),
        HEAD_DIM=D,
        BLOCK_D=BLOCK_D,
        EVEN_D=EVEN_D,
        GROUP_SIZE=GROUP_SIZE,
        USE_APPROX_DIV=HAS_FDIV_APPROX,
        USE_BLOCK_MASK=use_block_mask,
    )
    _flash_ra_attn_bwd_dq_kernel[lambda meta: (B * H, triton.cdiv(N_Q, meta["BLOCK_M"]))](
        q_c, k_c, v_c, do_c, dq, m_c, l_c, m_idx_c, d_vec, c_vec, am_c,
        *strides, H, N_Q, N_K, num_m64, num_n32, **shared_kw,
    )
    _flash_ra_attn_bwd_dkdv_kernel[lambda meta: (B * H, triton.cdiv(N_K, meta["BLOCK_N"]))](
        q_c, k_c, v_c, do_c, dk_buf, dv_buf, m_c, l_c, m_idx_c, d_vec, c_vec, am_c,
        *strides, H, N_Q, N_K, num_m64, num_n32, **shared_kw,
    )
    if GROUP_SIZE == 1:
        return dq, dk_buf, dv_buf
    dk = dk_buf.view(B, H_KV, GROUP_SIZE, N_K, D).sum(dim=2, dtype=torch.float32).to(k_c.dtype)
    dv = dv_buf.view(B, H_KV, GROUP_SIZE, N_K, D).sum(dim=2, dtype=torch.float32).to(v_c.dtype)
    return dq, dk, dv
