from __future__ import annotations

from collections import OrderedDict
import math
from typing import TYPE_CHECKING, Any, Optional, Tuple

from ._base import HAS_FDIV_APPROX, HAS_TRITON, _HAS_TORCH, _dev_idx, _int_spec, _launch_attn_bwd_cached, _ptr_aligned, tl, torch, triton
from .attention_fwd import _compute_causal_valid, _compute_decay_bias

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
        doc_base,
        doc_m,
        offs_m,
        offs_d,
        mask_m,
        mask_d,
        start_n,
        N_K,
        qk_scale,
        ra_scale,
        s_h,
        h_pen,
        hub_size,
        q_offset,
        m_scaled,
        coeff_i,
        d_i,
        c_acc,
        dq_acc,
        BLOCK_N: tl.constexpr,
        EVEN_D: tl.constexpr,
        MASK_N: tl.constexpr,
        MASK_CAUSAL: tl.constexpr,
        HAS_DOC_IDS: tl.constexpr,
        HAS_DECAY_BIAS: tl.constexpr,
        HAS_HUB: tl.constexpr,
        BIAS_IS_RATIONAL: tl.constexpr,
        CHECK_HUB_Q: tl.constexpr,
        USE_APPROX_DIV: tl.constexpr,
    ):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        if HAS_DOC_IDS:
            doc_n = tl.load(doc_base + offs_n, mask=(offs_n < N_K), other=0)
            doc_match = (doc_m[:, None] == doc_n[None, :]) & (doc_m[:, None] > 0)
            has_overlap = tl.sum(doc_match.to(tl.int32)) > 0
        else:
            has_overlap = True

        if has_overlap:
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

            score_scaled = tl.dot(q, tl.trans(k)) * qk_scale
            if HAS_DECAY_BIAS:
                bias = _compute_decay_bias(
                    offs_m, offs_n, q_offset, s_h, h_pen, hub_size,
                    HAS_HUB=HAS_HUB, BIAS_IS_RATIONAL=BIAS_IS_RATIONAL,
                )
                score_scaled = score_scaled + bias * ra_scale
            u = tl.maximum(m_scaled[:, None] - score_scaled, 0.0)
            if MASK_N or MASK_CAUSAL or HAS_DOC_IDS:
                valid = mask_m[:, None] & (offs_n[None, :] < N_K)
                if MASK_CAUSAL:
                    valid = valid & _compute_causal_valid(
                        offs_m, offs_n, q_offset, hub_size, HAS_HUB=HAS_HUB, CHECK_HUB_Q=CHECK_HUB_Q
                    )
                if HAS_DOC_IDS:
                    valid = valid & doc_match
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

    @triton.autotune(
        configs=_ATTN_BWD_DQ_CONFIGS,
        key=["N_Q", "N_K", "HEAD_DIM", "IS_CAUSAL", "HAS_DOC_IDS", "HAS_DECAY_BIAS", "HAS_HUB"],
    )
    @triton.jit
    def _flash_ra_attn_bwd_dq_kernel(
        q_ptr, k_ptr, v_ptr, do_ptr, dq_ptr, m_ptr, l_ptr, m_idx_ptr, d_ptr, c_ptr, doc_ids_ptr,
        bias_slopes_ptr, hub_penalty_ptr,
        stride_qb, stride_qh, stride_qn, stride_kb, stride_kh, stride_kn,
        stride_vb, stride_vh, stride_vn, stride_dob, stride_doh, stride_don, stride_db,
        H_Q, N_Q, N_K, hub_size, q_offset,
        sm_scale: tl.constexpr, ra_scale: tl.constexpr, IS_CAUSAL: tl.constexpr, HAS_DOC_IDS: tl.constexpr,
        HAS_DECAY_BIAS: tl.constexpr, HAS_HUB: tl.constexpr, BIAS_IS_RATIONAL: tl.constexpr, CHECK_HUB_Q: tl.constexpr,
        HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr, EVEN_D: tl.constexpr, GROUP_SIZE: tl.constexpr,
        USE_APPROX_DIV: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
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
        doc_base = doc_ids_ptr + b * stride_db
        stat_offset = bh_q.to(tl.int64) * N_Q

        s_h = tl.load(bias_slopes_ptr + h_q).to(tl.float32) if HAS_DECAY_BIAS else 0.0
        h_pen = tl.load(hub_penalty_ptr + h_q).to(tl.float32) if (HAS_DECAY_BIAS and HAS_HUB) else 0.0

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < N_Q
        mask_d = offs_d < HEAD_DIM
        q_mask = mask_m[:, None] if EVEN_D else (mask_m[:, None] & mask_d[None, :])

        q = tl.load(q_base + offs_m[:, None] * stride_qn + offs_d[None, :], mask=q_mask, other=0.0)
        do = tl.load(do_base + offs_m[:, None] * stride_don + offs_d[None, :], mask=q_mask, other=0.0)
        doc_m = tl.load(doc_base + offs_m, mask=mask_m, other=0) if HAS_DOC_IDS else tl.zeros([BLOCK_M], dtype=tl.int32)
        m_i = tl.load(m_ptr + stat_offset + offs_m, mask=mask_m, other=-float("inf")).to(tl.float32)
        l_i = tl.load(l_ptr + stat_offset + offs_m, mask=mask_m, other=1.0).to(tl.float32)
        coeff_i = (16.0 * ra_scale) / l_i
        m_idx_i = tl.load(m_idx_ptr + stat_offset + offs_m, mask=mask_m, other=N_K).to(tl.int32)
        d_i = tl.load(d_ptr + stat_offset + offs_m, mask=mask_m, other=0.0).to(tl.float32)

        qk_scale = sm_scale * ra_scale
        m_scaled = tl.where(m_i > -float("inf"), m_i * ra_scale, float("inf"))

        n_k_full = (N_K // BLOCK_N) * BLOCK_N
        if IS_CAUSAL:
            max_k_reach = (start_m + 1) * BLOCK_M + q_offset
            hi = tl.minimum(N_K, tl.maximum(hub_size, max_k_reach) if HAS_HUB else tl.maximum(0, max_k_reach))
            diag_start = (tl.maximum(0, start_m * BLOCK_M + q_offset) // BLOCK_N) * BLOCK_N
            hi_full = tl.minimum(n_k_full, diag_start)
            if CHECK_HUB_Q:
                hi_full = tl.where(start_m * BLOCK_M < hub_size, 0, hi_full)
        else:
            hi = N_K
            hi_full = n_k_full

        c_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
        dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        run_loops = (tl.max(doc_m, axis=0) > 0) if HAS_DOC_IDS else True
        if run_loops:
            for start_n in range(0, hi_full, BLOCK_N):
                c_acc, dq_acc = _ra_bwd_dq_step(
                    q, do, k_base, v_base, stride_kn, stride_vn, doc_base, doc_m, offs_m, offs_d, mask_m, mask_d,
                    start_n, N_K, qk_scale, ra_scale, s_h, h_pen, hub_size, q_offset, m_scaled, coeff_i, d_i, c_acc, dq_acc,
                    BLOCK_N=BLOCK_N, EVEN_D=EVEN_D, MASK_N=False, MASK_CAUSAL=False,
                    HAS_DOC_IDS=HAS_DOC_IDS, HAS_DECAY_BIAS=HAS_DECAY_BIAS, HAS_HUB=HAS_HUB,
                    BIAS_IS_RATIONAL=BIAS_IS_RATIONAL, CHECK_HUB_Q=CHECK_HUB_Q, USE_APPROX_DIV=USE_APPROX_DIV,
                )
            for start_n in range(hi_full, hi, BLOCK_N):
                c_acc, dq_acc = _ra_bwd_dq_step(
                    q, do, k_base, v_base, stride_kn, stride_vn, doc_base, doc_m, offs_m, offs_d, mask_m, mask_d,
                    start_n, N_K, qk_scale, ra_scale, s_h, h_pen, hub_size, q_offset, m_scaled, coeff_i, d_i, c_acc, dq_acc,
                    BLOCK_N=BLOCK_N, EVEN_D=EVEN_D, MASK_N=True, MASK_CAUSAL=IS_CAUSAL,
                    HAS_DOC_IDS=HAS_DOC_IDS, HAS_DECAY_BIAS=HAS_DECAY_BIAS, HAS_HUB=HAS_HUB,
                    BIAS_IS_RATIONAL=BIAS_IS_RATIONAL, CHECK_HUB_Q=CHECK_HUB_Q, USE_APPROX_DIV=USE_APPROX_DIV,
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
        k, v, q_base, do_base, stride_qn, stride_don, doc_base, doc_n,
        m_ptr, l_ptr, m_idx_ptr, d_ptr, c_ptr, stat_offset,
        offs_n, offs_d, mask_n, mask_d, start_m, N_Q, N_K,
        sm_scale, ra_scale, qk_scale, s_h, h_pen, hub_size, q_offset, dk_acc, dv_acc,
        BLOCK_M: tl.constexpr, EVEN_D: tl.constexpr, MASK_M: tl.constexpr,
        MASK_CAUSAL: tl.constexpr, HAS_DOC_IDS: tl.constexpr,
        HAS_DECAY_BIAS: tl.constexpr, HAS_HUB: tl.constexpr, BIAS_IS_RATIONAL: tl.constexpr,
        CHECK_HUB_Q: tl.constexpr, USE_APPROX_DIV: tl.constexpr,
    ):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        if HAS_DOC_IDS:
            doc_m = tl.load(doc_base + offs_m, mask=(offs_m < N_Q), other=0)
            doc_match = (doc_m[:, None] == doc_n[None, :]) & (doc_m[:, None] > 0)
            has_overlap = tl.sum(doc_match.to(tl.int32)) > 0
        else:
            has_overlap = True

        if has_overlap:
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
            score_scaled = tl.dot(q, tl.trans(k)) * qk_scale
            if HAS_DECAY_BIAS:
                bias = _compute_decay_bias(
                    offs_m, offs_n, q_offset, s_h, h_pen, hub_size,
                    HAS_HUB=HAS_HUB, BIAS_IS_RATIONAL=BIAS_IS_RATIONAL,
                )
                score_scaled = score_scaled + bias * ra_scale
            u = tl.maximum(m_scaled[:, None] - score_scaled, 0.0)

            if MASK_M or MASK_CAUSAL or HAS_DOC_IDS:
                valid = (offs_m[:, None] < N_Q) & mask_n[None, :]
                if MASK_CAUSAL:
                    valid = valid & _compute_causal_valid(
                        offs_m, offs_n, q_offset, hub_size, HAS_HUB=HAS_HUB, CHECK_HUB_Q=CHECK_HUB_Q
                    )
                if HAS_DOC_IDS:
                    valid = valid & doc_match
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
        key=["N_Q", "N_K", "HEAD_DIM", "IS_CAUSAL", "HAS_DOC_IDS", "HAS_DECAY_BIAS", "HAS_HUB"],
    )
    @triton.jit
    def _flash_ra_attn_bwd_dkdv_kernel(
        q_ptr, k_ptr, v_ptr, do_ptr, dk_ptr, dv_ptr, m_ptr, l_ptr, m_idx_ptr, d_ptr, c_ptr, doc_ids_ptr,
        bias_slopes_ptr, hub_penalty_ptr,
        stride_qb, stride_qh, stride_qn, stride_kb, stride_kh, stride_kn,
        stride_vb, stride_vh, stride_vn, stride_dob, stride_doh, stride_don, stride_db,
        H_Q, N_Q, N_K, hub_size, q_offset,
        sm_scale: tl.constexpr, ra_scale: tl.constexpr, IS_CAUSAL: tl.constexpr, HAS_DOC_IDS: tl.constexpr,
        HAS_DECAY_BIAS: tl.constexpr, HAS_HUB: tl.constexpr, BIAS_IS_RATIONAL: tl.constexpr, CHECK_HUB_Q: tl.constexpr,
        HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr, EVEN_D: tl.constexpr, GROUP_SIZE: tl.constexpr,
        USE_APPROX_DIV: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
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
        doc_base = doc_ids_ptr + b * stride_db
        stat_offset = bh_q.to(tl.int64) * N_Q

        s_h = tl.load(bias_slopes_ptr + h_q).to(tl.float32) if HAS_DECAY_BIAS else 0.0
        h_pen = tl.load(hub_penalty_ptr + h_q).to(tl.float32) if (HAS_DECAY_BIAS and HAS_HUB) else 0.0

        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        mask_n = offs_n < N_K
        mask_d = offs_d < HEAD_DIM
        kv_mask = mask_n[:, None] if EVEN_D else (mask_n[:, None] & mask_d[None, :])

        k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :], mask=kv_mask, other=0.0)
        v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :], mask=kv_mask, other=0.0)
        doc_n = tl.load(doc_base + offs_n, mask=mask_n, other=0) if HAS_DOC_IDS else tl.zeros([BLOCK_N], dtype=tl.int32)

        dk_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

        qk_scale = sm_scale * ra_scale
        bulk_end = (N_Q // BLOCK_M) * BLOCK_M
        if IS_CAUSAL:
            col_start = start_n * BLOCK_N
            col_end = (start_n + 1) * BLOCK_N
            min_i = tl.maximum(0, col_start - q_offset)
            full_i = tl.maximum(0, col_end - q_offset)
            if CHECK_HUB_Q:
                min_i = tl.maximum(min_i, hub_size)
                full_i = tl.maximum(full_i, hub_size)
            if HAS_HUB:
                min_i = tl.where(col_start < hub_size, 0, min_i)
            lo = (min_i // BLOCK_M) * BLOCK_M
            bulk_start = ((full_i + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
            diag_end = tl.minimum(N_Q, bulk_start)
            if HAS_HUB:
                is_pure_hub = col_end <= hub_size
                lo = tl.where(is_pure_hub, 0, lo)
                bulk_start = tl.where(is_pure_hub, 0, bulk_start)
                diag_end = tl.where(is_pure_hub, 0, diag_end)
            tail_start = tl.maximum(bulk_start, bulk_end)
        else:
            lo = 0
            bulk_start = 0
            diag_end = 0
            tail_start = bulk_end

        run_loops = (tl.max(doc_n, axis=0) > 0) if HAS_DOC_IDS else True
        if run_loops:
            if IS_CAUSAL:
                for start_m in range(lo, diag_end, BLOCK_M):
                    dk_acc, dv_acc = _ra_bwd_dkdv_step(
                        k, v, q_base, do_base, stride_qn, stride_don, doc_base, doc_n,
                        m_ptr, l_ptr, m_idx_ptr, d_ptr, c_ptr, stat_offset,
                        offs_n, offs_d, mask_n, mask_d, start_m, N_Q, N_K,
                        sm_scale, ra_scale, qk_scale, s_h, h_pen, hub_size, q_offset, dk_acc, dv_acc,
                        BLOCK_M=BLOCK_M, EVEN_D=EVEN_D, MASK_M=True, MASK_CAUSAL=True,
                        HAS_DOC_IDS=HAS_DOC_IDS, HAS_DECAY_BIAS=HAS_DECAY_BIAS, HAS_HUB=HAS_HUB,
                        BIAS_IS_RATIONAL=BIAS_IS_RATIONAL, CHECK_HUB_Q=CHECK_HUB_Q, USE_APPROX_DIV=USE_APPROX_DIV,
                    )
            for start_m in range(bulk_start, bulk_end, BLOCK_M):
                dk_acc, dv_acc = _ra_bwd_dkdv_step(
                    k, v, q_base, do_base, stride_qn, stride_don, doc_base, doc_n,
                    m_ptr, l_ptr, m_idx_ptr, d_ptr, c_ptr, stat_offset,
                    offs_n, offs_d, mask_n, mask_d, start_m, N_Q, N_K,
                    sm_scale, ra_scale, qk_scale, s_h, h_pen, hub_size, q_offset, dk_acc, dv_acc,
                    BLOCK_M=BLOCK_M, EVEN_D=EVEN_D, MASK_M=False, MASK_CAUSAL=False,
                    HAS_DOC_IDS=HAS_DOC_IDS, HAS_DECAY_BIAS=HAS_DECAY_BIAS, HAS_HUB=HAS_HUB,
                    BIAS_IS_RATIONAL=BIAS_IS_RATIONAL, CHECK_HUB_Q=CHECK_HUB_Q, USE_APPROX_DIV=USE_APPROX_DIV,
                )
            for start_m in range(tail_start, N_Q, BLOCK_M):
                dk_acc, dv_acc = _ra_bwd_dkdv_step(
                    k, v, q_base, do_base, stride_qn, stride_don, doc_base, doc_n,
                    m_ptr, l_ptr, m_idx_ptr, d_ptr, c_ptr, stat_offset,
                    offs_n, offs_d, mask_n, mask_d, start_m, N_Q, N_K,
                    sm_scale, ra_scale, qk_scale, s_h, h_pen, hub_size, q_offset, dk_acc, dv_acc,
                    BLOCK_M=BLOCK_M, EVEN_D=EVEN_D, MASK_M=True, MASK_CAUSAL=False,
                    HAS_DOC_IDS=HAS_DOC_IDS, HAS_DECAY_BIAS=HAS_DECAY_BIAS, HAS_HUB=HAS_HUB,
                    BIAS_IS_RATIONAL=BIAS_IS_RATIONAL, CHECK_HUB_Q=CHECK_HUB_Q, USE_APPROX_DIV=USE_APPROX_DIV,
                )

        tl.store(dk_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :], dk_acc, mask=kv_mask)
        tl.store(dv_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :], dv_acc, mask=kv_mask)
else:
    _flash_ra_attn_bwd_preprocess_kernel = None
    _flash_ra_attn_bwd_dq_kernel = None
    _flash_ra_attn_bwd_dkdv_kernel = None


_ATTN_BWD_DQ_CFG_CACHE: OrderedDict[Tuple[Any, ...], Tuple[Any, int]] = OrderedDict()
_ATTN_BWD_DKDV_CFG_CACHE: OrderedDict[Tuple[Any, ...], Tuple[Any, int]] = OrderedDict()


def triton_flash_ra_attention_backward(
    grad_out: Any, q: Any, k: Any, v: Any, out: Any, m: Any, l: Any, m_idx: Any,
    is_causal: bool = False, sm_scale: Optional[float] = None, ra_scale: float = 0.0625,
    doc_ids: Optional[Any] = None, *, bias_slopes: Optional[Any] = None,
    hub_penalty: Optional[Any] = None, bias_mode: Optional[str] = None,
    hub_size: int = 0, q_offset: int = 0, check_hub_q: bool = False,
) -> Optional[Tuple[Any, Any, Any]]:
    """Dispatches fused tiled Rational Attention backward pass to Triton SRAM kernels."""
    if not (
        HAS_TRITON and _HAS_TORCH
        and all(isinstance(t, torch.Tensor) for t in (q, k, v, grad_out, out, m, l, m_idx))
        and q.is_cuda and all(t.device == q.device for t in (k, v, grad_out, out, m, l, m_idx))
        and q.ndim == 4 and k.ndim == 4 and v.ndim == 4
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and k.dtype == q.dtype and v.dtype == q.dtype and out.dtype == q.dtype
    ):
        return None

    B, H, N_Q, D = q.shape
    H_KV, N_K = k.shape[1], k.shape[2]
    if (
        B <= 0 or H <= 0 or N_Q <= 0 or N_K <= 0 or D <= 0 or D > 256
        or k.shape[0] != B or v.shape[0] != B or v.shape[2] != N_K
        or k.shape[-1] != D or v.shape[-1] != D or v.shape[1] != H_KV
        or H_KV <= 0 or (H % H_KV != 0)
        or grad_out.shape != q.shape or out.shape != q.shape
        or m.shape != (B, H, N_Q) or l.shape != (B, H, N_Q) or m_idx.shape != (B, H, N_Q)
    ):
        return None

    has_doc = doc_ids is not None
    if has_doc:
        if not (isinstance(doc_ids, torch.Tensor) and doc_ids.device == q.device and doc_ids.ndim == 2 and doc_ids.shape == (B, N_Q) and N_Q == N_K):
            return None
        doc_c = doc_ids if (doc_ids.dtype == torch.int32 and doc_ids.stride(-1) == 1) else doc_ids.to(torch.int32).contiguous()
        stride_db = doc_c.stride(0)
    else:
        doc_c = q
        stride_db = 0

    has_decay = bias_slopes is not None
    hub_sz = int(hub_size)
    has_hub = hub_sz > 0
    if has_decay:
        if not (isinstance(bias_slopes, torch.Tensor) and bias_slopes.device == q.device and bias_slopes.numel() == H):
            return None
        slopes_c = bias_slopes.reshape(-1).to(dtype=torch.float32).contiguous()
        if has_hub:
            if not (isinstance(hub_penalty, torch.Tensor) and hub_penalty.device == q.device and hub_penalty.numel() == H):
                return None
            hub_pen_c = hub_penalty.reshape(-1).to(dtype=torch.float32).contiguous()
        else:
            hub_pen_c = slopes_c
        bias_is_rational = str(bias_mode or "linear").lower() == "rational"
    else:
        slopes_c = q
        hub_pen_c = q
        bias_is_rational = False

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

    d_vec = torch.empty((B, H, N_Q), device=q_c.device, dtype=torch.float32)
    c_vec = torch.empty_like(d_vec)
    _flash_ra_attn_bwd_preprocess_kernel[(B * H, triton.cdiv(N_Q, 64))](
        do_c, out_c, d_vec,
        do_c.stride(0), do_c.stride(1), do_c.stride(2),
        out_c.stride(0), out_c.stride(1), out_c.stride(2),
        H, N_Q, HEAD_DIM=D, BLOCK_D=BLOCK_D, EVEN_D=EVEN_D, BLOCK_M=64, num_warps=4, enable_fp_fusion=True,
    )

    dq = torch.empty(q_c.shape, device=q_c.device, dtype=q_c.dtype)
    dk_buf = torch.empty((B, H, N_K, D), device=k_c.device, dtype=k_c.dtype)
    dv_buf = torch.empty((B, H, N_K, D), device=v_c.device, dtype=v_c.dtype)

    strides = (
        q_c.stride(0), q_c.stride(1), q_c.stride(2),
        k_c.stride(0), k_c.stride(1), k_c.stride(2),
        v_c.stride(0), v_c.stride(1), v_c.stride(2),
        do_c.stride(0), do_c.stride(1), do_c.stride(2),
        stride_db,
    )
    shared_kw = dict(
        sm_scale=float(sm_scale), ra_scale=float(ra_scale), IS_CAUSAL=bool(is_causal),
        HAS_DOC_IDS=has_doc, HAS_DECAY_BIAS=has_decay, HAS_HUB=has_hub,
        BIAS_IS_RATIONAL=bias_is_rational, CHECK_HUB_Q=bool(check_hub_q and has_hub),
        HEAD_DIM=D, BLOCK_D=BLOCK_D, EVEN_D=EVEN_D, GROUP_SIZE=GROUP_SIZE,
        USE_APPROX_DIV=HAS_FDIV_APPROX, enable_fp_fusion=True,
    )
    int_spec = tuple(_int_spec(v) for v in (*strides, H, N_Q, N_K, hub_sz, int(q_offset)))
    ptr_ok = _ptr_aligned(q_c, k_c, v_c, do_c, dq, dk_buf, dv_buf, m_c, l_c, m_idx_c, d_vec, c_vec, doc_c, slopes_c, hub_pen_c)
    cfg_key = (
        _dev_idx(q_c.device), N_Q, N_K, D, bool(is_causal), has_doc, has_decay, has_hub, bias_is_rational,
        bool(check_hub_q and has_hub), GROUP_SIZE, float(sm_scale), float(ra_scale),
        q_c.dtype, m_c.dtype, l_c.dtype, m_idx_c.dtype, ptr_ok, int_spec,
    )
    _launch_attn_bwd_cached(
        _flash_ra_attn_bwd_dq_kernel, _ATTN_BWD_DQ_CFG_CACHE, cfg_key, B * H, N_Q, "BLOCK_M",
        (q_c, k_c, v_c, do_c, dq, m_c, l_c, m_idx_c, d_vec, c_vec, doc_c, slopes_c, hub_pen_c, *strides, H, N_Q, N_K, hub_sz, int(q_offset)),
        shared_kw, dev=q_c.device,
    )
    _launch_attn_bwd_cached(
        _flash_ra_attn_bwd_dkdv_kernel, _ATTN_BWD_DKDV_CFG_CACHE, cfg_key, B * H, N_K, "BLOCK_N",
        (q_c, k_c, v_c, do_c, dk_buf, dv_buf, m_c, l_c, m_idx_c, d_vec, c_vec, doc_c, slopes_c, hub_pen_c, *strides, H, N_Q, N_K, hub_sz, int(q_offset)),
        shared_kw, dev=q_c.device,
    )
    if GROUP_SIZE == 1:
        return dq, dk_buf, dv_buf
    dk = dk_buf.view(B, H_KV, GROUP_SIZE, N_K, D).sum(dim=2, dtype=torch.float32).to(k_c.dtype)
    dv = dv_buf.view(B, H_KV, GROUP_SIZE, N_K, D).sum(dim=2, dtype=torch.float32).to(v_c.dtype)
    return dq, dk, dv
