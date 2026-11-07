from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    HAS_TRITON: bool = False
    _ra_norm_fwd_kernel: Any = None
    _pos_ra_glu_fwd_kernel: Any = None
    _ra_softmax_fwd_kernel: Any = None
    _ra_softmax_bwd_kernel: Any = None
    _flash_ra_attn_fwd_kernel: Any = None
    _flash_ra_attn_bwd_dq_kernel: Any = None
    _flash_ra_attn_bwd_dkdv_kernel: Any = None
else:
    try:
        import triton
        import triton.language as tl
        HAS_TRITON = True
    except ImportError:
        HAS_TRITON = False
        triton = None
        tl = None

    if HAS_TRITON:
        @triton.jit
        def _ra_norm_fwd_kernel(
            x_ptr,
            y_ptr,
            weight_ptr,
            bias_ptr,
            stride_x_row,
            stride_y_row,
            D: tl.constexpr,
            eps: tl.constexpr,
            HAS_WEIGHT: tl.constexpr,
            HAS_BIAS: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            row_idx = tl.program_id(0)
            x_row = x_ptr + row_idx * stride_x_row
            y_row = y_ptr + row_idx * stride_y_row

            cols = tl.arange(0, BLOCK_SIZE)
            mask = cols < D

            x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            abs_x = tl.abs(x)

            l1_sum = tl.sum(abs_x, axis=0)
            l1_mean = l1_sum / D
            inv_scale = 1.0 / (l1_mean + eps)

            normed = x * inv_scale

            if HAS_WEIGHT:
                w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
                normed = normed * w

            if HAS_BIAS:
                b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                normed = normed + b

            tl.store(y_row + cols, normed, mask=mask)

        @triton.jit
        def _pos_ra_glu_fwd_kernel(
            gate_ptr,
            up_ptr,
            out_ptr,
            n_elements,
            b_val: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            z = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

            abs_z = tl.abs(z)
            b_term = b_val * (b_val + abs_z)
            ra_tanh = (b_term * z) / (b_term + z * z)
            if b_val == 1.0:
                sigma = 0.5 * (1.0 + ra_tanh)
            else:
                sigma = 0.5 * (1.0 + ra_tanh / b_val)

            out = u * sigma
            tl.store(out_ptr + offsets, out, mask=mask)

        @triton.jit
        def _ra_softmax_fwd_kernel(
            scores_ptr,
            out_ptr,
            stride_row,
            N: tl.constexpr,
            scale: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            row_idx = tl.program_id(0)
            row_scores = scores_ptr + row_idx * stride_row
            row_out = out_ptr + row_idx * stride_row

            cols = tl.arange(0, BLOCK_SIZE)
            mask = cols < N

            s = tl.load(row_scores + cols, mask=mask, other=-float("inf")).to(tl.float32)

            max_s = tl.max(s, axis=0)
            finite = mask & (s > -float("inf")) & (s < float("inf"))
            delta = tl.where(finite, tl.where(s == max_s, 0.0, max_s - s), float("inf"))

            u = delta * scale
            num = tl.maximum(1.0 - u, 0.0)
            den = 1.0 + u
            q = tl.where(finite, num / den, 0.0)

            q2 = q * q
            q4 = q2 * q2
            phi = tl.where(finite, q4 * q4, 0.0)

            sum_phi = tl.sum(phi, axis=0)
            safe_sum = tl.where(sum_phi > 0.0, sum_phi, 1.0)

            p = phi / safe_sum
            tl.store(row_out + cols, p, mask=mask)

        @triton.jit
        def _ra_softmax_bwd_kernel(
            scores_ptr,
            out_ptr,
            grad_out_ptr,
            grad_scores_ptr,
            stride_row,
            N: tl.constexpr,
            scale: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            row_idx = tl.program_id(0)
            offset = row_idx * stride_row

            cols = tl.arange(0, BLOCK_SIZE)
            mask = cols < N

            s = tl.load(scores_ptr + offset + cols, mask=mask, other=-float("inf")).to(tl.float32)
            p = tl.load(out_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
            g = tl.load(grad_out_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)

            max_s = tl.max(s, axis=0)
            finite = mask & (s > -float("inf")) & (s < float("inf"))
            is_max = (s == max_s) & finite
            first_max_col = tl.min(tl.where(is_max, cols, N), axis=0)
            is_first_max = (cols == first_max_col) & (first_max_col < N)

            delta = tl.where(finite, tl.where(s == max_s, 0.0, max_s - s), float("inf"))
            u = delta * scale
            active = finite & (u < 1.0)
            safe_u = tl.where(active, u, 0.0)

            sum_gp = tl.sum(tl.where(finite, g * p, 0.0), axis=0)
            w = (16.0 * scale) / (1.0 - safe_u * safe_u)
            ds_direct = tl.where(active, p * (g - sum_gp) * w, 0.0)

            sum_ds = tl.sum(ds_direct, axis=0)
            ds = ds_direct - tl.where(is_first_max, sum_ds, 0.0)
            tl.store(grad_scores_ptr + offset + cols, ds, mask=mask)

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
        ):
            start_m = tl.program_id(0)
            bh_idx = tl.program_id(1)

            q_base = q_ptr + bh_idx * stride_bh_q
            k_base = k_ptr + bh_idx * stride_bh_k
            v_base = v_ptr + bh_idx * stride_bh_v
            o_base = out_ptr + bh_idx * stride_bh_o
            stat_offset = bh_idx * stride_bh_stat

            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_d = tl.arange(0, BLOCK_D)
            mask_m = offs_m < N_Q
            mask_d = offs_d < HEAD_DIM

            q_ptrs = q_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
            q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

            hi = tl.minimum(N_K, (start_m + 1) * BLOCK_M) if IS_CAUSAL else N_K

            # Pass 1: Exact row-wise maximum and first-argmax index in SRAM
            m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
            m_idx_i = tl.full([BLOCK_M], N_K, dtype=tl.int32)

            for start_n in range(0, hi, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                mask_n = offs_n < N_K
                k_ptrs = k_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :]
                k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

                s = tl.dot(q, tl.trans(k), allow_tf32=False) * sm_scale
                valid = mask_m[:, None] & mask_n[None, :]
                if IS_CAUSAL:
                    valid = valid & (offs_m[:, None] >= offs_n[None, :])
                s = tl.where(valid, s, -float("inf"))

                tile_max = tl.max(s, axis=1)
                new_max = tl.maximum(m_i, tile_max)

                finite = valid & (s > -float("inf"))
                is_tile_max = (s == new_max[:, None]) & finite
                tile_first_col = tl.min(tl.where(is_tile_max, offs_n[None, :], N_K), axis=1)

                m_idx_i = tl.where(new_max > m_i, tile_first_col, tl.minimum(m_idx_i, tile_first_col))
                m_i = new_max

            # Pass 2: Fused Cayley-Padé Rational Kernel with Compact-Support Tile Skipping
            l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

            for start_n in range(0, hi, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                mask_n = offs_n < N_K
                k_ptrs = k_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :]
                k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

                s = tl.dot(q, tl.trans(k), allow_tf32=False) * sm_scale
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
                    num = tl.maximum(1.0 - safe_u, 0.0)
                    den = 1.0 + safe_u
                    q_rat = num / den
                    q2 = q_rat * q_rat
                    q4 = q2 * q2
                    phi = tl.where(active, q4 * q4, 0.0)

                    l_i += tl.sum(phi, axis=1)

                    v_ptrs = v_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :]
                    v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
                    acc += tl.dot(phi, v, allow_tf32=False)

            safe_l = tl.where(l_i > 0.0, l_i, 1.0)
            out = acc / safe_l[:, None]

            o_ptrs = o_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
            tl.store(o_ptrs, out, mask=mask_m[:, None] & mask_d[None, :])
            tl.store(m_ptr + stat_offset + offs_m, m_i, mask=mask_m)
            tl.store(l_ptr + stat_offset + offs_m, safe_l, mask=mask_m)
            tl.store(m_idx_ptr + stat_offset + offs_m, m_idx_i, mask=mask_m)

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
        ):
            start_m = tl.program_id(0)
            bh_idx = tl.program_id(1)

            q_base = q_ptr + bh_idx * stride_bh_q
            k_base = k_ptr + bh_idx * stride_bh_k
            v_base = v_ptr + bh_idx * stride_bh_v
            do_base = do_ptr + bh_idx * stride_bh_do
            dq_base = dq_ptr + bh_idx * stride_bh_q
            stat_offset = bh_idx * stride_bh_stat

            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_d = tl.arange(0, BLOCK_D)
            mask_m = offs_m < N_Q
            mask_d = offs_d < HEAD_DIM

            q = tl.load(
                q_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                mask=mask_m[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            do = tl.load(
                do_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                mask=mask_m[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            m_i = tl.load(m_ptr + stat_offset + offs_m, mask=mask_m, other=-float("inf")).to(tl.float32)
            l_i = tl.load(l_ptr + stat_offset + offs_m, mask=mask_m, other=1.0).to(tl.float32)
            m_idx_i = tl.load(m_idx_ptr + stat_offset + offs_m, mask=mask_m, other=N_K).to(tl.int32)
            d_i = tl.load(d_ptr + stat_offset + offs_m, mask=mask_m, other=0.0).to(tl.float32)

            hi = tl.minimum(N_K, (start_m + 1) * BLOCK_M) if IS_CAUSAL else N_K
            c_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

            # Pass 1: Accumulate row-wise sum of direct score gradients c_i = sum_k dS_direct_{ik}
            for start_n in range(0, hi, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                mask_n = offs_n < N_K
                k = tl.load(
                    k_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=mask_n[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)

                s = tl.dot(q, tl.trans(k), allow_tf32=False) * sm_scale
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
                    num = tl.maximum(1.0 - safe_u, 0.0)
                    den = 1.0 + safe_u
                    q_rat = num / den
                    q2 = q_rat * q_rat
                    q4 = q2 * q2
                    p = tl.where(active, (q4 * q4) / l_i[:, None], 0.0)

                    v = tl.load(
                        v_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                        mask=mask_n[:, None] & mask_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    dp = tl.dot(do, tl.trans(v), allow_tf32=False)
                    w = (16.0 * ra_scale) / (1.0 - safe_u * safe_u)
                    ds_direct = tl.where(active, p * (dp - d_i[:, None]) * w, 0.0)
                    c_acc += tl.sum(ds_direct, axis=1)

            tl.store(c_ptr + stat_offset + offs_m, c_acc, mask=mask_m)

            # Pass 2: Accumulate exact dQ in SRAM without atomic_add
            dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
            for start_n in range(0, hi, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                mask_n = offs_n < N_K
                k = tl.load(
                    k_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=mask_n[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)

                s = tl.dot(q, tl.trans(k), allow_tf32=False) * sm_scale
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
                    num = tl.maximum(1.0 - safe_u, 0.0)
                    den = 1.0 + safe_u
                    q_rat = num / den
                    q2 = q_rat * q_rat
                    q4 = q2 * q2
                    p = tl.where(active, (q4 * q4) / l_i[:, None], 0.0)

                    v = tl.load(
                        v_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                        mask=mask_n[:, None] & mask_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    dp = tl.dot(do, tl.trans(v), allow_tf32=False)
                    w = (16.0 * ra_scale) / (1.0 - safe_u * safe_u)
                    ds_direct = tl.where(active, p * (dp - d_i[:, None]) * w, 0.0)

                    is_first_max = finite & (offs_n[None, :] == m_idx_i[:, None])
                    ds = (ds_direct - tl.where(is_first_max, c_acc[:, None], 0.0)) * sm_scale
                    dq_acc += tl.dot(ds, k, allow_tf32=False)

            tl.store(
                dq_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                dq_acc,
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
        ):
            start_n = tl.program_id(0)
            bh_idx = tl.program_id(1)

            q_base = q_ptr + bh_idx * stride_bh_q
            k_base = k_ptr + bh_idx * stride_bh_k
            v_base = v_ptr + bh_idx * stride_bh_v
            do_base = do_ptr + bh_idx * stride_bh_do
            dk_base = dk_ptr + bh_idx * stride_bh_k
            dv_base = dv_ptr + bh_idx * stride_bh_v
            stat_offset = bh_idx * stride_bh_stat

            offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_d = tl.arange(0, BLOCK_D)
            mask_n = offs_n < N_K
            mask_d = offs_d < HEAD_DIM

            k = tl.load(
                k_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)

            dk_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
            dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

            lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M if IS_CAUSAL else 0

            for start_m in range(lo, N_Q, BLOCK_M):
                offs_m = start_m + tl.arange(0, BLOCK_M)
                mask_m = offs_m < N_Q

                q = tl.load(
                    q_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=mask_m[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                m_i = tl.load(m_ptr + stat_offset + offs_m, mask=mask_m, other=-float("inf")).to(tl.float32)

                s = tl.dot(q, tl.trans(k), allow_tf32=False) * sm_scale
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
                    ).to(tl.float32)
                    l_i = tl.load(l_ptr + stat_offset + offs_m, mask=mask_m, other=1.0).to(tl.float32)
                    m_idx_i = tl.load(m_idx_ptr + stat_offset + offs_m, mask=mask_m, other=N_K).to(tl.int32)
                    d_i = tl.load(d_ptr + stat_offset + offs_m, mask=mask_m, other=0.0).to(tl.float32)
                    c_i = tl.load(c_ptr + stat_offset + offs_m, mask=mask_m, other=0.0).to(tl.float32)

                    active = finite & (u < 1.0)
                    safe_u = tl.where(active, u, 0.0)
                    num = tl.maximum(1.0 - safe_u, 0.0)
                    den = 1.0 + safe_u
                    q_rat = num / den
                    q2 = q_rat * q_rat
                    q4 = q2 * q2
                    p = tl.where(active, (q4 * q4) / l_i[:, None], 0.0)

                    dv_acc += tl.dot(tl.trans(p), do, allow_tf32=False)

                    dp = tl.dot(do, tl.trans(v), allow_tf32=False)
                    w = (16.0 * ra_scale) / (1.0 - safe_u * safe_u)
                    ds_direct = tl.where(active, p * (dp - d_i[:, None]) * w, 0.0)

                    is_first_max = finite & (offs_n[None, :] == m_idx_i[:, None])
                    ds = (ds_direct - tl.where(is_first_max, c_i[:, None], 0.0)) * sm_scale

                    dk_acc += tl.dot(tl.trans(ds), q, allow_tf32=False)

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
        _ra_norm_fwd_kernel = None
        _pos_ra_glu_fwd_kernel = None
        _ra_softmax_fwd_kernel = None
        _ra_softmax_bwd_kernel = None
        _flash_ra_attn_fwd_kernel = None
        _flash_ra_attn_bwd_dq_kernel = None
        _flash_ra_attn_bwd_dkdv_kernel = None

from ._compat import _HAS_TORCH, torch


def triton_ra_norm(
    x: Any,
    weight: Optional[Any] = None,
    bias: Optional[Any] = None,
    eps: float = 1e-6,
) -> Any:
    """Dispatches RaNorm execution to Triton GPU kernel when available."""
    if not (HAS_TRITON and _HAS_TORCH and isinstance(x, torch.Tensor) and x.is_cuda):
        return None

    if not x.is_contiguous():
        x = x.contiguous()
    if weight is not None and not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    D = x.shape[-1]
    if D > 8192 or D & (D - 1) != 0:
        BLOCK_SIZE = triton.next_power_of_2(D)  # type: ignore[attr-defined]
    else:
        BLOCK_SIZE = max(16, D)

    y = torch.empty_like(x)
    n_rows = x.numel() // D

    grid = (n_rows,)
    _ra_norm_fwd_kernel[grid](
        x,
        y,
        weight if weight is not None else x,
        bias if bias is not None else x,
        D,
        D,
        D=D,
        eps=eps,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


def triton_pos_ra_glu(
    gate: Any,
    up: Any,
    b: float = 1.0,
) -> Any:
    """Dispatches PosRaGLU forward execution to Triton GPU kernel."""
    if not (HAS_TRITON and _HAS_TORCH and isinstance(gate, torch.Tensor) and gate.is_cuda):
        return None

    if not gate.is_contiguous():
        gate = gate.contiguous()
    if not up.is_contiguous():
        up = up.contiguous()

    n_elements = gate.numel()
    out = torch.empty_like(gate)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # type: ignore[attr-defined]

    _pos_ra_glu_fwd_kernel[grid](
        gate,
        up,
        out,
        n_elements,
        b_val=float(b),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def triton_ra_softmax(
    scores: Any,
    scale: float = 0.0625,
) -> Any:
    """Dispatches ra_softmax forward execution to Triton GPU kernel."""
    if not (HAS_TRITON and _HAS_TORCH and isinstance(scores, torch.Tensor) and scores.is_cuda):
        return None

    if not scores.is_contiguous():
        scores = scores.contiguous()

    N = scores.shape[-1]
    BLOCK_SIZE = max(16, triton.next_power_of_2(N))  # type: ignore[attr-defined]
    if BLOCK_SIZE > 8192:
        return None

    out = torch.empty_like(scores)
    n_rows = scores.numel() // N
    grid = (n_rows,)

    _ra_softmax_fwd_kernel[grid](
        scores,
        out,
        N,
        N=N,
        scale=float(scale),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def triton_ra_softmax_backward(
    scores: Any,
    out: Any,
    grad_out: Any,
    scale: float = 0.0625,
) -> Any:
    """Dispatches ra_softmax backward execution to Triton GPU kernel."""
    if not (HAS_TRITON and _HAS_TORCH and isinstance(scores, torch.Tensor) and scores.is_cuda):
        return None

    if not scores.is_contiguous():
        scores = scores.contiguous()
    if not out.is_contiguous():
        out = out.contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    N = scores.shape[-1]
    BLOCK_SIZE = max(16, triton.next_power_of_2(N))  # type: ignore[attr-defined]
    if BLOCK_SIZE > 8192:
        return None

    grad_scores = torch.empty_like(scores)
    n_rows = scores.numel() // N
    grid = (n_rows,)

    _ra_softmax_bwd_kernel[grid](
        scores,
        out,
        grad_out,
        grad_scores,
        N,
        N=N,
        scale=float(scale),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return grad_scores


def triton_flash_ra_attention(
    q: Any,
    k: Any,
    v: Any,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    ra_scale: float = 0.0625,
) -> Optional[Tuple[Any, Any, Any, Any]]:
    """
    Dispatches fused tiled Rational Attention (Flash-RaAttention) forward pass to Triton SRAM kernel.
    Returns (out, m, l, m_idx) without materializing the [B, H, N_Q, N_K] attention matrix in HBM.
    """
    if not (
        HAS_TRITON
        and _HAS_TORCH
        and isinstance(q, torch.Tensor)
        and q.is_cuda
        and q.ndim == 4
    ):
        return None

    B, H, N_Q, D = q.shape
    N_K = k.shape[2]
    if D > 256 or k.shape[-1] != D or v.shape[-1] != D:
        return None

    BLOCK_D = max(16, triton.next_power_of_2(D))  # type: ignore[attr-defined]
    BLOCK_M = 64 if N_Q >= 64 else max(16, triton.next_power_of_2(N_Q))  # type: ignore[attr-defined]
    BLOCK_N = 64 if N_K >= 64 else max(16, triton.next_power_of_2(N_K))  # type: ignore[attr-defined]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()

    out = torch.empty_like(q_c)
    bh = B * H
    m = torch.empty((B, H, N_Q), device=q.device, dtype=torch.float32)
    l = torch.empty((B, H, N_Q), device=q.device, dtype=torch.float32)
    m_idx = torch.empty((B, H, N_Q), device=q.device, dtype=torch.int32)

    grid = (triton.cdiv(N_Q, BLOCK_M), bh)  # type: ignore[attr-defined]
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
    )
    return out, m, l, m_idx


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
    """
    Dispatches fused tiled Rational Attention (Flash-RaAttention) backward pass to Triton SRAM kernels.
    Recomputes attention tiles in SRAM without materializing [B, H, N_Q, N_K] tensors in HBM.
    """
    if not (
        HAS_TRITON
        and _HAS_TORCH
        and isinstance(q, torch.Tensor)
        and q.is_cuda
        and q.ndim == 4
    ):
        return None

    B, H, N_Q, D = q.shape
    N_K = k.shape[2]
    if D > 256 or k.shape[-1] != D or v.shape[-1] != D:
        return None

    BLOCK_D = max(16, triton.next_power_of_2(D))  # type: ignore[attr-defined]
    BLOCK_M = 64 if N_Q >= 64 else max(16, triton.next_power_of_2(N_Q))  # type: ignore[attr-defined]
    BLOCK_N = 64 if N_K >= 64 else max(16, triton.next_power_of_2(N_K))  # type: ignore[attr-defined]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    do_c = grad_out.contiguous()
    out_c = out.contiguous()

    # Row-wise inner product D_i = sum_d dO_{i,d} * O_{i,d}
    d_vec = (do_c.to(torch.float32) * out_c.to(torch.float32)).sum(dim=-1).contiguous()
    c_vec = torch.empty_like(d_vec)

    dq = torch.empty_like(q_c)
    dk = torch.empty_like(k_c)
    dv = torch.empty_like(v_c)
    bh = B * H

    grid_m = (triton.cdiv(N_Q, BLOCK_M), bh)  # type: ignore[attr-defined]
    _flash_ra_attn_bwd_dq_kernel[grid_m](
        q_c,
        k_c,
        v_c,
        do_c,
        dq,
        m,
        l,
        m_idx,
        d_vec,
        c_vec,
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
    )

    grid_n = (triton.cdiv(N_K, BLOCK_N), bh)  # type: ignore[attr-defined]
    _flash_ra_attn_bwd_dkdv_kernel[grid_n](
        q_c,
        k_c,
        v_c,
        do_c,
        dk,
        dv,
        m,
        l,
        m_idx,
        d_vec,
        c_vec,
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
    )

    return dq, dk, dv
