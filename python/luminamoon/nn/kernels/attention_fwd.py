from __future__ import annotations

from collections import OrderedDict
import math
from typing import TYPE_CHECKING, Any, Optional, Tuple

from ._base import HAS_FDIV_APPROX, HAS_TRITON, _HAS_TORCH, _dev_idx, _int_spec, _launch_ck_autotuned, _ptr_aligned, tl, torch, triton

if TYPE_CHECKING:
    _compute_decay_bias: Any = None
    _compute_causal_valid: Any = None
    _flash_ra_attn_fwd_kernel: Any = None
elif HAS_TRITON:
    _ATTN_FWD_CONFIGS = [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    ]

    @triton.jit
    def _compute_decay_bias(
        offs_m,
        offs_n,
        q_offset,
        s_h,
        h_pen,
        hub_size,
        HAS_HUB: tl.constexpr,
        BIAS_IS_RATIONAL: tl.constexpr,
    ):
        raw_dist = tl.maximum((offs_m[:, None] + q_offset - offs_n[None, :]).to(tl.float32), 0.0)
        if BIAS_IS_RATIONAL:
            text_bias = -tl.log(tl.maximum(1.0 + s_h * raw_dist, 1e-8))
        else:
            text_bias = -s_h * raw_dist
        if HAS_HUB:
            bias = tl.where(offs_n[None, :] < hub_size, -h_pen, text_bias)
        else:
            bias = text_bias
        return tl.minimum(tl.maximum(bias, -10.0), 0.0)

    @triton.jit
    def _compute_causal_valid(
        offs_m,
        offs_n,
        q_offset,
        hub_size,
        HAS_HUB: tl.constexpr,
        CHECK_HUB_Q: tl.constexpr,
    ):
        if HAS_HUB:
            is_hub_k = offs_n[None, :] < hub_size
            if CHECK_HUB_Q:
                return is_hub_k | ((offs_m[:, None] >= hub_size) & (offs_n[None, :] <= (offs_m[:, None] + q_offset)))
            return is_hub_k | (offs_n[None, :] <= (offs_m[:, None] + q_offset))
        return offs_n[None, :] <= (offs_m[:, None] + q_offset)

    @triton.jit
    def _ra_fwd_pass1_step(
        q,
        k_base,
        stride_kn,
        doc_base,
        doc_m,
        offs_m,
        offs_d,
        mask_m,
        mask_d,
        start_n,
        N_K,
        sm_scale,
        s_h,
        h_pen,
        hub_size,
        q_offset,
        m_i,
        m_idx_i,
        BLOCK_N: tl.constexpr,
        EVEN_D: tl.constexpr,
        MASK_N: tl.constexpr,
        MASK_CAUSAL: tl.constexpr,
        HAS_DOC_IDS: tl.constexpr,
        HAS_DECAY_BIAS: tl.constexpr,
        HAS_HUB: tl.constexpr,
        BIAS_IS_RATIONAL: tl.constexpr,
        CHECK_HUB_Q: tl.constexpr,
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
            if MASK_N:
                mask_n = offs_n < N_K
                k_mask = mask_n[:, None] if EVEN_D else (mask_n[:, None] & mask_d[None, :])
                k = tl.load(k_ptrs, mask=k_mask, other=0.0)
            elif EVEN_D:
                k = tl.load(k_ptrs)
            else:
                k = tl.load(k_ptrs, mask=mask_d[None, :], other=0.0)

            s = tl.dot(q, tl.trans(k)) * sm_scale
            if HAS_DECAY_BIAS:
                s = s + _compute_decay_bias(
                    offs_m, offs_n, q_offset, s_h, h_pen, hub_size,
                    HAS_HUB=HAS_HUB, BIAS_IS_RATIONAL=BIAS_IS_RATIONAL,
                )
            if MASK_N or MASK_CAUSAL or HAS_DOC_IDS:
                valid = mask_m[:, None] & (offs_n[None, :] < N_K)
                if MASK_CAUSAL:
                    valid = valid & _compute_causal_valid(
                        offs_m, offs_n, q_offset, hub_size, HAS_HUB=HAS_HUB, CHECK_HUB_Q=CHECK_HUB_Q
                    )
                if HAS_DOC_IDS:
                    valid = valid & doc_match
                s = tl.where(valid, s, -float("inf"))

            # Single fused max+argmin reduction; monotonic start_n ensures strict > preserves left-most tie-breaking.
            tile_max, tile_local_idx = tl.max(
                s, axis=1, return_indices=True, return_indices_tie_break_left=True
            )
            tile_col = start_n + tile_local_idx.to(tl.int32)
            update = (tile_max > m_i) & (tile_max > -float("inf"))
            m_idx_i = tl.where(update, tile_col, m_idx_i)
            m_i = tl.maximum(m_i, tile_max)
        return m_i, m_idx_i

    @triton.jit
    def _ra_fwd_pass2_step(
        q,
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
        l_i,
        acc,
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
                q_rat = tl.fdiv(1.0 - safe_u, 1.0 + safe_u, approx=True)
            else:
                q_rat = tl.fdiv(1.0 - safe_u, 1.0 + safe_u)
            q2 = q_rat * q_rat
            q4 = q2 * q2
            phi = tl.where(active, q4 * q4, 0.0)

            l_i += tl.sum(phi, axis=1)
            acc += tl.dot(phi.to(v.dtype), v)
        return l_i, acc

    @triton.autotune(
        configs=_ATTN_FWD_CONFIGS,
        key=["N_Q", "N_K", "HEAD_DIM", "IS_CAUSAL", "HAS_DOC_IDS", "HAS_DECAY_BIAS", "HAS_HUB"],
    )
    @triton.jit
    def _flash_ra_attn_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        m_ptr,
        l_ptr,
        m_idx_ptr,
        doc_ids_ptr,
        bias_slopes_ptr,
        hub_penalty_ptr,
        stride_qb,
        stride_qh,
        stride_qn,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_ob,
        stride_oh,
        stride_on,
        stride_db,
        H_Q,
        N_Q,
        N_K,
        hub_size,
        q_offset,
        sm_scale: tl.constexpr,
        ra_scale: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        HAS_DOC_IDS: tl.constexpr,
        HAS_DECAY_BIAS: tl.constexpr,
        HAS_HUB: tl.constexpr,
        BIAS_IS_RATIONAL: tl.constexpr,
        CHECK_HUB_Q: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        EVEN_D: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        USE_APPROX_DIV: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        bh_q = tl.program_id(0)
        pid_m = tl.program_id(1)
        # Schedule longest causal query tiles first to eliminate tail-wave SM bubbles.
        start_m = (tl.num_programs(1) - 1 - pid_m) if IS_CAUSAL else pid_m

        b = (bh_q // H_Q).to(tl.int64)
        h_q = (bh_q % H_Q).to(tl.int64)
        h_kv = h_q // GROUP_SIZE

        q_base = q_ptr + b * stride_qb + h_q * stride_qh
        k_base = k_ptr + b * stride_kb + h_kv * stride_kh
        v_base = v_ptr + b * stride_vb + h_kv * stride_vh
        o_base = out_ptr + b * stride_ob + h_q * stride_oh
        doc_base = doc_ids_ptr + b * stride_db
        stat_offset = bh_q.to(tl.int64) * N_Q

        s_h = tl.load(bias_slopes_ptr + h_q).to(tl.float32) if HAS_DECAY_BIAS else 0.0
        h_pen = tl.load(hub_penalty_ptr + h_q).to(tl.float32) if (HAS_DECAY_BIAS and HAS_HUB) else 0.0

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < N_Q
        mask_d = offs_d < HEAD_DIM

        q_ptrs = q_base + offs_m[:, None] * stride_qn + offs_d[None, :]
        q_mask = mask_m[:, None] if EVEN_D else (mask_m[:, None] & mask_d[None, :])
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)
        doc_m = (
            tl.load(doc_base + offs_m, mask=mask_m, other=0)
            if HAS_DOC_IDS
            else tl.zeros([BLOCK_M], dtype=tl.int32)
        )

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

        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        m_idx_i = tl.full([BLOCK_M], N_K, dtype=tl.int32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        run_loops = (tl.max(doc_m, axis=0) > 0) if HAS_DOC_IDS else True
        if run_loops:
            for start_n in range(0, hi_full, BLOCK_N):
                m_i, m_idx_i = _ra_fwd_pass1_step(
                    q, k_base, stride_kn, doc_base, doc_m, offs_m, offs_d, mask_m, mask_d, start_n, N_K,
                    sm_scale, s_h, h_pen, hub_size, q_offset, m_i, m_idx_i, BLOCK_N=BLOCK_N, EVEN_D=EVEN_D,
                    MASK_N=False, MASK_CAUSAL=False, HAS_DOC_IDS=HAS_DOC_IDS,
                    HAS_DECAY_BIAS=HAS_DECAY_BIAS, HAS_HUB=HAS_HUB,
                    BIAS_IS_RATIONAL=BIAS_IS_RATIONAL, CHECK_HUB_Q=CHECK_HUB_Q,
                )
            for start_n in range(hi_full, hi, BLOCK_N):
                m_i, m_idx_i = _ra_fwd_pass1_step(
                    q, k_base, stride_kn, doc_base, doc_m, offs_m, offs_d, mask_m, mask_d, start_n, N_K,
                    sm_scale, s_h, h_pen, hub_size, q_offset, m_i, m_idx_i, BLOCK_N=BLOCK_N, EVEN_D=EVEN_D,
                    MASK_N=True, MASK_CAUSAL=IS_CAUSAL, HAS_DOC_IDS=HAS_DOC_IDS,
                    HAS_DECAY_BIAS=HAS_DECAY_BIAS, HAS_HUB=HAS_HUB,
                    BIAS_IS_RATIONAL=BIAS_IS_RATIONAL, CHECK_HUB_Q=CHECK_HUB_Q,
                )

            qk_scale = sm_scale * ra_scale
            # Use +inf when m_i is -inf so u evaluates to +inf (inactive) after clamping at 0.
            m_scaled = tl.where(m_i > -float("inf"), m_i * ra_scale, float("inf"))

            for start_n in range(0, hi_full, BLOCK_N):
                l_i, acc = _ra_fwd_pass2_step(
                    q, k_base, v_base, stride_kn, stride_vn, doc_base, doc_m, offs_m, offs_d, mask_m, mask_d,
                    start_n, N_K, qk_scale, ra_scale, s_h, h_pen, hub_size, q_offset, m_scaled, l_i, acc,
                    BLOCK_N=BLOCK_N, EVEN_D=EVEN_D, MASK_N=False, MASK_CAUSAL=False,
                    HAS_DOC_IDS=HAS_DOC_IDS, HAS_DECAY_BIAS=HAS_DECAY_BIAS, HAS_HUB=HAS_HUB,
                    BIAS_IS_RATIONAL=BIAS_IS_RATIONAL, CHECK_HUB_Q=CHECK_HUB_Q, USE_APPROX_DIV=USE_APPROX_DIV,
                )
            for start_n in range(hi_full, hi, BLOCK_N):
                l_i, acc = _ra_fwd_pass2_step(
                    q, k_base, v_base, stride_kn, stride_vn, doc_base, doc_m, offs_m, offs_d, mask_m, mask_d,
                    start_n, N_K, qk_scale, ra_scale, s_h, h_pen, hub_size, q_offset, m_scaled, l_i, acc,
                    BLOCK_N=BLOCK_N, EVEN_D=EVEN_D, MASK_N=True, MASK_CAUSAL=IS_CAUSAL,
                    HAS_DOC_IDS=HAS_DOC_IDS, HAS_DECAY_BIAS=HAS_DECAY_BIAS, HAS_HUB=HAS_HUB,
                    BIAS_IS_RATIONAL=BIAS_IS_RATIONAL, CHECK_HUB_Q=CHECK_HUB_Q, USE_APPROX_DIV=USE_APPROX_DIV,
                )

        safe_l = tl.where(l_i > 0.0, l_i, 1.0)
        out = acc / safe_l[:, None]

        o_ptrs = o_base + offs_m[:, None] * stride_on + offs_d[None, :]
        tl.store(o_ptrs, out, mask=q_mask)
        tl.store(m_ptr + stat_offset + offs_m, m_i, mask=mask_m)
        tl.store(l_ptr + stat_offset + offs_m, safe_l, mask=mask_m)
        tl.store(m_idx_ptr + stat_offset + offs_m, m_idx_i, mask=mask_m)
else:
    _compute_decay_bias = None
    _compute_causal_valid = None
    _flash_ra_attn_fwd_kernel = None


_ATTN_FWD_CFG_CACHE: OrderedDict[Tuple[Any, ...], Tuple[Any, int]] = OrderedDict()


def triton_flash_ra_attention(
    q: Any,
    k: Any,
    v: Any,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    ra_scale: float = 0.0625,
    doc_ids: Optional[Any] = None,
    *,
    bias_slopes: Optional[Any] = None,
    hub_penalty: Optional[Any] = None,
    bias_mode: Optional[str] = None,
    hub_size: int = 0,
    q_offset: int = 0,
    check_hub_q: bool = False,
) -> Optional[Tuple[Any, Any, Any, Any]]:
    """Dispatches fused tiled Rational Attention forward pass to Triton SRAM kernel."""
    if not (
        HAS_TRITON
        and _HAS_TORCH
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
        return None

    B, H, N_Q, D = q.shape
    H_KV = k.shape[1]
    N_K = k.shape[2]
    if (
        B <= 0
        or H <= 0
        or N_Q <= 0
        or N_K <= 0
        or D <= 0
        or D > 256
        or k.shape[0] != B
        or v.shape[0] != B
        or v.shape[2] != N_K
        or k.shape[-1] != D
        or v.shape[-1] != D
        or v.shape[1] != H_KV
        or H_KV <= 0
        or (H % H_KV != 0)
    ):
        return None

    has_doc = doc_ids is not None
    if has_doc:
        if not (
            isinstance(doc_ids, torch.Tensor)
            and doc_ids.device == q.device
            and doc_ids.ndim == 2
            and doc_ids.shape == (B, N_Q)
            and N_Q == N_K
        ):
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

    out = torch.empty(q_c.shape, device=q_c.device, dtype=q_c.dtype)
    m = torch.empty((B, H, N_Q), device=q_c.device, dtype=torch.float32)
    l = torch.empty((B, H, N_Q), device=q_c.device, dtype=torch.float32)
    m_idx = torch.empty((B, H, N_Q), device=q_c.device, dtype=torch.int32)

    fwd_args = (
        q_c, k_c, v_c, out, m, l, m_idx, doc_c, slopes_c, hub_pen_c,
        q_c.stride(0), q_c.stride(1), q_c.stride(2),
        k_c.stride(0), k_c.stride(1), k_c.stride(2),
        v_c.stride(0), v_c.stride(1), v_c.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        stride_db, H, N_Q, N_K, hub_sz, int(q_offset),
    )
    cfg_key = (
        _dev_idx(q_c.device), N_Q, N_K, D, bool(is_causal), has_doc, has_decay, has_hub, bias_is_rational,
        bool(check_hub_q and has_hub), GROUP_SIZE, float(sm_scale), float(ra_scale), q_c.dtype,
        _ptr_aligned(q_c, k_c, v_c, out, m, l, m_idx, doc_c, slopes_c, hub_pen_c),
        tuple(_int_spec(v) for v in fwd_args[10:]),
    )
    fwd_kw = dict(
        sm_scale=float(sm_scale), ra_scale=float(ra_scale), IS_CAUSAL=bool(is_causal),
        HAS_DOC_IDS=has_doc, HAS_DECAY_BIAS=has_decay, HAS_HUB=has_hub,
        BIAS_IS_RATIONAL=bias_is_rational, CHECK_HUB_Q=bool(check_hub_q and has_hub),
        HEAD_DIM=D, BLOCK_D=BLOCK_D, EVEN_D=EVEN_D, GROUP_SIZE=GROUP_SIZE,
        USE_APPROX_DIV=HAS_FDIV_APPROX, enable_fp_fusion=True,
    )
    _launch_ck_autotuned(
        _flash_ra_attn_fwd_kernel, _ATTN_FWD_CFG_CACHE, cfg_key, q_c.device,
        lambda bm: (B * H, (N_Q + bm - 1) // bm, 1),
        lambda meta: (B * H, triton.cdiv(N_Q, meta["BLOCK_M"])),
        "BLOCK_M", fwd_args, fwd_kw,
    )
    return out, m, l, m_idx
