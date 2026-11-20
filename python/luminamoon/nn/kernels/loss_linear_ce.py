from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

from ._base import HAS_TRITON, _HAS_TORCH, tl, torch, triton

if TYPE_CHECKING:
    _fused_linear_ce_fwd_kernel: Any = None
    _fused_linear_ce_bwd_dx_kernel: Any = None
    _fused_linear_ce_bwd_dw_db_kernel: Any = None
elif HAS_TRITON:
    @triton.jit
    def _fused_linear_ce_fwd_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        target_ptr,
        loss_ptr,
        lse_ptr,
        M,
        V,
        D,
        ignore_index,
        label_smoothing: tl.constexpr,
        logit_scale: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        target_m = tl.load(target_ptr + offs_m, mask=mask_m, other=ignore_index)
        valid_m = mask_m & (target_m != ignore_index) & (target_m >= 0) & (target_m < V)

        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        s_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        z_target_i = tl.zeros([BLOCK_M], dtype=tl.float32)

        for start_v in range(0, V, BLOCK_V):
            offs_v = start_v + tl.arange(0, BLOCK_V)
            mask_v = offs_v < V

            logits_tile = tl.zeros([BLOCK_M, BLOCK_V], dtype=tl.float32)
            for start_d in range(0, D, BLOCK_D):
                offs_d = start_d + tl.arange(0, BLOCK_D)
                mask_d = offs_d < D
                h_tile = tl.load(
                    x_ptr + offs_m[:, None] * D + offs_d[None, :],
                    mask=mask_m[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                w_tile = tl.load(
                    w_ptr + offs_v[:, None] * D + offs_d[None, :],
                    mask=mask_v[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                logits_tile += tl.dot(h_tile, tl.trans(w_tile), allow_tf32=False)

            if HAS_BIAS:
                b_tile = tl.load(b_ptr + offs_v, mask=mask_v, other=0.0).to(tl.float32)
                logits_tile += b_tile[None, :]
            logits_tile = logits_tile * logit_scale

            valid_mv = valid_m[:, None] & mask_v[None, :]
            s_i += tl.sum(tl.where(valid_mv, logits_tile, 0.0), axis=1)

            is_target = valid_mv & (offs_v[None, :] == target_m[:, None])
            z_target_i += tl.sum(tl.where(is_target, logits_tile, 0.0), axis=1)

            masked_logits = tl.where(valid_mv, logits_tile, -float("inf"))
            tile_max = tl.max(masked_logits, axis=1)
            m_new = tl.maximum(m_i, tile_max)

            has_prev = m_i > -float("inf")
            safe_prev_diff = tl.where(has_prev, m_i - m_new, 0.0)
            alpha = tl.where(has_prev, tl.exp(safe_prev_diff), 0.0)

            has_cur = valid_mv & (m_new[:, None] > -float("inf"))
            safe_cur_diff = tl.where(has_cur, masked_logits - m_new[:, None], 0.0)
            p_tile = tl.where(has_cur, tl.exp(safe_cur_diff), 0.0)

            l_i = l_i * alpha + tl.sum(p_tile, axis=1)
            m_i = m_new

        safe_l = tl.where(valid_m & (l_i > 0.0), l_i, 1.0)
        lse_i = tl.where(valid_m & (l_i > 0.0), m_i + tl.log(safe_l), 0.0)
        loss_i = tl.where(
            valid_m,
            lse_i - (1.0 - label_smoothing) * z_target_i - (label_smoothing / V) * s_i,
            0.0,
        )
        tl.store(loss_ptr + offs_m, loss_i, mask=mask_m)
        tl.store(lse_ptr + offs_m, lse_i, mask=mask_m)

    @triton.jit
    def _fused_linear_ce_bwd_dx_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        target_ptr,
        lse_ptr,
        grad_loss_ptr,
        dx_ptr,
        M,
        V,
        D,
        ignore_index,
        label_smoothing: tl.constexpr,
        logit_scale: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        target_m = tl.load(target_ptr + offs_m, mask=mask_m, other=ignore_index)
        valid_m = mask_m & (target_m != ignore_index) & (target_m >= 0) & (target_m < V)
        lse_m = tl.load(lse_ptr + offs_m, mask=valid_m, other=0.0).to(tl.float32)
        grad_m = tl.load(grad_loss_ptr + offs_m, mask=valid_m, other=0.0).to(tl.float32) * logit_scale

        for start_v in range(0, V, BLOCK_V):
            offs_v = start_v + tl.arange(0, BLOCK_V)
            mask_v = offs_v < V

            logits_tile = tl.zeros([BLOCK_M, BLOCK_V], dtype=tl.float32)
            for start_d in range(0, D, BLOCK_D):
                offs_d = start_d + tl.arange(0, BLOCK_D)
                mask_d = offs_d < D
                h_tile = tl.load(
                    x_ptr + offs_m[:, None] * D + offs_d[None, :],
                    mask=mask_m[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                w_tile = tl.load(
                    w_ptr + offs_v[:, None] * D + offs_d[None, :],
                    mask=mask_v[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                logits_tile += tl.dot(h_tile, tl.trans(w_tile), allow_tf32=False)

            if HAS_BIAS:
                b_tile = tl.load(b_ptr + offs_v, mask=mask_v, other=0.0).to(tl.float32)
                logits_tile += b_tile[None, :]
            logits_tile = logits_tile * logit_scale

            valid_mv = valid_m[:, None] & mask_v[None, :]
            safe_exp_arg = tl.where(valid_mv, logits_tile - lse_m[:, None], 0.0)
            p_tile = tl.where(valid_mv, tl.exp(safe_exp_arg), 0.0)
            is_target = valid_mv & (offs_v[None, :] == target_m[:, None])
            target_term = tl.where(is_target, 1.0 - label_smoothing, 0.0)
            smooth_term = tl.where(valid_mv, label_smoothing / V, 0.0)
            dz_tile = (p_tile - target_term - smooth_term) * grad_m[:, None]

            for start_d in range(0, D, BLOCK_D):
                offs_d = start_d + tl.arange(0, BLOCK_D)
                mask_d = offs_d < D
                w_tile = tl.load(
                    w_ptr + offs_v[:, None] * D + offs_d[None, :],
                    mask=mask_v[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                dx_part = tl.dot(dz_tile, w_tile, allow_tf32=False)
                dx_ptrs = dx_ptr + offs_m[:, None] * D + offs_d[None, :]
                mask_md = mask_m[:, None] & mask_d[None, :]
                if start_v == 0:
                    tl.store(dx_ptrs, dx_part, mask=mask_md)
                else:
                    dx_prev = tl.load(dx_ptrs, mask=mask_md, other=0.0).to(tl.float32)
                    tl.store(dx_ptrs, dx_prev + dx_part, mask=mask_md)

    @triton.jit
    def _fused_linear_ce_bwd_dw_db_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        target_ptr,
        lse_ptr,
        grad_loss_ptr,
        dw_ptr,
        db_ptr,
        M,
        V,
        D,
        ignore_index,
        label_smoothing: tl.constexpr,
        logit_scale: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        COMPUTE_DW: tl.constexpr,
        COMPUTE_DB: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_v = tl.program_id(0)
        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V
        db_acc = tl.zeros([BLOCK_V], dtype=tl.float32)

        for start_m in range(0, M, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < M

            target_m = tl.load(target_ptr + offs_m, mask=mask_m, other=ignore_index)
            valid_m = mask_m & (target_m != ignore_index) & (target_m >= 0) & (target_m < V)
            lse_m = tl.load(lse_ptr + offs_m, mask=valid_m, other=0.0).to(tl.float32)
            grad_m = tl.load(grad_loss_ptr + offs_m, mask=valid_m, other=0.0).to(tl.float32) * logit_scale

            logits_tile = tl.zeros([BLOCK_M, BLOCK_V], dtype=tl.float32)
            for start_d in range(0, D, BLOCK_D):
                offs_d = start_d + tl.arange(0, BLOCK_D)
                mask_d = offs_d < D
                h_tile = tl.load(
                    x_ptr + offs_m[:, None] * D + offs_d[None, :],
                    mask=mask_m[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                w_tile = tl.load(
                    w_ptr + offs_v[:, None] * D + offs_d[None, :],
                    mask=mask_v[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                logits_tile += tl.dot(h_tile, tl.trans(w_tile), allow_tf32=False)

            if HAS_BIAS:
                b_tile = tl.load(b_ptr + offs_v, mask=mask_v, other=0.0).to(tl.float32)
                logits_tile += b_tile[None, :]
            logits_tile = logits_tile * logit_scale

            valid_mv = valid_m[:, None] & mask_v[None, :]
            safe_exp_arg = tl.where(valid_mv, logits_tile - lse_m[:, None], 0.0)
            p_tile = tl.where(valid_mv, tl.exp(safe_exp_arg), 0.0)
            is_target = valid_mv & (offs_v[None, :] == target_m[:, None])
            target_term = tl.where(is_target, 1.0 - label_smoothing, 0.0)
            smooth_term = tl.where(valid_mv, label_smoothing / V, 0.0)
            dz_tile = (p_tile - target_term - smooth_term) * grad_m[:, None]

            if COMPUTE_DB:
                db_acc += tl.sum(dz_tile, axis=0)

            if COMPUTE_DW:
                for start_d in range(0, D, BLOCK_D):
                    offs_d = start_d + tl.arange(0, BLOCK_D)
                    mask_d = offs_d < D
                    h_tile = tl.load(
                        x_ptr + offs_m[:, None] * D + offs_d[None, :],
                        mask=mask_m[:, None] & mask_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    dw_part = tl.dot(tl.trans(dz_tile), h_tile, allow_tf32=False)
                    dw_ptrs = dw_ptr + offs_v[:, None] * D + offs_d[None, :]
                    mask_vd = mask_v[:, None] & mask_d[None, :]
                    if start_m == 0:
                        tl.store(dw_ptrs, dw_part, mask=mask_vd)
                    else:
                        dw_prev = tl.load(dw_ptrs, mask=mask_vd, other=0.0).to(tl.float32)
                        tl.store(dw_ptrs, dw_prev + dw_part, mask=mask_vd)

        if COMPUTE_DB:
            tl.store(db_ptr + offs_v, db_acc, mask=mask_v)
else:
    _fused_linear_ce_fwd_kernel = None
    _fused_linear_ce_bwd_dx_kernel = None
    _fused_linear_ce_bwd_dw_db_kernel = None


def _tile_sizes(M: int, V: int, D: int) -> Tuple[int, int, int]:
    return (
        64 if M >= 64 else (32 if M >= 32 else 16),
        64 if V >= 64 else (32 if V >= 32 else 16),
        64 if D >= 64 else (32 if D >= 32 else 16),
    )


def triton_fused_linear_cross_entropy(
    x_2d: Any,
    weight: Any,
    target_1d: Any,
    bias: Optional[Any] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
) -> Optional[Tuple[Any, Any]]:
    """Dispatches SRAM-fused Linear + Cross-Entropy forward pass to Triton GPU kernel."""
    if not (
        HAS_TRITON
        and _HAS_TORCH
        and isinstance(x_2d, torch.Tensor)
        and x_2d.is_cuda
        and x_2d.ndim == 2
        and weight.ndim == 2
    ):
        return None

    M, D = x_2d.shape
    V = weight.shape[0]
    if weight.shape[1] != D or M == 0 or V == 0 or D == 0:
        return None

    BLOCK_M, BLOCK_V, BLOCK_D = _tile_sizes(M, V, D)
    x_c = x_2d.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous() if bias is not None else x_c
    t_c = target_1d.contiguous()

    losses = torch.empty((M,), device=x_2d.device, dtype=torch.float32)
    lse = torch.empty((M,), device=x_2d.device, dtype=torch.float32)

    _fused_linear_ce_fwd_kernel[(triton.cdiv(M, BLOCK_M),)](
        x_c, w_c, b_c, t_c, losses, lse, M, V, D, int(ignore_index),
        label_smoothing=float(label_smoothing),
        logit_scale=float(logit_scale),
        HAS_BIAS=bias is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_V=BLOCK_V,
        BLOCK_D=BLOCK_D,
    )
    return losses, lse


def triton_fused_linear_cross_entropy_backward(
    grad_loss_1d: Any,
    x_2d: Any,
    weight: Any,
    target_1d: Any,
    lse_1d: Any,
    bias: Optional[Any] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    need_dx: bool = True,
    need_dw: bool = True,
    need_db: bool = False,
) -> Optional[Tuple[Optional[Any], Optional[Any], Optional[Any]]]:
    """Dispatches SRAM-fused Linear + Cross-Entropy backward pass to Triton GPU kernels."""
    if not (
        HAS_TRITON
        and _HAS_TORCH
        and isinstance(x_2d, torch.Tensor)
        and x_2d.is_cuda
        and x_2d.ndim == 2
        and weight.ndim == 2
    ):
        return None

    M, D = x_2d.shape
    V = weight.shape[0]
    if weight.shape[1] != D or M == 0 or V == 0 or D == 0:
        return None

    BLOCK_M, BLOCK_V, BLOCK_D = _tile_sizes(M, V, D)
    x_c = x_2d.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous() if bias is not None else x_c
    t_c = target_1d.contiguous()
    lse_c = lse_1d.contiguous()
    g_c = grad_loss_1d.to(torch.float32).contiguous()

    dx_out: Optional[Any] = None
    dw_out: Optional[Any] = None
    db_out: Optional[Any] = None

    if need_dx:
        dx_f32 = torch.zeros((M, D), device=x_2d.device, dtype=torch.float32)
        _fused_linear_ce_bwd_dx_kernel[(triton.cdiv(M, BLOCK_M),)](
            x_c, w_c, b_c, t_c, lse_c, g_c, dx_f32, M, V, D, int(ignore_index),
            label_smoothing=float(label_smoothing),
            logit_scale=float(logit_scale),
            HAS_BIAS=bias is not None,
            BLOCK_M=BLOCK_M,
            BLOCK_V=BLOCK_V,
            BLOCK_D=BLOCK_D,
        )
        dx_out = dx_f32.to(x_2d.dtype)

    if need_dw or need_db:
        dw_f32 = torch.zeros((V, D), device=weight.device, dtype=torch.float32) if need_dw else x_c
        db_f32 = torch.zeros((V,), device=weight.device, dtype=torch.float32) if need_db else x_c
        _fused_linear_ce_bwd_dw_db_kernel[(triton.cdiv(V, BLOCK_V),)](
            x_c, w_c, b_c, t_c, lse_c, g_c, dw_f32, db_f32, M, V, D, int(ignore_index),
            label_smoothing=float(label_smoothing),
            logit_scale=float(logit_scale),
            HAS_BIAS=bias is not None,
            COMPUTE_DW=bool(need_dw),
            COMPUTE_DB=bool(need_db),
            BLOCK_M=BLOCK_M,
            BLOCK_V=BLOCK_V,
            BLOCK_D=BLOCK_D,
        )
        if need_dw:
            dw_out = dw_f32.to(weight.dtype)
        if need_db and bias is not None:
            db_out = db_f32.to(bias.dtype)

    return dx_out, dw_out, db_out
