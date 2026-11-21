from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional, Tuple

from ._base import HAS_TRITON, tl, torch, triton

if TYPE_CHECKING:
    _cce_bwd_joint_kernel: Any = None
    _cce_bwd_dx_kernel: Any = None
    _cce_bwd_dw_kernel: Any = None
elif HAS_TRITON:
    _CCE_BWD_JOINT_CONFIGS = [
        triton.Config({"BLOCK_M": 64, "BLOCK_V": 128, "BLOCK_D": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_V": 128, "BLOCK_D": 64, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_V": 64, "BLOCK_D": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_V": 64, "BLOCK_D": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    ]

    _CCE_BWD_DET_CONFIGS = [
        triton.Config({"BLOCK_M": 64, "BLOCK_V": 128, "BLOCK_D": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_V": 64, "BLOCK_D": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_V": 64, "BLOCK_D": 64}, num_warps=4, num_stages=3),
    ]

    @triton.autotune(configs=_CCE_BWD_JOINT_CONFIGS, key=["V", "D"])
    @triton.jit
    def _cce_bwd_joint_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        lse_ptr,
        g_scaled_ptr,
        dx_ptr,
        dw_ptr,
        db_ptr,
        stride_xm,
        stride_xd,
        stride_wv,
        stride_wd,
        stride_dxm,
        stride_dxd,
        stride_dwv,
        stride_dwd,
        M,
        V,
        D,
        logit_scale,
        log_filter_eps,
        HAS_BIAS: tl.constexpr,
        HAS_LOGIT_SCALE: tl.constexpr,
        EVEN_D: tl.constexpr,
        USE_FILTER: tl.constexpr,
        NEED_DX: tl.constexpr,
        NEED_DW: tl.constexpr,
        NEED_DB: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_v = tl.cdiv(V, BLOCK_V)
        num_pid_in_group = GROUP_SIZE_M * num_pid_v
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_v = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_m = offs_m < M
        mask_v = offs_v < V

        g_scaled = tl.load(g_scaled_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        active_m = mask_m & (g_scaled != 0.0)
        if tl.sum(active_m.to(tl.int32), axis=0) == 0:
            return
        lse_m = tl.load(lse_ptr + offs_m, mask=active_m, other=0.0).to(tl.float32)

        acc = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            x_blk = tl.load(
                x_ptr + offs_m[:, None].to(tl.int64) * stride_xm + offs_d[None, :] * stride_xd,
                mask=active_m[:, None] if EVEN_D else (active_m[:, None] & mask_d[None, :]),
                other=0.0,
            )
            w_blk = tl.load(
                w_ptr + offs_v[:, None].to(tl.int64) * stride_wv + offs_d[None, :] * stride_wd,
                mask=mask_v[:, None] if EVEN_D else (mask_v[:, None] & mask_d[None, :]),
                other=0.0,
            )
            acc = tl.dot(x_blk, tl.trans(w_blk), acc)

        if HAS_BIAS:
            b_v = tl.load(bias_ptr + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            acc = acc + b_v[None, :]
        if HAS_LOGIT_SCALE:
            acc = acc * logit_scale

        # Mask inactive rows to -inf before exp() so invalid rows never yield inf * 0 = NaN.
        z_shift = tl.where(active_m[:, None] & mask_v[None, :], acc - lse_m[:, None], -float("inf"))
        if USE_FILTER:
            if tl.max(z_shift) < log_filter_eps:
                return

        p_f32 = tl.exp(z_shift) * g_scaled[:, None]
        if NEED_DB:
            tl.atomic_add(db_ptr + offs_v, tl.sum(p_f32, axis=0), mask=mask_v, sem="relaxed")

        if not (NEED_DX or NEED_DW):
            return

        p_lp = p_f32.to(x_ptr.dtype.element_ty)
        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            if NEED_DX:
                w_blk = tl.load(
                    w_ptr + offs_v[:, None].to(tl.int64) * stride_wv + offs_d[None, :] * stride_wd,
                    mask=mask_v[:, None] if EVEN_D else (mask_v[:, None] & mask_d[None, :]),
                    other=0.0,
                )
                tl.atomic_add(
                    dx_ptr + offs_m[:, None].to(tl.int64) * stride_dxm + offs_d[None, :] * stride_dxd,
                    tl.dot(p_lp, w_blk),
                    mask=active_m[:, None] if EVEN_D else (active_m[:, None] & mask_d[None, :]),
                    sem="relaxed",
                )
            if NEED_DW:
                x_blk = tl.load(
                    x_ptr + offs_m[:, None].to(tl.int64) * stride_xm + offs_d[None, :] * stride_xd,
                    mask=active_m[:, None] if EVEN_D else (active_m[:, None] & mask_d[None, :]),
                    other=0.0,
                )
                tl.atomic_add(
                    dw_ptr + offs_v[:, None].to(tl.int64) * stride_dwv + offs_d[None, :] * stride_dwd,
                    tl.dot(tl.trans(p_lp), x_blk),
                    mask=mask_v[:, None] if EVEN_D else (mask_v[:, None] & mask_d[None, :]),
                    sem="relaxed",
                )

    @triton.autotune(configs=_CCE_BWD_DET_CONFIGS, key=["V", "D"])
    @triton.jit
    def _cce_bwd_dx_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        lse_ptr,
        g_scaled_ptr,
        dx_ptr,
        stride_xm,
        stride_xd,
        stride_wv,
        stride_wd,
        stride_dxm,
        stride_dxd,
        M,
        V,
        D,
        logit_scale,
        HAS_BIAS: tl.constexpr,
        HAS_LOGIT_SCALE: tl.constexpr,
        EVEN_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        g_scaled = tl.load(g_scaled_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        active_m = mask_m & (g_scaled != 0.0)
        if tl.sum(active_m.to(tl.int32), axis=0) == 0:
            return
        lse_m = tl.load(lse_ptr + offs_m, mask=active_m, other=0.0).to(tl.float32)

        for v0 in range(0, V, BLOCK_V):
            offs_v = v0 + tl.arange(0, BLOCK_V)
            mask_v = offs_v < V
            acc = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
            for d0 in range(0, D, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D)
                mask_d = offs_d < D
                x_blk = tl.load(
                    x_ptr + offs_m[:, None].to(tl.int64) * stride_xm + offs_d[None, :] * stride_xd,
                    mask=active_m[:, None] if EVEN_D else (active_m[:, None] & mask_d[None, :]),
                    other=0.0,
                )
                w_blk = tl.load(
                    w_ptr + offs_v[:, None].to(tl.int64) * stride_wv + offs_d[None, :] * stride_wd,
                    mask=mask_v[:, None] if EVEN_D else (mask_v[:, None] & mask_d[None, :]),
                    other=0.0,
                )
                acc = tl.dot(x_blk, tl.trans(w_blk), acc)

            if HAS_BIAS:
                b_v = tl.load(bias_ptr + offs_v, mask=mask_v, other=0.0).to(tl.float32)
                acc = acc + b_v[None, :]
            if HAS_LOGIT_SCALE:
                acc = acc * logit_scale

            z_shift = tl.where(active_m[:, None] & mask_v[None, :], acc - lse_m[:, None], -float("inf"))
            p_lp = (tl.exp(z_shift) * g_scaled[:, None]).to(x_ptr.dtype.element_ty)

            for d0 in range(0, D, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D)
                mask_d = offs_d < D
                w_blk = tl.load(
                    w_ptr + offs_v[:, None].to(tl.int64) * stride_wv + offs_d[None, :] * stride_wd,
                    mask=mask_v[:, None] if EVEN_D else (mask_v[:, None] & mask_d[None, :]),
                    other=0.0,
                )
                dx_ptrs = dx_ptr + offs_m[:, None].to(tl.int64) * stride_dxm + offs_d[None, :] * stride_dxd
                dx_mask = active_m[:, None] if EVEN_D else (active_m[:, None] & mask_d[None, :])
                cur = tl.load(dx_ptrs, mask=dx_mask, other=0.0)
                tl.store(dx_ptrs, cur + tl.dot(p_lp, w_blk), mask=dx_mask)

    @triton.autotune(configs=_CCE_BWD_DET_CONFIGS, key=["V", "D"])
    @triton.jit
    def _cce_bwd_dw_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        target_ptr,
        lse_ptr,
        g_scaled_ptr,
        c_target_ptr,
        dw_ptr,
        db_ptr,
        stride_xm,
        stride_xd,
        stride_wv,
        stride_wd,
        stride_dwv,
        stride_dwd,
        M,
        V,
        D,
        logit_scale,
        HAS_BIAS: tl.constexpr,
        HAS_LOGIT_SCALE: tl.constexpr,
        EVEN_D: tl.constexpr,
        NEED_DW: tl.constexpr,
        NEED_DB: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_v = tl.program_id(0)
        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V
        b_v = tl.load(bias_ptr + offs_v, mask=mask_v, other=0.0).to(tl.float32) if HAS_BIAS else 0.0
        db_acc = tl.zeros((BLOCK_V,), dtype=tl.float32)

        for m0 in range(0, M, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            mask_m = offs_m < M
            g_scaled = tl.load(g_scaled_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            active_m = mask_m & (g_scaled != 0.0)
            if tl.sum(active_m.to(tl.int32), axis=0) != 0:
                lse_m = tl.load(lse_ptr + offs_m, mask=active_m, other=0.0).to(tl.float32)
                t_m = tl.load(target_ptr + offs_m, mask=active_m, other=-1)
                c_m = tl.load(c_target_ptr + offs_m, mask=active_m, other=0.0).to(tl.float32)
                acc = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    offs_d = d0 + tl.arange(0, BLOCK_D)
                    mask_d = offs_d < D
                    x_blk = tl.load(
                        x_ptr + offs_m[:, None].to(tl.int64) * stride_xm + offs_d[None, :] * stride_xd,
                        mask=active_m[:, None] if EVEN_D else (active_m[:, None] & mask_d[None, :]),
                        other=0.0,
                    )
                    w_blk = tl.load(
                        w_ptr + offs_v[:, None].to(tl.int64) * stride_wv + offs_d[None, :] * stride_wd,
                        mask=mask_v[:, None] if EVEN_D else (mask_v[:, None] & mask_d[None, :]),
                        other=0.0,
                    )
                    acc = tl.dot(x_blk, tl.trans(w_blk), acc)

                if HAS_BIAS:
                    acc = acc + b_v[None, :]
                if HAS_LOGIT_SCALE:
                    acc = acc * logit_scale

                z_shift = tl.where(active_m[:, None] & mask_v[None, :], acc - lse_m[:, None], -float("inf"))
                # Include target term inline here so deterministic mode avoids CUDA atomic index_add_.
                tgt_match = active_m[:, None] & mask_v[None, :] & (offs_v[None, :] == t_m[:, None])
                p_f32 = tl.exp(z_shift) * g_scaled[:, None] + tl.where(tgt_match, c_m[:, None], 0.0)
                if NEED_DB:
                    db_acc += tl.sum(p_f32, axis=0)
                if NEED_DW:
                    p_lp = p_f32.to(x_ptr.dtype.element_ty)
                    for d0 in range(0, D, BLOCK_D):
                        offs_d = d0 + tl.arange(0, BLOCK_D)
                        mask_d = offs_d < D
                        x_blk = tl.load(
                            x_ptr + offs_m[:, None].to(tl.int64) * stride_xm + offs_d[None, :] * stride_xd,
                            mask=active_m[:, None] if EVEN_D else (active_m[:, None] & mask_d[None, :]),
                            other=0.0,
                        )
                        dw_ptrs = dw_ptr + offs_v[:, None].to(tl.int64) * stride_dwv + offs_d[None, :] * stride_dwd
                        dw_mask = mask_v[:, None] if EVEN_D else (mask_v[:, None] & mask_d[None, :])
                        cur = tl.load(dw_ptrs, mask=dw_mask, other=0.0)
                        tl.store(dw_ptrs, cur + tl.dot(tl.trans(p_lp), x_blk), mask=dw_mask)

        if NEED_DB:
            cur_db = tl.load(db_ptr + offs_v, mask=mask_v, other=0.0)
            tl.store(db_ptr + offs_v, cur_db + db_acc, mask=mask_v)
else:
    _cce_bwd_joint_kernel = None
    _cce_bwd_dx_kernel = None
    _cce_bwd_dw_kernel = None


def triton_cce_bwd(
    x_2d: Any,
    weight: Any,
    target_1d: Any,
    lse_1d: Any,
    grad_loss_1d: Any,
    bias: Optional[Any] = None,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    need_dx: bool = True,
    need_dw: bool = True,
    need_db: bool = False,
    deterministic: Optional[bool] = None,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Computes exact (dX, dW, dB) via algebraic target/smoothing split + SRAM-tiled softmax GEMM."""
    M, D = x_2d.shape
    V = weight.shape[0]
    x_c = x_2d if x_2d.stride(1) == 1 else x_2d.contiguous()
    w_c = weight if weight.stride(1) == 1 else weight.contiguous()
    t_c = target_1d if target_1d.is_contiguous() else target_1d.contiguous()
    lse_c = lse_1d.to(torch.float32).contiguous()
    has_bias = bias is not None
    b_c = bias.to(torch.float32).contiguous() if has_bias else x_c

    valid_mask = (t_c != int(ignore_index)) & (t_c >= 0) & (t_c < V)
    safe_t = torch.where(valid_mask, t_c, 0).to(torch.long)
    g_scaled = torch.where(valid_mask, grad_loss_1d.to(torch.float32) * float(logit_scale), 0.0).contiguous()
    c_target = (-(1.0 - float(label_smoothing)) * g_scaled).contiguous()

    dx_f32 = torch.zeros((M, D), device=x_c.device, dtype=torch.float32) if need_dx else x_c
    dw_f32 = torch.zeros((V, D), device=w_c.device, dtype=torch.float32) if need_dw else w_c
    db_f32 = torch.zeros((V,), device=w_c.device, dtype=torch.float32) if (need_db and has_bias) else x_c

    use_det = bool(torch.are_deterministic_algorithms_enabled()) if deterministic is None else bool(deterministic)
    if need_dx:
        dx_f32.add_(w_c[safe_t].to(torch.float32) * c_target.unsqueeze(-1))
    if not use_det:
        if need_dw:
            dw_f32.index_add_(0, safe_t, x_c.to(torch.float32) * c_target.unsqueeze(-1))
        if need_db and has_bias:
            db_f32.index_add_(0, safe_t, c_target)

    if label_smoothing > 0.0:
        u_smooth = -(float(label_smoothing) / float(V)) * g_scaled
        if need_dx:
            w_sum = w_c.sum(dim=0, dtype=torch.float32)
            dx_f32.add_(u_smooth.unsqueeze(-1) * w_sum.unsqueeze(0))
        if need_dw:
            s_x = (u_smooth.unsqueeze(-1) * x_c.to(torch.float32)).sum(dim=0)
            dw_f32.add_(s_x.unsqueeze(0))
        if need_db and has_bias:
            db_f32.add_(u_smooth.sum())

    even_d = (D % 64) == 0
    if use_det:
        if need_dx:
            _cce_bwd_dx_kernel[lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)](
                x_c, w_c, b_c, lse_c, g_scaled, dx_f32,
                x_c.stride(0), x_c.stride(1), w_c.stride(0), w_c.stride(1),
                dx_f32.stride(0), dx_f32.stride(1),
                M, V, D, float(logit_scale),
                HAS_BIAS=has_bias, HAS_LOGIT_SCALE=float(logit_scale) != 1.0, EVEN_D=even_d,
            )
        if need_dw or (need_db and has_bias):
            t_det = torch.where(valid_mask, t_c, -1).to(torch.int64).contiguous()
            _cce_bwd_dw_kernel[lambda meta: (triton.cdiv(V, meta["BLOCK_V"]),)](
                x_c, w_c, b_c, t_det, lse_c, g_scaled, c_target, dw_f32, db_f32,
                x_c.stride(0), x_c.stride(1), w_c.stride(0), w_c.stride(1),
                dw_f32.stride(0), dw_f32.stride(1),
                M, V, D, float(logit_scale),
                HAS_BIAS=has_bias, HAS_LOGIT_SCALE=float(logit_scale) != 1.0,
                EVEN_D=even_d, NEED_DW=need_dw, NEED_DB=bool(need_db and has_bias),
            )
    else:
        # Strict probability-mass bound: skipped tail mass per row is < V * (2^-10 / V) = 2^-10 (< 0.001).
        log_filter_eps = math.log(1.0 / (1024.0 * float(max(V, 1))))
        _cce_bwd_joint_kernel[
            lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(V, meta["BLOCK_V"]),)
        ](
            x_c, w_c, b_c, lse_c, g_scaled, dx_f32, dw_f32, db_f32,
            x_c.stride(0), x_c.stride(1), w_c.stride(0), w_c.stride(1),
            dx_f32.stride(0) if need_dx else 0, dx_f32.stride(1) if need_dx else 0,
            dw_f32.stride(0) if need_dw else 0, dw_f32.stride(1) if need_dw else 0,
            M, V, D, float(logit_scale), float(log_filter_eps),
            HAS_BIAS=has_bias, HAS_LOGIT_SCALE=float(logit_scale) != 1.0,
            EVEN_D=even_d, USE_FILTER=True,
            NEED_DX=need_dx, NEED_DW=need_dw, NEED_DB=bool(need_db and has_bias),
        )

    dx_out = dx_f32.to(x_2d.dtype) if need_dx else None
    dw_out = dw_f32.to(weight.dtype) if need_dw else None
    db_out = db_f32.to(bias.dtype) if (need_db and has_bias) else None
    return dx_out, dw_out, db_out
