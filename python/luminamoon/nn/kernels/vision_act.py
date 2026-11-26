from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional, Tuple

from ._base import HAS_TRITON, _HAS_TORCH, tl, torch, triton

_SQRT_2_OVER_PI: float = math.sqrt(2.0 / math.pi)
_INV_SQRT_2_OVER_PI: float = 1.0 / _SQRT_2_OVER_PI

if TYPE_CHECKING:
    _ra_gelu_fwd_kernel: Any = None
    _ra_gelu_bwd_kernel: Any = None
    _ra_bn2d_sum_x_kernel: Any = None
    _ra_bn2d_sum_mad_kernel: Any = None
    _ra_bn2d_norm_fwd_kernel: Any = None
    _ra_bn2d_bwd_reduce_kernel: Any = None
    _ra_bn2d_bwd_dx_kernel: Any = None
elif HAS_TRITON:

    @triton.jit
    def _ra_gelu_fwd_kernel(
        x_ptr,
        out_ptr,
        n_elements,
        k_val: tl.constexpr,
        inv_2k: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        z = k_val * x
        abs_z = tl.abs(z)
        inv_m = 1.0 / tl.maximum(1.0, abs_z)
        ub = inv_m
        ua = abs_z * inv_m
        uz = z * inv_m
        inv_dn = 1.0 / (ub * (ub + ua) + ua * ua)
        q = (0.5 * ub * ub) * inv_dn
        out = tl.where(z >= 0.0, x * (1.0 - q), (inv_2k * ub) * uz * inv_dn)
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _ra_gelu_bwd_kernel(
        grad_out_ptr,
        x_ptr,
        dx_ptr,
        n_elements,
        k_val: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        g = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        z = k_val * x
        abs_z = tl.abs(z)
        inv_m = 1.0 / tl.maximum(1.0, abs_z)
        ub = inv_m
        ua = abs_z * inv_m
        inv_dn = 1.0 / (ub * (ub + ua) + ua * ua)
        q = (0.5 * ub * ub) * inv_dn
        is_pos = z >= 0.0
        sigma = tl.where(is_pos, 1.0 - q, q)
        deriv = tl.where(
            is_pos,
            sigma + q * ua * (ub + 2.0 * ua) * inv_dn,
            q * (ub * ub - ua * ua) * inv_dn,
        )
        tl.store(dx_ptr + offsets, g * deriv, mask=mask)

    @triton.jit
    def _ra_bn2d_sum_x_kernel(
        x_ptr,
        partial_sum_ptr,
        stride_b,
        S,
        M,
        n_splits,
        chunk_m,
        BLOCK_M: tl.constexpr,
    ):
        c = tl.program_id(0).to(tl.int64)
        sp = tl.program_id(1)
        start = sp * chunk_m
        end = tl.minimum(start + chunk_m, M)
        c_off = c * S

        acc = tl.zeros([BLOCK_M], dtype=tl.float32)
        offs_base = tl.arange(0, BLOCK_M)
        pos = start
        while pos < end:
            idx = pos + offs_base
            mask = idx < end
            b = idx // S
            s = idx - b * S
            ptr_off = b.to(tl.int64) * stride_b + c_off + s.to(tl.int64)
            x = tl.load(x_ptr + ptr_off, mask=mask, other=0.0).to(tl.float32)
            acc += x
            pos += BLOCK_M
        tl.store(partial_sum_ptr + c * n_splits + sp, tl.sum(acc, axis=0))

    @triton.jit
    def _ra_bn2d_sum_mad_kernel(
        x_ptr,
        mean_ptr,
        partial_mad_ptr,
        stride_b,
        S,
        M,
        n_splits,
        chunk_m,
        BLOCK_M: tl.constexpr,
    ):
        c = tl.program_id(0).to(tl.int64)
        sp = tl.program_id(1)
        start = sp * chunk_m
        end = tl.minimum(start + chunk_m, M)
        c_off = c * S
        mean_c = tl.load(mean_ptr + c).to(tl.float32)

        acc = tl.zeros([BLOCK_M], dtype=tl.float32)
        offs_base = tl.arange(0, BLOCK_M)
        pos = start
        while pos < end:
            idx = pos + offs_base
            mask = idx < end
            b = idx // S
            s = idx - b * S
            ptr_off = b.to(tl.int64) * stride_b + c_off + s.to(tl.int64)
            x = tl.load(x_ptr + ptr_off, mask=mask, other=0.0).to(tl.float32)
            acc += tl.where(mask, tl.abs(x - mean_c), 0.0)
            pos += BLOCK_M
        tl.store(partial_mad_ptr + c * n_splits + sp, tl.sum(acc, axis=0))

    @triton.jit
    def _ra_bn2d_norm_fwd_kernel(
        x_ptr,
        y_ptr,
        mean_ptr,
        inv_d_ptr,
        weight_ptr,
        bias_ptr,
        stride_b,
        S,
        M,
        HAS_AFFINE: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        c = tl.program_id(0).to(tl.int64)
        tile = tl.program_id(1).to(tl.int64)
        idx = tile * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = idx < M
        b = idx // S
        s = idx - b * S
        ptr_off = b * stride_b + c * S + s

        mean_c = tl.load(mean_ptr + c).to(tl.float32)
        inv_d_c = tl.load(inv_d_ptr + c).to(tl.float32)
        x = tl.load(x_ptr + ptr_off, mask=mask, other=0.0).to(tl.float32)
        if HAS_AFFINE:
            w_c = tl.load(weight_ptr + c).to(tl.float32)
            b_c = tl.load(bias_ptr + c).to(tl.float32)
            y = (x - mean_c) * (inv_d_c * w_c) + b_c
        else:
            y = (x - mean_c) * inv_d_c
        tl.store(y_ptr + ptr_off, y, mask=mask)

    @triton.jit
    def _ra_bn2d_bwd_reduce_kernel(
        grad_y_ptr,
        x_ptr,
        mean_ptr,
        inv_d_ptr,
        p_gy_ptr,
        p_gy_xhat_ptr,
        p_sgn_ptr,
        stride_b,
        S,
        M,
        n_splits,
        chunk_m,
        BLOCK_M: tl.constexpr,
    ):
        c = tl.program_id(0).to(tl.int64)
        sp = tl.program_id(1)
        start = sp * chunk_m
        end = tl.minimum(start + chunk_m, M)
        c_off = c * S
        mean_c = tl.load(mean_ptr + c).to(tl.float32)
        inv_d_c = tl.load(inv_d_ptr + c).to(tl.float32)

        acc_gy = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc_gy_xh = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc_sgn = tl.zeros([BLOCK_M], dtype=tl.float32)
        offs_base = tl.arange(0, BLOCK_M)
        pos = start
        while pos < end:
            idx = pos + offs_base
            mask = idx < end
            b = idx // S
            s = idx - b * S
            ptr_off = b.to(tl.int64) * stride_b + c_off + s.to(tl.int64)
            gy = tl.load(grad_y_ptr + ptr_off, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(x_ptr + ptr_off, mask=mask, other=0.0).to(tl.float32)
            d = x - mean_c
            sgn = tl.where(mask, (d > 0.0).to(tl.float32) - (d < 0.0).to(tl.float32), 0.0)
            acc_gy += gy
            acc_gy_xh += gy * (d * inv_d_c)
            acc_sgn += sgn
            pos += BLOCK_M

        out_off = c * n_splits + sp
        tl.store(p_gy_ptr + out_off, tl.sum(acc_gy, axis=0))
        tl.store(p_gy_xhat_ptr + out_off, tl.sum(acc_gy_xh, axis=0))
        tl.store(p_sgn_ptr + out_off, tl.sum(acc_sgn, axis=0))

    @triton.jit
    def _ra_bn2d_bwd_dx_kernel(
        grad_y_ptr,
        x_ptr,
        dx_ptr,
        mean_ptr,
        inv_d_ptr,
        weight_ptr,
        u_bar_ptr,
        c1_ptr,
        s_bar_ptr,
        stride_b,
        S,
        M,
        HAS_WEIGHT: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        c = tl.program_id(0).to(tl.int64)
        tile = tl.program_id(1).to(tl.int64)
        idx = tile * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = idx < M
        b = idx // S
        s = idx - b * S
        ptr_off = b * stride_b + c * S + s

        mean_c = tl.load(mean_ptr + c).to(tl.float32)
        inv_d_c = tl.load(inv_d_ptr + c).to(tl.float32)
        u_bar_c = tl.load(u_bar_ptr + c).to(tl.float32)
        c1_c = tl.load(c1_ptr + c).to(tl.float32)
        s_bar_c = tl.load(s_bar_ptr + c).to(tl.float32)
        gy = tl.load(grad_y_ptr + ptr_off, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + ptr_off, mask=mask, other=0.0).to(tl.float32)
        if HAS_WEIGHT:
            w_c = tl.load(weight_ptr + c).to(tl.float32)
            u = gy * w_c
        else:
            u = gy
        d = x - mean_c
        sgn = (d > 0.0).to(tl.float32) - (d < 0.0).to(tl.float32)
        dx = inv_d_c * ((u - u_bar_c) - c1_c * (sgn - s_bar_c))
        tl.store(dx_ptr + ptr_off, dx, mask=mask)
else:
    _ra_gelu_fwd_kernel = None
    _ra_gelu_bwd_kernel = None
    _ra_bn2d_sum_x_kernel = None
    _ra_bn2d_sum_mad_kernel = None
    _ra_bn2d_norm_fwd_kernel = None
    _ra_bn2d_bwd_reduce_kernel = None
    _ra_bn2d_bwd_dx_kernel = None


def _bn2d_split_plan(C: int, M: int) -> Tuple[int, int]:
    # Target ~512 concurrent CTAs across SMs while giving each CTA at least 4096 elements.
    max_by_m = max(1, (M + 4095) // 4096)
    target_by_c = max(1, (512 + C - 1) // C)
    n_splits = min(64, max_by_m, target_by_c)
    chunk_m = (M + n_splits - 1) // n_splits
    return n_splits, chunk_m


def triton_ra_gelu(x: Any) -> Any:
    """Dispatches RaGELU forward execution to a fused Triton GPU kernel."""
    if not (
        HAS_TRITON and _HAS_TORCH and isinstance(x, torch.Tensor) and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return None
    if x.numel() == 0:
        return torch.empty_like(x)
    x_c = x if x.is_contiguous() else x.contiguous()
    n_elements = x_c.numel()
    out = torch.empty_like(x_c)
    BLOCK_SIZE = 1024
    _ra_gelu_fwd_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        x_c, out, n_elements, k_val=_SQRT_2_OVER_PI, inv_2k=0.5 * _INV_SQRT_2_OVER_PI,
        BLOCK_SIZE=BLOCK_SIZE, enable_fp_fusion=True,
    )
    return out


def triton_ra_gelu_backward(grad_out: Any, x: Any) -> Any:
    """Dispatches RaGELU backward execution to a fused Triton GPU kernel."""
    if not (
        HAS_TRITON and _HAS_TORCH and isinstance(x, torch.Tensor) and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return None
    if x.numel() == 0:
        return torch.empty_like(x)
    g_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
    x_c = x if x.is_contiguous() else x.contiguous()
    n_elements = x_c.numel()
    dx = torch.empty_like(x_c)
    BLOCK_SIZE = 1024
    _ra_gelu_bwd_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        g_c, x_c, dx, n_elements, k_val=_SQRT_2_OVER_PI, BLOCK_SIZE=BLOCK_SIZE, enable_fp_fusion=True,
    )
    return dx


def triton_ra_batch_norm2d(
    x: Any,
    weight: Optional[Any],
    bias: Optional[Any],
    running_mean: Optional[Any],
    running_mad: Optional[Any],
    *,
    training: bool,
    eps: float = 1e-5,
) -> Optional[Tuple[Any, Any, Any, Any]]:
    """Fused Triton RaBatchNorm2d forward returning `(y, mean_f32, mad_f32, inv_d_f32)`."""
    if not (
        HAS_TRITON and _HAS_TORCH and isinstance(x, torch.Tensor) and x.is_cuda
        and x.ndim == 4 and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return None
    B, C, H, W = x.shape
    S = H * W
    M = B * S
    stride_b = C * S
    if C == 0 or M == 0:
        return None

    x_c = x if x.is_contiguous() else x.contiguous()
    has_affine = weight is not None and bias is not None
    w_c = (weight if weight.is_contiguous() else weight.contiguous()) if has_affine else x_c
    b_c = (bias if bias.is_contiguous() else bias.contiguous()) if has_affine else x_c

    if training:
        n_splits, chunk_m = _bn2d_split_plan(C, M)
        p_sum = torch.empty((C, n_splits), device=x.device, dtype=torch.float32)
        _ra_bn2d_sum_x_kernel[(C, n_splits)](
            x_c, p_sum, stride_b, S, M, n_splits, chunk_m, BLOCK_M=1024, num_warps=4, enable_fp_fusion=True,
        )
        mean_f32 = (p_sum.squeeze(1) if n_splits == 1 else p_sum.sum(dim=1)) / float(M)
        p_mad = torch.empty((C, n_splits), device=x.device, dtype=torch.float32)
        _ra_bn2d_sum_mad_kernel[(C, n_splits)](
            x_c, mean_f32, p_mad, stride_b, S, M, n_splits, chunk_m, BLOCK_M=1024, num_warps=4, enable_fp_fusion=True,
        )
        mad_f32 = (p_mad.squeeze(1) if n_splits == 1 else p_mad.sum(dim=1)) / float(M)
    else:
        if running_mean is None or running_mad is None:
            return None
        mean_f32 = running_mean.to(torch.float32).contiguous()
        mad_f32 = running_mad.to(torch.float32).contiguous()

    inv_d_f32 = 1.0 / (mad_f32 + float(eps))
    y = torch.empty_like(x_c)
    BLOCK_M = 1024
    _ra_bn2d_norm_fwd_kernel[(C, triton.cdiv(M, BLOCK_M))](
        x_c, y, mean_f32, inv_d_f32, w_c, b_c, stride_b, S, M,
        HAS_AFFINE=has_affine, BLOCK_M=BLOCK_M, num_warps=4, enable_fp_fusion=True,
    )
    return y, mean_f32, mad_f32, inv_d_f32


def triton_ra_batch_norm2d_backward(
    grad_y: Any,
    x: Any,
    mean_f32: Any,
    inv_d_f32: Any,
    weight: Optional[Any],
    *,
    training: bool,
    need_dx: bool,
    need_dw: bool,
    need_db: bool,
    bias_dtype: Any = None,
) -> Optional[Tuple[Optional[Any], Optional[Any], Optional[Any]]]:
    """Fused 2-kernel Triton RaBatchNorm2d backward returning `(dx, dw, db)`."""
    if not (
        HAS_TRITON and _HAS_TORCH and isinstance(x, torch.Tensor) and x.is_cuda
        and x.ndim == 4 and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return None
    B, C, H, W = x.shape
    S = H * W
    M = B * S
    stride_b = C * S
    if C == 0 or M == 0:
        return None

    gy_c = grad_y if grad_y.is_contiguous() else grad_y.contiguous()
    x_c = x if x.is_contiguous() else x.contiguous()
    m_c = mean_f32 if mean_f32.is_contiguous() else mean_f32.contiguous()
    id_c = inv_d_f32 if inv_d_f32.is_contiguous() else inv_d_f32.contiguous()
    has_w = weight is not None
    w_c = (weight if weight.is_contiguous() else weight.contiguous()) if has_w else x_c

    n_splits, chunk_m = _bn2d_split_plan(C, M)
    p_gy = torch.empty((C, n_splits), device=x.device, dtype=torch.float32)
    p_gy_xh = torch.empty((C, n_splits), device=x.device, dtype=torch.float32)
    p_sgn = torch.empty((C, n_splits), device=x.device, dtype=torch.float32)
    _ra_bn2d_bwd_reduce_kernel[(C, n_splits)](
        gy_c, x_c, m_c, id_c, p_gy, p_gy_xh, p_sgn, stride_b, S, M, n_splits, chunk_m,
        BLOCK_M=1024, num_warps=4, enable_fp_fusion=True,
    )
    sum_gy = p_gy.squeeze(1) if n_splits == 1 else p_gy.sum(dim=1)
    sum_gy_xh = p_gy_xh.squeeze(1) if n_splits == 1 else p_gy_xh.sum(dim=1)
    sum_sgn = p_sgn.squeeze(1) if n_splits == 1 else p_sgn.sum(dim=1)

    dw = sum_gy_xh.to(weight.dtype) if (need_dw and weight is not None) else None
    db_dt = bias_dtype if bias_dtype is not None else (weight.dtype if weight is not None else x.dtype)
    db = sum_gy.to(db_dt) if need_db else None

    dx: Optional[Any] = None
    if need_dx:
        if training:
            inv_m = 1.0 / float(M)
            w_f32 = w_c.to(torch.float32) if has_w else None
            u_bar = (sum_gy * w_f32 if w_f32 is not None else sum_gy) * inv_m
            c1 = (sum_gy_xh * w_f32 if w_f32 is not None else sum_gy_xh) * inv_m
            s_bar = sum_sgn * inv_m
        else:
            # In eval mode, mean and mad are detached running stats so dx = gy * w * inv_d.
            zeros = torch.zeros((C,), device=x.device, dtype=torch.float32)
            u_bar, c1, s_bar = zeros, zeros, zeros
        dx = torch.empty_like(x_c)
        BLOCK_M = 1024
        _ra_bn2d_bwd_dx_kernel[(C, triton.cdiv(M, BLOCK_M))](
            gy_c, x_c, dx, m_c, id_c, w_c, u_bar, c1, s_bar, stride_b, S, M,
            HAS_WEIGHT=has_w, BLOCK_M=BLOCK_M, num_warps=4, enable_fp_fusion=True,
        )
    return dx, dw, db
