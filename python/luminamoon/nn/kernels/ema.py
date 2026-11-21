from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

from ._base import HAS_TRITON, _HAS_TORCH, tl, torch, triton

if TYPE_CHECKING:
    _fused_ra_ema_fwd_kernel: Any = None
    _fused_ra_ema_bwd_kernel: Any = None
elif HAS_TRITON:
    @triton.jit
    def _fused_ra_ema_fwd_kernel(
        x_ptr,
        decay_ptr,
        h0_ptr,
        out_ptr,
        h_last_ptr,
        T,
        D,
        HAS_H0: tl.constexpr,
        STORE_H_LAST: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_d = tl.program_id(0)
        pid_b = tl.program_id(1)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        if HAS_H0:
            h = tl.load(h0_ptr + pid_b * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        else:
            h = tl.zeros([BLOCK_D], dtype=tl.float32)

        b_base = pid_b * T * D
        for t in range(0, T):
            idx = b_base + t * D + offs_d
            x_t = tl.load(x_ptr + idx, mask=mask_d, other=0.0).to(tl.float32)
            a_t = tl.load(decay_ptr + idx, mask=mask_d, other=0.0).to(tl.float32)
            h = a_t * h + (1.0 - a_t) * x_t
            tl.store(out_ptr + idx, h, mask=mask_d)

        if STORE_H_LAST:
            tl.store(h_last_ptr + pid_b * D + offs_d, h, mask=mask_d)

    @triton.jit
    def _fused_ra_ema_bwd_kernel(
        grad_out_ptr,
        dh_last_ptr,
        x_ptr,
        decay_ptr,
        h0_ptr,
        out_ptr,
        dx_ptr,
        ddecay_ptr,
        dh0_ptr,
        T,
        D,
        HAS_H0: tl.constexpr,
        HAS_DH_LAST: tl.constexpr,
        COMPUTE_DH0: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_d = tl.program_id(0)
        pid_b = tl.program_id(1)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        if HAS_DH_LAST:
            delta = tl.load(dh_last_ptr + pid_b * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        else:
            delta = tl.zeros([BLOCK_D], dtype=tl.float32)

        b_base = pid_b * T * D
        for step in range(0, T):
            t = T - 1 - step
            idx = b_base + t * D + offs_d
            go_t = tl.load(grad_out_ptr + idx, mask=mask_d, other=0.0).to(tl.float32)
            a_t = tl.load(decay_ptr + idx, mask=mask_d, other=0.0).to(tl.float32)
            x_t = tl.load(x_ptr + idx, mask=mask_d, other=0.0).to(tl.float32)

            delta = delta + go_t
            dx_t = (1.0 - a_t) * delta

            if t > 0:
                prev_idx = b_base + (t - 1) * D + offs_d
                h_prev = tl.load(out_ptr + prev_idx, mask=mask_d, other=0.0).to(tl.float32)
            elif HAS_H0:
                h_prev = tl.load(h0_ptr + pid_b * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
            else:
                h_prev = tl.zeros([BLOCK_D], dtype=tl.float32)

            da_t = (h_prev - x_t) * delta
            tl.store(dx_ptr + idx, dx_t, mask=mask_d)
            tl.store(ddecay_ptr + idx, da_t, mask=mask_d)
            delta = a_t * delta

        if COMPUTE_DH0:
            tl.store(dh0_ptr + pid_b * D + offs_d, delta, mask=mask_d)
else:
    _fused_ra_ema_fwd_kernel = None
    _fused_ra_ema_bwd_kernel = None


def _select_ema_block_d(B: int, D: int) -> int:
    target_per_b = max(1, (128 + B - 1) // B)
    raw = max(16, D // target_per_b)
    return min(256, max(16, triton.next_power_of_2(min(raw, 256))))


def triton_fused_ra_ema_fwd(
    x_3d: Any,
    decay_3d: Any,
    h0_2d: Optional[Any] = None,
) -> Optional[Tuple[Any, Any]]:
    """Dispatches Fused Rational EMA forward recurrence to a single-launch Triton SRAM kernel."""
    if not (
        HAS_TRITON
        and _HAS_TORCH
        and isinstance(x_3d, torch.Tensor)
        and x_3d.is_cuda
        and x_3d.ndim == 3
        and x_3d.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return None

    B, T, D = x_3d.shape
    if B == 0 or T == 0 or D == 0:
        return None

    x_c = x_3d.contiguous()
    a_c = decay_3d.contiguous()
    has_h0 = h0_2d is not None
    h0_c = h0_2d.contiguous() if has_h0 else x_c

    out = torch.empty_like(x_c)
    h_last = torch.empty((B, D), device=x_c.device, dtype=x_c.dtype)

    BLOCK_D = _select_ema_block_d(B, D)
    grid = (triton.cdiv(D, BLOCK_D), B)
    _fused_ra_ema_fwd_kernel[grid](
        x_c,
        a_c,
        h0_c,
        out,
        h_last,
        T,
        D,
        HAS_H0=has_h0,
        STORE_H_LAST=True,
        BLOCK_D=BLOCK_D,
    )
    return out, h_last


def triton_fused_ra_ema_bwd(
    grad_out_3d: Any,
    dh_last_2d: Optional[Any],
    x_3d: Any,
    decay_3d: Any,
    h0_2d: Optional[Any],
    out_3d: Any,
    need_dh0: bool = False,
) -> Optional[Tuple[Any, Any, Optional[Any]]]:
    """Dispatches Fused Rational EMA reverse-time adjoint recurrence to a single-launch Triton SRAM kernel."""
    if not (
        HAS_TRITON
        and _HAS_TORCH
        and isinstance(x_3d, torch.Tensor)
        and x_3d.is_cuda
        and x_3d.ndim == 3
        and x_3d.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return None

    B, T, D = x_3d.shape
    if B == 0 or T == 0 or D == 0:
        return None

    go_c = grad_out_3d.contiguous()
    x_c = x_3d.contiguous()
    a_c = decay_3d.contiguous()
    out_c = out_3d.contiguous()
    has_h0 = h0_2d is not None
    h0_c = h0_2d.contiguous() if has_h0 else x_c
    has_dh_last = dh_last_2d is not None
    dh_last_c = dh_last_2d.contiguous() if has_dh_last else x_c

    dx = torch.empty_like(x_c)
    ddecay = torch.empty_like(a_c, dtype=torch.float32)
    dh0 = torch.empty((B, D), device=x_c.device, dtype=x_c.dtype) if (has_h0 and need_dh0) else x_c

    BLOCK_D = _select_ema_block_d(B, D)
    grid = (triton.cdiv(D, BLOCK_D), B)
    _fused_ra_ema_bwd_kernel[grid](
        go_c,
        dh_last_c,
        x_c,
        a_c,
        h0_c,
        out_c,
        dx,
        ddecay,
        dh0,
        T,
        D,
        HAS_H0=has_h0,
        HAS_DH_LAST=has_dh_last,
        COMPUTE_DH0=bool(has_h0 and need_dh0),
        BLOCK_D=BLOCK_D,
    )
    return dx, ddecay.to(decay_3d.dtype), (dh0 if (has_h0 and need_dh0) else None)
