from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

from ._base import HAS_TRITON, _HAS_TORCH, tl, torch, triton

if TYPE_CHECKING:
    _ra_norm_fwd_kernel: Any = None
    _ra_norm_bwd_kernel: Any = None
    _pos_ra_glu_fwd_kernel: Any = None
    _pos_ra_glu_bwd_kernel: Any = None
    _ra_tanh_fwd_kernel: Any = None
    _ra_tanh_bwd_kernel: Any = None
elif HAS_TRITON:
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
        l1_mean = tl.sum(tl.abs(x), axis=0) / D
        normed = x * (1.0 / (l1_mean + eps))

        if HAS_WEIGHT:
            w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            normed = normed * w

        if HAS_BIAS:
            b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            normed = normed + b

        tl.store(y_row + cols, normed, mask=mask)

    @triton.jit
    def _ra_norm_bwd_kernel(
        grad_y_ptr,
        x_ptr,
        weight_ptr,
        dx_ptr,
        dw_ptr,
        db_ptr,
        n_rows,
        stride_row,
        D: tl.constexpr,
        eps: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_progs = tl.num_programs(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < D

        w = (
            tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            if HAS_WEIGHT
            else tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        )
        dw_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        db_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        for row_idx in range(pid, n_rows, num_progs):
            offset = row_idx * stride_row
            gy = tl.load(grad_y_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(x_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
            l1_mean = tl.sum(tl.abs(x), axis=0) / D
            inv_denom = 1.0 / (l1_mean + eps)
            x_hat = x * inv_denom

            if HAS_WEIGHT:
                dy_w = gy * w
                dw_acc += gy * x_hat
            else:
                dy_w = gy

            if HAS_BIAS:
                db_acc += gy

            c = tl.sum(tl.where(mask, dy_w * x_hat, 0.0), axis=0) / D
            sgn_x = tl.where(x > 0.0, 1.0, tl.where(x < 0.0, -1.0, 0.0))
            dx = inv_denom * (dy_w - sgn_x * c)
            tl.store(dx_ptr + offset + cols, dx, mask=mask)

        if HAS_WEIGHT:
            tl.atomic_add(dw_ptr + cols, dw_acc, mask=mask)
        if HAS_BIAS:
            tl.atomic_add(db_ptr + cols, db_acc, mask=mask)

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

        b_term = b_val * (b_val + tl.abs(z))
        ra_tanh = (b_term * z) / (b_term + z * z)
        if b_val == 1.0:
            sigma = 0.5 * (1.0 + ra_tanh)
        else:
            sigma = 0.5 * (1.0 + ra_tanh / b_val)

        tl.store(out_ptr + offsets, u * sigma, mask=mask)

    @triton.jit
    def _pos_ra_glu_bwd_kernel(
        grad_out_ptr,
        gate_ptr,
        up_ptr,
        dgate_ptr,
        dup_ptr,
        n_elements,
        b_val: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        g = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        abs_z = tl.abs(z)
        b_term = b_val * (b_val + abs_z)
        denom = b_term + z * z
        ra_tanh = (b_term * z) / denom
        if b_val == 1.0:
            sigma = 0.5 * (1.0 + ra_tanh)
        else:
            sigma = 0.5 * (1.0 + ra_tanh / b_val)

        d_sigma = (0.5 * (b_val * b_val) * (b_val + 2.0 * abs_z)) / (denom * denom)
        tl.store(dgate_ptr + offsets, g * u * d_sigma, mask=mask)
        tl.store(dup_ptr + offsets, g * sigma, mask=mask)

    @triton.jit
    def _ra_tanh_fwd_kernel(
        x_ptr,
        out_ptr,
        n_elements,
        b_val: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b_term = b_val * (b_val + tl.abs(x))
        out = (b_term * x) / (b_term + x * x)
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _ra_tanh_bwd_kernel(
        grad_out_ptr,
        x_ptr,
        dx_ptr,
        n_elements,
        b_val: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        g = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        abs_x = tl.abs(x)
        b_term = b_val * (b_val + abs_x)
        denom = b_term + x * x
        deriv = ((b_val * b_val) * (b_val + 2.0 * abs_x)) / (denom * denom)
        tl.store(dx_ptr + offsets, g * deriv, mask=mask)
else:
    _ra_norm_fwd_kernel = None
    _ra_norm_bwd_kernel = None
    _pos_ra_glu_fwd_kernel = None
    _pos_ra_glu_bwd_kernel = None
    _ra_tanh_fwd_kernel = None
    _ra_tanh_bwd_kernel = None


def triton_ra_norm(
    x: Any,
    weight: Optional[Any] = None,
    bias: Optional[Any] = None,
    eps: float = 1e-6,
) -> Any:
    """Dispatches RaNorm execution to Triton GPU kernel when available."""
    if not (HAS_TRITON and _HAS_TORCH and isinstance(x, torch.Tensor) and x.is_cuda):
        return None

    D = x.shape[-1]
    if D == 0 or D > 8192:
        return None
    BLOCK_SIZE = max(16, triton.next_power_of_2(D))

    if not x.is_contiguous():
        x = x.contiguous()
    if weight is not None and not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    y = torch.empty_like(x)
    n_rows = x.numel() // D
    num_warps = 4 if BLOCK_SIZE <= 1024 else 8
    _ra_norm_fwd_kernel[(n_rows,)](
        x,
        y,
        weight if weight is not None else x,
        bias if bias is not None else x,
        D,
        D,
        D=D,
        eps=float(eps),
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y


def triton_ra_norm_backward(
    grad_y: Any,
    x: Any,
    weight: Optional[Any] = None,
    has_bias: bool = False,
    eps: float = 1e-6,
) -> Optional[Tuple[Any, Optional[Any], Optional[Any]]]:
    """Dispatches RaNorm backward execution to persistent Triton GPU kernel when available."""
    if not (HAS_TRITON and _HAS_TORCH and isinstance(x, torch.Tensor) and x.is_cuda):
        return None
    if hasattr(torch, "are_deterministic_algorithms_enabled") and torch.are_deterministic_algorithms_enabled():
        if weight is not None or has_bias:
            return None

    D = x.shape[-1]
    if D == 0 or D > 8192:
        return None
    BLOCK_SIZE = max(16, triton.next_power_of_2(D))

    gy_c = grad_y if grad_y.is_contiguous() else grad_y.contiguous()
    x_c = x if x.is_contiguous() else x.contiguous()
    w_c = (weight if weight.is_contiguous() else weight.contiguous()) if weight is not None else None

    dx = torch.empty_like(x_c)
    has_w = w_c is not None
    dw_f32 = torch.zeros((D,), device=x.device, dtype=torch.float32) if has_w else dx
    db_f32 = torch.zeros((D,), device=x.device, dtype=torch.float32) if has_bias else dx

    n_rows = x_c.numel() // D
    num_progs = min(n_rows, 256) if (has_w or has_bias) else n_rows
    num_warps = 4 if BLOCK_SIZE <= 1024 else 8
    _ra_norm_bwd_kernel[(max(1, num_progs),)](
        gy_c,
        x_c,
        w_c if w_c is not None else x_c,
        dx,
        dw_f32,
        db_f32,
        n_rows,
        D,
        D=D,
        eps=float(eps),
        HAS_WEIGHT=has_w,
        HAS_BIAS=bool(has_bias),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    dw = dw_f32.to(w_c.dtype) if w_c is not None else None
    db = db_f32.to(w_c.dtype if w_c is not None else x_c.dtype) if has_bias else None
    return dx, dw, db


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
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _pos_ra_glu_fwd_kernel[grid](
        gate,
        up,
        out,
        n_elements,
        b_val=float(b),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def triton_pos_ra_glu_backward(
    grad_out: Any,
    gate: Any,
    up: Any,
    b: float = 1.0,
) -> Optional[Tuple[Any, Any]]:
    """Dispatches PosRaGLU backward execution to Triton GPU kernel."""
    if not (HAS_TRITON and _HAS_TORCH and isinstance(gate, torch.Tensor) and gate.is_cuda):
        return None

    g_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
    gate_c = gate if gate.is_contiguous() else gate.contiguous()
    up_c = up if up.is_contiguous() else up.contiguous()

    n_elements = gate_c.numel()
    dgate = torch.empty_like(gate_c)
    dup = torch.empty_like(up_c)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _pos_ra_glu_bwd_kernel[grid](
        g_c,
        gate_c,
        up_c,
        dgate,
        dup,
        n_elements,
        b_val=float(b),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return dgate, dup


def triton_ra_tanh(
    x: Any,
    b: float = 1.0,
) -> Any:
    """Dispatches RaTanh forward execution to Triton GPU kernel."""
    if not (HAS_TRITON and _HAS_TORCH and isinstance(x, torch.Tensor) and x.is_cuda):
        return None

    x_c = x if x.is_contiguous() else x.contiguous()
    n_elements = x_c.numel()
    out = torch.empty_like(x_c)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _ra_tanh_fwd_kernel[grid](
        x_c,
        out,
        n_elements,
        b_val=float(b),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def triton_ra_tanh_backward(
    grad_out: Any,
    x: Any,
    b: float = 1.0,
) -> Any:
    """Dispatches RaTanh backward execution to Triton GPU kernel."""
    if not (HAS_TRITON and _HAS_TORCH and isinstance(x, torch.Tensor) and x.is_cuda):
        return None

    g_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
    x_c = x if x.is_contiguous() else x.contiguous()
    n_elements = x_c.numel()
    dx = torch.empty_like(x_c)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _ra_tanh_bwd_kernel[grid](
        g_c,
        x_c,
        dx,
        n_elements,
        b_val=float(b),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return dx
