from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ._base import HAS_TRITON, _HAS_TORCH, tl, torch, triton

if TYPE_CHECKING:
    _ra_norm_fwd_kernel: Any = None
    _pos_ra_glu_fwd_kernel: Any = None
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
else:
    _ra_norm_fwd_kernel = None
    _pos_ra_glu_fwd_kernel = None


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
        BLOCK_SIZE = triton.next_power_of_2(D)
    else:
        BLOCK_SIZE = max(16, D)

    y = torch.empty_like(x)
    n_rows = x.numel() // D
    _ra_norm_fwd_kernel[(n_rows,)](
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
