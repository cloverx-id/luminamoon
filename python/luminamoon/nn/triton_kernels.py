from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    HAS_TRITON: bool = False
    _ra_norm_fwd_kernel: Any = None
    _pos_ra_glu_fwd_kernel: Any = None
    _ra_softmax_fwd_kernel: Any = None
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
            sigma = 0.5 * (1.0 + ra_tanh)

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
            delta = max_s - s

            u = delta * scale
            num = tl.maximum(1.0 - u, 0.0)
            den = 1.0 + u
            q = num / den

            q2 = q * q
            q4 = q2 * q2
            phi = q4 * q4

            phi = tl.where(mask, phi, 0.0)
            sum_phi = tl.sum(phi, axis=0)

            p = phi / sum_phi
            tl.store(row_out + cols, p, mask=mask)
    else:
        _ra_norm_fwd_kernel = None
        _pos_ra_glu_fwd_kernel = None
        _ra_softmax_fwd_kernel = None

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
        x.stride(-2) if x.ndim > 1 else D,
        y.stride(-2) if y.ndim > 1 else D,
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

    N = scores.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(N)  # type: ignore[attr-defined]
    if BLOCK_SIZE > 8192:
        return None

    out = torch.empty_like(scores)
    n_rows = scores.numel() // N
    grid = (n_rows,)

    _ra_softmax_fwd_kernel[grid](
        scores,
        out,
        scores.stride(-2) if scores.ndim > 1 else N,
        N=N,
        scale=float(scale),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
