from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._base import HAS_TRITON, _HAS_TORCH, tl, torch, triton

if TYPE_CHECKING:
    _ra_softmax_fwd_kernel: Any = None
    _ra_softmax_bwd_kernel: Any = None
elif HAS_TRITON:
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
        tl.store(row_out + cols, phi / safe_sum, mask=mask)

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
else:
    _ra_softmax_fwd_kernel = None
    _ra_softmax_bwd_kernel = None


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
    BLOCK_SIZE = max(16, triton.next_power_of_2(N))
    if BLOCK_SIZE > 8192:
        return None

    out = torch.empty_like(scores)
    n_rows = scores.numel() // N
    _ra_softmax_fwd_kernel[(n_rows,)](
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
    BLOCK_SIZE = max(16, triton.next_power_of_2(N))
    if BLOCK_SIZE > 8192:
        return None

    grad_scores = torch.empty_like(scores)
    n_rows = scores.numel() // N
    _ra_softmax_bwd_kernel[(n_rows,)](
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
