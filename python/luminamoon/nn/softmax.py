from __future__ import annotations

from typing import Any, Tuple

from ._compat import (
    _HAS_NUMPY,
    _HAS_RUST_SIMD,
    _HAS_TORCH,
    _HAS_TRITON,
    _is_torch_compiling,
    _rs_engine,
    np,
    torch,
)
from .kernels import triton_ra_softmax, triton_ra_softmax_backward

_MAX_CHUNK_ELEMENTS: int = 4_194_304


def _ra_softmax_chunk_fwd(s_chunk: Any, dim: int, scale: float) -> Any:
    """Computes Cayley-Pade rational softmax on a single chunk without building an autograd graph."""
    finite = torch.isfinite(s_chunk)
    max_s = s_chunk.max(dim=dim, keepdim=True)[0]
    delta = torch.where(finite, torch.where(s_chunk == max_s, 0.0, max_s - s_chunk), float("inf"))
    u = delta * scale
    u.mul_(-1.0).add_(1.0).clamp_(min=0.0)
    den = delta.mul_(scale).add_(1.0)
    q = torch.where(finite, u.div_(den), torch.zeros_like(u))
    q.square_()
    q.square_()
    phi = q.square_()
    sum_phi = phi.sum(dim=dim, keepdim=True).clamp(min=1e-38)
    return phi.div_(sum_phi)


def _ra_softmax_chunk_bwd(s_chunk: Any, p_chunk: Any, g_chunk: Any, dim: int, scale: float) -> Any:
    """Computes exact closed-form backward gradient of Cayley-Pade rational softmax."""
    orig_dtype = s_chunk.dtype
    s_f32 = s_chunk.to(torch.float32) if orig_dtype in (torch.float16, torch.bfloat16) else s_chunk
    p_f32 = p_chunk.to(torch.float32) if orig_dtype in (torch.float16, torch.bfloat16) else p_chunk
    g_f32 = g_chunk.to(torch.float32) if orig_dtype in (torch.float16, torch.bfloat16) else g_chunk

    finite = torch.isfinite(s_f32)
    max_s, max_idx = s_f32.max(dim=dim, keepdim=True)
    delta = torch.where(finite, torch.where(s_f32 == max_s, 0.0, max_s - s_f32), float("inf"))
    u = delta * scale
    active = finite & (u < 1.0)
    safe_u = torch.where(active, u, torch.zeros_like(u))

    sum_gp = (g_f32 * p_f32).sum(dim=dim, keepdim=True)
    w = (16.0 * scale) / (1.0 - safe_u * safe_u)
    ds_direct = torch.where(active, p_f32 * (g_f32 - sum_gp) * w, torch.zeros_like(p_f32))

    sum_ds = ds_direct.sum(dim=dim, keepdim=True)
    ds_direct.scatter_add_(dim, max_idx, -sum_ds)
    return ds_direct.to(orig_dtype) if orig_dtype in (torch.float16, torch.bfloat16) else ds_direct


def _ra_softmax_forward_no_grad(scores: Any, dim: int, scale: float) -> Any:
    """Memory-bounded forward execution of ra_softmax outside autograd."""
    ndim = scores.ndim
    norm_dim = dim if dim >= 0 else ndim + dim

    if (
        _HAS_TRITON
        and scores.is_cuda
        and norm_dim == ndim - 1
        and scores.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        triton_p = triton_ra_softmax(scores, scale=scale)
        if triton_p is not None:
            return triton_p

    if (
        _HAS_RUST_SIMD
        and not _is_torch_compiling()
        and not scores.is_cuda
        and scores.dtype == torch.float32
        and scores.is_contiguous()
        and norm_dim == ndim - 1
    ):
        out = torch.empty_like(scores)
        cols = scores.shape[-1]
        rows = scores.numel() // cols
        _rs_engine.ra_softmax_f32_pyo3(scores, out, rows, cols, scale)
        return out

    if norm_dim == ndim - 1 and scores.numel() > _MAX_CHUNK_ELEMENTS:
        cols = scores.shape[-1]
        rows = scores.numel() // cols
        flat_in = scores.reshape(rows, cols)
        flat_out = torch.empty_like(flat_in)
        chunk_rows = max(1, _MAX_CHUNK_ELEMENTS // max(1, cols))
        for r0 in range(0, rows, chunk_rows):
            r1 = min(rows, r0 + chunk_rows)
            flat_out[r0:r1] = _ra_softmax_chunk_fwd(flat_in[r0:r1], dim=-1, scale=scale)
        return flat_out.reshape(scores.shape)

    return _ra_softmax_chunk_fwd(scores, dim=norm_dim, scale=scale)


def _ra_softmax_backward_no_grad(scores: Any, out: Any, grad_out: Any, dim: int, scale: float) -> Any:
    """Memory-bounded backward execution of ra_softmax using closed-form analytical derivative."""
    ndim = scores.ndim
    norm_dim = dim if dim >= 0 else ndim + dim

    if (
        _HAS_TRITON
        and scores.is_cuda
        and norm_dim == ndim - 1
        and scores.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        triton_ds = triton_ra_softmax_backward(scores, out, grad_out, scale=scale)
        if triton_ds is not None:
            return triton_ds

    if norm_dim == ndim - 1 and scores.numel() > _MAX_CHUNK_ELEMENTS:
        cols = scores.shape[-1]
        rows = scores.numel() // cols
        flat_s = scores.reshape(rows, cols)
        flat_p = out.reshape(rows, cols)
        flat_g = grad_out.reshape(rows, cols)
        flat_ds = torch.empty_like(flat_s)
        chunk_rows = max(1, _MAX_CHUNK_ELEMENTS // max(1, cols))
        for r0 in range(0, rows, chunk_rows):
            r1 = min(rows, r0 + chunk_rows)
            flat_ds[r0:r1] = _ra_softmax_chunk_bwd(
                flat_s[r0:r1], flat_p[r0:r1], flat_g[r0:r1], dim=-1, scale=scale
            )
        return flat_ds.reshape(scores.shape)

    return _ra_softmax_chunk_bwd(scores, out, grad_out, dim=norm_dim, scale=scale)


_RaSoftmaxFunction: Any = None

if _HAS_TORCH:

    class _RaSoftmaxFunctionImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, scores: Any, dim: int, scale: float) -> Any:
            with torch.no_grad():
                out = _ra_softmax_forward_no_grad(scores, dim=dim, scale=scale)
            ctx.save_for_backward(scores, out)
            ctx.dim = dim
            ctx.scale = scale
            return out

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Tuple[Any, None, None]:  # type: ignore[override]
            scores, out = ctx.saved_tensors
            with torch.no_grad():
                grad_scores = _ra_softmax_backward_no_grad(
                    scores, out, grad_out, dim=ctx.dim, scale=ctx.scale
                )
            return grad_scores, None, None

    _RaSoftmaxFunction = _RaSoftmaxFunctionImpl


def ra_softmax(scores: Any, dim: int = -1, scale: float = 0.0625) -> Any:
    """Cayley-Pade Rational Softmax with compact support and closed-form autograd."""
    scale = float(scale)

    if _HAS_TORCH and isinstance(scores, torch.Tensor):
        if scores.requires_grad and torch.is_grad_enabled():
            return _RaSoftmaxFunction.apply(scores, int(dim), scale)
        with torch.no_grad():
            return _ra_softmax_forward_no_grad(scores, dim=int(dim), scale=scale)

    if _HAS_NUMPY and isinstance(scores, np.ndarray):
        if (
            _HAS_RUST_SIMD
            and not _is_torch_compiling()
            and scores.dtype == np.float32
            and scores.flags.c_contiguous
            and dim in (-1, scores.ndim - 1)
        ):
            out = np.empty_like(scores)
            cols = scores.shape[-1]
            rows = scores.size // cols
            _rs_engine.ra_softmax_f32_pyo3(scores, out, rows, cols, scale)
            return out

        finite = np.isfinite(scores)
        max_scores = np.max(scores, axis=dim, keepdims=True)
        delta = np.where(finite, np.where(scores == max_scores, 0.0, max_scores - scores), np.inf)
        u = delta * scale
        q = np.where(finite, np.maximum(1.0 - u, 0.0) / (1.0 + u), 0.0)
        q2 = q * q
        q4 = q2 * q2
        phi = q4 * q4
        sum_phi = np.maximum(np.sum(phi, axis=dim, keepdims=True), 1e-38)
        return phi / sum_phi

    raise TypeError(f"ra_softmax expects torch.Tensor or numpy.ndarray, got {type(scores)}")


__all__ = ["ra_softmax"]
