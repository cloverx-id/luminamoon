"""PyTorch custom_op registration for RaMoE grouped GEMM and packed activations."""

from typing import Any, Callable, Optional, Tuple, cast

from ._compat import _HAS_TORCH, torch, triton

_HAS_CUSTOM_OP: bool = bool(
    _HAS_TORCH
    and torch is not None
    and hasattr(torch, "library")
    and hasattr(torch.library, "custom_op")
)

if _HAS_CUSTOM_OP:
    import torch as _torch

    def _custom_op(
        name: str, *, mutates_args: Tuple[str, ...] = ()
    ) -> Callable[[Callable[..., Any]], Any]:
        return cast(Any, _torch.library.custom_op(name, mutates_args=mutates_args))

    def _validate_moe_gemm(
        x: _torch.Tensor, w: _torch.Tensor, offs: _torch.Tensor, gather_idx: Optional[_torch.Tensor],
        grad_out: Optional[_torch.Tensor] = None,
    ) -> None:
        if triton is None:
            raise RuntimeError("Triton is required for luminamoon::moe_grouped_gemm custom ops.")
        if not (x.is_cuda and w.device == x.device and offs.device == x.device):
            raise ValueError("moe_grouped_gemm requires x, w, and offs on the same CUDA device.")
        if x.ndim != 2 or w.ndim != 3 or offs.ndim != 1:
            raise ValueError(f"Expected x(2D), w(3D), offs(1D), got shapes {tuple(x.shape)}, {tuple(w.shape)}, {tuple(offs.shape)}.")
        if x.shape[1] != w.shape[1] or offs.shape[0] != w.shape[0]:
            raise ValueError(f"Dimension mismatch: x={tuple(x.shape)}, w={tuple(w.shape)}, offs={tuple(offs.shape)}.")
        if gather_idx is not None and (gather_idx.device != x.device or gather_idx.ndim != 1):
            raise ValueError("gather_idx must be a 1D tensor on the same CUDA device as x.")
        if grad_out is not None:
            m_exp = gather_idx.shape[0] if gather_idx is not None else x.shape[0]
            if grad_out.device != x.device or tuple(grad_out.shape) != (m_exp, w.shape[2]):
                raise ValueError(f"grad_out must reside on {x.device} with shape {(m_exp, w.shape[2])}, got {tuple(grad_out.shape)}.")

    def _validate_ra_packed(
        gu: _torch.Tensor, b: float, grad_out: Optional[_torch.Tensor] = None
    ) -> Tuple[int, int]:
        if triton is None:
            raise RuntimeError("Triton is required for luminamoon::ra_gated_packed custom ops.")
        if not gu.is_cuda or gu.ndim != 2 or gu.shape[1] % 2 != 0:
            raise ValueError(f"ra_gated_packed expects 2D CUDA tensor with even width, got {tuple(gu.shape)} on {gu.device}.")
        if not (float(b) > 0.0):
            raise ValueError(f"ra_gated_packed requires b > 0, got {b}.")
        M, two_h = gu.shape
        H = two_h // 2
        if grad_out is not None and (grad_out.device != gu.device or tuple(grad_out.shape) != (M, H)):
            raise ValueError(f"grad_out must reside on {gu.device} with shape {(M, H)}, got {tuple(grad_out.shape)}.")
        return M, H

    @_custom_op("luminamoon::moe_grouped_gemm_fwd", mutates_args=())
    def _op_moe_gemm_fwd(
        x: _torch.Tensor, w: _torch.Tensor, offs: _torch.Tensor, gather_idx: Optional[_torch.Tensor]
    ) -> _torch.Tensor:
        from .kernels.moe_grouped_gemm import _launch_m_grouped_gemm

        _validate_moe_gemm(x, w, offs, gather_idx)
        x_c = x if x.is_contiguous() else x.contiguous()
        w_c = w if w.is_contiguous() else w.contiguous()
        offs_c = offs if (offs.dtype == _torch.int32 and offs.is_contiguous()) else offs.to(_torch.int32).contiguous()
        g_c = gather_idx.contiguous() if gather_idx is not None else None
        return _launch_m_grouped_gemm(x_c, w_c, offs_c, gather_idx=g_c)

    @_op_moe_gemm_fwd.register_fake
    def _fake_moe_gemm_fwd(
        x: _torch.Tensor, w: _torch.Tensor, offs: _torch.Tensor, gather_idx: Optional[_torch.Tensor]
    ) -> _torch.Tensor:
        m_out = gather_idx.shape[0] if gather_idx is not None else x.shape[0]
        return x.new_empty((m_out, w.shape[2]))

    @_custom_op("luminamoon::moe_grouped_gemm_bwd", mutates_args=())
    def _op_moe_gemm_bwd(
        grad_out: _torch.Tensor, x: _torch.Tensor, w: _torch.Tensor,
        offs: _torch.Tensor, gather_idx: Optional[_torch.Tensor], need_dx: bool, need_dw: bool,
    ) -> Tuple[_torch.Tensor, _torch.Tensor]:
        from .kernels.moe_grouped_gemm import _launch_k_grouped_dw, _launch_m_grouped_gemm

        _validate_moe_gemm(x, w, offs, gather_idx, grad_out=grad_out)
        x_c = x if x.is_contiguous() else x.contiguous()
        w_c = w if w.is_contiguous() else w.contiguous()
        offs_c = offs if (offs.dtype == _torch.int32 and offs.is_contiguous()) else offs.to(_torch.int32).contiguous()
        g_c = gather_idx.contiguous() if gather_idx is not None else None
        dy_in = grad_out if grad_out.dtype == w_c.dtype else grad_out.to(w_c.dtype)
        dy_c = dy_in if dy_in.is_contiguous() else dy_in.contiguous()
        dx = x.new_empty(0)
        if need_dx:
            dx_perm = _launch_m_grouped_gemm(dy_c, w_c.transpose(-1, -2), offs_c)
            if g_c is not None:
                dx = _torch.zeros_like(x_c)
                dx.index_add_(0, g_c, dx_perm)
            else:
                dx = dx_perm
        dw = _launch_k_grouped_dw(x_c, dy_c, offs_c, w_c.dtype, gather_idx=g_c) if need_dw else w.new_empty(0)
        return dx, dw

    @_op_moe_gemm_bwd.register_fake
    def _fake_moe_gemm_bwd(
        grad_out: _torch.Tensor, x: _torch.Tensor, w: _torch.Tensor,
        offs: _torch.Tensor, gather_idx: Optional[_torch.Tensor], need_dx: bool, need_dw: bool,
    ) -> Tuple[_torch.Tensor, _torch.Tensor]:
        return (x.new_empty(x.shape) if need_dx else x.new_empty(0)), (w.new_empty(w.shape) if need_dw else w.new_empty(0))

    def _setup_moe_gemm_ctx(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        x, w, offs, gather_idx = inputs
        ctx.save_for_backward(x, w, offs, gather_idx)

    def _bwd_moe_gemm_ctx(ctx: Any, grad_out: _torch.Tensor) -> Tuple[Any, Any, None, None]:
        x, w, offs, gather_idx = ctx.saved_tensors
        need_dx, need_dw = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
        dx, dw = _torch.ops.luminamoon.moe_grouped_gemm_bwd(grad_out, x, w, offs, gather_idx, need_dx, need_dw)
        return (dx if need_dx else None, dw if need_dw else None, None, None)

    _op_moe_gemm_fwd.register_autograd(_bwd_moe_gemm_ctx, setup_context=_setup_moe_gemm_ctx)

    @_custom_op("luminamoon::ra_gated_packed_fwd", mutates_args=())
    def _op_ra_packed_fwd(gu: _torch.Tensor, b: float, is_swiglu: bool) -> _torch.Tensor:
        from .kernels.moe_grouped_gemm import _ra_gated_packed_fwd_kernel

        M, H = _validate_ra_packed(gu, b)
        gu_c = gu if gu.is_contiguous() else gu.contiguous()
        out = _torch.empty((M, H), device=gu_c.device, dtype=gu_c.dtype)
        if M > 0 and H > 0:
            BLOCK_H = min(1024, max(64, triton.next_power_of_2(H)))
            _ra_gated_packed_fwd_kernel[(M, triton.cdiv(H, BLOCK_H))](
                gu_c, out, M, H, gu_c.stride(0), out.stride(0), b_val=float(b), IS_SWIGLU=is_swiglu,
                BLOCK_H=BLOCK_H, enable_fp_fusion=True,
            )
        return out

    @_op_ra_packed_fwd.register_fake
    def _fake_ra_packed_fwd(gu: _torch.Tensor, b: float, is_swiglu: bool) -> _torch.Tensor:
        return gu.new_empty((gu.shape[0], gu.shape[1] // 2))

    @_custom_op("luminamoon::ra_gated_packed_bwd", mutates_args=())
    def _op_ra_packed_bwd(grad_out: _torch.Tensor, gu: _torch.Tensor, b: float, is_swiglu: bool) -> _torch.Tensor:
        from .kernels.moe_grouped_gemm import _ra_gated_packed_bwd_kernel

        M, H = _validate_ra_packed(gu, b, grad_out=grad_out)
        gu_c = gu if gu.is_contiguous() else gu.contiguous()
        g_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
        dgu = _torch.empty_like(gu_c)
        if M > 0 and H > 0:
            BLOCK_H = min(1024, max(64, triton.next_power_of_2(H)))
            _ra_gated_packed_bwd_kernel[(M, triton.cdiv(H, BLOCK_H))](
                g_c, gu_c, dgu, M, H, g_c.stride(0), gu_c.stride(0), dgu.stride(0),
                b_val=float(b), IS_SWIGLU=is_swiglu, BLOCK_H=BLOCK_H, enable_fp_fusion=True,
            )
        return dgu

    @_op_ra_packed_bwd.register_fake
    def _fake_ra_packed_bwd(grad_out: _torch.Tensor, gu: _torch.Tensor, b: float, is_swiglu: bool) -> _torch.Tensor:
        return gu.new_empty(gu.shape)

    def _setup_ra_packed_ctx(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        gu, b, is_swiglu = inputs
        ctx.save_for_backward(gu)
        ctx.b = float(b)
        ctx.is_swiglu = bool(is_swiglu)

    def _bwd_ra_packed_ctx(ctx: Any, grad_out: _torch.Tensor) -> Tuple[Any, None, None]:
        (gu,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None
        return _torch.ops.luminamoon.ra_gated_packed_bwd(grad_out, gu, ctx.b, ctx.is_swiglu), None, None

    _op_ra_packed_fwd.register_autograd(_bwd_ra_packed_ctx, setup_context=_setup_ra_packed_ctx)


def compiled_moe_grouped_gemm(x: Any, w: Any, offs: Any, gather_idx: Optional[Any] = None) -> Any:
    return torch.ops.luminamoon.moe_grouped_gemm_fwd(x, w, offs, gather_idx)


def compiled_ra_gated_packed(gu: Any, b: float, is_swiglu: bool) -> Any:
    return torch.ops.luminamoon.ra_gated_packed_fwd(gu, float(b), bool(is_swiglu))


__all__ = ["_HAS_CUSTOM_OP", "compiled_moe_grouped_gemm", "compiled_ra_gated_packed"]
