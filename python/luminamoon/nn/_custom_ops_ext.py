from typing import Any, Optional, Tuple

from ._compat import _HAS_TORCH, torch, triton
from ._loss_ops import _linear_ce_backward_impl, _linear_ce_forward_impl

_HAS_CUSTOM_OP_EXT: bool = bool(
    _HAS_TORCH
    and torch is not None
    and hasattr(torch, "library")
    and hasattr(torch.library, "custom_op")
)

if _HAS_CUSTOM_OP_EXT:
    import torch as _torch

    # 1. Adaptive Chunked LinearCrossEntropy
    @_torch.library.custom_op("luminamoon::linear_ce_fwd", mutates_args=())
    def _op_lce_fwd(
        x: _torch.Tensor,
        weight: _torch.Tensor,
        target: _torch.Tensor,
        bias: Optional[_torch.Tensor],
        ignore_index: int,
        reduction: str,
        label_smoothing: float,
        logit_scale: float,
        chunk_size: int,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
        c_opt = chunk_size if chunk_size > 0 else None
        return _linear_ce_forward_impl(
            x, weight, target, bias, ignore_index, reduction, label_smoothing, logit_scale, c_opt
        )

    @_op_lce_fwd.register_fake
    def _fake_lce_fwd(
        x: _torch.Tensor,
        weight: _torch.Tensor,
        target: _torch.Tensor,
        bias: Optional[_torch.Tensor],
        ignore_index: int,
        reduction: str,
        label_smoothing: float,
        logit_scale: float,
        chunk_size: int,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
        out_dt = (
            x.dtype
            if x.dtype in (_torch.float16, _torch.bfloat16, _torch.float32, _torch.float64)
            else _torch.float32
        )
        M = x.numel() // max(1, x.shape[-1])
        loss_shape = target.shape if reduction == "none" else ()
        return (
            x.new_empty(loss_shape, dtype=out_dt),
            x.new_empty((M,), dtype=_torch.float32),
            x.new_empty((), dtype=_torch.int64),
        )

    @_torch.library.custom_op("luminamoon::linear_ce_bwd", mutates_args=())
    def _op_lce_bwd(
        grad_output: _torch.Tensor,
        x: _torch.Tensor,
        weight: _torch.Tensor,
        target: _torch.Tensor,
        bias: Optional[_torch.Tensor],
        lse_1d: _torch.Tensor,
        valid_count: _torch.Tensor,
        ignore_index: int,
        reduction: str,
        label_smoothing: float,
        logit_scale: float,
        chunk_size: int,
        need_dx: bool,
        need_dw: bool,
        need_db: bool,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
        c_opt = chunk_size if chunk_size > 0 else None
        dx, dw, db = _linear_ce_backward_impl(
            grad_output, x.reshape(-1, x.shape[-1]), weight, bias, target.reshape(-1),
            lse_1d, valid_count, ignore_index, reduction, label_smoothing, logit_scale,
            c_opt, x.shape, need_dx, need_dw, need_db,
        )
        empty = x.new_empty(0)
        return (
            dx if dx is not None else empty,
            dw if dw is not None else empty,
            db if db is not None else empty,
        )

    @_op_lce_bwd.register_fake
    def _fake_lce_bwd(
        grad_output: _torch.Tensor,
        x: _torch.Tensor,
        weight: _torch.Tensor,
        target: _torch.Tensor,
        bias: Optional[_torch.Tensor],
        lse_1d: _torch.Tensor,
        valid_count: _torch.Tensor,
        ignore_index: int,
        reduction: str,
        label_smoothing: float,
        logit_scale: float,
        chunk_size: int,
        need_dx: bool,
        need_dw: bool,
        need_db: bool,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
        empty = x.new_empty(0)
        return (
            _torch.empty_like(x) if need_dx else empty,
            _torch.empty_like(weight) if need_dw else empty,
            _torch.empty_like(bias) if (need_db and bias is not None) else empty,
        )

    def _setup_lce_ctx(ctx: Any, inputs: Tuple[Any, ...], output: Tuple[Any, Any, Any]) -> None:
        x, weight, target, bias, ignore_index, reduction, label_smoothing, logit_scale, chunk_size = inputs
        _, lse_1d, valid_count = output
        ctx.save_for_backward(x, weight, target, bias, lse_1d, valid_count)
        ctx.has_bias = bias is not None
        ctx.ignore_index = int(ignore_index)
        ctx.reduction = str(reduction)
        ctx.label_smoothing = float(label_smoothing)
        ctx.logit_scale = float(logit_scale)
        ctx.chunk_size = int(chunk_size)

    def _bwd_lce_ctx(ctx: Any, grad_out: _torch.Tensor, _glse: Any, _gcnt: Any) -> Tuple[Any, ...]:
        x, weight, target, bias, lse_1d, valid_count = ctx.saved_tensors
        need_dx, need_dw = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
        need_db = bool(ctx.has_bias and ctx.needs_input_grad[3])
        dx, dw, db = _torch.ops.luminamoon.linear_ce_bwd(
            grad_out, x, weight, target, bias, lse_1d, valid_count,
            ctx.ignore_index, ctx.reduction, ctx.label_smoothing, ctx.logit_scale,
            ctx.chunk_size, need_dx, need_dw, need_db,
        )
        return (
            dx if need_dx else None,
            dw if need_dw else None,
            None,
            db if need_db else None,
            None, None, None, None, None,
        )

    _op_lce_fwd.register_autograd(_bwd_lce_ctx, setup_context=_setup_lce_ctx)

    # 2. RaMoE Grouped GEMM & Packed Activation
    @_torch.library.custom_op("luminamoon::moe_grouped_gemm_fwd", mutates_args=())
    def _op_moe_gemm_fwd(
        x: _torch.Tensor,
        w: _torch.Tensor,
        offs: _torch.Tensor,
        gather_idx: Optional[_torch.Tensor],
    ) -> _torch.Tensor:
        from .kernels.moe_grouped_gemm import _launch_m_grouped_gemm

        x_c = x if x.is_contiguous() else x.contiguous()
        w_c = w if w.is_contiguous() else w.contiguous()
        offs_c = offs if (offs.dtype == _torch.int32 and offs.is_contiguous()) else offs.to(_torch.int32).contiguous()
        g_c = gather_idx.contiguous() if gather_idx is not None else None
        return _launch_m_grouped_gemm(x_c, w_c, offs_c, gather_idx=g_c)

    @_op_moe_gemm_fwd.register_fake
    def _fake_moe_gemm_fwd(
        x: _torch.Tensor,
        w: _torch.Tensor,
        offs: _torch.Tensor,
        gather_idx: Optional[_torch.Tensor],
    ) -> _torch.Tensor:
        m_out = gather_idx.shape[0] if gather_idx is not None else x.shape[0]
        return x.new_empty((m_out, w.shape[2]))

    @_torch.library.custom_op("luminamoon::moe_grouped_gemm_bwd", mutates_args=())
    def _op_moe_gemm_bwd(
        grad_out: _torch.Tensor,
        x: _torch.Tensor,
        w: _torch.Tensor,
        offs: _torch.Tensor,
        gather_idx: Optional[_torch.Tensor],
        need_dx: bool,
        need_dw: bool,
    ) -> Tuple[_torch.Tensor, _torch.Tensor]:
        from .kernels.moe_grouped_gemm import _launch_k_grouped_dw, _launch_m_grouped_gemm

        x_c = x if x.is_contiguous() else x.contiguous()
        w_c = w if w.is_contiguous() else w.contiguous()
        offs_c = offs if (offs.dtype == _torch.int32 and offs.is_contiguous()) else offs.to(_torch.int32).contiguous()
        g_c = gather_idx.contiguous() if gather_idx is not None else None
        dy_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
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
        grad_out: _torch.Tensor,
        x: _torch.Tensor,
        w: _torch.Tensor,
        offs: _torch.Tensor,
        gather_idx: Optional[_torch.Tensor],
        need_dx: bool,
        need_dw: bool,
    ) -> Tuple[_torch.Tensor, _torch.Tensor]:
        return (
            _torch.empty_like(x) if need_dx else x.new_empty(0),
            _torch.empty_like(w) if need_dw else w.new_empty(0),
        )

    def _setup_moe_gemm_ctx(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        x, w, offs, gather_idx = inputs
        ctx.save_for_backward(x, w, offs, gather_idx)

    def _bwd_moe_gemm_ctx(ctx: Any, grad_out: _torch.Tensor) -> Tuple[Any, Any, None, None]:
        x, w, offs, gather_idx = ctx.saved_tensors
        need_dx, need_dw = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
        dx, dw = _torch.ops.luminamoon.moe_grouped_gemm_bwd(
            grad_out, x, w, offs, gather_idx, need_dx, need_dw
        )
        return (dx if need_dx else None, dw if need_dw else None, None, None)

    _op_moe_gemm_fwd.register_autograd(_bwd_moe_gemm_ctx, setup_context=_setup_moe_gemm_ctx)

    @_torch.library.custom_op("luminamoon::ra_gated_packed_fwd", mutates_args=())
    def _op_ra_packed_fwd(gu: _torch.Tensor, b: float, is_swiglu: bool) -> _torch.Tensor:
        from .kernels.moe_grouped_gemm import _ra_gated_packed_fwd_kernel

        gu_c = gu if gu.is_contiguous() else gu.contiguous()
        M, two_h = gu_c.shape
        H = two_h // 2
        out = _torch.empty((M, H), device=gu_c.device, dtype=gu_c.dtype)
        if M > 0 and H > 0:
            BLOCK_H = min(1024, max(64, triton.next_power_of_2(H)))
            _ra_gated_packed_fwd_kernel[(M, triton.cdiv(H, BLOCK_H))](
                gu_c, out, M, H, gu_c.stride(0), out.stride(0),
                b_val=float(b), IS_SWIGLU=is_swiglu, BLOCK_H=BLOCK_H,
            )
        return out

    @_op_ra_packed_fwd.register_fake
    def _fake_ra_packed_fwd(gu: _torch.Tensor, b: float, is_swiglu: bool) -> _torch.Tensor:
        return gu.new_empty((gu.shape[0], gu.shape[1] // 2))

    @_torch.library.custom_op("luminamoon::ra_gated_packed_bwd", mutates_args=())
    def _op_ra_packed_bwd(
        grad_out: _torch.Tensor, gu: _torch.Tensor, b: float, is_swiglu: bool
    ) -> _torch.Tensor:
        from .kernels.moe_grouped_gemm import _ra_gated_packed_bwd_kernel

        gu_c = gu if gu.is_contiguous() else gu.contiguous()
        g_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
        M, two_h = gu_c.shape
        H = two_h // 2
        dgu = _torch.empty_like(gu_c)
        if M > 0 and H > 0:
            BLOCK_H = min(1024, max(64, triton.next_power_of_2(H)))
            _ra_gated_packed_bwd_kernel[(M, triton.cdiv(H, BLOCK_H))](
                g_c, gu_c, dgu, M, H, g_c.stride(0), gu_c.stride(0), dgu.stride(0),
                b_val=float(b), IS_SWIGLU=is_swiglu, BLOCK_H=BLOCK_H,
            )
        return dgu

    @_op_ra_packed_bwd.register_fake
    def _fake_ra_packed_bwd(
        grad_out: _torch.Tensor, gu: _torch.Tensor, b: float, is_swiglu: bool
    ) -> _torch.Tensor:
        return _torch.empty_like(gu)

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


def compiled_linear_ce(
    x: Any,
    weight: Any,
    target: Any,
    bias: Optional[Any],
    ignore_index: int,
    reduction: str,
    label_smoothing: float,
    logit_scale: float,
    chunk_size: Optional[int],
) -> Any:
    c_int = int(chunk_size) if (chunk_size is not None and int(chunk_size) > 0) else 0
    return torch.ops.luminamoon.linear_ce_fwd(
        x, weight, target, bias, int(ignore_index), str(reduction),
        float(label_smoothing), float(logit_scale), c_int,
    )[0]


def compiled_moe_grouped_gemm(x: Any, w: Any, offs: Any, gather_idx: Optional[Any] = None) -> Any:
    return torch.ops.luminamoon.moe_grouped_gemm_fwd(x, w, offs, gather_idx)


def compiled_ra_gated_packed(gu: Any, b: float, is_swiglu: bool) -> Any:
    return torch.ops.luminamoon.ra_gated_packed_fwd(gu, float(b), bool(is_swiglu))


__all__ = [
    "compiled_linear_ce",
    "compiled_moe_grouped_gemm",
    "compiled_ra_gated_packed",
]
