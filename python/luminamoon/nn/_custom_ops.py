from typing import Any, Callable, Optional, Tuple, cast

from ._attention_gqa import (
    _gqa_matmul_ds_k,
    _gqa_matmul_dst_q,
    _gqa_matmul_pt_do,
    _gqa_matmul_pv,
    _gqa_matmul_qk,
)
from ._attention_ops import _ra_attn_scores_bwd, _ra_attn_weights_fwd
from ._compat import _HAS_TORCH, torch, triton
from ._loss_ops import _linear_ce_backward_impl, _linear_ce_forward_impl, _resolve_cuda_lp_dtype
from .kernels import triton_flash_ra_attention, triton_flash_ra_attention_backward

_HAS_CUSTOM_OP: bool = bool(
    _HAS_TORCH
    and torch is not None
    and hasattr(torch, "library")
    and hasattr(torch.library, "custom_op")
)


def _decode_lp_dtype(code: int) -> Optional[Any]:
    if code == 1:
        return torch.bfloat16
    if code == 2:
        return torch.float16
    return None


def _encode_lp_dtype(dt: Optional[Any]) -> int:
    if dt is torch.bfloat16:
        return 1
    if dt is torch.float16:
        return 2
    return 0


if _HAS_CUSTOM_OP:
    import torch as _torch

    def _custom_op(
        name: str, *, mutates_args: Tuple[str, ...] = ()
    ) -> Callable[[Callable[..., Any]], Any]:
        return cast(Any, _torch.library.custom_op(name, mutates_args=mutates_args))

    # 1. Flash-RaAttention (O(N) SRAM Triton kernel, supports optional native doc_ids packing)
    @_custom_op("luminamoon::ra_attention_fwd", mutates_args=())
    def _op_ra_attn_fwd(
        q: _torch.Tensor, k: _torch.Tensor, v: _torch.Tensor,
        is_causal: bool, sm_scale: float, ra_scale: float,
        doc_ids: Optional[_torch.Tensor] = None,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor, _torch.Tensor]:
        res = triton_flash_ra_attention(
            q, k, v, is_causal=is_causal, sm_scale=sm_scale, ra_scale=ra_scale, doc_ids=doc_ids
        )
        if res is not None:
            return res[0], res[1], res[2], res[3]
        orig_dt = q.dtype
        work_dt = _torch.float32 if orig_dt in (_torch.float16, _torch.bfloat16) else orig_dt
        q_s = (q.to(work_dt) * sm_scale).to(orig_dt) if work_dt != orig_dt else q * sm_scale
        scores = _gqa_matmul_qk(q_s, k).to(work_dt)
        if doc_ids is not None:
            same_doc = (doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)) & (doc_ids.unsqueeze(2) > 0)
            scores = scores.masked_fill(~same_doc.unsqueeze(1), float("-inf"))
        p, m, l, m_idx = _ra_attn_weights_fwd(scores, ra_scale, q_start=0, is_causal=is_causal, use_simd=False)
        out = _gqa_matmul_pv(p.to(orig_dt), v)
        return (
            out.contiguous(),
            m.squeeze(-1).to(_torch.float32).contiguous(),
            l.squeeze(-1).to(_torch.float32).contiguous(),
            m_idx.squeeze(-1).to(_torch.int32).contiguous(),
        )

    @_op_ra_attn_fwd.register_fake
    def _fake_ra_attn_fwd(
        q: _torch.Tensor, k: _torch.Tensor, v: _torch.Tensor,
        is_causal: bool, sm_scale: float, ra_scale: float,
        doc_ids: Optional[_torch.Tensor] = None,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor, _torch.Tensor]:
        B, H, N_Q, _ = q.shape
        return (
            q.new_empty(q.shape),
            q.new_empty((B, H, N_Q), dtype=_torch.float32),
            q.new_empty((B, H, N_Q), dtype=_torch.float32),
            q.new_empty((B, H, N_Q), dtype=_torch.int32),
        )

    @_custom_op("luminamoon::ra_attention_bwd", mutates_args=())
    def _op_ra_attn_bwd(
        grad_out: _torch.Tensor, q: _torch.Tensor, k: _torch.Tensor, v: _torch.Tensor,
        out: _torch.Tensor, m: _torch.Tensor, l: _torch.Tensor, m_idx: _torch.Tensor,
        is_causal: bool, sm_scale: float, ra_scale: float,
        doc_ids: Optional[_torch.Tensor] = None,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
        res = triton_flash_ra_attention_backward(
            grad_out, q, k, v, out, m, l, m_idx,
            is_causal=is_causal, sm_scale=sm_scale, ra_scale=ra_scale, doc_ids=doc_ids,
        )
        if res is not None:
            return res
        orig_dt = q.dtype
        work_dt = _torch.float32 if orig_dt in (_torch.float16, _torch.bfloat16) else orig_dt
        gemm_dt = orig_dt if (q.is_cuda and orig_dt in (_torch.float16, _torch.bfloat16)) else work_dt
        H_kv = k.shape[1]
        d_all = (grad_out.to(work_dt) * out.to(work_dt)).sum(dim=-1, keepdim=True)
        q_g, k_g, v_g, do_g = q.to(gemm_dt), k.to(gemm_dt), v.to(gemm_dt), grad_out.to(gemm_dt)
        q_s = (q.to(work_dt) * sm_scale).to(gemm_dt) if gemm_dt != work_dt else q_g * sm_scale
        scores = _gqa_matmul_qk(q_s, k_g).to(work_dt)
        if doc_ids is not None:
            same_doc = (doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)) & (doc_ids.unsqueeze(2) > 0)
            scores = scores.masked_fill(~same_doc.unsqueeze(1), float("-inf"))
        p_g, ds_d = _ra_attn_scores_bwd(
            scores, m.unsqueeze(-1).to(work_dt), l.unsqueeze(-1).to(work_dt),
            m_idx.unsqueeze(-1).to(_torch.int64),
            do_g, v_g, d_all, ra_scale, work_dt, gemm_dt, q_start=0, is_causal=is_causal, use_simd=False,
        )
        dv = _gqa_matmul_pt_do(p_g, do_g, H_kv).to(orig_dt).contiguous()
        ds_g = ds_d.to(gemm_dt) if ds_d.dtype != gemm_dt else ds_d
        dq = _gqa_matmul_ds_k(ds_g, k_g).mul_(sm_scale).to(orig_dt).contiguous()
        dk = _gqa_matmul_dst_q(ds_g, q_s, H_kv).to(orig_dt).contiguous()
        return dq, dk, dv

    @_op_ra_attn_bwd.register_fake
    def _fake_ra_attn_bwd(
        grad_out: _torch.Tensor, q: _torch.Tensor, k: _torch.Tensor, v: _torch.Tensor,
        out: _torch.Tensor, m: _torch.Tensor, l: _torch.Tensor, m_idx: _torch.Tensor,
        is_causal: bool, sm_scale: float, ra_scale: float,
        doc_ids: Optional[_torch.Tensor] = None,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
        return q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)

    def _setup_ra_attn_ctx(ctx: Any, inputs: Tuple[Any, ...], output: Tuple[Any, ...]) -> None:
        q, k, v, is_causal, sm_scale, ra_scale, doc_ids = inputs
        out, m, l, m_idx = output
        ctx.save_for_backward(q, k, v, out, m, l, m_idx, doc_ids)
        ctx.is_causal, ctx.sm_scale, ctx.ra_scale = bool(is_causal), float(sm_scale), float(ra_scale)

    def _bwd_ra_attn_ctx(
        ctx: Any, grad_out: _torch.Tensor, _gm: Any, _gl: Any, _gi: Any
    ) -> Tuple[Any, Any, Any, None, None, None, None]:
        q, k, v, out, m, l, m_idx, doc_ids = ctx.saved_tensors
        dq, dk, dv = _torch.ops.luminamoon.ra_attention_bwd(
            grad_out, q, k, v, out, m, l, m_idx, ctx.is_causal, ctx.sm_scale, ctx.ra_scale, doc_ids
        )
        return (
            dq if ctx.needs_input_grad[0] else None,
            dk if ctx.needs_input_grad[1] else None,
            dv if ctx.needs_input_grad[2] else None,
            None, None, None, None,
        )

    _op_ra_attn_fwd.register_autograd(_bwd_ra_attn_ctx, setup_context=_setup_ra_attn_ctx)

    # 2. Adaptive 400 MiB Chunked LinearCrossEntropy (AMP BF16/FP16 Tensor Core aware)
    @_custom_op("luminamoon::linear_ce_fwd", mutates_args=())
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
        lp_dtype_code: int,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
        c_opt = chunk_size if chunk_size > 0 else None
        return _linear_ce_forward_impl(
            x, weight, target, bias, ignore_index, reduction, label_smoothing, logit_scale,
            c_opt, forced_lp_dtype=_decode_lp_dtype(lp_dtype_code),
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
        lp_dtype_code: int,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
        out_dt = (
            x.dtype
            if x.dtype in (_torch.float16, _torch.bfloat16, _torch.float32, _torch.float64)
            else _torch.float32
        )
        M = x.numel() // max(1, x.shape[-1])
        loss_shape = target.shape if reduction == "none" else ()
        lse_dt = _torch.float64 if (x.dtype == _torch.float64 and weight.dtype == _torch.float64) else _torch.float32
        return (
            x.new_empty(loss_shape, dtype=out_dt),
            x.new_empty((M,), dtype=lse_dt),
            x.new_empty((), dtype=_torch.int64),
        )

    @_custom_op("luminamoon::linear_ce_bwd", mutates_args=())
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
        lp_dtype_code: int,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
        c_opt = chunk_size if chunk_size > 0 else None
        dx, dw, db = _linear_ce_backward_impl(
            grad_output, x.reshape(-1, x.shape[-1]), weight, bias, target.reshape(-1),
            lse_1d, valid_count, ignore_index, reduction, label_smoothing, logit_scale,
            c_opt, x.shape, need_dx, need_dw, need_db,
            forced_lp_dtype=_decode_lp_dtype(lp_dtype_code),
        )
        empty = x.new_empty(0)
        return (
            dx.contiguous() if dx is not None else empty,
            dw.contiguous() if dw is not None else empty,
            db.contiguous() if db is not None else empty,
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
        lp_dtype_code: int,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
        empty = x.new_empty(0)
        return (
            x.new_empty(x.shape) if need_dx else empty,
            weight.new_empty(weight.shape) if need_dw else empty,
            bias.new_empty(bias.shape) if (need_db and bias is not None) else empty,
        )

    def _setup_lce_ctx(ctx: Any, inputs: Tuple[Any, ...], output: Tuple[Any, Any, Any]) -> None:
        x, weight, target, bias, ignore_index, reduction, label_smoothing, logit_scale, chunk_size, lp_code = inputs
        _, lse_1d, valid_count = output
        ctx.save_for_backward(x, weight, target, bias, lse_1d, valid_count)
        ctx.has_bias = bias is not None
        ctx.ignore_index = int(ignore_index)
        ctx.reduction = str(reduction)
        ctx.label_smoothing = float(label_smoothing)
        ctx.logit_scale = float(logit_scale)
        ctx.chunk_size = int(chunk_size)
        ctx.lp_dtype_code = int(lp_code)

    def _bwd_lce_ctx(ctx: Any, grad_out: _torch.Tensor, _glse: Any, _gcnt: Any) -> Tuple[Any, ...]:
        x, weight, target, bias, lse_1d, valid_count = ctx.saved_tensors
        need_dx, need_dw = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
        need_db = bool(ctx.has_bias and ctx.needs_input_grad[3])
        dx, dw, db = _torch.ops.luminamoon.linear_ce_bwd(
            grad_out, x, weight, target, bias, lse_1d, valid_count,
            ctx.ignore_index, ctx.reduction, ctx.label_smoothing, ctx.logit_scale,
            ctx.chunk_size, need_dx, need_dw, need_db, ctx.lp_dtype_code,
        )
        return (
            dx if need_dx else None,
            dw if need_dw else None,
            None,
            db if need_db else None,
            None, None, None, None, None, None,
        )

    _op_lce_fwd.register_autograd(_bwd_lce_ctx, setup_context=_setup_lce_ctx)

    # 3. RaMoE Zero-Copy Grouped GEMM & Packed Activation
    @_custom_op("luminamoon::moe_grouped_gemm_fwd", mutates_args=())
    def _op_moe_gemm_fwd(
        x: _torch.Tensor, w: _torch.Tensor, offs: _torch.Tensor, gather_idx: Optional[_torch.Tensor]
    ) -> _torch.Tensor:
        from .kernels.moe_grouped_gemm import _launch_m_grouped_gemm

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

        gu_c = gu if gu.is_contiguous() else gu.contiguous()
        M, two_h = gu_c.shape
        H = two_h // 2
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

        gu_c = gu if gu.is_contiguous() else gu.contiguous()
        g_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
        M, two_h = gu_c.shape
        H = two_h // 2
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


def compiled_ra_attention(
    q: Any, k: Any, v: Any, is_causal: bool, sm_scale: float, ra_scale: float, doc_ids: Optional[Any] = None,
) -> Any:
    return torch.ops.luminamoon.ra_attention_fwd(
        q, k, v, bool(is_causal), float(sm_scale), float(ra_scale), doc_ids
    )[0]


def compiled_linear_ce(
    x: Any, weight: Any, target: Any, bias: Optional[Any],
    ignore_index: int, reduction: str, label_smoothing: float, logit_scale: float, chunk_size: Optional[int],
) -> Any:
    c_int = int(chunk_size) if (chunk_size is not None and int(chunk_size) > 0) else 0
    lp_code = _encode_lp_dtype(_resolve_cuda_lp_dtype(x.reshape(-1, x.shape[-1]), weight))
    return torch.ops.luminamoon.linear_ce_fwd(
        x, weight, target, bias, int(ignore_index), str(reduction),
        float(label_smoothing), float(logit_scale), c_int, lp_code,
    )[0]


def compiled_moe_grouped_gemm(x: Any, w: Any, offs: Any, gather_idx: Optional[Any] = None) -> Any:
    return torch.ops.luminamoon.moe_grouped_gemm_fwd(x, w, offs, gather_idx)


def compiled_ra_gated_packed(gu: Any, b: float, is_swiglu: bool) -> Any:
    return torch.ops.luminamoon.ra_gated_packed_fwd(gu, float(b), bool(is_swiglu))


__all__ = [
    "_HAS_CUSTOM_OP",
    "compiled_ra_attention",
    "compiled_linear_ce",
    "compiled_moe_grouped_gemm",
    "compiled_ra_gated_packed",
]
