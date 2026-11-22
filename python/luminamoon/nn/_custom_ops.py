from typing import Any, Optional, Tuple

from ._attention_gqa import (
    _gqa_matmul_ds_k,
    _gqa_matmul_dst_q,
    _gqa_matmul_pt_do,
    _gqa_matmul_pv,
    _gqa_matmul_qk,
)
from ._attention_ops import _ra_attn_scores_bwd, _ra_attn_weights_fwd
from ._compat import _HAS_TORCH, torch
from ._custom_ops_ext import (
    compiled_linear_ce,
    compiled_moe_grouped_gemm,
    compiled_ra_gated_packed,
)
from .kernels import (
    triton_flash_ra_attention,
    triton_flash_ra_attention_backward,
    triton_ra_glu,
    triton_ra_glu_backward,
    triton_ra_norm,
    triton_ra_norm_backward,
    triton_ra_swiglu,
    triton_ra_swiglu_backward,
    triton_ra_tanh,
    triton_ra_tanh_backward,
)

_HAS_CUSTOM_OP: bool = bool(
    _HAS_TORCH
    and torch is not None
    and hasattr(torch, "library")
    and hasattr(torch.library, "custom_op")
)

if _HAS_CUSTOM_OP:
    import torch as _torch

    # 1. RaNorm
    @_torch.library.custom_op("luminamoon::ra_norm_fwd", mutates_args=())
    def _op_ra_norm_fwd(
        x: _torch.Tensor,
        weight: Optional[_torch.Tensor],
        bias: Optional[_torch.Tensor],
        eps: float,
    ) -> Tuple[_torch.Tensor, _torch.Tensor]:
        res = triton_ra_norm(x, weight, bias, eps=eps, return_inv_denom=True)
        if res is not None:
            return res
        x_f32 = x.to(_torch.float32) if x.dtype in (_torch.float16, _torch.bfloat16) else x
        inv_d = (1.0 / (x_f32.abs().mean(dim=-1, keepdim=True) + eps)).to(_torch.float32)
        normed = x_f32 * inv_d
        if weight is not None:
            normed = normed * weight.to(normed.dtype)
        if bias is not None:
            normed = normed + bias.to(normed.dtype)
        out = normed.to(x.dtype).contiguous()
        rows = x.numel() // max(1, x.shape[-1]) if x.numel() > 0 else 0
        return out, inv_d.reshape(rows).contiguous()

    @_op_ra_norm_fwd.register_fake
    def _fake_ra_norm_fwd(
        x: _torch.Tensor,
        weight: Optional[_torch.Tensor],
        bias: Optional[_torch.Tensor],
        eps: float,
    ) -> Tuple[_torch.Tensor, _torch.Tensor]:
        rows = x.numel() // max(1, x.shape[-1]) if x.ndim >= 1 else 0
        return x.new_empty(x.shape), x.new_empty((rows,), dtype=_torch.float32)

    @_torch.library.custom_op("luminamoon::ra_norm_bwd", mutates_args=())
    def _op_ra_norm_bwd(
        grad_y: _torch.Tensor,
        x: _torch.Tensor,
        weight: Optional[_torch.Tensor],
        inv_denom: _torch.Tensor,
        has_bias: bool,
        eps: float,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
        res = triton_ra_norm_backward(
            grad_y, x, weight=weight, has_bias=has_bias, eps=eps, inv_denom=inv_denom
        )
        if res is not None:
            dx, dw, db = res
            empty = x.new_empty(0)
            return dx.reshape(x.shape), (dw if dw is not None else empty), (db if db is not None else empty)
        x_f32 = x.to(_torch.float32) if x.dtype in (_torch.float16, _torch.bfloat16) else x
        gy_f32 = grad_y.to(_torch.float32) if grad_y.dtype in (_torch.float16, _torch.bfloat16) else grad_y
        inv_d = inv_denom.reshape(x.shape[:-1] + (1,)).to(x_f32.dtype)
        x_hat = x_f32 * inv_d
        dy_w = gy_f32 * weight.to(x_f32.dtype) if weight is not None else gy_f32
        c = (dy_w * x_hat).mean(dim=-1, keepdim=True)
        dx = (inv_d * (dy_w - _torch.sign(x_f32) * c)).to(x.dtype).contiguous()
        reduce_dims = tuple(range(x.ndim - 1))
        dw = (gy_f32 * x_hat).sum(dim=reduce_dims).to(weight.dtype).contiguous() if weight is not None else x.new_empty(0)
        b_dt = weight.dtype if weight is not None else x.dtype
        db = gy_f32.sum(dim=reduce_dims).to(b_dt).contiguous() if has_bias else x.new_empty(0)
        return dx, dw, db

    @_op_ra_norm_bwd.register_fake
    def _fake_ra_norm_bwd(
        grad_y: _torch.Tensor,
        x: _torch.Tensor,
        weight: Optional[_torch.Tensor],
        inv_denom: _torch.Tensor,
        has_bias: bool,
        eps: float,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
        dw = weight.new_empty(weight.shape) if weight is not None else x.new_empty(0)
        b_dt = weight.dtype if weight is not None else x.dtype
        db = x.new_empty((x.shape[-1],), dtype=b_dt) if has_bias else x.new_empty(0)
        return x.new_empty(x.shape), dw, db

    def _setup_ra_norm_ctx(ctx: Any, inputs: Tuple[Any, ...], output: Tuple[Any, Any]) -> None:
        x, weight, bias, eps = inputs
        _, inv_denom = output
        ctx.save_for_backward(x, weight, inv_denom)
        ctx.has_weight = weight is not None
        ctx.has_bias = bias is not None
        ctx.eps = float(eps)

    def _bwd_ra_norm_ctx(ctx: Any, grad_out: _torch.Tensor, _grad_inv: _torch.Tensor) -> Tuple[Any, Any, Any, None]:
        x, weight, inv_denom = ctx.saved_tensors
        dx, dw, db = _torch.ops.luminamoon.ra_norm_bwd(
            grad_out, x, weight, inv_denom, ctx.has_bias, ctx.eps
        )
        return (
            dx if ctx.needs_input_grad[0] else None,
            dw if (ctx.has_weight and ctx.needs_input_grad[1]) else None,
            db if (ctx.has_bias and ctx.needs_input_grad[2]) else None,
            None,
        )

    _op_ra_norm_fwd.register_autograd(_bwd_ra_norm_ctx, setup_context=_setup_ra_norm_ctx)

    # 2. RaSwiGLU / RaGLU
    @_torch.library.custom_op("luminamoon::ra_gated_fwd", mutates_args=())
    def _op_ra_gated_fwd(
        gate: _torch.Tensor, up: _torch.Tensor, b: float, is_swiglu: bool
    ) -> _torch.Tensor:
        out = triton_ra_swiglu(gate, up, b=b) if is_swiglu else triton_ra_glu(gate, up, b=b)
        if out is not None:
            return out
        z = gate.to(_torch.float32) if gate.dtype in (_torch.float16, _torch.bfloat16) else gate
        u = up.to(_torch.float32) if up.dtype in (_torch.float16, _torch.bfloat16) else up
        abs_z = z.abs()
        if is_swiglu:
            q = (0.5 * b * b) / (b * (b + abs_z) + z.square())
            sigma = _torch.where(z >= 0.0, 1.0 - q, q)
            return ((z * sigma) * u).to(gate.dtype).contiguous()
        b_term = b * (b + abs_z)
        return (((b_term * z) / (b_term + z.square())) * u).to(gate.dtype).contiguous()

    @_op_ra_gated_fwd.register_fake
    def _fake_ra_gated_fwd(
        gate: _torch.Tensor, up: _torch.Tensor, b: float, is_swiglu: bool
    ) -> _torch.Tensor:
        return gate.new_empty(gate.shape)

    @_torch.library.custom_op("luminamoon::ra_gated_bwd", mutates_args=())
    def _op_ra_gated_bwd(
        grad_out: _torch.Tensor, gate: _torch.Tensor, up: _torch.Tensor, b: float, is_swiglu: bool
    ) -> Tuple[_torch.Tensor, _torch.Tensor]:
        res = (
            triton_ra_swiglu_backward(grad_out, gate, up, b=b)
            if is_swiglu
            else triton_ra_glu_backward(grad_out, gate, up, b=b)
        )
        if res is not None:
            return res[0].reshape(gate.shape), res[1].reshape(up.shape)
        z = gate.to(_torch.float32) if gate.dtype in (_torch.float16, _torch.bfloat16) else gate
        u = up.to(_torch.float32) if up.dtype in (_torch.float16, _torch.bfloat16) else up
        g = grad_out.to(_torch.float32) if grad_out.dtype in (_torch.float16, _torch.bfloat16) else grad_out
        abs_z = z.abs()
        z_sq = z.square()
        if is_swiglu:
            inv_d = 1.0 / (b * (b + abs_z) + z_sq)
            b2 = b * b
            q = (0.5 * b2) * inv_d
            is_pos = z >= 0.0
            sigma = _torch.where(is_pos, 1.0 - q, q)
            s = z * sigma
            ds_dz = _torch.where(is_pos, sigma + z * q * (b + 2.0 * abs_z) * inv_d, q * (b2 - z_sq) * inv_d)
            return (g * u * ds_dz).to(gate.dtype).contiguous(), (g * s).to(up.dtype).contiguous()
        b_term = b * (b + abs_z)
        inv_d = 1.0 / (b_term + z_sq)
        ra_t = (b_term * z) * inv_d
        deriv = (b * b * b) * (b + 2.0 * abs_z) * (inv_d * inv_d)
        return (g * u * deriv).to(gate.dtype).contiguous(), (g * ra_t).to(up.dtype).contiguous()

    @_op_ra_gated_bwd.register_fake
    def _fake_ra_gated_bwd(
        grad_out: _torch.Tensor, gate: _torch.Tensor, up: _torch.Tensor, b: float, is_swiglu: bool
    ) -> Tuple[_torch.Tensor, _torch.Tensor]:
        return gate.new_empty(gate.shape), up.new_empty(up.shape)

    def _setup_ra_gated_ctx(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        gate, up, b, is_swiglu = inputs
        ctx.save_for_backward(gate, up)
        ctx.b = float(b)
        ctx.is_swiglu = bool(is_swiglu)

    def _bwd_ra_gated_ctx(ctx: Any, grad_out: _torch.Tensor) -> Tuple[Any, Any, None, None]:
        gate, up = ctx.saved_tensors
        dg, du = _torch.ops.luminamoon.ra_gated_bwd(grad_out, gate, up, ctx.b, ctx.is_swiglu)
        return (dg if ctx.needs_input_grad[0] else None, du if ctx.needs_input_grad[1] else None, None, None)

    _op_ra_gated_fwd.register_autograd(_bwd_ra_gated_ctx, setup_context=_setup_ra_gated_ctx)

    # 3. RaTanh
    @_torch.library.custom_op("luminamoon::ra_tanh_fwd", mutates_args=())
    def _op_ra_tanh_fwd(x: _torch.Tensor, b: float) -> _torch.Tensor:
        out = triton_ra_tanh(x, b=b)
        if out is not None:
            return out
        b_term = b * (b + x.abs())
        return ((b_term * x) / (b_term + x.square())).contiguous()

    @_op_ra_tanh_fwd.register_fake
    def _fake_ra_tanh_fwd(x: _torch.Tensor, b: float) -> _torch.Tensor:
        return x.new_empty(x.shape)

    @_torch.library.custom_op("luminamoon::ra_tanh_bwd", mutates_args=())
    def _op_ra_tanh_bwd(grad_out: _torch.Tensor, x: _torch.Tensor, b: float) -> _torch.Tensor:
        dx = triton_ra_tanh_backward(grad_out, x, b=b)
        if dx is not None:
            return dx.reshape(x.shape)
        x_f32 = x.to(_torch.float32) if x.dtype in (_torch.float16, _torch.bfloat16) else x
        g_f32 = grad_out.to(_torch.float32) if grad_out.dtype in (_torch.float16, _torch.bfloat16) else grad_out
        abs_x = x_f32.abs()
        denom = b * (b + abs_x) + x_f32.square()
        return (g_f32 * ((b * b * b) * (b + 2.0 * abs_x)) / denom.square()).to(x.dtype).contiguous()

    @_op_ra_tanh_bwd.register_fake
    def _fake_ra_tanh_bwd(grad_out: _torch.Tensor, x: _torch.Tensor, b: float) -> _torch.Tensor:
        return x.new_empty(x.shape)

    def _setup_ra_tanh_ctx(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        x, b = inputs
        ctx.save_for_backward(x)
        ctx.b = float(b)

    def _bwd_ra_tanh_ctx(ctx: Any, grad_out: _torch.Tensor) -> Tuple[Any, None]:
        (x,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None
        return _torch.ops.luminamoon.ra_tanh_bwd(grad_out, x, ctx.b), None

    _op_ra_tanh_fwd.register_autograd(_bwd_ra_tanh_ctx, setup_context=_setup_ra_tanh_ctx)

    # 4. Flash-RaAttention (with Block-Sparsity Bitmask)
    @_torch.library.custom_op("luminamoon::ra_attention_fwd", mutates_args=())
    def _op_ra_attn_fwd(
        q: _torch.Tensor,
        k: _torch.Tensor,
        v: _torch.Tensor,
        is_causal: bool,
        sm_scale: float,
        ra_scale: float,
        need_block_mask: bool,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor, _torch.Tensor, _torch.Tensor]:
        res = triton_flash_ra_attention(
            q, k, v, is_causal=is_causal, sm_scale=sm_scale, ra_scale=ra_scale,
            return_block_mask=need_block_mask,
        )
        if res is not None:
            if need_block_mask:
                return res[0], res[1], res[2], res[3], res[4]
            return res[0], res[1], res[2], res[3], q.new_zeros((1,), dtype=_torch.uint8)
        orig_dt = q.dtype
        work_dt = _torch.float32 if orig_dt in (_torch.float16, _torch.bfloat16) else orig_dt
        q_s = (q.to(work_dt) * sm_scale).to(orig_dt) if work_dt != orig_dt else q * sm_scale
        scores = _gqa_matmul_qk(q_s, k).to(work_dt)
        p, m, l, m_idx = _ra_attn_weights_fwd(scores, ra_scale, q_start=0, is_causal=is_causal, use_simd=False)
        out = _gqa_matmul_pv(p.to(orig_dt), v)
        B, H, N_Q, _ = q.shape
        N_K = k.shape[2]
        am = (
            _torch.ones((B * H, (N_Q + 63) // 64, (N_K + 31) // 32), device=q.device, dtype=_torch.uint8)
            if need_block_mask
            else q.new_zeros((1,), dtype=_torch.uint8)
        )
        return (
            out.contiguous(),
            m.squeeze(-1).to(_torch.float32).contiguous(),
            l.squeeze(-1).to(_torch.float32).contiguous(),
            m_idx.squeeze(-1).to(_torch.int32).contiguous(),
            am,
        )

    @_op_ra_attn_fwd.register_fake
    def _fake_ra_attn_fwd(
        q: _torch.Tensor, k: _torch.Tensor, v: _torch.Tensor,
        is_causal: bool, sm_scale: float, ra_scale: float, need_block_mask: bool,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor, _torch.Tensor, _torch.Tensor]:
        B, H, N_Q, _ = q.shape
        N_K = k.shape[2]
        am_shape = (B * H, (N_Q + 63) // 64, (N_K + 31) // 32) if need_block_mask else (1,)
        return (
            q.new_empty(q.shape),
            q.new_empty((B, H, N_Q), dtype=_torch.float32),
            q.new_empty((B, H, N_Q), dtype=_torch.float32),
            q.new_empty((B, H, N_Q), dtype=_torch.int32),
            q.new_empty(am_shape, dtype=_torch.uint8),
        )

    @_torch.library.custom_op("luminamoon::ra_attention_bwd", mutates_args=())
    def _op_ra_attn_bwd(
        grad_out: _torch.Tensor, q: _torch.Tensor, k: _torch.Tensor, v: _torch.Tensor,
        out: _torch.Tensor, m: _torch.Tensor, l: _torch.Tensor, m_idx: _torch.Tensor,
        active_mask: _torch.Tensor, is_causal: bool, sm_scale: float, ra_scale: float, use_block_mask: bool,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
        res = triton_flash_ra_attention_backward(
            grad_out, q, k, v, out, m, l, m_idx,
            is_causal=is_causal, sm_scale=sm_scale, ra_scale=ra_scale,
            active_mask=active_mask if use_block_mask else None,
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
        p_g, ds_d = _ra_attn_scores_bwd(
            scores, m.unsqueeze(-1).to(work_dt), l.unsqueeze(-1).to(work_dt), m_idx.unsqueeze(-1).to(_torch.int64),
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
        active_mask: _torch.Tensor, is_causal: bool, sm_scale: float, ra_scale: float, use_block_mask: bool,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
        return q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)

    def _setup_ra_attn_ctx(ctx: Any, inputs: Tuple[Any, ...], output: Tuple[Any, ...]) -> None:
        q, k, v, is_causal, sm_scale, ra_scale, need_block_mask = inputs
        out, m, l, m_idx, active_mask = output
        ctx.save_for_backward(q, k, v, out, m, l, m_idx, active_mask)
        ctx.is_causal = bool(is_causal)
        ctx.sm_scale = float(sm_scale)
        ctx.ra_scale = float(ra_scale)
        ctx.use_block_mask = bool(need_block_mask)

    def _bwd_ra_attn_ctx(
        ctx: Any, grad_out: _torch.Tensor, _gm: Any, _gl: Any, _gi: Any, _ga: Any
    ) -> Tuple[Any, Any, Any, None, None, None, None]:
        q, k, v, out, m, l, m_idx, active_mask = ctx.saved_tensors
        dq, dk, dv = _torch.ops.luminamoon.ra_attention_bwd(
            grad_out, q, k, v, out, m, l, m_idx, active_mask,
            ctx.is_causal, ctx.sm_scale, ctx.ra_scale, ctx.use_block_mask,
        )
        return (
            dq if ctx.needs_input_grad[0] else None,
            dk if ctx.needs_input_grad[1] else None,
            dv if ctx.needs_input_grad[2] else None,
            None, None, None, None,
        )

    _op_ra_attn_fwd.register_autograd(_bwd_ra_attn_ctx, setup_context=_setup_ra_attn_ctx)


def compiled_ra_norm(x: Any, weight: Optional[Any], bias: Optional[Any], eps: float) -> Any:
    return torch.ops.luminamoon.ra_norm_fwd(x, weight, bias, float(eps))[0]


def compiled_ra_gated(gate: Any, up: Any, b: float, is_swiglu: bool) -> Any:
    return torch.ops.luminamoon.ra_gated_fwd(gate, up, float(b), bool(is_swiglu))


def compiled_ra_tanh(x: Any, b: float) -> Any:
    return torch.ops.luminamoon.ra_tanh_fwd(x, float(b))


def compiled_ra_attention(
    q: Any, k: Any, v: Any, is_causal: bool, sm_scale: float, ra_scale: float
) -> Any:
    need_mask = bool(torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad))
    return torch.ops.luminamoon.ra_attention_fwd(
        q, k, v, bool(is_causal), float(sm_scale), float(ra_scale), need_mask
    )[0]


__all__ = [
    "_HAS_CUSTOM_OP",
    "compiled_ra_norm",
    "compiled_ra_gated",
    "compiled_ra_tanh",
    "compiled_ra_attention",
    "compiled_linear_ce",
    "compiled_moe_grouped_gemm",
    "compiled_ra_gated_packed",
]
