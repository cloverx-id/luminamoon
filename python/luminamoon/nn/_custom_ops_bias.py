"""PyTorch custom_op registration for SRAM-fused DistanceDecayBias Flash-RaAttention."""

from typing import Any, Callable, Optional, Tuple, cast

from ._attention_gqa import (
    _gqa_matmul_ds_k,
    _gqa_matmul_dst_q,
    _gqa_matmul_pt_do,
    _gqa_matmul_pv,
    _gqa_matmul_qk,
)
from ._attention_ops import _ra_attn_scores_bwd, _ra_attn_weights_fwd
from ._compat import _HAS_TORCH, torch
from .kernels import triton_flash_ra_attention, triton_flash_ra_attention_backward

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

    def _build_bias_and_causal_mask_torch(
        n_q: int,
        n_k: int,
        device: Any,
        bias_slopes: Optional[_torch.Tensor],
        hub_penalty: Optional[_torch.Tensor],
        bias_mode: str,
        hub_size: int,
        q_offset: int,
        check_hub_q: bool,
        is_causal: bool,
    ) -> Optional[_torch.Tensor]:
        q_idx = _torch.arange(n_q, device=device, dtype=_torch.int64).unsqueeze(1)
        k_idx = _torch.arange(n_k, device=device, dtype=_torch.int64).unsqueeze(0)
        mask: Optional[_torch.Tensor] = None
        if bias_slopes is not None:
            raw_dist = (q_idx + int(q_offset) - k_idx).clamp(min=0).to(_torch.float32).view(1, 1, n_q, n_k)
            s_h = bias_slopes.reshape(1, -1, 1, 1).to(device=device, dtype=_torch.float32)
            if bias_mode == "rational":
                text_bias = -_torch.log((1.0 + s_h * raw_dist).clamp(min=1e-8))
            else:
                text_bias = -s_h * raw_dist
            if hub_size > 0 and hub_penalty is not None:
                h_pen = hub_penalty.reshape(1, -1, 1, 1).to(device=device, dtype=_torch.float32)
                is_hub_k = (k_idx < hub_size).view(1, 1, 1, n_k)
                mask = _torch.where(is_hub_k, -h_pen, text_bias).clamp(min=-10.0, max=0.0)
            else:
                mask = text_bias.clamp(min=-10.0, max=0.0)
        if is_causal:
            if hub_size > 0:
                is_hub_k = k_idx < hub_size
                if check_hub_q:
                    allowed = is_hub_k | ((q_idx >= hub_size) & (k_idx <= (q_idx + int(q_offset))))
                else:
                    allowed = is_hub_k | (k_idx <= (q_idx + int(q_offset)))
            else:
                allowed = k_idx <= (q_idx + int(q_offset))
            causal_add = _torch.where(
                allowed.view(1, 1, n_q, n_k),
                _torch.zeros((), device=device, dtype=_torch.float32),
                _torch.full((), float("-inf"), device=device, dtype=_torch.float32),
            )
            mask = causal_add if mask is None else (mask + causal_add)
        return mask

    @_custom_op("luminamoon::ra_attention_bias_fwd", mutates_args=())
    def _op_ra_attn_bias_fwd(
        q: _torch.Tensor, k: _torch.Tensor, v: _torch.Tensor,
        bias_slopes: Optional[_torch.Tensor], hub_penalty: Optional[_torch.Tensor],
        bias_mode: str, hub_size: int, q_offset: int, check_hub_q: bool,
        is_causal: bool, sm_scale: float, ra_scale: float,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor, _torch.Tensor]:
        mode_opt = bias_mode if bias_mode else None
        res = triton_flash_ra_attention(
            q, k, v, is_causal=is_causal, sm_scale=sm_scale, ra_scale=ra_scale,
            bias_slopes=bias_slopes, hub_penalty=hub_penalty, bias_mode=mode_opt,
            hub_size=hub_size, q_offset=q_offset, check_hub_q=check_hub_q,
        )
        if res is not None:
            return res[0], res[1], res[2], res[3]
        orig_dt = q.dtype
        work_dt = _torch.float32 if orig_dt in (_torch.float16, _torch.bfloat16) else orig_dt
        q_s = (q.to(work_dt) * sm_scale).to(orig_dt) if work_dt != orig_dt else q * sm_scale
        scores = _gqa_matmul_qk(q_s, k).to(work_dt)
        add_mask = _build_bias_and_causal_mask_torch(
            q.shape[2], k.shape[2], q.device, bias_slopes, hub_penalty, bias_mode,
            hub_size, q_offset, check_hub_q, is_causal,
        )
        if add_mask is not None:
            scores = scores + add_mask.to(work_dt)
        p, m, l, m_idx = _ra_attn_weights_fwd(scores, ra_scale, q_start=0, is_causal=False, use_simd=False)
        out = _gqa_matmul_pv(p.to(orig_dt), v)
        return (
            out.contiguous(),
            m.squeeze(-1).to(_torch.float32).contiguous(),
            l.squeeze(-1).to(_torch.float32).contiguous(),
            m_idx.squeeze(-1).to(_torch.int32).contiguous(),
        )

    @_op_ra_attn_bias_fwd.register_fake
    def _fake_ra_attn_bias_fwd(
        q: _torch.Tensor, k: _torch.Tensor, v: _torch.Tensor,
        bias_slopes: Optional[_torch.Tensor], hub_penalty: Optional[_torch.Tensor],
        bias_mode: str, hub_size: int, q_offset: int, check_hub_q: bool,
        is_causal: bool, sm_scale: float, ra_scale: float,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor, _torch.Tensor]:
        B, H, N_Q, _ = q.shape
        return (
            q.new_empty(q.shape),
            q.new_empty((B, H, N_Q), dtype=_torch.float32),
            q.new_empty((B, H, N_Q), dtype=_torch.float32),
            q.new_empty((B, H, N_Q), dtype=_torch.int32),
        )

    @_custom_op("luminamoon::ra_attention_bias_bwd", mutates_args=())
    def _op_ra_attn_bias_bwd(
        grad_out: _torch.Tensor, q: _torch.Tensor, k: _torch.Tensor, v: _torch.Tensor,
        out: _torch.Tensor, m: _torch.Tensor, l: _torch.Tensor, m_idx: _torch.Tensor,
        bias_slopes: Optional[_torch.Tensor], hub_penalty: Optional[_torch.Tensor],
        bias_mode: str, hub_size: int, q_offset: int, check_hub_q: bool,
        is_causal: bool, sm_scale: float, ra_scale: float,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
        mode_opt = bias_mode if bias_mode else None
        res = triton_flash_ra_attention_backward(
            grad_out, q, k, v, out, m, l, m_idx, is_causal=is_causal,
            sm_scale=sm_scale, ra_scale=ra_scale, bias_slopes=bias_slopes,
            hub_penalty=hub_penalty, bias_mode=mode_opt, hub_size=hub_size,
            q_offset=q_offset, check_hub_q=check_hub_q,
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
        add_mask = _build_bias_and_causal_mask_torch(
            q.shape[2], k.shape[2], q.device, bias_slopes, hub_penalty, bias_mode,
            hub_size, q_offset, check_hub_q, is_causal,
        )
        if add_mask is not None:
            scores = scores + add_mask.to(work_dt)
        p_g, ds_d = _ra_attn_scores_bwd(
            scores, m.unsqueeze(-1).to(work_dt), l.unsqueeze(-1).to(work_dt),
            m_idx.unsqueeze(-1).to(_torch.int64),
            do_g, v_g, d_all, ra_scale, work_dt, gemm_dt, q_start=0, is_causal=False, use_simd=False,
        )
        dv = _gqa_matmul_pt_do(p_g, do_g, H_kv).to(orig_dt).contiguous()
        ds_g = ds_d.to(gemm_dt) if ds_d.dtype != gemm_dt else ds_d
        dq = _gqa_matmul_ds_k(ds_g, k_g).mul_(sm_scale).to(orig_dt).contiguous()
        dk = _gqa_matmul_dst_q(ds_g, q_s, H_kv).to(orig_dt).contiguous()
        return dq, dk, dv

    @_op_ra_attn_bias_bwd.register_fake
    def _fake_ra_attn_bias_bwd(
        grad_out: _torch.Tensor, q: _torch.Tensor, k: _torch.Tensor, v: _torch.Tensor,
        out: _torch.Tensor, m: _torch.Tensor, l: _torch.Tensor, m_idx: _torch.Tensor,
        bias_slopes: Optional[_torch.Tensor], hub_penalty: Optional[_torch.Tensor],
        bias_mode: str, hub_size: int, q_offset: int, check_hub_q: bool,
        is_causal: bool, sm_scale: float, ra_scale: float,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
        return q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)

    def _setup_ra_attn_bias_ctx(ctx: Any, inputs: Tuple[Any, ...], output: Tuple[Any, ...]) -> None:
        (
            q, k, v, bias_slopes, hub_penalty, bias_mode,
            hub_size, q_offset, check_hub_q, is_causal, sm_scale, ra_scale,
        ) = inputs
        out, m, l, m_idx = output
        ctx.save_for_backward(q, k, v, out, m, l, m_idx, bias_slopes, hub_penalty)
        ctx.bias_mode, ctx.hub_size, ctx.q_offset = str(bias_mode), int(hub_size), int(q_offset)
        ctx.check_hub_q, ctx.is_causal = bool(check_hub_q), bool(is_causal)
        ctx.sm_scale, ctx.ra_scale = float(sm_scale), float(ra_scale)

    def _bwd_ra_attn_bias_ctx(
        ctx: Any, grad_out: _torch.Tensor, _gm: Any, _gl: Any, _gi: Any
    ) -> Tuple[Any, ...]:
        q, k, v, out, m, l, m_idx, bias_slopes, hub_penalty = ctx.saved_tensors
        dq, dk, dv = _torch.ops.luminamoon.ra_attention_bias_bwd(
            grad_out, q, k, v, out, m, l, m_idx, bias_slopes, hub_penalty,
            ctx.bias_mode, ctx.hub_size, ctx.q_offset, ctx.check_hub_q,
            ctx.is_causal, ctx.sm_scale, ctx.ra_scale,
        )
        return (
            dq if ctx.needs_input_grad[0] else None,
            dk if ctx.needs_input_grad[1] else None,
            dv if ctx.needs_input_grad[2] else None,
            None, None, None, None, None, None, None, None, None,
        )

    _op_ra_attn_bias_fwd.register_autograd(_bwd_ra_attn_bias_ctx, setup_context=_setup_ra_attn_bias_ctx)


def compiled_ra_attention_bias(
    q: Any,
    k: Any,
    v: Any,
    *,
    bias_slopes: Optional[Any],
    hub_penalty: Optional[Any],
    bias_mode: Optional[str],
    hub_size: int,
    q_offset: int,
    check_hub_q: bool,
    is_causal: bool,
    sm_scale: float,
    ra_scale: float,
) -> Any:
    return torch.ops.luminamoon.ra_attention_bias_fwd(
        q, k, v, bias_slopes, hub_penalty, str(bias_mode or ""),
        int(hub_size), int(q_offset), bool(check_hub_q), bool(is_causal),
        float(sm_scale), float(ra_scale),
    )[0]


__all__ = ["_HAS_CUSTOM_OP", "compiled_ra_attention_bias"]
