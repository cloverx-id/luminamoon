from __future__ import annotations

import math
from typing import Any, Optional

from ._compat import _HAS_TORCH, _HAS_TRITON, _is_torch_compiling, torch
from .kernels import (
    can_use_triton_cce,
    triton_cce_bwd,
    triton_cce_fwd,
    triton_fused_ce_chunk_bwd_inplace,
    triton_fused_ce_chunk_fwd,
)

_DEFAULT_CUDA_CHUNK_SIZE: int = 2048
_DEFAULT_CPU_CHUNK_SIZE: int = 512


def _resolve_chunk_size(M: int, chunk_size: Optional[int], is_cuda: bool = False) -> int:
    if chunk_size is not None and int(chunk_size) > 0:
        return max(1, min(int(M), int(chunk_size)))
    default_c = _DEFAULT_CUDA_CHUNK_SIZE if is_cuda else _DEFAULT_CPU_CHUNK_SIZE
    return max(1, min(int(M), default_c))


def _validate_ce_args(
    reduction: str,
    label_smoothing: float,
    logit_scale: float,
) -> str:
    red = reduction.strip().lower()
    if red not in ("mean", "sum", "none"):
        raise ValueError(
            f"Invalid reduction '{reduction}'. Expected one of: 'mean', 'sum', 'none'."
        )
    if not (0.0 <= float(label_smoothing) < 1.0) or math.isnan(label_smoothing):
        raise ValueError(f"label_smoothing must be in [0.0, 1.0), got {label_smoothing}.")
    if not math.isfinite(logit_scale) or float(logit_scale) <= 0.0:
        raise ValueError(f"logit_scale must be a positive finite float, got {logit_scale}.")
    return red


_FusedLinearCrossEntropyFunction: Any = None

if _HAS_TORCH and torch is not None:

    class _FusedLinearCrossEntropyFunctionImpl(torch.autograd.Function):
        """Chunked / Triton SRAM-fused Linear + Cross-Entropy autograd function."""

        @staticmethod
        def forward(
            ctx: Any,
            x: Any,
            weight: Any,
            target: Any,
            bias: Optional[Any],
            ignore_index: int = -100,
            reduction: str = "mean",
            label_smoothing: float = 0.0,
            logit_scale: float = 1.0,
            chunk_size: Optional[int] = None,
        ) -> Any:
            orig_target_shape = target.shape
            D = x.shape[-1]
            V = weight.shape[0]
            x_2d = x.reshape(-1, D)
            target_1d = target.reshape(-1)
            M = x_2d.shape[0]
            out_dtype = (
                x.dtype
                if x.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
                else torch.float32
            )

            if M == 0:
                empty_lse = torch.empty((0,), device=x.device, dtype=torch.float32)
                valid_count = torch.zeros((), device=x.device, dtype=torch.int64)
                ctx.save_for_backward(
                    x_2d, weight,
                    bias if bias is not None else torch.empty(0, device=x.device),
                    target_1d, empty_lse, valid_count,
                )
                ctx.has_bias = bias is not None
                ctx.ignore_index = int(ignore_index)
                ctx.reduction = reduction
                ctx.label_smoothing = float(label_smoothing)
                ctx.logit_scale = float(logit_scale)
                ctx.chunk_size = chunk_size
                ctx.x_shape = x.shape
                ctx.use_cce = False
                if reduction == "none":
                    return torch.empty(orig_target_shape, device=x.device, dtype=out_dtype)
                return torch.zeros((), device=x.device, dtype=out_dtype)

            use_cce = (not _is_torch_compiling()) and can_use_triton_cce(x_2d, weight, chunk_size)
            if use_cce:
                losses_1d, lse_1d = triton_cce_fwd(
                    x_2d,
                    weight,
                    target_1d,
                    bias=bias,
                    ignore_index=ignore_index,
                    label_smoothing=label_smoothing,
                    logit_scale=logit_scale,
                )
            else:
                eff_chunk = _resolve_chunk_size(M, chunk_size, is_cuda=bool(x_2d.is_cuda))
                losses_1d = torch.empty((M,), device=x.device, dtype=torch.float32)
                lse_1d = torch.empty((M,), device=x.device, dtype=torch.float32)

                use_native_lp_gemm = (
                    bool(x_2d.is_cuda)
                    and x_2d.dtype == weight.dtype
                    and x_2d.dtype in (torch.float16, torch.bfloat16)
                )
                use_triton_ce = bool(_HAS_TRITON and x_2d.is_cuda)
                w_mm = weight if use_native_lp_gemm else weight.to(torch.float32)
                b_f32 = bias.to(torch.float32) if bias is not None else None
                # Keep z_buf in BF16/FP16 when Triton handles FP32 upcast, bias, and logit_scale in registers.
                z_buf_dtype = x_2d.dtype if (use_native_lp_gemm and use_triton_ce) else torch.float32
                z_buf = torch.empty((min(M, eff_chunk), V), device=x.device, dtype=z_buf_dtype)

                for start in range(0, M, eff_chunk):
                    end = min(start + eff_chunk, M)
                    c = end - start
                    z_chunk = z_buf[:c]
                    if use_native_lp_gemm and z_buf_dtype == x_2d.dtype:
                        torch.mm(x_2d[start:end], w_mm.t(), out=z_chunk)
                    elif use_native_lp_gemm:
                        z_chunk.copy_(torch.mm(x_2d[start:end], w_mm.t()))
                    else:
                        torch.mm(x_2d[start:end].to(torch.float32), w_mm.t(), out=z_chunk)

                    t_c = target_1d[start:end]
                    if use_triton_ce and triton_fused_ce_chunk_fwd(
                        z_chunk, t_c, losses_1d[start:end], lse_1d[start:end],
                        ignore_index=ignore_index, label_smoothing=label_smoothing,
                        bias=b_f32, logit_scale=logit_scale,
                    ):
                        continue

                    z_f32 = z_chunk.to(torch.float32) if z_chunk.dtype != torch.float32 else z_chunk
                    if b_f32 is not None:
                        z_f32.add_(b_f32)
                    if logit_scale != 1.0:
                        z_f32.mul_(logit_scale)

                    valid_c = (t_c != ignore_index) & (t_c >= 0) & (t_c < V)
                    safe_t = torch.where(valid_c, t_c, 0).to(torch.long)

                    lse_c = torch.where(valid_c, torch.logsumexp(z_f32, dim=-1), 0.0)
                    z_target = z_f32.gather(dim=-1, index=safe_t.unsqueeze(-1)).squeeze(-1)

                    if label_smoothing > 0.0:
                        s_c = z_f32.sum(dim=-1)
                        loss_c = lse_c - (1.0 - label_smoothing) * z_target - (label_smoothing / V) * s_c
                    else:
                        loss_c = lse_c - z_target

                    losses_1d[start:end] = torch.where(valid_c, loss_c, 0.0)
                    lse_1d[start:end] = lse_c

            valid_mask_all = (target_1d != ignore_index) & (target_1d >= 0) & (target_1d < V)
            valid_count = valid_mask_all.sum()

            ctx.save_for_backward(
                x_2d, weight,
                bias if bias is not None else torch.empty(0, device=x.device),
                target_1d, lse_1d, valid_count,
            )
            ctx.has_bias = bias is not None
            ctx.ignore_index = int(ignore_index)
            ctx.reduction = reduction
            ctx.label_smoothing = float(label_smoothing)
            ctx.logit_scale = float(logit_scale)
            ctx.chunk_size = chunk_size
            ctx.x_shape = x.shape
            ctx.use_cce = use_cce

            if reduction == "none":
                return losses_1d.view(orig_target_shape).to(out_dtype)
            if reduction == "sum":
                return losses_1d.sum().to(out_dtype)

            denom = valid_count.clamp(min=1).to(torch.float32)
            mean_loss = torch.where(
                valid_count > 0,
                losses_1d.sum() / denom,
                torch.zeros((), device=x.device, dtype=torch.float32),
            )
            return mean_loss.to(out_dtype)

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> Any:  # type: ignore[override]
            x_2d, weight, bias_saved, target_1d, lse_1d, valid_count = ctx.saved_tensors
            has_bias: bool = ctx.has_bias
            bias = bias_saved if has_bias else None
            ignore_index: int = ctx.ignore_index
            reduction: str = ctx.reduction
            label_smoothing: float = ctx.label_smoothing
            logit_scale: float = ctx.logit_scale
            chunk_size: Optional[int] = ctx.chunk_size
            x_shape = ctx.x_shape

            M, D = x_2d.shape
            V = weight.shape[0]
            need_dx = bool(ctx.needs_input_grad[0])
            need_dw = bool(ctx.needs_input_grad[1])
            need_db = bool(has_bias and ctx.needs_input_grad[3])

            if M == 0 or not (need_dx or need_dw or need_db):
                dx_empty = torch.zeros(x_shape, device=x_2d.device, dtype=x_2d.dtype) if need_dx else None
                dw_empty = torch.zeros_like(weight) if need_dw else None
                db_empty = torch.zeros_like(bias) if (need_db and bias is not None) else None
                return dx_empty, dw_empty, None, db_empty, None, None, None, None, None

            valid_mask_all = (target_1d != ignore_index) & (target_1d >= 0) & (target_1d < V)
            if reduction == "none":
                grad_loss_1d = torch.where(valid_mask_all, grad_output.reshape(-1).to(torch.float32), 0.0)
            elif reduction == "sum":
                g_scalar = grad_output.reshape(()).to(torch.float32)
                grad_loss_1d = torch.where(valid_mask_all, g_scalar, 0.0)
            else:
                g_scalar = grad_output.reshape(()).to(torch.float32)
                denom = valid_count.clamp(min=1).to(torch.float32)
                scale_val = torch.where(
                    valid_count > 0,
                    g_scalar / denom,
                    torch.zeros((), device=x_2d.device, dtype=torch.float32),
                )
                grad_loss_1d = torch.where(valid_mask_all, scale_val, 0.0)

            if getattr(ctx, "use_cce", False):
                dx_2d, dw_out, db_out = triton_cce_bwd(
                    x_2d,
                    weight,
                    target_1d,
                    lse_1d,
                    grad_loss_1d,
                    bias=bias,
                    ignore_index=ignore_index,
                    label_smoothing=label_smoothing,
                    logit_scale=logit_scale,
                    need_dx=need_dx,
                    need_dw=need_dw,
                    need_db=need_db,
                )
                dx_out = dx_2d.reshape(x_shape) if dx_2d is not None else None
                return dx_out, dw_out, None, db_out, None, None, None, None, None

            eff_chunk = _resolve_chunk_size(M, chunk_size, is_cuda=bool(x_2d.is_cuda))
            use_native_lp_gemm = (
                bool(x_2d.is_cuda)
                and x_2d.dtype == weight.dtype
                and x_2d.dtype in (torch.float16, torch.bfloat16)
            )
            use_triton_ce = bool(_HAS_TRITON and x_2d.is_cuda)
            w_mm = weight if use_native_lp_gemm else weight.to(torch.float32)
            b_f32 = bias.to(torch.float32) if bias is not None else None

            gemm_dtype = x_2d.dtype if use_native_lp_gemm else torch.float32
            z_buf_dtype = gemm_dtype if (use_native_lp_gemm and use_triton_ce) else torch.float32
            dx_buf = torch.empty((M, D), device=x_2d.device, dtype=gemm_dtype) if need_dx else None
            dw_buf = torch.zeros((V, D), device=weight.device, dtype=gemm_dtype) if need_dw else None
            db_f32 = torch.zeros((V,), device=weight.device, dtype=torch.float32) if need_db else None
            z_buf = torch.empty((min(M, eff_chunk), V), device=x_2d.device, dtype=z_buf_dtype)

            for start in range(0, M, eff_chunk):
                end = min(start + eff_chunk, M)
                c = end - start
                z_chunk = z_buf[:c]
                x_c = x_2d[start:end] if use_native_lp_gemm else x_2d[start:end].to(torch.float32)

                if use_native_lp_gemm and z_buf_dtype == gemm_dtype:
                    torch.mm(x_c, w_mm.t(), out=z_chunk)
                elif use_native_lp_gemm:
                    z_chunk.copy_(torch.mm(x_c, w_mm.t()))
                else:
                    torch.mm(x_c, w_mm.t(), out=z_chunk)

                t_c = target_1d[start:end]
                used_triton_bwd = False
                if use_triton_ce:
                    used_triton_bwd = triton_fused_ce_chunk_bwd_inplace(
                        z_chunk, t_c, lse_1d[start:end], grad_loss_1d[start:end],
                        ignore_index=ignore_index, label_smoothing=label_smoothing,
                        logit_scale=logit_scale, bias=b_f32,
                    )

                if used_triton_bwd:
                    if db_f32 is not None:
                        db_f32.add_(z_chunk.sum(dim=0, dtype=torch.float32))
                    z_mm = z_chunk if z_chunk.dtype == gemm_dtype else z_chunk.to(gemm_dtype)
                else:
                    z_f32 = z_chunk.to(torch.float32) if z_chunk.dtype != torch.float32 else z_chunk
                    if b_f32 is not None:
                        z_f32.add_(b_f32)
                    if logit_scale != 1.0:
                        z_f32.mul_(logit_scale)
                    lse_c = lse_1d[start:end].unsqueeze(-1)
                    valid_c = (t_c != ignore_index) & (t_c >= 0) & (t_c < V)
                    safe_t = torch.where(valid_c, t_c, 0).to(torch.long).unsqueeze(-1)
                    z_f32.sub_(lse_c)
                    z_f32.masked_fill_(~valid_c.unsqueeze(-1), 0.0)
                    z_f32.exp_()
                    if label_smoothing > 0.0:
                        z_f32.sub_(label_smoothing / V)
                    z_f32.scatter_add_(
                        dim=-1,
                        index=safe_t,
                        src=torch.full(
                            (c, 1), -(1.0 - label_smoothing), device=z_f32.device, dtype=torch.float32
                        ),
                    )
                    z_f32.mul_((grad_loss_1d[start:end] * logit_scale).unsqueeze(-1))
                    if db_f32 is not None:
                        db_f32.add_(z_f32.sum(dim=0))
                    z_mm = z_f32.to(gemm_dtype) if use_native_lp_gemm else z_f32

                if dx_buf is not None:
                    torch.mm(z_mm, w_mm, out=dx_buf[start:end])
                if dw_buf is not None:
                    dw_buf.addmm_(z_mm.t(), x_c)

            dx_out = dx_buf.reshape(x_shape).to(x_2d.dtype) if dx_buf is not None else None
            dw_out = dw_buf.to(weight.dtype) if dw_buf is not None else None
            db_out = db_f32.to(bias.dtype) if (db_f32 is not None and bias is not None) else None
            return dx_out, dw_out, None, db_out, None, None, None, None, None

    _FusedLinearCrossEntropyFunction = _FusedLinearCrossEntropyFunctionImpl
