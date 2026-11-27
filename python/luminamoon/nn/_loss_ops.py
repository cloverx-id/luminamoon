from __future__ import annotations

import math
from typing import Any, Optional

from ._compat import _HAS_TORCH, _HAS_TRITON, torch
from .kernels import triton_fused_ce_chunk_bwd_inplace, triton_fused_ce_chunk_fwd

_CUDA_TARGET_Z_BYTES: int = 400 * 1024 * 1024
_CUDA_MIN_CHUNK_SIZE: int = 512
_CUDA_MAX_CHUNK_SIZE: int = 4096
_DEFAULT_CUDA_CHUNK_SIZE: int = 2048
_DEFAULT_CPU_CHUNK_SIZE: int = 512


def _resolve_chunk_size(
    M: int,
    chunk_size: Optional[int],
    is_cuda: bool = False,
    vocab_size: Optional[int] = None,
    elem_bytes: int = 2,
) -> int:
    if chunk_size is not None and int(chunk_size) > 0:
        return max(1, min(int(M), int(chunk_size)))
    if not is_cuda:
        return max(1, min(int(M), _DEFAULT_CPU_CHUNK_SIZE))
    if vocab_size is not None and int(vocab_size) > 0:
        raw_c = _CUDA_TARGET_Z_BYTES // max(1, int(vocab_size) * max(1, int(elem_bytes)))
        aligned_c = (raw_c // 256) * 256
        default_c = max(_CUDA_MIN_CHUNK_SIZE, min(_CUDA_MAX_CHUNK_SIZE, aligned_c))
    else:
        default_c = _DEFAULT_CUDA_CHUNK_SIZE
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


def _resolve_cuda_lp_dtype(
    x_2d: Any, weight: Any, forced_lp_dtype: Optional[Any] = None
) -> Optional[Any]:
    if not x_2d.is_cuda:
        return None
    if forced_lp_dtype in (torch.float16, torch.bfloat16):
        return forced_lp_dtype
    if x_2d.dtype in (torch.float16, torch.bfloat16) and weight.dtype in (
        torch.float16, torch.bfloat16, torch.float32
    ):
        return x_2d.dtype
    try:
        amp_on = torch.is_autocast_enabled("cuda")
    except TypeError:
        amp_on = torch.is_autocast_enabled()
    if amp_on:
        amp_dt = (
            torch.get_autocast_dtype("cuda")
            if hasattr(torch, "get_autocast_dtype")
            else torch.get_autocast_gpu_dtype()
        )
        if (
            amp_dt in (torch.float16, torch.bfloat16)
            and x_2d.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
        ):
            return amp_dt
    return None


def _linear_ce_forward_impl(
    x: Any,
    weight: Any,
    target: Any,
    bias: Optional[Any],
    ignore_index: int,
    reduction: str,
    label_smoothing: float,
    logit_scale: float,
    chunk_size: Optional[int],
    forced_lp_dtype: Optional[Any] = None,
) -> tuple[Any, Any, Any]:
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
        loss = (
            torch.empty(orig_target_shape, device=x.device, dtype=out_dtype)
            if reduction == "none"
            else torch.zeros((), device=x.device, dtype=out_dtype)
        )
        return loss, empty_lse, valid_count

    lp_dtype = _resolve_cuda_lp_dtype(x_2d, weight, forced_lp_dtype=forced_lp_dtype)
    use_native_lp_gemm = lp_dtype is not None
    work_dtype = (
        torch.float64
        if (not use_native_lp_gemm and x_2d.dtype == torch.float64 and weight.dtype == torch.float64)
        else torch.float32
    )
    gemm_dtype = lp_dtype if lp_dtype is not None else work_dtype
    use_triton_ce = bool(_HAS_TRITON and x_2d.is_cuda and work_dtype == torch.float32)
    w_mm = weight if weight.dtype == gemm_dtype else weight.to(gemm_dtype)
    b_work = bias.to(work_dtype) if bias is not None else None
    # Keep z_buf in BF16/FP16 when Triton handles FP32 upcast, bias, and logit_scale in registers.
    z_buf_dtype = gemm_dtype if (use_native_lp_gemm and use_triton_ce) else work_dtype
    eff_chunk = _resolve_chunk_size(
        M, chunk_size, is_cuda=bool(x_2d.is_cuda), vocab_size=V, elem_bytes=z_buf_dtype.itemsize
    )
    losses_1d = torch.empty((M,), device=x.device, dtype=work_dtype)
    lse_1d = torch.empty((M,), device=x.device, dtype=work_dtype)
    z_buf = torch.empty((min(M, eff_chunk), V), device=x.device, dtype=z_buf_dtype)

    for start in range(0, M, eff_chunk):
        end = min(start + eff_chunk, M)
        c = end - start
        z_chunk = z_buf[:c]
        x_c = x_2d[start:end] if x_2d.dtype == gemm_dtype else x_2d[start:end].to(gemm_dtype)
        if z_buf_dtype == gemm_dtype:
            torch.mm(x_c, w_mm.t(), out=z_chunk)
        else:
            z_chunk.copy_(torch.mm(x_c, w_mm.t()))

        t_c = target_1d[start:end]
        if use_triton_ce and triton_fused_ce_chunk_fwd(
            z_chunk, t_c, losses_1d[start:end], lse_1d[start:end],
            ignore_index=ignore_index, label_smoothing=label_smoothing,
            bias=b_work, logit_scale=logit_scale,
        ):
            continue

        z_work = z_chunk.to(work_dtype) if z_chunk.dtype != work_dtype else z_chunk
        if b_work is not None:
            z_work.add_(b_work)
        if logit_scale != 1.0:
            z_work.mul_(logit_scale)

        valid_c = (t_c != ignore_index) & (t_c >= 0) & (t_c < V)
        safe_t = torch.where(valid_c, t_c, 0).to(torch.long)

        lse_c = torch.where(valid_c, torch.logsumexp(z_work, dim=-1), 0.0)
        z_target = z_work.gather(dim=-1, index=safe_t.unsqueeze(-1)).squeeze(-1)

        if label_smoothing > 0.0:
            s_c = z_work.sum(dim=-1)
            loss_c = lse_c - (1.0 - label_smoothing) * z_target - (label_smoothing / V) * s_c
        else:
            loss_c = lse_c - z_target

        losses_1d[start:end] = torch.where(valid_c, loss_c, 0.0)
        lse_1d[start:end] = lse_c

    valid_mask_all = (target_1d != ignore_index) & (target_1d >= 0) & (target_1d < V)
    valid_count = valid_mask_all.sum()

    if reduction == "none":
        loss_out = losses_1d.view(orig_target_shape).to(out_dtype)
    elif reduction == "sum":
        loss_out = losses_1d.sum().to(out_dtype)
    else:
        denom = valid_count.clamp(min=1).to(work_dtype)
        mean_loss = torch.where(
            valid_count > 0,
            losses_1d.sum() / denom,
            torch.zeros((), device=x.device, dtype=work_dtype),
        )
        loss_out = mean_loss.to(out_dtype)
    return loss_out, lse_1d, valid_count


def _linear_ce_backward_impl(
    grad_output: Any,
    x_2d: Any,
    weight: Any,
    bias: Optional[Any],
    target_1d: Any,
    lse_1d: Any,
    valid_count: Any,
    ignore_index: int,
    reduction: str,
    label_smoothing: float,
    logit_scale: float,
    chunk_size: Optional[int],
    x_shape: Any,
    need_dx: bool,
    need_dw: bool,
    need_db: bool,
    forced_lp_dtype: Optional[Any] = None,
) -> tuple[Optional[Any], Optional[Any], Optional[Any]]:
    M, D = x_2d.shape
    V = weight.shape[0]
    if M == 0 or not (need_dx or need_dw or need_db):
        dx_empty = torch.zeros(x_shape, device=x_2d.device, dtype=x_2d.dtype) if need_dx else None
        dw_empty = torch.zeros_like(weight) if need_dw else None
        db_empty = torch.zeros_like(bias) if (need_db and bias is not None) else None
        return dx_empty, dw_empty, db_empty

    lp_dtype = _resolve_cuda_lp_dtype(x_2d, weight, forced_lp_dtype=forced_lp_dtype)
    use_native_lp_gemm = lp_dtype is not None
    work_dtype = (
        torch.float64
        if (not use_native_lp_gemm and x_2d.dtype == torch.float64 and weight.dtype == torch.float64)
        else torch.float32
    )

    valid_mask_all = (target_1d != ignore_index) & (target_1d >= 0) & (target_1d < V)
    gemm_dtype = lp_dtype if lp_dtype is not None else work_dtype
    # Normalize FP16 in-place dZ peak scale to 64.0 (2^6) across all reduction modes so large-vocab
    # non-target gradients (p_j / M ~ 1e-9) stay in normal FP16 range without GradScaler overflow.
    use_fp16_prescale = gemm_dtype == torch.float16
    fp16_dz_scale = 64.0

    if reduction == "none":
        grad_loss_1d = torch.where(valid_mask_all, grad_output.reshape(-1).to(work_dtype), 0.0)
        ref_mag = grad_loss_1d.abs().max() if use_fp16_prescale else None
    elif reduction == "sum":
        base_scale = grad_output.reshape(()).to(work_dtype)
        grad_loss_1d = torch.where(valid_mask_all, base_scale, 0.0)
        ref_mag = base_scale.abs() if use_fp16_prescale else None
    else:
        g_scalar = grad_output.reshape(()).to(work_dtype)
        denom = valid_count.clamp(min=1).to(work_dtype)
        base_scale = torch.where(
            valid_count > 0,
            g_scalar / denom,
            torch.zeros((), device=x_2d.device, dtype=work_dtype),
        )
        grad_loss_1d = torch.where(valid_mask_all, base_scale, 0.0)
        ref_mag = base_scale.abs() if use_fp16_prescale else None

    post_dx_db_scale = None
    post_dw_scale = None
    if use_fp16_prescale and ref_mag is not None:
        safe_mag = torch.where(ref_mag > 0.0, ref_mag, torch.ones_like(ref_mag))
        prescale_factor = fp16_dz_scale / (safe_mag * logit_scale)
        grad_loss_1d = grad_loss_1d * prescale_factor
        post_dx_db_scale = (safe_mag * logit_scale) / fp16_dz_scale
        post_dw_scale = safe_mag * (float(M) * logit_scale)

    use_triton_ce = bool(_HAS_TRITON and x_2d.is_cuda and work_dtype == torch.float32)
    w_mm = weight if weight.dtype == gemm_dtype else weight.to(gemm_dtype)
    b_work = bias.to(work_dtype) if bias is not None else None

    z_buf_dtype = gemm_dtype if (use_native_lp_gemm and use_triton_ce) else work_dtype
    eff_chunk = _resolve_chunk_size(
        M, chunk_size, is_cuda=bool(x_2d.is_cuda), vocab_size=V, elem_bytes=z_buf_dtype.itemsize
    )
    dx_buf = torch.empty((M, D), device=x_2d.device, dtype=gemm_dtype) if need_dx else None
    dw_buf = torch.empty((V, D), device=weight.device, dtype=gemm_dtype) if need_dw else None
    db_work = torch.zeros((V,), device=weight.device, dtype=work_dtype) if need_db else None
    z_buf = torch.empty((min(M, eff_chunk), V), device=x_2d.device, dtype=z_buf_dtype)
    dw_gemm_alpha = (1.0 / (float(M) * fp16_dz_scale)) if use_fp16_prescale else 1.0

    for start in range(0, M, eff_chunk):
        end = min(start + eff_chunk, M)
        c = end - start
        z_chunk = z_buf[:c]
        x_c = x_2d[start:end] if x_2d.dtype == gemm_dtype else x_2d[start:end].to(gemm_dtype)

        if z_buf_dtype == gemm_dtype:
            torch.mm(x_c, w_mm.t(), out=z_chunk)
        else:
            z_chunk.copy_(torch.mm(x_c, w_mm.t()))

        t_c = target_1d[start:end]
        used_triton_bwd = False
        if use_triton_ce:
            used_triton_bwd = triton_fused_ce_chunk_bwd_inplace(
                z_chunk, t_c, lse_1d[start:end], grad_loss_1d[start:end],
                ignore_index=ignore_index, label_smoothing=label_smoothing,
                logit_scale=logit_scale, bias=b_work,
            )

        if used_triton_bwd:
            if db_work is not None:
                db_work.add_(z_chunk.sum(dim=0, dtype=work_dtype))
            z_mm = z_chunk if z_chunk.dtype == gemm_dtype else z_chunk.to(gemm_dtype)
        else:
            z_work = z_chunk.to(work_dtype) if z_chunk.dtype != work_dtype else z_chunk
            if b_work is not None:
                z_work.add_(b_work)
            if logit_scale != 1.0:
                z_work.mul_(logit_scale)
            lse_c = lse_1d[start:end].to(work_dtype).unsqueeze(-1)
            valid_c = (t_c != ignore_index) & (t_c >= 0) & (t_c < V)
            safe_t = torch.where(valid_c, t_c, 0).to(torch.long).unsqueeze(-1)
            z_work.sub_(lse_c)
            z_work.masked_fill_(~valid_c.unsqueeze(-1), 0.0)
            z_work.exp_()
            if label_smoothing > 0.0:
                z_work.sub_(label_smoothing / V)
            z_work.scatter_add_(
                dim=-1,
                index=safe_t,
                src=torch.full(
                    (c, 1), -(1.0 - label_smoothing), device=z_work.device, dtype=work_dtype
                ),
            )
            z_work.mul_((grad_loss_1d[start:end] * logit_scale).unsqueeze(-1))
            if db_work is not None:
                db_work.add_(z_work.sum(dim=0))
            z_mm = z_work.to(gemm_dtype) if use_native_lp_gemm else z_work

        if dx_buf is not None:
            torch.mm(z_mm, w_mm, out=dx_buf[start:end])
        if dw_buf is not None:
            if start == 0:
                torch.mm(z_mm.t(), x_c, out=dw_buf)
                if use_fp16_prescale:
                    dw_buf.mul_(dw_gemm_alpha)
            elif use_fp16_prescale:
                dw_buf.addmm_(z_mm.t(), x_c, beta=1.0, alpha=dw_gemm_alpha)
            else:
                dw_buf.addmm_(z_mm.t(), x_c)

    if use_fp16_prescale and post_dx_db_scale is not None and post_dw_scale is not None:
        dx_out = (
            (dx_buf.to(work_dtype) * post_dx_db_scale).reshape(x_shape).to(x_2d.dtype)
            if dx_buf is not None
            else None
        )
        dw_out = (
            (dw_buf.to(work_dtype) * post_dw_scale).to(weight.dtype)
            if dw_buf is not None
            else None
        )
        db_out = (
            (db_work * post_dx_db_scale).to(bias.dtype)
            if (db_work is not None and bias is not None)
            else None
        )
    else:
        dx_out = dx_buf.reshape(x_shape).to(x_2d.dtype) if dx_buf is not None else None
        dw_out = dw_buf.to(weight.dtype) if dw_buf is not None else None
        db_out = db_work.to(bias.dtype) if (db_work is not None and bias is not None) else None
    return dx_out, dw_out, db_out


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
            lp_dtype = _resolve_cuda_lp_dtype(x.reshape(-1, x.shape[-1]), weight)
            loss_out, lse_1d, valid_count = _linear_ce_forward_impl(
                x, weight, target, bias, int(ignore_index), reduction,
                float(label_smoothing), float(logit_scale), chunk_size,
                forced_lp_dtype=lp_dtype,
            )
            ctx.save_for_backward(
                x.reshape(-1, x.shape[-1]), weight,
                bias if bias is not None else torch.empty(0, device=x.device),
                target.reshape(-1), lse_1d, valid_count,
            )
            ctx.has_bias = bias is not None
            ctx.ignore_index = int(ignore_index)
            ctx.reduction = reduction
            ctx.label_smoothing = float(label_smoothing)
            ctx.logit_scale = float(logit_scale)
            ctx.chunk_size = chunk_size
            ctx.x_shape = x.shape
            ctx.lp_dtype = lp_dtype
            return loss_out

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> Any:  # type: ignore[override]
            x_2d, weight, bias_saved, target_1d, lse_1d, valid_count = ctx.saved_tensors
            bias = bias_saved if ctx.has_bias else None
            dx_out, dw_out, db_out = _linear_ce_backward_impl(
                grad_output, x_2d, weight, bias, target_1d, lse_1d, valid_count,
                ctx.ignore_index, ctx.reduction, ctx.label_smoothing, ctx.logit_scale,
                ctx.chunk_size, ctx.x_shape,
                bool(ctx.needs_input_grad[0]),
                bool(ctx.needs_input_grad[1]),
                bool(ctx.has_bias and ctx.needs_input_grad[3]),
                forced_lp_dtype=ctx.lp_dtype,
            )
            return dx_out, dw_out, None, db_out, None, None, None, None, None

    _FusedLinearCrossEntropyFunction = _FusedLinearCrossEntropyFunctionImpl
