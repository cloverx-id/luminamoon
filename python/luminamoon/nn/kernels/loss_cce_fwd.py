from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

from ._base import HAS_TRITON, _HAS_TORCH, tl, torch, triton

# Threshold where avoiding [C, V] HBM materialization beats chunked cuBLAS on L4/A100/H100.
_CCE_MIN_VOCAB: int = 16384
_CCE_MIN_DIM: int = 32
_CCE_SPLIT_V_TARGET_CTAS: int = 48

if TYPE_CHECKING:
    _cce_target_logit_kernel: Any = None
    _cce_lse_fwd_kernel: Any = None
elif HAS_TRITON:
    _CCE_FWD_CONFIGS = [
        triton.Config({"BLOCK_M": 64, "BLOCK_V": 128, "BLOCK_D": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_V": 128, "BLOCK_D": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_V": 64, "BLOCK_D": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_V": 64, "BLOCK_D": 64}, num_warps=4, num_stages=4),
    ]

    @triton.jit
    def _cce_target_logit_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        target_ptr,
        out_ptr,
        stride_xm,
        stride_xd,
        stride_wv,
        stride_wd,
        M,
        V,
        D,
        ignore_index,
        logit_scale,
        HAS_BIAS: tl.constexpr,
        HAS_LOGIT_SCALE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= M:
            return
        t = tl.load(target_ptr + row)
        valid = (t != ignore_index) & (t >= 0) & (t < V)
        if not valid:
            tl.store(out_ptr + row, 0.0)
            return

        acc = 0.0
        x_row = x_ptr + row.to(tl.int64) * stride_xm
        w_row = w_ptr + t.to(tl.int64) * stride_wv
        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            xv = tl.load(x_row + offs_d * stride_xd, mask=mask_d, other=0.0).to(tl.float32)
            wv = tl.load(w_row + offs_d * stride_wd, mask=mask_d, other=0.0).to(tl.float32)
            acc += tl.sum(xv * wv, axis=0)
        if HAS_BIAS:
            acc += tl.load(bias_ptr + t).to(tl.float32)
        if HAS_LOGIT_SCALE:
            acc *= logit_scale
        tl.store(out_ptr + row, acc)

    @triton.autotune(configs=_CCE_FWD_CONFIGS, key=["V", "D"])
    @triton.jit
    def _cce_lse_fwd_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        lse_ptr,
        stride_xm,
        stride_xd,
        stride_wv,
        stride_wd,
        stride_lse_s,
        M,
        V,
        D,
        v_chunk_size,
        logit_scale,
        HAS_BIAS: tl.constexpr,
        HAS_LOGIT_SCALE: tl.constexpr,
        EVEN_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_s = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        v_start = pid_s * v_chunk_size
        v_end = tl.minimum(v_start + v_chunk_size, V)

        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for v0 in range(v_start, v_end, BLOCK_V):
            offs_v = v0 + tl.arange(0, BLOCK_V)
            mask_v = offs_v < v_end
            acc = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
            for d0 in range(0, D, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D)
                mask_d = offs_d < D
                x_blk = tl.load(
                    x_ptr + offs_m[:, None].to(tl.int64) * stride_xm + offs_d[None, :] * stride_xd,
                    mask=mask_m[:, None] if EVEN_D else (mask_m[:, None] & mask_d[None, :]),
                    other=0.0,
                )
                w_blk = tl.load(
                    w_ptr + offs_v[:, None].to(tl.int64) * stride_wv + offs_d[None, :] * stride_wd,
                    mask=mask_v[:, None] if EVEN_D else (mask_v[:, None] & mask_d[None, :]),
                    other=0.0,
                )
                acc = tl.dot(x_blk, tl.trans(w_blk), acc)

            if HAS_BIAS:
                b_v = tl.load(bias_ptr + offs_v, mask=mask_v, other=0.0).to(tl.float32)
                acc = acc + b_v[None, :]
            if HAS_LOGIT_SCALE:
                acc = acc * logit_scale

            acc = tl.where(mask_v[None, :], acc, -float("inf"))
            m_tile = tl.max(acc, axis=1)
            m_new = tl.maximum(m_i, m_tile)
            alpha = tl.where(m_i > -float("inf"), tl.exp(m_i - m_new), 0.0)
            p_tile = tl.where(mask_v[None, :], tl.exp(acc - m_new[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(p_tile, axis=1)
            m_i = m_new

        lse_out = tl.where(l_i > 0.0, m_i + tl.log(l_i), -float("inf"))
        tl.store(lse_ptr + pid_s.to(tl.int64) * stride_lse_s + offs_m, lse_out, mask=mask_m)
else:
    _cce_target_logit_kernel = None
    _cce_lse_fwd_kernel = None


def can_use_triton_cce(
    x_2d: Any,
    weight: Any,
    chunk_size: Optional[int],
    *,
    min_vocab: int = _CCE_MIN_VOCAB,
) -> bool:
    """Returns True when SRAM-tiled Cut Cross-Entropy should replace chunked HBM matmuls."""
    if not (
        HAS_TRITON
        and _HAS_TORCH
        and chunk_size is None
        and isinstance(x_2d, torch.Tensor)
        and isinstance(weight, torch.Tensor)
        and x_2d.is_cuda
        and weight.is_cuda
        and x_2d.ndim == 2
        and weight.ndim == 2
        and x_2d.dtype == weight.dtype
        and x_2d.dtype in (torch.float16, torch.bfloat16)
    ):
        return False
    M, D = x_2d.shape
    V = weight.shape[0]
    return M > 0 and D >= _CCE_MIN_DIM and V >= int(min_vocab)


def triton_cce_fwd(
    x_2d: Any,
    weight: Any,
    target_1d: Any,
    bias: Optional[Any] = None,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
) -> Tuple[Any, Any]:
    """Computes exact per-token loss and LSE without materializing `[C, V]` logits in HBM."""
    M, D = x_2d.shape
    V = weight.shape[0]
    x_c = x_2d if x_2d.stride(1) == 1 else x_2d.contiguous()
    w_c = weight if weight.stride(1) == 1 else weight.contiguous()
    t_c = target_1d if target_1d.is_contiguous() else target_1d.contiguous()
    has_bias = bias is not None
    b_c = bias.to(torch.float32).contiguous() if has_bias else x_c

    est_m_ctas = max(1, (M + 63) // 64)
    num_splits = 1
    if est_m_ctas < _CCE_SPLIT_V_TARGET_CTAS and V >= 256:
        num_splits = min(8, max(1, (_CCE_SPLIT_V_TARGET_CTAS + est_m_ctas - 1) // est_m_ctas))
    v_chunk = ((V + num_splits - 1) // num_splits + 127) // 128 * 128
    num_splits = max(1, (V + v_chunk - 1) // v_chunk)

    partial_lse = torch.empty((num_splits, M), device=x_c.device, dtype=torch.float32)
    even_d = (D % 64) == 0
    _cce_lse_fwd_kernel[lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), num_splits)](
        x_c,
        w_c,
        b_c,
        partial_lse,
        x_c.stride(0),
        x_c.stride(1),
        w_c.stride(0),
        w_c.stride(1),
        partial_lse.stride(0),
        M,
        V,
        D,
        v_chunk,
        float(logit_scale),
        HAS_BIAS=has_bias,
        HAS_LOGIT_SCALE=float(logit_scale) != 1.0,
        EVEN_D=even_d,
    )
    lse_1d = partial_lse[0] if num_splits == 1 else torch.logsumexp(partial_lse, dim=0)

    z_target = torch.empty((M,), device=x_c.device, dtype=torch.float32)
    block_d_tgt = min(1024, max(64, triton.next_power_of_2(D)))
    _cce_target_logit_kernel[(M,)](
        x_c,
        w_c,
        b_c,
        t_c,
        z_target,
        x_c.stride(0),
        x_c.stride(1),
        w_c.stride(0),
        w_c.stride(1),
        M,
        V,
        D,
        int(ignore_index),
        float(logit_scale),
        HAS_BIAS=has_bias,
        HAS_LOGIT_SCALE=float(logit_scale) != 1.0,
        BLOCK_D=block_d_tgt,
    )

    valid_mask = (t_c != int(ignore_index)) & (t_c >= 0) & (t_c < V)
    lse_clean = torch.where(valid_mask, lse_1d, 0.0)
    if label_smoothing > 0.0:
        w_mean = w_c.mean(dim=0, dtype=torch.float32)
        z_mean = x_c.to(torch.float32) @ w_mean
        if has_bias:
            z_mean = z_mean + b_c.mean()
        if logit_scale != 1.0:
            z_mean = z_mean * float(logit_scale)
        loss_raw = lse_clean - (1.0 - float(label_smoothing)) * z_target - float(label_smoothing) * z_mean
    else:
        loss_raw = lse_clean - z_target
    losses_1d = torch.where(valid_mask, loss_raw, 0.0)
    return losses_1d, lse_clean
