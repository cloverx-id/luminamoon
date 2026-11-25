from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ._base import HAS_TRITON, _HAS_TORCH, tl, torch, triton

if TYPE_CHECKING:
    _fused_ce_chunk_fwd_kernel: Any = None
    _fused_ce_chunk_bwd_inplace_kernel: Any = None
elif HAS_TRITON:
    def _prune_ce_configs(configs: Any, named_args: Any, **kwargs: Any) -> Any:
        v_size = int(named_args.get("V", kwargs.get("V", 4096)))
        max_blk = max(1024, triton.next_power_of_2(min(v_size, 4096)))
        kept = [c for c in configs if c.kwargs["BLOCK_V"] <= max_blk]
        return kept if kept else configs[:1]

    _CE_CHUNK_CONFIGS = [
        triton.Config({"BLOCK_V": 1024}, num_warps=4),
        triton.Config({"BLOCK_V": 2048}, num_warps=8),
        triton.Config({"BLOCK_V": 4096}, num_warps=8),
        triton.Config({"BLOCK_V": 4096}, num_warps=16),
    ]

    @triton.autotune(
        configs=_CE_CHUNK_CONFIGS,
        key=["V"],
        prune_configs_by={"early_config_prune": _prune_ce_configs},
    )
    @triton.jit
    def _fused_ce_chunk_fwd_kernel(
        z_ptr,
        bias_ptr,
        target_ptr,
        loss_ptr,
        lse_ptr,
        stride_z_row,
        V,
        ignore_index,
        label_smoothing,
        logit_scale,
        HAS_BIAS: tl.constexpr,
        HAS_LOGIT_SCALE: tl.constexpr,
        HAS_LABEL_SMOOTHING: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        target_i = tl.load(target_ptr + row_idx)
        valid_i = (target_i != ignore_index) & (target_i >= 0) & (target_i < V)

        if not valid_i:
            tl.store(loss_ptr + row_idx, 0.0)
            tl.store(lse_ptr + row_idx, 0.0)
            return

        row_base = z_ptr + row_idx.to(tl.int64) * stride_z_row
        m_i = -float("inf")
        l_i = 0.0
        s_i = 0.0

        for start_v in range(0, V, BLOCK_V):
            offs_v = start_v + tl.arange(0, BLOCK_V)
            mask_v = offs_v < V
            z_tile = tl.load(row_base + offs_v, mask=mask_v, other=-float("inf")).to(tl.float32)
            if HAS_BIAS:
                b_tile = tl.load(bias_ptr + offs_v, mask=mask_v, other=0.0).to(tl.float32)
                z_tile = tl.where(mask_v, z_tile + b_tile, -float("inf"))
            if HAS_LOGIT_SCALE:
                z_tile = tl.where(mask_v, z_tile * logit_scale, -float("inf"))

            if HAS_LABEL_SMOOTHING:
                s_i += tl.sum(tl.where(mask_v, z_tile, 0.0), axis=0)
            tile_max = tl.max(z_tile, axis=0)
            m_new = tl.maximum(m_i, tile_max)
            alpha = tl.where(m_i > -float("inf"), tl.exp(m_i - m_new), 0.0)
            p_tile = tl.where(mask_v, tl.exp(z_tile - m_new), 0.0)
            l_i = l_i * alpha + tl.sum(p_tile, axis=0)
            m_i = m_new

        lse_i = m_i + tl.log(l_i)
        z_target = tl.load(row_base + target_i).to(tl.float32)
        if HAS_BIAS:
            z_target += tl.load(bias_ptr + target_i).to(tl.float32)
        if HAS_LOGIT_SCALE:
            z_target *= logit_scale
        if HAS_LABEL_SMOOTHING:
            loss_i = lse_i - (1.0 - label_smoothing) * z_target - (label_smoothing / V) * s_i
        else:
            loss_i = lse_i - z_target
        tl.store(loss_ptr + row_idx, loss_i)
        tl.store(lse_ptr + row_idx, lse_i)

    @triton.autotune(
        configs=_CE_CHUNK_CONFIGS,
        key=["V"],
        prune_configs_by={"early_config_prune": _prune_ce_configs},
    )
    @triton.jit
    def _fused_ce_chunk_bwd_inplace_kernel(
        z_ptr,
        bias_ptr,
        target_ptr,
        lse_ptr,
        grad_loss_ptr,
        stride_z_row,
        V,
        ignore_index,
        label_smoothing,
        logit_scale,
        HAS_BIAS: tl.constexpr,
        HAS_LOGIT_SCALE: tl.constexpr,
        HAS_LABEL_SMOOTHING: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        target_i = tl.load(target_ptr + row_idx)
        valid_i = (target_i != ignore_index) & (target_i >= 0) & (target_i < V)
        row_base = z_ptr + row_idx.to(tl.int64) * stride_z_row

        if not valid_i:
            for start_v in range(0, V, BLOCK_V):
                offs_v = start_v + tl.arange(0, BLOCK_V)
                mask_v = offs_v < V
                tl.store(row_base + offs_v, 0.0, mask=mask_v)
            return

        lse_i = tl.load(lse_ptr + row_idx).to(tl.float32)
        grad_i = tl.load(grad_loss_ptr + row_idx).to(tl.float32) * logit_scale
        target_weight = 1.0 - label_smoothing
        smooth_term = label_smoothing / V

        for start_v in range(0, V, BLOCK_V):
            offs_v = start_v + tl.arange(0, BLOCK_V)
            mask_v = offs_v < V
            z_tile = tl.load(row_base + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            if HAS_BIAS:
                b_tile = tl.load(bias_ptr + offs_v, mask=mask_v, other=0.0).to(tl.float32)
                z_tile = z_tile + b_tile
            if HAS_LOGIT_SCALE:
                z_tile = z_tile * logit_scale
            p_tile = tl.exp(z_tile - lse_i)
            target_term = tl.where(offs_v == target_i, target_weight, 0.0)
            if HAS_LABEL_SMOOTHING:
                dz_tile = (p_tile - target_term - smooth_term) * grad_i
            else:
                dz_tile = (p_tile - target_term) * grad_i
            tl.store(row_base + offs_v, dz_tile, mask=mask_v)
else:
    _fused_ce_chunk_fwd_kernel = None
    _fused_ce_chunk_bwd_inplace_kernel = None


def triton_fused_ce_chunk_fwd(
    z_chunk: Any,
    target_chunk: Any,
    losses_out: Any,
    lse_out: Any,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    bias: Optional[Any] = None,
    logit_scale: float = 1.0,
) -> bool:
    """Dispatches 1-row-per-token online SRAM LogSumExp + Cross-Entropy forward pass on a `(C, V)` chunk."""
    if not (
        HAS_TRITON
        and _HAS_TORCH
        and isinstance(z_chunk, torch.Tensor)
        and z_chunk.is_cuda
        and z_chunk.ndim == 2
        and z_chunk.stride(1) == 1
        and z_chunk.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return False

    C, V = z_chunk.shape
    if C == 0 or V == 0:
        return False

    has_bias = bias is not None
    b_c = (bias if bias.is_contiguous() else bias.contiguous()) if has_bias else z_chunk
    t_c = target_chunk if target_chunk.is_contiguous() else target_chunk.contiguous()
    _fused_ce_chunk_fwd_kernel[(C,)](
        z_chunk,
        b_c,
        t_c,
        losses_out,
        lse_out,
        z_chunk.stride(0),
        V,
        int(ignore_index),
        float(label_smoothing),
        float(logit_scale),
        HAS_BIAS=has_bias,
        HAS_LOGIT_SCALE=float(logit_scale) != 1.0,
        HAS_LABEL_SMOOTHING=float(label_smoothing) > 0.0,
        enable_fp_fusion=True,
    )
    return True


def triton_fused_ce_chunk_bwd_inplace(
    z_chunk: Any,
    target_chunk: Any,
    lse_chunk: Any,
    grad_loss_chunk: Any,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    bias: Optional[Any] = None,
) -> bool:
    """Overwrites `z_chunk` `(C, V)` in-place with analytical logit gradients in a single SRAM pass."""
    if not (
        HAS_TRITON
        and _HAS_TORCH
        and isinstance(z_chunk, torch.Tensor)
        and z_chunk.is_cuda
        and z_chunk.ndim == 2
        and z_chunk.stride(1) == 1
        and z_chunk.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return False

    C, V = z_chunk.shape
    if C == 0 or V == 0:
        return False

    has_bias = bias is not None
    b_c = (bias if bias.is_contiguous() else bias.contiguous()) if has_bias else z_chunk
    t_c = target_chunk if target_chunk.is_contiguous() else target_chunk.contiguous()
    _fused_ce_chunk_bwd_inplace_kernel[(C,)](
        z_chunk,
        b_c,
        t_c,
        lse_chunk,
        grad_loss_chunk,
        z_chunk.stride(0),
        V,
        int(ignore_index),
        float(label_smoothing),
        float(logit_scale),
        HAS_BIAS=has_bias,
        HAS_LOGIT_SCALE=float(logit_scale) != 1.0,
        HAS_LABEL_SMOOTHING=float(label_smoothing) > 0.0,
        enable_fp_fusion=True,
    )
    return True
