from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._base import HAS_TRITON, _HAS_TORCH, tl, torch, triton

if TYPE_CHECKING:
    _fused_ce_chunk_fwd_kernel: Any = None
    _fused_ce_chunk_bwd_inplace_kernel: Any = None
elif HAS_TRITON:
    @triton.jit
    def _fused_ce_chunk_fwd_kernel(
        z_ptr,
        target_ptr,
        loss_ptr,
        lse_ptr,
        stride_z_row,
        V,
        ignore_index,
        label_smoothing: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        target_i = tl.load(target_ptr + row_idx)
        valid_i = (target_i != ignore_index) & (target_i >= 0) & (target_i < V)

        if not valid_i:
            tl.store(loss_ptr + row_idx, 0.0)
            tl.store(lse_ptr + row_idx, 0.0)
            return

        row_base = z_ptr + row_idx * stride_z_row
        m_i = -float("inf")
        l_i = 0.0
        s_i = 0.0

        for start_v in range(0, V, BLOCK_V):
            offs_v = start_v + tl.arange(0, BLOCK_V)
            mask_v = offs_v < V
            z_tile = tl.load(row_base + offs_v, mask=mask_v, other=-float("inf")).to(tl.float32)

            s_i += tl.sum(tl.where(mask_v, z_tile, 0.0), axis=0)
            tile_max = tl.max(z_tile, axis=0)
            m_new = tl.maximum(m_i, tile_max)
            alpha = tl.exp(m_i - m_new) if m_i > -float("inf") else 0.0
            p_tile = tl.where(mask_v, tl.exp(z_tile - m_new), 0.0)
            l_i = l_i * alpha + tl.sum(p_tile, axis=0)
            m_i = m_new

        lse_i = m_i + tl.log(l_i)
        z_target = tl.load(row_base + target_i).to(tl.float32)
        loss_i = lse_i - (1.0 - label_smoothing) * z_target - (label_smoothing / V) * s_i
        tl.store(loss_ptr + row_idx, loss_i)
        tl.store(lse_ptr + row_idx, lse_i)

    @triton.jit
    def _fused_ce_chunk_bwd_inplace_kernel(
        z_ptr,
        target_ptr,
        lse_ptr,
        grad_loss_ptr,
        stride_z_row,
        V,
        ignore_index,
        label_smoothing: tl.constexpr,
        logit_scale: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        target_i = tl.load(target_ptr + row_idx)
        valid_i = (target_i != ignore_index) & (target_i >= 0) & (target_i < V)
        row_base = z_ptr + row_idx * stride_z_row

        if not valid_i:
            for start_v in range(0, V, BLOCK_V):
                offs_v = start_v + tl.arange(0, BLOCK_V)
                mask_v = offs_v < V
                tl.store(row_base + offs_v, 0.0, mask=mask_v)
            return

        lse_i = tl.load(lse_ptr + row_idx).to(tl.float32)
        grad_i = tl.load(grad_loss_ptr + row_idx).to(tl.float32) * logit_scale

        for start_v in range(0, V, BLOCK_V):
            offs_v = start_v + tl.arange(0, BLOCK_V)
            mask_v = offs_v < V
            z_tile = tl.load(row_base + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            p_tile = tl.exp(z_tile - lse_i)
            target_term = tl.where(offs_v == target_i, 1.0 - label_smoothing, 0.0)
            dz_tile = (p_tile - target_term - (label_smoothing / V)) * grad_i
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
) -> bool:
    """Dispatches 1-row-per-token online SRAM LogSumExp + Cross-Entropy forward pass on a `(C, V)` chunk."""
    if not (
        HAS_TRITON
        and _HAS_TORCH
        and isinstance(z_chunk, torch.Tensor)
        and z_chunk.is_cuda
        and z_chunk.ndim == 2
    ):
        return False

    C, V = z_chunk.shape
    if C == 0 or V == 0:
        return False

    BLOCK_V = min(4096, max(64, triton.next_power_of_2(min(V, 4096))))
    _fused_ce_chunk_fwd_kernel[(C,)](
        z_chunk,
        target_chunk,
        losses_out,
        lse_out,
        z_chunk.stride(0),
        V,
        int(ignore_index),
        label_smoothing=float(label_smoothing),
        BLOCK_V=BLOCK_V,
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
) -> bool:
    """Overwrites `z_chunk` `(C, V)` in-place with analytical logit gradients in a single SRAM pass."""
    if not (
        HAS_TRITON
        and _HAS_TORCH
        and isinstance(z_chunk, torch.Tensor)
        and z_chunk.is_cuda
        and z_chunk.ndim == 2
    ):
        return False

    C, V = z_chunk.shape
    if C == 0 or V == 0:
        return False

    BLOCK_V = min(4096, max(64, triton.next_power_of_2(min(V, 4096))))
    _fused_ce_chunk_bwd_inplace_kernel[(C,)](
        z_chunk,
        target_chunk,
        lse_chunk,
        grad_loss_chunk,
        z_chunk.stride(0),
        V,
        int(ignore_index),
        label_smoothing=float(label_smoothing),
        logit_scale=float(logit_scale),
        BLOCK_V=BLOCK_V,
    )
    return True
