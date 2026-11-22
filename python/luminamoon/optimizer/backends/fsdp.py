from __future__ import annotations

import math
from typing import List

import torch
from torch import Tensor

from ..config import StepOptions


def tanh_pade_f32(x_in: float) -> float:
    if math.isnan(x_in):
        return float("nan")
    x = max(min(x_in, 7.9988117), -7.9988117)
    x2 = x * x
    p = -2.7607684e-16 * x2 + 2.000188e-13
    q = 1.1982584e-6 * x2 + 1.1853471e-4
    p = p * x2 - 8.604672e-11
    q = q * x2 + 2.2684346e-3
    p = p * x2 + 5.122297e-8
    q = q * x2 + 4.8935246e-3
    p = p * x2 + 1.4857224e-5
    p = p * x2 + 6.3726193e-4
    p = p * x2 + 4.8935246e-3
    return (x * p) / q


def radial_scale_scalar(r: float, is_triangle: bool) -> float:
    if is_triangle:
        return (1.0 + r) / (1.0 + r + r * r)
    if r < 1e-4:
        return 1.0
    return tanh_pade_f32(r) / max(r, 1e-8)


def sync_fsdp_scratch(sc: Tensor, local_numel: int) -> float:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sync_buf = torch.empty((4,), device=sc.device, dtype=torch.float32)
        sync_buf[:3].copy_(sc[:3])
        sync_buf[3] = float(local_numel)
        torch.distributed.all_reduce(sync_buf, op=torch.distributed.ReduceOp.SUM)
        sc[:3].copy_(sync_buf[:3])
        return max(float(sync_buf[3].item()), 1.0)
    return max(float(local_numel), 1.0)


def sync_fsdp_scratch_batch(scratches: List[Tensor], local_numels: List[int]) -> List[float]:
    n_params = len(scratches)
    if n_params == 0:
        return []
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return [max(float(n), 1.0) for n in local_numels]
    if n_params == 1:
        return [sync_fsdp_scratch(scratches[0], local_numels[0])]
    device = scratches[0].device
    sync_buf = torch.empty((n_params, 4), device=device, dtype=torch.float32)
    for idx, (sc, numel) in enumerate(zip(scratches, local_numels)):
        sync_buf[idx, :3].copy_(sc[:3])
        sync_buf[idx, 3] = float(numel)
    torch.distributed.all_reduce(sync_buf, op=torch.distributed.ReduceOp.SUM)
    totals: List[float] = []
    for idx, sc in enumerate(scratches):
        sc[:3].copy_(sync_buf[idx, :3])
        totals.append(max(float(sync_buf[idx, 3].item()), 1.0))
    return totals


def fallback_post_step_ema(
    sc: Tensor,
    n_elements: int,
    opts: StepOptions,
    *,
    scratch_synced: bool,
    synced_total_elements: float = 0.0,
    ema_decay_override: float | None = None,
) -> None:
    ema_decay = opts.ema_decay if ema_decay_override is None else ema_decay_override
    total_elements = (
        synced_total_elements
        if (opts.fsdp_sync and scratch_synced and synced_total_elements > 0.0)
        else float(n_elements)
    )
    if opts.fsdp_sync and not scratch_synced and opts.needs_ema_sync:
        total_elements = sync_fsdp_scratch(sc, n_elements)
    if opts.cautious:
        mask_sum = sc[0].item()
        m_bar = max(min(mask_sum / max(total_elements, 1.0), 1.0), opts.clamp_min)
        target_cautious = 1.0 / m_bar
        sc[3] = ema_decay * sc[3].item() + (1.0 - ema_decay) * target_cautious
    if opts.bound_mode == 1:
        u_sq_sum = max(sc[1].item(), 0.0)
        p_sq_sum = max(sc[2].item(), 0.0)
        c_scale = sc[3].item() if opts.cautious else 1.0
        norm_p = math.sqrt(p_sq_sum)
        norm_delta = math.sqrt(u_sq_sum) * c_scale
        radius = opts.bound_ratio * max(norm_p, 1.0)
        r = (opts.lr * norm_delta) / radius
        target_bound = radial_scale_scalar(r, opts.is_triangle)
        old_bound = sc[4].item()
        gamma_eff = min(ema_decay, 0.10) if target_bound < old_bound else ema_decay
        sc[4] = gamma_eff * old_bound + (1.0 - gamma_eff) * target_bound
    sc[:3].zero_()
    sc[5].zero_()


def sync_empty_fsdp_shard(
    sc: Tensor,
    step: int,
    opts: StepOptions,
    *,
    scratch_synced: bool = False,
    synced_total_elements: float = 0.0,
) -> None:
    if not scratch_synced:
        sc[:3].zero_()
    sc[5].zero_()
    if opts.needs_ema_sync:
        tot_elems = (
            sync_fsdp_scratch(sc, 0)
            if not scratch_synced
            else (synced_total_elements if synced_total_elements > 0.0 else 1.0)
        )
        eff_decay = opts.ema_decay if (opts.fused_single_pass and step > 1) else 0.0
        fallback_post_step_ema(
            sc,
            0,
            opts,
            scratch_synced=True,
            synced_total_elements=tot_elems,
            ema_decay_override=eff_decay,
        )
    else:
        sc[:3].zero_()
        sc[5].zero_()
