from __future__ import annotations

import math
from typing import List
import torch
from torch import Tensor


def _sync_fsdp_scratch(sc: Tensor, local_numel: int) -> float:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sync_buf = torch.empty((4,), device=sc.device, dtype=torch.float32)
        sync_buf[:3].copy_(sc[:3])
        sync_buf[3] = float(local_numel)
        torch.distributed.all_reduce(sync_buf, op=torch.distributed.ReduceOp.SUM)
        sc[:3].copy_(sync_buf[:3])
        return max(float(sync_buf[3].item()), 1.0)
    return max(float(local_numel), 1.0)


def _sync_fsdp_diff_only(sc: Tensor, local_numel: int) -> float:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sync_buf = torch.empty((2,), device=sc.device, dtype=torch.float32)
        sync_buf[0] = sc[5]
        sync_buf[1] = float(local_numel)
        torch.distributed.all_reduce(sync_buf, op=torch.distributed.ReduceOp.SUM)
        sc[5].copy_(sync_buf[0])
        return max(float(sync_buf[1].item()), 1.0)
    return max(float(local_numel), 1.0)


def _sync_fsdp_diff_batch(scratches: List[Tensor], local_numels: List[int]) -> List[float]:
    n_params = len(scratches)
    if n_params == 0:
        return []
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return [max(float(n), 1.0) for n in local_numels]
    if n_params == 1:
        return [_sync_fsdp_diff_only(scratches[0], local_numels[0])]
    device = scratches[0].device
    sync_buf = torch.empty((n_params, 2), device=device, dtype=torch.float32)
    for idx, (sc, numel) in enumerate(zip(scratches, local_numels)):
        sync_buf[idx, 0] = sc[5]
        sync_buf[idx, 1] = float(numel)
    torch.distributed.all_reduce(sync_buf, op=torch.distributed.ReduceOp.SUM)
    totals: List[float] = []
    for idx, sc in enumerate(scratches):
        sc[5].copy_(sync_buf[idx, 0])
        totals.append(max(float(sync_buf[idx, 1].item()), 1.0))
    return totals


def _sync_fsdp_scratch_batch(scratches: List[Tensor], local_numels: List[int]) -> List[float]:
    n_params = len(scratches)
    if n_params == 0:
        return []
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return [max(float(n), 1.0) for n in local_numels]
    if n_params == 1:
        return [_sync_fsdp_scratch(scratches[0], local_numels[0])]
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


def _fallback_post_step_ema(
    sc: Tensor,
    n_elements: int,
    lr: float,
    clamp_min: float,
    ema_decay: float,
    bound_mode: int,
    bound_ratio: float,
    cautious: bool,
    fsdp_sync: bool,
    scratch_synced: bool,
    synced_total_elements: float = 0.0,
) -> None:
    total_elements = synced_total_elements if (fsdp_sync and scratch_synced and synced_total_elements > 0.0) else float(n_elements)
    if fsdp_sync and not scratch_synced and (cautious or bound_mode == 1):
        total_elements = _sync_fsdp_scratch(sc, n_elements)
    if cautious:
        mask_sum = sc[0].item()
        m_bar = max(min(mask_sum / max(total_elements, 1.0), 1.0), clamp_min)
        target_cautious = 1.0 / m_bar
        old_cautious = sc[3].item()
        sc[3] = ema_decay * old_cautious + (1.0 - ema_decay) * target_cautious
    if bound_mode == 1:
        u_sq_sum = max(sc[1].item(), 0.0)
        p_sq_sum = max(sc[2].item(), 0.0)
        c_scale = sc[3].item() if cautious else 1.0
        norm_p = math.sqrt(p_sq_sum)
        norm_delta = math.sqrt(u_sq_sum) * c_scale
        R = bound_ratio * max(norm_p, 1.0)
        r = (lr * norm_delta) / R
        target_bound = 1.0 if r < 1e-4 else (math.tanh(r) / max(r, 1e-8))
        old_bound = sc[4].item()
        sc[4] = ema_decay * old_bound + (1.0 - ema_decay) * target_bound
    sc[:3].zero_()
    sc[5].zero_()


def _sync_empty_fsdp_shard(
    sc: Tensor,
    step: int,
    lr: float,
    clamp_min: float,
    ema_decay: float,
    bound_mode: int,
    bound_ratio: float,
    cautious: bool,
    centered_defect: bool,
    fused_single_pass: bool,
    diff_synced: bool = False,
    scratch_synced: bool = False,
    synced_total_elements: float = 0.0,
) -> None:
    if not scratch_synced:
        sc[:3].zero_()
    sc[5].zero_()
    if centered_defect and not diff_synced:
        _sync_fsdp_diff_only(sc, 0)
    if cautious or bound_mode == 1:
        tot_elems = (
            _sync_fsdp_scratch(sc, 0)
            if not scratch_synced
            else (synced_total_elements if synced_total_elements > 0.0 else 1.0)
        )
        eff_decay = ema_decay if (fused_single_pass and step > 1) else 0.0
        _fallback_post_step_ema(
            sc, 0, lr, clamp_min, eff_decay, bound_mode, bound_ratio,
            cautious, True, True, synced_total_elements=tot_elems,
        )
    else:
        sc[:3].zero_()
        sc[5].zero_()
