from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from ..config import StepOptions

ProcessGroupTuple = Tuple[Any, ...]


def is_dtensor(t: Any) -> bool:
    return hasattr(t, "to_local") and hasattr(t, "device_mesh") and hasattr(t, "placements")


def _to_local_tensor(t: Any) -> Tensor:
    return t.to_local() if is_dtensor(t) else t


def unwrap_local(t: Optional[Tensor]) -> Optional[Tensor]:
    if t is None:
        return None
    return _to_local_tensor(t)


def extract_shard_process_groups(p: Any) -> ProcessGroupTuple:
    """Return process groups along DTensor Shard dimensions, or (None,) for standard tensors."""
    if not is_dtensor(p):
        return (None,)
    mesh = p.device_mesh
    placements = p.placements
    groups: List[Any] = []
    for dim_idx, pl in enumerate(placements):
        if type(pl).__name__ == "Shard" or hasattr(pl, "dim"):
            groups.append(mesh.get_group(dim_idx))
    return tuple(groups)


def prepare_fsdp_local_views(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Optional[Tensor]],
    master_params: Optional[List[Tensor]],
) -> Tuple[
    List[Tensor],
    List[Tensor],
    List[Tensor],
    List[Optional[Tensor]],
    Optional[List[Tensor]],
    List[ProcessGroupTuple],
]:
    """Unwrap FSDP2 DTensor shards to local Tensor views while extracting per-param shard groups."""
    shard_groups = [extract_shard_process_groups(p) for p in params]
    if not any(is_dtensor(p) for p in params):
        return params, grads, exp_avgs, exp_avg_sqs, master_params, shard_groups

    loc_p: List[Tensor] = [_to_local_tensor(p) for p in params]
    loc_g: List[Tensor] = []
    for g in grads:
        g_l = _to_local_tensor(g)
        loc_g.append(g_l if g_l.is_contiguous() else g_l.contiguous())
    loc_m: List[Tensor] = [_to_local_tensor(m) for m in exp_avgs]
    loc_v: List[Optional[Tensor]] = [_to_local_tensor(v) if v is not None else None for v in exp_avg_sqs]
    loc_mp: Optional[List[Tensor]] = None
    if master_params is not None:
        loc_mp = [
            loc_p[i] if (mp is params[i]) else _to_local_tensor(mp)
            for i, mp in enumerate(master_params)
        ]
    return loc_p, loc_g, loc_m, loc_v, loc_mp, shard_groups


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


def sync_fsdp_scratch(
    sc: Tensor,
    local_numel: int,
    process_groups: ProcessGroupTuple = (None,),
) -> float:
    if not process_groups:
        return max(float(local_numel), 1.0)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sync_buf = torch.empty((4,), device=sc.device, dtype=torch.float32)
        sync_buf[:3].copy_(sc[:3])
        sync_buf[3] = float(local_numel)
        for pg in process_groups:
            if pg is None:
                torch.distributed.all_reduce(sync_buf, op=torch.distributed.ReduceOp.SUM)
            else:
                torch.distributed.all_reduce(sync_buf, op=torch.distributed.ReduceOp.SUM, group=pg)
        sc[:3].copy_(sync_buf[:3])
        return max(float(sync_buf[3].item()), 1.0)
    return max(float(local_numel), 1.0)


def sync_fsdp_scratch_batch(
    scratches: List[Tensor],
    local_numels: List[int],
    process_groups_list: Optional[List[ProcessGroupTuple]] = None,
) -> List[float]:
    n_params = len(scratches)
    if n_params == 0:
        return []
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return [max(float(n), 1.0) for n in local_numels]

    if process_groups_list is None:
        pg_list: List[ProcessGroupTuple] = [(None,)] * n_params
    else:
        pg_list = process_groups_list

    # Group indices by identical shard process-group tuple (preserving deterministic order).
    buckets: Dict[Tuple[int, ...], Tuple[ProcessGroupTuple, List[int]]] = {}
    for idx, groups in enumerate(pg_list):
        key = tuple(0 if g is None else id(g) for g in groups)
        if key not in buckets:
            buckets[key] = (groups, [idx])
        else:
            buckets[key][1].append(idx)

    totals: List[float] = [1.0] * n_params
    for groups, idxs in buckets.values():
        if not groups:
            for idx in idxs:
                totals[idx] = max(float(local_numels[idx]), 1.0)
            continue
        if len(idxs) == 1:
            idx0 = idxs[0]
            totals[idx0] = sync_fsdp_scratch(scratches[idx0], local_numels[idx0], process_groups=groups)
            continue
        device = scratches[idxs[0]].device
        sync_buf = torch.empty((len(idxs), 4), device=device, dtype=torch.float32)
        for row_i, idx in enumerate(idxs):
            sync_buf[row_i, :3].copy_(scratches[idx][:3])
            sync_buf[row_i, 3] = float(local_numels[idx])
        for pg in groups:
            if pg is None:
                torch.distributed.all_reduce(sync_buf, op=torch.distributed.ReduceOp.SUM)
            else:
                torch.distributed.all_reduce(sync_buf, op=torch.distributed.ReduceOp.SUM, group=pg)
        for row_i, idx in enumerate(idxs):
            scratches[idx][:3].copy_(sync_buf[row_i, :3])
            totals[idx] = max(float(sync_buf[row_i, 3].item()), 1.0)
    return totals


def fallback_post_step_ema(
    sc: Tensor,
    n_elements: int,
    opts: StepOptions,
    *,
    scratch_synced: bool,
    synced_total_elements: float = 0.0,
    ema_decay_override: float | None = None,
    process_groups: ProcessGroupTuple = (None,),
) -> None:
    ema_decay = opts.ema_decay if ema_decay_override is None else ema_decay_override
    total_elements = (
        synced_total_elements
        if (opts.fsdp_sync and scratch_synced and synced_total_elements > 0.0)
        else float(n_elements)
    )
    if opts.fsdp_sync and not scratch_synced and opts.needs_ema_sync:
        total_elements = sync_fsdp_scratch(sc, n_elements, process_groups=process_groups)
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
    process_groups: ProcessGroupTuple = (None,),
) -> None:
    if not scratch_synced:
        sc[:3].zero_()
    sc[5].zero_()
    if opts.needs_ema_sync:
        tot_elems = (
            sync_fsdp_scratch(sc, 0, process_groups=process_groups)
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
            process_groups=process_groups,
        )
    else:
        sc[:3].zero_()
        sc[5].zero_()
