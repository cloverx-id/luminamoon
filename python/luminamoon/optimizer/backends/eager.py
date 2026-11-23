from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from ..config import StepOptions
from ..constants import EAGER_ELEM_CHUNK_SIZE
from ..torch_ops import (
    _apply_bound_scalar,
    _apply_master_update,
    _decode_state_f32,
    _encode_state_from_f32,
    _sr_apply_step_val,
    _tanh_pade_torch,
    _temper_belief_defect_torch,
)
from .fsdp import (
    fallback_post_step_ema,
    radial_scale_scalar,
    sync_empty_fsdp_shard,
    sync_fsdp_scratch,
    sync_fsdp_scratch_batch,
)


def _compute_update_chunk(
    g_c: Tensor,
    m_c: Tensor,
    v_c: Optional[Tensor],
    opts: StepOptions,
    c1: float,
    c2: float,
    use_rtv: bool = False,
    inv_sqrt_bc2_prev: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    g_f32 = g_c.float()
    valid_g = torch.isfinite(g_f32) & (torch.abs(g_f32) < 1.0e38)
    g_clean = torch.where(valid_g, g_f32.clamp(-1e18, 1e18), torch.zeros_like(g_f32))
    m_f32 = _decode_state_f32(m_c)

    if opts.is_dual:
        assert v_c is not None
        v_f32 = _decode_state_f32(v_c)
        m_new = m_f32.mul(opts.beta1).add(g_clean, alpha=1.0 - opts.beta1)
        diff = (g_clean - (m_f32 if opts.is_t2 else m_new)).clamp(-1e18, 1e18)
        abs_d = torch.abs(diff)
        d_mem = (
            _temper_belief_defect_torch(
                abs_d, v_f32, m_f32, inv_sqrt_bc2_prev, opts.variance_temper, opts.eps
            )
            if use_rtv
            else abs_d
        )
        v_new = v_f32.mul(opts.beta2).addcmul(d_mem, d_mem, value=1.0 - opts.beta2)
        if opts.is_t2:
            beta1_sq = opts.beta1 * opts.beta1
            one_minus_beta1_sq = 1.0 - beta1_sq
            num = m_f32.mul(beta1_sq).add(g_clean, alpha=one_minus_beta1_sq)
            den = (
                torch.abs(m_f32)
                .mul(beta1_sq)
                .add(torch.abs(g_clean), alpha=one_minus_beta1_sq)
                .add_(abs_d.mul(c2))
                .add_(v_new.clamp_min(0.0).sqrt().mul(c1))
                .add_(opts.eps)
            )
            u = num / den
        else:
            clm_m = m_new.mul(opts.beta1).add(g_clean, alpha=1.0 - opts.beta1)
            sigma = v_new.clamp_min(0.0).sqrt().mul(c1).add(c2)
            u = _tanh_pade_torch(clm_m / sigma.clamp_min(1e-12))
        return u, g_clean, m_new, v_new

    beta1_sq = opts.beta1 * opts.beta1
    one_minus_beta1_sq = 1.0 - beta1_sq
    clm_m = m_f32.mul(beta1_sq).add(g_clean, alpha=one_minus_beta1_sq)
    diff = (g_clean - m_f32).clamp(-1e18, 1e18)
    clm_env = (
        torch.abs(m_f32)
        .mul(beta1_sq)
        .add(torch.abs(g_clean), alpha=one_minus_beta1_sq)
        .add_(torch.abs(diff).mul(opts.lam))
        .add_(opts.eps)
    )
    u = clm_m / clm_env
    m_new = m_f32.mul(opts.beta1).add(g_clean, alpha=1.0 - opts.beta1)
    return u, g_clean, m_new, None


def compute_pass1_eager_one(
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Optional[Tensor],
    ref_p: Tensor,
    sc: Tensor,
    opts: StepOptions,
    step: int,
    chunk_size: int = EAGER_ELEM_CHUNK_SIZE,
) -> None:
    sc[:3].zero_()
    n_el = p.numel()
    if n_el == 0:
        return

    c1, c2 = opts.dual_coefficients(step) if opts.is_dual else (0.0, 0.0)
    use_rtv = opts.use_rtv(step)
    inv_sqrt_bc2_prev = opts.inv_sqrt_bc2_prev(step)

    flat_g = (grad if grad.is_contiguous() else grad.contiguous()).view(-1)
    flat_m = (exp_avg if exp_avg.is_contiguous() else exp_avg.contiguous()).view(-1)
    flat_v = (
        (exp_avg_sq if exp_avg_sq.is_contiguous() else exp_avg_sq.contiguous()).view(-1)
        if (opts.is_dual and exp_avg_sq is not None)
        else None
    )
    flat_ref = (ref_p if ref_p.is_contiguous() else ref_p.contiguous()).view(-1)

    mask_sum_acc = 0.0
    u_sq_sum_acc = 0.0
    p_sq_sum_acc = 0.0
    for offset in range(0, n_el, chunk_size):
        end = min(offset + chunk_size, n_el)
        u, g_clean, _, _ = _compute_update_chunk(
            flat_g[offset:end],
            flat_m[offset:end],
            flat_v[offset:end] if flat_v is not None else None,
            opts,
            c1,
            c2,
            use_rtv,
            inv_sqrt_bc2_prev,
        )
        if opts.cautious:
            m_mask = (u * g_clean > 0.0).float()
            mask_sum_acc += m_mask.sum().item()
            u_eff = u * m_mask
        else:
            u_eff = u
        if opts.bound_mode == 1:
            u_sq_sum_acc += torch.sum(u_eff * u_eff).item()
            p_sq_sum_acc += torch.sum(flat_ref[offset:end].float() ** 2).item()

    if opts.cautious:
        sc[0] = mask_sum_acc
    if opts.bound_mode == 1:
        sc[1] = u_sq_sum_acc
        sc[2] = p_sq_sum_acc


def _step_eager_param(
    opt: Any,
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Optional[Tensor],
    ref_p: Tensor,
    step: int,
    sc: Optional[Tensor],
    opts: StepOptions,
    *,
    param_idx: int,
    scratch_synced: bool,
    scratch_total_elements: float,
    defer_fsdp_post_ema: bool,
    process_groups: Tuple[Any, ...] = (None,),
    chunk_size: int = EAGER_ELEM_CHUNK_SIZE,
) -> None:
    n_elem = p.numel()
    has_master = ref_p is not p
    c1, c2 = opts.dual_coefficients(step) if opts.is_dual else (0.0, 0.0)
    use_rtv = opts.use_rtv(step)
    inv_sqrt_bc2_prev = opts.inv_sqrt_bc2_prev(step)

    p_work = p if p.is_contiguous() else p.contiguous()
    g_work = grad if grad.is_contiguous() else grad.contiguous()
    m_work = exp_avg if exp_avg.is_contiguous() else exp_avg.contiguous()
    v_work = (
        (exp_avg_sq if exp_avg_sq.is_contiguous() else exp_avg_sq.contiguous())
        if (opts.is_dual and exp_avg_sq is not None)
        else None
    )
    ref_work = (ref_p if ref_p.is_contiguous() else ref_p.contiguous()) if has_master else p_work

    flat_p, flat_g, flat_m = p_work.view(-1), g_work.view(-1), m_work.view(-1)
    flat_v = v_work.view(-1) if v_work is not None else None
    flat_ref = ref_work.view(-1)

    safe_step = max(step, 1)
    seed_m = (opts.seed + param_idx * 10007) & 0x7FFFFFFF
    seed_p = (seed_m + 0x55555555) & 0x7FFFFFFF
    seed_v = (seed_m + 0x2A6D365B) & 0x7FFFFFFF
    weyl_m = (safe_step * 0x6A09E667) & 0xFFFFFFFF
    weyl_v = (safe_step * 0xBB67AE85) & 0xFFFFFFFF
    weyl_p = (safe_step * 0x9E3779B9) & 0xFFFFFFFF

    use_single_pass = (opts.fused_single_pass and sc is not None and step > 1) or not opts.needs_ema_sync

    if use_single_pass:
        cautious_scale = sc[3].item() if (opts.cautious and sc is not None and step > 1) else 1.0
        bound_scale = sc[4].item() if (opts.bound_mode == 1 and sc is not None and step > 1) else 1.0
    else:
        if opts.fsdp_sync and scratch_synced and sc is not None:
            mask_sum_acc, u_sq_sum_acc, p_sq_sum_acc = sc[0].item(), sc[1].item(), sc[2].item()
            total_elements = scratch_total_elements if scratch_total_elements > 0.0 else float(n_elem)
        else:
            tmp_sc = sc if sc is not None else torch.zeros((6,), device=p.device, dtype=torch.float32)
            compute_pass1_eager_one(p, grad, exp_avg, exp_avg_sq, ref_p, tmp_sc, opts, step, chunk_size)
            total_elements = (
                sync_fsdp_scratch(tmp_sc, n_elem, process_groups=process_groups)
                if (opts.fsdp_sync and not scratch_synced)
                else float(n_elem)
            )
            mask_sum_acc, u_sq_sum_acc, p_sq_sum_acc = tmp_sc[0].item(), tmp_sc[1].item(), tmp_sc[2].item()

        cautious_scale = 1.0
        if opts.cautious:
            m_bar = max(min(mask_sum_acc / max(total_elements, 1.0), 1.0), opts.clamp_min)
            cautious_scale = 1.0 / m_bar
            if sc is not None:
                sc[3] = cautious_scale

        bound_scale = 1.0
        if opts.bound_mode == 1:
            norm_p = math.sqrt(max(p_sq_sum_acc, 0.0))
            norm_u = math.sqrt(max(u_sq_sum_acc, 0.0)) * cautious_scale
            radius = opts.bound_ratio * max(norm_p, 1.0)
            bound_scale = radial_scale_scalar((opts.lr * norm_u) / radius, opts.is_triangle)
            if sc is not None:
                sc[4] = bound_scale

    radial_tensor = torch.tensor(bound_scale, device=p.device, dtype=torch.float32) if opts.bound_mode == 1 else None
    mask_sum_post = u_sq_sum_post = p_sq_sum_post = 0.0

    for offset in range(0, n_elem, chunk_size):
        end = min(offset + chunk_size, n_elem)
        p_c, g_c, m_c, ref_c = flat_p[offset:end], flat_g[offset:end], flat_m[offset:end], flat_ref[offset:end]
        v_c = flat_v[offset:end] if flat_v is not None else None

        u, g_clean, m_new, v_new = _compute_update_chunk(g_c, m_c, v_c, opts, c1, c2, use_rtv, inv_sqrt_bc2_prev)
        _encode_state_from_f32(m_c, m_new, use_sr=opts.use_sr, seed=seed_m, offset_base=offset, step_offset=weyl_m)
        if v_c is not None and v_new is not None:
            _encode_state_from_f32(v_c, v_new, use_sr=opts.use_sr, seed=seed_v, offset_base=offset, step_offset=weyl_v)

        if opts.cautious:
            m_mask = (u * g_clean > 0.0).float()
            delta_theta = u * m_mask * cautious_scale
            if use_single_pass:
                mask_sum_post += m_mask.sum().item()
        else:
            m_mask, delta_theta = None, u

        step_val = opts.lr * delta_theta
        if opts.lr == 0.0:
            step_val = torch.zeros_like(step_val)
        else:
            step_val = _apply_bound_scalar(
                step_val, ref_c.float(), opts.bound, opts.bound_type, opts.bound_ratio, p.dtype,
                radial_scale=radial_tensor, is_triangle=opts.is_triangle, bound_floor=opts.bound_floor,
            )

        if use_single_pass and opts.bound_mode == 1:
            u_eff = (u * m_mask) if (opts.cautious and m_mask is not None) else u
            u_sq_sum_post += torch.sum(u_eff * u_eff).item()
            p_sq_sum_post += torch.sum(ref_c.float() ** 2).item()

        if has_master:
            _apply_master_update(ref_c, p_c, step_val, opts.lr, opts.weight_decay, opts.cautious_weight_decay)
        else:
            _sr_apply_step_val(
                p_c, step_val, opts.lr, opts.weight_decay, opts.use_sr,
                seed=seed_p, offset_base=offset, step_offset=weyl_p,
                cautious_weight_decay=opts.cautious_weight_decay,
            )

    if sc is not None:
        if use_single_pass and opts.needs_ema_sync:
            if not (opts.fsdp_sync and scratch_synced):
                sc[:3].zero_()
                if opts.cautious:
                    sc[0] = mask_sum_post
                if opts.bound_mode == 1:
                    sc[1] = u_sq_sum_post
                    sc[2] = p_sq_sum_post
            if not defer_fsdp_post_ema:
                fallback_post_step_ema(
                    sc, n_elem, opts, scratch_synced=scratch_synced,
                    synced_total_elements=scratch_total_elements, process_groups=process_groups,
                )
        else:
            sc[:3].zero_()
            sc[5].zero_()

    if not p.is_contiguous():
        p.copy_(p_work)
    if not exp_avg.is_contiguous():
        exp_avg.copy_(m_work)
    if v_work is not None and exp_avg_sq is not None and not exp_avg_sq.is_contiguous():
        exp_avg_sq.copy_(v_work)
    if has_master and not ref_p.is_contiguous():
        ref_p.copy_(ref_work)


def step_eager_group(
    opt: Any,
    opts: StepOptions,
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Optional[Tensor]],
    master_params: Optional[List[Tensor]],
    steps: List[Any],
    scratches: Optional[List[Tensor]],
    *,
    scratch_synced: bool = False,
    scratch_total_elements: float = 0.0,
    orig_indices: Optional[List[int]] = None,
    defer_fsdp_post_ema: bool = False,
    shard_groups: Optional[List[Tuple[Any, ...]]] = None,
) -> None:
    pg_all: List[Tuple[Any, ...]] = shard_groups if shard_groups is not None else [(None,)] * len(params)
    batch_post_ema = bool(
        opts.fsdp_sync and not scratch_synced and not defer_fsdp_post_ema
        and scratches is not None and len(params) > 1 and opts.needs_ema_sync
    )
    batched_totals: Dict[int, float] = {}
    if batch_post_ema and scratches is not None:
        pass1_indices = [
            idx
            for idx, st in enumerate(steps)
            if not (opts.fused_single_pass and int(st.item() if isinstance(st, torch.Tensor) else st) > 1)
        ]
        if len(pass1_indices) > 1:
            for idx in pass1_indices:
                p, sc = params[idx], scratches[idx]
                sc[:3].zero_()
                if p.numel() == 0:
                    continue
                st_raw = steps[idx]
                st_int = int(st_raw.item() if isinstance(st_raw, torch.Tensor) else st_raw)
                ref_p = master_params[idx] if (master_params is not None and master_params[idx] is not p) else p
                compute_pass1_eager_one(
                    p, grads[idx], exp_avgs[idx], exp_avg_sqs[idx] if opts.is_dual else None, ref_p, sc, opts, st_int
                )
            totals = sync_fsdp_scratch_batch(
                [scratches[idx] for idx in pass1_indices],
                [params[idx].numel() for idx in pass1_indices],
                process_groups_list=[pg_all[idx] for idx in pass1_indices],
            )
            batched_totals = dict(zip(pass1_indices, totals))

    pending_ema: List[Tuple[Tensor, int, Tuple[Any, ...]]] = []
    for i, p in enumerate(params):
        step_raw = steps[i]
        step = int(step_raw.item() if isinstance(step_raw, torch.Tensor) else step_raw)
        global_idx = orig_indices[i] if orig_indices is not None else i
        ref_p = master_params[i] if (master_params is not None and master_params[i] is not p) else p
        sc = scratches[i] if scratches is not None else None
        pg_i = pg_all[i]
        cur_synced = scratch_synced or (i in batched_totals)
        cur_total = batched_totals[i] if i in batched_totals else scratch_total_elements
        defer_cur = defer_fsdp_post_ema or (batch_post_ema and opts.fused_single_pass and step > 1)

        if p.numel() == 0:
            if opts.fsdp_sync and sc is not None:
                if not cur_synced:
                    sc[:3].zero_()
                sc[5].zero_()
                if defer_cur and opts.needs_ema_sync:
                    if batch_post_ema:
                        pending_ema.append((sc, 0, pg_i))
                else:
                    sync_empty_fsdp_shard(
                        sc, step, opts, scratch_synced=cur_synced,
                        synced_total_elements=cur_total, process_groups=pg_i,
                    )
            continue

        _step_eager_param(
            opt, p, grads[i], exp_avgs[i], exp_avg_sqs[i] if opts.is_dual else None,
            ref_p, step, sc, opts, param_idx=global_idx, scratch_synced=cur_synced,
            scratch_total_elements=cur_total, defer_fsdp_post_ema=defer_cur, process_groups=pg_i,
        )
        if batch_post_ema and defer_cur and sc is not None:
            pending_ema.append((sc, p.numel(), pg_i))

    if pending_ema:
        totals = sync_fsdp_scratch_batch(
            [it[0] for it in pending_ema],
            [it[1] for it in pending_ema],
            process_groups_list=[it[2] for it in pending_ema],
        )
        for (sc_p, n_el_p, pg_p), tot_el in zip(pending_ema, totals):
            fallback_post_step_ema(
                sc_p, n_el_p, opts, scratch_synced=True, synced_total_elements=tot_el, process_groups=pg_p
            )
