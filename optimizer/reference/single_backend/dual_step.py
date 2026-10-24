from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from ..utils.fsdp_sync import (
    _fallback_post_step_ema,
    _sync_empty_fsdp_shard,
    _sync_fsdp_diff_only,
    _sync_fsdp_scratch,
    _sync_fsdp_scratch_batch,
)
from ..utils.math_sr import _compute_step_val, _sr_apply_step_val, _sr_quantize
from .dual_chunked import SingleDualChunkedMixin


class SingleDualStepMixin(SingleDualChunkedMixin):
    def _single_step_dual(
        self, params, grads, exp_avgs, exp_avg_sqs, master_params, steps, scratches,
        lr, beta1, beta2, eps, weight_decay, tau, cautious, clamp_min,
        use_sr, bound=True, bound_type="coordinate", bound_ratio=0.05,
        fused_single_pass=False, ema_decay=0.8, fsdp_sync: bool = False, seed: int = 1337,
        centered_defect: bool = False,
        diff_synced: bool = False, scratch_synced: bool = False,
        diff_total_elements: float = 0.0, scratch_total_elements: float = 0.0,
        orig_indices: Optional[List[int]] = None,
        defer_fsdp_post_ema: bool = False,
    ):
        bound_mode = 0
        if bound and bound_ratio > 0.0:
            bound_mode = 1 if bound_type == "radial" else 2

        base_step = int(steps[0].item() if isinstance(steps[0], torch.Tensor) else steps[0])
        safe_base_step = max(base_step, 1)
        bc1_nes_base = 1.0 - (beta1 ** (safe_base_step + 1))
        bc2_base = max(1.0 - (beta2 ** safe_base_step), 1e-15)
        c1_base = (bc1_nes_base * tau) / math.sqrt(bc2_base)
        c2_default = eps * bc1_nes_base * tau
        c2_fp16 = max(eps, 1e-4) * bc1_nes_base * tau

        ELEM_CHUNK_SIZE = 2_097_152

        batched_diff_totals: Optional[List[float]] = None
        if centered_defect and fsdp_sync and not diff_synced and scratches is not None and len(params) > 1:
            batched_diff_totals = self._precompute_fsdp_diff_single(
                params, grads, exp_avgs, scratches, ELEM_CHUNK_SIZE
            )

        needs_ema_sync = cautious or bound_mode == 1
        batch_post_ema = bool(
            fsdp_sync and not scratch_synced and not defer_fsdp_post_ema
            and scratches is not None and len(params) > 1 and needs_ema_sync
        )
        batched_scratch_totals: Dict[int, float] = {}
        if batch_post_ema:
            pass1_indices = [
                idx for idx, st in enumerate(steps)
                if not (fused_single_pass and int(st.item() if isinstance(st, torch.Tensor) else st) > 1)
            ]
            if scratches is not None and len(pass1_indices) > 1:
                batched_scratch_totals = self._precompute_fsdp_pass1_single_dual(
                    pass1_indices, params, grads, exp_avgs, exp_avg_sqs, master_params,
                    steps, scratches, beta1, beta2, eps, tau, base_step, c1_base, c2_default, c2_fp16,
                    bound_mode, cautious, centered_defect, batched_diff_totals, ELEM_CHUNK_SIZE
                )
        pending_ema_single: List[Tuple[Tensor, int]] = []

        for i in range(len(params)):
            p, grad, exp_avg, exp_avg_sq, step_raw = params[i], grads[i], exp_avgs[i], exp_avg_sqs[i], steps[i]
            step = int(step_raw.item() if isinstance(step_raw, torch.Tensor) else step_raw)
            global_param_idx = orig_indices[i] if orig_indices is not None else i
            param_has_master = (master_params is not None) and (master_params[i] is not p)
            ref_p = master_params[i] if (param_has_master and master_params is not None) else p
            sc = scratches[i] if scratches is not None else None
            n_elem = p.numel()
            cur_diff_synced = diff_synced or (batched_diff_totals is not None)
            cur_diff_total = batched_diff_totals[i] if batched_diff_totals is not None else diff_total_elements
            cur_scratch_synced = scratch_synced or (i in batched_scratch_totals)
            cur_scratch_total = batched_scratch_totals[i] if (i in batched_scratch_totals) else scratch_total_elements
            defer_cur_ema = defer_fsdp_post_ema or (batch_post_ema and fused_single_pass and step > 1)

            if n_elem == 0:
                if fsdp_sync and sc is not None:
                    if not cur_scratch_synced:
                        sc[:3].zero_()
                    if not cur_diff_synced:
                        sc[5].zero_()
                        if centered_defect:
                            _sync_fsdp_diff_only(sc, 0)
                    if defer_cur_ema and needs_ema_sync:
                        if batch_post_ema:
                            pending_ema_single.append((sc, 0))
                    else:
                        _sync_empty_fsdp_shard(
                            sc, step, lr, clamp_min, ema_decay,
                            bound_mode, bound_ratio, cautious, centered_defect, fused_single_pass,
                            diff_synced=True, scratch_synced=cur_scratch_synced,
                            synced_total_elements=cur_scratch_total,
                        )
                continue

            safe_step = max(step, 1)
            if step == base_step:
                c1 = c1_base
                c2 = c2_fp16 if p.dtype == torch.float16 else c2_default
            else:
                effective_eps = max(eps, 1e-4) if p.dtype == torch.float16 else eps
                bc1_nes = 1.0 - (beta1 ** (safe_step + 1))
                bc2 = max(1.0 - (beta2 ** safe_step), 1e-15)
                c1 = (bc1_nes * tau) / math.sqrt(bc2)
                c2 = effective_eps * bc1_nes * tau

            if n_elem > ELEM_CHUNK_SIZE:
                self._single_step_dual_element_chunked(
                    p, grad, exp_avg, exp_avg_sq, ref_p, step, sc,
                    c1, c2, lr, beta1, beta2, weight_decay, cautious, clamp_min,
                    use_sr, bound, bound_type, bound_ratio,
                    fused_single_pass, ema_decay, fsdp_sync, seed, ELEM_CHUNK_SIZE,
                    centered_defect,
                    diff_synced=cur_diff_synced, scratch_synced=cur_scratch_synced,
                    diff_total_elements=cur_diff_total,
                    scratch_total_elements=cur_scratch_total,
                    param_index=global_param_idx,
                    defer_fsdp_post_ema=defer_cur_ema,
                )
                if batch_post_ema and defer_cur_ema and sc is not None:
                    pending_ema_single.append((sc, n_elem))
                continue

            need_sr_m = use_sr and (exp_avg.dtype in (torch.bfloat16, torch.float16))
            need_sr_p = use_sr and (not param_has_master) and (p.dtype in (torch.bfloat16, torch.float16))
            gen_m, gen_p = self._get_step_generators(
                p.device, seed, step, global_param_idx, need_sr_m, need_sr_p
            )

            grad_clean = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            g_f32 = grad_clean.float().clamp(-1e18, 1e18)
            m_f32 = exp_avg.float()
            v_f32 = exp_avg_sq.float()

            m_new = m_f32.mul(beta1).add(g_f32, alpha=1.0 - beta1)
            clm_m = m_new.mul(beta1).add(g_f32, alpha=1.0 - beta1)
            diff = (g_f32 - m_new).clamp(-1e18, 1e18)
            if centered_defect:
                if fsdp_sync and sc is not None:
                    if not cur_diff_synced:
                        sc[5].copy_((g_f32 - m_f32).clamp(-1e18, 1e18).sum().to(torch.float32))
                        tot_cd = _sync_fsdp_diff_only(sc, n_elem)
                    else:
                        tot_cd = cur_diff_total if cur_diff_total > 0.0 else float(n_elem)
                    if tot_cd > 1.0:
                        diff = (diff - (sc[5] / tot_cd) * beta1).clamp(-1e18, 1e18)
                    sc[5].zero_()
                elif diff.numel() > 1:
                    diff_mean = (g_f32 - m_f32).clamp(-1e18, 1e18).mean() * beta1
                    diff = (diff - diff_mean).clamp(-1e18, 1e18)

            v_new = v_f32.mul(beta2).addcmul(diff, diff, value=1.0 - beta2)
            if p.dtype == torch.float16 or exp_avg.dtype == torch.float16 or exp_avg_sq.dtype == torch.float16:
                v_new = v_new.clamp(0.0, 65000.0)

            sigma = v_new.clamp_min(0.0).sqrt().mul(c1).add(c2)
            u = torch.tanh(clm_m / sigma.clamp_min(1e-12))

            if exp_avg.dtype in (torch.bfloat16, torch.float16):
                _sr_quantize(exp_avg, m_new, use_sr=use_sr, generator=gen_m)
            else:
                exp_avg.copy_(m_new.to(dtype=exp_avg.dtype))
            exp_avg_sq.copy_(v_new.to(dtype=exp_avg_sq.dtype))

            if fused_single_pass and sc is not None and step > 1:
                cautious_scale = sc[3].clone()
                bound_scale = sc[4].clone()

                if cautious:
                    m_mask = (u * g_f32 > 0).float()
                    delta_theta = u * m_mask * cautious_scale
                else:
                    m_mask = None
                    delta_theta = u

                step_val = _compute_step_val(
                    delta_theta, ref_p, lr, bound, bound_type, bound_ratio,
                    target_dtype=p.dtype,
                    radial_scale=bound_scale if bound_mode == 1 else None,
                )

                if not (fsdp_sync and cur_scratch_synced):
                    sc[:3].zero_()
                    if cautious and m_mask is not None:
                        sc[0].copy_(m_mask.sum().to(torch.float32))
                    if bound_mode == 1:
                        u_eff = (u * m_mask) if (cautious and m_mask is not None) else u
                        sc[1].copy_(torch.sum(u_eff.float() ** 2))
                        sc[2].copy_(torch.sum(ref_p.float() ** 2))

                if defer_cur_ema:
                    if batch_post_ema:
                        pending_ema_single.append((sc, n_elem))
                else:
                    total_elements = float(n_elem)
                    if fsdp_sync and needs_ema_sync:
                        if not cur_scratch_synced:
                            total_elements = _sync_fsdp_scratch(sc, n_elem)
                        else:
                            total_elements = cur_scratch_total if cur_scratch_total > 0.0 else float(n_elem)

                    if cautious and m_mask is not None:
                        m_bar = (sc[0] / total_elements).clamp_(min=clamp_min, max=1.0)
                        target_cautious = 1.0 / m_bar
                        new_cautious = ema_decay * cautious_scale + (1.0 - ema_decay) * target_cautious
                        sc[3].copy_(new_cautious)

                    if bound_mode == 1:
                        c_eff = sc[3] if cautious else 1.0
                        norm_delta = torch.sqrt(torch.clamp_min(sc[1], 0.0)) * c_eff
                        norm_p = torch.sqrt(torch.clamp_min(sc[2], 0.0))
                        R = bound_ratio * torch.clamp_min(norm_p, 1.0)
                        r = (lr * norm_delta) / R
                        target_bound = torch.where(r < 1e-4, torch.ones_like(r), torch.tanh(r) / torch.clamp_min(r, 1e-8))
                        new_bound = ema_decay * bound_scale + (1.0 - ema_decay) * target_bound
                        sc[4].copy_(new_bound)

                    sc[:3].zero_()
                    sc[5].zero_()
            else:
                update = u
                mask: Optional[Tensor] = None
                if sc is not None and not (fsdp_sync and cur_scratch_synced):
                    sc[:3].zero_()

                if cautious:
                    mask = (u * g_f32 > 0).float()
                    if sc is not None and not (fsdp_sync and cur_scratch_synced):
                        sc[0].copy_(mask.sum().to(torch.float32))

                u_sq_sum: Optional[Tensor] = None
                p_sq_sum: Optional[Tensor] = None
                if bound_mode == 1 and not (fsdp_sync and cur_scratch_synced and sc is not None):
                    u_eff = (u * mask) if (cautious and mask is not None) else u
                    u_sq_sum = torch.sum(u_eff.float() ** 2)
                    p_sq_sum = torch.sum(ref_p.float() ** 2)
                    if sc is not None:
                        sc[1].copy_(u_sq_sum)
                        sc[2].copy_(p_sq_sum)

                total_elements = float(n_elem)
                if fsdp_sync and sc is not None and needs_ema_sync:
                    if not cur_scratch_synced:
                        total_elements = _sync_fsdp_scratch(sc, n_elem)
                    else:
                        total_elements = cur_scratch_total if cur_scratch_total > 0.0 else float(n_elem)

                cautious_scale_val: Optional[Tensor] = None
                if cautious and mask is not None:
                    if fsdp_sync and sc is not None:
                        m_bar = (sc[0] / total_elements).clamp_(min=clamp_min, max=1.0)
                    else:
                        m_bar = mask.mean().clamp_(min=clamp_min, max=1.0)
                    cautious_scale_val = 1.0 / m_bar
                    update = u.mul(mask).mul(cautious_scale_val)
                    if sc is not None:
                        sc[3].copy_(cautious_scale_val.to(torch.float32))

                b_scale: Optional[Tensor] = None
                if bound_mode == 1:
                    if sc is not None:
                        c_scale = sc[3] if cautious else 1.0
                        norm_u = torch.sqrt(torch.clamp_min(sc[1], 0.0)) * c_scale
                        norm_p = torch.sqrt(torch.clamp_min(sc[2], 0.0))
                    else:
                        assert u_sq_sum is not None and p_sq_sum is not None
                        c_scale_f = cautious_scale_val.float() if (cautious and cautious_scale_val is not None) else 1.0
                        norm_u = torch.sqrt(torch.clamp_min(u_sq_sum, 0.0)) * c_scale_f
                        norm_p = torch.sqrt(torch.clamp_min(p_sq_sum, 0.0))
                    R = bound_ratio * torch.clamp_min(norm_p, 1.0)
                    r = (lr * norm_u) / R
                    b_scale = torch.where(r < 1e-4, torch.ones_like(r), torch.tanh(r) / torch.clamp_min(r, 1e-8))
                    if sc is not None:
                        sc[4].copy_(b_scale.to(torch.float32))

                step_val = _compute_step_val(
                    update, ref_p, lr, bound, bound_type, bound_ratio,
                    target_dtype=p.dtype, radial_scale=b_scale,
                )
                if sc is not None:
                    sc[:3].zero_()
                    sc[5].zero_()

            if param_has_master and master_params is not None:
                mp = master_params[i]
                if weight_decay != 0.0:
                    mp.mul_(1.0 - lr * weight_decay)
                mp.add_(step_val, alpha=-1.0)
                if p.dtype == torch.float16:
                    p.copy_(mp.clamp(-65504.0, 65504.0).to(dtype=p.dtype))
                elif p.dtype == torch.bfloat16:
                    p.copy_(mp.clamp(-3.3895313892515355e38, 3.3895313892515355e38).to(dtype=p.dtype))
                else:
                    p.copy_(mp.to(dtype=p.dtype))
            else:
                _sr_apply_step_val(p, step_val, lr, weight_decay, use_sr, generator=gen_p)

        if pending_ema_single:
            sc_list = [item[0] for item in pending_ema_single]
            numels = [item[1] for item in pending_ema_single]
            totals = _sync_fsdp_scratch_batch(sc_list, numels)
            for (sc_p, n_el_p), tot_el in zip(pending_ema_single, totals):
                _fallback_post_step_ema(
                    sc_p, n_el_p, lr, clamp_min, ema_decay,
                    bound_mode, bound_ratio, cautious, True, True,
                    synced_total_elements=tot_el
                )
