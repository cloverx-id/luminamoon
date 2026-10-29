from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from ..utils.fsdp_sync import (
    _fallback_post_step_ema,
    _sync_empty_fsdp_shard,
    _sync_fsdp_scratch,
    _sync_fsdp_scratch_batch,
)
from ..utils.math_sr import _compute_step_val, _sr_apply_step_val, _sr_quantize
from .tdclm_chunked import SingleTDCLMChunkedMixin


class SingleTDCLMStepMixin(SingleTDCLMChunkedMixin):
    def _single_step_tdclm(
        self, params, grads, exp_avgs, master_params, steps, scratches,
        lr, beta1, eps, weight_decay, lam, cautious, clamp_min,
        use_sr, bound=True, bound_type="coordinate", bound_ratio=0.05,
        fused_single_pass: bool = False, ema_decay: float = 0.8,
        fsdp_sync: bool = False, seed: int = 1337,
        scratch_synced: bool = False,
        scratch_total_elements: float = 0.0,
        orig_indices: Optional[List[int]] = None,
        defer_fsdp_post_ema: bool = False,
    ):
        bound_mode = 0
        if bound and bound_ratio > 0.0:
            bound_mode = 1 if bound_type == "radial" else 2

        ELEM_CHUNK_SIZE = 2_097_152

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
                batched_scratch_totals = self._precompute_fsdp_pass1_single_tdclm(
                    pass1_indices, params, grads, exp_avgs, master_params, scratches,
                    beta1, eps, lam, bound_mode, cautious, ELEM_CHUNK_SIZE
                )
        pending_ema_single: List[Tuple[Tensor, int]] = []

        for i in range(len(params)):
            p, grad, exp_avg, step_raw = params[i], grads[i], exp_avgs[i], steps[i]
            step = int(step_raw.item() if isinstance(step_raw, torch.Tensor) else step_raw)
            global_param_idx = orig_indices[i] if orig_indices is not None else i
            param_has_master = (master_params is not None) and (master_params[i] is not p)
            ref_p = master_params[i] if (param_has_master and master_params is not None) else p
            sc = scratches[i] if scratches is not None else None
            n_elem = p.numel()
            cur_scratch_synced = scratch_synced or (i in batched_scratch_totals)
            cur_scratch_total = batched_scratch_totals[i] if (i in batched_scratch_totals) else scratch_total_elements
            defer_cur_ema = defer_fsdp_post_ema or (batch_post_ema and fused_single_pass and step > 1)

            if n_elem == 0:
                if fsdp_sync and sc is not None:
                    if not cur_scratch_synced:
                        sc[:3].zero_()
                    sc[5].zero_()
                    if defer_cur_ema and needs_ema_sync:
                        if batch_post_ema:
                            pending_ema_single.append((sc, 0))
                    else:
                        _sync_empty_fsdp_shard(
                            sc, step, lr, clamp_min, ema_decay,
                            bound_mode, bound_ratio, cautious, fused_single_pass,
                            scratch_synced=cur_scratch_synced,
                            synced_total_elements=cur_scratch_total,
                        )
                continue

            if n_elem > ELEM_CHUNK_SIZE:
                self._single_step_tdclm_element_chunked(
                    p, grad, exp_avg, ref_p, step, sc,
                    lr, beta1, eps, weight_decay, lam, cautious, clamp_min,
                    use_sr, bound, bound_type, bound_ratio,
                    fused_single_pass, ema_decay, fsdp_sync, seed, ELEM_CHUNK_SIZE,
                    scratch_synced=cur_scratch_synced,
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

            beta1_sq = beta1 * beta1
            one_minus_beta1_sq = 1.0 - beta1_sq
            effective_eps = max(eps, 1e-4) if p.dtype == torch.float16 else eps

            clm_m = m_f32.mul(beta1_sq).add(g_f32, alpha=one_minus_beta1_sq)

            diff_f32 = (g_f32 - m_f32).clamp(-1e18, 1e18)

            clm_defect = torch.abs(diff_f32).mul(lam)
            clm_env = (
                torch.abs(m_f32)
                .mul(beta1_sq)
                .add(torch.abs(g_f32), alpha=one_minus_beta1_sq)
                .add_(clm_defect)
                .add_(effective_eps)
            )
            u = clm_m / clm_env
            del clm_m, clm_env, clm_defect, diff_f32

            m_new_f32 = m_f32.mul(beta1).add(g_f32, alpha=1.0 - beta1)
            if exp_avg.dtype in (torch.bfloat16, torch.float16):
                _sr_quantize(exp_avg, m_new_f32, use_sr=use_sr, generator=gen_m)
            else:
                exp_avg.copy_(m_new_f32.to(dtype=exp_avg.dtype))
            del m_new_f32, m_f32

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
                mask: Optional[Tensor] = None
                if sc is not None and not (fsdp_sync and cur_scratch_synced):
                    sc[:3].zero_()

                if cautious:
                    mask = (u * g_f32 > 0).float()
                    if sc is not None and mask is not None and not (fsdp_sync and cur_scratch_synced):
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
                else:
                    update = u

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
            del step_val, grad_clean, g_f32

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

    def _dispatch_single(
        self, params, grads, exp_avgs, exp_avg_sqs, master_params, steps, scratches,
        lr, beta1, beta2, eps, weight_decay, tau, lam,
        cautious, clamp_min, buf_count, use_sr, bound, bound_type, bound_ratio,
        fused_single_pass, ema_decay, fsdp_sync: bool = False, seed: int = 1337,
        orig_indices: Optional[List[int]] = None,
    ):
        primary_dev = params[0].device
        is_single_device = all(p.device == primary_dev for p in params)

        if is_single_device:
            if buf_count == 2:
                self._single_step_dual(
                    params, grads, exp_avgs, exp_avg_sqs, master_params, steps, scratches,
                    lr, beta1, beta2, eps, weight_decay, tau, cautious, clamp_min,
                    use_sr, bound, bound_type, bound_ratio,
                    fused_single_pass, ema_decay, fsdp_sync, seed,
                    orig_indices=orig_indices
                )
            else:
                self._single_step_tdclm(
                    params, grads, exp_avgs, master_params, steps, scratches,
                    lr, beta1, eps, weight_decay, lam, cautious, clamp_min,
                    use_sr, bound, bound_type, bound_ratio,
                    fused_single_pass, ema_decay, fsdp_sync, seed,
                    orig_indices=orig_indices
                )
            return

        device_groups: Dict[torch.device, List[int]] = {}
        for idx, p in enumerate(params):
            dev = p.device
            if dev not in device_groups:
                device_groups[dev] = []
            device_groups[dev].append(idx)

        for dev, indices in device_groups.items():
            sub_p = [params[i] for i in indices]
            sub_g = [grads[i] for i in indices]
            sub_m = [exp_avgs[i] for i in indices]
            sub_st = [steps[i] for i in indices]
            sub_sc = [scratches[i] for i in indices]
            sub_mp = [master_params[i] for i in indices] if master_params is not None else None
            mapped_orig = [orig_indices[i] for i in indices] if orig_indices is not None else indices

            if buf_count == 2:
                sub_v = [exp_avg_sqs[i] for i in indices]
                self._single_step_dual(
                    sub_p, sub_g, sub_m, sub_v, sub_mp, sub_st, sub_sc,
                    lr, beta1, beta2, eps, weight_decay, tau, cautious, clamp_min,
                    use_sr, bound, bound_type, bound_ratio,
                    fused_single_pass, ema_decay, fsdp_sync, seed,
                    orig_indices=mapped_orig
                )
            else:
                self._single_step_tdclm(
                    sub_p, sub_g, sub_m, sub_mp, sub_st, sub_sc,
                    lr, beta1, eps, weight_decay, lam, cautious, clamp_min,
                    use_sr, bound, bound_type, bound_ratio,
                    fused_single_pass, ema_decay, fsdp_sync, seed,
                    orig_indices=mapped_orig
                )
