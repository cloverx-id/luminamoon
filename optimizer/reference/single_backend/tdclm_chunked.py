from __future__ import annotations

import math

import torch

from ..utils.fsdp_sync import _sync_fsdp_scratch
from ..utils.math_sr import _sr_quantize
from .dual_step import SingleDualStepMixin


class SingleTDCLMChunkedMixin(SingleDualStepMixin):
    def _single_step_tdclm_element_chunked(
        self, p, grad, exp_avg, ref_p, step, sc,
        lr, beta1, eps, weight_decay, lam, cautious, clamp_min,
        use_sr, bound, bound_type, bound_ratio,
        fused_single_pass, ema_decay, fsdp_sync, seed, chunk_size,
        scratch_synced: bool = False,
        scratch_total_elements: float = 0.0,
        param_index: int = 0,
        defer_fsdp_post_ema: bool = False,
    ):
        n_elem = p.numel()
        has_master = ref_p is not None and ref_p is not p
        p_work = p if p.is_contiguous() else p.contiguous()
        g_work = grad if grad.is_contiguous() else grad.contiguous()
        m_work = exp_avg if exp_avg.is_contiguous() else exp_avg.contiguous()
        if has_master and ref_p is not None:
            ref_work = ref_p if ref_p.is_contiguous() else ref_p.contiguous()
        else:
            ref_work = p_work

        flat_p = p_work.view(-1)
        flat_g = g_work.view(-1)
        flat_m = m_work.view(-1)
        flat_ref = ref_work.view(-1)

        need_sr_m = use_sr and (exp_avg.dtype in (torch.bfloat16, torch.float16))
        need_sr_p = use_sr and (not has_master) and (p.dtype in (torch.bfloat16, torch.float16))
        gen_m, gen_p = self._get_step_generators(
            p.device, seed, step, param_index, need_sr_m, need_sr_p
        )

        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq
        effective_eps = max(eps, 1e-4) if p.dtype == torch.float16 else eps

        mask_sum_acc = 0.0
        u_sq_sum_acc = 0.0
        p_sq_sum_acc = 0.0
        needs_global_reduction = cautious or (bound and bound_type == "radial" and bound_ratio > 0.0)

        if (fused_single_pass and sc is not None and step > 1) or not needs_global_reduction:
            cautious_scale = sc[3].item() if (cautious and sc is not None and step > 1) else 1.0
            bound_scale = sc[4].item() if (bound and bound_type == "radial" and bound_ratio > 0.0 and sc is not None and step > 1) else 1.0

            for offset in range(0, n_elem, chunk_size):
                cur = min(chunk_size, n_elem - offset)
                p_c = flat_p[offset : offset + cur]
                g_c = flat_g[offset : offset + cur]
                m_c = flat_m[offset : offset + cur]
                ref_c = flat_ref[offset : offset + cur]

                g_clean = torch.nan_to_num(g_c, nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                m_f32 = m_c.float()

                clm_m = m_f32.mul(beta1_sq).add(g_clean, alpha=one_minus_beta1_sq)
                diff_f32 = (g_clean - m_f32).clamp(-1e18, 1e18)
                clm_defect = torch.abs(diff_f32).mul(lam)
                clm_env = torch.abs(m_f32).mul(beta1_sq).add(torch.abs(g_clean), alpha=one_minus_beta1_sq).add_(clm_defect).add_(effective_eps)
                u = clm_m / clm_env

                m_new = m_f32.mul(beta1).add(g_clean, alpha=1.0 - beta1)
                if m_c.dtype in (torch.bfloat16, torch.float16):
                    _sr_quantize(m_c, m_new, use_sr=use_sr, generator=gen_m)
                else:
                    m_c.copy_(m_new.to(dtype=m_c.dtype))

                if cautious:
                    m_mask = (u * g_clean > 0.0).float()
                    delta_theta = u * m_mask * cautious_scale
                    mask_sum_acc += m_mask.sum().item()
                else:
                    m_mask = None
                    delta_theta = u

                step_val = lr * delta_theta

                if bound and bound_ratio > 0.0:
                    if bound_type == "radial":
                        step_val = step_val * bound_scale
                    elif bound_type == "coordinate":
                        B_i = bound_ratio * torch.sqrt(ref_c.float().square() + 1e-4)
                        step_val = B_i * torch.tanh(step_val / B_i)

                if p.dtype == torch.float16:
                    b_safe = torch.clamp_min(torch.abs(ref_c.float()) * 0.01, 1.5e-4)
                    step_val = b_safe * torch.tanh(step_val / b_safe)

                if bound and bound_type == "radial" and bound_ratio > 0.0:
                    u_eff = (u * m_mask) if (cautious and m_mask is not None) else u
                    u_sq_sum_acc += torch.sum(u_eff ** 2).item()
                    p_sq_sum_acc += torch.sum(ref_c.float() ** 2).item()

                if has_master:
                    if weight_decay != 0.0:
                        ref_c.mul_(1.0 - lr * weight_decay)
                    ref_c.add_(step_val, alpha=-1.0)
                    if p_c.dtype == torch.float16:
                        p_c.copy_(ref_c.clamp(-65504.0, 65504.0).to(dtype=p_c.dtype))
                    elif p_c.dtype == torch.bfloat16:
                        p_c.copy_(ref_c.clamp(-3.3895313892515355e38, 3.3895313892515355e38).to(dtype=p_c.dtype))
                    else:
                        p_c.copy_(ref_c.to(dtype=p_c.dtype))
                else:
                    p_fp32 = p_c.float()
                    if weight_decay != 0.0:
                        p_fp32.mul_(1.0 - lr * weight_decay)
                    p_fp32.add_(step_val, alpha=-1.0)
                    _sr_quantize(p_c, p_fp32, use_sr=use_sr, generator=gen_p)

            if sc is not None:
                if needs_global_reduction:
                    if not (fsdp_sync and scratch_synced):
                        sc[:3].zero_()
                        if cautious:
                            sc[0] = mask_sum_acc
                        if bound and bound_type == "radial" and bound_ratio > 0.0:
                            sc[1] = u_sq_sum_acc
                            sc[2] = p_sq_sum_acc

                    if not defer_fsdp_post_ema:
                        total_elements = float(n_elem)
                        if fsdp_sync:
                            if not scratch_synced:
                                total_elements = _sync_fsdp_scratch(sc, n_elem)
                            else:
                                total_elements = scratch_total_elements if scratch_total_elements > 0.0 else float(n_elem)
                            mask_sum_acc = sc[0].item()
                            u_sq_sum_acc = sc[1].item()
                            p_sq_sum_acc = sc[2].item()

                        if cautious:
                            m_bar = max(min(mask_sum_acc / total_elements, 1.0), clamp_min)
                            target_cautious = 1.0 / m_bar
                            new_cautious = ema_decay * cautious_scale + (1.0 - ema_decay) * target_cautious
                            sc[3] = new_cautious

                        if bound and bound_type == "radial" and bound_ratio > 0.0:
                            c_eff = sc[3].item() if cautious else 1.0
                            norm_delta = math.sqrt(max(u_sq_sum_acc, 0.0)) * c_eff
                            norm_p = math.sqrt(max(p_sq_sum_acc, 0.0))
                            R = bound_ratio * max(norm_p, 1.0)
                            r = (lr * norm_delta) / R
                            target_bound = 1.0 if r < 1e-4 else math.tanh(r) / max(r, 1e-8)
                            new_bound = ema_decay * bound_scale + (1.0 - ema_decay) * target_bound
                            sc[4] = new_bound

                        sc[:3].zero_()
                        sc[5].zero_()
                else:
                    sc[:3].zero_()
                    sc[5].zero_()
        else:
            if fsdp_sync and scratch_synced and sc is not None:
                mask_sum_acc = sc[0].item()
                u_sq_sum_acc = sc[1].item()
                p_sq_sum_acc = sc[2].item()
                total_elements = scratch_total_elements if scratch_total_elements > 0.0 else float(n_elem)
            else:
                for offset in range(0, n_elem, chunk_size):
                    cur = min(chunk_size, n_elem - offset)
                    g_c = flat_g[offset : offset + cur]
                    m_c = flat_m[offset : offset + cur]
                    ref_c = flat_ref[offset : offset + cur]

                    g_clean = torch.nan_to_num(g_c, nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                    m_f32 = m_c.float()

                    clm_m = m_f32.mul(beta1_sq).add(g_clean, alpha=one_minus_beta1_sq)
                    diff_f32 = (g_clean - m_f32).clamp(-1e18, 1e18)
                    clm_defect = torch.abs(diff_f32).mul(lam)
                    clm_env = (
                        torch.abs(m_f32)
                        .mul(beta1_sq)
                        .add(torch.abs(g_clean), alpha=one_minus_beta1_sq)
                        .add_(clm_defect)
                        .add_(effective_eps)
                    )
                    u = clm_m / clm_env

                    if cautious:
                        m_mask = (u * g_clean > 0.0).float()
                        mask_sum_acc += m_mask.sum().item()
                        u_eff = u * m_mask
                    else:
                        u_eff = u

                    if bound and bound_type == "radial" and bound_ratio > 0.0:
                        u_sq_sum_acc += torch.sum(u_eff ** 2).item()
                        p_sq_sum_acc += torch.sum(ref_c.float() ** 2).item()

                total_elements = float(n_elem)
                if sc is not None:
                    sc[:3].zero_()
                    if cautious:
                        sc[0] = mask_sum_acc
                    if bound and bound_type == "radial" and bound_ratio > 0.0:
                        sc[1] = u_sq_sum_acc
                        sc[2] = p_sq_sum_acc

                    if fsdp_sync and not scratch_synced:
                        total_elements = _sync_fsdp_scratch(sc, n_elem)
                        mask_sum_acc = sc[0].item()
                        u_sq_sum_acc = sc[1].item()
                        p_sq_sum_acc = sc[2].item()

            cautious_scale = 1.0
            if cautious:
                m_bar = max(min(mask_sum_acc / total_elements, 1.0), clamp_min)
                cautious_scale = 1.0 / m_bar
                if sc is not None:
                    sc[3] = cautious_scale

            bound_scale = 1.0
            if bound and bound_type == "radial" and bound_ratio > 0.0:
                norm_p = math.sqrt(max(p_sq_sum_acc, 0.0))
                norm_u = math.sqrt(max(u_sq_sum_acc, 0.0)) * cautious_scale
                R = bound_ratio * max(norm_p, 1.0)
                r = (lr * norm_u) / R
                bound_scale = 1.0 if r < 1e-4 else math.tanh(r) / max(r, 1e-8)
                if sc is not None:
                    sc[4] = bound_scale

            for offset in range(0, n_elem, chunk_size):
                cur = min(chunk_size, n_elem - offset)
                p_c = flat_p[offset : offset + cur]
                g_c = flat_g[offset : offset + cur]
                m_c = flat_m[offset : offset + cur]
                ref_c = flat_ref[offset : offset + cur]

                g_clean = torch.nan_to_num(g_c, nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                m_f32 = m_c.float()

                clm_m = m_f32.mul(beta1_sq).add(g_clean, alpha=one_minus_beta1_sq)
                diff_f32 = (g_clean - m_f32).clamp(-1e18, 1e18)
                clm_defect = torch.abs(diff_f32).mul(lam)
                clm_env = (
                    torch.abs(m_f32)
                    .mul(beta1_sq)
                    .add(torch.abs(g_clean), alpha=one_minus_beta1_sq)
                    .add_(clm_defect)
                    .add_(effective_eps)
                )
                u = clm_m / clm_env

                m_new = m_f32.mul(beta1).add(g_clean, alpha=1.0 - beta1)
                if m_c.dtype in (torch.bfloat16, torch.float16):
                    _sr_quantize(m_c, m_new, use_sr=use_sr, generator=gen_m)
                else:
                    m_c.copy_(m_new.to(dtype=m_c.dtype))

                if cautious:
                    m_mask = (u * g_clean > 0.0).float()
                    delta_theta = u * m_mask * cautious_scale
                else:
                    delta_theta = u

                step_val = lr * delta_theta

                if bound and bound_ratio > 0.0:
                    if bound_type == "radial":
                        step_val = step_val * bound_scale
                    elif bound_type == "coordinate":
                        B_i = bound_ratio * torch.sqrt(ref_c.float().square() + 1e-4)
                        step_val = B_i * torch.tanh(step_val / B_i)

                if p.dtype == torch.float16:
                    b_safe = torch.clamp_min(torch.abs(ref_c.float()) * 0.01, 1.5e-4)
                    step_val = b_safe * torch.tanh(step_val / b_safe)

                if has_master:
                    if weight_decay != 0.0:
                        ref_c.mul_(1.0 - lr * weight_decay)
                    ref_c.add_(step_val, alpha=-1.0)
                    if p_c.dtype == torch.float16:
                        p_c.copy_(ref_c.clamp(-65504.0, 65504.0).to(dtype=p_c.dtype))
                    elif p_c.dtype == torch.bfloat16:
                        p_c.copy_(ref_c.clamp(-3.3895313892515355e38, 3.3895313892515355e38).to(dtype=p_c.dtype))
                    else:
                        p_c.copy_(ref_c.to(dtype=p_c.dtype))
                else:
                    p_fp32 = p_c.float()
                    if weight_decay != 0.0:
                        p_fp32.mul_(1.0 - lr * weight_decay)
                    p_fp32.add_(step_val, alpha=-1.0)
                    _sr_quantize(p_c, p_fp32, use_sr=use_sr, generator=gen_p)

            if sc is not None:
                sc[:3].zero_()
                sc[5].zero_()

        if not p.is_contiguous():
            p.copy_(p_work)
        if not exp_avg.is_contiguous():
            exp_avg.copy_(m_work)
        if has_master and ref_p is not None and not ref_p.is_contiguous():
            ref_p.copy_(ref_work)
