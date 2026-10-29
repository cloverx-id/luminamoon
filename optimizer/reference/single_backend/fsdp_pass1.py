from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
from torch import Tensor

from ..core.state_manager import LuminaVStateMixin
from ..utils.fsdp_sync import _sync_fsdp_scratch_batch


class SingleFSDPPass1Mixin(LuminaVStateMixin):
    @staticmethod
    def _compute_pass1_single_dual_one(
        p: Tensor,
        grad: Tensor,
        exp_avg: Tensor,
        exp_avg_sq: Tensor,
        ref_p: Tensor,
        sc: Tensor,
        beta1: float,
        beta2: float,
        c1: float,
        c2: float,
        bound_mode: int,
        cautious: bool,
        chunk_size: int = 2_097_152,
    ) -> None:
        sc[:3].zero_()
        n_el = p.numel()
        if n_el == 0:
            return
        clamp_v = (p.dtype == torch.float16) or (exp_avg.dtype == torch.float16) or (exp_avg_sq.dtype == torch.float16)
        if n_el <= chunk_size:
            g_clean = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
            m_f32 = exp_avg.float()
            v_f32 = exp_avg_sq.float()
            m_new = m_f32.mul(beta1).add(g_clean, alpha=1.0 - beta1)
            clm_m = m_new.mul(beta1).add(g_clean, alpha=1.0 - beta1)
            diff = (g_clean - m_new).clamp(-1e18, 1e18)
            v_new = v_f32.mul(beta2).addcmul(diff, diff, value=1.0 - beta2)
            if clamp_v:
                v_new = v_new.clamp(0.0, 65000.0)
            sigma = v_new.clamp_min(0.0).sqrt().mul(c1).add(c2)
            u = torch.tanh(clm_m / sigma.clamp_min(1e-12))
            mask = (u * g_clean > 0.0).float() if cautious else None
            if cautious and mask is not None:
                sc[0].copy_(mask.sum().to(torch.float32))
            if bound_mode == 1:
                u_eff = (u * mask) if (cautious and mask is not None) else u
                sc[1].copy_(torch.sum(u_eff.float() ** 2))
                sc[2].copy_(torch.sum(ref_p.float() ** 2))
        else:
            flat_g = (grad if grad.is_contiguous() else grad.contiguous()).view(-1)
            flat_m = (exp_avg if exp_avg.is_contiguous() else exp_avg.contiguous()).view(-1)
            flat_v = (exp_avg_sq if exp_avg_sq.is_contiguous() else exp_avg_sq.contiguous()).view(-1)
            flat_ref = (ref_p if ref_p.is_contiguous() else ref_p.contiguous()).view(-1)
            mask_sum_acc = 0.0
            u_sq_sum_acc = 0.0
            p_sq_sum_acc = 0.0
            for offset in range(0, n_el, chunk_size):
                cur = min(chunk_size, n_el - offset)
                g_c = torch.nan_to_num(flat_g[offset : offset + cur], nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                m_f32 = flat_m[offset : offset + cur].float()
                v_f32 = flat_v[offset : offset + cur].float()
                m_new = m_f32.mul(beta1).add(g_c, alpha=1.0 - beta1)
                clm_m = m_new.mul(beta1).add(g_c, alpha=1.0 - beta1)
                diff = (g_c - m_new).clamp(-1e18, 1e18)
                v_new = v_f32.mul(beta2).addcmul(diff, diff, value=1.0 - beta2)
                if clamp_v:
                    v_new = v_new.clamp(0.0, 65000.0)
                sigma = v_new.clamp_min(0.0).sqrt().mul(c1).add(c2)
                u = torch.tanh(clm_m / sigma.clamp_min(1e-12))
                if cautious:
                    m_mask = (u * g_c > 0.0).float()
                    mask_sum_acc += m_mask.sum().item()
                    u_eff = u * m_mask
                else:
                    u_eff = u
                if bound_mode == 1:
                    u_sq_sum_acc += torch.sum(u_eff ** 2).item()
                    p_sq_sum_acc += torch.sum(flat_ref[offset : offset + cur].float() ** 2).item()
            if cautious:
                sc[0] = mask_sum_acc
            if bound_mode == 1:
                sc[1] = u_sq_sum_acc
                sc[2] = p_sq_sum_acc

    @staticmethod
    def _compute_pass1_single_tdclm_one(
        p: Tensor,
        grad: Tensor,
        exp_avg: Tensor,
        ref_p: Tensor,
        sc: Tensor,
        beta1: float,
        eps: float,
        lam: float,
        bound_mode: int,
        cautious: bool,
        chunk_size: int = 2_097_152,
    ) -> None:
        sc[:3].zero_()
        n_el = p.numel()
        if n_el == 0:
            return
        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq
        effective_eps = max(eps, 1e-4) if p.dtype == torch.float16 else eps
        if n_el <= chunk_size:
            g_clean = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
            m_f32 = exp_avg.float()
            clm_m = m_f32.mul(beta1_sq).add(g_clean, alpha=one_minus_beta1_sq)
            diff_f32 = (g_clean - m_f32).clamp(-1e18, 1e18)
            clm_env = (
                torch.abs(m_f32)
                .mul(beta1_sq)
                .add(torch.abs(g_clean), alpha=one_minus_beta1_sq)
                .add_(torch.abs(diff_f32).mul(lam))
                .add_(effective_eps)
            )
            u = clm_m / clm_env
            mask = (u * g_clean > 0.0).float() if cautious else None
            if cautious and mask is not None:
                sc[0].copy_(mask.sum().to(torch.float32))
            if bound_mode == 1:
                u_eff = (u * mask) if (cautious and mask is not None) else u
                sc[1].copy_(torch.sum(u_eff.float() ** 2))
                sc[2].copy_(torch.sum(ref_p.float() ** 2))
        else:
            flat_g = (grad if grad.is_contiguous() else grad.contiguous()).view(-1)
            flat_m = (exp_avg if exp_avg.is_contiguous() else exp_avg.contiguous()).view(-1)
            flat_ref = (ref_p if ref_p.is_contiguous() else ref_p.contiguous()).view(-1)
            mask_sum_acc = 0.0
            u_sq_sum_acc = 0.0
            p_sq_sum_acc = 0.0
            for offset in range(0, n_el, chunk_size):
                cur = min(chunk_size, n_el - offset)
                g_c = torch.nan_to_num(flat_g[offset : offset + cur], nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                m_f32 = flat_m[offset : offset + cur].float()
                clm_m = m_f32.mul(beta1_sq).add(g_c, alpha=one_minus_beta1_sq)
                diff_f32 = (g_c - m_f32).clamp(-1e18, 1e18)
                clm_env = (
                    torch.abs(m_f32)
                    .mul(beta1_sq)
                    .add(torch.abs(g_c), alpha=one_minus_beta1_sq)
                    .add_(torch.abs(diff_f32).mul(lam))
                    .add_(effective_eps)
                )
                u = clm_m / clm_env
                if cautious:
                    m_mask = (u * g_c > 0.0).float()
                    mask_sum_acc += m_mask.sum().item()
                    u_eff = u * m_mask
                else:
                    u_eff = u
                if bound_mode == 1:
                    u_sq_sum_acc += torch.sum(u_eff ** 2).item()
                    p_sq_sum_acc += torch.sum(flat_ref[offset : offset + cur].float() ** 2).item()
            if cautious:
                sc[0] = mask_sum_acc
            if bound_mode == 1:
                sc[1] = u_sq_sum_acc
                sc[2] = p_sq_sum_acc

    @classmethod
    def _precompute_fsdp_pass1_single_dual(
        cls,
        pass1_indices: List[int],
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        master_params: Optional[List[Tensor]],
        steps: List[int],
        scratches: List[Tensor],
        beta1: float,
        beta2: float,
        eps: float,
        tau: float,
        base_step: int,
        c1_base: float,
        c2_default: float,
        c2_fp16: float,
        bound_mode: int,
        cautious: bool,
        chunk_size: int = 2_097_152,
    ) -> Dict[int, float]:
        has_master = master_params is not None
        for idx in pass1_indices:
            p = params[idx]
            sc = scratches[idx]
            sc[:3].zero_()
            n_el = p.numel()
            if n_el == 0:
                continue
            step_raw = steps[idx]
            step = int(step_raw.item() if isinstance(step_raw, torch.Tensor) else step_raw)
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
            param_has_master = has_master and master_params is not None and (master_params[idx] is not p)
            ref_p = master_params[idx] if (param_has_master and master_params is not None) else p
            cls._compute_pass1_single_dual_one(
                p, grads[idx], exp_avgs[idx], exp_avg_sqs[idx], ref_p, sc,
                beta1, beta2, c1, c2, bound_mode, cautious, chunk_size
            )
        synced_totals = _sync_fsdp_scratch_batch(
            [scratches[idx] for idx in pass1_indices],
            [params[idx].numel() for idx in pass1_indices],
        )
        return {idx: tot for idx, tot in zip(pass1_indices, synced_totals)}

    @classmethod
    def _precompute_fsdp_pass1_single_tdclm(
        cls,
        pass1_indices: List[int],
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        master_params: Optional[List[Tensor]],
        scratches: List[Tensor],
        beta1: float,
        eps: float,
        lam: float,
        bound_mode: int,
        cautious: bool,
        chunk_size: int = 2_097_152,
    ) -> Dict[int, float]:
        has_master = master_params is not None
        for idx in pass1_indices:
            p = params[idx]
            sc = scratches[idx]
            sc[:3].zero_()
            n_el = p.numel()
            if n_el == 0:
                continue
            param_has_master = has_master and master_params is not None and (master_params[idx] is not p)
            ref_p = master_params[idx] if (param_has_master and master_params is not None) else p
            cls._compute_pass1_single_tdclm_one(
                p, grads[idx], exp_avgs[idx], ref_p, sc,
                beta1, eps, lam, bound_mode, cautious, chunk_size
            )
        synced_totals = _sync_fsdp_scratch_batch(
            [scratches[idx] for idx in pass1_indices],
            [params[idx].numel() for idx in pass1_indices],
        )
        return {idx: tot for idx, tot in zip(pass1_indices, synced_totals)}
