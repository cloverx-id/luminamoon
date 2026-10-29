from __future__ import annotations

import contextlib
import logging
import math
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor

from ..kernels import (
    _lumina_dual_fused_single_pass_kernel,
    _lumina_dual_pass1_kernel,
    _lumina_dual_pass2_kernel,
    _lumina_dual_single_pass_kernel,
    _lumina_post_step_ema_kernel,
)
from ..utils.fsdp_sync import (
    _fallback_post_step_ema,
    _sync_empty_fsdp_shard,
    _sync_fsdp_scratch,
    _sync_fsdp_scratch_batch,
)
from .base import TritonBaseMixin

logger = logging.getLogger("LuminaV")


class TritonDualMixin(TritonBaseMixin):
    def _precompute_fsdp_pass1_triton_dual(
        self,
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
    ) -> Dict[int, float]:
        has_master = master_params is not None
        for idx in pass1_indices:
            p = params[idx]
            sc = scratches[idx]
            sc[:3].zero_()
            n_el = p.numel()
            if n_el == 0:
                continue
            grad = grads[idx]
            exp_avg = exp_avgs[idx]
            exp_avg_sq = exp_avg_sqs[idx]
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
            mp_ref = master_params[idx] if (param_has_master and master_params is not None) else None
            try:
                (
                    n_elements, b_size, n_warps, c_grid, dt_mode, s_dt_mode, _,
                    p_contig, grad_contig, exp_avg_contig, exp_avg_sq_contig, mp_contig
                ) = self._prepare_triton_buffers(p, grad, exp_avg, exp_avg_sq, mp_ref, param_has_master)
                _lumina_dual_pass1_kernel[c_grid](
                    p_contig, mp_contig, grad_contig, exp_avg_contig, exp_avg_sq_contig,
                    sc,
                    n_elements, beta1, beta2, c1, c2,
                    bound_mode, dt_mode, s_dt_mode,
                    cautious, param_has_master,
                    BLOCK_SIZE=b_size, num_warps=n_warps
                )
            except Exception:
                ref_p = mp_ref if (param_has_master and mp_ref is not None) else p
                self._compute_pass1_single_dual_one(
                    p, grad, exp_avg, exp_avg_sq, ref_p, sc,
                    beta1, beta2, c1, c2, bound_mode, cautious
                )
        synced_totals = _sync_fsdp_scratch_batch(
            [scratches[idx] for idx in pass1_indices],
            [params[idx].numel() for idx in pass1_indices],
        )
        return {idx: tot for idx, tot in zip(pass1_indices, synced_totals)}

    def _triton_step_dual(
        self, params, grads, exp_avgs, exp_avg_sqs, master_params, steps, scratches,
        lr, beta1, beta2, eps, weight_decay, tau, cautious, clamp_min,
        use_sr, bound=True, bound_type="coordinate", bound_ratio=0.05,
        fused_single_pass=False, ema_decay=0.8, fsdp_sync: bool = False, seed: int = 1337,
        orig_indices: Optional[List[int]] = None,
        completed_indices: Optional[Set[int]] = None
    ):
        bound_mode = 0
        if bound and bound_ratio > 0.0:
            bound_mode = 1 if bound_type == "radial" else 2

        has_master = master_params is not None

        base_step = int(steps[0].item() if isinstance(steps[0], torch.Tensor) else steps[0])
        safe_base_step = max(base_step, 1)
        bc1_nes_base = 1.0 - (beta1 ** (safe_base_step + 1))
        bc2_base = max(1.0 - (beta2 ** safe_base_step), 1e-15)
        c1_base = (bc1_nes_base * tau) / math.sqrt(bc2_base)
        c2_default = eps * bc1_nes_base * tau
        c2_fp16 = max(eps, 1e-4) * bc1_nes_base * tau

        pending_ema: List[Tuple[Tensor, int, bool]] = []
        needs_ema_sync = cautious or bound_mode == 1

        batched_scratch_totals: Dict[int, float] = {}
        if fsdp_sync and needs_ema_sync and len(params) > 1:
            pass1_indices = [
                idx for idx, st in enumerate(steps)
                if not (fused_single_pass and int(st.item() if isinstance(st, torch.Tensor) else st) > 1)
            ]
            if len(pass1_indices) > 1:
                batched_scratch_totals = self._precompute_fsdp_pass1_triton_dual(
                    pass1_indices, params, grads, exp_avgs, exp_avg_sqs, master_params,
                    steps, scratches, beta1, beta2, eps, tau, base_step, c1_base, c2_default, c2_fp16,
                    bound_mode, cautious
                )

        for i in range(len(params)):
            p, grad, exp_avg, exp_avg_sq, step_raw, scratch_param = (
                params[i], grads[i], exp_avgs[i], exp_avg_sqs[i], steps[i], scratches[i]
            )
            step = int(step_raw.item() if isinstance(step_raw, torch.Tensor) else step_raw)
            global_param_idx = orig_indices[i] if orig_indices is not None else i
            cur_scratch_synced = i in batched_scratch_totals
            cur_scratch_total = batched_scratch_totals[i] if cur_scratch_synced else float(p.numel())

            if p.numel() == 0:
                if fsdp_sync:
                    if not cur_scratch_synced:
                        scratch_param[:3].zero_()
                    scratch_param[5].zero_()
                    if fused_single_pass and step > 1 and needs_ema_sync:
                        pending_ema.append((scratch_param, 0, False))
                    else:
                        _sync_empty_fsdp_shard(
                            scratch_param, step, lr, clamp_min, ema_decay,
                            bound_mode, bound_ratio, cautious, fused_single_pass,
                            scratch_synced=cur_scratch_synced,
                            synced_total_elements=cur_scratch_total,
                        )
                if completed_indices is not None:
                    completed_indices.add(global_param_idx)
                continue

            param_has_master = has_master and master_params is not None and (master_params[i] is not p)
            mp_ref = master_params[i] if (param_has_master and master_params is not None) else None
            p_contig = p
            exp_avg_contig = exp_avg
            exp_avg_sq_contig = exp_avg_sq
            mp_contig = mp_ref if param_has_master and mp_ref is not None else p
            is_all_contig = True
            scratch_synced = cur_scratch_synced
            param_updated = False
            ema_updated = False
            copied_back = False
            n_elements = p.numel()
            total_elements = cur_scratch_total
            defer_this_ema = bool(fsdp_sync and fused_single_pass and step > 1 and needs_ema_sync)

            try:
                (
                    n_elements, b_size, n_warps, c_grid, dt_mode, s_dt_mode, is_all_contig,
                    p_contig, grad_contig, exp_avg_contig, exp_avg_sq_contig, mp_contig
                ) = self._prepare_triton_buffers(p, grad, exp_avg, exp_avg_sq, mp_ref, param_has_master, reuse_contig=cur_scratch_synced)

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

                kernel_seed = (seed + step * 0x9E3779B9 + global_param_idx * 10007) & 0x7FFFFFFF

                if not cautious and bound_mode != 1:
                    _lumina_dual_single_pass_kernel[c_grid](
                        p_contig, mp_contig, grad_contig, exp_avg_contig, exp_avg_sq_contig,
                        scratch_param,
                        n_elements, beta1, beta2, c1, c2, lr, weight_decay,
                        bound_mode, bound_ratio,
                        kernel_seed, dt_mode, s_dt_mode, use_sr, param_has_master,
                        BLOCK_SIZE=b_size, num_warps=n_warps
                    )
                    param_updated = True
                    ema_updated = True
                elif fused_single_pass and step > 1:
                    scratch_param[:3].zero_()
                    _lumina_dual_fused_single_pass_kernel[c_grid](
                        p_contig, mp_contig, grad_contig, exp_avg_contig, exp_avg_sq_contig,
                        scratch_param,
                        n_elements, beta1, beta2, c1, c2, lr, weight_decay,
                        bound_mode, bound_ratio,
                        kernel_seed, dt_mode, s_dt_mode, use_sr,
                        cautious, param_has_master, BLOCK_SIZE=b_size, num_warps=n_warps
                    )
                    param_updated = True
                    if defer_this_ema:
                        pending_ema.append((scratch_param, n_elements, True))
                        ema_updated = True
                    else:
                        total_elements = float(n_elements)
                        if fsdp_sync and needs_ema_sync:
                            total_elements = _sync_fsdp_scratch(scratch_param, n_elements)
                            scratch_synced = True
                        _lumina_post_step_ema_kernel[(1,)](
                            scratch_param, total_elements, lr, clamp_min, ema_decay,
                            bound_mode, bound_ratio, cautious, num_warps=1
                        )
                        ema_updated = True
                else:
                    if not scratch_synced:
                        scratch_param[:3].zero_()
                        _lumina_dual_pass1_kernel[c_grid](
                            p_contig, mp_contig, grad_contig, exp_avg_contig, exp_avg_sq_contig,
                            scratch_param,
                            n_elements, beta1, beta2, c1, c2,
                            bound_mode, dt_mode, s_dt_mode,
                            cautious, param_has_master,
                            BLOCK_SIZE=b_size, num_warps=n_warps
                        )
                        total_elements = float(n_elements)
                        if fsdp_sync and needs_ema_sync:
                            total_elements = _sync_fsdp_scratch(scratch_param, n_elements)
                            scratch_synced = True
                    _lumina_dual_pass2_kernel[c_grid](
                        p_contig, mp_contig, grad_contig, exp_avg_contig, exp_avg_sq_contig,
                        scratch_param,
                        n_elements, total_elements, beta1, beta2, c1, c2, lr, weight_decay, clamp_min,
                        bound_mode, bound_ratio,
                        kernel_seed, dt_mode, s_dt_mode, use_sr, cautious, param_has_master,
                        BLOCK_SIZE=b_size, num_warps=n_warps
                    )
                    param_updated = True
                    scratch_param[:3].zero_()
                    ema_updated = True

                if not is_all_contig:
                    self._copy_back_non_contig(
                        p, p_contig, exp_avg, exp_avg_contig, exp_avg_sq, exp_avg_sq_contig,
                        mp_ref, mp_contig, param_has_master
                    )
                    copied_back = True

                if completed_indices is not None:
                    completed_indices.add(global_param_idx)
            except Exception as e:
                if not self._warned_triton_failure:
                    err_line = str(e).strip().splitlines()[0] if str(e).strip() else "Unknown error"
                    logger.warning(f"LuminaV: Triton kernel execution failed ({err_line}). Falling back to PyTorch single-step.")
                    self._warned_triton_failure = True

                if p.is_cuda:
                    with contextlib.suppress(Exception):
                        torch.cuda.synchronize(p.device)

                if param_updated:
                    if not is_all_contig and not copied_back:
                        self._copy_back_non_contig(
                            p, p_contig, exp_avg, exp_avg_contig, exp_avg_sq, exp_avg_sq_contig,
                            mp_ref, mp_contig, param_has_master
                        )
                    if (fused_single_pass and step > 1) and needs_ema_sync and not defer_this_ema and not ema_updated:
                        _fallback_post_step_ema(
                            scratch_param, n_elements, lr, clamp_min, ema_decay,
                            bound_mode, bound_ratio, cautious, fsdp_sync, scratch_synced,
                            synced_total_elements=total_elements
                        )
                else:
                    if not scratch_synced:
                        scratch_param[:3].zero_()
                    scratch_param[5].zero_()
                    if step <= 1:
                        scratch_param[3] = 1.0
                        scratch_param[4] = 1.0
                    self._single_step_dual(
                        [p], [grad], [exp_avg], [exp_avg_sq],
                        [mp_ref] if param_has_master else None,
                        [step], [scratch_param], lr, beta1, beta2, eps, weight_decay, tau,
                        cautious, clamp_min, use_sr, bound, bound_type, bound_ratio,
                        fused_single_pass, ema_decay, fsdp_sync, seed,
                        scratch_synced=scratch_synced,
                        scratch_total_elements=total_elements,
                        orig_indices=[global_param_idx],
                        defer_fsdp_post_ema=defer_this_ema,
                    )
                    if defer_this_ema:
                        pending_ema.append((scratch_param, n_elements, False))
                if completed_indices is not None:
                    completed_indices.add(global_param_idx)

        if pending_ema:
            self._finalize_coalesced_fsdp_ema_triton(
                pending_ema, lr, clamp_min, ema_decay, bound_mode, bound_ratio, cautious
            )
            if completed_indices is not None:
                for idx_done in range(len(params)):
                    completed_indices.add(orig_indices[idx_done] if orig_indices is not None else idx_done)
