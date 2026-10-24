from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from ..kernels import (
    _lumina_diff_sum_kernel,
    _lumina_post_step_ema_kernel,
    triton,
)
from ..single_backend import LuminaVSingleMixin
from ..utils.fsdp_sync import (
    _fallback_post_step_ema,
    _sync_fsdp_diff_batch,
    _sync_fsdp_scratch_batch,
)

logger = logging.getLogger("LuminaV")


class TritonBaseMixin(LuminaVSingleMixin):
    def _prepare_triton_buffers(
        self,
        p: Tensor,
        grad: Tensor,
        exp_avg: Tensor,
        exp_avg_sq: Optional[Tensor],
        mp: Optional[Tensor],
        param_has_master: bool,
        reuse_contig: bool = False,
    ):
        state = self.state[p]
        n_elements = p.numel()
        curr_dt_mode = 1 if p.dtype == torch.bfloat16 else (2 if p.dtype == torch.float16 else 0)
        curr_s_dt_mode = 1 if exp_avg.dtype == torch.bfloat16 else (2 if exp_avg.dtype == torch.float16 else 0)
        cached_plan = state.get("_plan")
        if (
            cached_plan is None
            or len(cached_plan) != 6
            or cached_plan[0] != n_elements
            or cached_plan[4] != curr_dt_mode
            or cached_plan[5] != curr_s_dt_mode
        ):
            if n_elements >= 2_097_152:
                b_size = 2048
                n_warps = 8
            else:
                b_size = 1024
                n_warps = 4
            c_grid = (triton.cdiv(n_elements, b_size),)
            dt_mode = curr_dt_mode
            s_dt_mode = curr_s_dt_mode
            state["_plan"] = (n_elements, b_size, n_warps, c_grid, dt_mode, s_dt_mode)
        else:
            n_elements, b_size, n_warps, c_grid, dt_mode, s_dt_mode = cached_plan

        is_orig_contig = p.is_contiguous()
        is_grad_contig = grad.is_contiguous()
        is_m_contig = exp_avg.is_contiguous()
        is_v_contig = (exp_avg_sq is None) or exp_avg_sq.is_contiguous()
        is_mp_contig = (not param_has_master) or (mp is not None and mp.is_contiguous())
        is_all_contig = is_orig_contig and is_grad_contig and is_m_contig and is_v_contig and is_mp_contig

        if is_all_contig:
            p_contig = p
            grad_contig = grad
            exp_avg_contig = exp_avg
            exp_avg_sq_contig = exp_avg_sq
            mp_contig = mp if (param_has_master and mp is not None) else p
        else:
            if not self._warned_non_contiguous:
                logger.warning(
                    "LuminaV detected non-contiguous tensor buffer(s). An explicit contiguous copy "
                    "was created for Triton kernel execution."
                )
                self._warned_non_contiguous = True

            curr_step_tag = state.get("step")
            can_reuse = bool(reuse_contig and curr_step_tag is not None and state.get("_contig_step") == curr_step_tag)

            if is_orig_contig:
                p_contig = p
            else:
                need_p_alloc = (
                    "p_contig" not in state
                    or state["p_contig"].shape != p.shape
                    or state["p_contig"].dtype != p.dtype
                )
                if need_p_alloc:
                    state["p_contig"] = torch.empty_like(p, memory_format=torch.contiguous_format)
                p_contig = state["p_contig"]
                if need_p_alloc or not can_reuse:
                    p_contig.copy_(p)

            grad_contig = grad if is_grad_contig else grad.contiguous()

            if is_m_contig:
                exp_avg_contig = exp_avg
            else:
                need_m_alloc = (
                    "exp_avg_contig" not in state
                    or state["exp_avg_contig"].shape != exp_avg.shape
                    or state["exp_avg_contig"].dtype != exp_avg.dtype
                )
                if need_m_alloc:
                    state["exp_avg_contig"] = torch.empty_like(exp_avg, memory_format=torch.contiguous_format)
                exp_avg_contig = state["exp_avg_contig"]
                if need_m_alloc or not can_reuse:
                    exp_avg_contig.copy_(exp_avg)

            if exp_avg_sq is None:
                exp_avg_sq_contig = None
            elif is_v_contig:
                exp_avg_sq_contig = exp_avg_sq
            else:
                need_v_alloc = (
                    "exp_avg_sq_contig" not in state
                    or state["exp_avg_sq_contig"].shape != exp_avg_sq.shape
                    or state["exp_avg_sq_contig"].dtype != exp_avg_sq.dtype
                )
                if need_v_alloc:
                    state["exp_avg_sq_contig"] = torch.empty_like(exp_avg_sq, memory_format=torch.contiguous_format)
                exp_avg_sq_contig = state["exp_avg_sq_contig"]
                if need_v_alloc or not can_reuse:
                    exp_avg_sq_contig.copy_(exp_avg_sq)

            if param_has_master and mp is not None:
                if is_mp_contig:
                    mp_contig = mp
                else:
                    need_mp_alloc = (
                        "mp_contig" not in state
                        or state["mp_contig"].shape != mp.shape
                        or state["mp_contig"].dtype != mp.dtype
                    )
                    if need_mp_alloc:
                        state["mp_contig"] = torch.empty_like(mp, memory_format=torch.contiguous_format)
                    mp_contig = state["mp_contig"]
                    if need_mp_alloc or not can_reuse:
                        mp_contig.copy_(mp)
            else:
                mp_contig = p_contig

            state["_contig_step"] = curr_step_tag

        return (
            n_elements, b_size, n_warps, c_grid, dt_mode, s_dt_mode, is_all_contig,
            p_contig, grad_contig, exp_avg_contig, exp_avg_sq_contig, mp_contig
        )

    @staticmethod
    def _copy_back_non_contig(
        p: Tensor,
        p_contig: Tensor,
        exp_avg: Tensor,
        exp_avg_contig: Tensor,
        exp_avg_sq: Optional[Tensor],
        exp_avg_sq_contig: Optional[Tensor],
        mp: Optional[Tensor],
        mp_contig: Tensor,
        param_has_master: bool,
    ) -> None:
        if not p.is_contiguous():
            p.copy_(p_contig)
        if not exp_avg.is_contiguous():
            exp_avg.copy_(exp_avg_contig)
        if exp_avg_sq is not None and exp_avg_sq_contig is not None and not exp_avg_sq.is_contiguous():
            exp_avg_sq.copy_(exp_avg_sq_contig)
        if param_has_master and mp is not None and not mp.is_contiguous():
            mp.copy_(mp_contig)

    def _precompute_fsdp_diff_triton(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        scratches: List[Tensor],
    ) -> List[float]:
        for p, grad, exp_avg, sc in zip(params, grads, exp_avgs, scratches):
            sc[5].zero_()
            n_el = p.numel()
            if n_el == 0:
                continue
            try:
                g_c = grad if grad.is_contiguous() else grad.contiguous()
                m_c = exp_avg if exp_avg.is_contiguous() else exp_avg.contiguous()
                b_size = 2048 if n_el >= 2_097_152 else 1024
                n_warps = 8 if n_el >= 2_097_152 else 4
                c_grid = (triton.cdiv(n_el, b_size),)
                _lumina_diff_sum_kernel[c_grid](
                    g_c, m_c, sc, n_el, BLOCK_SIZE=b_size, num_warps=n_warps
                )
            except Exception:
                chunk_sz = 2_097_152
                if n_el <= chunk_sz:
                    g_clean = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                    sc[5].copy_((g_clean - exp_avg.float()).clamp(-1e18, 1e18).sum().to(torch.float32))
                else:
                    flat_g = (grad if grad.is_contiguous() else grad.contiguous()).view(-1)
                    flat_m = (exp_avg if exp_avg.is_contiguous() else exp_avg.contiguous()).view(-1)
                    diff_acc = 0.0
                    for offset in range(0, n_el, chunk_sz):
                        cur = min(chunk_sz, n_el - offset)
                        g_c_sub = torch.nan_to_num(flat_g[offset : offset + cur], nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                        m_c_sub = flat_m[offset : offset + cur].float()
                        diff_acc += (g_c_sub - m_c_sub).clamp(-1e18, 1e18).sum().item()
                    sc[5] = diff_acc
        return _sync_fsdp_diff_batch(scratches, [p.numel() for p in params])

    def _finalize_coalesced_fsdp_ema_triton(
        self,
        pending_ema: List[Tuple[Tensor, int, bool]],
        lr: float,
        clamp_min: float,
        ema_decay: float,
        bound_mode: int,
        bound_ratio: float,
        cautious: bool,
    ) -> None:
        if not pending_ema:
            return
        sc_list = [item[0] for item in pending_ema]
        numels = [item[1] for item in pending_ema]
        try:
            totals = _sync_fsdp_scratch_batch(sc_list, numels)
            batch_synced = True
        except Exception:
            totals = [max(float(n), 1.0) for n in numels]
            batch_synced = False
        for (sc_p, n_el_p, allow_triton), tot_el in zip(pending_ema, totals):
            if batch_synced and allow_triton and n_el_p > 0:
                try:
                    _lumina_post_step_ema_kernel[(1,)](
                        sc_p, tot_el, lr, clamp_min, ema_decay,
                        bound_mode, bound_ratio, cautious, num_warps=1
                    )
                    continue
                except Exception:
                    pass
            _fallback_post_step_ema(
                sc_p, n_el_p, lr, clamp_min, ema_decay,
                bound_mode, bound_ratio, cautious, True, batch_synced,
                synced_total_elements=tot_el
            )
