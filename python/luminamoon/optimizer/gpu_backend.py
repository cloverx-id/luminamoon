from __future__ import annotations

import contextlib
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from torch import Tensor

from .fsdp_sync import (
    _fallback_post_step_ema,
    _sync_empty_fsdp_shard,
    _sync_fsdp_scratch,
    _sync_fsdp_scratch_batch,
)
from .torch_ops import (
    _compute_step_val,
    _decode_state_f32,
    _encode_state_from_f32,
    _get_device_context,
    _sr_apply_step_val,
    _sr_quantize,
)
from .triton_kernels import (
    HAS_TRITON,
    _lumina_dual_fused_single_pass_kernel,
    _lumina_dual_pass1_kernel,
    _lumina_dual_pass2_kernel,
    _lumina_dual_single_pass_kernel,
    _lumina_post_step_ema_kernel,
    _lumina_tdclm_pass1_kernel,
    _lumina_tdclm_pass2_kernel,
    _lumina_tdclm_single_pass_kernel,
    triton,
)

logger = logging.getLogger("LuminaV.GPU")


class LuminaVGpuMixin:
    seed: int
    state: Any
    param_groups: List[Dict[str, Any]]
    _generators: Dict[Tuple[str, int, int], torch.Generator]
    _warned_non_contiguous: bool
    _warned_triton_failure: bool

    def _get_generator(
        self,
        device: torch.device,
        seed: Optional[int] = None,
        slot: int = 0,
        reseed: bool = False,
    ) -> torch.Generator:
        effective_seed = int(getattr(self, "seed", 1337) if seed is None else seed) & 0x7FFFFFFF
        dev_type = device.type
        if device.index is not None:
            dev_idx = device.index
        elif dev_type == "cuda" and torch.cuda.is_available():
            dev_idx = torch.cuda.current_device()
        elif dev_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            dev_idx = torch.xpu.current_device()
        elif hasattr(torch, "accelerator") and hasattr(torch.accelerator, "current_device_index") and torch.accelerator.is_available():
            dev_idx = getattr(torch.accelerator, "current_device_index")()
        else:
            dev_idx = 0
        key = (dev_type, dev_idx, slot if reseed else effective_seed)
        if key not in self._generators:
            try:
                gen = torch.Generator(device=device)
            except (RuntimeError, ValueError, TypeError):
                gen = torch.Generator(device="cpu")
            gen.manual_seed(effective_seed)
            self._generators[key] = gen
        elif reseed:
            self._generators[key].manual_seed(effective_seed)
        return self._generators[key]

    def _get_step_generators(
        self,
        device: torch.device,
        seed: int,
        step: int,
        param_index: int,
        need_m: bool,
        need_p: bool,
    ) -> Tuple[Optional[torch.Generator], Optional[torch.Generator]]:
        base_seed = (seed + step * 0x9E3779B9 + param_index * 10007) & 0x7FFFFFFF
        gen_m = self._get_generator(device, seed=base_seed, slot=0, reseed=True) if need_m else None
        seed_p = (base_seed + 0x55555555) & 0x7FFFFFFF
        gen_p = self._get_generator(device, seed=seed_p, slot=1, reseed=True) if need_p else None
        return gen_m, gen_p

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
        state = self.state.get(p) if hasattr(self, "state") and hasattr(self.state, "get") else None
        n_elements = p.numel()
        curr_dt_mode = 1 if p.dtype == torch.bfloat16 else (2 if p.dtype == torch.float16 else 0)
        curr_s_dt_mode = 1 if exp_avg.dtype == torch.bfloat16 else (2 if exp_avg.dtype == torch.float16 else 0)
        cached_plan = state.get("_plan") if state is not None else None
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
            if state is not None:
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
            if not getattr(self, "_warned_non_contiguous", False):
                logger.warning(
                    "LuminaV detected non-contiguous tensor buffer(s). An explicit contiguous copy "
                    "was created for Triton kernel execution."
                )
                self._warned_non_contiguous = True

            if state is None:
                state = self.state[p]
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

    def _prewarm_all_configs(self) -> None:
        """Pre-compiles Triton GPU kernels on 64-element dummy buffers (`lr=0.0`) so Step 1 and Step 2 execute without JIT compilation latency."""
        if not HAS_TRITON or triton is None:
            return
        if not hasattr(self, "param_groups") or not self.param_groups:
            return

        deterministic_mode = (
            torch.are_deterministic_algorithms_enabled()
            if hasattr(torch, "are_deterministic_algorithms_enabled")
            else False
        )
        seen_configs: Set[Any] = getattr(self, "_prewarmed_triton_configs", set())
        setattr(self, "_prewarmed_triton_configs", seen_configs)

        fused_buckets: List[Dict[str, Any]] = getattr(self, "_fused_buckets", [])

        for group in self.param_groups:
            if not bool(group.get("prewarm_jit", True)):
                continue
            exec_mode = group.get("execution", 0)
            if exec_mode in ("auto", "foreach", 0) and deterministic_mode:
                continue
            if exec_mode not in ("auto", "foreach", "triton", 0, 1):
                continue

            buf_count = int(group.get("buffer", 2))
            master_mode = int(group.get("master_weights", 0))
            cautious = bool(group.get("cautious", True))
            bound = bool(group.get("bound", True))
            bound_type = group.get("bound_type", "coordinate")
            bound_ratio = float(group.get("bound_ratio", 0.05))
            bound_mode = (
                (1 if bound_type in ("radial", 1) else 2)
                if (bound and bound_ratio > 0.0)
                else 0
            )
            fused_single_pass = bool(group.get("fused_single_pass", True))
            use_sr = bool(group.get("stochastic_rounding", True))
            beta1, beta2 = group.get("betas", (0.9, 0.999))
            beta1, beta2 = float(beta1), float(beta2)
            eps = float(group.get("eps", 1e-8))
            tau = float(group.get("tau", 0.8))
            lam = float(group.get("lam", 0.3))
            clamp_min = float(group.get("cautious_clamp_min", 0.5))
            ema_decay = float(group.get("ema_decay", 0.95 if buf_count == 1 else 0.80))

            targets: Set[Tuple[torch.device, torch.dtype, int, int]] = set()
            group_buckets = [b for b in fused_buckets if b.get("group") is group]
            if bool(group.get("fuse_tensors", False)) and group_buckets:
                for b in group_buckets:
                    dev = b["device"]
                    dt = b["dtype"]
                    if dev.type in ("cuda", "xpu"):
                        n_el = int(b["flat_p"].numel())
                        b_size, n_warps = (2048, 8) if n_el >= 2_097_152 else (1024, 4)
                        targets.add((dev, dt, b_size, n_warps))
            else:
                for p in group.get("params", []):
                    if isinstance(p, torch.Tensor) and p.device.type in ("cuda", "xpu") and p.numel() > 0:
                        n_el = int(p.numel())
                        b_size, n_warps = (2048, 8) if n_el >= 2_097_152 else (1024, 4)
                        targets.add((p.device, p.dtype, b_size, n_warps))

            for dev, dt, b_size, n_warps in targets:
                dt_mode = 1 if dt == torch.bfloat16 else (2 if dt == torch.float16 else 0)
                state_dtype = torch.float32 if (master_mode == 2 or dt == torch.float32) else dt
                s_dt_mode = 1 if state_dtype == torch.bfloat16 else (2 if state_dtype == torch.float16 else 0)
                param_has_master = (master_mode > 0) and (dt in (torch.float16, torch.bfloat16))
                cfg_key = (
                    dev.type,
                    dev.index or 0,
                    dt_mode,
                    s_dt_mode,
                    param_has_master,
                    buf_count,
                    cautious,
                    bound_mode,
                    fused_single_pass,
                    use_sr,
                    b_size,
                    n_warps,
                )
                if cfg_key in seen_configs:
                    continue
                seen_configs.add(cfg_key)

                with contextlib.suppress(Exception), _get_device_context(dev):
                    dummy_p = torch.zeros(64, dtype=dt, device=dev)
                    dummy_g = torch.zeros(64, dtype=dt, device=dev)
                    dummy_m = torch.zeros(64, dtype=state_dtype, device=dev)
                    dummy_v = (
                        torch.zeros(64, dtype=state_dtype, device=dev)
                        if buf_count == 2
                        else None
                    )
                    dummy_mp = (
                        torch.zeros(64, dtype=torch.float32, device=dev)
                        if param_has_master
                        else dummy_p
                    )
                    dummy_sc = torch.tensor(
                        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                        dtype=torch.float32,
                        device=dev,
                    )

                    if buf_count == 1:
                        if not cautious and bound_mode != 1:
                            _lumina_tdclm_single_pass_kernel[(1,)](
                                dummy_p, dummy_mp, dummy_g, dummy_m,
                                dummy_sc,
                                64, beta1, 0.0, 0.0, eps, lam,
                                bound_mode, bound_ratio,
                                1337, dt_mode, s_dt_mode,
                                use_sr, cautious, param_has_master,
                                BLOCK_SIZE=b_size, num_warps=n_warps,
                            )
                        else:
                            _lumina_tdclm_pass1_kernel[(1,)](
                                dummy_p, dummy_mp, dummy_g, dummy_m,
                                dummy_sc,
                                64, beta1, eps, lam,
                                bound_mode, dt_mode, s_dt_mode,
                                cautious, param_has_master,
                                BLOCK_SIZE=b_size, num_warps=n_warps,
                            )
                            _lumina_tdclm_pass2_kernel[(1,)](
                                dummy_p, dummy_mp, dummy_g, dummy_m,
                                dummy_sc,
                                64, 64.0, beta1, 0.0, 0.0, eps, lam, clamp_min,
                                bound_mode, bound_ratio,
                                1337, dt_mode, s_dt_mode, use_sr,
                                cautious, param_has_master,
                                BLOCK_SIZE=b_size, num_warps=n_warps,
                            )
                            if fused_single_pass:
                                dummy_sc[:3].zero_()
                                dummy_sc[3] = 1.0
                                dummy_sc[4] = 1.0
                                _lumina_tdclm_single_pass_kernel[(1,)](
                                    dummy_p, dummy_mp, dummy_g, dummy_m,
                                    dummy_sc,
                                    64, beta1, 0.0, 0.0, eps, lam,
                                    bound_mode, bound_ratio,
                                    1337, dt_mode, s_dt_mode,
                                    use_sr, cautious, param_has_master,
                                    BLOCK_SIZE=b_size, num_warps=n_warps,
                                )
                                _lumina_post_step_ema_kernel[(1,)](
                                    dummy_sc, 64.0, 0.0, clamp_min, ema_decay,
                                    bound_mode, bound_ratio, cautious, num_warps=1,
                                )
                    else:
                        bc1_nes = 1.0 - (beta1 ** 2)
                        bc2 = max(1.0 - beta2, 1e-15)
                        c1 = (bc1_nes * tau) / math.sqrt(bc2)
                        c2 = eps * bc1_nes * tau
                        if not cautious and bound_mode != 1:
                            _lumina_dual_single_pass_kernel[(1,)](
                                dummy_p, dummy_mp, dummy_g, dummy_m, dummy_v,
                                dummy_sc,
                                64, beta1, beta2, c1, c2, 0.0, 0.0,
                                bound_mode, bound_ratio,
                                1337, dt_mode, s_dt_mode, use_sr, param_has_master,
                                BLOCK_SIZE=b_size, num_warps=n_warps,
                            )
                        else:
                            _lumina_dual_pass1_kernel[(1,)](
                                dummy_p, dummy_mp, dummy_g, dummy_m, dummy_v,
                                dummy_sc,
                                64, beta1, beta2, c1, c2,
                                bound_mode, dt_mode, s_dt_mode,
                                cautious, param_has_master,
                                BLOCK_SIZE=b_size, num_warps=n_warps,
                            )
                            _lumina_dual_pass2_kernel[(1,)](
                                dummy_p, dummy_mp, dummy_g, dummy_m, dummy_v,
                                dummy_sc,
                                64, 64.0, beta1, beta2, c1, c2, 0.0, 0.0, clamp_min,
                                bound_mode, bound_ratio,
                                1337, dt_mode, s_dt_mode, use_sr, cautious, param_has_master,
                                BLOCK_SIZE=b_size, num_warps=n_warps,
                            )
                            if fused_single_pass:
                                dummy_sc[:3].zero_()
                                dummy_sc[3] = 1.0
                                dummy_sc[4] = 1.0
                                _lumina_dual_fused_single_pass_kernel[(1,)](
                                    dummy_p, dummy_mp, dummy_g, dummy_m, dummy_v,
                                    dummy_sc,
                                    64, beta1, beta2, c1, c2, 0.0, 0.0,
                                    bound_mode, bound_ratio,
                                    1337, dt_mode, s_dt_mode, use_sr,
                                    cautious, param_has_master,
                                    BLOCK_SIZE=b_size, num_warps=n_warps,
                                )
                                _lumina_post_step_ema_kernel[(1,)](
                                    dummy_sc, 64.0, 0.0, clamp_min, ema_decay,
                                    bound_mode, bound_ratio, cautious, num_warps=1,
                                )

                    if dev.type == "cuda" and torch.cuda.is_available():
                        torch.cuda.synchronize(dev)
                    del dummy_p, dummy_g, dummy_m, dummy_v, dummy_mp, dummy_sc

    def _dispatch_triton(
        self, params, grads, exp_avgs, exp_avg_sqs, master_params, steps, scratches,
        lr, beta1, beta2, eps, weight_decay, tau, lam,
        cautious, clamp_min, buf_count, use_sr, bound, bound_type, bound_ratio,
        fused_single_pass, ema_decay, fsdp_sync: bool = False, seed: int = 1337,
        completed_indices: Optional[Set[int]] = None,
        orig_indices: Optional[List[int]] = None,
    ):
        primary_dev = params[0].device
        is_single_device = all(p.device == primary_dev for p in params)

        if is_single_device:
            with _get_device_context(primary_dev):
                if buf_count == 2:
                    self._triton_step_dual(
                        params, grads, exp_avgs, exp_avg_sqs, master_params, steps, scratches,
                        lr, beta1, beta2, eps, weight_decay, tau, cautious, clamp_min,
                        use_sr, bound, bound_type, bound_ratio, fused_single_pass, ema_decay,
                        fsdp_sync, seed,
                        orig_indices=orig_indices, completed_indices=completed_indices
                    )
                else:
                    self._triton_step_tdclm(
                        params, grads, exp_avgs, master_params, steps, scratches,
                        lr, beta1, eps, weight_decay, lam, cautious, clamp_min,
                        use_sr, bound, bound_type, bound_ratio,
                        fused_single_pass=fused_single_pass, ema_decay=ema_decay,
                        fsdp_sync=fsdp_sync, seed=seed,
                        orig_indices=orig_indices, completed_indices=completed_indices
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
            sub_orig = [orig_indices[i] if orig_indices is not None else i for i in indices]

            with _get_device_context(dev):
                if buf_count == 2:
                    sub_v = [exp_avg_sqs[i] for i in indices]
                    self._triton_step_dual(
                        sub_p, sub_g, sub_m, sub_v, sub_mp, sub_st, sub_sc,
                        lr, beta1, beta2, eps, weight_decay, tau, cautious, clamp_min,
                        use_sr, bound, bound_type, bound_ratio, fused_single_pass, ema_decay,
                        fsdp_sync, seed,
                        orig_indices=sub_orig, completed_indices=completed_indices
                    )
                else:
                    self._triton_step_tdclm(
                        sub_p, sub_g, sub_m, sub_mp, sub_st, sub_sc,
                        lr, beta1, eps, weight_decay, lam, cautious, clamp_min,
                        use_sr, bound, bound_type, bound_ratio,
                        fused_single_pass=fused_single_pass, ema_decay=ema_decay,
                        fsdp_sync=fsdp_sync, seed=seed,
                        orig_indices=sub_orig, completed_indices=completed_indices
                    )

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
            m_f32 = _decode_state_f32(exp_avg)
            v_f32 = _decode_state_f32(exp_avg_sq)
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
                m_f32 = _decode_state_f32(flat_m[offset : offset + cur])
                v_f32 = _decode_state_f32(flat_v[offset : offset + cur])
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
        effective_eps = eps
        if n_el <= chunk_size:
            g_clean = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
            m_f32 = _decode_state_f32(exp_avg)
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
                m_f32 = _decode_state_f32(flat_m[offset : offset + cur])
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

    def _precompute_fsdp_pass1_triton_dual(
        self,
        pass1_indices: List[int],
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        master_params: Optional[List[Tensor]],
        steps: List[Any],
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
                effective_eps = eps
                bc1_nes = 1.0 - (beta1 ** (safe_step + 1))
                bc2 = max(1.0 - (beta2 ** safe_step), 1e-15)
                c1 = (bc1_nes * tau) / math.sqrt(bc2)
                c2 = effective_eps * bc1_nes * tau
            param_has_master = (master_params is not None) and (master_params[idx] is not p)
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

    def _precompute_fsdp_pass1_triton_tdclm(
        self,
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
    ) -> Dict[int, float]:
        for idx in pass1_indices:
            p = params[idx]
            sc = scratches[idx]
            sc[:3].zero_()
            n_el = p.numel()
            if n_el == 0:
                continue
            grad = grads[idx]
            exp_avg = exp_avgs[idx]
            param_has_master = (master_params is not None) and (master_params[idx] is not p)
            mp_ref = master_params[idx] if (param_has_master and master_params is not None) else None
            try:
                (
                    n_elements, b_size, n_warps, c_grid, dt_mode, s_dt_mode, _,
                    p_contig, grad_contig, exp_avg_contig, _, mp_contig
                ) = self._prepare_triton_buffers(p, grad, exp_avg, None, mp_ref, param_has_master)
                _lumina_tdclm_pass1_kernel[c_grid](
                    p_contig, mp_contig, grad_contig, exp_avg_contig,
                    sc,
                    n_elements, beta1, eps, lam,
                    bound_mode, dt_mode, s_dt_mode,
                    cautious, param_has_master,
                    BLOCK_SIZE=b_size, num_warps=n_warps
                )
            except Exception:
                ref_p = mp_ref if (param_has_master and mp_ref is not None) else p
                self._compute_pass1_single_tdclm_one(
                    p, grad, exp_avg, ref_p, sc,
                    beta1, eps, lam, bound_mode, cautious
                )
        synced_totals = _sync_fsdp_scratch_batch(
            [scratches[idx] for idx in pass1_indices],
            [params[idx].numel() for idx in pass1_indices],
        )
        return {idx: tot for idx, tot in zip(pass1_indices, synced_totals)}

    @classmethod
    def _precompute_fsdp_pass1_single_dual(
        cls,
        pass1_indices: List[int],
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        master_params: Optional[List[Tensor]],
        steps: List[Any],
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
                effective_eps = eps
                bc1_nes = 1.0 - (beta1 ** (safe_step + 1))
                bc2 = max(1.0 - (beta2 ** safe_step), 1e-15)
                c1 = (bc1_nes * tau) / math.sqrt(bc2)
                c2 = effective_eps * bc1_nes * tau
            param_has_master = (master_params is not None) and (master_params[idx] is not p)
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
        for idx in pass1_indices:
            p = params[idx]
            sc = scratches[idx]
            sc[:3].zero_()
            n_el = p.numel()
            if n_el == 0:
                continue
            param_has_master = (master_params is not None) and (master_params[idx] is not p)
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

        base_step = int(steps[0].item() if isinstance(steps[0], torch.Tensor) else steps[0])
        safe_base_step = max(base_step, 1)
        bc1_nes_base = 1.0 - (beta1 ** (safe_base_step + 1))
        bc2_base = max(1.0 - (beta2 ** safe_base_step), 1e-15)
        c1_base = (bc1_nes_base * tau) / math.sqrt(bc2_base)
        c2_default = eps * bc1_nes_base * tau
        c2_fp16 = c2_default

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

            param_has_master = (master_params is not None) and (master_params[i] is not p)
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
                    effective_eps = eps
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
                if not getattr(self, "_warned_triton_failure", False):
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

    def _triton_step_tdclm(
        self, params, grads, exp_avgs, master_params, steps, scratches,
        lr, beta1, eps, weight_decay, lam, cautious, clamp_min,
        use_sr, bound=True, bound_type="coordinate", bound_ratio=0.05,
        fused_single_pass: bool = True, ema_decay: float = 0.8,
        fsdp_sync: bool = False, seed: int = 1337,
        orig_indices: Optional[List[int]] = None,
        completed_indices: Optional[Set[int]] = None
    ):
        bound_mode = 0
        if bound and bound_ratio > 0.0:
            bound_mode = 1 if bound_type == "radial" else 2

        pending_ema: List[Tuple[Tensor, int, bool]] = []
        needs_ema_sync = cautious or bound_mode == 1

        batched_scratch_totals: Dict[int, float] = {}
        if fsdp_sync and needs_ema_sync and len(params) > 1:
            pass1_indices = [
                idx for idx, st in enumerate(steps)
                if not (fused_single_pass and int(st.item() if isinstance(st, torch.Tensor) else st) > 1)
            ]
            if len(pass1_indices) > 1:
                batched_scratch_totals = self._precompute_fsdp_pass1_triton_tdclm(
                    pass1_indices, params, grads, exp_avgs, master_params, scratches,
                    beta1, eps, lam, bound_mode, cautious
                )

        for i in range(len(params)):
            p, grad, exp_avg, step_raw, scratch_param = (
                params[i], grads[i], exp_avgs[i], steps[i], scratches[i]
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

            param_has_master = (master_params is not None) and (master_params[i] is not p)
            mp_ref = master_params[i] if (param_has_master and master_params is not None) else None
            p_contig = p
            exp_avg_contig = exp_avg
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
                    p_contig, grad_contig, exp_avg_contig, _, mp_contig
                ) = self._prepare_triton_buffers(p, grad, exp_avg, None, mp_ref, param_has_master, reuse_contig=cur_scratch_synced)

                kernel_seed = (seed + step * 0x9E3779B9 + global_param_idx * 10007) & 0x7FFFFFFF

                if (fused_single_pass and step > 1) or (not cautious and bound_mode != 1):
                    if needs_ema_sync:
                        scratch_param[:3].zero_()
                    _lumina_tdclm_single_pass_kernel[c_grid](
                        p_contig, mp_contig, grad_contig, exp_avg_contig,
                        scratch_param,
                        n_elements, beta1, lr, weight_decay, eps, lam,
                        bound_mode, bound_ratio,
                        kernel_seed, dt_mode, s_dt_mode,
                        use_sr, cautious, param_has_master,
                        BLOCK_SIZE=b_size, num_warps=n_warps
                    )
                    param_updated = True
                    if needs_ema_sync:
                        if defer_this_ema:
                            pending_ema.append((scratch_param, n_elements, True))
                            ema_updated = True
                        else:
                            total_elements = float(n_elements)
                            if fsdp_sync:
                                total_elements = _sync_fsdp_scratch(scratch_param, n_elements)
                                scratch_synced = True
                            _lumina_post_step_ema_kernel[(1,)](
                                scratch_param, total_elements, lr, clamp_min, ema_decay,
                                bound_mode, bound_ratio, cautious, num_warps=1
                            )
                            ema_updated = True
                    else:
                        ema_updated = True
                else:
                    if not scratch_synced:
                        scratch_param[:3].zero_()
                        _lumina_tdclm_pass1_kernel[c_grid](
                            p_contig, mp_contig, grad_contig, exp_avg_contig,
                            scratch_param,
                            n_elements, beta1, eps, lam,
                            bound_mode, dt_mode, s_dt_mode,
                            cautious, param_has_master,
                            BLOCK_SIZE=b_size, num_warps=n_warps
                        )
                        total_elements = float(n_elements)
                        if fsdp_sync and needs_ema_sync:
                            total_elements = _sync_fsdp_scratch(scratch_param, n_elements)
                            scratch_synced = True
                    _lumina_tdclm_pass2_kernel[c_grid](
                        p_contig, mp_contig, grad_contig, exp_avg_contig,
                        scratch_param,
                        n_elements, total_elements, beta1, lr, weight_decay, eps, lam, clamp_min,
                        bound_mode, bound_ratio,
                        kernel_seed, dt_mode, s_dt_mode, use_sr,
                        cautious, param_has_master,
                        BLOCK_SIZE=b_size, num_warps=n_warps
                    )
                    param_updated = True
                    scratch_param[:3].zero_()
                    ema_updated = True

                if not is_all_contig:
                    self._copy_back_non_contig(
                        p, p_contig, exp_avg, exp_avg_contig, None, None,
                        mp_ref, mp_contig, param_has_master
                    )
                    copied_back = True

                if completed_indices is not None:
                    completed_indices.add(global_param_idx)
            except Exception as e:
                if not getattr(self, "_warned_triton_failure", False):
                    err_line = str(e).strip().splitlines()[0] if str(e).strip() else "Unknown error"
                    logger.warning(f"LuminaV: Triton kernel execution failed ({err_line}). Falling back to PyTorch single-step.")
                    self._warned_triton_failure = True

                if p.is_cuda:
                    with contextlib.suppress(Exception):
                        torch.cuda.synchronize(p.device)

                if param_updated:
                    if not is_all_contig and not copied_back:
                        self._copy_back_non_contig(
                            p, p_contig, exp_avg, exp_avg_contig, None, None,
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
                    self._single_step_tdclm(
                        [p], [grad], [exp_avg],
                        [mp_ref] if param_has_master else None,
                        [step], [scratch_param], lr, beta1, eps, weight_decay, lam,
                        cautious, clamp_min, use_sr, bound, bound_type, bound_ratio,
                        fused_single_pass=fused_single_pass, ema_decay=ema_decay,
                        fsdp_sync=fsdp_sync, seed=seed,
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

    def _single_step_dual(
        self, params, grads, exp_avgs, exp_avg_sqs, master_params, steps, scratches,
        lr, beta1, beta2, eps, weight_decay, tau, cautious, clamp_min,
        use_sr, bound=True, bound_type="coordinate", bound_ratio=0.05,
        fused_single_pass=False, ema_decay=0.8, fsdp_sync: bool = False, seed: int = 1337,
        scratch_synced: bool = False,
        scratch_total_elements: float = 0.0,
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
        c2_fp16 = c2_default

        ELEM_CHUNK_SIZE = 2_097_152

        needs_ema_sync = cautious or bound_mode == 1
        batch_post_ema = bool(
            fsdp_sync and not scratch_synced and not defer_fsdp_post_ema
            and scratches is not None and len(params) > 1 and needs_ema_sync
        )
        batched_scratch_totals: Dict[int, float] = {}
        if batch_post_ema and scratches is not None:
            pass1_indices = [
                idx for idx, st in enumerate(steps)
                if not (fused_single_pass and int(st.item() if isinstance(st, torch.Tensor) else st) > 1)
            ]
            if len(pass1_indices) > 1:
                batched_scratch_totals = self._precompute_fsdp_pass1_single_dual(
                    pass1_indices, params, grads, exp_avgs, exp_avg_sqs, master_params,
                    steps, scratches, beta1, beta2, eps, tau, base_step, c1_base, c2_default, c2_fp16,
                    bound_mode, cautious, ELEM_CHUNK_SIZE
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

            safe_step = max(step, 1)
            if step == base_step:
                c1 = c1_base
                c2 = c2_fp16 if p.dtype == torch.float16 else c2_default
            else:
                effective_eps = eps
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
            m_f32 = _decode_state_f32(exp_avg)
            v_f32 = _decode_state_f32(exp_avg_sq)

            m_new = m_f32.mul(beta1).add(g_f32, alpha=1.0 - beta1)
            clm_m = m_new.mul(beta1).add(g_f32, alpha=1.0 - beta1)
            diff = (g_f32 - m_new).clamp(-1e18, 1e18)

            v_new = v_f32.mul(beta2).addcmul(diff, diff, value=1.0 - beta2)
            if p.dtype == torch.float16 or exp_avg.dtype == torch.float16 or exp_avg_sq.dtype == torch.float16:
                v_new = v_new.clamp(0.0, 65000.0)

            sigma = v_new.clamp_min(0.0).sqrt().mul(c1).add(c2)
            u = torch.tanh(clm_m / sigma.clamp_min(1e-12))

            _encode_state_from_f32(exp_avg, m_new, use_sr=use_sr, generator=gen_m)
            _encode_state_from_f32(exp_avg_sq, v_new, use_sr=False, generator=None)

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

    def _single_step_dual_element_chunked(
        self, p, grad, exp_avg, exp_avg_sq, ref_p, step, sc,
        c1, c2, lr, beta1, beta2, weight_decay, cautious, clamp_min,
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
        v_work = exp_avg_sq if exp_avg_sq.is_contiguous() else exp_avg_sq.contiguous()
        if has_master and ref_p is not None:
            ref_work = ref_p if ref_p.is_contiguous() else ref_p.contiguous()
        else:
            ref_work = p_work

        flat_p = p_work.view(-1)
        flat_g = g_work.view(-1)
        flat_m = m_work.view(-1)
        flat_v = v_work.view(-1)
        flat_ref = ref_work.view(-1)

        need_sr_m = use_sr and (exp_avg.dtype in (torch.bfloat16, torch.float16))
        need_sr_p = use_sr and (not has_master) and (p.dtype in (torch.bfloat16, torch.float16))
        gen_m, gen_p = self._get_step_generators(
            p.device, seed, step, param_index, need_sr_m, need_sr_p
        )

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
                v_c = flat_v[offset : offset + cur]
                ref_c = flat_ref[offset : offset + cur]

                g_clean = torch.nan_to_num(g_c, nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                m_f32 = _decode_state_f32(m_c)
                v_f32 = _decode_state_f32(v_c)

                m_new = m_f32.mul(beta1).add(g_clean, alpha=1.0 - beta1)
                clm_m = m_new.mul(beta1).add(g_clean, alpha=1.0 - beta1)
                diff = (g_clean - m_new).clamp(-1e18, 1e18)
                v_new = v_f32.mul(beta2).addcmul(diff, diff, value=1.0 - beta2)
                if p.dtype == torch.float16 or m_c.dtype == torch.float16 or v_c.dtype == torch.float16:
                    v_new = v_new.clamp(0.0, 65000.0)

                sigma = v_new.clamp_min(0.0).sqrt().mul(c1).add(c2)
                u = torch.tanh(clm_m / sigma.clamp_min(1e-12))

                _encode_state_from_f32(m_c, m_new, use_sr=use_sr, generator=gen_m)
                _encode_state_from_f32(v_c, v_new, use_sr=False, generator=None)

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
                    v_c = flat_v[offset : offset + cur]
                    ref_c = flat_ref[offset : offset + cur]

                    g_clean = torch.nan_to_num(g_c, nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                    m_f32 = _decode_state_f32(m_c)
                    v_f32 = _decode_state_f32(v_c)

                    m_new = m_f32.mul(beta1).add(g_clean, alpha=1.0 - beta1)
                    clm_m = m_new.mul(beta1).add(g_clean, alpha=1.0 - beta1)
                    diff = (g_clean - m_new).clamp(-1e18, 1e18)
                    v_new = v_f32.mul(beta2).addcmul(diff, diff, value=1.0 - beta2)
                    if p.dtype == torch.float16 or m_c.dtype == torch.float16 or v_c.dtype == torch.float16:
                        v_new = v_new.clamp(0.0, 65000.0)

                    sigma = v_new.clamp_min(0.0).sqrt().mul(c1).add(c2)
                    u = torch.tanh(clm_m / sigma.clamp_min(1e-12))

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
                v_c = flat_v[offset : offset + cur]
                ref_c = flat_ref[offset : offset + cur]

                g_clean = torch.nan_to_num(g_c, nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                m_f32 = _decode_state_f32(m_c)
                v_f32 = _decode_state_f32(v_c)

                m_new = m_f32.mul(beta1).add(g_clean, alpha=1.0 - beta1)
                clm_m = m_new.mul(beta1).add(g_clean, alpha=1.0 - beta1)
                diff = (g_clean - m_new).clamp(-1e18, 1e18)
                v_new = v_f32.mul(beta2).addcmul(diff, diff, value=1.0 - beta2)
                if p.dtype == torch.float16 or m_c.dtype == torch.float16 or v_c.dtype == torch.float16:
                    v_new = v_new.clamp(0.0, 65000.0)

                sigma = v_new.clamp_min(0.0).sqrt().mul(c1).add(c2)
                u = torch.tanh(clm_m / sigma.clamp_min(1e-12))

                _encode_state_from_f32(m_c, m_new, use_sr=use_sr, generator=gen_m)
                _encode_state_from_f32(v_c, v_new, use_sr=False, generator=None)

                delta_theta = (u * (u * g_clean > 0.0).float() * cautious_scale) if cautious else u
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
        if not exp_avg_sq.is_contiguous():
            exp_avg_sq.copy_(v_work)
        if has_master and ref_p is not None and not ref_p.is_contiguous():
            ref_p.copy_(ref_work)

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
        if batch_post_ema and scratches is not None:
            pass1_indices = [
                idx for idx, st in enumerate(steps)
                if not (fused_single_pass and int(st.item() if isinstance(st, torch.Tensor) else st) > 1)
            ]
            if len(pass1_indices) > 1:
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
            m_f32 = _decode_state_f32(exp_avg)

            beta1_sq = beta1 * beta1
            one_minus_beta1_sq = 1.0 - beta1_sq
            effective_eps = eps

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
            u: Tensor = clm_m / clm_env
            del clm_m, clm_env, clm_defect, diff_f32

            m_new_f32 = m_f32.mul(beta1).add(g_f32, alpha=1.0 - beta1)
            _encode_state_from_f32(exp_avg, m_new_f32, use_sr=use_sr, generator=gen_m)
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
        effective_eps = eps

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
                m_f32 = _decode_state_f32(m_c)

                clm_m = m_f32.mul(beta1_sq).add(g_clean, alpha=one_minus_beta1_sq)
                diff_f32 = (g_clean - m_f32).clamp(-1e18, 1e18)
                clm_defect = torch.abs(diff_f32).mul(lam)
                clm_env = torch.abs(m_f32).mul(beta1_sq).add(torch.abs(g_clean), alpha=one_minus_beta1_sq).add_(clm_defect).add_(effective_eps)
                u = clm_m / clm_env

                m_new = m_f32.mul(beta1).add(g_clean, alpha=1.0 - beta1)
                _encode_state_from_f32(m_c, m_new, use_sr=use_sr, generator=gen_m)

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
                    m_f32 = _decode_state_f32(m_c)

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
                m_f32 = _decode_state_f32(m_c)

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
                _encode_state_from_f32(m_c, m_new, use_sr=use_sr, generator=gen_m)

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

    def _step_gpu_group(
        self,
        group: Dict[str, Any],
        params_with_grad: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Optional[Tensor]],
        master_params: List[Optional[Tensor]],
        scratches: List[Tensor],
        steps: List[Any],
        orig_indices: Optional[List[int]] = None,
    ) -> None:
        if not params_with_grad:
            return

        buf_count = int(group["buffer"])
        master_mode = group["master_weights"]
        fsdp_sync = bool(group.get("fsdp_sync", False))
        cautious = bool(group["cautious"])
        bound = bool(group.get("bound", True))
        bound_type = group.get("bound_type", "coordinate")
        if isinstance(bound_type, int):
            bound_type = "radial" if bound_type == 1 else "coordinate"
        bound_ratio = float(group.get("bound_ratio", 0.05))

        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        beta1, beta2 = float(beta1), float(beta2)
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])
        tau = float(group["tau"])
        lam = float(group.get("lam", 0.3))
        clamp_min = float(group["cautious_clamp_min"])
        use_sr = bool(group["stochastic_rounding"])
        exec_mode = group["execution"]
        fused_single_pass = bool(group.get("fused_single_pass", True))
        ema_decay = float(group.get("ema_decay", 0.95 if buf_count == 1 else 0.80))
        seed = int(group.get("seed", getattr(self, "seed", 1337)))

        all_accelerator = all(p.device.type in ("cuda", "xpu") for p in params_with_grad)
        deterministic_mode = (
            torch.are_deterministic_algorithms_enabled()
            if hasattr(torch, "are_deterministic_algorithms_enabled")
            else False
        )

        if exec_mode in ("auto", "foreach", 0):
            exec_mode = "triton" if (HAS_TRITON and all_accelerator and not deterministic_mode) else "single"
        if exec_mode in ("triton", 1) and not (HAS_TRITON and all_accelerator):
            exec_mode = "single"

        has_master = (master_mode not in (0, "none", False)) and any(
            p.dtype != torch.float32 for p in params_with_grad
        )
        m_list: Optional[List[Tensor]] = (
            [mp if mp is not None else p for mp, p in zip(master_params, params_with_grad)]
            if has_master
            else None
        )
        v_list: List[Tensor] = [v for v in exp_avg_sqs if v is not None] if buf_count == 2 else []

        if exec_mode in ("triton", 1) and HAS_TRITON and all_accelerator:
            completed_indices: Set[int] = set()
            try:
                self._dispatch_triton(
                    params_with_grad, grads, exp_avgs, v_list, m_list, steps, scratches,
                    lr, beta1, beta2, eps, weight_decay, tau, lam,
                    cautious, clamp_min, buf_count, use_sr, bound, bound_type, bound_ratio,
                    fused_single_pass, ema_decay, fsdp_sync, seed,
                    completed_indices=completed_indices,
                    orig_indices=orig_indices,
                )
            except Exception as e:
                if not getattr(self, "_warned_triton_failure", False):
                    err_line = str(e).strip().splitlines()[0] if str(e).strip() else "Execution failed"
                    logger.warning(f"LuminaV: Triton dispatch failed ({err_line}), falling back to single.")
                    self._warned_triton_failure = True
                uncompleted = [
                    i for i in range(len(params_with_grad))
                    if (orig_indices[i] if orig_indices is not None else i) not in completed_indices
                ]
                if uncompleted:
                    for idx_u in uncompleted:
                        scratches[idx_u][:3].zero_()
                        scratches[idx_u][5].zero_()
                        st_u = int(steps[idx_u].item() if isinstance(steps[idx_u], torch.Tensor) else steps[idx_u])
                        if st_u <= 1:
                            scratches[idx_u][3] = 1.0
                            scratches[idx_u][4] = 1.0
                    sub_p = [params_with_grad[i] for i in uncompleted]
                    sub_g = [grads[i] for i in uncompleted]
                    sub_m = [exp_avgs[i] for i in uncompleted]
                    sub_mp = [m_list[i] for i in uncompleted] if m_list is not None else None
                    sub_st = [steps[i] for i in uncompleted]
                    sub_sc = [scratches[i] for i in uncompleted]
                    sub_orig = [orig_indices[i] if orig_indices is not None else i for i in uncompleted]
                    if buf_count == 2:
                        sub_v = [v_list[i] for i in uncompleted]
                        self._single_step_dual(
                            sub_p, sub_g, sub_m, sub_v, sub_mp, sub_st, sub_sc,
                            lr, beta1, beta2, eps, weight_decay, tau, cautious, clamp_min,
                            use_sr, bound, bound_type, bound_ratio,
                            fused_single_pass, ema_decay, fsdp_sync, seed,
                            orig_indices=sub_orig,
                        )
                    else:
                        self._single_step_tdclm(
                            sub_p, sub_g, sub_m, sub_mp, sub_st, sub_sc,
                            lr, beta1, eps, weight_decay, lam, cautious, clamp_min,
                            use_sr, bound, bound_type, bound_ratio,
                            fused_single_pass, ema_decay, fsdp_sync, seed,
                            orig_indices=sub_orig,
                        )
        else:
            if buf_count == 2:
                self._single_step_dual(
                    params_with_grad, grads, exp_avgs, v_list, m_list, steps, scratches,
                    lr, beta1, beta2, eps, weight_decay, tau, cautious, clamp_min,
                    use_sr, bound, bound_type, bound_ratio,
                    fused_single_pass, ema_decay, fsdp_sync, seed,
                    orig_indices=orig_indices,
                )
            else:
                self._single_step_tdclm(
                    params_with_grad, grads, exp_avgs, m_list, steps, scratches,
                    lr, beta1, eps, weight_decay, lam, cautious, clamp_min,
                    use_sr, bound, bound_type, bound_ratio,
                    fused_single_pass, ema_decay, fsdp_sync, seed,
                    orig_indices=orig_indices,
                )
