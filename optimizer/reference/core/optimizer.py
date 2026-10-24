from __future__ import annotations

import logging
from typing import Callable, Optional, Set

import torch
from torch import Tensor

from ..kernels import HAS_TRITON
from ..triton_backend import LuminaVTritonMixin

__all__ = ["LuminaV"]
__version__ = "1.5.3"
__author__ = "Silver Moon (cloverxion)"
__organization__ = "Lumina Moon (cloverx-id)"
__license__ = "Apache-2.0"

logger = logging.getLogger("LuminaV")


class LuminaV(LuminaVTritonMixin):
    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:  # type: ignore
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self._validate_and_normalize_group(group)
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            master_params = []
            steps = []
            scratches = []
            buf_count = group["buffer"]
            master_mode = group["master_weights"]
            fsdp_sync = group.get("fsdp_sync", False)
            cautious = group["cautious"]
            bound = group.get("bound", True)
            bound_type = group.get("bound_type", "coordinate")
            bound_ratio = group.get("bound_ratio", 0.05)
            centered_defect = group.get("centered_defect", False)

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("LuminaV does not support sparse gradients.")
                if p.numel() == 0 and not fsdp_sync:
                    continue

                state = self.state[p]
                target_state_dtype = torch.float32 if master_mode == "full" else p.dtype
                if len(state) == 0:
                    state["step"] = 0
                    if master_mode != "none" and p.dtype != torch.float32:
                        state["master_param"] = p.detach().clone().float().contiguous()

                    state["exp_avg"] = torch.zeros_like(p, dtype=target_state_dtype, memory_format=torch.contiguous_format)
                    if buf_count == 2:
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=target_state_dtype, memory_format=torch.contiguous_format)

                    scratch = torch.zeros((6,), device=p.device, dtype=torch.float32)
                    scratch[3] = 1.0
                    scratch[4] = 1.0
                    state["scratch"] = scratch
                    state["buf_count"] = buf_count
                else:
                    if master_mode != "none":
                        if p.dtype == torch.float32:
                            state.pop("master_param", None)
                        elif "master_param" not in state:
                            state["master_param"] = p.detach().clone().float().contiguous()
                        elif state["master_param"].device != p.device or state["master_param"].dtype != torch.float32:
                            state["master_param"] = state["master_param"].to(device=p.device, dtype=torch.float32, non_blocking=True)
                    else:
                        state.pop("master_param", None)

                    if "scratch" not in state or state["scratch"].numel() < 6:
                        scratch = torch.zeros((6,), device=p.device, dtype=torch.float32)
                        scratch[3] = 1.0
                        scratch[4] = 1.0
                        if "scratch" in state and isinstance(state["scratch"], Tensor):
                            old_sc = state["scratch"].to(device=p.device, dtype=torch.float32)
                            scratch[:old_sc.numel()].copy_(old_sc)
                        state["scratch"] = scratch
                        state["buf_count"] = buf_count
                    elif state["scratch"].device != p.device or state["scratch"].dtype != torch.float32:
                        state["scratch"] = state["scratch"].to(device=p.device, dtype=torch.float32, non_blocking=True)
                        state["buf_count"] = buf_count
                    elif state.get("buf_count") != buf_count:
                        state["scratch"].zero_()
                        state["scratch"][3] = 1.0
                        state["scratch"][4] = 1.0
                        state["buf_count"] = buf_count

                    if not cautious:
                        state["scratch"][3] = 1.0
                    if not (bound and bound_type == "radial" and bound_ratio > 0.0):
                        state["scratch"][4] = 1.0

                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p, dtype=target_state_dtype, memory_format=torch.contiguous_format)
                    elif state["exp_avg"].device != p.device or state["exp_avg"].dtype != target_state_dtype:
                        state["exp_avg"] = state["exp_avg"].to(device=p.device, dtype=target_state_dtype, non_blocking=True)

                    if buf_count == 2:
                        if "exp_avg_sq" not in state:
                            state["exp_avg_sq"] = torch.zeros_like(p, dtype=target_state_dtype, memory_format=torch.contiguous_format)
                        elif state["exp_avg_sq"].device != p.device or state["exp_avg_sq"].dtype != target_state_dtype:
                            state["exp_avg_sq"] = state["exp_avg_sq"].to(device=p.device, dtype=target_state_dtype, non_blocking=True)
                    else:
                        state.pop("exp_avg_sq", None)

                params_with_grad.append(p)
                grads.append(grad)
                exp_avgs.append(state["exp_avg"])
                scratches.append(state["scratch"])
                if buf_count == 2:
                    exp_avg_sqs.append(state["exp_avg_sq"])

                if master_mode != "none":
                    master_params.append(state["master_param"] if "master_param" in state else p)

                curr_step = state.get("step", 0)
                step_val = int(curr_step.item() if isinstance(curr_step, torch.Tensor) else curr_step) + 1
                state["step"] = step_val
                steps.append(step_val)

            if not params_with_grad:
                continue

            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            tau = group["tau"]
            lam = group.get("lam", 0.3)
            clamp_min = group["cautious_clamp_min"]
            use_sr = group["stochastic_rounding"]
            exec_mode = group["execution"]
            fused_single_pass = group.get("fused_single_pass", True)
            ema_decay = group.get("ema_decay", 0.8)
            seed = group.get("seed", self.seed)

            all_accelerator = all(p.device.type in ("cuda", "xpu") for p in params_with_grad)

            deterministic_mode = False
            if hasattr(torch, "are_deterministic_algorithms_enabled"):
                deterministic_mode = torch.are_deterministic_algorithms_enabled()

            if exec_mode in ("auto", "foreach"):
                if HAS_TRITON and all_accelerator and not deterministic_mode:
                    exec_mode = "triton"
                else:
                    exec_mode = "single"

            if exec_mode == "triton" and not (HAS_TRITON and all_accelerator):
                exec_mode = "single"

            has_master = (master_mode != "none") and any(p.dtype != torch.float32 for p in params_with_grad)
            m_list = master_params if has_master else None

            if exec_mode == "triton" and HAS_TRITON and all_accelerator:
                completed_indices: Set[int] = set()
                try:
                    self._dispatch_triton(
                        params_with_grad, grads, exp_avgs, exp_avg_sqs, m_list, steps, scratches,
                        lr, beta1, beta2, eps, weight_decay, tau, lam,
                        cautious, clamp_min, buf_count, use_sr, bound, bound_type, bound_ratio,
                        fused_single_pass, ema_decay, fsdp_sync, seed, centered_defect,
                        completed_indices=completed_indices
                    )
                except Exception as e:
                    if not self._warned_triton_failure:
                        err_line = str(e).strip().splitlines()[0] if str(e).strip() else "Execution failed"
                        logger.warning(f"LuminaV: Triton dispatch failed ({err_line}), falling back to single.")
                        self._warned_triton_failure = True
                    uncompleted = [i for i in range(len(params_with_grad)) if i not in completed_indices]
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
                        sub_v = [exp_avg_sqs[i] for i in uncompleted] if buf_count == 2 else None
                        sub_mp = [m_list[i] for i in uncompleted] if m_list is not None else None
                        sub_st = [steps[i] for i in uncompleted]
                        sub_sc = [scratches[i] for i in uncompleted]
                        self._dispatch_single(
                            sub_p, sub_g, sub_m, sub_v, sub_mp, sub_st, sub_sc,
                            lr, beta1, beta2, eps, weight_decay, tau, lam,
                            cautious, clamp_min, buf_count, use_sr, bound, bound_type, bound_ratio,
                            fused_single_pass, ema_decay, fsdp_sync, seed, centered_defect,
                            orig_indices=uncompleted
                        )
            else:
                self._dispatch_single(
                    params_with_grad, grads, exp_avgs, exp_avg_sqs, m_list, steps, scratches,
                    lr, beta1, beta2, eps, weight_decay, tau, lam,
                    cautious, clamp_min, buf_count, use_sr, bound, bound_type, bound_ratio,
                    fused_single_pass, ema_decay, fsdp_sync, seed, centered_defect
                )

        return loss
