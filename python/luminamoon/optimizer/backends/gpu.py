from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import torch
from torch import Tensor

from ..config import StepOptions
from .eager import step_eager_group
from .triton import HAS_TRITON, dispatch_triton_group

logger = logging.getLogger("LuminaV.GPU")


def step_gpu_group(
    opt: Any,
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

    opts = StepOptions.from_group(group, fallback_seed=getattr(opt, "seed", 1337))
    all_accelerator = all(p.device.type in ("cuda", "xpu") for p in params_with_grad)
    deterministic_mode = (
        torch.are_deterministic_algorithms_enabled()
        if hasattr(torch, "are_deterministic_algorithms_enabled")
        else False
    )

    exec_mode = opts.execution
    if exec_mode == "auto":
        exec_mode = "triton" if (HAS_TRITON and all_accelerator and not deterministic_mode) else "eager"
    elif exec_mode == "triton" and not (HAS_TRITON and all_accelerator):
        exec_mode = "eager"

    has_master = (opts.master_mode > 0) and any(p.dtype != torch.float32 for p in params_with_grad)
    m_list: Optional[List[Tensor]] = (
        [mp if mp is not None else p for mp, p in zip(master_params, params_with_grad)]
        if has_master
        else None
    )
    v_list: List[Optional[Tensor]] = exp_avg_sqs if opts.is_dual else []

    if exec_mode == "triton" and HAS_TRITON and all_accelerator:
        completed_indices: Set[int] = set()
        try:
            dispatch_triton_group(
                opt, opts, params_with_grad, grads, exp_avgs, v_list, m_list,
                steps, scratches, orig_indices=orig_indices, completed_indices=completed_indices,
            )
            return
        except Exception as exc:
            if not getattr(opt, "_warned_triton_failure", False):
                err_line = str(exc).strip().splitlines()[0] if str(exc).strip() else "Execution failed"
                logger.warning("LuminaV: Triton dispatch failed (%s), falling back to eager.", err_line)
                opt._warned_triton_failure = True

            uncompleted = [
                i
                for i in range(len(params_with_grad))
                if (orig_indices[i] if orig_indices is not None else i) not in completed_indices
            ]
            if not uncompleted:
                return
            for idx_u in uncompleted:
                scratches[idx_u][:3].zero_()
                scratches[idx_u][5].zero_()
                st_u = int(steps[idx_u].item() if isinstance(steps[idx_u], torch.Tensor) else steps[idx_u])
                if st_u <= 1:
                    scratches[idx_u][3] = 1.0
                    scratches[idx_u][4] = 1.0
            step_eager_group(
                opt, opts,
                [params_with_grad[i] for i in uncompleted],
                [grads[i] for i in uncompleted],
                [exp_avgs[i] for i in uncompleted],
                [v_list[i] for i in uncompleted] if opts.is_dual else [],
                [m_list[i] for i in uncompleted] if m_list is not None else None,
                [steps[i] for i in uncompleted],
                [scratches[i] for i in uncompleted],
                orig_indices=[orig_indices[i] if orig_indices is not None else i for i in uncompleted],
            )
            return

    step_eager_group(
        opt, opts, params_with_grad, grads, exp_avgs, v_list, m_list,
        steps, scratches, orig_indices=orig_indices,
    )
