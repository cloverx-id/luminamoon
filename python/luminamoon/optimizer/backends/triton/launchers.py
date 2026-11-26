from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
from torch import Tensor

from ...config import StepOptions
from .common import _lumina_post_step_ema_kernel, triton
from .dual import (
    _lumina_dual_fused_single_pass_kernel,
    _lumina_dual_pass1_kernel,
    _lumina_dual_pass2_kernel,
    _lumina_dual_single_pass_kernel,
)
from .tdclm import (
    _lumina_tdclm_pass1_kernel,
    _lumina_tdclm_pass2_kernel,
    _lumina_tdclm_single_pass_kernel,
)
from .triangle import (
    _lumina_t1_single_pass_kernel,
    _lumina_t2_fused_single_pass_kernel,
    _lumina_t2_pass1_kernel,
    _lumina_t2_pass2_kernel,
    _lumina_t2_single_pass_kernel,
)

logger = logging.getLogger("LuminaV.GPU")


def dtype_mode_code(dt: torch.dtype) -> int:
    if dt == torch.bfloat16:
        return 1
    if dt == torch.float16:
        return 2
    return 0


def block_config(n_elements: int) -> Tuple[int, int]:
    return (2048, 8) if n_elements >= 2_097_152 else (1024, 4)


def _get_or_alloc_contig(
    state: dict[str, Any],
    key: str,
    src: Tensor,
    can_reuse: bool,
) -> Tensor:
    need_alloc = key not in state or state[key].shape != src.shape or state[key].dtype != src.dtype
    if need_alloc:
        state[key] = torch.empty_like(src, memory_format=torch.contiguous_format)
    buf = state[key]
    if need_alloc or not can_reuse:
        buf.copy_(src)
    return buf


def prepare_triton_buffers(
    opt: Any,
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Optional[Tensor],
    mp: Optional[Tensor],
    param_has_master: bool,
    reuse_contig: bool = False,
):
    state = opt.state.get(p) if hasattr(opt, "state") and hasattr(opt.state, "get") else None
    n_elements = p.numel()
    dt_mode = dtype_mode_code(p.dtype)
    s_dt_mode = dtype_mode_code(exp_avg.dtype)

    cached_plan = state.get("_plan") if state is not None else None
    if (
        cached_plan is None
        or len(cached_plan) != 6
        or cached_plan[0] != n_elements
        or cached_plan[4] != dt_mode
        or cached_plan[5] != s_dt_mode
    ):
        b_size, n_warps = block_config(n_elements)
        c_grid = (triton.cdiv(n_elements, b_size),)
        if state is not None:
            state["_plan"] = (n_elements, b_size, n_warps, c_grid, dt_mode, s_dt_mode)
    else:
        n_elements, b_size, n_warps, c_grid, dt_mode, s_dt_mode = cached_plan

    is_all_contig = (
        p.is_contiguous()
        and grad.is_contiguous()
        and exp_avg.is_contiguous()
        and (exp_avg_sq is None or exp_avg_sq.is_contiguous())
        and (not param_has_master or (mp is not None and mp.is_contiguous()))
    )
    if is_all_contig:
        mp_contig = mp if (param_has_master and mp is not None) else p
        return (
            n_elements, b_size, n_warps, c_grid, dt_mode, s_dt_mode, True,
            p, grad, exp_avg, exp_avg_sq, mp_contig,
        )

    if not getattr(opt, "_warned_non_contiguous", False):
        logger.warning(
            "LuminaV detected non-contiguous tensor buffer(s). An explicit contiguous copy "
            "was created for Triton kernel execution."
        )
        opt._warned_non_contiguous = True

    if state is None:
        state = opt.state[p]
    curr_step_tag = state.get("step")
    can_reuse = bool(reuse_contig and curr_step_tag is not None and state.get("_contig_step") == curr_step_tag)

    p_contig = p if p.is_contiguous() else _get_or_alloc_contig(state, "p_contig", p, can_reuse)
    grad_contig = grad if grad.is_contiguous() else grad.contiguous()
    m_contig = exp_avg if exp_avg.is_contiguous() else _get_or_alloc_contig(state, "exp_avg_contig", exp_avg, can_reuse)
    v_contig = (
        None
        if exp_avg_sq is None
        else (exp_avg_sq if exp_avg_sq.is_contiguous() else _get_or_alloc_contig(state, "exp_avg_sq_contig", exp_avg_sq, can_reuse))
    )
    if param_has_master and mp is not None:
        mp_contig = mp if mp.is_contiguous() else _get_or_alloc_contig(state, "mp_contig", mp, can_reuse)
    else:
        mp_contig = p_contig

    state["_contig_step"] = curr_step_tag
    return (
        n_elements, b_size, n_warps, c_grid, dt_mode, s_dt_mode, False,
        p_contig, grad_contig, m_contig, v_contig, mp_contig,
    )


def copy_back_non_contig(
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


def launch_pass1(
    grid: Tuple[int, ...],
    p: Tensor,
    mp: Tensor,
    g: Tensor,
    m: Tensor,
    v: Optional[Tensor],
    sc: Tensor,
    n_elements: int,
    opts: StepOptions,
    c1: float,
    c2: float,
    dt_mode: int,
    s_dt_mode: int,
    has_master: bool,
    b_size: int,
    n_warps: int,
    *,
    step: int = 1,
) -> None:
    use_rtv = opts.use_rtv(step)
    inv_sqrt_bc2_prev = opts.inv_sqrt_bc2_prev(step)
    if opts.buffer == "t2":
        _lumina_t2_pass1_kernel[grid](
            p, mp, g, m, v, sc, n_elements, opts.beta1, opts.beta2, c1, c2, opts.eps,
            opts.variance_temper, inv_sqrt_bc2_prev,
            opts.bound_mode, dt_mode, s_dt_mode, opts.cautious, has_master, use_rtv, opts.use_rpm,
            BLOCK_SIZE=b_size, num_warps=n_warps, enable_fp_fusion=True,
        )
    elif opts.buffer == 2:
        _lumina_dual_pass1_kernel[grid](
            p, mp, g, m, v, sc, n_elements, opts.beta1, opts.beta2, c1, c2, opts.eps,
            opts.variance_temper, inv_sqrt_bc2_prev,
            opts.bound_mode, dt_mode, s_dt_mode, opts.cautious, has_master, use_rtv, opts.use_rpm,
            BLOCK_SIZE=b_size, num_warps=n_warps, enable_fp_fusion=True,
        )
    else:
        _lumina_tdclm_pass1_kernel[grid](
            p, mp, g, m, sc, n_elements, opts.beta1, opts.eps, opts.lam,
            opts.bound_mode, dt_mode, s_dt_mode, opts.cautious, has_master, opts.use_rpm,
            BLOCK_SIZE=b_size, num_warps=n_warps, enable_fp_fusion=True,
        )


def launch_pass2(
    grid: Tuple[int, ...],
    p: Tensor,
    mp: Tensor,
    g: Tensor,
    m: Tensor,
    v: Optional[Tensor],
    sc: Tensor,
    n_elements: int,
    total_elements: float,
    opts: StepOptions,
    c1: float,
    c2: float,
    lr: float,
    weight_decay: float,
    kernel_seed: int,
    step_i32: int,
    dt_mode: int,
    s_dt_mode: int,
    eff_use_sr: bool,
    has_master: bool,
    b_size: int,
    n_warps: int,
    *,
    step: int = 1,
) -> None:
    use_rtv = opts.use_rtv(step)
    inv_sqrt_bc2_prev = opts.inv_sqrt_bc2_prev(step)
    if opts.buffer == "t2":
        _lumina_t2_pass2_kernel[grid](
            p, mp, g, m, v, sc, n_elements, total_elements,
            opts.beta1, opts.beta2, c1, c2, lr, weight_decay, opts.eps, opts.clamp_min,
            opts.bound_mode, opts.bound_ratio, opts.bound_floor, opts.variance_temper, inv_sqrt_bc2_prev,
            kernel_seed, step_i32, dt_mode, s_dt_mode, eff_use_sr, opts.cautious, has_master,
            use_rtv, opts.use_rpm, opts.cautious_weight_decay, BLOCK_SIZE=b_size, num_warps=n_warps, enable_fp_fusion=True,
        )
    elif opts.buffer == 2:
        _lumina_dual_pass2_kernel[grid](
            p, mp, g, m, v, sc, n_elements, total_elements,
            opts.beta1, opts.beta2, c1, c2, lr, weight_decay, opts.eps, opts.clamp_min,
            opts.bound_mode, opts.bound_ratio, opts.bound_floor, opts.variance_temper, inv_sqrt_bc2_prev,
            kernel_seed, step_i32, dt_mode, s_dt_mode, eff_use_sr, opts.cautious, has_master,
            use_rtv, opts.use_rpm, opts.cautious_weight_decay, BLOCK_SIZE=b_size, num_warps=n_warps, enable_fp_fusion=True,
        )
    else:
        _lumina_tdclm_pass2_kernel[grid](
            p, mp, g, m, sc, n_elements, total_elements,
            opts.beta1, lr, weight_decay, opts.eps, opts.lam, opts.clamp_min,
            opts.bound_mode, opts.bound_ratio, opts.bound_floor, kernel_seed, step_i32, dt_mode, s_dt_mode,
            eff_use_sr, opts.cautious, has_master, opts.use_rpm, opts.cautious_weight_decay, BLOCK_SIZE=b_size,
            is_triangle=opts.is_triangle, num_warps=n_warps, enable_fp_fusion=True,
        )


def launch_single_pass(
    grid: Tuple[int, ...],
    p: Tensor,
    mp: Tensor,
    g: Tensor,
    m: Tensor,
    v: Optional[Tensor],
    sc: Tensor,
    n_elements: int,
    opts: StepOptions,
    c1: float,
    c2: float,
    lr: float,
    weight_decay: float,
    kernel_seed: int,
    step_i32: int,
    dt_mode: int,
    s_dt_mode: int,
    eff_use_sr: bool,
    has_master: bool,
    b_size: int,
    n_warps: int,
    *,
    fused_ema: bool,
    step: int = 1,
) -> None:
    use_rtv = opts.use_rtv(step)
    inv_sqrt_bc2_prev = opts.inv_sqrt_bc2_prev(step)
    if opts.buffer == "t2":
        if fused_ema:
            _lumina_t2_fused_single_pass_kernel[grid](
                p, mp, g, m, v, sc, n_elements, opts.beta1, opts.beta2, c1, c2, lr, weight_decay, opts.eps,
                opts.bound_mode, opts.bound_ratio, opts.bound_floor, opts.variance_temper, inv_sqrt_bc2_prev,
                kernel_seed, step_i32, dt_mode, s_dt_mode, eff_use_sr, opts.cautious, has_master,
                use_rtv, opts.use_rpm, opts.cautious_weight_decay, BLOCK_SIZE=b_size, num_warps=n_warps, enable_fp_fusion=True,
            )
        else:
            _lumina_t2_single_pass_kernel[grid](
                p, mp, g, m, v, sc, n_elements, opts.beta1, opts.beta2, c1, c2, lr, weight_decay, opts.eps,
                opts.bound_mode, opts.bound_ratio, opts.bound_floor, opts.variance_temper, inv_sqrt_bc2_prev,
                kernel_seed, step_i32, dt_mode, s_dt_mode, eff_use_sr, has_master,
                use_rtv, opts.use_rpm, opts.cautious_weight_decay, BLOCK_SIZE=b_size, num_warps=n_warps, enable_fp_fusion=True,
            )
    elif opts.buffer == 2:
        if fused_ema:
            _lumina_dual_fused_single_pass_kernel[grid](
                p, mp, g, m, v, sc, n_elements, opts.beta1, opts.beta2, c1, c2, lr, weight_decay, opts.eps,
                opts.bound_mode, opts.bound_ratio, opts.bound_floor, opts.variance_temper, inv_sqrt_bc2_prev,
                kernel_seed, step_i32, dt_mode, s_dt_mode, eff_use_sr, opts.cautious, has_master,
                use_rtv, opts.use_rpm, opts.cautious_weight_decay, BLOCK_SIZE=b_size, num_warps=n_warps, enable_fp_fusion=True,
            )
        else:
            _lumina_dual_single_pass_kernel[grid](
                p, mp, g, m, v, sc, n_elements, opts.beta1, opts.beta2, c1, c2, lr, weight_decay, opts.eps,
                opts.bound_mode, opts.bound_ratio, opts.bound_floor, opts.variance_temper, inv_sqrt_bc2_prev,
                kernel_seed, step_i32, dt_mode, s_dt_mode, eff_use_sr, has_master,
                use_rtv, opts.use_rpm, opts.cautious_weight_decay, BLOCK_SIZE=b_size, num_warps=n_warps, enable_fp_fusion=True,
            )
    else:
        kernel = _lumina_t1_single_pass_kernel if opts.is_triangle else _lumina_tdclm_single_pass_kernel
        kernel[grid](
            p, mp, g, m, sc, n_elements, opts.beta1, lr, weight_decay, opts.eps, opts.lam,
            opts.bound_mode, opts.bound_ratio, opts.bound_floor, kernel_seed, step_i32, dt_mode, s_dt_mode,
            eff_use_sr, opts.cautious, has_master, opts.use_rpm, opts.cautious_weight_decay,
            BLOCK_SIZE=b_size, num_warps=n_warps, enable_fp_fusion=True,
        )


def launch_post_step_ema(
    sc: Tensor,
    total_elements: float,
    opts: StepOptions,
    *,
    lr_override: Optional[float] = None,
) -> None:
    lr = opts.lr if lr_override is None else lr_override
    _lumina_post_step_ema_kernel[(1,)](
        sc, total_elements, lr, opts.clamp_min, opts.ema_decay,
        opts.bound_mode, opts.bound_ratio, opts.cautious,
        is_triangle=opts.is_triangle, num_warps=1, enable_fp_fusion=True,
    )
