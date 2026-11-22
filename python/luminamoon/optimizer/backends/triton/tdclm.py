from __future__ import annotations

from typing import Any

from .common import (
    HAS_TRITON,
    _derive_seeds_and_weyl_single,
    _load_state_buffer,
    _store_param,
    _store_state_buffer,
    _triton_apply_step_bounds,
    _triton_apply_wd,
    _triton_tanh_fast,
    tl,
    triton,
)

if HAS_TRITON:

    @triton.jit(do_not_specialize=["seed", "step_i32"])
    def _lumina_tdclm_single_pass_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr,
        scratch_ptr,
        n_elements, beta1, lr, weight_decay, eps, lam,
        bound_mode: tl.constexpr, bound_ratio, bound_floor,
        seed, step_i32, dtype_mode: tl.constexpr, state_dtype_mode: tl.constexpr, use_sr: tl.constexpr,
        cautious: tl.constexpr, has_master: tl.constexpr, cautious_wd: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        seed_m, seed_p, weyl_m, weyl_p = _derive_seeds_and_weyl_single(seed, step_i32)

        cautious_scale = tl.load(scratch_ptr + 3)
        bound_scale = tl.load(scratch_ptr + 4)

        if has_master:
            p = tl.load(master_p_ptr + offsets, mask=mask, other=0.0)
        else:
            p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        p_init = p

        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        is_finite_g = (g == g) & (g < 1e38) & (g > -1e38)
        g = tl.where(is_finite_g, tl.maximum(tl.minimum(g, 1e18), -1e18), 0.0)
        m = _load_state_buffer(exp_avg_ptr, offsets, mask, state_dtype_mode)

        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq
        m_new = beta1 * m + (1.0 - beta1) * g
        clm_m = beta1_sq * m + one_minus_beta1_sq * g
        diff = tl.maximum(tl.minimum(g - m, 1e18), -1e18)
        clm_env = beta1_sq * tl.abs(m) + one_minus_beta1_sq * tl.abs(g) + lam * tl.abs(diff) + eps

        u = clm_m / clm_env
        m_mask = tl.where((u * g) > 0.0, 1.0, 0.0)

        _store_state_buffer(exp_avg_ptr, offsets, m_new, mask, seed_m, weyl_m, state_dtype_mode, use_sr)

        delta_theta = (u * m_mask) * cautious_scale if cautious else u
        step_val = lr * delta_theta

        step_val = _triton_apply_step_bounds(
            step_val, p_init, bound_ratio, bound_floor, bound_scale, bound_mode, dtype_mode, False
        )

        p = _triton_apply_wd(p_init, step_val, lr, weight_decay, cautious_wd)
        p_updated = p - step_val

        if has_master:
            tl.store(master_p_ptr + offsets, p_updated, mask=mask)
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, weyl_p, dtype_mode, False)
        else:
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, weyl_p, dtype_mode, use_sr)

        if cautious:
            tl.atomic_add(scratch_ptr + 0, tl.sum(tl.where(mask, m_mask, 0.0), axis=0))

        if bound_mode == 1:
            u_eff = (u * m_mask) if cautious else u
            tl.atomic_add(scratch_ptr + 1, tl.sum(tl.where(mask, u_eff * u_eff, 0.0), axis=0))
            tl.atomic_add(scratch_ptr + 2, tl.sum(tl.where(mask, p_init * p_init, 0.0), axis=0))

    @triton.jit
    def _lumina_tdclm_pass1_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr,
        scratch_ptr,
        n_elements, beta1, eps, lam,
        bound_mode: tl.constexpr, dtype_mode: tl.constexpr, state_dtype_mode: tl.constexpr,
        cautious: tl.constexpr, has_master: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        is_finite_g = (g == g) & (g < 1e38) & (g > -1e38)
        g = tl.where(is_finite_g, tl.maximum(tl.minimum(g, 1e18), -1e18), 0.0)
        m = _load_state_buffer(exp_avg_ptr, offsets, mask, state_dtype_mode)

        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq
        diff = tl.maximum(tl.minimum(g - m, 1e18), -1e18)
        clm_m = beta1_sq * m + one_minus_beta1_sq * g
        clm_env = beta1_sq * tl.abs(m) + one_minus_beta1_sq * tl.abs(g) + lam * tl.abs(diff) + eps
        u = clm_m / clm_env
        m_mask = tl.where((u * g) > 0.0, 1.0, 0.0)

        if cautious:
            tl.atomic_add(scratch_ptr + 0, tl.sum(tl.where(mask, m_mask, 0.0), axis=0))

        if bound_mode == 1:
            u_eff = (u * m_mask) if cautious else u
            tl.atomic_add(scratch_ptr + 1, tl.sum(tl.where(mask, u_eff * u_eff, 0.0), axis=0))
            if has_master:
                p = tl.load(master_p_ptr + offsets, mask=mask, other=0.0)
            else:
                p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            tl.atomic_add(scratch_ptr + 2, tl.sum(tl.where(mask, p * p, 0.0), axis=0))

    @triton.jit(do_not_specialize=["seed", "step_i32"])
    def _lumina_tdclm_pass2_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr,
        scratch_ptr,
        n_elements, total_elements, beta1, lr, weight_decay, eps, lam, clamp_min,
        bound_mode: tl.constexpr, bound_ratio, bound_floor,
        seed, step_i32, dtype_mode: tl.constexpr, state_dtype_mode: tl.constexpr, use_sr: tl.constexpr,
        cautious: tl.constexpr, has_master: tl.constexpr, cautious_wd: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        is_triangle: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        seed_m, seed_p, weyl_m, weyl_p = _derive_seeds_and_weyl_single(seed, step_i32)

        cautious_scale = 1.0
        if cautious:
            mask_sum = tl.load(scratch_ptr + 0)
            m_bar = tl.minimum(tl.maximum(mask_sum / total_elements, clamp_min), 1.0)
            cautious_scale = 1.0 / m_bar

        bound_scale = 1.0
        if bound_mode == 1:
            u_sq_sum = tl.load(scratch_ptr + 1)
            p_sq_sum = tl.load(scratch_ptr + 2)
            norm_p = tl.sqrt(tl.maximum(p_sq_sum, 0.0))
            norm_delta = tl.sqrt(tl.maximum(u_sq_sum, 0.0)) * cautious_scale
            radius = bound_ratio * tl.maximum(norm_p, 1.0)
            r = (lr * norm_delta) / radius
            if is_triangle:
                bound_scale = (1.0 + r) / (1.0 + r + r * r)
            else:
                bound_scale = tl.where(r < 1e-4, 1.0, _triton_tanh_fast(r) / tl.maximum(r, 1e-8))

        if has_master:
            p = tl.load(master_p_ptr + offsets, mask=mask, other=0.0)
        else:
            p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        p_init = p

        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        is_finite_g = (g == g) & (g < 1e38) & (g > -1e38)
        g = tl.where(is_finite_g, tl.maximum(tl.minimum(g, 1e18), -1e18), 0.0)
        m = _load_state_buffer(exp_avg_ptr, offsets, mask, state_dtype_mode)

        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq
        diff = tl.maximum(tl.minimum(g - m, 1e18), -1e18)
        m_new = beta1 * m + (1.0 - beta1) * g
        clm_m = beta1_sq * m + one_minus_beta1_sq * g
        clm_env = beta1_sq * tl.abs(m) + one_minus_beta1_sq * tl.abs(g) + lam * tl.abs(diff) + eps
        u = clm_m / clm_env
        m_mask = tl.where((u * g) > 0.0, 1.0, 0.0)

        _store_state_buffer(exp_avg_ptr, offsets, m_new, mask, seed_m, weyl_m, state_dtype_mode, use_sr)

        delta_theta = (u * m_mask) * cautious_scale if cautious else u
        step_val = lr * delta_theta

        step_val = _triton_apply_step_bounds(
            step_val, p_init, bound_ratio, bound_floor, bound_scale, bound_mode, dtype_mode, is_triangle
        )

        p = _triton_apply_wd(p_init, step_val, lr, weight_decay, cautious_wd)
        p_updated = p - step_val

        if has_master:
            tl.store(master_p_ptr + offsets, p_updated, mask=mask)
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, weyl_p, dtype_mode, False)
        else:
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, weyl_p, dtype_mode, use_sr)

        if tl.program_id(axis=0) == 0:
            if cautious:
                tl.store(scratch_ptr + 3, cautious_scale)
            if bound_mode == 1:
                tl.store(scratch_ptr + 4, bound_scale)

else:
    _lumina_tdclm_single_pass_kernel: Any = None
    _lumina_tdclm_pass1_kernel: Any = None
    _lumina_tdclm_pass2_kernel: Any = None
