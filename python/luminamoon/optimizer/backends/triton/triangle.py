from __future__ import annotations

from typing import Any

from .common import (
    HAS_TRITON,
    _derive_seeds_and_weyl_dual,
    _derive_seeds_and_weyl_single,
    _load_state_buffer,
    _store_param,
    _store_state_buffer,
    _triton_apply_rpm,
    _triton_apply_step_bounds,
    _triton_apply_wd,
    _triton_temper_defect,
    tl,
    triton,
)

if HAS_TRITON:

    @triton.jit(do_not_specialize=["seed", "step_i32"])
    def _lumina_t1_single_pass_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr,
        scratch_ptr,
        n_elements, beta1, lr, weight_decay, eps, lam,
        bound_mode: tl.constexpr, bound_ratio, bound_floor,
        seed, step_i32, dtype_mode: tl.constexpr, state_dtype_mode: tl.constexpr, use_sr: tl.constexpr,
        cautious: tl.constexpr, has_master: tl.constexpr, use_rpm: tl.constexpr, cautious_wd: tl.constexpr,
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
        if use_rpm:
            m = _triton_apply_rpm(m, g, beta1, eps, BLOCK_SIZE)

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
            step_val, p_init, bound_ratio, bound_floor, bound_scale, bound_mode, dtype_mode, True
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

    @triton.jit(do_not_specialize=["seed", "step_i32"])
    def _lumina_t2_fused_single_pass_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
        scratch_ptr,
        n_elements, beta1, beta2, c1, c2, lr, weight_decay, eps,
        bound_mode: tl.constexpr, bound_ratio, bound_floor, variance_temper, inv_sqrt_bc2_prev,
        seed, step_i32, dtype_mode: tl.constexpr, state_dtype_mode: tl.constexpr, use_sr: tl.constexpr,
        cautious: tl.constexpr, has_master: tl.constexpr,
        use_rtv: tl.constexpr, use_rpm: tl.constexpr, cautious_wd: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        seed_m, seed_p, seed_v, weyl_m, weyl_v, weyl_p = _derive_seeds_and_weyl_dual(seed, step_i32)

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
        if use_rpm:
            m = _triton_apply_rpm(m, g, beta1, eps, BLOCK_SIZE)
        v = _load_state_buffer(exp_avg_sq_ptr, offsets, mask, state_dtype_mode)

        m_new = beta1 * m + (1.0 - beta1) * g
        diff = tl.maximum(tl.minimum(g - m, 1e18), -1e18)
        abs_d = tl.abs(diff)
        if use_rtv:
            d_mem = _triton_temper_defect(abs_d, v, m, inv_sqrt_bc2_prev, variance_temper, eps)
        else:
            d_mem = abs_d
        v_new = beta2 * v + (1.0 - beta2) * (d_mem * d_mem)

        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq
        num = beta1_sq * m + one_minus_beta1_sq * g
        sigma_t = tl.sqrt(tl.maximum(v_new, 0.0)) * c1
        den = beta1_sq * tl.abs(m) + one_minus_beta1_sq * tl.abs(g) + c2 * abs_d + sigma_t + eps

        u = num / den
        m_mask = tl.where((u * g) > 0.0, 1.0, 0.0)

        _store_state_buffer(exp_avg_ptr, offsets, m_new, mask, seed_m, weyl_m, state_dtype_mode, use_sr)
        _store_state_buffer(exp_avg_sq_ptr, offsets, v_new, mask, seed_v, weyl_v, state_dtype_mode, use_sr)

        delta_theta = (u * m_mask) * cautious_scale if cautious else u
        step_val = lr * delta_theta

        step_val = _triton_apply_step_bounds(
            step_val, p_init, bound_ratio, bound_floor, bound_scale, bound_mode, dtype_mode, True
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

    @triton.jit(do_not_specialize=["seed", "step_i32"])
    def _lumina_t2_single_pass_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
        scratch_ptr,
        n_elements, beta1, beta2, c1, c2, lr, weight_decay, eps,
        bound_mode: tl.constexpr, bound_ratio, bound_floor, variance_temper, inv_sqrt_bc2_prev,
        seed, step_i32, dtype_mode: tl.constexpr, state_dtype_mode: tl.constexpr, use_sr: tl.constexpr,
        has_master: tl.constexpr, use_rtv: tl.constexpr, use_rpm: tl.constexpr, cautious_wd: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        seed_m, seed_p, seed_v, weyl_m, weyl_v, weyl_p = _derive_seeds_and_weyl_dual(seed, step_i32)

        if has_master:
            p = tl.load(master_p_ptr + offsets, mask=mask, other=0.0)
        else:
            p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        p_init = p

        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        is_finite_g = (g == g) & (g < 1e38) & (g > -1e38)
        g = tl.where(is_finite_g, tl.maximum(tl.minimum(g, 1e18), -1e18), 0.0)
        m = _load_state_buffer(exp_avg_ptr, offsets, mask, state_dtype_mode)
        if use_rpm:
            m = _triton_apply_rpm(m, g, beta1, eps, BLOCK_SIZE)
        v = _load_state_buffer(exp_avg_sq_ptr, offsets, mask, state_dtype_mode)

        m_new = beta1 * m + (1.0 - beta1) * g
        diff = tl.maximum(tl.minimum(g - m, 1e18), -1e18)
        abs_d = tl.abs(diff)
        if use_rtv:
            d_mem = _triton_temper_defect(abs_d, v, m, inv_sqrt_bc2_prev, variance_temper, eps)
        else:
            d_mem = abs_d
        v_new = beta2 * v + (1.0 - beta2) * (d_mem * d_mem)

        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq
        num = beta1_sq * m + one_minus_beta1_sq * g
        sigma_t = tl.sqrt(tl.maximum(v_new, 0.0)) * c1
        den = beta1_sq * tl.abs(m) + one_minus_beta1_sq * tl.abs(g) + c2 * abs_d + sigma_t + eps

        u = num / den

        _store_state_buffer(exp_avg_ptr, offsets, m_new, mask, seed_m, weyl_m, state_dtype_mode, use_sr)
        _store_state_buffer(exp_avg_sq_ptr, offsets, v_new, mask, seed_v, weyl_v, state_dtype_mode, use_sr)

        bound_scale = tl.load(scratch_ptr + 4) if bound_mode == 1 else 1.0
        step_val = _triton_apply_step_bounds(
            lr * u, p_init, bound_ratio, bound_floor, bound_scale, bound_mode, dtype_mode, True
        )

        p = _triton_apply_wd(p_init, step_val, lr, weight_decay, cautious_wd)
        p_updated = p - step_val

        if has_master:
            tl.store(master_p_ptr + offsets, p_updated, mask=mask)
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, weyl_p, dtype_mode, False)
        else:
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, weyl_p, dtype_mode, use_sr)

    @triton.jit
    def _lumina_t2_pass1_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
        scratch_ptr,
        n_elements, beta1, beta2, c1, c2, eps, variance_temper, inv_sqrt_bc2_prev,
        bound_mode: tl.constexpr, dtype_mode: tl.constexpr, state_dtype_mode: tl.constexpr,
        cautious: tl.constexpr, has_master: tl.constexpr, use_rtv: tl.constexpr, use_rpm: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        is_finite_g = (g == g) & (g < 1e38) & (g > -1e38)
        g = tl.where(is_finite_g, tl.maximum(tl.minimum(g, 1e18), -1e18), 0.0)
        m = _load_state_buffer(exp_avg_ptr, offsets, mask, state_dtype_mode)
        if use_rpm:
            m = _triton_apply_rpm(m, g, beta1, eps, BLOCK_SIZE)
        v = _load_state_buffer(exp_avg_sq_ptr, offsets, mask, state_dtype_mode)

        diff = tl.maximum(tl.minimum(g - m, 1e18), -1e18)
        abs_d = tl.abs(diff)
        if use_rtv:
            d_mem = _triton_temper_defect(abs_d, v, m, inv_sqrt_bc2_prev, variance_temper, eps)
        else:
            d_mem = abs_d
        v_new = beta2 * v + (1.0 - beta2) * (d_mem * d_mem)

        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq
        num = beta1_sq * m + one_minus_beta1_sq * g
        sigma_t = tl.sqrt(tl.maximum(v_new, 0.0)) * c1
        den = beta1_sq * tl.abs(m) + one_minus_beta1_sq * tl.abs(g) + c2 * abs_d + sigma_t + eps

        u = num / den
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
    def _lumina_t2_pass2_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
        scratch_ptr,
        n_elements, total_elements, beta1, beta2, c1, c2, lr, weight_decay, eps, clamp_min,
        bound_mode: tl.constexpr, bound_ratio, bound_floor, variance_temper, inv_sqrt_bc2_prev,
        seed, step_i32, dtype_mode: tl.constexpr, state_dtype_mode: tl.constexpr, use_sr: tl.constexpr,
        cautious: tl.constexpr, has_master: tl.constexpr,
        use_rtv: tl.constexpr, use_rpm: tl.constexpr, cautious_wd: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        seed_m, seed_p, seed_v, weyl_m, weyl_v, weyl_p = _derive_seeds_and_weyl_dual(seed, step_i32)

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
            bound_scale = (1.0 + r) / (1.0 + r + r * r)

        if has_master:
            p = tl.load(master_p_ptr + offsets, mask=mask, other=0.0)
        else:
            p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        p_init = p

        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        is_finite_g = (g == g) & (g < 1e38) & (g > -1e38)
        g = tl.where(is_finite_g, tl.maximum(tl.minimum(g, 1e18), -1e18), 0.0)
        m = _load_state_buffer(exp_avg_ptr, offsets, mask, state_dtype_mode)
        if use_rpm:
            m = _triton_apply_rpm(m, g, beta1, eps, BLOCK_SIZE)
        v = _load_state_buffer(exp_avg_sq_ptr, offsets, mask, state_dtype_mode)

        m_new = beta1 * m + (1.0 - beta1) * g
        diff = tl.maximum(tl.minimum(g - m, 1e18), -1e18)
        abs_d = tl.abs(diff)
        if use_rtv:
            d_mem = _triton_temper_defect(abs_d, v, m, inv_sqrt_bc2_prev, variance_temper, eps)
        else:
            d_mem = abs_d
        v_new = beta2 * v + (1.0 - beta2) * (d_mem * d_mem)

        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq
        num = beta1_sq * m + one_minus_beta1_sq * g
        sigma_t = tl.sqrt(tl.maximum(v_new, 0.0)) * c1
        den = beta1_sq * tl.abs(m) + one_minus_beta1_sq * tl.abs(g) + c2 * abs_d + sigma_t + eps

        u = num / den
        m_mask = tl.where((u * g) > 0.0, 1.0, 0.0)

        _store_state_buffer(exp_avg_ptr, offsets, m_new, mask, seed_m, weyl_m, state_dtype_mode, use_sr)
        _store_state_buffer(exp_avg_sq_ptr, offsets, v_new, mask, seed_v, weyl_v, state_dtype_mode, use_sr)

        delta_theta = (u * m_mask) * cautious_scale if cautious else u
        step_val = lr * delta_theta

        step_val = _triton_apply_step_bounds(
            step_val, p_init, bound_ratio, bound_floor, bound_scale, bound_mode, dtype_mode, True
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
    _lumina_t1_single_pass_kernel: Any = None
    _lumina_t2_fused_single_pass_kernel: Any = None
    _lumina_t2_single_pass_kernel: Any = None
    _lumina_t2_pass1_kernel: Any = None
    _lumina_t2_pass2_kernel: Any = None
