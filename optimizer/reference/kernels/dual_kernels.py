from __future__ import annotations

from typing import Any

from .common import (
    HAS_TRITON,
    _store_param,
    _store_state_buffer,
    _triton_tanh_fast,
    tl,
    triton,
)

if HAS_TRITON:
    @triton.jit
    def _lumina_dual_fused_single_pass_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
        scratch_ptr,
        n_elements, total_elements, beta1, beta2, c1, c2, lr, weight_decay,
        bound_mode: tl.constexpr, bound_ratio,
        seed, dtype_mode: tl.constexpr, state_dtype_mode: tl.constexpr, use_sr: tl.constexpr,
        cautious: tl.constexpr, has_master: tl.constexpr, centered_defect: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        seed_m = seed
        seed_p = (seed + 0x55555555) & 0x7FFFFFFF

        mask_sum_ptr = scratch_ptr + 0
        u_sq_sum_ptr = scratch_ptr + 1
        p_sq_sum_ptr = scratch_ptr + 2
        cautious_scale_ptr = scratch_ptr + 3
        bound_scale_ptr = scratch_ptr + 4

        cautious_scale = tl.load(cautious_scale_ptr)
        bound_scale = tl.load(bound_scale_ptr)

        if has_master:
            p = tl.load(master_p_ptr + offsets, mask=mask, other=0.0)
        else:
            p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        p_init = p

        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        is_finite_g = (g == g) & (g < 1e38) & (g > -1e38)
        g = tl.where(is_finite_g, tl.maximum(tl.minimum(g, 1e18), -1e18), 0.0)
        m = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        m_new = beta1 * m + (1.0 - beta1) * g
        clm_m = beta1 * m_new + (1.0 - beta1) * g
        diff = tl.maximum(tl.minimum(g - m_new, 1e18), -1e18)
        if centered_defect:
            diff_mean = (tl.load(scratch_ptr + 5) / total_elements) * beta1
            diff = tl.maximum(tl.minimum(diff - diff_mean, 1e18), -1e18)
        v_new = beta2 * v + (1.0 - beta2) * (diff * diff)
        if dtype_mode == 2 or state_dtype_mode == 2:
            v_new = tl.minimum(v_new, 65000.0)

        sigma = tl.sqrt(tl.maximum(v_new, 0.0)) * c1 + c2
        u = _triton_tanh_fast(clm_m / tl.maximum(sigma, 1e-12))
        m_mask = tl.where((u * g) > 0.0, 1.0, 0.0)

        _store_state_buffer(exp_avg_ptr, offsets, m_new, mask, seed_m, state_dtype_mode, use_sr)
        _store_state_buffer(exp_avg_sq_ptr, offsets, v_new, mask, seed_m, state_dtype_mode, False)

        if cautious:
            delta_theta = (u * m_mask) * cautious_scale
        else:
            delta_theta = u
        step_val = lr * delta_theta

        if bound_mode == 1:
            step_val = step_val * bound_scale
        elif bound_mode == 2:
            B_i = bound_ratio * tl.sqrt(p_init * p_init + 1e-4)
            step_val = B_i * _triton_tanh_fast(step_val / B_i)

        if dtype_mode == 2:
            b_safe = tl.maximum(tl.abs(p_init) * 0.01, 1.5e-4)
            step_val = b_safe * _triton_tanh_fast(step_val / b_safe)

        if weight_decay != 0.0:
            p = p * (1.0 - lr * weight_decay)
        p_updated = p - step_val

        if has_master:
            tl.store(master_p_ptr + offsets, p_updated, mask=mask)
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, dtype_mode, False)
        else:
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, dtype_mode, use_sr)

        if cautious:
            block_sum = tl.sum(tl.where(mask, m_mask, 0.0), axis=0)
            tl.atomic_add(mask_sum_ptr, block_sum)

        if bound_mode == 1:
            u_eff = (u * m_mask) if cautious else u
            block_u_sq = tl.sum(tl.where(mask, u_eff * u_eff, 0.0), axis=0)
            tl.atomic_add(u_sq_sum_ptr, block_u_sq)

            block_p_sq = tl.sum(tl.where(mask, p_init * p_init, 0.0), axis=0)
            tl.atomic_add(p_sq_sum_ptr, block_p_sq)

    @triton.jit
    def _lumina_dual_pass1_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
        scratch_ptr,
        n_elements, total_elements, beta1, beta2, c1, c2,
        bound_mode: tl.constexpr, dtype_mode: tl.constexpr, state_dtype_mode: tl.constexpr,
        cautious: tl.constexpr, has_master: tl.constexpr,
        centered_defect: tl.constexpr, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        mask_sum_ptr = scratch_ptr + 0
        u_sq_sum_ptr = scratch_ptr + 1
        p_sq_sum_ptr = scratch_ptr + 2

        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        is_finite_g = (g == g) & (g < 1e38) & (g > -1e38)
        g = tl.where(is_finite_g, tl.maximum(tl.minimum(g, 1e18), -1e18), 0.0)
        m = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        m_new = beta1 * m + (1.0 - beta1) * g
        clm_m = beta1 * m_new + (1.0 - beta1) * g
        diff = tl.maximum(tl.minimum(g - m_new, 1e18), -1e18)
        if centered_defect:
            diff_mean = (tl.load(scratch_ptr + 5) / total_elements) * beta1
            diff = tl.maximum(tl.minimum(diff - diff_mean, 1e18), -1e18)
        v_new = beta2 * v + (1.0 - beta2) * (diff * diff)
        if dtype_mode == 2 or state_dtype_mode == 2:
            v_new = tl.minimum(v_new, 65000.0)

        sigma = tl.sqrt(tl.maximum(v_new, 0.0)) * c1 + c2
        u = _triton_tanh_fast(clm_m / tl.maximum(sigma, 1e-12))
        m_mask = tl.where((u * g) > 0.0, 1.0, 0.0)

        if cautious:
            block_sum = tl.sum(tl.where(mask, m_mask, 0.0), axis=0)
            tl.atomic_add(mask_sum_ptr, block_sum)

        if bound_mode == 1:
            u_eff = (u * m_mask) if cautious else u
            block_u_sq = tl.sum(tl.where(mask, u_eff * u_eff, 0.0), axis=0)
            tl.atomic_add(u_sq_sum_ptr, block_u_sq)

            if has_master:
                p = tl.load(master_p_ptr + offsets, mask=mask, other=0.0)
            else:
                p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            block_p_sq = tl.sum(tl.where(mask, p * p, 0.0), axis=0)
            tl.atomic_add(p_sq_sum_ptr, block_p_sq)

    @triton.jit
    def _lumina_dual_pass2_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
        scratch_ptr,
        n_elements, total_elements, beta1, beta2, c1, c2, lr, weight_decay, clamp_min,
        bound_mode: tl.constexpr, bound_ratio,
        seed, dtype_mode: tl.constexpr, state_dtype_mode: tl.constexpr, use_sr: tl.constexpr,
        cautious: tl.constexpr, has_master: tl.constexpr, centered_defect: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        seed_m = seed
        seed_p = (seed + 0x55555555) & 0x7FFFFFFF

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
            R = bound_ratio * tl.maximum(norm_p, 1.0)
            r = (lr * norm_delta) / R
            bound_scale = tl.where(r < 1e-4, 1.0, _triton_tanh_fast(r) / tl.maximum(r, 1e-8))

        if has_master:
            p = tl.load(master_p_ptr + offsets, mask=mask, other=0.0)
        else:
            p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        p_init = p

        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        is_finite_g = (g == g) & (g < 1e38) & (g > -1e38)
        g = tl.where(is_finite_g, tl.maximum(tl.minimum(g, 1e18), -1e18), 0.0)
        m = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        m_new = beta1 * m + (1.0 - beta1) * g
        clm_m = beta1 * m_new + (1.0 - beta1) * g
        diff = tl.maximum(tl.minimum(g - m_new, 1e18), -1e18)
        if centered_defect:
            diff_mean = (tl.load(scratch_ptr + 5) / total_elements) * beta1
            diff = tl.maximum(tl.minimum(diff - diff_mean, 1e18), -1e18)
        v_new = beta2 * v + (1.0 - beta2) * (diff * diff)
        if dtype_mode == 2 or state_dtype_mode == 2:
            v_new = tl.minimum(v_new, 65000.0)

        sigma = tl.sqrt(tl.maximum(v_new, 0.0)) * c1 + c2
        u = _triton_tanh_fast(clm_m / tl.maximum(sigma, 1e-12))
        m_mask = tl.where((u * g) > 0.0, 1.0, 0.0)

        _store_state_buffer(exp_avg_ptr, offsets, m_new, mask, seed_m, state_dtype_mode, use_sr)
        _store_state_buffer(exp_avg_sq_ptr, offsets, v_new, mask, seed_m, state_dtype_mode, False)

        delta_theta = (u * m_mask) * cautious_scale if cautious else u
        step_val = lr * delta_theta

        if bound_mode == 1:
            step_val = step_val * bound_scale
        elif bound_mode == 2:
            B_i = bound_ratio * tl.sqrt(p_init * p_init + 1e-4)
            step_val = B_i * _triton_tanh_fast(step_val / B_i)

        if dtype_mode == 2:
            b_safe = tl.maximum(tl.abs(p_init) * 0.01, 1.5e-4)
            step_val = b_safe * _triton_tanh_fast(step_val / b_safe)

        if weight_decay != 0.0:
            p = p * (1.0 - lr * weight_decay)
        p_updated = p - step_val

        if has_master:
            tl.store(master_p_ptr + offsets, p_updated, mask=mask)
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, dtype_mode, False)
        else:
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, dtype_mode, use_sr)

        if tl.program_id(axis=0) == 0:
            if cautious:
                tl.store(scratch_ptr + 3, cautious_scale)
            if bound_mode == 1:
                tl.store(scratch_ptr + 4, bound_scale)

    @triton.jit
    def _lumina_dual_single_pass_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
        scratch_ptr,
        n_elements, total_elements, beta1, beta2, c1, c2, lr, weight_decay,
        bound_mode: tl.constexpr, bound_ratio,
        seed, dtype_mode: tl.constexpr, state_dtype_mode: tl.constexpr, use_sr: tl.constexpr,
        has_master: tl.constexpr, centered_defect: tl.constexpr, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        seed_m = seed
        seed_p = (seed + 0x55555555) & 0x7FFFFFFF

        if has_master:
            p = tl.load(master_p_ptr + offsets, mask=mask, other=0.0)
        else:
            p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        p_init = p

        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        is_finite_g = (g == g) & (g < 1e38) & (g > -1e38)
        g = tl.where(is_finite_g, tl.maximum(tl.minimum(g, 1e18), -1e18), 0.0)
        m = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        m_new = beta1 * m + (1.0 - beta1) * g
        clm_m = beta1 * m_new + (1.0 - beta1) * g
        diff = tl.maximum(tl.minimum(g - m_new, 1e18), -1e18)
        if centered_defect:
            diff_mean = (tl.load(scratch_ptr + 5) / total_elements) * beta1
            diff = tl.maximum(tl.minimum(diff - diff_mean, 1e18), -1e18)
        v_new = beta2 * v + (1.0 - beta2) * (diff * diff)
        if dtype_mode == 2 or state_dtype_mode == 2:
            v_new = tl.minimum(v_new, 65000.0)

        sigma = tl.sqrt(tl.maximum(v_new, 0.0)) * c1 + c2
        u = _triton_tanh_fast(clm_m / tl.maximum(sigma, 1e-12))

        _store_state_buffer(exp_avg_ptr, offsets, m_new, mask, seed_m, state_dtype_mode, use_sr)
        _store_state_buffer(exp_avg_sq_ptr, offsets, v_new, mask, seed_m, state_dtype_mode, False)

        step_val = lr * u
        if bound_mode == 2:
            B_i = bound_ratio * tl.sqrt(p_init * p_init + 1e-4)
            step_val = B_i * _triton_tanh_fast(step_val / B_i)

        if dtype_mode == 2:
            b_safe = tl.maximum(tl.abs(p_init) * 0.01, 1.5e-4)
            step_val = b_safe * _triton_tanh_fast(step_val / b_safe)

        if weight_decay != 0.0:
            p = p * (1.0 - lr * weight_decay)
        p_updated = p - step_val

        if has_master:
            tl.store(master_p_ptr + offsets, p_updated, mask=mask)
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, dtype_mode, False)
        else:
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, dtype_mode, use_sr)
else:
    _lumina_dual_fused_single_pass_kernel: Any = None
    _lumina_dual_pass1_kernel: Any = None
    _lumina_dual_pass2_kernel: Any = None
    _lumina_dual_single_pass_kernel: Any = None
