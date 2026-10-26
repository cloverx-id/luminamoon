from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import triton.language as tl
    triton: Any = None
    HAS_TRITON = True
else:
    triton = None
    tl = None
    try:
        import triton
        import triton.language as tl
        HAS_TRITON = True
    except ImportError:
        HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _prng(seed, idx):
        z = (idx + seed - 1640531527).to(tl.int32)
        z = ((z ^ (z >> 16)) * -2048173461).to(tl.int32)
        z = ((z ^ (z >> 13)) * -1028477387).to(tl.int32)
        z = (z ^ (z >> 16)).to(tl.int32)
        return z

    @triton.jit
    def _sr_bf16_bits(val_f32, seed, offsets):
        p_int = val_f32.to(tl.int32, bitcast=True)
        abs_int = p_int & 0x7FFFFFFF
        noise = (_prng(seed, offsets) & 0xFFFF).to(tl.int32)
        rounded_abs = tl.minimum((abs_int + noise) & -65536, 0x7F7F0000)
        sign_bit = p_int & -2147483648
        out_int = sign_bit | rounded_abs
        return out_int.to(tl.float32, bitcast=True).to(tl.bfloat16)

    @triton.jit
    def _sr_fp16_bits(val_f32, seed, offsets):
        p_int = val_f32.to(tl.int32, bitcast=True)
        abs_int = p_int & 0x7FFFFFFF
        exp_f32 = (abs_int >> 23) & 0xFF
        sub_shift = tl.minimum(tl.maximum(113 - exp_f32, 0), 11)
        total_shift = 13 + sub_shift
        mask_bits = (1 << total_shift) - 1
        noise = _prng(seed, offsets) & mask_bits
        rounded_abs = tl.where(exp_f32 < 103, 0, (abs_int + noise) & (~mask_bits))
        rounded_abs = tl.minimum(rounded_abs, 0x477FE000)
        sign_bit = p_int & -2147483648
        out_int = sign_bit | rounded_abs
        return out_int.to(tl.float32, bitcast=True).to(tl.float16)

    @triton.jit
    def _store_param(p_ptr, offsets, p_val, mask, seed, dtype_mode: tl.constexpr, use_sr: tl.constexpr):
        if use_sr:
            if dtype_mode == 1:
                p_out = _sr_bf16_bits(p_val, seed, offsets)
                tl.store(p_ptr + offsets, p_out, mask=mask)
            elif dtype_mode == 2:
                p_out = _sr_fp16_bits(p_val, seed, offsets)
                tl.store(p_ptr + offsets, p_out, mask=mask)
            else:
                tl.store(p_ptr + offsets, p_val, mask=mask)
        else:
            if dtype_mode == 1:
                p_clamped = tl.maximum(tl.minimum(p_val, 3.3895313892515355e38), -3.3895313892515355e38)
                tl.store(p_ptr + offsets, p_clamped.to(tl.bfloat16), mask=mask)
            elif dtype_mode == 2:
                p_clamped = tl.maximum(tl.minimum(p_val, 65504.0), -65504.0)
                tl.store(p_ptr + offsets, p_clamped.to(tl.float16), mask=mask)
            else:
                tl.store(p_ptr + offsets, p_val, mask=mask)

    @triton.jit
    def _store_state_buffer(
        ptr, offsets, val, mask, seed,
        state_dtype_mode: tl.constexpr, use_sr: tl.constexpr
    ):
        if use_sr:
            if state_dtype_mode == 1:
                p_out = _sr_bf16_bits(val, seed, offsets)
                tl.store(ptr + offsets, p_out, mask=mask)
            elif state_dtype_mode == 2:
                p_out = _sr_fp16_bits(val, seed, offsets)
                tl.store(ptr + offsets, p_out, mask=mask)
            else:
                tl.store(ptr + offsets, val, mask=mask)
        else:
            if state_dtype_mode == 1:
                val_clamped = tl.maximum(tl.minimum(val, 3.3895313892515355e38), -3.3895313892515355e38)
                tl.store(ptr + offsets, val_clamped.to(tl.bfloat16), mask=mask)
            elif state_dtype_mode == 2:
                val_clamped = tl.maximum(tl.minimum(val, 65504.0), -65504.0)
                tl.store(ptr + offsets, val_clamped.to(tl.float16), mask=mask)
            else:
                tl.store(ptr + offsets, val, mask=mask)

    @triton.jit
    def _triton_tanh_fast(x):
        x_c = tl.maximum(tl.minimum(x, 7.9988117), -7.9988117)
        x2 = x_c * x_c
        p = -2.7607684e-16 * x2 + 2.000188e-13
        q = 1.1982584e-6 * x2 + 1.1853471e-4
        p = p * x2 + -8.604672e-11
        q = q * x2 + 2.2684346e-3
        p = p * x2 + 5.122297e-8
        q = q * x2 + 4.8935246e-3
        p = p * x2 + 1.4857224e-5
        p = p * x2 + 6.3726193e-4
        p = p * x2 + 4.8935246e-3
        return (x_c * p) / q

    @triton.jit
    def _lumina_diff_sum_kernel(
        grad_ptr, exp_avg_ptr, scratch_ptr,
        n_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        is_finite_g = (g == g) & (g < 1e38) & (g > -1e38)
        g = tl.where(is_finite_g, tl.maximum(tl.minimum(g, 1e18), -1e18), 0.0)
        m = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        diff_gm = tl.maximum(tl.minimum(g - m, 1e18), -1e18)
        block_diff_sum = tl.sum(tl.where(mask, diff_gm, 0.0), axis=0)
        tl.atomic_add(scratch_ptr + 5, block_diff_sum)

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
    def _lumina_tdclm_single_pass_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr,
        scratch_ptr,
        n_elements, total_elements, beta1, lr, weight_decay, eps, lam,
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

        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq
        effective_eps = tl.maximum(eps, 1e-4) if dtype_mode == 2 else eps

        m_new = beta1 * m + (1.0 - beta1) * g
        clm_m = beta1_sq * m + one_minus_beta1_sq * g

        diff = tl.maximum(tl.minimum(g - m, 1e18), -1e18)
        if centered_defect:
            diff_mean = tl.load(scratch_ptr + 5) / total_elements
            diff = tl.maximum(tl.minimum(diff - diff_mean, 1e18), -1e18)

        clm_env = (
            beta1_sq * tl.abs(m)
            + one_minus_beta1_sq * tl.abs(g)
            + lam * tl.abs(diff)
            + effective_eps
        )

        u = clm_m / clm_env
        m_mask = tl.where((u * g) > 0.0, 1.0, 0.0)

        _store_state_buffer(exp_avg_ptr, offsets, m_new, mask, seed_m, state_dtype_mode, use_sr)

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
    def _lumina_tdclm_pass1_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr,
        scratch_ptr,
        n_elements, total_elements, beta1, eps, lam,
        bound_mode: tl.constexpr, dtype_mode: tl.constexpr,
        cautious: tl.constexpr, has_master: tl.constexpr, centered_defect: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
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

        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq
        effective_eps = tl.maximum(eps, 1e-4) if dtype_mode == 2 else eps

        diff = tl.maximum(tl.minimum(g - m, 1e18), -1e18)
        if centered_defect:
            diff_mean = tl.load(scratch_ptr + 5) / total_elements
            diff = tl.maximum(tl.minimum(diff - diff_mean, 1e18), -1e18)

        clm_m = beta1_sq * m + one_minus_beta1_sq * g
        clm_env = beta1_sq * tl.abs(m) + one_minus_beta1_sq * tl.abs(g) + lam * tl.abs(diff) + effective_eps
        u = clm_m / clm_env
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
    def _lumina_tdclm_pass2_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr,
        scratch_ptr,
        n_elements, total_elements, beta1, lr, weight_decay, eps, lam, clamp_min,
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

        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq
        effective_eps = tl.maximum(eps, 1e-4) if dtype_mode == 2 else eps

        diff = tl.maximum(tl.minimum(g - m, 1e18), -1e18)
        if centered_defect:
            diff_mean = tl.load(scratch_ptr + 5) / total_elements
            diff = tl.maximum(tl.minimum(diff - diff_mean, 1e18), -1e18)

        m_new = beta1 * m + (1.0 - beta1) * g
        clm_m = beta1_sq * m + one_minus_beta1_sq * g
        clm_env = beta1_sq * tl.abs(m) + one_minus_beta1_sq * tl.abs(g) + lam * tl.abs(diff) + effective_eps
        u = clm_m / clm_env
        m_mask = tl.where((u * g) > 0.0, 1.0, 0.0)

        _store_state_buffer(exp_avg_ptr, offsets, m_new, mask, seed_m, state_dtype_mode, use_sr)

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
    def _lumina_post_step_ema_kernel(
        scratch_ptr, total_elements, lr, clamp_min, ema_decay,
        bound_mode: tl.constexpr, bound_ratio, cautious: tl.constexpr
    ):
        if tl.program_id(axis=0) == 0:
            mask_sum_ptr = scratch_ptr + 0
            u_sq_sum_ptr = scratch_ptr + 1
            p_sq_sum_ptr = scratch_ptr + 2
            cautious_scale_ptr = scratch_ptr + 3
            bound_scale_ptr = scratch_ptr + 4

            if cautious:
                mask_sum = tl.load(mask_sum_ptr)
                m_bar = tl.minimum(tl.maximum(mask_sum / total_elements, clamp_min), 1.0)
                target_cautious_scale = 1.0 / m_bar
                old_cautious = tl.load(cautious_scale_ptr)
                new_cautious = ema_decay * old_cautious + (1.0 - ema_decay) * target_cautious_scale
                tl.store(cautious_scale_ptr, new_cautious)

            if bound_mode == 1:
                u_sq_sum = tl.load(u_sq_sum_ptr)
                p_sq_sum = tl.load(p_sq_sum_ptr)
                c_scale = tl.load(cautious_scale_ptr) if cautious else 1.0

                norm_p = tl.sqrt(tl.maximum(p_sq_sum, 0.0))
                norm_delta = tl.sqrt(tl.maximum(u_sq_sum, 0.0)) * c_scale
                R = bound_ratio * tl.maximum(norm_p, 1.0)
                r = (lr * norm_delta) / R
                target_scale = tl.where(r < 1e-4, 1.0, _triton_tanh_fast(r) / tl.maximum(r, 1e-8))
                old_bound = tl.load(bound_scale_ptr)
                new_bound = ema_decay * old_bound + (1.0 - ema_decay) * target_scale
                tl.store(bound_scale_ptr, new_bound)

            tl.store(mask_sum_ptr, 0.0)
            tl.store(u_sq_sum_ptr, 0.0)
            tl.store(p_sq_sum_ptr, 0.0)
            tl.store(scratch_ptr + 5, 0.0)

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
    _lumina_diff_sum_kernel = None
    _lumina_dual_fused_single_pass_kernel = None
    _lumina_dual_pass1_kernel = None
    _lumina_dual_pass2_kernel = None
    _lumina_dual_single_pass_kernel = None
    _lumina_post_step_ema_kernel = None
    _lumina_tdclm_pass1_kernel = None
    _lumina_tdclm_pass2_kernel = None
    _lumina_tdclm_single_pass_kernel = None
