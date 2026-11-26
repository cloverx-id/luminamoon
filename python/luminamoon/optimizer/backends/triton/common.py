from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import triton.language as _tl

    tl = _tl
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
    def _derive_seeds_and_weyl_single(seed, step_i32):
        # Promote scalar args to 0-D tl.int32 tensors so Triton's equal_to_1
        # specialization (e.g., step_i32 == 1 on Step 1) never produces a raw Python int.
        seed_t = seed + tl.zeros([], dtype=tl.int32)
        step_t = step_i32 + tl.zeros([], dtype=tl.int32)
        seed_m = seed_t
        seed_p = (seed_t + 0x55555555) & 0x7FFFFFFF
        weyl_m = (step_t * 1779033703).to(tl.int32)
        weyl_p = (step_t * -1640531527).to(tl.int32)
        return seed_m, seed_p, weyl_m, weyl_p

    @triton.jit
    def _derive_seeds_and_weyl_dual(seed, step_i32):
        seed_t = seed + tl.zeros([], dtype=tl.int32)
        step_t = step_i32 + tl.zeros([], dtype=tl.int32)
        seed_m = seed_t
        seed_p = (seed_t + 0x55555555) & 0x7FFFFFFF
        seed_v = (seed_t + 0x2A6D365B) & 0x7FFFFFFF
        weyl_m = (step_t * 1779033703).to(tl.int32)
        weyl_v = (step_t * -1150833019).to(tl.int32)
        weyl_p = (step_t * -1640531527).to(tl.int32)
        return seed_m, seed_p, seed_v, weyl_m, weyl_v, weyl_p

    @triton.jit
    def _prng(seed, idx):
        z = (idx + seed - 1640531527).to(tl.int32)
        z = ((z ^ ((z >> 16) & 0xFFFF)) * -2048173461).to(tl.int32)
        z = ((z ^ ((z >> 13) & 0x7FFFF)) * -1028477387).to(tl.int32)
        z = (z ^ ((z >> 16) & 0xFFFF)).to(tl.int32)
        return z

    @triton.jit
    def _prng_weyl(seed, idx, step_offset):
        return (_prng(seed, idx) + step_offset).to(tl.int32)

    @triton.jit
    def _sr_bf16_bits(val_f32, seed, offsets, step_offset):
        p_int = val_f32.to(tl.int32, bitcast=True)
        abs_int = p_int & 0x7FFFFFFF
        is_nan = abs_int > 0x7F800000
        noise = ((_prng_weyl(seed, offsets, step_offset) >> 16) & 0xFFFF).to(tl.int32)
        rounded_abs = tl.minimum((abs_int + noise) & -65536, 0x7F7F0000)
        sign_bit = p_int & -2147483648
        out_int = tl.where(is_nan, (p_int | 0x00400000) & -65536, sign_bit | rounded_abs)
        return out_int.to(tl.float32, bitcast=True).to(tl.bfloat16)

    @triton.jit
    def _sr_fp16_bits(val_f32, seed, offsets, step_offset):
        p_int = val_f32.to(tl.int32, bitcast=True)
        abs_int = p_int & 0x7FFFFFFF
        is_nan = abs_int > 0x7F800000
        exp_f32 = (abs_int >> 23) & 0xFF
        sub_shift = tl.minimum(tl.maximum(113 - exp_f32, 0), 11)
        total_shift = 13 + sub_shift
        mask_bits = (1 << total_shift) - 1
        noise = (_prng_weyl(seed, offsets, step_offset) >> (32 - total_shift)) & mask_bits
        rounded_abs = tl.where(exp_f32 < 103, 0, (abs_int + noise) & (~mask_bits))
        rounded_abs = tl.minimum(rounded_abs, 0x477FE000)
        sign_bit = p_int & -2147483648
        out_int = tl.where(is_nan, p_int | 0x00400000, sign_bit | rounded_abs)
        return out_int.to(tl.float32, bitcast=True).to(tl.float16)

    @triton.jit
    def _store_param(
        p_ptr, offsets, p_val, mask, seed, step_offset, dtype_mode: tl.constexpr, use_sr: tl.constexpr
    ):
        if use_sr:
            if dtype_mode == 1:
                tl.store(p_ptr + offsets, _sr_bf16_bits(p_val, seed, offsets, step_offset), mask=mask)
            elif dtype_mode == 2:
                tl.store(p_ptr + offsets, _sr_fp16_bits(p_val, seed, offsets, step_offset), mask=mask)
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
    def _sr_bf16_in_fp16_container(val_f32, seed, offsets, step_offset):
        p_int = val_f32.to(tl.int32, bitcast=True)
        abs_int = p_int & 0x7FFFFFFF
        is_nan = abs_int > 0x7F800000
        noise = ((_prng_weyl(seed, offsets, step_offset) >> 16) & 0xFFFF).to(tl.int32)
        rounded_abs = tl.minimum((abs_int + noise) & -65536, 0x7F7F0000)
        sign_bit = p_int & -2147483648
        out_int = tl.where(is_nan, (p_int | 0x00400000) & -65536, sign_bit | rounded_abs)
        out_i16 = (out_int >> 16).to(tl.int16)
        return out_i16.to(tl.float16, bitcast=True)

    @triton.jit
    def _rne_bf16_in_fp16_container(val_f32):
        raw_int = val_f32.to(tl.int32, bitcast=True)
        is_nan = (raw_int & 0x7FFFFFFF) > 0x7F800000
        val_clamped = tl.maximum(tl.minimum(val_f32, 3.3895313892515355e38), -3.3895313892515355e38)
        p_int = val_clamped.to(tl.int32, bitcast=True)
        abs_int = p_int & 0x7FFFFFFF
        sign_bit = p_int & -2147483648
        lsb = (abs_int >> 16) & 1
        rounded_abs = tl.minimum((abs_int + 0x7FFF + lsb) & -65536, 0x7F7F0000)
        out_int = tl.where(is_nan, (raw_int | 0x00400000) & -65536, sign_bit | rounded_abs)
        out_i16 = (out_int >> 16).to(tl.int16)
        return out_i16.to(tl.float16, bitcast=True)

    @triton.jit
    def _load_state_buffer(ptr, offsets, mask, state_dtype_mode: tl.constexpr):
        if state_dtype_mode == 2:
            raw_f16 = tl.load(ptr + offsets, mask=mask, other=0.0)
            bits_i32 = raw_f16.to(tl.int16, bitcast=True).to(tl.int32) << 16
            return bits_i32.to(tl.float32, bitcast=True)
        return tl.load(ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    @triton.jit
    def _store_state_buffer(
        ptr, offsets, val, mask, seed, step_offset, state_dtype_mode: tl.constexpr, use_sr: tl.constexpr
    ):
        if use_sr:
            if state_dtype_mode == 1:
                tl.store(ptr + offsets, _sr_bf16_bits(val, seed, offsets, step_offset), mask=mask)
            elif state_dtype_mode == 2:
                tl.store(ptr + offsets, _sr_bf16_in_fp16_container(val, seed, offsets, step_offset), mask=mask)
            else:
                tl.store(ptr + offsets, val, mask=mask)
        else:
            if state_dtype_mode == 1:
                val_clamped = tl.maximum(tl.minimum(val, 3.3895313892515355e38), -3.3895313892515355e38)
                tl.store(ptr + offsets, val_clamped.to(tl.bfloat16), mask=mask)
            elif state_dtype_mode == 2:
                tl.store(ptr + offsets, _rne_bf16_in_fp16_container(val), mask=mask)
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
    def _triton_rational_bound(step_val, bound):
        b = tl.maximum(bound, 0.0)
        abs_s = tl.abs(step_val)
        m = tl.minimum(b, abs_s)
        big_m = tl.maximum(b, abs_s)
        pos = big_m > 0.0
        safe_m = tl.where(pos, big_m, 1.0)
        t = tl.where(pos, m / safe_m, 0.0)
        one_plus_t = 1.0 + t
        mag = m * (one_plus_t / (one_plus_t + t * t))
        sign_bit = step_val.to(tl.int32, bitcast=True) & -2147483648
        mag_bits = mag.to(tl.int32, bitcast=True) & 0x7FFFFFFF
        return (mag_bits | sign_bit).to(tl.float32, bitcast=True)

    @triton.jit
    def _triton_temper_defect(abs_d, v_prev, m_prev, inv_sqrt_bc2_prev, variance_temper, eps):
        sigma_prev = tl.sqrt(tl.maximum(v_prev, 0.0)) * inv_sqrt_bc2_prev
        scale_ref = tl.maximum(sigma_prev, tl.abs(m_prev))
        return _triton_rational_bound(abs_d, variance_temper * scale_ref + eps)

    @triton.jit
    def _triton_apply_rpm(m, g, beta1, eps, BLOCK_SIZE: tl.constexpr):
        m0, m1 = tl.split(tl.reshape(m, [BLOCK_SIZE // 2, 2]))
        g0, g1 = tl.split(tl.reshape(g, [BLOCK_SIZE // 2, 2]))
        c = m0 * g1 - m1 * g0
        d = m0 * g0 + m1 * g1
        s = (1.0 - beta1) * c / (2.0 * tl.abs(d) + tl.abs(c) + eps)
        s2 = s * s
        inv = 1.0 / (1.0 + s2)
        c_rot = (1.0 - s2) * inv
        s_rot = (2.0 * s) * inv
        return tl.reshape(tl.join(m0 * c_rot - m1 * s_rot, m0 * s_rot + m1 * c_rot), [BLOCK_SIZE])

    @triton.jit
    def _triton_apply_wd(p_init, step_val, lr, weight_decay, cautious_wd: tl.constexpr):
        if weight_decay != 0.0:
            p_dec = p_init * (1.0 - lr * weight_decay)
            if cautious_wd:
                return tl.where((step_val * p_init) >= 0.0, p_dec, p_init)
            return p_dec
        return p_init

    @triton.jit
    def _triton_apply_step_bounds(
        step_val,
        p_init,
        bound_ratio,
        bound_floor,
        bound_scale,
        bound_mode: tl.constexpr,
        dtype_mode: tl.constexpr,
        is_triangle: tl.constexpr,
    ):
        if bound_mode == 1:
            step_val = step_val * bound_scale
        elif bound_mode == 2:
            if is_triangle:
                b_i = bound_ratio * (tl.abs(p_init) + bound_floor)
            else:
                b_i = bound_ratio * tl.sqrt(p_init * p_init + bound_floor * bound_floor)
            if dtype_mode == 2:
                b_safe = tl.maximum(tl.abs(p_init) * 0.01, 1.5e-4)
                b_i = tl.minimum(b_i, b_safe)
            if is_triangle:
                step_val = _triton_rational_bound(step_val, b_i)
            else:
                step_val = b_i * _triton_tanh_fast(step_val / tl.maximum(b_i, 1e-12))

        if bound_mode != 2 and dtype_mode == 2:
            b_safe = tl.maximum(tl.abs(p_init) * 0.01, 1.5e-4)
            if is_triangle:
                step_val = _triton_rational_bound(step_val, b_safe)
            else:
                step_val = b_safe * _triton_tanh_fast(step_val / b_safe)
        return step_val

    @triton.jit
    def _lumina_post_step_ema_kernel(
        scratch_ptr,
        total_elements,
        lr,
        clamp_min,
        ema_decay,
        bound_mode: tl.constexpr,
        bound_ratio,
        cautious: tl.constexpr,
        is_triangle: tl.constexpr,
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
                target_cautious = 1.0 / m_bar
                old_cautious = tl.load(cautious_scale_ptr)
                tl.store(cautious_scale_ptr, ema_decay * old_cautious + (1.0 - ema_decay) * target_cautious)

            if bound_mode == 1:
                u_sq_sum = tl.load(u_sq_sum_ptr)
                p_sq_sum = tl.load(p_sq_sum_ptr)
                c_scale = tl.load(cautious_scale_ptr) if cautious else 1.0
                norm_p = tl.sqrt(tl.maximum(p_sq_sum, 0.0))
                norm_delta = tl.sqrt(tl.maximum(u_sq_sum, 0.0)) * c_scale
                radius = bound_ratio * tl.maximum(norm_p, 1.0)
                r = (lr * norm_delta) / radius
                if is_triangle:
                    target_scale = (1.0 + r) / (1.0 + r + r * r)
                else:
                    target_scale = tl.where(r < 1e-4, 1.0, _triton_tanh_fast(r) / tl.maximum(r, 1e-8))
                old_bound = tl.load(bound_scale_ptr)
                gamma_eff = tl.where(target_scale < old_bound, tl.minimum(ema_decay, 0.10), ema_decay)
                tl.store(bound_scale_ptr, gamma_eff * old_bound + (1.0 - gamma_eff) * target_scale)

            tl.store(mask_sum_ptr, 0.0)
            tl.store(u_sq_sum_ptr, 0.0)
            tl.store(p_sq_sum_ptr, 0.0)

else:
    _derive_seeds_and_weyl_single: Any = None
    _derive_seeds_and_weyl_dual: Any = None
    _load_state_buffer: Any = None
    _store_param: Any = None
    _store_state_buffer: Any = None
    _triton_tanh_fast: Any = None
    _triton_rational_bound: Any = None
    _triton_temper_defect: Any = None
    _triton_apply_rpm: Any = None
    _triton_apply_wd: Any = None
    _triton_apply_step_bounds: Any = None
    _lumina_post_step_ema_kernel: Any = None
