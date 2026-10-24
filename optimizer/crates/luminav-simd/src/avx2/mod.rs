pub mod dual;
pub mod tdclm;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256, __m256i, _CMP_LT_OQ, _CMP_ORD_Q, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32,
    _mm_movehdup_ps, _mm_movehl_ps, _mm256_add_epi32, _mm256_add_ps, _mm256_and_ps,
    _mm256_and_si256, _mm256_andnot_ps, _mm256_andnot_si256, _mm256_blendv_ps, _mm256_castps_si256,
    _mm256_castps256_ps128, _mm256_castsi256_ps, _mm256_cmp_ps, _mm256_cmpgt_epi32, _mm256_div_ps,
    _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_max_epi32, _mm256_max_ps,
    _mm256_min_epi32, _mm256_min_ps, _mm256_movemask_ps, _mm256_mul_ps, _mm256_mullo_epi32,
    _mm256_or_si256, _mm256_set_epi32, _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_setzero_si256, _mm256_sllv_epi32, _mm256_sqrt_ps, _mm256_srai_epi32, _mm256_srli_epi32,
    _mm256_storeu_ps, _mm256_storeu_si256, _mm256_sub_epi32, _mm256_sub_ps, _mm256_xor_si256,
};

#[cfg(target_arch = "x86_64")]
use luminav_core::{DTypeMode, sanitize_gradient, stochastic_round_fp16};

#[cfg(target_arch = "x86_64")]
pub use dual::{
    dual_pass1_chunk_avx2, dual_pass2_chunk_avx2, dual_single_pass_chunk_avx2, dual_step_avx2,
};
#[cfg(target_arch = "x86_64")]
pub use tdclm::{
    tdclm_pass1_chunk_avx2, tdclm_pass2_chunk_avx2, tdclm_single_pass_chunk_avx2, tdclm_step_avx2,
};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn reduce_add_ps_avx2_pair(v_lo: __m256, v_hi: __m256) -> f32 {
    let sum256 = _mm256_add_ps(v_lo, v_hi);
    let lo128 = _mm256_castps256_ps128(sum256);
    let hi128 = _mm256_extractf128_ps::<1>(sum256);
    let sum128 = _mm_add_ps(lo128, hi128);

    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);

    let hi32 = _mm_movehdup_ps(sum64);
    let sum32 = _mm_add_ss(sum64, hi32);
    _mm_cvtss_f32(sum32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn abs_ps_avx2(v: __m256) -> __m256 {
    _mm256_andnot_ps(_mm256_set1_ps(-0.0_f32), v)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn clamp_diff_avx2(v: __m256) -> __m256 {
    let v_hi = _mm256_set1_ps(1.0e18_f32);
    let v_lo = _mm256_set1_ps(-1.0e18_f32);
    _mm256_min_ps(v_hi, _mm256_max_ps(v_lo, v))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn sanitize_gradient_avx2(v_g: __m256) -> __m256 {
    unsafe {
        let v_max_g = _mm256_set1_ps(1.0e38_f32);
        let m_ord = _mm256_cmp_ps::<_CMP_ORD_Q>(v_g, v_g);
        let m_lt = _mm256_cmp_ps::<_CMP_LT_OQ>(abs_ps_avx2(v_g), v_max_g);
        let v_valid = _mm256_and_ps(_mm256_and_ps(m_ord, m_lt), v_g);
        clamp_diff_avx2(v_valid)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn prng_avx2(v_seed: __m256i, v_offsets: __m256i) -> __m256i {
    let c0 = _mm256_set1_epi32(-1_640_531_527);
    let c1 = _mm256_set1_epi32(-2_048_173_461);
    let c2 = _mm256_set1_epi32(-1_028_477_387);

    let mut z = _mm256_add_epi32(_mm256_add_epi32(v_offsets, v_seed), c0);
    z = _mm256_mullo_epi32(_mm256_xor_si256(z, _mm256_srai_epi32::<16>(z)), c1);
    z = _mm256_mullo_epi32(_mm256_xor_si256(z, _mm256_srai_epi32::<13>(z)), c2);
    _mm256_xor_si256(z, _mm256_srai_epi32::<16>(z))
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn stochastic_round_bf16_avx2(p: __m256, v_seed: __m256i, v_offsets: __m256i) -> __m256 {
    let p_int = _mm256_castps_si256(p);
    let abs_int = _mm256_and_si256(p_int, _mm256_set1_epi32(0x7FFF_FFFF));
    let m_finite = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x7F80_0000), abs_int);

    let noise = _mm256_and_si256(prng_avx2(v_seed, v_offsets), _mm256_set1_epi32(0xFFFF));
    let added_abs = _mm256_and_si256(_mm256_add_epi32(abs_int, noise), _mm256_set1_epi32(!0xFFFF));
    let rounded_abs = _mm256_min_epi32(added_abs, _mm256_set1_epi32(0x7F7F_0000));
    let sign_bit = _mm256_and_si256(p_int, _mm256_set1_epi32(i32::MIN));
    let out_int = _mm256_or_si256(sign_bit, rounded_abs);
    let out_f32 = _mm256_castsi256_ps(out_int);

    _mm256_blendv_ps(p, out_f32, _mm256_castsi256_ps(m_finite))
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn round_to_nearest_bf16_avx2(p: __m256) -> __m256 {
    let p_int = _mm256_castps_si256(p);
    let abs_int = _mm256_and_si256(p_int, _mm256_set1_epi32(0x7FFF_FFFF));
    let m_finite = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x7F80_0000), abs_int);

    let lsb = _mm256_and_si256(_mm256_srli_epi32::<16>(abs_int), _mm256_set1_epi32(1));
    let bias = _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), lsb);
    let added_abs = _mm256_and_si256(_mm256_add_epi32(abs_int, bias), _mm256_set1_epi32(!0xFFFF));
    let rounded_abs = _mm256_min_epi32(added_abs, _mm256_set1_epi32(0x7F7F_0000));
    let sign_bit = _mm256_and_si256(p_int, _mm256_set1_epi32(i32::MIN));
    let out_f32 = _mm256_castsi256_ps(_mm256_or_si256(sign_bit, rounded_abs));

    _mm256_blendv_ps(p, out_f32, _mm256_castsi256_ps(m_finite))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn stochastic_round_fp16_avx2(
    v_val: __m256,
    v_seed: __m256i,
    v_offsets: __m256i,
) -> __m256 {
    unsafe {
        let p_int = _mm256_castps_si256(v_val);
        let abs_int = _mm256_and_si256(p_int, _mm256_set1_epi32(0x7FFF_FFFF));
        let exp_f32 = _mm256_and_si256(_mm256_srli_epi32::<23>(abs_int), _mm256_set1_epi32(0xFF));

        let diff_113 = _mm256_sub_epi32(_mm256_set1_epi32(113), exp_f32);
        let sub_shift = _mm256_min_epi32(
            _mm256_max_epi32(diff_113, _mm256_setzero_si256()),
            _mm256_set1_epi32(11),
        );
        let total_shift = _mm256_add_epi32(_mm256_set1_epi32(13), sub_shift);
        let one = _mm256_set1_epi32(1);
        let mask_bits = _mm256_sub_epi32(_mm256_sllv_epi32(one, total_shift), one);

        let noise = _mm256_and_si256(prng_avx2(v_seed, v_offsets), mask_bits);
        let added = _mm256_andnot_si256(mask_bits, _mm256_add_epi32(abs_int, noise));

        let m_not_underflow = _mm256_cmpgt_epi32(exp_f32, _mm256_set1_epi32(102));
        let rounded_abs = _mm256_and_si256(added, m_not_underflow);
        let sign_bit = _mm256_and_si256(p_int, _mm256_set1_epi32(i32::MIN));
        let out_int = _mm256_or_si256(sign_bit, rounded_abs);

        let exp_out = _mm256_srli_epi32::<23>(rounded_abs);
        let m_valid_in = _mm256_cmpgt_epi32(_mm256_set1_epi32(143), exp_f32);
        let m_valid_out = _mm256_cmpgt_epi32(_mm256_set1_epi32(143), exp_out);
        let valid_mask = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_and_si256(
            m_valid_in,
            m_valid_out,
        )));

        if valid_mask == 0xFF {
            _mm256_castsi256_ps(out_int)
        } else {
            let mut lanes = [0.0_f32; 8];
            let mut seeds = [0_i32; 8];
            let mut offs = [0_i32; 8];
            _mm256_storeu_ps(lanes.as_mut_ptr(), v_val);
            _mm256_storeu_si256(seeds.as_mut_ptr().cast(), v_seed);
            _mm256_storeu_si256(offs.as_mut_ptr().cast(), v_offsets);
            for i in 0..8 {
                lanes[i] = stochastic_round_fp16(lanes[i], seeds[i], offs[i]);
            }
            _mm256_loadu_ps(lanes.as_ptr())
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn round_to_nearest_fp16_avx2(v_val: __m256) -> __m256 {
    unsafe {
        let p_int = _mm256_castps_si256(v_val);
        let abs_int = _mm256_and_si256(p_int, _mm256_set1_epi32(0x7FFF_FFFF));
        let exp_f32 = _mm256_and_si256(_mm256_srli_epi32::<23>(abs_int), _mm256_set1_epi32(0xFF));

        let lsb = _mm256_and_si256(_mm256_srli_epi32::<13>(abs_int), _mm256_set1_epi32(1));
        let bias = _mm256_add_epi32(_mm256_set1_epi32(0x0FFF), lsb);
        let added = _mm256_and_si256(_mm256_add_epi32(abs_int, bias), _mm256_set1_epi32(!0x1FFF));

        let m_normal = _mm256_cmpgt_epi32(exp_f32, _mm256_set1_epi32(112));
        let m_underflow = _mm256_cmpgt_epi32(_mm256_set1_epi32(102), exp_f32);
        let rounded_abs = _mm256_and_si256(added, m_normal);
        let sign_bit = _mm256_and_si256(p_int, _mm256_set1_epi32(i32::MIN));
        let out_int = _mm256_or_si256(sign_bit, rounded_abs);

        let exp_out = _mm256_srli_epi32::<23>(rounded_abs);
        let m_fast_range = _mm256_or_si256(m_normal, m_underflow);
        let m_valid_in = _mm256_cmpgt_epi32(_mm256_set1_epi32(143), exp_f32);
        let m_valid_out = _mm256_cmpgt_epi32(_mm256_set1_epi32(143), exp_out);
        let valid_mask = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_and_si256(
            m_fast_range,
            _mm256_and_si256(m_valid_in, m_valid_out),
        )));

        if valid_mask == 0xFF {
            _mm256_castsi256_ps(out_int)
        } else {
            let mut lanes = [0.0_f32; 8];
            _mm256_storeu_ps(lanes.as_mut_ptr(), v_val);
            for v in &mut lanes {
                *v = luminav_core::round_to_nearest_fp16(*v);
            }
            _mm256_loadu_ps(lanes.as_ptr())
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn quantize_avx2(
    v_val: __m256,
    v_seed: __m256i,
    v_offsets: __m256i,
    dtype_mode: DTypeMode,
    use_sr: bool,
) -> __m256 {
    unsafe {
        match dtype_mode {
            DTypeMode::Float32 => v_val,
            DTypeMode::BFloat16 => {
                if use_sr {
                    stochastic_round_bf16_avx2(v_val, v_seed, v_offsets)
                } else {
                    round_to_nearest_bf16_avx2(v_val)
                }
            }
            DTypeMode::Float16 => {
                if use_sr {
                    stochastic_round_fp16_avx2(v_val, v_seed, v_offsets)
                } else {
                    round_to_nearest_fp16_avx2(v_val)
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn tanh_pade_avx2(v_x_in: __m256) -> __m256 {
    let v_clamp_hi = _mm256_set1_ps(7.998_811_7_f32);
    let v_clamp_lo = _mm256_set1_ps(-7.998_811_7_f32);

    let v_a13 = _mm256_set1_ps(-2.760_768_4e-16_f32);
    let v_a11 = _mm256_set1_ps(2.000_188e-13_f32);
    let v_a9 = _mm256_set1_ps(-8.604_672e-11_f32);
    let v_a7 = _mm256_set1_ps(5.122_297e-8_f32);
    let v_a5 = _mm256_set1_ps(1.485_722_4e-5_f32);
    let v_a3 = _mm256_set1_ps(6.372_619_3e-4_f32);
    let v_a1 = _mm256_set1_ps(4.893_524_6e-3_f32);

    let v_b6 = _mm256_set1_ps(1.198_258_4e-6_f32);
    let v_b4 = _mm256_set1_ps(1.185_347_1e-4_f32);
    let v_b2 = _mm256_set1_ps(2.268_434_6e-3_f32);
    let v_b0 = _mm256_set1_ps(4.893_524_6e-3_f32);

    let v_x = _mm256_min_ps(v_clamp_hi, _mm256_max_ps(v_clamp_lo, v_x_in));
    let v_x2 = _mm256_mul_ps(v_x, v_x);

    let mut v_p = _mm256_fmadd_ps(v_a13, v_x2, v_a11);
    let mut v_q = _mm256_fmadd_ps(v_b6, v_x2, v_b4);
    v_p = _mm256_fmadd_ps(v_p, v_x2, v_a9);
    v_q = _mm256_fmadd_ps(v_q, v_x2, v_b2);
    v_p = _mm256_fmadd_ps(v_p, v_x2, v_a7);
    v_q = _mm256_fmadd_ps(v_q, v_x2, v_b0);
    v_p = _mm256_fmadd_ps(v_p, v_x2, v_a5);
    v_p = _mm256_fmadd_ps(v_p, v_x2, v_a3);
    v_p = _mm256_fmadd_ps(v_p, v_x2, v_a1);

    _mm256_div_ps(_mm256_mul_ps(v_x, v_p), v_q)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn apply_coordinate_bound_avx2(
    v_step: __m256,
    v_p_init: __m256,
    v_bound_ratio: __m256,
) -> __m256 {
    unsafe {
        let v_eps = _mm256_set1_ps(1.0e-4_f32);
        let v_norm = _mm256_sqrt_ps(_mm256_fmadd_ps(v_p_init, v_p_init, v_eps));
        let v_bi = _mm256_mul_ps(v_bound_ratio, v_norm);
        _mm256_mul_ps(v_bi, tanh_pade_avx2(_mm256_div_ps(v_step, v_bi)))
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn apply_fp16_safety_bound_avx2(v_step: __m256, v_p_init: __m256) -> __m256 {
    unsafe {
        let v_scale = _mm256_set1_ps(0.01_f32);
        let v_min = _mm256_set1_ps(1.5e-4_f32);
        let v_bsafe = _mm256_max_ps(_mm256_mul_ps(abs_ps_avx2(v_p_init), v_scale), v_min);
        _mm256_mul_ps(v_bsafe, tanh_pade_avx2(_mm256_div_ps(v_step, v_bsafe)))
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn offset_vector_avx2(base_offset: i32) -> __m256i {
    _mm256_add_epi32(
        _mm256_set1_epi32(base_offset),
        _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn compute_diff_sum_chunk_avx2(grad: &[f32], exp_avg: &[f32]) -> f32 {
    unsafe {
        let n = grad.len();
        let simd_len = n & !15;
        let mut v_acc_lo = _mm256_setzero_ps();
        let mut v_acc_hi = _mm256_setzero_ps();
        let mut i = 0;
        while i < simd_len {
            let v_g_lo = sanitize_gradient_avx2(_mm256_loadu_ps(grad.as_ptr().add(i)));
            let v_m_lo = _mm256_loadu_ps(exp_avg.as_ptr().add(i));
            v_acc_lo = _mm256_add_ps(v_acc_lo, clamp_diff_avx2(_mm256_sub_ps(v_g_lo, v_m_lo)));

            let v_g_hi = sanitize_gradient_avx2(_mm256_loadu_ps(grad.as_ptr().add(i + 8)));
            let v_m_hi = _mm256_loadu_ps(exp_avg.as_ptr().add(i + 8));
            v_acc_hi = _mm256_add_ps(v_acc_hi, clamp_diff_avx2(_mm256_sub_ps(v_g_hi, v_m_hi)));

            i += 16;
        }
        let mut sum = reduce_add_ps_avx2_pair(v_acc_lo, v_acc_hi);
        for idx in simd_len..n {
            let g = sanitize_gradient(grad[idx]);
            sum += (g - exp_avg[idx]).clamp(-1.0e18_f32, 1.0e18_f32);
        }
        sum
    }
}
