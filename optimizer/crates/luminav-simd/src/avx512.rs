#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m512, __m512i, __mmask16, _CMP_GT_OQ, _CMP_LT_OQ, _CMP_ORD_Q, _mm_add_ps, _mm_add_ss,
    _mm_cvtss_f32, _mm_movehdup_ps, _mm_movehl_ps, _mm256_add_ps, _mm256_castps256_ps128,
    _mm256_extractf128_ps, _mm512_abs_ps, _mm512_add_epi32, _mm512_add_ps, _mm512_and_si512,
    _mm512_castps_si512, _mm512_castps512_ps256, _mm512_castsi512_ps, _mm512_cmp_epi32_mask,
    _mm512_cmp_ps_mask, _mm512_div_ps, _mm512_extractf32x8_ps, _mm512_fmadd_ps, _mm512_fnmadd_ps,
    _mm512_loadu_ps, _mm512_mask_mov_epi32, _mm512_maskz_mov_epi32, _mm512_maskz_mov_ps,
    _mm512_max_epi32, _mm512_max_ps, _mm512_min_epi32, _mm512_min_ps, _mm512_mul_ps,
    _mm512_mullo_epi32, _mm512_or_si512, _mm512_set_epi32, _mm512_set1_epi32, _mm512_set1_ps,
    _mm512_setzero_ps, _mm512_setzero_si512, _mm512_sllv_epi32, _mm512_sqrt_ps, _mm512_srai_epi32,
    _mm512_srli_epi32, _mm512_storeu_ps, _mm512_storeu_si512, _mm512_sub_epi32, _mm512_sub_ps,
    _mm512_xor_si512,
};

#[cfg(target_arch = "x86_64")]
use luminav_core::{
    BiasCorrectionCoeffs, BoundType, BufferMode, DTypeMode, LuminaError, LuminaVConfig,
    TensorViewMut, apply_coordinate_bound, apply_fp16_safety_bound, compute_bias_correction,
    compute_cautious_scale, compute_radial_bound_scale, derive_kernel_seeds, quantize_value,
    sanitize_gradient, stochastic_round_fp16, tanh_pade_f32, update_post_step_ema,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct ChunkMetrics {
    pub mask_count: u64,
    pub u_sq_sum: f32,
    pub p_sq_sum: f32,
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn _mm512_srad_epi32<const IMM8: u32>(a: __m512i) -> __m512i {
    _mm512_srai_epi32::<IMM8>(a)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[inline]
pub unsafe fn reduce_add_ps_avx512(v: __m512) -> f32 {
    let lo256 = _mm512_castps512_ps256(v);
    let hi256 = _mm512_extractf32x8_ps::<1>(v);
    let sum256 = _mm256_add_ps(lo256, hi256);

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
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[inline]
pub unsafe fn clamp_diff_avx512(v: __m512) -> __m512 {
    let v_clamp_hi = _mm512_set1_ps(1.0e18_f32);
    let v_clamp_lo = _mm512_set1_ps(-1.0e18_f32);
    _mm512_min_ps(v_clamp_hi, _mm512_max_ps(v_clamp_lo, v))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[inline]
pub unsafe fn sanitize_gradient_avx512(v_g: __m512) -> __m512 {
    let v_max_g = _mm512_set1_ps(1.0e38_f32);
    let k_ord: __mmask16 = _mm512_cmp_ps_mask::<_CMP_ORD_Q>(v_g, v_g);
    let k_lt: __mmask16 = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(_mm512_abs_ps(v_g), v_max_g);
    _mm512_maskz_mov_ps(k_ord & k_lt, clamp_diff_avx512(v_g))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[inline]
pub unsafe fn prng_avx512(v_seed: __m512i, v_offsets: __m512i) -> __m512i {
    let c0 = _mm512_set1_epi32(-1_640_531_527);
    let c1 = _mm512_set1_epi32(-2_048_173_461);
    let c2 = _mm512_set1_epi32(-1_028_477_387);

    let mut z = _mm512_add_epi32(_mm512_add_epi32(v_offsets, v_seed), c0);
    z = _mm512_mullo_epi32(_mm512_xor_si512(z, _mm512_srad_epi32::<16>(z)), c1);
    z = _mm512_mullo_epi32(_mm512_xor_si512(z, _mm512_srad_epi32::<13>(z)), c2);
    _mm512_xor_si512(z, _mm512_srad_epi32::<16>(z))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[inline]
pub unsafe fn stochastic_round_bf16_avx512(
    v_val: __m512,
    v_seed: __m512i,
    v_offsets: __m512i,
) -> __m512 {
    let p_int = _mm512_castps_si512(v_val);
    let abs_int = _mm512_and_si512(p_int, _mm512_set1_epi32(0x7FFF_FFFF));
    let k_finite: __mmask16 = _mm512_cmp_epi32_mask::<1>(abs_int, _mm512_set1_epi32(0x7F80_0000));

    let noise = _mm512_and_si512(prng_avx512(v_seed, v_offsets), _mm512_set1_epi32(0xFFFF));
    let rounded_abs =
        _mm512_and_si512(_mm512_add_epi32(abs_int, noise), _mm512_set1_epi32(!0xFFFF));
    let sign_bit = _mm512_and_si512(p_int, _mm512_set1_epi32(i32::MIN));
    let out_int = _mm512_or_si512(sign_bit, rounded_abs);
    let res_int = _mm512_mask_mov_epi32(p_int, k_finite, out_int);
    _mm512_castsi512_ps(res_int)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[inline]
pub unsafe fn round_to_nearest_bf16_avx512(v_val: __m512) -> __m512 {
    let p_int = _mm512_castps_si512(v_val);
    let abs_int = _mm512_and_si512(p_int, _mm512_set1_epi32(0x7FFF_FFFF));
    let k_finite: __mmask16 = _mm512_cmp_epi32_mask::<1>(abs_int, _mm512_set1_epi32(0x7F80_0000));

    let lsb = _mm512_and_si512(_mm512_srli_epi32::<16>(p_int), _mm512_set1_epi32(1));
    let bias = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);
    let rounded = _mm512_and_si512(_mm512_add_epi32(p_int, bias), _mm512_set1_epi32(!0xFFFF));
    let res_int = _mm512_mask_mov_epi32(p_int, k_finite, rounded);
    _mm512_castsi512_ps(res_int)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[inline]
pub unsafe fn stochastic_round_fp16_avx512(
    v_val: __m512,
    v_seed: __m512i,
    v_offsets: __m512i,
) -> __m512 {
    let p_int = _mm512_castps_si512(v_val);
    let abs_int = _mm512_and_si512(p_int, _mm512_set1_epi32(0x7FFF_FFFF));
    let exp_f32 = _mm512_and_si512(_mm512_srli_epi32::<23>(abs_int), _mm512_set1_epi32(0xFF));

    let diff_113 = _mm512_sub_epi32(_mm512_set1_epi32(113), exp_f32);
    let sub_shift = _mm512_min_epi32(
        _mm512_max_epi32(diff_113, _mm512_setzero_si512()),
        _mm512_set1_epi32(11),
    );
    let total_shift = _mm512_add_epi32(_mm512_set1_epi32(13), sub_shift);
    let one = _mm512_set1_epi32(1);
    let mask_bits = _mm512_sub_epi32(_mm512_sllv_epi32(one, total_shift), one);

    let noise = _mm512_and_si512(prng_avx512(v_seed, v_offsets), mask_bits);
    let not_mask = _mm512_xor_si512(mask_bits, _mm512_set1_epi32(-1));
    let added = _mm512_and_si512(_mm512_add_epi32(abs_int, noise), not_mask);

    let k_not_underflow: __mmask16 = _mm512_cmp_epi32_mask::<5>(exp_f32, _mm512_set1_epi32(103));
    let rounded_abs = _mm512_maskz_mov_epi32(k_not_underflow, added);
    let sign_bit = _mm512_and_si512(p_int, _mm512_set1_epi32(i32::MIN));
    let out_int = _mm512_or_si512(sign_bit, rounded_abs);

    let exp_out = _mm512_srli_epi32::<23>(rounded_abs);
    let k_valid: __mmask16 = _mm512_cmp_epi32_mask::<2>(exp_f32, _mm512_set1_epi32(142))
        & _mm512_cmp_epi32_mask::<2>(exp_out, _mm512_set1_epi32(142));

    if k_valid == 0xFFFF {
        _mm512_castsi512_ps(out_int)
    } else {
        let mut lanes = [0.0_f32; 16];
        let mut seeds = [0_i32; 16];
        let mut offs = [0_i32; 16];
        _mm512_storeu_ps(lanes.as_mut_ptr(), v_val);
        _mm512_storeu_si512(seeds.as_mut_ptr().cast(), v_seed);
        _mm512_storeu_si512(offs.as_mut_ptr().cast(), v_offsets);
        for i in 0..16 {
            lanes[i] = stochastic_round_fp16(lanes[i], seeds[i], offs[i]);
        }
        _mm512_loadu_ps(lanes.as_ptr())
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[inline]
pub unsafe fn round_to_nearest_fp16_avx512(v_val: __m512) -> __m512 {
    let p_int = _mm512_castps_si512(v_val);
    let abs_int = _mm512_and_si512(p_int, _mm512_set1_epi32(0x7FFF_FFFF));
    let exp_f32 = _mm512_and_si512(_mm512_srli_epi32::<23>(abs_int), _mm512_set1_epi32(0xFF));

    let lsb = _mm512_and_si512(_mm512_srli_epi32::<13>(abs_int), _mm512_set1_epi32(1));
    let bias = _mm512_add_epi32(_mm512_set1_epi32(0x0FFF), lsb);
    let added = _mm512_and_si512(_mm512_add_epi32(abs_int, bias), _mm512_set1_epi32(!0x1FFF));

    let k_normal: __mmask16 = _mm512_cmp_epi32_mask::<5>(exp_f32, _mm512_set1_epi32(113));
    let k_underflow: __mmask16 = _mm512_cmp_epi32_mask::<1>(exp_f32, _mm512_set1_epi32(102));
    let rounded_abs = _mm512_maskz_mov_epi32(k_normal, added);
    let sign_bit = _mm512_and_si512(p_int, _mm512_set1_epi32(i32::MIN));
    let out_int = _mm512_or_si512(sign_bit, rounded_abs);

    let exp_out = _mm512_srli_epi32::<23>(rounded_abs);
    let k_valid: __mmask16 = (k_normal | k_underflow)
        & _mm512_cmp_epi32_mask::<2>(exp_f32, _mm512_set1_epi32(142))
        & _mm512_cmp_epi32_mask::<2>(exp_out, _mm512_set1_epi32(142));

    if k_valid == 0xFFFF {
        _mm512_castsi512_ps(out_int)
    } else {
        let mut lanes = [0.0_f32; 16];
        _mm512_storeu_ps(lanes.as_mut_ptr(), v_val);
        for v in &mut lanes {
            *v = luminav_core::round_to_nearest_fp16(*v);
        }
        _mm512_loadu_ps(lanes.as_ptr())
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[inline]
pub unsafe fn quantize_avx512(
    v_val: __m512,
    v_seed: __m512i,
    v_offsets: __m512i,
    dtype_mode: DTypeMode,
    use_sr: bool,
) -> __m512 {
    match dtype_mode {
        DTypeMode::Float32 => v_val,
        DTypeMode::BFloat16 => {
            if use_sr {
                stochastic_round_bf16_avx512(v_val, v_seed, v_offsets)
            } else {
                round_to_nearest_bf16_avx512(v_val)
            }
        }
        DTypeMode::Float16 => {
            if use_sr {
                stochastic_round_fp16_avx512(v_val, v_seed, v_offsets)
            } else {
                round_to_nearest_fp16_avx512(v_val)
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[inline]
pub unsafe fn tanh_pade_avx512(v_x_in: __m512) -> __m512 {
    let v_clamp_hi = _mm512_set1_ps(7.998_811_7_f32);
    let v_clamp_lo = _mm512_set1_ps(-7.998_811_7_f32);

    let v_a13 = _mm512_set1_ps(-2.760_768_4e-16_f32);
    let v_a11 = _mm512_set1_ps(2.000_188e-13_f32);
    let v_a9 = _mm512_set1_ps(-8.604_672e-11_f32);
    let v_a7 = _mm512_set1_ps(5.122_297e-8_f32);
    let v_a5 = _mm512_set1_ps(1.485_722_4e-5_f32);
    let v_a3 = _mm512_set1_ps(6.372_619_3e-4_f32);
    let v_a1 = _mm512_set1_ps(4.893_524_6e-3_f32);

    let v_b6 = _mm512_set1_ps(1.198_258_4e-6_f32);
    let v_b4 = _mm512_set1_ps(1.185_347_1e-4_f32);
    let v_b2 = _mm512_set1_ps(2.268_434_6e-3_f32);
    let v_b0 = _mm512_set1_ps(4.893_524_6e-3_f32);

    let v_x = _mm512_min_ps(v_clamp_hi, _mm512_max_ps(v_clamp_lo, v_x_in));
    let v_x2 = _mm512_mul_ps(v_x, v_x);

    let mut v_p = _mm512_fmadd_ps(v_a13, v_x2, v_a11);
    let mut v_q = _mm512_fmadd_ps(v_b6, v_x2, v_b4);
    v_p = _mm512_fmadd_ps(v_p, v_x2, v_a9);
    v_q = _mm512_fmadd_ps(v_q, v_x2, v_b2);
    v_p = _mm512_fmadd_ps(v_p, v_x2, v_a7);
    v_q = _mm512_fmadd_ps(v_q, v_x2, v_b0);
    v_p = _mm512_fmadd_ps(v_p, v_x2, v_a5);
    v_p = _mm512_fmadd_ps(v_p, v_x2, v_a3);
    v_p = _mm512_fmadd_ps(v_p, v_x2, v_a1);

    _mm512_div_ps(_mm512_mul_ps(v_x, v_p), v_q)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[inline]
unsafe fn apply_coordinate_bound_avx512(
    v_step: __m512,
    v_p_init: __m512,
    v_bound_ratio: __m512,
) -> __m512 {
    let v_eps = _mm512_set1_ps(1.0e-4_f32);
    let v_norm = _mm512_sqrt_ps(_mm512_fmadd_ps(v_p_init, v_p_init, v_eps));
    let v_bi = _mm512_mul_ps(v_bound_ratio, v_norm);
    _mm512_mul_ps(v_bi, tanh_pade_avx512(_mm512_div_ps(v_step, v_bi)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[inline]
unsafe fn apply_fp16_safety_bound_avx512(v_step: __m512, v_p_init: __m512) -> __m512 {
    let v_scale = _mm512_set1_ps(0.01_f32);
    let v_min = _mm512_set1_ps(1.5e-4_f32);
    let v_bsafe = _mm512_max_ps(_mm512_mul_ps(_mm512_abs_ps(v_p_init), v_scale), v_min);
    _mm512_mul_ps(v_bsafe, tanh_pade_avx512(_mm512_div_ps(v_step, v_bsafe)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
pub unsafe fn compute_diff_sum_chunk_avx512(grad: &[f32], exp_avg: &[f32]) -> f32 {
    let n = grad.len();
    let simd_len = n & !15;
    let mut v_acc = _mm512_setzero_ps();
    let mut i = 0;
    while i < simd_len {
        let v_g = sanitize_gradient_avx512(_mm512_loadu_ps(grad.as_ptr().add(i)));
        let v_m = _mm512_loadu_ps(exp_avg.as_ptr().add(i));
        v_acc = _mm512_add_ps(v_acc, clamp_diff_avx512(_mm512_sub_ps(v_g, v_m)));
        i += 16;
    }
    let mut sum = reduce_add_ps_avx512(v_acc);
    for idx in simd_len..n {
        let g = sanitize_gradient(grad[idx]);
        let m = exp_avg[idx];
        sum += (g - m).clamp(-1.0e18_f32, 1.0e18_f32);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[inline]
unsafe fn offset_vector_avx512(base_offset: i32) -> __m512i {
    _mm512_add_epi32(
        _mm512_set1_epi32(base_offset),
        _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn tdclm_single_pass_chunk_avx512(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    config: &LuminaVConfig,
    dtype_mode: DTypeMode,
    state_dtype_mode: DTypeMode,
    chunk_offset: usize,
    diff_mean: f32,
    cautious_scale: f32,
    bound_scale: f32,
    seed_m: i32,
    seed_p: i32,
) -> ChunkMetrics {
    let n = param.len();
    let simd_len = n & !15;
    let has_master = config.master_weights.is_enabled() && master_param.is_some();
    let bound_mode = config.effective_bound_type();

    let beta1 = config.betas.0;
    let one_minus_beta1 = 1.0_f32 - beta1;
    let beta1_sq = beta1 * beta1;
    let one_minus_beta1_sq = 1.0_f32 - beta1_sq;
    let effective_eps = if dtype_mode == DTypeMode::Float16 {
        config.eps.max(1.0e-4_f32)
    } else {
        config.eps
    };
    let lr_wd = config.lr * config.weight_decay;

    let v_beta1 = _mm512_set1_ps(beta1);
    let v_one_minus_beta1 = _mm512_set1_ps(one_minus_beta1);
    let v_beta1_sq = _mm512_set1_ps(beta1_sq);
    let v_one_minus_beta1_sq = _mm512_set1_ps(one_minus_beta1_sq);
    let v_lam = _mm512_set1_ps(config.lam);
    let v_eps = _mm512_set1_ps(effective_eps);
    let v_diff_mean = _mm512_set1_ps(diff_mean);
    let v_cautious_scale = _mm512_set1_ps(cautious_scale);
    let v_bound_scale = _mm512_set1_ps(bound_scale);
    let v_bound_ratio = _mm512_set1_ps(config.bound_ratio);
    let v_lr = _mm512_set1_ps(config.lr);
    let v_lr_wd = _mm512_set1_ps(lr_wd);
    let v_zero = _mm512_setzero_ps();
    let v_seed_m = _mm512_set1_epi32(seed_m);
    let v_seed_p = _mm512_set1_epi32(seed_p);

    let mut mask_count: u64 = 0;
    let mut v_u_sq_acc = _mm512_setzero_ps();
    let mut v_p_sq_acc = _mm512_setzero_ps();

    let mut mp_slice = master_param;

    let mut i = 0;
    while i < simd_len {
        let global_offset = (chunk_offset + i) as i32;
        let v_offsets = offset_vector_avx512(global_offset);

        let v_p_init = match &mp_slice {
            Some(mp) if has_master => _mm512_loadu_ps(mp.as_ptr().add(i)),
            _ => _mm512_loadu_ps(param.as_ptr().add(i)),
        };
        let v_g = sanitize_gradient_avx512(_mm512_loadu_ps(grad.as_ptr().add(i)));
        let v_m = _mm512_loadu_ps(exp_avg.as_ptr().add(i));

        let v_g_scaled = _mm512_mul_ps(v_one_minus_beta1, v_g);
        let v_m_new = _mm512_fmadd_ps(v_beta1, v_m, v_g_scaled);
        let v_clm_m = _mm512_fmadd_ps(v_beta1_sq, v_m, _mm512_mul_ps(v_one_minus_beta1_sq, v_g));
        let v_diff_gm = clamp_diff_avx512(_mm512_sub_ps(v_g, v_m));
        let v_diff = clamp_diff_avx512(_mm512_sub_ps(v_diff_gm, v_diff_mean));

        let v_env0 = _mm512_fmadd_ps(v_beta1_sq, _mm512_abs_ps(v_m), v_eps);
        let v_env1 = _mm512_fmadd_ps(v_one_minus_beta1_sq, _mm512_abs_ps(v_g), v_env0);
        let v_clm_env = _mm512_fmadd_ps(v_lam, _mm512_abs_ps(v_diff), v_env1);
        let v_u = _mm512_div_ps(v_clm_m, v_clm_env);

        let k_mask: __mmask16 = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(_mm512_mul_ps(v_u, v_g), v_zero);
        let v_u_masked = _mm512_maskz_mov_ps(k_mask, v_u);

        let v_m_stored = quantize_avx512(
            v_m_new,
            v_seed_m,
            v_offsets,
            state_dtype_mode,
            config.stochastic_rounding,
        );
        _mm512_storeu_ps(exp_avg.as_mut_ptr().add(i), v_m_stored);

        let v_delta_theta = if config.cautious {
            _mm512_mul_ps(v_u_masked, v_cautious_scale)
        } else {
            v_u
        };
        let mut v_step = _mm512_mul_ps(v_lr, v_delta_theta);

        match bound_mode {
            BoundType::Radial => {
                v_step = _mm512_mul_ps(v_step, v_bound_scale);
            }
            BoundType::Coordinate => {
                v_step = apply_coordinate_bound_avx512(v_step, v_p_init, v_bound_ratio);
            }
            BoundType::None => {}
        }

        if dtype_mode == DTypeMode::Float16 {
            v_step = apply_fp16_safety_bound_avx512(v_step, v_p_init);
        }

        let v_p_decayed = if config.weight_decay != 0.0_f32 {
            _mm512_fnmadd_ps(v_lr_wd, v_p_init, v_p_init)
        } else {
            v_p_init
        };
        let v_p_updated = _mm512_sub_ps(v_p_decayed, v_step);

        if let Some(mp) = &mut mp_slice
            && has_master
        {
            _mm512_storeu_ps(mp.as_mut_ptr().add(i), v_p_updated);
            let v_p_q = quantize_avx512(v_p_updated, v_seed_p, v_offsets, dtype_mode, false);
            _mm512_storeu_ps(param.as_mut_ptr().add(i), v_p_q);
        } else {
            let v_p_q = quantize_avx512(
                v_p_updated,
                v_seed_p,
                v_offsets,
                dtype_mode,
                config.stochastic_rounding,
            );
            _mm512_storeu_ps(param.as_mut_ptr().add(i), v_p_q);
        }

        if config.cautious {
            mask_count += u64::from(k_mask.count_ones());
        }
        if bound_mode == BoundType::Radial {
            let v_u_eff = if config.cautious { v_u_masked } else { v_u };
            v_u_sq_acc = _mm512_add_ps(v_u_sq_acc, _mm512_mul_ps(v_u_eff, v_u_eff));
            v_p_sq_acc = _mm512_add_ps(v_p_sq_acc, _mm512_mul_ps(v_p_init, v_p_init));
        }

        i += 16;
    }

    let mut u_sq_sum = if bound_mode == BoundType::Radial {
        reduce_add_ps_avx512(v_u_sq_acc)
    } else {
        0.0_f32
    };
    let mut p_sq_sum = if bound_mode == BoundType::Radial {
        reduce_add_ps_avx512(v_p_sq_acc)
    } else {
        0.0_f32
    };

    for idx in simd_len..n {
        let offset = (chunk_offset + idx) as i32;
        let p_init = match &mp_slice {
            Some(mp) if has_master => mp[idx],
            _ => param[idx],
        };
        let g = sanitize_gradient(grad[idx]);
        let m = exp_avg[idx];

        let m_new = beta1.mul_add(m, one_minus_beta1 * g);
        let clm_m = beta1_sq.mul_add(m, one_minus_beta1_sq * g);
        let diff_gm = (g - m).clamp(-1.0e18_f32, 1.0e18_f32);
        let diff = (diff_gm - diff_mean).clamp(-1.0e18_f32, 1.0e18_f32);

        let env0 = beta1_sq.mul_add(m.abs(), effective_eps);
        let env1 = one_minus_beta1_sq.mul_add(g.abs(), env0);
        let clm_env = config.lam.mul_add(diff.abs(), env1);
        let u = clm_m / clm_env;
        let mask_active = (u * g) > 0.0_f32;
        let m_mask = if mask_active { 1.0_f32 } else { 0.0_f32 };

        exp_avg[idx] = quantize_value(
            m_new,
            seed_m,
            offset,
            state_dtype_mode,
            config.stochastic_rounding,
        );

        let delta_theta = if config.cautious {
            (u * m_mask) * cautious_scale
        } else {
            u
        };
        let mut step_val = config.lr * delta_theta;

        match bound_mode {
            BoundType::Radial => {
                step_val *= bound_scale;
            }
            BoundType::Coordinate => {
                step_val = apply_coordinate_bound(step_val, p_init, config.bound_ratio);
            }
            BoundType::None => {}
        }

        if dtype_mode == DTypeMode::Float16 {
            step_val = apply_fp16_safety_bound(step_val, p_init);
        }

        let p_decayed = if config.weight_decay != 0.0_f32 {
            (-lr_wd).mul_add(p_init, p_init)
        } else {
            p_init
        };
        let p_updated = p_decayed - step_val;

        if let Some(mp) = &mut mp_slice
            && has_master
        {
            mp[idx] = p_updated;
            param[idx] = quantize_value(p_updated, seed_p, offset, dtype_mode, false);
        } else {
            param[idx] = quantize_value(
                p_updated,
                seed_p,
                offset,
                dtype_mode,
                config.stochastic_rounding,
            );
        }

        if config.cautious && mask_active {
            mask_count += 1;
        }
        if bound_mode == BoundType::Radial {
            let u_eff = if config.cautious { u * m_mask } else { u };
            u_sq_sum += u_eff * u_eff;
            p_sq_sum += p_init * p_init;
        }
    }

    ChunkMetrics {
        mask_count,
        u_sq_sum,
        p_sq_sum,
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
pub unsafe fn tdclm_pass1_chunk_avx512(
    param: &[f32],
    master_param: Option<&[f32]>,
    grad: &[f32],
    exp_avg: &[f32],
    config: &LuminaVConfig,
    dtype_mode: DTypeMode,
    diff_mean: f32,
) -> ChunkMetrics {
    let n = param.len();
    let simd_len = n & !15;
    let has_master = config.master_weights.is_enabled() && master_param.is_some();
    let bound_mode = config.effective_bound_type();

    let beta1 = config.betas.0;
    let beta1_sq = beta1 * beta1;
    let one_minus_beta1_sq = 1.0_f32 - beta1_sq;
    let effective_eps = if dtype_mode == DTypeMode::Float16 {
        config.eps.max(1.0e-4_f32)
    } else {
        config.eps
    };

    let v_beta1_sq = _mm512_set1_ps(beta1_sq);
    let v_one_minus_beta1_sq = _mm512_set1_ps(one_minus_beta1_sq);
    let v_lam = _mm512_set1_ps(config.lam);
    let v_eps = _mm512_set1_ps(effective_eps);
    let v_diff_mean = _mm512_set1_ps(diff_mean);
    let v_zero = _mm512_setzero_ps();

    let mut mask_count: u64 = 0;
    let mut v_u_sq_acc = _mm512_setzero_ps();
    let mut v_p_sq_acc = _mm512_setzero_ps();

    let mut i = 0;
    while i < simd_len {
        let v_g = sanitize_gradient_avx512(_mm512_loadu_ps(grad.as_ptr().add(i)));
        let v_m = _mm512_loadu_ps(exp_avg.as_ptr().add(i));
        let v_diff_gm = clamp_diff_avx512(_mm512_sub_ps(v_g, v_m));
        let v_diff = clamp_diff_avx512(_mm512_sub_ps(v_diff_gm, v_diff_mean));

        let v_clm_m = _mm512_fmadd_ps(v_beta1_sq, v_m, _mm512_mul_ps(v_one_minus_beta1_sq, v_g));
        let v_env0 = _mm512_fmadd_ps(v_beta1_sq, _mm512_abs_ps(v_m), v_eps);
        let v_env1 = _mm512_fmadd_ps(v_one_minus_beta1_sq, _mm512_abs_ps(v_g), v_env0);
        let v_clm_env = _mm512_fmadd_ps(v_lam, _mm512_abs_ps(v_diff), v_env1);
        let v_u = _mm512_div_ps(v_clm_m, v_clm_env);

        let k_mask: __mmask16 = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(_mm512_mul_ps(v_u, v_g), v_zero);

        if config.cautious {
            mask_count += u64::from(k_mask.count_ones());
        }
        if bound_mode == BoundType::Radial {
            let v_u_eff = if config.cautious {
                _mm512_maskz_mov_ps(k_mask, v_u)
            } else {
                v_u
            };
            let v_p_val = match master_param {
                Some(mp) if has_master => _mm512_loadu_ps(mp.as_ptr().add(i)),
                _ => _mm512_loadu_ps(param.as_ptr().add(i)),
            };
            v_u_sq_acc = _mm512_add_ps(v_u_sq_acc, _mm512_mul_ps(v_u_eff, v_u_eff));
            v_p_sq_acc = _mm512_add_ps(v_p_sq_acc, _mm512_mul_ps(v_p_val, v_p_val));
        }
        i += 16;
    }

    let mut u_sq_sum = if bound_mode == BoundType::Radial {
        reduce_add_ps_avx512(v_u_sq_acc)
    } else {
        0.0_f32
    };
    let mut p_sq_sum = if bound_mode == BoundType::Radial {
        reduce_add_ps_avx512(v_p_sq_acc)
    } else {
        0.0_f32
    };

    for idx in simd_len..n {
        let g = sanitize_gradient(grad[idx]);
        let m = exp_avg[idx];
        let diff_gm = (g - m).clamp(-1.0e18_f32, 1.0e18_f32);
        let diff = (diff_gm - diff_mean).clamp(-1.0e18_f32, 1.0e18_f32);

        let clm_m = beta1_sq.mul_add(m, one_minus_beta1_sq * g);
        let env0 = beta1_sq.mul_add(m.abs(), effective_eps);
        let env1 = one_minus_beta1_sq.mul_add(g.abs(), env0);
        let clm_env = config.lam.mul_add(diff.abs(), env1);
        let u = clm_m / clm_env;
        let mask_active = (u * g) > 0.0_f32;
        let m_mask = if mask_active { 1.0_f32 } else { 0.0_f32 };

        if config.cautious && mask_active {
            mask_count += 1;
        }
        if bound_mode == BoundType::Radial {
            let u_eff = if config.cautious { u * m_mask } else { u };
            let p_val = match master_param {
                Some(mp) if has_master => mp[idx],
                _ => param[idx],
            };
            u_sq_sum += u_eff * u_eff;
            p_sq_sum += p_val * p_val;
        }
    }

    ChunkMetrics {
        mask_count,
        u_sq_sum,
        p_sq_sum,
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn tdclm_pass2_chunk_avx512(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    config: &LuminaVConfig,
    dtype_mode: DTypeMode,
    state_dtype_mode: DTypeMode,
    chunk_offset: usize,
    diff_mean: f32,
    cautious_scale: f32,
    bound_scale: f32,
    seed_m: i32,
    seed_p: i32,
) {
    let n = param.len();
    let simd_len = n & !15;
    let has_master = config.master_weights.is_enabled() && master_param.is_some();
    let bound_mode = config.effective_bound_type();

    let beta1 = config.betas.0;
    let one_minus_beta1 = 1.0_f32 - beta1;
    let beta1_sq = beta1 * beta1;
    let one_minus_beta1_sq = 1.0_f32 - beta1_sq;
    let effective_eps = if dtype_mode == DTypeMode::Float16 {
        config.eps.max(1.0e-4_f32)
    } else {
        config.eps
    };
    let lr_wd = config.lr * config.weight_decay;

    let v_beta1 = _mm512_set1_ps(beta1);
    let v_one_minus_beta1 = _mm512_set1_ps(one_minus_beta1);
    let v_beta1_sq = _mm512_set1_ps(beta1_sq);
    let v_one_minus_beta1_sq = _mm512_set1_ps(one_minus_beta1_sq);
    let v_lam = _mm512_set1_ps(config.lam);
    let v_eps = _mm512_set1_ps(effective_eps);
    let v_diff_mean = _mm512_set1_ps(diff_mean);
    let v_cautious_scale = _mm512_set1_ps(cautious_scale);
    let v_bound_scale = _mm512_set1_ps(bound_scale);
    let v_bound_ratio = _mm512_set1_ps(config.bound_ratio);
    let v_lr = _mm512_set1_ps(config.lr);
    let v_lr_wd = _mm512_set1_ps(lr_wd);
    let v_zero = _mm512_setzero_ps();
    let v_seed_m = _mm512_set1_epi32(seed_m);
    let v_seed_p = _mm512_set1_epi32(seed_p);

    let mut mp_slice = master_param;

    let mut i = 0;
    while i < simd_len {
        let global_offset = (chunk_offset + i) as i32;
        let v_offsets = offset_vector_avx512(global_offset);

        let v_p_init = match &mp_slice {
            Some(mp) if has_master => _mm512_loadu_ps(mp.as_ptr().add(i)),
            _ => _mm512_loadu_ps(param.as_ptr().add(i)),
        };
        let v_g = sanitize_gradient_avx512(_mm512_loadu_ps(grad.as_ptr().add(i)));
        let v_m = _mm512_loadu_ps(exp_avg.as_ptr().add(i));
        let v_diff_gm = clamp_diff_avx512(_mm512_sub_ps(v_g, v_m));
        let v_diff = clamp_diff_avx512(_mm512_sub_ps(v_diff_gm, v_diff_mean));

        let v_m_new = _mm512_fmadd_ps(v_beta1, v_m, _mm512_mul_ps(v_one_minus_beta1, v_g));
        let v_clm_m = _mm512_fmadd_ps(v_beta1_sq, v_m, _mm512_mul_ps(v_one_minus_beta1_sq, v_g));
        let v_env0 = _mm512_fmadd_ps(v_beta1_sq, _mm512_abs_ps(v_m), v_eps);
        let v_env1 = _mm512_fmadd_ps(v_one_minus_beta1_sq, _mm512_abs_ps(v_g), v_env0);
        let v_clm_env = _mm512_fmadd_ps(v_lam, _mm512_abs_ps(v_diff), v_env1);
        let v_u = _mm512_div_ps(v_clm_m, v_clm_env);

        let k_mask: __mmask16 = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(_mm512_mul_ps(v_u, v_g), v_zero);
        let v_u_masked = _mm512_maskz_mov_ps(k_mask, v_u);

        let v_m_stored = quantize_avx512(
            v_m_new,
            v_seed_m,
            v_offsets,
            state_dtype_mode,
            config.stochastic_rounding,
        );
        _mm512_storeu_ps(exp_avg.as_mut_ptr().add(i), v_m_stored);

        let v_delta_theta = if config.cautious {
            _mm512_mul_ps(v_u_masked, v_cautious_scale)
        } else {
            v_u
        };
        let mut v_step = _mm512_mul_ps(v_lr, v_delta_theta);

        match bound_mode {
            BoundType::Radial => {
                v_step = _mm512_mul_ps(v_step, v_bound_scale);
            }
            BoundType::Coordinate => {
                v_step = apply_coordinate_bound_avx512(v_step, v_p_init, v_bound_ratio);
            }
            BoundType::None => {}
        }

        if dtype_mode == DTypeMode::Float16 {
            v_step = apply_fp16_safety_bound_avx512(v_step, v_p_init);
        }

        let v_p_decayed = if config.weight_decay != 0.0_f32 {
            _mm512_fnmadd_ps(v_lr_wd, v_p_init, v_p_init)
        } else {
            v_p_init
        };
        let v_p_updated = _mm512_sub_ps(v_p_decayed, v_step);

        if let Some(mp) = &mut mp_slice
            && has_master
        {
            _mm512_storeu_ps(mp.as_mut_ptr().add(i), v_p_updated);
            let v_p_q = quantize_avx512(v_p_updated, v_seed_p, v_offsets, dtype_mode, false);
            _mm512_storeu_ps(param.as_mut_ptr().add(i), v_p_q);
        } else {
            let v_p_q = quantize_avx512(
                v_p_updated,
                v_seed_p,
                v_offsets,
                dtype_mode,
                config.stochastic_rounding,
            );
            _mm512_storeu_ps(param.as_mut_ptr().add(i), v_p_q);
        }

        i += 16;
    }

    for idx in simd_len..n {
        let offset = (chunk_offset + idx) as i32;
        let p_init = match &mp_slice {
            Some(mp) if has_master => mp[idx],
            _ => param[idx],
        };
        let g = sanitize_gradient(grad[idx]);
        let m = exp_avg[idx];
        let diff_gm = (g - m).clamp(-1.0e18_f32, 1.0e18_f32);
        let diff = (diff_gm - diff_mean).clamp(-1.0e18_f32, 1.0e18_f32);

        let m_new = beta1.mul_add(m, one_minus_beta1 * g);
        let clm_m = beta1_sq.mul_add(m, one_minus_beta1_sq * g);
        let env0 = beta1_sq.mul_add(m.abs(), effective_eps);
        let env1 = one_minus_beta1_sq.mul_add(g.abs(), env0);
        let clm_env = config.lam.mul_add(diff.abs(), env1);
        let u = clm_m / clm_env;
        let m_mask = if (u * g) > 0.0_f32 { 1.0_f32 } else { 0.0_f32 };

        exp_avg[idx] = quantize_value(
            m_new,
            seed_m,
            offset,
            state_dtype_mode,
            config.stochastic_rounding,
        );

        let delta_theta = if config.cautious {
            (u * m_mask) * cautious_scale
        } else {
            u
        };
        let mut step_val = config.lr * delta_theta;

        match bound_mode {
            BoundType::Radial => {
                step_val *= bound_scale;
            }
            BoundType::Coordinate => {
                step_val = apply_coordinate_bound(step_val, p_init, config.bound_ratio);
            }
            BoundType::None => {}
        }

        if dtype_mode == DTypeMode::Float16 {
            step_val = apply_fp16_safety_bound(step_val, p_init);
        }

        let p_decayed = if config.weight_decay != 0.0_f32 {
            (-lr_wd).mul_add(p_init, p_init)
        } else {
            p_init
        };
        let p_updated = p_decayed - step_val;

        if let Some(mp) = &mut mp_slice
            && has_master
        {
            mp[idx] = p_updated;
            param[idx] = quantize_value(p_updated, seed_p, offset, dtype_mode, false);
        } else {
            param[idx] = quantize_value(
                p_updated,
                seed_p,
                offset,
                dtype_mode,
                config.stochastic_rounding,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dual_single_pass_chunk_avx512(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    exp_avg_sq: &mut [f32],
    config: &LuminaVConfig,
    coeffs: &BiasCorrectionCoeffs,
    dtype_mode: DTypeMode,
    state_dtype_mode: DTypeMode,
    chunk_offset: usize,
    diff_mean_scaled: f32,
    cautious_scale: f32,
    bound_scale: f32,
    seed_m: i32,
    seed_p: i32,
    accumulate_metrics: bool,
) -> ChunkMetrics {
    let n = param.len();
    let simd_len = n & !15;
    let has_master = config.master_weights.is_enabled() && master_param.is_some();
    let bound_mode = config.effective_bound_type();
    let (beta1, beta2) = config.betas;
    let one_minus_beta1 = 1.0_f32 - beta1;
    let one_minus_beta2 = 1.0_f32 - beta2;
    let lr_wd = config.lr * config.weight_decay;
    let clamp_v = dtype_mode == DTypeMode::Float16 || state_dtype_mode == DTypeMode::Float16;

    let v_beta1 = _mm512_set1_ps(beta1);
    let v_one_minus_beta1 = _mm512_set1_ps(one_minus_beta1);
    let v_beta2 = _mm512_set1_ps(beta2);
    let v_one_minus_beta2 = _mm512_set1_ps(one_minus_beta2);
    let v_c1 = _mm512_set1_ps(coeffs.c1);
    let v_c2 = _mm512_set1_ps(coeffs.c2);
    let v_diff_mean_scaled = _mm512_set1_ps(diff_mean_scaled);
    let v_cautious_scale = _mm512_set1_ps(cautious_scale);
    let v_bound_scale = _mm512_set1_ps(bound_scale);
    let v_bound_ratio = _mm512_set1_ps(config.bound_ratio);
    let v_lr = _mm512_set1_ps(config.lr);
    let v_lr_wd = _mm512_set1_ps(lr_wd);
    let v_zero = _mm512_setzero_ps();
    let v_sigma_min = _mm512_set1_ps(1.0e-12_f32);
    let v_vmax = _mm512_set1_ps(65_000.0_f32);
    let v_seed_m = _mm512_set1_epi32(seed_m);
    let v_seed_p = _mm512_set1_epi32(seed_p);

    let mut mask_count: u64 = 0;
    let mut v_u_sq_acc = _mm512_setzero_ps();
    let mut v_p_sq_acc = _mm512_setzero_ps();

    let mut mp_slice = master_param;

    let mut i = 0;
    while i < simd_len {
        let global_offset = (chunk_offset + i) as i32;
        let v_offsets = offset_vector_avx512(global_offset);

        let v_p_init = match &mp_slice {
            Some(mp) if has_master => _mm512_loadu_ps(mp.as_ptr().add(i)),
            _ => _mm512_loadu_ps(param.as_ptr().add(i)),
        };
        let v_g = sanitize_gradient_avx512(_mm512_loadu_ps(grad.as_ptr().add(i)));
        let v_m = _mm512_loadu_ps(exp_avg.as_ptr().add(i));
        let v_v = _mm512_loadu_ps(exp_avg_sq.as_ptr().add(i));

        let v_g_scaled = _mm512_mul_ps(v_one_minus_beta1, v_g);
        let v_m_new = _mm512_fmadd_ps(v_beta1, v_m, v_g_scaled);
        let v_clm_m = _mm512_fmadd_ps(v_beta1, v_m_new, v_g_scaled);
        let v_diff_gm = clamp_diff_avx512(_mm512_sub_ps(v_g, v_m_new));
        let v_diff = clamp_diff_avx512(_mm512_sub_ps(v_diff_gm, v_diff_mean_scaled));

        let v_diff_sq = _mm512_mul_ps(v_diff, v_diff);
        let mut v_v_new =
            _mm512_fmadd_ps(v_beta2, v_v, _mm512_mul_ps(v_one_minus_beta2, v_diff_sq));
        if clamp_v {
            v_v_new = _mm512_min_ps(v_v_new, v_vmax);
        }

        let v_sigma = _mm512_fmadd_ps(_mm512_sqrt_ps(_mm512_max_ps(v_v_new, v_zero)), v_c1, v_c2);
        let v_u = tanh_pade_avx512(_mm512_div_ps(v_clm_m, _mm512_max_ps(v_sigma, v_sigma_min)));

        let k_mask: __mmask16 = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(_mm512_mul_ps(v_u, v_g), v_zero);
        let v_u_masked = _mm512_maskz_mov_ps(k_mask, v_u);

        let v_m_stored = quantize_avx512(
            v_m_new,
            v_seed_m,
            v_offsets,
            state_dtype_mode,
            config.stochastic_rounding,
        );
        _mm512_storeu_ps(exp_avg.as_mut_ptr().add(i), v_m_stored);

        let v_v_stored = quantize_avx512(v_v_new, v_seed_m, v_offsets, state_dtype_mode, false);
        _mm512_storeu_ps(exp_avg_sq.as_mut_ptr().add(i), v_v_stored);

        let v_delta_theta = if config.cautious && accumulate_metrics {
            _mm512_mul_ps(v_u_masked, v_cautious_scale)
        } else {
            v_u
        };
        let mut v_step = _mm512_mul_ps(v_lr, v_delta_theta);

        match bound_mode {
            BoundType::Radial => {
                v_step = _mm512_mul_ps(v_step, v_bound_scale);
            }
            BoundType::Coordinate => {
                v_step = apply_coordinate_bound_avx512(v_step, v_p_init, v_bound_ratio);
            }
            BoundType::None => {}
        }

        if dtype_mode == DTypeMode::Float16 {
            v_step = apply_fp16_safety_bound_avx512(v_step, v_p_init);
        }

        let v_p_decayed = if config.weight_decay != 0.0_f32 {
            _mm512_fnmadd_ps(v_lr_wd, v_p_init, v_p_init)
        } else {
            v_p_init
        };
        let v_p_updated = _mm512_sub_ps(v_p_decayed, v_step);

        if let Some(mp) = &mut mp_slice
            && has_master
        {
            _mm512_storeu_ps(mp.as_mut_ptr().add(i), v_p_updated);
            let v_p_q = quantize_avx512(v_p_updated, v_seed_p, v_offsets, dtype_mode, false);
            _mm512_storeu_ps(param.as_mut_ptr().add(i), v_p_q);
        } else {
            let v_p_q = quantize_avx512(
                v_p_updated,
                v_seed_p,
                v_offsets,
                dtype_mode,
                config.stochastic_rounding,
            );
            _mm512_storeu_ps(param.as_mut_ptr().add(i), v_p_q);
        }

        if accumulate_metrics {
            if config.cautious {
                mask_count += u64::from(k_mask.count_ones());
            }
            if bound_mode == BoundType::Radial {
                let v_u_eff = if config.cautious { v_u_masked } else { v_u };
                v_u_sq_acc = _mm512_add_ps(v_u_sq_acc, _mm512_mul_ps(v_u_eff, v_u_eff));
                v_p_sq_acc = _mm512_add_ps(v_p_sq_acc, _mm512_mul_ps(v_p_init, v_p_init));
            }
        }

        i += 16;
    }

    let mut u_sq_sum = if accumulate_metrics && bound_mode == BoundType::Radial {
        reduce_add_ps_avx512(v_u_sq_acc)
    } else {
        0.0_f32
    };
    let mut p_sq_sum = if accumulate_metrics && bound_mode == BoundType::Radial {
        reduce_add_ps_avx512(v_p_sq_acc)
    } else {
        0.0_f32
    };

    for idx in simd_len..n {
        let offset = (chunk_offset + idx) as i32;
        let p_init = match &mp_slice {
            Some(mp) if has_master => mp[idx],
            _ => param[idx],
        };
        let g = sanitize_gradient(grad[idx]);
        let m = exp_avg[idx];
        let v = exp_avg_sq[idx];

        let m_new = beta1.mul_add(m, one_minus_beta1 * g);
        let clm_m = beta1.mul_add(m_new, one_minus_beta1 * g);
        let diff_gm = (g - m_new).clamp(-1.0e18_f32, 1.0e18_f32);
        let diff = (diff_gm - diff_mean_scaled).clamp(-1.0e18_f32, 1.0e18_f32);

        let mut v_new = beta2.mul_add(v, one_minus_beta2 * (diff * diff));
        if clamp_v {
            v_new = v_new.min(65_000.0_f32);
        }

        let sigma = v_new.max(0.0_f32).sqrt().mul_add(coeffs.c1, coeffs.c2);
        let u = tanh_pade_f32(clm_m / sigma.max(1.0e-12_f32));
        let mask_active = (u * g) > 0.0_f32;
        let m_mask = if mask_active { 1.0_f32 } else { 0.0_f32 };

        exp_avg[idx] = quantize_value(
            m_new,
            seed_m,
            offset,
            state_dtype_mode,
            config.stochastic_rounding,
        );
        exp_avg_sq[idx] = quantize_value(v_new, seed_m, offset, state_dtype_mode, false);

        let delta_theta = if config.cautious && accumulate_metrics {
            (u * m_mask) * cautious_scale
        } else {
            u
        };
        let mut step_val = config.lr * delta_theta;

        match bound_mode {
            BoundType::Radial => {
                step_val *= bound_scale;
            }
            BoundType::Coordinate => {
                step_val = apply_coordinate_bound(step_val, p_init, config.bound_ratio);
            }
            BoundType::None => {}
        }

        if dtype_mode == DTypeMode::Float16 {
            step_val = apply_fp16_safety_bound(step_val, p_init);
        }

        let p_decayed = if config.weight_decay != 0.0_f32 {
            (-lr_wd).mul_add(p_init, p_init)
        } else {
            p_init
        };
        let p_updated = p_decayed - step_val;

        if let Some(mp) = &mut mp_slice
            && has_master
        {
            mp[idx] = p_updated;
            param[idx] = quantize_value(p_updated, seed_p, offset, dtype_mode, false);
        } else {
            param[idx] = quantize_value(
                p_updated,
                seed_p,
                offset,
                dtype_mode,
                config.stochastic_rounding,
            );
        }

        if accumulate_metrics {
            if config.cautious && mask_active {
                mask_count += 1;
            }
            if bound_mode == BoundType::Radial {
                let u_eff = if config.cautious { u * m_mask } else { u };
                u_sq_sum += u_eff * u_eff;
                p_sq_sum += p_init * p_init;
            }
        }
    }

    ChunkMetrics {
        mask_count,
        u_sq_sum,
        p_sq_sum,
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dual_pass1_chunk_avx512(
    param: &[f32],
    master_param: Option<&[f32]>,
    grad: &[f32],
    exp_avg: &[f32],
    exp_avg_sq: &[f32],
    config: &LuminaVConfig,
    coeffs: &BiasCorrectionCoeffs,
    dtype_mode: DTypeMode,
    state_dtype_mode: DTypeMode,
    diff_mean_scaled: f32,
) -> ChunkMetrics {
    let n = param.len();
    let simd_len = n & !15;
    let has_master = config.master_weights.is_enabled() && master_param.is_some();
    let bound_mode = config.effective_bound_type();
    let (beta1, beta2) = config.betas;
    let one_minus_beta1 = 1.0_f32 - beta1;
    let one_minus_beta2 = 1.0_f32 - beta2;
    let clamp_v = dtype_mode == DTypeMode::Float16 || state_dtype_mode == DTypeMode::Float16;

    let v_beta1 = _mm512_set1_ps(beta1);
    let v_one_minus_beta1 = _mm512_set1_ps(one_minus_beta1);
    let v_beta2 = _mm512_set1_ps(beta2);
    let v_one_minus_beta2 = _mm512_set1_ps(one_minus_beta2);
    let v_c1 = _mm512_set1_ps(coeffs.c1);
    let v_c2 = _mm512_set1_ps(coeffs.c2);
    let v_diff_mean_scaled = _mm512_set1_ps(diff_mean_scaled);
    let v_zero = _mm512_setzero_ps();
    let v_sigma_min = _mm512_set1_ps(1.0e-12_f32);
    let v_vmax = _mm512_set1_ps(65_000.0_f32);

    let mut mask_count: u64 = 0;
    let mut v_u_sq_acc = _mm512_setzero_ps();
    let mut v_p_sq_acc = _mm512_setzero_ps();

    let mut i = 0;
    while i < simd_len {
        let v_g = sanitize_gradient_avx512(_mm512_loadu_ps(grad.as_ptr().add(i)));
        let v_m = _mm512_loadu_ps(exp_avg.as_ptr().add(i));
        let v_v = _mm512_loadu_ps(exp_avg_sq.as_ptr().add(i));

        let v_g_scaled = _mm512_mul_ps(v_one_minus_beta1, v_g);
        let v_m_new = _mm512_fmadd_ps(v_beta1, v_m, v_g_scaled);
        let v_clm_m = _mm512_fmadd_ps(v_beta1, v_m_new, v_g_scaled);
        let v_diff_gm = clamp_diff_avx512(_mm512_sub_ps(v_g, v_m_new));
        let v_diff = clamp_diff_avx512(_mm512_sub_ps(v_diff_gm, v_diff_mean_scaled));

        let v_diff_sq = _mm512_mul_ps(v_diff, v_diff);
        let mut v_v_new =
            _mm512_fmadd_ps(v_beta2, v_v, _mm512_mul_ps(v_one_minus_beta2, v_diff_sq));
        if clamp_v {
            v_v_new = _mm512_min_ps(v_v_new, v_vmax);
        }

        let v_sigma = _mm512_fmadd_ps(_mm512_sqrt_ps(_mm512_max_ps(v_v_new, v_zero)), v_c1, v_c2);
        let v_u = tanh_pade_avx512(_mm512_div_ps(v_clm_m, _mm512_max_ps(v_sigma, v_sigma_min)));

        let k_mask: __mmask16 = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(_mm512_mul_ps(v_u, v_g), v_zero);

        if config.cautious {
            mask_count += u64::from(k_mask.count_ones());
        }
        if bound_mode == BoundType::Radial {
            let v_u_eff = if config.cautious {
                _mm512_maskz_mov_ps(k_mask, v_u)
            } else {
                v_u
            };
            let v_p_val = match master_param {
                Some(mp) if has_master => _mm512_loadu_ps(mp.as_ptr().add(i)),
                _ => _mm512_loadu_ps(param.as_ptr().add(i)),
            };
            v_u_sq_acc = _mm512_add_ps(v_u_sq_acc, _mm512_mul_ps(v_u_eff, v_u_eff));
            v_p_sq_acc = _mm512_add_ps(v_p_sq_acc, _mm512_mul_ps(v_p_val, v_p_val));
        }

        i += 16;
    }

    let mut u_sq_sum = if bound_mode == BoundType::Radial {
        reduce_add_ps_avx512(v_u_sq_acc)
    } else {
        0.0_f32
    };
    let mut p_sq_sum = if bound_mode == BoundType::Radial {
        reduce_add_ps_avx512(v_p_sq_acc)
    } else {
        0.0_f32
    };

    for idx in simd_len..n {
        let g = sanitize_gradient(grad[idx]);
        let m = exp_avg[idx];
        let v = exp_avg_sq[idx];

        let m_new = beta1.mul_add(m, one_minus_beta1 * g);
        let clm_m = beta1.mul_add(m_new, one_minus_beta1 * g);
        let diff_gm = (g - m_new).clamp(-1.0e18_f32, 1.0e18_f32);
        let diff = (diff_gm - diff_mean_scaled).clamp(-1.0e18_f32, 1.0e18_f32);

        let mut v_new = beta2.mul_add(v, one_minus_beta2 * (diff * diff));
        if clamp_v {
            v_new = v_new.min(65_000.0_f32);
        }

        let sigma = v_new.max(0.0_f32).sqrt().mul_add(coeffs.c1, coeffs.c2);
        let u = tanh_pade_f32(clm_m / sigma.max(1.0e-12_f32));
        let mask_active = (u * g) > 0.0_f32;
        let m_mask = if mask_active { 1.0_f32 } else { 0.0_f32 };

        if config.cautious && mask_active {
            mask_count += 1;
        }
        if bound_mode == BoundType::Radial {
            let u_eff = if config.cautious { u * m_mask } else { u };
            let p_val = match master_param {
                Some(mp) if has_master => mp[idx],
                _ => param[idx],
            };
            u_sq_sum += u_eff * u_eff;
            p_sq_sum += p_val * p_val;
        }
    }

    ChunkMetrics {
        mask_count,
        u_sq_sum,
        p_sq_sum,
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dual_pass2_chunk_avx512(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    exp_avg_sq: &mut [f32],
    config: &LuminaVConfig,
    coeffs: &BiasCorrectionCoeffs,
    dtype_mode: DTypeMode,
    state_dtype_mode: DTypeMode,
    chunk_offset: usize,
    diff_mean_scaled: f32,
    cautious_scale: f32,
    bound_scale: f32,
    seed_m: i32,
    seed_p: i32,
) {
    let n = param.len();
    let simd_len = n & !15;
    let has_master = config.master_weights.is_enabled() && master_param.is_some();
    let bound_mode = config.effective_bound_type();
    let (beta1, beta2) = config.betas;
    let one_minus_beta1 = 1.0_f32 - beta1;
    let one_minus_beta2 = 1.0_f32 - beta2;
    let lr_wd = config.lr * config.weight_decay;
    let clamp_v = dtype_mode == DTypeMode::Float16 || state_dtype_mode == DTypeMode::Float16;

    let v_beta1 = _mm512_set1_ps(beta1);
    let v_one_minus_beta1 = _mm512_set1_ps(one_minus_beta1);
    let v_beta2 = _mm512_set1_ps(beta2);
    let v_one_minus_beta2 = _mm512_set1_ps(one_minus_beta2);
    let v_c1 = _mm512_set1_ps(coeffs.c1);
    let v_c2 = _mm512_set1_ps(coeffs.c2);
    let v_diff_mean_scaled = _mm512_set1_ps(diff_mean_scaled);
    let v_cautious_scale = _mm512_set1_ps(cautious_scale);
    let v_bound_scale = _mm512_set1_ps(bound_scale);
    let v_bound_ratio = _mm512_set1_ps(config.bound_ratio);
    let v_lr = _mm512_set1_ps(config.lr);
    let v_lr_wd = _mm512_set1_ps(lr_wd);
    let v_zero = _mm512_setzero_ps();
    let v_sigma_min = _mm512_set1_ps(1.0e-12_f32);
    let v_vmax = _mm512_set1_ps(65_000.0_f32);
    let v_seed_m = _mm512_set1_epi32(seed_m);
    let v_seed_p = _mm512_set1_epi32(seed_p);

    let mut mp_slice = master_param;

    let mut i = 0;
    while i < simd_len {
        let global_offset = (chunk_offset + i) as i32;
        let v_offsets = offset_vector_avx512(global_offset);

        let v_p_init = match &mp_slice {
            Some(mp) if has_master => _mm512_loadu_ps(mp.as_ptr().add(i)),
            _ => _mm512_loadu_ps(param.as_ptr().add(i)),
        };
        let v_g = sanitize_gradient_avx512(_mm512_loadu_ps(grad.as_ptr().add(i)));
        let v_m = _mm512_loadu_ps(exp_avg.as_ptr().add(i));
        let v_v = _mm512_loadu_ps(exp_avg_sq.as_ptr().add(i));

        let v_g_scaled = _mm512_mul_ps(v_one_minus_beta1, v_g);
        let v_m_new = _mm512_fmadd_ps(v_beta1, v_m, v_g_scaled);
        let v_clm_m = _mm512_fmadd_ps(v_beta1, v_m_new, v_g_scaled);
        let v_diff_gm = clamp_diff_avx512(_mm512_sub_ps(v_g, v_m_new));
        let v_diff = clamp_diff_avx512(_mm512_sub_ps(v_diff_gm, v_diff_mean_scaled));

        let v_diff_sq = _mm512_mul_ps(v_diff, v_diff);
        let mut v_v_new =
            _mm512_fmadd_ps(v_beta2, v_v, _mm512_mul_ps(v_one_minus_beta2, v_diff_sq));
        if clamp_v {
            v_v_new = _mm512_min_ps(v_v_new, v_vmax);
        }

        let v_sigma = _mm512_fmadd_ps(_mm512_sqrt_ps(_mm512_max_ps(v_v_new, v_zero)), v_c1, v_c2);
        let v_u = tanh_pade_avx512(_mm512_div_ps(v_clm_m, _mm512_max_ps(v_sigma, v_sigma_min)));

        let k_mask: __mmask16 = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(_mm512_mul_ps(v_u, v_g), v_zero);
        let v_u_masked = _mm512_maskz_mov_ps(k_mask, v_u);

        let v_m_stored = quantize_avx512(
            v_m_new,
            v_seed_m,
            v_offsets,
            state_dtype_mode,
            config.stochastic_rounding,
        );
        _mm512_storeu_ps(exp_avg.as_mut_ptr().add(i), v_m_stored);

        let v_v_stored = quantize_avx512(v_v_new, v_seed_m, v_offsets, state_dtype_mode, false);
        _mm512_storeu_ps(exp_avg_sq.as_mut_ptr().add(i), v_v_stored);

        let v_delta_theta = if config.cautious {
            _mm512_mul_ps(v_u_masked, v_cautious_scale)
        } else {
            v_u
        };
        let mut v_step = _mm512_mul_ps(v_lr, v_delta_theta);

        match bound_mode {
            BoundType::Radial => {
                v_step = _mm512_mul_ps(v_step, v_bound_scale);
            }
            BoundType::Coordinate => {
                v_step = apply_coordinate_bound_avx512(v_step, v_p_init, v_bound_ratio);
            }
            BoundType::None => {}
        }

        if dtype_mode == DTypeMode::Float16 {
            v_step = apply_fp16_safety_bound_avx512(v_step, v_p_init);
        }

        let v_p_decayed = if config.weight_decay != 0.0_f32 {
            _mm512_fnmadd_ps(v_lr_wd, v_p_init, v_p_init)
        } else {
            v_p_init
        };
        let v_p_updated = _mm512_sub_ps(v_p_decayed, v_step);

        if let Some(mp) = &mut mp_slice
            && has_master
        {
            _mm512_storeu_ps(mp.as_mut_ptr().add(i), v_p_updated);
            let v_p_q = quantize_avx512(v_p_updated, v_seed_p, v_offsets, dtype_mode, false);
            _mm512_storeu_ps(param.as_mut_ptr().add(i), v_p_q);
        } else {
            let v_p_q = quantize_avx512(
                v_p_updated,
                v_seed_p,
                v_offsets,
                dtype_mode,
                config.stochastic_rounding,
            );
            _mm512_storeu_ps(param.as_mut_ptr().add(i), v_p_q);
        }

        i += 16;
    }

    for idx in simd_len..n {
        let offset = (chunk_offset + idx) as i32;
        let p_init = match &mp_slice {
            Some(mp) if has_master => mp[idx],
            _ => param[idx],
        };
        let g = sanitize_gradient(grad[idx]);
        let m = exp_avg[idx];
        let v = exp_avg_sq[idx];

        let m_new = beta1.mul_add(m, one_minus_beta1 * g);
        let clm_m = beta1.mul_add(m_new, one_minus_beta1 * g);
        let diff_gm = (g - m_new).clamp(-1.0e18_f32, 1.0e18_f32);
        let diff = (diff_gm - diff_mean_scaled).clamp(-1.0e18_f32, 1.0e18_f32);

        let mut v_new = beta2.mul_add(v, one_minus_beta2 * (diff * diff));
        if clamp_v {
            v_new = v_new.min(65_000.0_f32);
        }

        let sigma = v_new.max(0.0_f32).sqrt().mul_add(coeffs.c1, coeffs.c2);
        let u = tanh_pade_f32(clm_m / sigma.max(1.0e-12_f32));
        let m_mask = if (u * g) > 0.0_f32 { 1.0_f32 } else { 0.0_f32 };

        exp_avg[idx] = quantize_value(
            m_new,
            seed_m,
            offset,
            state_dtype_mode,
            config.stochastic_rounding,
        );
        exp_avg_sq[idx] = quantize_value(v_new, seed_m, offset, state_dtype_mode, false);

        let delta_theta = if config.cautious {
            (u * m_mask) * cautious_scale
        } else {
            u
        };
        let mut step_val = config.lr * delta_theta;

        match bound_mode {
            BoundType::Radial => {
                step_val *= bound_scale;
            }
            BoundType::Coordinate => {
                step_val = apply_coordinate_bound(step_val, p_init, config.bound_ratio);
            }
            BoundType::None => {}
        }

        if dtype_mode == DTypeMode::Float16 {
            step_val = apply_fp16_safety_bound(step_val, p_init);
        }

        let p_decayed = if config.weight_decay != 0.0_f32 {
            (-lr_wd).mul_add(p_init, p_init)
        } else {
            p_init
        };
        let p_updated = p_decayed - step_val;

        if let Some(mp) = &mut mp_slice
            && has_master
        {
            mp[idx] = p_updated;
            param[idx] = quantize_value(p_updated, seed_p, offset, dtype_mode, false);
        } else {
            param[idx] = quantize_value(
                p_updated,
                seed_p,
                offset,
                dtype_mode,
                config.stochastic_rounding,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
pub unsafe fn tdclm_step_avx512(
    view: &mut TensorViewMut<'_>,
    config: &LuminaVConfig,
    step: u64,
    param_index: usize,
) -> Result<(), LuminaError> {
    config.validate()?;
    let has_master = config.master_weights.is_enabled();
    let n_elements = view.validate(BufferMode::TdClm, has_master)?;
    if n_elements == 0 {
        return Ok(());
    }

    let total_elements = n_elements as f32;
    let bound_mode = config.effective_bound_type();
    let (seed_m, seed_p) = derive_kernel_seeds(config.seed, step, param_index);

    let use_cd = config.centered_defect && n_elements > 1;
    let diff_mean = if use_cd {
        let mut diff_sum = 0.0_f32;
        let mut start = 0;
        while start < n_elements {
            let end = (start + luminav_core::reference_cpu::ELEM_CHUNK_SIZE).min(n_elements);
            diff_sum +=
                compute_diff_sum_chunk_avx512(&view.grad[start..end], &view.exp_avg[start..end]);
            start = end;
        }
        view.scratch.set_diff_sum(diff_sum);
        diff_sum / total_elements
    } else {
        0.0_f32
    };

    let use_single_pass = (config.fused_single_pass && step > 1)
        || (!config.cautious && bound_mode != BoundType::Radial);

    if use_single_pass {
        view.scratch.set_mask_sum(0.0);
        view.scratch.set_u_sq_sum(0.0);
        view.scratch.set_p_sq_sum(0.0);

        let cautious_scale = view.scratch.cautious_scale();
        let bound_scale = view.scratch.bound_scale();

        let mut total_mask: u64 = 0;
        let mut total_u_sq = 0.0_f32;
        let mut total_p_sq = 0.0_f32;

        let mut start = 0;
        while start < n_elements {
            let end = (start + luminav_core::reference_cpu::ELEM_CHUNK_SIZE).min(n_elements);
            let mp_chunk = view
                .master_param
                .as_deref_mut()
                .map(|mp| &mut mp[start..end]);
            let metrics = tdclm_single_pass_chunk_avx512(
                &mut view.param[start..end],
                mp_chunk,
                &view.grad[start..end],
                &mut view.exp_avg[start..end],
                config,
                view.dtype_mode,
                view.state_dtype_mode,
                start,
                diff_mean,
                cautious_scale,
                bound_scale,
                seed_m,
                seed_p,
            );
            total_mask += metrics.mask_count;
            total_u_sq += metrics.u_sq_sum;
            total_p_sq += metrics.p_sq_sum;
            start = end;
        }

        view.scratch.set_mask_sum(total_mask as f32);
        view.scratch.set_u_sq_sum(total_u_sq);
        view.scratch.set_p_sq_sum(total_p_sq);

        if config.cautious || bound_mode == BoundType::Radial {
            update_post_step_ema(view.scratch, total_elements, config);
        } else {
            view.scratch.reset_accumulators();
        }
    } else {
        view.scratch.set_mask_sum(0.0);
        view.scratch.set_u_sq_sum(0.0);
        view.scratch.set_p_sq_sum(0.0);

        let mut total_mask: u64 = 0;
        let mut total_u_sq = 0.0_f32;
        let mut total_p_sq = 0.0_f32;

        let mut start = 0;
        while start < n_elements {
            let end = (start + luminav_core::reference_cpu::ELEM_CHUNK_SIZE).min(n_elements);
            let mp_chunk = view.master_param.as_deref().map(|mp| &mp[start..end]);
            let metrics = tdclm_pass1_chunk_avx512(
                &view.param[start..end],
                mp_chunk,
                &view.grad[start..end],
                &view.exp_avg[start..end],
                config,
                view.dtype_mode,
                diff_mean,
            );
            total_mask += metrics.mask_count;
            total_u_sq += metrics.u_sq_sum;
            total_p_sq += metrics.p_sq_sum;
            start = end;
        }

        let mask_sum = total_mask as f32;
        view.scratch.set_mask_sum(mask_sum);
        view.scratch.set_u_sq_sum(total_u_sq);
        view.scratch.set_p_sq_sum(total_p_sq);

        let cautious_scale = if config.cautious {
            compute_cautious_scale(mask_sum, total_elements, config.cautious_clamp_min)
        } else {
            1.0_f32
        };

        let bound_scale = if bound_mode == BoundType::Radial {
            compute_radial_bound_scale(
                total_u_sq,
                total_p_sq,
                cautious_scale,
                config.lr,
                config.bound_ratio,
            )
        } else {
            1.0_f32
        };

        let mut start = 0;
        while start < n_elements {
            let end = (start + luminav_core::reference_cpu::ELEM_CHUNK_SIZE).min(n_elements);
            let mp_chunk = view
                .master_param
                .as_deref_mut()
                .map(|mp| &mut mp[start..end]);
            tdclm_pass2_chunk_avx512(
                &mut view.param[start..end],
                mp_chunk,
                &view.grad[start..end],
                &mut view.exp_avg[start..end],
                config,
                view.dtype_mode,
                view.state_dtype_mode,
                start,
                diff_mean,
                cautious_scale,
                bound_scale,
                seed_m,
                seed_p,
            );
            start = end;
        }

        view.scratch.reset_accumulators();
        view.scratch.set_cautious_scale(cautious_scale);
        view.scratch.set_bound_scale(bound_scale);
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
pub unsafe fn dual_step_avx512(
    view: &mut TensorViewMut<'_>,
    config: &LuminaVConfig,
    step: u64,
    param_index: usize,
) -> Result<(), LuminaError> {
    config.validate()?;
    let has_master = config.master_weights.is_enabled();
    let n_elements = view.validate(BufferMode::Dual, has_master)?;
    if n_elements == 0 {
        return Ok(());
    }

    let total_elements = n_elements as f32;
    let (beta1, beta2) = config.betas;
    let coeffs =
        compute_bias_correction(step, beta1, beta2, config.eps, config.tau, view.dtype_mode);
    let bound_mode = config.effective_bound_type();
    let (seed_m, seed_p) = derive_kernel_seeds(config.seed, step, param_index);

    let use_cd = config.centered_defect && n_elements > 1;
    let diff_mean_scaled = if use_cd {
        let mut diff_sum = 0.0_f32;
        let mut start = 0;
        while start < n_elements {
            let end = (start + luminav_core::reference_cpu::ELEM_CHUNK_SIZE).min(n_elements);
            diff_sum +=
                compute_diff_sum_chunk_avx512(&view.grad[start..end], &view.exp_avg[start..end]);
            start = end;
        }
        view.scratch.set_diff_sum(diff_sum);
        (diff_sum / total_elements) * beta1
    } else {
        0.0_f32
    };

    let exp_avg_sq = view
        .exp_avg_sq
        .as_deref_mut()
        .ok_or(LuminaError::MissingSecondaryStateBuffer)?;

    if config.fused_single_pass && step > 1 {
        view.scratch.set_mask_sum(0.0);
        view.scratch.set_u_sq_sum(0.0);
        view.scratch.set_p_sq_sum(0.0);

        let cautious_scale = view.scratch.cautious_scale();
        let bound_scale = view.scratch.bound_scale();

        let mut total_mask: u64 = 0;
        let mut total_u_sq = 0.0_f32;
        let mut total_p_sq = 0.0_f32;

        let mut start = 0;
        while start < n_elements {
            let end = (start + luminav_core::reference_cpu::ELEM_CHUNK_SIZE).min(n_elements);
            let mp_chunk = view
                .master_param
                .as_deref_mut()
                .map(|mp| &mut mp[start..end]);
            let metrics = dual_single_pass_chunk_avx512(
                &mut view.param[start..end],
                mp_chunk,
                &view.grad[start..end],
                &mut view.exp_avg[start..end],
                &mut exp_avg_sq[start..end],
                config,
                &coeffs,
                view.dtype_mode,
                view.state_dtype_mode,
                start,
                diff_mean_scaled,
                cautious_scale,
                bound_scale,
                seed_m,
                seed_p,
                true,
            );
            total_mask += metrics.mask_count;
            total_u_sq += metrics.u_sq_sum;
            total_p_sq += metrics.p_sq_sum;
            start = end;
        }

        view.scratch.set_mask_sum(total_mask as f32);
        view.scratch.set_u_sq_sum(total_u_sq);
        view.scratch.set_p_sq_sum(total_p_sq);

        update_post_step_ema(view.scratch, total_elements, config);
    } else if !config.cautious && bound_mode != BoundType::Radial {
        let mut start = 0;
        while start < n_elements {
            let end = (start + luminav_core::reference_cpu::ELEM_CHUNK_SIZE).min(n_elements);
            let mp_chunk = view
                .master_param
                .as_deref_mut()
                .map(|mp| &mut mp[start..end]);
            dual_single_pass_chunk_avx512(
                &mut view.param[start..end],
                mp_chunk,
                &view.grad[start..end],
                &mut view.exp_avg[start..end],
                &mut exp_avg_sq[start..end],
                config,
                &coeffs,
                view.dtype_mode,
                view.state_dtype_mode,
                start,
                diff_mean_scaled,
                1.0_f32,
                1.0_f32,
                seed_m,
                seed_p,
                false,
            );
            start = end;
        }
        view.scratch.reset_accumulators();
    } else {
        view.scratch.set_mask_sum(0.0);
        view.scratch.set_u_sq_sum(0.0);
        view.scratch.set_p_sq_sum(0.0);

        let mut total_mask: u64 = 0;
        let mut total_u_sq = 0.0_f32;
        let mut total_p_sq = 0.0_f32;

        let mut start = 0;
        while start < n_elements {
            let end = (start + luminav_core::reference_cpu::ELEM_CHUNK_SIZE).min(n_elements);
            let mp_chunk = view.master_param.as_deref().map(|mp| &mp[start..end]);
            let metrics = dual_pass1_chunk_avx512(
                &view.param[start..end],
                mp_chunk,
                &view.grad[start..end],
                &view.exp_avg[start..end],
                &exp_avg_sq[start..end],
                config,
                &coeffs,
                view.dtype_mode,
                view.state_dtype_mode,
                diff_mean_scaled,
            );
            total_mask += metrics.mask_count;
            total_u_sq += metrics.u_sq_sum;
            total_p_sq += metrics.p_sq_sum;
            start = end;
        }

        let mask_sum = total_mask as f32;
        view.scratch.set_mask_sum(mask_sum);
        view.scratch.set_u_sq_sum(total_u_sq);
        view.scratch.set_p_sq_sum(total_p_sq);

        let cautious_scale = if config.cautious {
            compute_cautious_scale(mask_sum, total_elements, config.cautious_clamp_min)
        } else {
            1.0_f32
        };

        let bound_scale = if bound_mode == BoundType::Radial {
            compute_radial_bound_scale(
                total_u_sq,
                total_p_sq,
                cautious_scale,
                config.lr,
                config.bound_ratio,
            )
        } else {
            1.0_f32
        };

        let mut start = 0;
        while start < n_elements {
            let end = (start + luminav_core::reference_cpu::ELEM_CHUNK_SIZE).min(n_elements);
            let mp_chunk = view
                .master_param
                .as_deref_mut()
                .map(|mp| &mut mp[start..end]);
            dual_pass2_chunk_avx512(
                &mut view.param[start..end],
                mp_chunk,
                &view.grad[start..end],
                &mut view.exp_avg[start..end],
                &mut exp_avg_sq[start..end],
                config,
                &coeffs,
                view.dtype_mode,
                view.state_dtype_mode,
                start,
                diff_mean_scaled,
                cautious_scale,
                bound_scale,
                seed_m,
                seed_p,
            );
            start = end;
        }

        view.scratch.reset_accumulators();
        view.scratch.set_cautious_scale(cautious_scale);
        view.scratch.set_bound_scale(bound_scale);
    }

    Ok(())
}
