pub mod dual;
pub mod tdclm;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m512, __m512i, __mmask16, _CMP_LT_OQ, _CMP_ORD_Q, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32,
    _mm_movehdup_ps, _mm_movehl_ps, _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps,
    _mm512_abs_ps, _mm512_add_epi32, _mm512_and_epi32, _mm512_and_si512,
    _mm512_castps_si512, _mm512_castps512_ps256, _mm512_castsi512_ps, _mm512_cmp_epi32_mask,
    _mm512_cmp_ps_mask, _mm512_cmpgt_epi32_mask, _mm512_div_ps, _mm512_extractf32x8_ps,
    _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_mask_blend_ps, _mm512_maskz_mov_epi32,
    _mm512_maskz_mov_ps, _mm512_max_epi32, _mm512_max_ps, _mm512_min_epi32, _mm512_min_ps,
    _mm512_mul_ps, _mm512_mullo_epi32, _mm512_or_epi32, _mm512_or_si512, _mm512_set_epi32,
    _mm512_set1_epi32, _mm512_set1_ps, _mm512_setzero_si512, _mm512_sllv_epi32, _mm512_sqrt_ps,
    _mm512_srai_epi32, _mm512_srli_epi32, _mm512_storeu_ps, _mm512_storeu_si512,
    _mm512_sub_epi32, _mm512_xor_si512,
};

#[cfg(target_arch = "x86_64")]
use luminav_core::{DTypeMode, stochastic_round_fp16};

#[cfg(target_arch = "x86_64")]
pub use dual::{
    dual_pass1_chunk_avx512, dual_pass2_chunk_avx512, dual_single_pass_chunk_avx512,
    dual_step_avx512,
};
#[cfg(target_arch = "x86_64")]
pub use tdclm::{
    tdclm_pass1_chunk_avx512, tdclm_pass2_chunk_avx512, tdclm_single_pass_chunk_avx512,
    tdclm_step_avx512,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct ChunkMetrics {
    pub mask_count: u64,
    pub u_sq_sum: f32,
    pub p_sq_sum: f32,
}

impl ChunkMetrics {
    #[inline]
    pub fn combine(self, other: Self) -> Self {
        Self {
            mask_count: self.mask_count + other.mask_count,
            u_sq_sum: self.u_sq_sum + other.u_sq_sum,
            p_sq_sum: self.p_sq_sum + other.p_sq_sum,
        }
    }
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
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn stochastic_round_bf16_avx512(
    p: __m512,
    v_seed: __m512i,
    v_offsets: __m512i,
) -> __m512 {
    let p_int = _mm512_castps_si512(p);
    let abs_int = _mm512_and_epi32(p_int, _mm512_set1_epi32(0x7FFF_FFFF));
    let m_finite = _mm512_cmpgt_epi32_mask(_mm512_set1_epi32(0x7F80_0000), abs_int);

    let noise = _mm512_and_epi32(prng_avx512(v_seed, v_offsets), _mm512_set1_epi32(0xFFFF));
    let added_abs = _mm512_and_epi32(_mm512_add_epi32(abs_int, noise), _mm512_set1_epi32(!0xFFFF));
    let rounded_abs = _mm512_min_epi32(added_abs, _mm512_set1_epi32(0x7F7F_0000));
    let sign_bit = _mm512_and_epi32(p_int, _mm512_set1_epi32(i32::MIN));
    let out_int = _mm512_or_epi32(sign_bit, rounded_abs);
    let out_f32 = _mm512_castsi512_ps(out_int);

    _mm512_mask_blend_ps(m_finite, p, out_f32)
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn round_to_nearest_bf16_avx512(p: __m512) -> __m512 {
    let p_int = _mm512_castps_si512(p);
    let abs_int = _mm512_and_epi32(p_int, _mm512_set1_epi32(0x7FFF_FFFF));
    let m_finite = _mm512_cmpgt_epi32_mask(_mm512_set1_epi32(0x7F80_0000), abs_int);

    let lsb = _mm512_and_epi32(_mm512_srli_epi32::<16>(abs_int), _mm512_set1_epi32(1));
    let bias = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);
    let added_abs = _mm512_and_epi32(_mm512_add_epi32(abs_int, bias), _mm512_set1_epi32(!0xFFFF));
    let rounded_abs = _mm512_min_epi32(added_abs, _mm512_set1_epi32(0x7F7F_0000));
    let sign_bit = _mm512_and_epi32(p_int, _mm512_set1_epi32(i32::MIN));
    let out_f32 = _mm512_castsi512_ps(_mm512_or_epi32(sign_bit, rounded_abs));

    _mm512_mask_blend_ps(m_finite, p, out_f32)
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
#[inline]
unsafe fn offset_vector_avx512(base_offset: i32) -> __m512i {
    _mm512_add_epi32(
        _mm512_set1_epi32(base_offset),
        _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    )
}
