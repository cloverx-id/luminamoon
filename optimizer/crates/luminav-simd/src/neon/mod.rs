pub mod dual;
pub mod tdclm;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    float32x4_t, uint32x4_t, vabsq_f32, vadd_f32, vaddq_f32, vaddq_s32, vaddq_u32, vandq_u32,
    vbicq_u32, vbslq_f32, vceqq_f32, vcgeq_f32, vcgtq_f32, vcgtq_u32, vcltq_f32, vcltq_u32,
    vdivq_f32, vdupq_n_f32, vdupq_n_s32, vdupq_n_u32, veorq_u32, vfmaq_f32, vfmsq_f32,
    vget_high_f32, vget_lane_f32, vget_low_f32, vld1q_f32, vld1q_u32, vmaxq_f32, vmaxq_s32,
    vminq_f32, vminq_s32, vminq_u32, vminvq_u32, vmulq_f32, vmulq_u32, vnegq_s32, vorrq_u32,
    vreinterpretq_f32_u32, vreinterpretq_s32_u32, vreinterpretq_u32_f32, vshlq_u32, vshrq_n_u32,
    vsqrtq_f32, vst1q_f32, vst1q_u32, vsubq_s32, vsubq_u32,
};

#[cfg(target_arch = "aarch64")]
use luminav_core::{DTypeMode, stochastic_round_fp16_weyl};

#[cfg(target_arch = "aarch64")]
pub use dual::{
    dual_pass1_chunk_neon, dual_pass2_chunk_neon, dual_single_pass_chunk_neon, dual_step_neon,
};
#[cfg(target_arch = "aarch64")]
pub use tdclm::{
    tdclm_pass1_chunk_neon, tdclm_pass2_chunk_neon, tdclm_single_pass_chunk_neon, tdclm_step_neon,
};

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn reduce_add_ps_neon_quad(
    v0: float32x4_t,
    v1: float32x4_t,
    v2: float32x4_t,
    v3: float32x4_t,
) -> f32 {
    // Matches the 16-lane tree order of AVX-512 / AVX2 / reduce_sum_16.
    let sum4 = vaddq_f32(vaddq_f32(v0, v2), vaddq_f32(v1, v3));
    let sum2 = vadd_f32(vget_low_f32(sum4), vget_high_f32(sum4));
    vget_lane_f32::<0>(sum2) + vget_lane_f32::<1>(sum2)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn clamp_diff_neon(v: float32x4_t) -> float32x4_t {
    vminq_f32(
        vdupq_n_f32(1.0e18_f32),
        vmaxq_f32(vdupq_n_f32(-1.0e18_f32), v),
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn sanitize_gradient_neon(v_g: float32x4_t) -> float32x4_t {
    let m_ord = vceqq_f32(v_g, v_g);
    let m_lt = vcltq_f32(vabsq_f32(v_g), vdupq_n_f32(1.0e38_f32));
    let m_valid = vandq_u32(m_ord, m_lt);
    vbslq_f32(m_valid, clamp_diff_neon(v_g), vdupq_n_f32(0.0_f32))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn prng_neon(v_seed: uint32x4_t, v_offsets: uint32x4_t) -> uint32x4_t {
    let c0 = vdupq_n_u32(0x9E37_79B9_u32);
    let c1 = vdupq_n_u32(0x85EB_5A6B_u32);
    let c2 = vdupq_n_u32(0xC2B2_AE35_u32);

    let mut z = vaddq_u32(vaddq_u32(v_offsets, v_seed), c0);
    z = vmulq_u32(veorq_u32(z, vshrq_n_u32::<16>(z)), c1);
    z = vmulq_u32(veorq_u32(z, vshrq_n_u32::<13>(z)), c2);
    veorq_u32(z, vshrq_n_u32::<16>(z))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn prng_weyl_neon(
    v_seed: uint32x4_t,
    v_offsets: uint32x4_t,
    v_step_offset: uint32x4_t,
) -> uint32x4_t {
    vaddq_u32(prng_neon(v_seed, v_offsets), v_step_offset)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn stochastic_round_bf16_weyl_neon(
    p: float32x4_t,
    v_seed: uint32x4_t,
    v_offsets: uint32x4_t,
    v_step_offset: uint32x4_t,
) -> float32x4_t {
    let p_u = vreinterpretq_u32_f32(p);
    let abs_u = vandq_u32(p_u, vdupq_n_u32(0x7FFF_FFFF));
    let m_not_nan = vcltq_u32(abs_u, vdupq_n_u32(0x7F80_0001));

    let noise = vshrq_n_u32::<16>(prng_weyl_neon(v_seed, v_offsets, v_step_offset));
    let added_abs = vbicq_u32(vaddq_u32(abs_u, noise), vdupq_n_u32(0xFFFF));
    let rounded_abs = vminq_u32(added_abs, vdupq_n_u32(0x7F7F_0000));
    let sign_bit = vandq_u32(p_u, vdupq_n_u32(0x8000_0000));
    let out_f32 = vreinterpretq_f32_u32(vorrq_u32(sign_bit, rounded_abs));

    let qnan_f32 = vreinterpretq_f32_u32(vbicq_u32(
        vorrq_u32(p_u, vdupq_n_u32(0x0040_0000)),
        vdupq_n_u32(0xFFFF),
    ));
    vbslq_f32(m_not_nan, out_f32, qnan_f32)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn stochastic_round_bf16_neon(
    p: float32x4_t,
    v_seed: uint32x4_t,
    v_offsets: uint32x4_t,
) -> float32x4_t {
    stochastic_round_bf16_weyl_neon(p, v_seed, v_offsets, vdupq_n_u32(0))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn round_to_nearest_bf16_neon(p: float32x4_t) -> float32x4_t {
    let p_u = vreinterpretq_u32_f32(p);
    let abs_u = vandq_u32(p_u, vdupq_n_u32(0x7FFF_FFFF));
    let m_not_nan = vcltq_u32(abs_u, vdupq_n_u32(0x7F80_0001));

    let lsb = vandq_u32(vshrq_n_u32::<16>(abs_u), vdupq_n_u32(1));
    let bias = vaddq_u32(vdupq_n_u32(0x7FFF), lsb);
    let added_abs = vbicq_u32(vaddq_u32(abs_u, bias), vdupq_n_u32(0xFFFF));
    let rounded_abs = vminq_u32(added_abs, vdupq_n_u32(0x7F7F_0000));
    let sign_bit = vandq_u32(p_u, vdupq_n_u32(0x8000_0000));
    let out_f32 = vreinterpretq_f32_u32(vorrq_u32(sign_bit, rounded_abs));

    let qnan_f32 = vreinterpretq_f32_u32(vbicq_u32(
        vorrq_u32(p_u, vdupq_n_u32(0x0040_0000)),
        vdupq_n_u32(0xFFFF),
    ));
    vbslq_f32(m_not_nan, out_f32, qnan_f32)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn stochastic_round_fp16_weyl_neon(
    v_val: float32x4_t,
    v_seed: uint32x4_t,
    v_offsets: uint32x4_t,
    v_step_offset: uint32x4_t,
) -> float32x4_t {
    let p_u = vreinterpretq_u32_f32(v_val);
    let abs_u = vandq_u32(p_u, vdupq_n_u32(0x7FFF_FFFF));
    let exp_f32 = vandq_u32(vshrq_n_u32::<23>(abs_u), vdupq_n_u32(0xFF));

    let diff_113 = vsubq_s32(vdupq_n_s32(113), vreinterpretq_s32_u32(exp_f32));
    let sub_shift = vminq_s32(vmaxq_s32(diff_113, vdupq_n_s32(0)), vdupq_n_s32(11));
    let total_shift = vaddq_s32(vdupq_n_s32(13), sub_shift);
    let one = vdupq_n_u32(1);
    let mask_bits = vsubq_u32(vshlq_u32(one, total_shift), one);
    let neg_rshift = vnegq_s32(vsubq_s32(vdupq_n_s32(32), total_shift));

    let noise = vshlq_u32(prng_weyl_neon(v_seed, v_offsets, v_step_offset), neg_rshift);
    let added = vbicq_u32(vaddq_u32(abs_u, noise), mask_bits);

    let m_not_underflow = vcgtq_u32(exp_f32, vdupq_n_u32(102));
    let rounded_abs = vandq_u32(added, m_not_underflow);
    let sign_bit = vandq_u32(p_u, vdupq_n_u32(0x8000_0000));
    let out_u = vorrq_u32(sign_bit, rounded_abs);

    let exp_out = vshrq_n_u32::<23>(rounded_abs);
    let m_valid = vandq_u32(
        vcltq_u32(exp_f32, vdupq_n_u32(143)),
        vcltq_u32(exp_out, vdupq_n_u32(143)),
    );

    if vminvq_u32(m_valid) != 0 {
        vreinterpretq_f32_u32(out_u)
    } else {
        let mut lanes = [0.0_f32; 4];
        let mut seeds = [0_u32; 4];
        let mut offs = [0_u32; 4];
        let mut step_offs = [0_u32; 4];
        vst1q_f32(lanes.as_mut_ptr(), v_val);
        vst1q_u32(seeds.as_mut_ptr(), v_seed);
        vst1q_u32(offs.as_mut_ptr(), v_offsets);
        vst1q_u32(step_offs.as_mut_ptr(), v_step_offset);
        for i in 0..4 {
            lanes[i] =
                stochastic_round_fp16_weyl(lanes[i], seeds[i] as i32, offs[i] as i32, step_offs[i]);
        }
        vld1q_f32(lanes.as_ptr())
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn stochastic_round_fp16_neon(
    v_val: float32x4_t,
    v_seed: uint32x4_t,
    v_offsets: uint32x4_t,
) -> float32x4_t {
    stochastic_round_fp16_weyl_neon(v_val, v_seed, v_offsets, vdupq_n_u32(0))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn round_to_nearest_fp16_neon(v_val: float32x4_t) -> float32x4_t {
    let p_u = vreinterpretq_u32_f32(v_val);
    let abs_u = vandq_u32(p_u, vdupq_n_u32(0x7FFF_FFFF));
    let exp_f32 = vandq_u32(vshrq_n_u32::<23>(abs_u), vdupq_n_u32(0xFF));

    let lsb = vandq_u32(vshrq_n_u32::<13>(abs_u), vdupq_n_u32(1));
    let bias = vaddq_u32(vdupq_n_u32(0x0FFF), lsb);
    let added = vbicq_u32(vaddq_u32(abs_u, bias), vdupq_n_u32(0x1FFF));

    let m_normal = vcgtq_u32(exp_f32, vdupq_n_u32(112));
    let m_underflow = vcltq_u32(exp_f32, vdupq_n_u32(102));
    let rounded_abs = vandq_u32(added, m_normal);
    let sign_bit = vandq_u32(p_u, vdupq_n_u32(0x8000_0000));
    let out_u = vorrq_u32(sign_bit, rounded_abs);

    let exp_out = vshrq_n_u32::<23>(rounded_abs);
    let m_fast_range = vorrq_u32(m_normal, m_underflow);
    let m_valid = vandq_u32(
        m_fast_range,
        vandq_u32(
            vcltq_u32(exp_f32, vdupq_n_u32(143)),
            vcltq_u32(exp_out, vdupq_n_u32(143)),
        ),
    );

    if vminvq_u32(m_valid) != 0 {
        vreinterpretq_f32_u32(out_u)
    } else {
        let mut lanes = [0.0_f32; 4];
        vst1q_f32(lanes.as_mut_ptr(), v_val);
        for v in &mut lanes {
            *v = luminav_core::round_to_nearest_fp16(*v);
        }
        vld1q_f32(lanes.as_ptr())
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn quantize_neon(
    v_val: float32x4_t,
    v_seed: uint32x4_t,
    v_offsets: uint32x4_t,
    v_step_offset: uint32x4_t,
    dtype_mode: DTypeMode,
    use_sr: bool,
) -> float32x4_t {
    match dtype_mode {
        DTypeMode::Float32 => v_val,
        DTypeMode::BFloat16 => {
            if use_sr {
                stochastic_round_bf16_weyl_neon(v_val, v_seed, v_offsets, v_step_offset)
            } else {
                round_to_nearest_bf16_neon(v_val)
            }
        }
        DTypeMode::Float16 => {
            if use_sr {
                stochastic_round_fp16_weyl_neon(v_val, v_seed, v_offsets, v_step_offset)
            } else {
                round_to_nearest_fp16_neon(v_val)
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn tanh_pade_neon(v_x_in: float32x4_t) -> float32x4_t {
    let v_x = vminq_f32(
        vdupq_n_f32(7.998_811_7_f32),
        vmaxq_f32(vdupq_n_f32(-7.998_811_7_f32), v_x_in),
    );
    let v_x2 = vmulq_f32(v_x, v_x);

    let mut v_p = vfmaq_f32(
        vdupq_n_f32(2.000_188e-13_f32),
        vdupq_n_f32(-2.760_768_4e-16_f32),
        v_x2,
    );
    let mut v_q = vfmaq_f32(
        vdupq_n_f32(1.185_347_1e-4_f32),
        vdupq_n_f32(1.198_258_4e-6_f32),
        v_x2,
    );
    v_p = vfmaq_f32(vdupq_n_f32(-8.604_672e-11_f32), v_p, v_x2);
    v_q = vfmaq_f32(vdupq_n_f32(2.268_434_6e-3_f32), v_q, v_x2);
    v_p = vfmaq_f32(vdupq_n_f32(5.122_297e-8_f32), v_p, v_x2);
    v_q = vfmaq_f32(vdupq_n_f32(4.893_524_6e-3_f32), v_q, v_x2);
    v_p = vfmaq_f32(vdupq_n_f32(1.485_722_4e-5_f32), v_p, v_x2);
    v_p = vfmaq_f32(vdupq_n_f32(6.372_619_3e-4_f32), v_p, v_x2);
    v_p = vfmaq_f32(vdupq_n_f32(4.893_524_6e-3_f32), v_p, v_x2);

    vdivq_f32(vmulq_f32(v_x, v_p), v_q)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(super) unsafe fn apply_rational_bound_neon(
    v_step: float32x4_t,
    v_bound: float32x4_t,
) -> float32x4_t {
    let v_zero = vdupq_n_f32(0.0_f32);
    let v_one = vdupq_n_f32(1.0_f32);
    let v_b = vmaxq_f32(v_bound, v_zero);
    let v_abs_s = vabsq_f32(v_step);
    let v_m = vminq_f32(v_b, v_abs_s);
    let v_big_m = vmaxq_f32(v_b, v_abs_s);
    let m_pos = vcgtq_f32(v_big_m, v_zero);
    let v_safe_m = vbslq_f32(m_pos, v_big_m, v_one);
    let v_t = vbslq_f32(m_pos, vdivq_f32(v_m, v_safe_m), v_zero);
    let v_one_plus_t = vaddq_f32(v_one, v_t);
    let v_den = vfmaq_f32(v_one_plus_t, v_t, v_t);
    let v_mag = vmulq_f32(v_m, vdivq_f32(v_one_plus_t, v_den));
    vbslq_f32(vdupq_n_u32(0x8000_0000), v_step, v_mag)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(super) unsafe fn apply_decoupled_weight_decay_neon(
    v_p_init: float32x4_t,
    v_step: float32x4_t,
    v_lr_wd: float32x4_t,
    cautious_wd: bool,
) -> float32x4_t {
    let v_decayed = vfmsq_f32(v_p_init, v_lr_wd, v_p_init);
    if cautious_wd {
        let m_ge = vcgeq_f32(vmulq_f32(v_step, v_p_init), vdupq_n_f32(0.0_f32));
        vbslq_f32(m_ge, v_decayed, v_p_init)
    } else {
        v_decayed
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(super) unsafe fn apply_coordinate_bound_neon(
    v_step: float32x4_t,
    v_p_init: float32x4_t,
    v_bound_ratio: float32x4_t,
    v_bound_floor: float32x4_t,
    is_fp16: bool,
) -> float32x4_t {
    let v_eps = vmulq_f32(v_bound_floor, v_bound_floor);
    let v_norm = vsqrtq_f32(vfmaq_f32(v_eps, v_p_init, v_p_init));
    let mut v_bi = vmulq_f32(v_bound_ratio, v_norm);
    if is_fp16 {
        let v_bsafe = vmaxq_f32(
            vmulq_f32(vabsq_f32(v_p_init), vdupq_n_f32(0.01_f32)),
            vdupq_n_f32(1.5e-4_f32),
        );
        v_bi = vminq_f32(v_bi, v_bsafe);
    }
    let v_bi_safe = vmaxq_f32(v_bi, vdupq_n_f32(1.0e-12_f32));
    vmulq_f32(v_bi, tanh_pade_neon(vdivq_f32(v_step, v_bi_safe)))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(super) unsafe fn apply_coordinate_bound_t2_neon(
    v_step: float32x4_t,
    v_p_init: float32x4_t,
    v_bound_ratio: float32x4_t,
    v_bound_floor: float32x4_t,
    is_fp16: bool,
) -> float32x4_t {
    let v_abs_p = vabsq_f32(v_p_init);
    let mut v_bi = vmulq_f32(v_bound_ratio, vaddq_f32(v_abs_p, v_bound_floor));
    if is_fp16 {
        let v_bsafe = vmaxq_f32(
            vmulq_f32(v_abs_p, vdupq_n_f32(0.01_f32)),
            vdupq_n_f32(1.5e-4_f32),
        );
        v_bi = vminq_f32(v_bi, v_bsafe);
    }
    apply_rational_bound_neon(v_step, v_bi)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(super) unsafe fn apply_fp16_safety_bound_neon(
    v_step: float32x4_t,
    v_p_init: float32x4_t,
) -> float32x4_t {
    let v_bsafe = vmaxq_f32(
        vmulq_f32(vabsq_f32(v_p_init), vdupq_n_f32(0.01_f32)),
        vdupq_n_f32(1.5e-4_f32),
    );
    vmulq_f32(v_bsafe, tanh_pade_neon(vdivq_f32(v_step, v_bsafe)))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(super) unsafe fn apply_fp16_safety_bound_t2_neon(
    v_step: float32x4_t,
    v_p_init: float32x4_t,
) -> float32x4_t {
    let v_bsafe = vmaxq_f32(
        vmulq_f32(vabsq_f32(v_p_init), vdupq_n_f32(0.01_f32)),
        vdupq_n_f32(1.5e-4_f32),
    );
    apply_rational_bound_neon(v_step, v_bsafe)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(super) unsafe fn offset_vector_neon(base_offset: i32) -> uint32x4_t {
    const LANE_OFFSETS: [u32; 4] = [0, 1, 2, 3];
    vaddq_u32(
        vdupq_n_u32(base_offset as u32),
        vld1q_u32(LANE_OFFSETS.as_ptr()),
    )
}
