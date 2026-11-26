use cubecl::prelude::*;

#[cube]
#[allow(clippy::eq_op)]
pub fn sanitize_gradient_cube(g: f32) -> f32 {
    let abs_g = f32::abs(g);
    let is_valid = (g == g) && (abs_g < 1.0e38_f32);
    if is_valid {
        f32::clamp(g, -1.0e18_f32, 1.0e18_f32)
    } else {
        0.0_f32
    }
}

#[cube]
pub fn tanh_pade_cube(x_in: f32) -> f32 {
    let x = f32::clamp(x_in, -7.998_811_7_f32, 7.998_811_7_f32);
    let x2 = x * x;

    let mut p = -2.760_768_4e-16_f32 * x2 + 2.000_188e-13_f32;
    let mut q = 1.198_258_4e-6_f32 * x2 + 1.185_347_1e-4_f32;
    p = p * x2 - 8.604_672e-11_f32;
    q = q * x2 + 2.268_434_6e-3_f32;
    p = p * x2 + 5.122_297e-8_f32;
    q = q * x2 + 4.893_524_6e-3_f32;
    p = p * x2 + 1.485_722_4e-5_f32;
    p = p * x2 + 6.372_619_3e-4_f32;
    p = p * x2 + 4.893_524_6e-3_f32;

    (x * p) / q
}

#[cube]
pub fn rational_bound_cube(step_val: f32, bound: f32) -> f32 {
    let b = f32::max(bound, 0.0_f32);
    let abs_s = f32::abs(step_val);
    let m = f32::min(b, abs_s);
    let big_m = f32::max(b, abs_s);
    let t = if big_m > 0.0_f32 { m / big_m } else { 0.0_f32 };
    let one_plus_t = 1.0_f32 + t;
    let mag = m * (one_plus_t / (one_plus_t + t * t));
    let sign_bit = u32::reinterpret(step_val) & 0x8000_0000_u32;
    f32::reinterpret((u32::reinterpret(mag) & 0x7FFF_FFFF_u32) | sign_bit)
}

#[cube]
pub fn temper_belief_defect_cube(
    abs_d: f32,
    v_prev: f32,
    m_prev: f32,
    inv_sqrt_bc2_prev: f32,
    variance_temper: f32,
    eps: f32,
) -> f32 {
    let sigma_prev = f32::sqrt(f32::max(v_prev, 0.0_f32)) * inv_sqrt_bc2_prev;
    let scale_ref = f32::max(sigma_prev, f32::abs(m_prev));
    rational_bound_cube(abs_d, variance_temper * scale_ref + eps)
}

#[cube]
pub fn apply_rpm_cube(m: f32, m_p: f32, g: f32, g_p: f32, one_minus_beta1: f32, eps: f32) -> f32 {
    let c = m * g_p - m_p * g;
    let d = m * g + m_p * g_p;
    let s = (one_minus_beta1 * c) / (2.0_f32 * f32::abs(d) + f32::abs(c) + eps);
    let s2 = s * s;
    let inv = 1.0_f32 / (1.0_f32 + s2);
    let c_rot = (1.0_f32 - s2) * inv;
    let s_rot = (2.0_f32 * s) * inv;
    m * c_rot - m_p * s_rot
}

#[cube]
pub fn decoupled_weight_decay_cube(
    p_init: f32,
    step_val: f32,
    lr_wd: f32,
    #[comptime] cautious_wd: bool,
) -> f32 {
    if lr_wd == 0.0_f32 {
        p_init
    } else if comptime!(cautious_wd) {
        if step_val * p_init >= 0.0_f32 {
            p_init - lr_wd * p_init
        } else {
            p_init
        }
    } else {
        p_init - lr_wd * p_init
    }
}

#[cube]
pub fn coord_bound_cube(
    step_val: f32,
    p_init: f32,
    bound_ratio: f32,
    bound_floor: f32,
    dtype_mode: u32,
) -> f32 {
    let mut b_i = bound_ratio * f32::sqrt(p_init * p_init + bound_floor * bound_floor);
    if dtype_mode == 2_u32 {
        let b_safe = f32::max(f32::abs(p_init) * 0.01_f32, 1.5e-4_f32);
        b_i = f32::min(b_i, b_safe);
    }
    b_i * tanh_pade_cube(step_val / f32::max(b_i, 1.0e-12_f32))
}

#[cube]
pub fn coord_bound_t2_cube(
    step_val: f32,
    p_init: f32,
    bound_ratio: f32,
    bound_floor: f32,
    dtype_mode: u32,
) -> f32 {
    let abs_p = f32::abs(p_init);
    let mut b_i = bound_ratio * (abs_p + bound_floor);
    if dtype_mode == 2_u32 {
        let b_safe = f32::max(abs_p * 0.01_f32, 1.5e-4_f32);
        b_i = f32::min(b_i, b_safe);
    }
    rational_bound_cube(step_val, b_i)
}

#[cube]
pub fn fp16_safety_bound_cube(step_val: f32, p_init: f32) -> f32 {
    let b_safe = f32::max(f32::abs(p_init) * 0.01_f32, 1.5e-4_f32);
    b_safe * tanh_pade_cube(step_val / b_safe)
}

#[cube]
pub fn fp16_safety_bound_t2_cube(step_val: f32, p_init: f32) -> f32 {
    let b_safe = f32::max(f32::abs(p_init) * 0.01_f32, 1.5e-4_f32);
    rational_bound_cube(step_val, b_safe)
}

#[cube]
pub fn weyl_prng_cube(base_seed: u32, idx: u32, step_offset: u32) -> u32 {
    let mut z = idx + base_seed - 1_640_531_527_u32;
    z = (z ^ (z >> 16_u32)) * 0x85EB_5A6B_u32;
    z = (z ^ (z >> 13_u32)) * 0xC2B2_AE35_u32;
    (z ^ (z >> 16_u32)) + step_offset
}

#[cube]
pub fn quantize_bf16_weyl_cube(
    val: f32,
    base_seed: u32,
    idx: u32,
    step_offset: u32,
    stochastic: u32,
) -> f32 {
    let bits = u32::reinterpret(val);
    let abs_bits = bits & 0x7FFF_FFFF_u32;
    let sign_bit = bits & 0x8000_0000_u32;
    if abs_bits > 0x7F80_0000_u32 {
        f32::reinterpret((bits | 0x0040_0000_u32) & 0xFFFF_0000_u32)
    } else if stochastic != 0_u32 {
        let rand16 = weyl_prng_cube(base_seed, idx, step_offset) >> 16_u32;
        let added = (abs_bits + rand16) & 0xFFFF_0000_u32;
        let rounded_abs = u32::min(added, 0x7F7F_0000_u32);
        f32::reinterpret(sign_bit | rounded_abs)
    } else {
        let lsb = (abs_bits >> 16_u32) & 1_u32;
        let added = (abs_bits + 0x7FFF_u32 + lsb) & 0xFFFF_0000_u32;
        let rounded_abs = u32::min(added, 0x7F7F_0000_u32);
        f32::reinterpret(sign_bit | rounded_abs)
    }
}

#[cube]
pub fn quantize_fp16_weyl_cube(
    val: f32,
    base_seed: u32,
    idx: u32,
    step_offset: u32,
    stochastic: u32,
) -> f32 {
    let bits = u32::reinterpret(val);
    let abs_bits = bits & 0x7FFF_FFFF_u32;
    let sign_bit = bits & 0x8000_0000_u32;
    if abs_bits > 0x7F80_0000_u32 {
        f32::reinterpret(bits | 0x0040_0000_u32)
    } else {
        let exp_f32 = (abs_bits >> 23_u32) & 0xFF_u32;
        if stochastic != 0_u32 {
            if exp_f32 < 103_u32 {
                f32::reinterpret(sign_bit)
            } else {
                let sub_shift = if exp_f32 < 113_u32 {
                    u32::min(113_u32 - exp_f32, 11_u32)
                } else {
                    0_u32
                };
                let total_shift = 13_u32 + sub_shift;
                let mask_bits = (1_u32 << total_shift) - 1_u32;
                let noise = weyl_prng_cube(base_seed, idx, step_offset) >> (32_u32 - total_shift);
                let added_abs = (abs_bits + noise) & (mask_bits ^ 0xFFFF_FFFF_u32);
                let rounded_abs = u32::min(added_abs, 0x477F_E000_u32);
                f32::reinterpret(sign_bit | rounded_abs)
            }
        } else if exp_f32 < 102_u32 {
            f32::reinterpret(sign_bit)
        } else if exp_f32 < 113_u32 {
            // Subnormal FP16 RNE: quantize magnitude in units of 2^-24 (5.9604645e-8).
            let scaled = f32::abs(val) * 16_777_216.0_f32;
            let q_floor = f32::floor(scaled);
            let frac = scaled - q_floor;
            let is_odd = (u32::cast_from(q_floor) & 1_u32) != 0_u32;
            let q = if frac > 0.5_f32 || (frac == 0.5_f32 && is_odd) {
                q_floor + 1.0_f32
            } else {
                q_floor
            };
            let out_abs = q * 5.960_464_5e-8_f32;
            f32::reinterpret(sign_bit | (u32::reinterpret(out_abs) & 0x7FFF_FFFF_u32))
        } else {
            let lsb = (abs_bits >> 13_u32) & 1_u32;
            let added_abs = (abs_bits + 0x0FFF_u32 + lsb) & 0xFFFF_E000_u32;
            let rounded_abs = u32::min(added_abs, 0x477F_E000_u32);
            f32::reinterpret(sign_bit | rounded_abs)
        }
    }
}

#[cube]
pub fn quantize_value_weyl_cube(
    val: f32,
    base_seed: u32,
    idx: u32,
    step_offset: u32,
    dtype_mode: u32,
    stochastic: u32,
) -> f32 {
    if dtype_mode == 1_u32 {
        quantize_bf16_weyl_cube(val, base_seed, idx, step_offset, stochastic)
    } else if dtype_mode == 2_u32 {
        quantize_fp16_weyl_cube(val, base_seed, idx, step_offset, stochastic)
    } else {
        val
    }
}

#[cube]
pub fn reduce_cube_sum(val: f32, #[comptime] cube_dim: usize) -> f32 {
    let tid = UNIT_POS_X as usize;
    let mut smem = Shared::<[f32]>::new_slice(cube_dim);
    smem[tid] = val;
    sync_cube();
    let mut stride = (CUBE_DIM_X as usize) / 2;
    while stride > 0 {
        if tid < stride {
            let a = smem[tid];
            let b = smem[tid + stride];
            smem[tid] = a + b;
        }
        sync_cube();
        stride /= 2;
    }
    smem[0]
}
