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
pub fn coord_bound_cube(step_val: f32, p_init: f32, bound_ratio: f32) -> f32 {
    let b_i = bound_ratio * f32::sqrt(p_init * p_init + 1.0e-4_f32);
    b_i * tanh_pade_cube(step_val / b_i)
}

#[cube]
pub fn coord_bound_t2_cube(step_val: f32, p_init: f32, bound_ratio: f32) -> f32 {
    let b_i = bound_ratio * (f32::abs(p_init) + 1.0e-2_f32);
    let b_bp = b_i * (b_i + f32::abs(step_val));
    (b_bp * step_val) / (step_val * step_val + b_bp)
}

#[cube]
pub fn fp16_safety_bound_cube(step_val: f32, p_init: f32) -> f32 {
    let b_safe = f32::max(f32::abs(p_init) * 0.01_f32, 1.5e-4_f32);
    b_safe * tanh_pade_cube(step_val / b_safe)
}

#[cube]
pub fn fp16_safety_bound_t2_cube(step_val: f32, p_init: f32) -> f32 {
    let b_safe = f32::max(f32::abs(p_init) * 0.01_f32, 1.5e-4_f32);
    let b_bp = b_safe * (b_safe + f32::abs(step_val));
    (b_bp * step_val) / (step_val * step_val + b_bp)
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
    let exp = bits & 0x7F80_0000_u32;
    let is_special = exp == 0x7F80_0000_u32;
    if is_special {
        val
    } else if stochastic != 0_u32 {
        let rand16 = weyl_prng_cube(base_seed, idx, step_offset) >> 16_u32;
        let rounded = (bits + rand16) & 0xFFFF_0000_u32;
        f32::reinterpret(rounded)
    } else {
        let lsb = (bits >> 16_u32) & 1_u32;
        let rounded = (bits + 0x7FFF_u32 + lsb) & 0xFFFF_0000_u32;
        f32::reinterpret(rounded)
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
    let clamped = f32::clamp(val, -65_504.0_f32, 65_504.0_f32);
    let bits = u32::reinterpret(clamped);
    let exp_biased = (bits >> 23_u32) & 0xFF_u32;
    if exp_biased < 113_u32 {
        // Subnormal / underflow region: preserve sign-zero cleanly in portable shader path.
        if f32::abs(clamped) < 5.960_464_5e-8_f32 {
            0.0_f32
        } else {
            let scaled = clamped * 16_777_216.0_f32;
            let q = if stochastic != 0_u32 {
                let u24 = f32::cast_from(weyl_prng_cube(base_seed, idx, step_offset) >> 8_u32)
                    * 5.960_464_5e-8_f32;
                f32::floor(scaled + u24)
            } else {
                f32::round(scaled)
            };
            q * 5.960_464_5e-8_f32
        }
    } else if stochastic != 0_u32 {
        let rand13 = weyl_prng_cube(base_seed, idx, step_offset) >> 19_u32;
        let rounded = (bits + rand13) & 0xFFFF_E000_u32;
        f32::clamp(f32::reinterpret(rounded), -65_504.0_f32, 65_504.0_f32)
    } else {
        let lsb = (bits >> 13_u32) & 1_u32;
        let rounded = (bits + 0x0FFF_u32 + lsb) & 0xFFFF_E000_u32;
        f32::clamp(f32::reinterpret(rounded), -65_504.0_f32, 65_504.0_f32)
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
