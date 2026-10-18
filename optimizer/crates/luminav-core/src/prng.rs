use crate::config::DTypeMode;

#[inline]
pub const fn derive_kernel_seeds(seed: u32, step: u64, param_index: usize) -> (i32, i32) {
    let step_term = (step as u128).wrapping_mul(0x9E37_79B9_u128);
    let idx_term = (param_index as u128).wrapping_mul(10_007_u128);
    let raw = (seed as u128)
        .wrapping_add(step_term)
        .wrapping_add(idx_term);
    let seed_m = (raw & 0x7FFF_FFFF_u128) as i32;
    let seed_p = ((seed_m as u64).wrapping_add(0x5555_5555_u64) & 0x7FFF_FFFF_u64) as i32;
    (seed_m, seed_p)
}

#[inline]
pub const fn _prng(seed: i32, idx: i32) -> i32 {
    let mut z = idx.wrapping_add(seed).wrapping_sub(1_640_531_527_i32);
    z = (z ^ (z >> 16)).wrapping_mul(-2_048_173_461_i32);
    z = (z ^ (z >> 13)).wrapping_mul(-1_028_477_387_i32);
    z ^ (z >> 16)
}

#[inline]
pub const fn f32_to_f16_bits_rne(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 255 {
        return if mant != 0 {
            sign | 0x7E00
        } else {
            sign | 0x7BFF
        };
    }

    let new_exp = exp - 112;
    if new_exp >= 31 {
        return sign | 0x7BFF;
    }

    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let full_mant = mant | 0x0080_0000;
        let shift = (14 - new_exp) as u32;
        let half_bit = 1_u32 << (shift - 1);
        let lsb = (full_mant >> shift) & 1;
        let rounded = (full_mant + half_bit - 1 + lsb) >> shift;
        return sign | (rounded as u16);
    }

    let lsb = (mant >> 13) & 1;
    let rounded = mant + 0x0FFF + lsb;
    let f16_body = ((new_exp as u32) << 10) + (rounded >> 13);
    if f16_body >= 0x7C00 {
        sign | 0x7BFF
    } else {
        sign | (f16_body as u16)
    }
}

#[inline]
pub const fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let mut m = mant;
        let mut e = 113_u32;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        return f32::from_bits(sign | (e << 23) | (m << 13));
    }

    if exp == 31 {
        return f32::from_bits(sign | 0x7F80_0000 | (mant << 13));
    }

    let f32_exp = (exp + 112) << 23;
    f32::from_bits(sign | f32_exp | (mant << 13))
}

#[inline]
pub const fn stochastic_round_fp16_bits(val_f32: f32, seed: i32, offset: i32) -> u16 {
    let p_int = val_f32.to_bits() as i32;
    let abs_int = p_int & 0x7FFF_FFFF;
    let exp_f32 = (abs_int >> 23) & 0xFF;

    let raw_sub = 113 - exp_f32;
    let clamped_low = if raw_sub > 0 { raw_sub } else { 0 };
    let sub_shift = if clamped_low < 11 { clamped_low } else { 11 };
    let total_shift = 13 + sub_shift;

    let mask_bits = (1_i32 << total_shift) - 1;
    let noise = _prng(seed, offset) & mask_bits;
    let added_abs = if exp_f32 < 103 {
        0_i32
    } else {
        abs_int.wrapping_add(noise) & (!mask_bits)
    };
    let rounded_abs = if added_abs > 0x477F_E000 {
        0x477F_E000
    } else {
        added_abs
    };
    let sign_bit = p_int & i32::MIN;
    let out_int = sign_bit | rounded_abs;
    let pre_rounded_f32 = f32::from_bits(out_int as u32);
    f32_to_f16_bits_rne(pre_rounded_f32)
}

#[inline]
pub const fn stochastic_round_fp16(val_f32: f32, seed: i32, offset: i32) -> f32 {
    f16_bits_to_f32(stochastic_round_fp16_bits(val_f32, seed, offset))
}

#[inline]
pub const fn round_to_nearest_fp16(val_f32: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits_rne(val_f32))
}

#[inline]
pub const fn stochastic_round_bf16_bits(val_f32: f32, seed: i32, offset: i32) -> u16 {
    let p_int = val_f32.to_bits() as i32;
    let abs_int = p_int & 0x7FFF_FFFF;
    if abs_int > 0x7F80_0000 {
        return (((p_int as u32) | 0x0040_0000) >> 16) as u16;
    }
    let noise = _prng(seed, offset) & 0xFFFF;
    let added = abs_int.wrapping_add(noise) & -65536_i32;
    let rounded_abs = if added > 0x7F7F_0000 || added < 0 {
        0x7F7F_0000
    } else {
        added
    };
    let sign_bit = p_int & i32::MIN;
    let out_int = sign_bit | rounded_abs;
    ((out_int as u32) >> 16) as u16
}

#[inline]
pub const fn stochastic_round_bf16(val_f32: f32, seed: i32, offset: i32) -> f32 {
    let bits = stochastic_round_bf16_bits(val_f32, seed, offset);
    f32::from_bits((bits as u32) << 16)
}

#[inline]
pub const fn round_to_nearest_bf16(val_f32: f32) -> f32 {
    let bits = val_f32.to_bits();
    let abs_bits = bits & 0x7FFF_FFFF;
    if abs_bits > 0x7F80_0000 {
        return f32::from_bits((bits | 0x0040_0000) & 0xFFFF_0000);
    }
    let sign = bits & 0x8000_0000;
    let lsb = (abs_bits >> 16) & 1;
    let rounded_abs = abs_bits.wrapping_add(0x7FFF + lsb) & 0xFFFF_0000;
    let clamped_abs = if rounded_abs > 0x7F7F_0000 {
        0x7F7F_0000
    } else {
        rounded_abs
    };
    f32::from_bits(sign | clamped_abs)
}

#[inline]
pub const fn quantize_value(
    val_f32: f32,
    seed: i32,
    offset: i32,
    dtype_mode: DTypeMode,
    use_sr: bool,
) -> f32 {
    match (dtype_mode, use_sr) {
        (DTypeMode::Float32, _) => val_f32,
        (DTypeMode::BFloat16, true) => stochastic_round_bf16(val_f32, seed, offset),
        (DTypeMode::BFloat16, false) => round_to_nearest_bf16(val_f32),
        (DTypeMode::Float16, true) => stochastic_round_fp16(val_f32, seed, offset),
        (DTypeMode::Float16, false) => round_to_nearest_fp16(val_f32),
    }
}
