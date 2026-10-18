use crate::config::{BoundType, DTypeMode, LuminaVConfig};
use crate::state::ScratchBuffer;

#[inline]
pub fn sanitize_gradient(g: f32) -> f32 {
    if g.is_finite() && g.abs() < 1.0e38_f32 {
        g.clamp(-1.0e18_f32, 1.0e18_f32)
    } else {
        0.0_f32
    }
}

#[inline]
pub fn clamp_defect(d: f32) -> f32 {
    d.clamp(-1.0e18_f32, 1.0e18_f32)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiasCorrectionCoeffs {
    pub bc1_nes: f32,
    pub bc2: f32,
    pub c1: f32,
    pub c2: f32,
    pub effective_eps: f32,
}

#[inline]
pub fn compute_bias_correction(
    step: u64,
    beta1: f32,
    beta2: f32,
    eps: f32,
    tau: f32,
    dtype_mode: DTypeMode,
) -> BiasCorrectionCoeffs {
    let safe_step = step.max(1);
    let effective_eps = if dtype_mode == DTypeMode::Float16 {
        eps.max(1.0e-4_f32)
    } else {
        eps
    };

    let beta1_f64 = f64::from(beta1);
    let beta2_f64 = f64::from(beta2);
    let tau_f64 = f64::from(tau);
    let eps_f64 = f64::from(effective_eps);

    let bc1_nes_f64 = if safe_step < (i32::MAX as u64) {
        1.0_f64 - beta1_f64.powi((safe_step + 1) as i32)
    } else {
        1.0_f64 - beta1_f64.powf((safe_step + 1) as f64)
    };

    let bc2_raw = if safe_step <= (i32::MAX as u64) {
        1.0_f64 - beta2_f64.powi(safe_step as i32)
    } else {
        1.0_f64 - beta2_f64.powf(safe_step as f64)
    };
    let bc2_f64 = bc2_raw.max(1.0e-15_f64);

    let c1 = ((bc1_nes_f64 * tau_f64) / bc2_f64.sqrt()) as f32;
    let c2 = (eps_f64 * bc1_nes_f64 * tau_f64) as f32;

    BiasCorrectionCoeffs {
        bc1_nes: bc1_nes_f64 as f32,
        bc2: bc2_f64 as f32,
        c1,
        c2,
        effective_eps,
    }
}

#[inline]
pub fn tanh_pade_f32(x_in: f32) -> f32 {
    if x_in.is_nan() {
        return f32::NAN;
    }
    const CLAMP_HI: f32 = 7.998_811_7_f32;
    const CLAMP_LO: f32 = -7.998_811_7_f32;

    const ALPHA_13: f32 = -2.760_768_4e-16_f32;
    const ALPHA_11: f32 = 2.000_188e-13_f32;
    const ALPHA_9: f32 = -8.604_672e-11_f32;
    const ALPHA_7: f32 = 5.122_297e-8_f32;
    const ALPHA_5: f32 = 1.485_722_4e-5_f32;
    const ALPHA_3: f32 = 6.372_619_3e-4_f32;
    const ALPHA_1: f32 = 4.893_524_6e-3_f32;

    const BETA_6: f32 = 1.198_258_4e-6_f32;
    const BETA_4: f32 = 1.185_347_1e-4_f32;
    const BETA_2: f32 = 2.268_434_6e-3_f32;
    const BETA_0: f32 = 4.893_524_6e-3_f32;

    let x = x_in.clamp(CLAMP_LO, CLAMP_HI);
    let x2 = x * x;

    let mut p = ALPHA_13.mul_add(x2, ALPHA_11);
    let mut q = BETA_6.mul_add(x2, BETA_4);
    p = p.mul_add(x2, ALPHA_9);
    q = q.mul_add(x2, BETA_2);
    p = p.mul_add(x2, ALPHA_7);
    q = q.mul_add(x2, BETA_0);
    p = p.mul_add(x2, ALPHA_5);
    p = p.mul_add(x2, ALPHA_3);
    p = p.mul_add(x2, ALPHA_1);

    (x * p) / q
}

#[inline]
pub fn compute_cautious_scale(mask_sum: f32, total_elements: f32, clamp_min: f32) -> f32 {
    let denom = total_elements.max(1.0_f32);
    let m_bar = (mask_sum / denom).clamp(clamp_min, 1.0_f32);
    1.0_f32 / m_bar
}

#[inline]
pub fn compute_radial_bound_scale(
    u_sq_sum: f32,
    p_sq_sum: f32,
    cautious_scale: f32,
    lr: f32,
    bound_ratio: f32,
) -> f32 {
    let norm_p = p_sq_sum.max(0.0_f32).sqrt();
    let norm_delta = u_sq_sum.max(0.0_f32).sqrt() * cautious_scale;
    let radius = bound_ratio * norm_p.max(1.0_f32);
    let r = (lr * norm_delta) / radius;
    if r < 1.0e-4_f32 {
        1.0_f32
    } else {
        tanh_pade_f32(r) / r.max(1.0e-8_f32)
    }
}

#[inline]
pub fn apply_coordinate_bound(step_val: f32, p_init: f32, bound_ratio: f32) -> f32 {
    let b_i = bound_ratio * p_init.mul_add(p_init, 1.0e-4_f32).sqrt();
    b_i * tanh_pade_f32(step_val / b_i)
}

#[inline]
pub fn apply_fp16_safety_bound(step_val: f32, p_init: f32) -> f32 {
    let b_safe = (p_init.abs() * 0.01_f32).max(1.5e-4_f32);
    b_safe * tanh_pade_f32(step_val / b_safe)
}

#[inline]
pub fn update_post_step_ema(
    scratch: &mut ScratchBuffer,
    total_elements: f32,
    config: &LuminaVConfig,
) {
    let bound_mode = config.effective_bound_type();
    if config.cautious {
        let target_cautious = compute_cautious_scale(
            scratch.mask_sum(),
            total_elements,
            config.cautious_clamp_min,
        );
        let old_cautious = scratch.cautious_scale();
        let new_cautious = config
            .ema_decay
            .mul_add(old_cautious, (1.0_f32 - config.ema_decay) * target_cautious);
        scratch.set_cautious_scale(new_cautious);
    }

    if bound_mode == BoundType::Radial {
        let c_scale = if config.cautious {
            scratch.cautious_scale()
        } else {
            1.0_f32
        };
        let target_bound = compute_radial_bound_scale(
            scratch.u_sq_sum(),
            scratch.p_sq_sum(),
            c_scale,
            config.lr,
            config.bound_ratio,
        );
        let old_bound = scratch.bound_scale();
        let new_bound = config
            .ema_decay
            .mul_add(old_bound, (1.0_f32 - config.ema_decay) * target_bound);
        scratch.set_bound_scale(new_bound);
    }

    scratch.reset_accumulators();
}

#[inline]
pub fn reduce_sum_16(lanes: &[f32; 16]) -> f32 {
    let s8_0 = lanes[0] + lanes[8];
    let s8_1 = lanes[1] + lanes[9];
    let s8_2 = lanes[2] + lanes[10];
    let s8_3 = lanes[3] + lanes[11];
    let s8_4 = lanes[4] + lanes[12];
    let s8_5 = lanes[5] + lanes[13];
    let s8_6 = lanes[6] + lanes[14];
    let s8_7 = lanes[7] + lanes[15];

    let s4_0 = s8_0 + s8_4;
    let s4_1 = s8_1 + s8_5;
    let s4_2 = s8_2 + s8_6;
    let s4_3 = s8_3 + s8_7;

    let s2_0 = s4_0 + s4_2;
    let s2_1 = s4_1 + s4_3;

    s2_0 + s2_1
}
