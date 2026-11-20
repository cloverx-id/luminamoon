#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    float32x4_t, vabsq_f32, vaddq_f32, vaddvq_u32, vbslq_f32, vcgtq_f32, vdivq_f32, vdupq_n_f32,
    vdupq_n_u32, vfmaq_f32, vfmsq_f32, vld1q_f32, vmaxq_f32, vminq_f32, vmulq_f32, vshrq_n_u32,
    vsqrtq_f32, vst1q_f32, vsubq_f32,
};

#[cfg(target_arch = "aarch64")]
use luminav_core::{
    BiasCorrectionCoeffs, BoundType, DTypeMode, LuminaError, LuminaVConfig, TensorViewMut,
    derive_variance_seed,
};

#[cfg(target_arch = "aarch64")]
use crate::SimdBackend;
#[cfg(target_arch = "aarch64")]
use crate::common::{ChunkMetrics, DualChunkCtx, DualParams};
#[cfg(target_arch = "aarch64")]
use crate::dispatch::dual::dual_step_sequential;
#[cfg(target_arch = "aarch64")]
use crate::scalar::dual::{dual_pass1_range, dual_update_range};

#[cfg(target_arch = "aarch64")]
use super::{
    apply_coordinate_bound_neon, apply_coordinate_bound_t2_neon, apply_fp16_safety_bound_neon,
    apply_fp16_safety_bound_t2_neon, clamp_diff_neon, offset_vector_neon, quantize_neon,
    reduce_add_ps_neon_quad, sanitize_gradient_neon, tanh_pade_neon,
};

#[cfg(target_arch = "aarch64")]
struct DualVecConstsNeon {
    beta1: float32x4_t,
    one_minus_beta1: float32x4_t,
    beta1_sq: float32x4_t,
    one_minus_beta1_sq: float32x4_t,
    beta2: float32x4_t,
    one_minus_beta2: float32x4_t,
    lam1: float32x4_t,
    c_sigma: float32x4_t,
    eps: float32x4_t,
    c1: float32x4_t,
    c2: float32x4_t,
    zero: float32x4_t,
    sigma_min: float32x4_t,
    vmax: float32x4_t,
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn build_dual_consts_neon(
    p: &DualParams,
    coeffs: &BiasCorrectionCoeffs,
) -> DualVecConstsNeon {
    DualVecConstsNeon {
        beta1: vdupq_n_f32(p.beta1),
        one_minus_beta1: vdupq_n_f32(p.one_minus_beta1),
        beta1_sq: vdupq_n_f32(p.beta1_sq),
        one_minus_beta1_sq: vdupq_n_f32(p.one_minus_beta1_sq),
        beta2: vdupq_n_f32(p.beta2),
        one_minus_beta2: vdupq_n_f32(p.one_minus_beta2),
        lam1: vdupq_n_f32(p.lam1),
        c_sigma: vdupq_n_f32(p.c_sigma),
        eps: vdupq_n_f32(p.effective_eps),
        c1: vdupq_n_f32(coeffs.c1),
        c2: vdupq_n_f32(coeffs.c2),
        zero: vdupq_n_f32(0.0_f32),
        sigma_min: vdupq_n_f32(1.0e-12_f32),
        vmax: vdupq_n_f32(65_000.0_f32),
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn compute_dual_u_neon(
    v_g: float32x4_t,
    v_m: float32x4_t,
    v_v: float32x4_t,
    c: &DualVecConstsNeon,
    is_t2: bool,
    clamp_v: bool,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let v_g_scaled = vmulq_f32(c.one_minus_beta1, v_g);
    let v_m_new = vfmaq_f32(v_g_scaled, c.beta1, v_m);
    let (v_u, v_v_new) = if is_t2 {
        let v_clm_m = vfmaq_f32(vmulq_f32(c.one_minus_beta1_sq, v_g), c.beta1_sq, v_m);
        let v_diff = clamp_diff_neon(vsubq_f32(v_g, v_m));
        let v_diff_sq = vmulq_f32(v_diff, v_diff);
        let mut v_v_new = vfmaq_f32(vmulq_f32(c.one_minus_beta2, v_diff_sq), c.beta2, v_v);
        if clamp_v {
            v_v_new = vminq_f32(v_v_new, c.vmax);
        }
        let v_env0 = vfmaq_f32(c.eps, c.beta1_sq, vabsq_f32(v_m));
        let v_env1 = vfmaq_f32(v_env0, c.one_minus_beta1_sq, vabsq_f32(v_g));
        let v_env2 = vfmaq_f32(v_env1, c.lam1, vabsq_f32(v_diff));
        let v_clm_env = vfmaq_f32(v_env2, c.c_sigma, vsqrtq_f32(vmaxq_f32(v_v_new, c.zero)));
        (vdivq_f32(v_clm_m, v_clm_env), v_v_new)
    } else {
        let v_clm_m = vfmaq_f32(v_g_scaled, c.beta1, v_m_new);
        let v_diff = clamp_diff_neon(vsubq_f32(v_g, v_m_new));
        let v_diff_sq = vmulq_f32(v_diff, v_diff);
        let mut v_v_new = vfmaq_f32(vmulq_f32(c.one_minus_beta2, v_diff_sq), c.beta2, v_v);
        if clamp_v {
            v_v_new = vminq_f32(v_v_new, c.vmax);
        }
        let v_sigma = vfmaq_f32(c.c2, vsqrtq_f32(vmaxq_f32(v_v_new, c.zero)), c.c1);
        (
            tanh_pade_neon(vdivq_f32(v_clm_m, vmaxq_f32(v_sigma, c.sigma_min))),
            v_v_new,
        )
    };
    (v_m_new, v_v_new, v_u)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dual_update_ctx_neon<const ACCUMULATE: bool, const CAUTIOUS_ALWAYS: bool>(
    param: &mut [f32],
    mut master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    exp_avg_sq: &mut [f32],
    ctx: &DualChunkCtx<'_>,
) -> ChunkMetrics {
    let n = param.len();
    let simd_len = n & !15;
    let has_master = ctx.config.master_weights.is_enabled() && master_param.is_some();
    let p = DualParams::new(ctx.config, ctx.coeffs, ctx.dtype_mode, ctx.state_dtype_mode);
    let c = build_dual_consts_neon(&p, ctx.coeffs);

    let v_cautious_scale = vdupq_n_f32(ctx.cautious_scale);
    let v_bound_scale = vdupq_n_f32(ctx.bound_scale);
    let v_bound_ratio = vdupq_n_f32(ctx.config.bound_ratio);
    let v_lr = vdupq_n_f32(ctx.config.lr);
    let v_lr_wd = vdupq_n_f32(p.lr_wd);
    let v_seed_m = vdupq_n_u32(ctx.seed_m as u32);
    let v_seed_p = vdupq_n_u32(ctx.seed_p as u32);
    let v_seed_v = vdupq_n_u32(derive_variance_seed(ctx.seed_m) as u32);
    let v_weyl_m = vdupq_n_u32(ctx.weyl_m);
    let v_weyl_p = vdupq_n_u32(ctx.weyl_p);
    let v_weyl_v = vdupq_n_u32(ctx.weyl_v);
    let cautious_enabled = ctx.config.cautious && (CAUTIOUS_ALWAYS || ctx.accumulate_metrics);

    let mut mask_count: u64 = 0;
    let mut v_u_sq = [vdupq_n_f32(0.0_f32); 4];
    let mut v_p_sq = [vdupq_n_f32(0.0_f32); 4];

    let mut i = 0;
    while i < simd_len {
        for q in 0..4 {
            let sub_i = i + q * 4;
            let v_offsets = offset_vector_neon((ctx.chunk_offset + sub_i) as i32);

            let v_p_init = match &master_param {
                Some(mp) if has_master => vld1q_f32(mp.as_ptr().add(sub_i)),
                _ => vld1q_f32(param.as_ptr().add(sub_i)),
            };
            let v_g = sanitize_gradient_neon(vld1q_f32(grad.as_ptr().add(sub_i)));
            let v_m = vld1q_f32(exp_avg.as_ptr().add(sub_i));
            let v_v = vld1q_f32(exp_avg_sq.as_ptr().add(sub_i));

            let (v_m_new, v_v_new, v_u) =
                compute_dual_u_neon(v_g, v_m, v_v, &c, p.is_t2, p.clamp_v);
            let m_pos = vcgtq_f32(vmulq_f32(v_u, v_g), c.zero);
            let v_u_masked = vbslq_f32(m_pos, v_u, c.zero);

            let v_m_stored = quantize_neon(
                v_m_new,
                v_seed_m,
                v_offsets,
                v_weyl_m,
                p.state_dtype_mode,
                ctx.config.stochastic_rounding,
            );
            vst1q_f32(exp_avg.as_mut_ptr().add(sub_i), v_m_stored);

            let v_v_stored = quantize_neon(
                v_v_new,
                v_seed_v,
                v_offsets,
                v_weyl_v,
                p.state_dtype_mode,
                ctx.config.stochastic_rounding,
            );
            vst1q_f32(exp_avg_sq.as_mut_ptr().add(sub_i), v_v_stored);

            let v_delta_theta = if cautious_enabled {
                vmulq_f32(v_u_masked, v_cautious_scale)
            } else {
                v_u
            };
            let mut v_step = vmulq_f32(v_lr, v_delta_theta);

            match p.bound_mode {
                BoundType::Radial => v_step = vmulq_f32(v_step, v_bound_scale),
                BoundType::Coordinate => {
                    v_step = if p.is_t2 {
                        apply_coordinate_bound_t2_neon(v_step, v_p_init, v_bound_ratio)
                    } else {
                        apply_coordinate_bound_neon(v_step, v_p_init, v_bound_ratio)
                    };
                }
                BoundType::None => {}
            }

            if ctx.dtype_mode == DTypeMode::Float16 {
                v_step = if p.is_t2 {
                    apply_fp16_safety_bound_t2_neon(v_step, v_p_init)
                } else {
                    apply_fp16_safety_bound_neon(v_step, v_p_init)
                };
            }

            let v_p_decayed = if ctx.config.weight_decay != 0.0_f32 {
                vfmsq_f32(v_p_init, v_lr_wd, v_p_init)
            } else {
                v_p_init
            };
            let v_p_updated = vsubq_f32(v_p_decayed, v_step);

            if let Some(mp) = &mut master_param
                && has_master
            {
                vst1q_f32(mp.as_mut_ptr().add(sub_i), v_p_updated);
                let v_p_q = quantize_neon(
                    v_p_updated,
                    v_seed_p,
                    v_offsets,
                    v_weyl_p,
                    ctx.dtype_mode,
                    false,
                );
                vst1q_f32(param.as_mut_ptr().add(sub_i), v_p_q);
            } else {
                let v_p_q = quantize_neon(
                    v_p_updated,
                    v_seed_p,
                    v_offsets,
                    v_weyl_p,
                    ctx.dtype_mode,
                    ctx.config.stochastic_rounding,
                );
                vst1q_f32(param.as_mut_ptr().add(sub_i), v_p_q);
            }

            if ACCUMULATE && ctx.accumulate_metrics {
                if ctx.config.cautious {
                    mask_count += u64::from(vaddvq_u32(vshrq_n_u32::<31>(m_pos)));
                }
                if p.bound_mode == BoundType::Radial {
                    let v_u_eff = if ctx.config.cautious { v_u_masked } else { v_u };
                    v_u_sq[q] = vaddq_f32(v_u_sq[q], vmulq_f32(v_u_eff, v_u_eff));
                    v_p_sq[q] = vaddq_f32(v_p_sq[q], vmulq_f32(v_p_init, v_p_init));
                }
            }
        }
        i += 16;
    }

    let simd_metrics = if ACCUMULATE && ctx.accumulate_metrics {
        let (u_sq_sum, p_sq_sum) = if p.bound_mode == BoundType::Radial {
            (
                reduce_add_ps_neon_quad(v_u_sq[0], v_u_sq[1], v_u_sq[2], v_u_sq[3]),
                reduce_add_ps_neon_quad(v_p_sq[0], v_p_sq[1], v_p_sq[2], v_p_sq[3]),
            )
        } else {
            (0.0_f32, 0.0_f32)
        };
        ChunkMetrics {
            mask_count,
            u_sq_sum,
            p_sq_sum,
        }
    } else {
        ChunkMetrics::default()
    };

    let tail = dual_update_range::<false, ACCUMULATE, CAUTIOUS_ALWAYS>(
        param,
        master_param,
        grad,
        exp_avg,
        exp_avg_sq,
        ctx,
        simd_len,
    );
    simd_metrics.combine(tail)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn dual_single_pass_ctx_neon(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    exp_avg_sq: &mut [f32],
    ctx: &DualChunkCtx<'_>,
) -> ChunkMetrics {
    dual_update_ctx_neon::<true, false>(param, master_param, grad, exp_avg, exp_avg_sq, ctx)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn dual_pass2_ctx_neon(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    exp_avg_sq: &mut [f32],
    ctx: &DualChunkCtx<'_>,
) {
    dual_update_ctx_neon::<false, true>(param, master_param, grad, exp_avg, exp_avg_sq, ctx);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dual_single_pass_chunk_neon(
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
    cautious_scale: f32,
    bound_scale: f32,
    seed_m: i32,
    seed_p: i32,
    accumulate_metrics: bool,
) -> ChunkMetrics {
    let ctx = DualChunkCtx {
        config,
        coeffs,
        dtype_mode,
        state_dtype_mode,
        chunk_offset,
        cautious_scale,
        bound_scale,
        seed_m,
        seed_p,
        weyl_m: 0,
        weyl_v: 0,
        weyl_p: 0,
        accumulate_metrics,
    };
    dual_single_pass_ctx_neon(param, master_param, grad, exp_avg, exp_avg_sq, &ctx)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dual_pass1_chunk_neon(
    param: &[f32],
    master_param: Option<&[f32]>,
    grad: &[f32],
    exp_avg: &[f32],
    exp_avg_sq: &[f32],
    config: &LuminaVConfig,
    coeffs: &BiasCorrectionCoeffs,
    dtype_mode: DTypeMode,
    state_dtype_mode: DTypeMode,
) -> ChunkMetrics {
    let n = param.len();
    let simd_len = n & !15;
    let has_master = config.master_weights.is_enabled() && master_param.is_some();
    let p = DualParams::new(config, coeffs, dtype_mode, state_dtype_mode);
    let c = build_dual_consts_neon(&p, coeffs);

    let mut mask_count: u64 = 0;
    let mut v_u_sq = [vdupq_n_f32(0.0_f32); 4];
    let mut v_p_sq = [vdupq_n_f32(0.0_f32); 4];

    let mut i = 0;
    while i < simd_len {
        for q in 0..4 {
            let sub_i = i + q * 4;
            let v_g = sanitize_gradient_neon(vld1q_f32(grad.as_ptr().add(sub_i)));
            let v_m = vld1q_f32(exp_avg.as_ptr().add(sub_i));
            let v_v = vld1q_f32(exp_avg_sq.as_ptr().add(sub_i));
            let (_, _, v_u) = compute_dual_u_neon(v_g, v_m, v_v, &c, p.is_t2, p.clamp_v);

            let m_pos = vcgtq_f32(vmulq_f32(v_u, v_g), c.zero);
            if config.cautious {
                mask_count += u64::from(vaddvq_u32(vshrq_n_u32::<31>(m_pos)));
            }
            if p.bound_mode == BoundType::Radial {
                let v_u_eff = if config.cautious {
                    vbslq_f32(m_pos, v_u, c.zero)
                } else {
                    v_u
                };
                let v_p_val = match master_param {
                    Some(mp) if has_master => vld1q_f32(mp.as_ptr().add(sub_i)),
                    _ => vld1q_f32(param.as_ptr().add(sub_i)),
                };
                v_u_sq[q] = vaddq_f32(v_u_sq[q], vmulq_f32(v_u_eff, v_u_eff));
                v_p_sq[q] = vaddq_f32(v_p_sq[q], vmulq_f32(v_p_val, v_p_val));
            }
        }
        i += 16;
    }

    let (u_sq_sum, p_sq_sum) = if p.bound_mode == BoundType::Radial {
        (
            reduce_add_ps_neon_quad(v_u_sq[0], v_u_sq[1], v_u_sq[2], v_u_sq[3]),
            reduce_add_ps_neon_quad(v_p_sq[0], v_p_sq[1], v_p_sq[2], v_p_sq[3]),
        )
    } else {
        (0.0_f32, 0.0_f32)
    };

    let tail = dual_pass1_range::<false>(
        param,
        master_param,
        grad,
        exp_avg,
        exp_avg_sq,
        config,
        coeffs,
        dtype_mode,
        state_dtype_mode,
        simd_len,
    );
    ChunkMetrics {
        mask_count,
        u_sq_sum,
        p_sq_sum,
    }
    .combine(tail)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dual_pass2_chunk_neon(
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
    cautious_scale: f32,
    bound_scale: f32,
    seed_m: i32,
    seed_p: i32,
) {
    let ctx = DualChunkCtx {
        config,
        coeffs,
        dtype_mode,
        state_dtype_mode,
        chunk_offset,
        cautious_scale,
        bound_scale,
        seed_m,
        seed_p,
        weyl_m: 0,
        weyl_v: 0,
        weyl_p: 0,
        accumulate_metrics: false,
    };
    dual_pass2_ctx_neon(param, master_param, grad, exp_avg, exp_avg_sq, &ctx);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dual_step_neon(
    view: &mut TensorViewMut<'_>,
    config: &LuminaVConfig,
    step: u64,
    param_index: usize,
) -> Result<(), LuminaError> {
    dual_step_sequential(view, config, step, param_index, SimdBackend::Neon)
}
