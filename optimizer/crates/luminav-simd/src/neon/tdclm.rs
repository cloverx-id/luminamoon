#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vabsq_f32, vaddq_f32, vaddvq_u32, vbslq_f32, vcgtq_f32, vdivq_f32, vdupq_n_f32, vdupq_n_u32,
    vfmaq_f32, vfmsq_f32, vld1q_f32, vmulq_f32, vshrq_n_u32, vst1q_f32, vsubq_f32,
};

#[cfg(target_arch = "aarch64")]
use luminav_core::{BoundType, DTypeMode, LuminaError, LuminaVConfig, TensorViewMut};

#[cfg(target_arch = "aarch64")]
use crate::SimdBackend;
#[cfg(target_arch = "aarch64")]
use crate::common::{ChunkMetrics, TdclmChunkCtx, TdclmParams};
#[cfg(target_arch = "aarch64")]
use crate::dispatch::tdclm::tdclm_step_sequential;
#[cfg(target_arch = "aarch64")]
use crate::scalar::tdclm::{tdclm_pass1_range, tdclm_update_range};

#[cfg(target_arch = "aarch64")]
use super::{
    apply_coordinate_bound_neon, apply_coordinate_bound_t2_neon, apply_fp16_safety_bound_neon,
    apply_fp16_safety_bound_t2_neon, clamp_diff_neon, offset_vector_neon, quantize_neon,
    reduce_add_ps_neon_quad, sanitize_gradient_neon,
};

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn tdclm_update_ctx_neon<const ACCUMULATE: bool>(
    param: &mut [f32],
    mut master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    ctx: &TdclmChunkCtx<'_>,
) -> ChunkMetrics {
    let n = param.len();
    let simd_len = n & !15;
    let has_master = ctx.config.master_weights.is_enabled() && master_param.is_some();
    let p = TdclmParams::new(ctx.config, ctx.state_dtype_mode);

    let v_beta1 = vdupq_n_f32(p.beta1);
    let v_one_minus_beta1 = vdupq_n_f32(p.one_minus_beta1);
    let v_beta1_sq = vdupq_n_f32(p.beta1_sq);
    let v_one_minus_beta1_sq = vdupq_n_f32(p.one_minus_beta1_sq);
    let v_lam = vdupq_n_f32(ctx.config.lam);
    let v_eps = vdupq_n_f32(p.effective_eps);
    let v_cautious_scale = vdupq_n_f32(ctx.cautious_scale);
    let v_bound_scale = vdupq_n_f32(ctx.bound_scale);
    let v_bound_ratio = vdupq_n_f32(ctx.config.bound_ratio);
    let v_lr = vdupq_n_f32(ctx.config.lr);
    let v_lr_wd = vdupq_n_f32(p.lr_wd);
    let v_zero = vdupq_n_f32(0.0_f32);
    let v_seed_m = vdupq_n_u32(ctx.seed_m as u32);
    let v_seed_p = vdupq_n_u32(ctx.seed_p as u32);
    let v_weyl_m = vdupq_n_u32(ctx.weyl_m);
    let v_weyl_p = vdupq_n_u32(ctx.weyl_p);

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

            let v_m_new = vfmaq_f32(vmulq_f32(v_one_minus_beta1, v_g), v_beta1, v_m);
            let v_clm_m = vfmaq_f32(vmulq_f32(v_one_minus_beta1_sq, v_g), v_beta1_sq, v_m);
            let v_diff = clamp_diff_neon(vsubq_f32(v_g, v_m));
            let v_env0 = vfmaq_f32(v_eps, v_beta1_sq, vabsq_f32(v_m));
            let v_env1 = vfmaq_f32(v_env0, v_one_minus_beta1_sq, vabsq_f32(v_g));
            let v_clm_env = vfmaq_f32(v_env1, v_lam, vabsq_f32(v_diff));
            let v_u = vdivq_f32(v_clm_m, v_clm_env);

            let m_pos = vcgtq_f32(vmulq_f32(v_u, v_g), v_zero);
            let v_u_masked = vbslq_f32(m_pos, v_u, v_zero);

            let v_m_stored = quantize_neon(
                v_m_new,
                v_seed_m,
                v_offsets,
                v_weyl_m,
                p.state_dtype_mode,
                ctx.config.stochastic_rounding,
            );
            vst1q_f32(exp_avg.as_mut_ptr().add(sub_i), v_m_stored);

            let v_delta_theta = if ctx.config.cautious {
                vmulq_f32(v_u_masked, v_cautious_scale)
            } else {
                v_u
            };
            let mut v_step = vmulq_f32(v_lr, v_delta_theta);

            match p.bound_mode {
                BoundType::Radial => v_step = vmulq_f32(v_step, v_bound_scale),
                BoundType::Coordinate => {
                    v_step = if p.is_t1 {
                        apply_coordinate_bound_t2_neon(v_step, v_p_init, v_bound_ratio)
                    } else {
                        apply_coordinate_bound_neon(v_step, v_p_init, v_bound_ratio)
                    };
                }
                BoundType::None => {}
            }

            if ctx.dtype_mode == DTypeMode::Float16 {
                v_step = if p.is_t1 {
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

            if ACCUMULATE {
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

    let simd_metrics = if ACCUMULATE {
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

    let tail =
        tdclm_update_range::<false, ACCUMULATE>(param, master_param, grad, exp_avg, ctx, simd_len);
    simd_metrics.combine(tail)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn tdclm_single_pass_ctx_neon(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    ctx: &TdclmChunkCtx<'_>,
) -> ChunkMetrics {
    tdclm_update_ctx_neon::<true>(param, master_param, grad, exp_avg, ctx)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn tdclm_pass2_ctx_neon(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    ctx: &TdclmChunkCtx<'_>,
) {
    tdclm_update_ctx_neon::<false>(param, master_param, grad, exp_avg, ctx);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn tdclm_single_pass_chunk_neon(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    config: &LuminaVConfig,
    dtype_mode: DTypeMode,
    state_dtype_mode: DTypeMode,
    chunk_offset: usize,
    cautious_scale: f32,
    bound_scale: f32,
    seed_m: i32,
    seed_p: i32,
) -> ChunkMetrics {
    let ctx = TdclmChunkCtx {
        config,
        dtype_mode,
        state_dtype_mode,
        chunk_offset,
        cautious_scale,
        bound_scale,
        seed_m,
        seed_p,
        weyl_m: 0,
        weyl_p: 0,
    };
    tdclm_single_pass_ctx_neon(param, master_param, grad, exp_avg, &ctx)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn tdclm_pass1_chunk_neon(
    param: &[f32],
    master_param: Option<&[f32]>,
    grad: &[f32],
    exp_avg: &[f32],
    config: &LuminaVConfig,
    dtype_mode: DTypeMode,
) -> ChunkMetrics {
    let _ = dtype_mode;
    let n = param.len();
    let simd_len = n & !15;
    let has_master = config.master_weights.is_enabled() && master_param.is_some();
    let p = TdclmParams::new(config, DTypeMode::Float32);

    let v_beta1_sq = vdupq_n_f32(p.beta1_sq);
    let v_one_minus_beta1_sq = vdupq_n_f32(p.one_minus_beta1_sq);
    let v_lam = vdupq_n_f32(config.lam);
    let v_eps = vdupq_n_f32(p.effective_eps);
    let v_zero = vdupq_n_f32(0.0_f32);

    let mut mask_count: u64 = 0;
    let mut v_u_sq = [vdupq_n_f32(0.0_f32); 4];
    let mut v_p_sq = [vdupq_n_f32(0.0_f32); 4];

    let mut i = 0;
    while i < simd_len {
        for q in 0..4 {
            let sub_i = i + q * 4;
            let v_g = sanitize_gradient_neon(vld1q_f32(grad.as_ptr().add(sub_i)));
            let v_m = vld1q_f32(exp_avg.as_ptr().add(sub_i));
            let v_diff = clamp_diff_neon(vsubq_f32(v_g, v_m));

            let v_clm_m = vfmaq_f32(vmulq_f32(v_one_minus_beta1_sq, v_g), v_beta1_sq, v_m);
            let v_env0 = vfmaq_f32(v_eps, v_beta1_sq, vabsq_f32(v_m));
            let v_env1 = vfmaq_f32(v_env0, v_one_minus_beta1_sq, vabsq_f32(v_g));
            let v_clm_env = vfmaq_f32(v_env1, v_lam, vabsq_f32(v_diff));
            let v_u = vdivq_f32(v_clm_m, v_clm_env);

            let m_pos = vcgtq_f32(vmulq_f32(v_u, v_g), v_zero);
            if config.cautious {
                mask_count += u64::from(vaddvq_u32(vshrq_n_u32::<31>(m_pos)));
            }
            if p.bound_mode == BoundType::Radial {
                let v_u_eff = if config.cautious {
                    vbslq_f32(m_pos, v_u, v_zero)
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

    let tail = tdclm_pass1_range::<false>(param, master_param, grad, exp_avg, config, simd_len);
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
pub unsafe fn tdclm_pass2_chunk_neon(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    config: &LuminaVConfig,
    dtype_mode: DTypeMode,
    state_dtype_mode: DTypeMode,
    chunk_offset: usize,
    cautious_scale: f32,
    bound_scale: f32,
    seed_m: i32,
    seed_p: i32,
) {
    let ctx = TdclmChunkCtx {
        config,
        dtype_mode,
        state_dtype_mode,
        chunk_offset,
        cautious_scale,
        bound_scale,
        seed_m,
        seed_p,
        weyl_m: 0,
        weyl_p: 0,
    };
    tdclm_pass2_ctx_neon(param, master_param, grad, exp_avg, &ctx);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn tdclm_step_neon(
    view: &mut TensorViewMut<'_>,
    config: &LuminaVConfig,
    step: u64,
    param_index: usize,
) -> Result<(), LuminaError> {
    tdclm_step_sequential(view, config, step, param_index, SimdBackend::Neon)
}
