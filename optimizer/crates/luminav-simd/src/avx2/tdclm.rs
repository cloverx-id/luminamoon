#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _CMP_GT_OQ, _mm256_add_ps, _mm256_and_ps, _mm256_cmp_ps, _mm256_div_ps, _mm256_fmadd_ps,
    _mm256_fnmadd_ps, _mm256_loadu_ps, _mm256_movemask_ps, _mm256_mul_ps, _mm256_set1_epi32,
    _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps,
};

#[cfg(target_arch = "x86_64")]
use luminav_core::{BoundType, DTypeMode, LuminaError, LuminaVConfig, TensorViewMut};

#[cfg(target_arch = "x86_64")]
use crate::SimdBackend;
#[cfg(target_arch = "x86_64")]
use crate::common::{ChunkMetrics, TdclmChunkCtx, TdclmParams};
#[cfg(target_arch = "x86_64")]
use crate::dispatch::tdclm::tdclm_step_sequential;
#[cfg(target_arch = "x86_64")]
use crate::scalar::tdclm::{tdclm_pass1_range, tdclm_update_range};

#[cfg(target_arch = "x86_64")]
use super::{
    abs_ps_avx2, apply_coordinate_bound_avx2, apply_coordinate_bound_t2_avx2,
    apply_fp16_safety_bound_avx2, apply_fp16_safety_bound_t2_avx2, clamp_diff_avx2,
    offset_vector_avx2, quantize_avx2, reduce_add_ps_avx2_pair, sanitize_gradient_avx2,
};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tdclm_update_ctx_avx2<const ACCUMULATE: bool>(
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

    let v_beta1 = _mm256_set1_ps(p.beta1);
    let v_one_minus_beta1 = _mm256_set1_ps(p.one_minus_beta1);
    let v_beta1_sq = _mm256_set1_ps(p.beta1_sq);
    let v_one_minus_beta1_sq = _mm256_set1_ps(p.one_minus_beta1_sq);
    let v_lam = _mm256_set1_ps(ctx.config.lam);
    let v_eps = _mm256_set1_ps(p.effective_eps);
    let v_cautious_scale = _mm256_set1_ps(ctx.cautious_scale);
    let v_bound_scale = _mm256_set1_ps(ctx.bound_scale);
    let v_bound_ratio = _mm256_set1_ps(ctx.config.bound_ratio);
    let v_lr = _mm256_set1_ps(ctx.config.lr);
    let v_lr_wd = _mm256_set1_ps(p.lr_wd);
    let v_zero = _mm256_setzero_ps();
    let v_seed_m = _mm256_set1_epi32(ctx.seed_m);
    let v_seed_p = _mm256_set1_epi32(ctx.seed_p);

    let mut mask_count: u64 = 0;
    let mut v_u_sq_lo = _mm256_setzero_ps();
    let mut v_u_sq_hi = _mm256_setzero_ps();
    let mut v_p_sq_lo = _mm256_setzero_ps();
    let mut v_p_sq_hi = _mm256_setzero_ps();

    let mut i = 0;
    while i < simd_len {
        for half in 0..2 {
            let sub_i = i + half * 8;
            let v_offsets = offset_vector_avx2((ctx.chunk_offset + sub_i) as i32);

            let v_p_init = match &master_param {
                Some(mp) if has_master => _mm256_loadu_ps(mp.as_ptr().add(sub_i)),
                _ => _mm256_loadu_ps(param.as_ptr().add(sub_i)),
            };
            let v_g = sanitize_gradient_avx2(_mm256_loadu_ps(grad.as_ptr().add(sub_i)));
            let v_m = _mm256_loadu_ps(exp_avg.as_ptr().add(sub_i));

            let v_m_new = _mm256_fmadd_ps(v_beta1, v_m, _mm256_mul_ps(v_one_minus_beta1, v_g));
            let v_clm_m =
                _mm256_fmadd_ps(v_beta1_sq, v_m, _mm256_mul_ps(v_one_minus_beta1_sq, v_g));
            let v_diff = clamp_diff_avx2(_mm256_sub_ps(v_g, v_m));
            let v_env0 = _mm256_fmadd_ps(v_beta1_sq, abs_ps_avx2(v_m), v_eps);
            let v_env1 = _mm256_fmadd_ps(v_one_minus_beta1_sq, abs_ps_avx2(v_g), v_env0);
            let v_clm_env = _mm256_fmadd_ps(v_lam, abs_ps_avx2(v_diff), v_env1);
            let v_u = _mm256_div_ps(v_clm_m, v_clm_env);

            let v_mask_ps = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_mul_ps(v_u, v_g), v_zero);
            let v_u_masked = _mm256_and_ps(v_mask_ps, v_u);

            let v_m_stored = quantize_avx2(
                v_m_new,
                v_seed_m,
                v_offsets,
                p.state_dtype_mode,
                ctx.config.stochastic_rounding,
            );
            _mm256_storeu_ps(exp_avg.as_mut_ptr().add(sub_i), v_m_stored);

            let v_delta_theta = if ctx.config.cautious {
                _mm256_mul_ps(v_u_masked, v_cautious_scale)
            } else {
                v_u
            };
            let mut v_step = _mm256_mul_ps(v_lr, v_delta_theta);

            match p.bound_mode {
                BoundType::Radial => v_step = _mm256_mul_ps(v_step, v_bound_scale),
                BoundType::Coordinate => {
                    v_step = if p.is_t1 {
                        apply_coordinate_bound_t2_avx2(v_step, v_p_init, v_bound_ratio)
                    } else {
                        apply_coordinate_bound_avx2(v_step, v_p_init, v_bound_ratio)
                    };
                }
                BoundType::None => {}
            }

            if ctx.dtype_mode == DTypeMode::Float16 {
                v_step = if p.is_t1 {
                    apply_fp16_safety_bound_t2_avx2(v_step, v_p_init)
                } else {
                    apply_fp16_safety_bound_avx2(v_step, v_p_init)
                };
            }

            let v_p_decayed = if ctx.config.weight_decay != 0.0_f32 {
                _mm256_fnmadd_ps(v_lr_wd, v_p_init, v_p_init)
            } else {
                v_p_init
            };
            let v_p_updated = _mm256_sub_ps(v_p_decayed, v_step);

            if let Some(mp) = &mut master_param
                && has_master
            {
                _mm256_storeu_ps(mp.as_mut_ptr().add(sub_i), v_p_updated);
                let v_p_q = quantize_avx2(v_p_updated, v_seed_p, v_offsets, ctx.dtype_mode, false);
                _mm256_storeu_ps(param.as_mut_ptr().add(sub_i), v_p_q);
            } else {
                let v_p_q = quantize_avx2(
                    v_p_updated,
                    v_seed_p,
                    v_offsets,
                    ctx.dtype_mode,
                    ctx.config.stochastic_rounding,
                );
                _mm256_storeu_ps(param.as_mut_ptr().add(sub_i), v_p_q);
            }

            if ACCUMULATE {
                if ctx.config.cautious {
                    mask_count += u64::from((_mm256_movemask_ps(v_mask_ps) as u32).count_ones());
                }
                if p.bound_mode == BoundType::Radial {
                    let v_u_eff = if ctx.config.cautious { v_u_masked } else { v_u };
                    let v_u2 = _mm256_mul_ps(v_u_eff, v_u_eff);
                    let v_p2 = _mm256_mul_ps(v_p_init, v_p_init);
                    if half == 0 {
                        v_u_sq_lo = _mm256_add_ps(v_u_sq_lo, v_u2);
                        v_p_sq_lo = _mm256_add_ps(v_p_sq_lo, v_p2);
                    } else {
                        v_u_sq_hi = _mm256_add_ps(v_u_sq_hi, v_u2);
                        v_p_sq_hi = _mm256_add_ps(v_p_sq_hi, v_p2);
                    }
                }
            }
        }
        i += 16;
    }

    let simd_metrics = if ACCUMULATE {
        let (u_sq_sum, p_sq_sum) = if p.bound_mode == BoundType::Radial {
            (
                reduce_add_ps_avx2_pair(v_u_sq_lo, v_u_sq_hi),
                reduce_add_ps_avx2_pair(v_p_sq_lo, v_p_sq_hi),
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn tdclm_single_pass_ctx_avx2(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    ctx: &TdclmChunkCtx<'_>,
) -> ChunkMetrics {
    tdclm_update_ctx_avx2::<true>(param, master_param, grad, exp_avg, ctx)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn tdclm_pass2_ctx_avx2(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    ctx: &TdclmChunkCtx<'_>,
) {
    tdclm_update_ctx_avx2::<false>(param, master_param, grad, exp_avg, ctx);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn tdclm_single_pass_chunk_avx2(
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
    };
    tdclm_single_pass_ctx_avx2(param, master_param, grad, exp_avg, &ctx)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn tdclm_pass1_chunk_avx2(
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

    let v_beta1_sq = _mm256_set1_ps(p.beta1_sq);
    let v_one_minus_beta1_sq = _mm256_set1_ps(p.one_minus_beta1_sq);
    let v_lam = _mm256_set1_ps(config.lam);
    let v_eps = _mm256_set1_ps(p.effective_eps);
    let v_zero = _mm256_setzero_ps();

    let mut mask_count: u64 = 0;
    let mut v_u_sq_lo = _mm256_setzero_ps();
    let mut v_u_sq_hi = _mm256_setzero_ps();
    let mut v_p_sq_lo = _mm256_setzero_ps();
    let mut v_p_sq_hi = _mm256_setzero_ps();

    let mut i = 0;
    while i < simd_len {
        for half in 0..2 {
            let sub_i = i + half * 8;
            let v_g = sanitize_gradient_avx2(_mm256_loadu_ps(grad.as_ptr().add(sub_i)));
            let v_m = _mm256_loadu_ps(exp_avg.as_ptr().add(sub_i));
            let v_diff = clamp_diff_avx2(_mm256_sub_ps(v_g, v_m));

            let v_clm_m =
                _mm256_fmadd_ps(v_beta1_sq, v_m, _mm256_mul_ps(v_one_minus_beta1_sq, v_g));
            let v_env0 = _mm256_fmadd_ps(v_beta1_sq, abs_ps_avx2(v_m), v_eps);
            let v_env1 = _mm256_fmadd_ps(v_one_minus_beta1_sq, abs_ps_avx2(v_g), v_env0);
            let v_clm_env = _mm256_fmadd_ps(v_lam, abs_ps_avx2(v_diff), v_env1);
            let v_u = _mm256_div_ps(v_clm_m, v_clm_env);

            let v_mask_ps = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_mul_ps(v_u, v_g), v_zero);
            if config.cautious {
                mask_count += u64::from((_mm256_movemask_ps(v_mask_ps) as u32).count_ones());
            }
            if p.bound_mode == BoundType::Radial {
                let v_u_eff = if config.cautious {
                    _mm256_and_ps(v_mask_ps, v_u)
                } else {
                    v_u
                };
                let v_p_val = match master_param {
                    Some(mp) if has_master => _mm256_loadu_ps(mp.as_ptr().add(sub_i)),
                    _ => _mm256_loadu_ps(param.as_ptr().add(sub_i)),
                };
                let v_u2 = _mm256_mul_ps(v_u_eff, v_u_eff);
                let v_p2 = _mm256_mul_ps(v_p_val, v_p_val);
                if half == 0 {
                    v_u_sq_lo = _mm256_add_ps(v_u_sq_lo, v_u2);
                    v_p_sq_lo = _mm256_add_ps(v_p_sq_lo, v_p2);
                } else {
                    v_u_sq_hi = _mm256_add_ps(v_u_sq_hi, v_u2);
                    v_p_sq_hi = _mm256_add_ps(v_p_sq_hi, v_p2);
                }
            }
        }
        i += 16;
    }

    let (u_sq_sum, p_sq_sum) = if p.bound_mode == BoundType::Radial {
        (
            reduce_add_ps_avx2_pair(v_u_sq_lo, v_u_sq_hi),
            reduce_add_ps_avx2_pair(v_p_sq_lo, v_p_sq_hi),
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn tdclm_pass2_chunk_avx2(
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
    };
    tdclm_pass2_ctx_avx2(param, master_param, grad, exp_avg, &ctx);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn tdclm_step_avx2(
    view: &mut TensorViewMut<'_>,
    config: &LuminaVConfig,
    step: u64,
    param_index: usize,
) -> Result<(), LuminaError> {
    tdclm_step_sequential(view, config, step, param_index, SimdBackend::Avx2)
}
