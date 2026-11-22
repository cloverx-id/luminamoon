#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m512, __mmask16, _CMP_GT_OQ, _mm512_abs_ps, _mm512_add_ps, _mm512_cmp_ps_mask, _mm512_div_ps,
    _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_maskz_mov_ps, _mm512_max_ps, _mm512_mul_ps,
    _mm512_set1_epi32, _mm512_set1_ps, _mm512_setzero_ps, _mm512_sqrt_ps, _mm512_storeu_ps,
    _mm512_sub_ps,
};

#[cfg(target_arch = "x86_64")]
use luminav_core::{
    BiasCorrectionCoeffs, BoundType, DTypeMode, LuminaError, LuminaVConfig, TensorViewMut,
    derive_variance_seed,
};

#[cfg(target_arch = "x86_64")]
use crate::SimdBackend;
#[cfg(target_arch = "x86_64")]
use crate::common::{ChunkMetrics, DualChunkCtx, DualParams};
#[cfg(target_arch = "x86_64")]
use crate::dispatch::dual::dual_step_sequential;
#[cfg(target_arch = "x86_64")]
use crate::scalar::dual::{dual_pass1_range, dual_update_range};

#[cfg(target_arch = "x86_64")]
use super::{
    apply_coordinate_bound_avx512, apply_coordinate_bound_t2_avx512,
    apply_decoupled_weight_decay_avx512, apply_fp16_safety_bound_avx512,
    apply_fp16_safety_bound_t2_avx512, apply_rational_bound_avx512, clamp_diff_avx512,
    offset_vector_avx512, quantize_avx512, reduce_add_ps_avx512, sanitize_gradient_avx512,
    tanh_pade_avx512,
};

#[cfg(target_arch = "x86_64")]
struct DualVecConstsAvx512 {
    beta1: __m512,
    one_minus_beta1: __m512,
    beta1_sq: __m512,
    one_minus_beta1_sq: __m512,
    beta2: __m512,
    one_minus_beta2: __m512,
    lam1: __m512,
    c_sigma: __m512,
    eps: __m512,
    c1: __m512,
    c2: __m512,
    zero: __m512,
    sigma_min: __m512,
    inv_sqrt_bc2_prev: __m512,
    variance_temper: __m512,
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[inline]
unsafe fn build_dual_consts_avx512(
    p: &DualParams,
    coeffs: &BiasCorrectionCoeffs,
) -> DualVecConstsAvx512 {
    DualVecConstsAvx512 {
        beta1: _mm512_set1_ps(p.beta1),
        one_minus_beta1: _mm512_set1_ps(p.one_minus_beta1),
        beta1_sq: _mm512_set1_ps(p.beta1_sq),
        one_minus_beta1_sq: _mm512_set1_ps(p.one_minus_beta1_sq),
        beta2: _mm512_set1_ps(p.beta2),
        one_minus_beta2: _mm512_set1_ps(p.one_minus_beta2),
        lam1: _mm512_set1_ps(p.lam1),
        c_sigma: _mm512_set1_ps(p.c_sigma),
        eps: _mm512_set1_ps(p.effective_eps),
        c1: _mm512_set1_ps(coeffs.c1),
        c2: _mm512_set1_ps(coeffs.c2),
        zero: _mm512_setzero_ps(),
        sigma_min: _mm512_set1_ps(1.0e-12_f32),
        inv_sqrt_bc2_prev: _mm512_set1_ps(p.inv_sqrt_bc2_prev),
        variance_temper: _mm512_set1_ps(p.variance_temper),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[inline]
unsafe fn compute_dual_u_avx512(
    v_g: __m512,
    v_m: __m512,
    v_v: __m512,
    c: &DualVecConstsAvx512,
    is_t2: bool,
    use_rtv: bool,
) -> (__m512, __m512, __m512) {
    let v_g_scaled = _mm512_mul_ps(c.one_minus_beta1, v_g);
    let v_m_new = _mm512_fmadd_ps(c.beta1, v_m, v_g_scaled);
    let (v_u, v_v_new) = if is_t2 {
        let v_clm_m = _mm512_fmadd_ps(c.beta1_sq, v_m, _mm512_mul_ps(c.one_minus_beta1_sq, v_g));
        let v_diff = clamp_diff_avx512(_mm512_sub_ps(v_g, v_m));
        let v_abs_d = _mm512_abs_ps(v_diff);
        let v_d_mem = if use_rtv {
            let v_sigma_prev = _mm512_mul_ps(
                _mm512_sqrt_ps(_mm512_max_ps(v_v, c.zero)),
                c.inv_sqrt_bc2_prev,
            );
            let v_ref = _mm512_max_ps(v_sigma_prev, _mm512_abs_ps(v_m));
            apply_rational_bound_avx512(v_abs_d, _mm512_fmadd_ps(c.variance_temper, v_ref, c.eps))
        } else {
            v_abs_d
        };
        let v_diff_sq = _mm512_mul_ps(v_d_mem, v_d_mem);
        let v_v_new =
            _mm512_fmadd_ps(c.beta2, v_v, _mm512_mul_ps(c.one_minus_beta2, v_diff_sq));
        let v_env0 = _mm512_fmadd_ps(c.beta1_sq, _mm512_abs_ps(v_m), c.eps);
        let v_env1 = _mm512_fmadd_ps(c.one_minus_beta1_sq, _mm512_abs_ps(v_g), v_env0);
        let v_env2 = _mm512_fmadd_ps(c.lam1, v_abs_d, v_env1);
        let v_clm_env = _mm512_fmadd_ps(
            c.c_sigma,
            _mm512_sqrt_ps(_mm512_max_ps(v_v_new, c.zero)),
            v_env2,
        );
        (_mm512_div_ps(v_clm_m, v_clm_env), v_v_new)
    } else {
        let v_clm_m = _mm512_fmadd_ps(c.beta1, v_m_new, v_g_scaled);
        let v_diff = clamp_diff_avx512(_mm512_sub_ps(v_g, v_m_new));
        let v_abs_d = _mm512_abs_ps(v_diff);
        let v_d_mem = if use_rtv {
            let v_sigma_prev = _mm512_mul_ps(
                _mm512_sqrt_ps(_mm512_max_ps(v_v, c.zero)),
                c.inv_sqrt_bc2_prev,
            );
            let v_ref = _mm512_max_ps(v_sigma_prev, _mm512_abs_ps(v_m));
            apply_rational_bound_avx512(v_abs_d, _mm512_fmadd_ps(c.variance_temper, v_ref, c.eps))
        } else {
            v_abs_d
        };
        let v_diff_sq = _mm512_mul_ps(v_d_mem, v_d_mem);
        let v_v_new =
            _mm512_fmadd_ps(c.beta2, v_v, _mm512_mul_ps(c.one_minus_beta2, v_diff_sq));
        let v_sigma = _mm512_fmadd_ps(_mm512_sqrt_ps(_mm512_max_ps(v_v_new, c.zero)), c.c1, c.c2);
        (
            tanh_pade_avx512(_mm512_div_ps(v_clm_m, _mm512_max_ps(v_sigma, c.sigma_min))),
            v_v_new,
        )
    };
    (v_m_new, v_v_new, v_u)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
unsafe fn dual_update_ctx_avx512<const ACCUMULATE: bool, const CAUTIOUS_ALWAYS: bool>(
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
    let c = build_dual_consts_avx512(&p, ctx.coeffs);

    let v_cautious_scale = _mm512_set1_ps(ctx.cautious_scale);
    let v_bound_scale = _mm512_set1_ps(ctx.bound_scale);
    let v_bound_ratio = _mm512_set1_ps(ctx.config.bound_ratio);
    let v_bound_floor = _mm512_set1_ps(ctx.config.bound_floor);
    let v_lr = _mm512_set1_ps(ctx.config.lr);
    let v_lr_wd = _mm512_set1_ps(p.lr_wd);
    let v_seed_m = _mm512_set1_epi32(ctx.seed_m);
    let v_seed_p = _mm512_set1_epi32(ctx.seed_p);
    let v_seed_v = _mm512_set1_epi32(derive_variance_seed(ctx.seed_m));
    let v_weyl_m = _mm512_set1_epi32(ctx.weyl_m as i32);
    let v_weyl_p = _mm512_set1_epi32(ctx.weyl_p as i32);
    let v_weyl_v = _mm512_set1_epi32(ctx.weyl_v as i32);
    let cautious_enabled = ctx.config.cautious && (CAUTIOUS_ALWAYS || ctx.accumulate_metrics);

    let mut mask_count: u64 = 0;
    let mut v_u_sq_acc = _mm512_setzero_ps();
    let mut v_p_sq_acc = _mm512_setzero_ps();

    let mut i = 0;
    while i < simd_len {
        let v_offsets = offset_vector_avx512((ctx.chunk_offset + i) as i32);

        let v_p_init = match &master_param {
            Some(mp) if has_master => _mm512_loadu_ps(mp.as_ptr().add(i)),
            _ => _mm512_loadu_ps(param.as_ptr().add(i)),
        };
        let v_g = sanitize_gradient_avx512(_mm512_loadu_ps(grad.as_ptr().add(i)));
        let v_m = _mm512_loadu_ps(exp_avg.as_ptr().add(i));
        let v_v = _mm512_loadu_ps(exp_avg_sq.as_ptr().add(i));

        let (v_m_new, v_v_new, v_u) = compute_dual_u_avx512(v_g, v_m, v_v, &c, p.is_t2, p.use_rtv);
        let k_mask: __mmask16 = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(_mm512_mul_ps(v_u, v_g), c.zero);
        let v_u_masked = _mm512_maskz_mov_ps(k_mask, v_u);

        let v_m_stored = quantize_avx512(
            v_m_new,
            v_seed_m,
            v_offsets,
            v_weyl_m,
            p.state_dtype_mode,
            ctx.config.stochastic_rounding,
        );
        _mm512_storeu_ps(exp_avg.as_mut_ptr().add(i), v_m_stored);

        let v_v_stored = quantize_avx512(
            v_v_new,
            v_seed_v,
            v_offsets,
            v_weyl_v,
            p.state_dtype_mode,
            ctx.config.stochastic_rounding,
        );
        _mm512_storeu_ps(exp_avg_sq.as_mut_ptr().add(i), v_v_stored);

        let v_delta_theta = if cautious_enabled {
            _mm512_mul_ps(v_u_masked, v_cautious_scale)
        } else {
            v_u
        };
        let mut v_step = _mm512_mul_ps(v_lr, v_delta_theta);

        let is_fp16 = ctx.dtype_mode == DTypeMode::Float16;
        match p.bound_mode {
            BoundType::Radial => v_step = _mm512_mul_ps(v_step, v_bound_scale),
            BoundType::Coordinate => {
                v_step = if p.is_t2 {
                    apply_coordinate_bound_t2_avx512(
                        v_step,
                        v_p_init,
                        v_bound_ratio,
                        v_bound_floor,
                        is_fp16,
                    )
                } else {
                    apply_coordinate_bound_avx512(
                        v_step,
                        v_p_init,
                        v_bound_ratio,
                        v_bound_floor,
                        is_fp16,
                    )
                };
            }
            BoundType::None => {}
        }

        if is_fp16 && p.bound_mode != BoundType::Coordinate {
            v_step = if p.is_t2 {
                apply_fp16_safety_bound_t2_avx512(v_step, v_p_init)
            } else {
                apply_fp16_safety_bound_avx512(v_step, v_p_init)
            };
        }

        let v_p_decayed = if ctx.config.weight_decay != 0.0_f32 {
            apply_decoupled_weight_decay_avx512(
                v_p_init,
                v_step,
                v_lr_wd,
                ctx.config.cautious_weight_decay,
            )
        } else {
            v_p_init
        };
        let v_p_updated = _mm512_sub_ps(v_p_decayed, v_step);

        if let Some(mp) = &mut master_param
            && has_master
        {
            _mm512_storeu_ps(mp.as_mut_ptr().add(i), v_p_updated);
            let v_p_q = quantize_avx512(
                v_p_updated,
                v_seed_p,
                v_offsets,
                v_weyl_p,
                ctx.dtype_mode,
                false,
            );
            _mm512_storeu_ps(param.as_mut_ptr().add(i), v_p_q);
        } else {
            let v_p_q = quantize_avx512(
                v_p_updated,
                v_seed_p,
                v_offsets,
                v_weyl_p,
                ctx.dtype_mode,
                ctx.config.stochastic_rounding,
            );
            _mm512_storeu_ps(param.as_mut_ptr().add(i), v_p_q);
        }

        if ACCUMULATE && ctx.accumulate_metrics {
            if ctx.config.cautious {
                mask_count += u64::from(k_mask.count_ones());
            }
            if p.bound_mode == BoundType::Radial {
                let v_u_eff = if ctx.config.cautious { v_u_masked } else { v_u };
                v_u_sq_acc = _mm512_add_ps(v_u_sq_acc, _mm512_mul_ps(v_u_eff, v_u_eff));
                v_p_sq_acc = _mm512_add_ps(v_p_sq_acc, _mm512_mul_ps(v_p_init, v_p_init));
            }
        }
        i += 16;
    }

    let simd_metrics = if ACCUMULATE && ctx.accumulate_metrics {
        let (u_sq_sum, p_sq_sum) = if p.bound_mode == BoundType::Radial {
            (
                reduce_add_ps_avx512(v_u_sq_acc),
                reduce_add_ps_avx512(v_p_sq_acc),
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
pub(crate) unsafe fn dual_single_pass_ctx_avx512(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    exp_avg_sq: &mut [f32],
    ctx: &DualChunkCtx<'_>,
) -> ChunkMetrics {
    dual_update_ctx_avx512::<true, false>(param, master_param, grad, exp_avg, exp_avg_sq, ctx)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
pub(crate) unsafe fn dual_pass2_ctx_avx512(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    exp_avg_sq: &mut [f32],
    ctx: &DualChunkCtx<'_>,
) {
    dual_update_ctx_avx512::<false, true>(param, master_param, grad, exp_avg, exp_avg_sq, ctx);
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
    dual_single_pass_ctx_avx512(param, master_param, grad, exp_avg, exp_avg_sq, &ctx)
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
) -> ChunkMetrics {
    let n = param.len();
    let simd_len = n & !15;
    let has_master = config.master_weights.is_enabled() && master_param.is_some();
    let p = DualParams::new(config, coeffs, dtype_mode, state_dtype_mode);
    let c = build_dual_consts_avx512(&p, coeffs);

    let mut mask_count: u64 = 0;
    let mut v_u_sq_acc = _mm512_setzero_ps();
    let mut v_p_sq_acc = _mm512_setzero_ps();

    let mut i = 0;
    while i < simd_len {
        let v_g = sanitize_gradient_avx512(_mm512_loadu_ps(grad.as_ptr().add(i)));
        let v_m = _mm512_loadu_ps(exp_avg.as_ptr().add(i));
        let v_v = _mm512_loadu_ps(exp_avg_sq.as_ptr().add(i));
        let (_, _, v_u) = compute_dual_u_avx512(v_g, v_m, v_v, &c, p.is_t2, p.use_rtv);

        let k_mask: __mmask16 = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(_mm512_mul_ps(v_u, v_g), c.zero);
        if config.cautious {
            mask_count += u64::from(k_mask.count_ones());
        }
        if p.bound_mode == BoundType::Radial {
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

    let (u_sq_sum, p_sq_sum) = if p.bound_mode == BoundType::Radial {
        (
            reduce_add_ps_avx512(v_u_sq_acc),
            reduce_add_ps_avx512(v_p_sq_acc),
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
    dual_pass2_ctx_avx512(param, master_param, grad, exp_avg, exp_avg_sq, &ctx);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
pub unsafe fn dual_step_avx512(
    view: &mut TensorViewMut<'_>,
    config: &LuminaVConfig,
    step: u64,
    param_index: usize,
) -> Result<(), LuminaError> {
    dual_step_sequential(view, config, step, param_index, SimdBackend::Avx512)
}
