use luminav_core::{
    BiasCorrectionCoeffs, BoundType, DTypeMode, LuminaVConfig, derive_variance_seed,
    quantize_value, reduce_sum_16, sanitize_gradient, tanh_pade_f32,
};

use crate::common::{ChunkMetrics, DualChunkCtx, DualParams};
use crate::scalar::finalize_param_step;

#[inline(always)]
fn dual_compute_u_v(
    g: f32,
    m: f32,
    v: f32,
    p: &DualParams,
    coeffs: &BiasCorrectionCoeffs,
) -> (f32, f32, f32) {
    let m_new = p.beta1.mul_add(m, p.one_minus_beta1 * g);
    let (clm_m, diff) = if p.is_t2 {
        (
            p.beta1_sq.mul_add(m, p.one_minus_beta1_sq * g),
            (g - m).clamp(-1.0e18_f32, 1.0e18_f32),
        )
    } else {
        (
            p.beta1.mul_add(m_new, p.one_minus_beta1 * g),
            (g - m_new).clamp(-1.0e18_f32, 1.0e18_f32),
        )
    };

    let mut v_new = p.beta2.mul_add(v, p.one_minus_beta2 * (diff * diff));
    if p.clamp_v {
        v_new = v_new.min(65_000.0_f32);
    }

    let u = if p.is_t2 {
        let env0 = p.beta1_sq.mul_add(m.abs(), p.effective_eps);
        let env1 = p.one_minus_beta1_sq.mul_add(g.abs(), env0);
        let env2 = p.lam1.mul_add(diff.abs(), env1);
        let clm_env = p.c_sigma.mul_add(v_new.max(0.0_f32).sqrt(), env2);
        clm_m / clm_env
    } else {
        let sigma = v_new.max(0.0_f32).sqrt().mul_add(coeffs.c1, coeffs.c2);
        tanh_pade_f32(clm_m / sigma.max(1.0e-12_f32))
    };

    (m_new, v_new, u)
}

#[inline]
pub(crate) fn dual_update_range<
    const USE_16_LANE: bool,
    const ACCUMULATE: bool,
    const CAUTIOUS_ALWAYS: bool,
>(
    param: &mut [f32],
    mut master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    exp_avg_sq: &mut [f32],
    ctx: &DualChunkCtx<'_>,
    start_idx: usize,
) -> ChunkMetrics {
    let n = param.len();
    let has_master = ctx.config.master_weights.is_enabled() && master_param.is_some();
    let p = DualParams::new(ctx.config, ctx.coeffs, ctx.dtype_mode, ctx.state_dtype_mode);
    let seed_v = derive_variance_seed(ctx.seed_m);

    let simd_len = n & !15;
    let mut mask_count: u64 = 0;
    let mut u_sq_lanes = [0.0_f32; 16];
    let mut p_sq_lanes = [0.0_f32; 16];
    let mut u_sq_tail = 0.0_f32;
    let mut p_sq_tail = 0.0_f32;
    let cautious_enabled = ctx.config.cautious && (CAUTIOUS_ALWAYS || ctx.accumulate_metrics);

    for idx in start_idx..n {
        let offset = (ctx.chunk_offset + idx) as i32;
        let p_init = match &master_param {
            Some(mp) if has_master => mp[idx],
            _ => param[idx],
        };
        let g = sanitize_gradient(grad[idx]);
        let (m_new, v_new, u) = dual_compute_u_v(g, exp_avg[idx], exp_avg_sq[idx], &p, ctx.coeffs);
        let mask_active = (u * g) > 0.0_f32;
        let m_mask = if mask_active { 1.0_f32 } else { 0.0_f32 };

        exp_avg[idx] = quantize_value(
            m_new,
            ctx.seed_m,
            offset,
            p.state_dtype_mode,
            ctx.config.stochastic_rounding,
        );
        exp_avg_sq[idx] = quantize_value(
            v_new,
            seed_v,
            offset,
            p.state_dtype_mode,
            ctx.config.stochastic_rounding,
        );

        let p_updated = finalize_param_step(
            u,
            m_mask,
            p_init,
            ctx.config,
            p.bound_mode,
            p.is_t2,
            ctx.dtype_mode,
            cautious_enabled,
            ctx.cautious_scale,
            ctx.bound_scale,
            p.lr_wd,
        );

        if let Some(mp) = &mut master_param
            && has_master
        {
            mp[idx] = p_updated;
            param[idx] = quantize_value(p_updated, ctx.seed_p, offset, ctx.dtype_mode, false);
        } else {
            param[idx] = quantize_value(
                p_updated,
                ctx.seed_p,
                offset,
                ctx.dtype_mode,
                ctx.config.stochastic_rounding,
            );
        }

        if ACCUMULATE && ctx.accumulate_metrics {
            if ctx.config.cautious && mask_active {
                mask_count += 1;
            }
            if p.bound_mode == BoundType::Radial {
                let u_eff = if ctx.config.cautious { u * m_mask } else { u };
                let u_sq = u_eff * u_eff;
                let p_sq = p_init * p_init;
                if USE_16_LANE && idx < simd_len {
                    let lane = idx & 15;
                    u_sq_lanes[lane] += u_sq;
                    p_sq_lanes[lane] += p_sq;
                } else {
                    u_sq_tail += u_sq;
                    p_sq_tail += p_sq;
                }
            }
        }
    }

    let (u_sq_sum, p_sq_sum) =
        if ACCUMULATE && ctx.accumulate_metrics && p.bound_mode == BoundType::Radial {
            if USE_16_LANE {
                (
                    reduce_sum_16(&u_sq_lanes) + u_sq_tail,
                    reduce_sum_16(&p_sq_lanes) + p_sq_tail,
                )
            } else {
                (u_sq_tail, p_sq_tail)
            }
        } else {
            (0.0_f32, 0.0_f32)
        };

    ChunkMetrics {
        mask_count,
        u_sq_sum,
        p_sq_sum,
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn dual_pass1_range<const USE_16_LANE: bool>(
    param: &[f32],
    master_param: Option<&[f32]>,
    grad: &[f32],
    exp_avg: &[f32],
    exp_avg_sq: &[f32],
    config: &LuminaVConfig,
    coeffs: &BiasCorrectionCoeffs,
    dtype_mode: DTypeMode,
    state_dtype_mode: DTypeMode,
    start_idx: usize,
) -> ChunkMetrics {
    let n = param.len();
    let has_master = config.master_weights.is_enabled() && master_param.is_some();
    let p = DualParams::new(config, coeffs, dtype_mode, state_dtype_mode);

    let simd_len = n & !15;
    let mut mask_count: u64 = 0;
    let mut u_sq_lanes = [0.0_f32; 16];
    let mut p_sq_lanes = [0.0_f32; 16];
    let mut u_sq_tail = 0.0_f32;
    let mut p_sq_tail = 0.0_f32;

    for idx in start_idx..n {
        let g = sanitize_gradient(grad[idx]);
        let (_, _, u) = dual_compute_u_v(g, exp_avg[idx], exp_avg_sq[idx], &p, coeffs);
        let mask_active = (u * g) > 0.0_f32;
        let m_mask = if mask_active { 1.0_f32 } else { 0.0_f32 };

        if config.cautious && mask_active {
            mask_count += 1;
        }
        if p.bound_mode == BoundType::Radial {
            let u_eff = if config.cautious { u * m_mask } else { u };
            let p_val = match master_param {
                Some(mp) if has_master => mp[idx],
                _ => param[idx],
            };
            let u_sq = u_eff * u_eff;
            let p_sq = p_val * p_val;
            if USE_16_LANE && idx < simd_len {
                let lane = idx & 15;
                u_sq_lanes[lane] += u_sq;
                p_sq_lanes[lane] += p_sq;
            } else {
                u_sq_tail += u_sq;
                p_sq_tail += p_sq;
            }
        }
    }

    let (u_sq_sum, p_sq_sum) = if p.bound_mode == BoundType::Radial {
        if USE_16_LANE {
            (
                reduce_sum_16(&u_sq_lanes) + u_sq_tail,
                reduce_sum_16(&p_sq_lanes) + p_sq_tail,
            )
        } else {
            (u_sq_tail, p_sq_tail)
        }
    } else {
        (0.0_f32, 0.0_f32)
    };

    ChunkMetrics {
        mask_count,
        u_sq_sum,
        p_sq_sum,
    }
}
