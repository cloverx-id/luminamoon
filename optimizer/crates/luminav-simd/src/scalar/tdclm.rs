use luminav_core::{
    BoundType, DTypeMode, LuminaVConfig, quantize_value, reduce_sum_16, sanitize_gradient,
};

use crate::common::{ChunkMetrics, TdclmChunkCtx, TdclmParams};
use crate::scalar::finalize_param_step;

#[inline(always)]
fn tdclm_compute_u(g: f32, m: f32, p: &TdclmParams, lam: f32) -> (f32, f32) {
    let m_new = p.beta1.mul_add(m, p.one_minus_beta1 * g);
    let clm_m = p.beta1_sq.mul_add(m, p.one_minus_beta1_sq * g);
    let diff = (g - m).clamp(-1.0e18_f32, 1.0e18_f32);
    let env0 = p.beta1_sq.mul_add(m.abs(), p.effective_eps);
    let env1 = p.one_minus_beta1_sq.mul_add(g.abs(), env0);
    let clm_env = lam.mul_add(diff.abs(), env1);
    (m_new, clm_m / clm_env)
}

#[inline]
pub(crate) fn tdclm_update_range<const USE_16_LANE: bool, const ACCUMULATE: bool>(
    param: &mut [f32],
    mut master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    ctx: &TdclmChunkCtx<'_>,
    start_idx: usize,
) -> ChunkMetrics {
    let n = param.len();
    let has_master = ctx.config.master_weights.is_enabled() && master_param.is_some();
    let p = TdclmParams::new(ctx.config, ctx.state_dtype_mode);

    let simd_len = n & !15;
    let mut mask_count: u64 = 0;
    let mut u_sq_lanes = [0.0_f32; 16];
    let mut p_sq_lanes = [0.0_f32; 16];
    let mut u_sq_tail = 0.0_f32;
    let mut p_sq_tail = 0.0_f32;

    for idx in start_idx..n {
        let offset = (ctx.chunk_offset + idx) as i32;
        let p_init = match &master_param {
            Some(mp) if has_master => mp[idx],
            _ => param[idx],
        };
        let g = sanitize_gradient(grad[idx]);
        let (m_new, u) = tdclm_compute_u(g, exp_avg[idx], &p, ctx.config.lam);
        let mask_active = (u * g) > 0.0_f32;
        let m_mask = if mask_active { 1.0_f32 } else { 0.0_f32 };

        exp_avg[idx] = quantize_value(
            m_new,
            ctx.seed_m,
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
            p.is_t1,
            ctx.dtype_mode,
            ctx.config.cautious,
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

        if ACCUMULATE {
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

    let (u_sq_sum, p_sq_sum) = if ACCUMULATE && p.bound_mode == BoundType::Radial {
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

#[inline]
pub(crate) fn tdclm_pass1_range<const USE_16_LANE: bool>(
    param: &[f32],
    master_param: Option<&[f32]>,
    grad: &[f32],
    exp_avg: &[f32],
    config: &LuminaVConfig,
    start_idx: usize,
) -> ChunkMetrics {
    let n = param.len();
    let has_master = config.master_weights.is_enabled() && master_param.is_some();
    let p = TdclmParams::new(config, DTypeMode::Float32);

    let simd_len = n & !15;
    let mut mask_count: u64 = 0;
    let mut u_sq_lanes = [0.0_f32; 16];
    let mut p_sq_lanes = [0.0_f32; 16];
    let mut u_sq_tail = 0.0_f32;
    let mut p_sq_tail = 0.0_f32;

    for idx in start_idx..n {
        let g = sanitize_gradient(grad[idx]);
        let (_, u) = tdclm_compute_u(g, exp_avg[idx], &p, config.lam);
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
