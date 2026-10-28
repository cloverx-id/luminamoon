use crate::LuminaError;
use crate::config::{BoundType, BufferMode, DTypeMode, LuminaVConfig};
use crate::math::{
    apply_coordinate_bound, apply_fp16_safety_bound, compute_cautious_scale,
    compute_radial_bound_scale, reduce_sum_16, sanitize_gradient, update_post_step_ema,
};
use crate::prng::{derive_kernel_seeds, quantize_value};
use crate::state::TensorViewMut;

use super::{ELEM_CHUNK_SIZE, compute_diff_sum_scalar};

pub fn single_step_tdclm_scalar(
    view: &mut TensorViewMut<'_>,
    config: &LuminaVConfig,
    step: u64,
    param_index: usize,
) -> Result<(), LuminaError> {
    config.validate()?;
    let has_master = config.master_weights.is_enabled();
    let n_elements = view.validate(BufferMode::TdClm, has_master)?;
    if n_elements == 0 {
        return Ok(());
    }

    let total_elements = n_elements as f32;
    let beta1 = config.betas.0;
    let one_minus_beta1 = 1.0_f32 - beta1;
    let beta1_sq = beta1 * beta1;
    let one_minus_beta1_sq = 1.0_f32 - beta1_sq;
    let effective_eps = config.eps;
    let bound_mode = config.effective_bound_type();
    let (seed_m, seed_p) = derive_kernel_seeds(config.seed, step, param_index);
    let lr_wd = config.lr * config.weight_decay;
    let state_dtype_mode = if view.state_dtype_mode == DTypeMode::Float16 {
        DTypeMode::BFloat16
    } else {
        view.state_dtype_mode
    };

    let use_cd = config.centered_defect && n_elements > 1;
    let diff_mean = if use_cd {
        let diff_sum = compute_diff_sum_scalar(view.grad, view.exp_avg);
        view.scratch.set_diff_sum(diff_sum);
        diff_sum / total_elements
    } else {
        0.0_f32
    };

    let use_single_pass = (config.fused_single_pass && step > 1)
        || (!config.cautious && bound_mode != BoundType::Radial);

    if use_single_pass {
        view.scratch.set_mask_sum(0.0);
        view.scratch.set_u_sq_sum(0.0);
        view.scratch.set_p_sq_sum(0.0);

        let cautious_scale = view.scratch.cautious_scale();
        let bound_scale = view.scratch.bound_scale();

        let mut mask_count_total: u64 = 0;
        let mut u_sq_sum_total = 0.0_f32;
        let mut p_sq_sum_total = 0.0_f32;

        let mut chunk_start = 0;
        while chunk_start < n_elements {
            let chunk_end = (chunk_start + ELEM_CHUNK_SIZE).min(n_elements);
            let chunk_len = chunk_end - chunk_start;
            let simd_len = chunk_len & !15;
            let mut u_sq_lanes = [0.0_f32; 16];
            let mut p_sq_lanes = [0.0_f32; 16];
            let mut u_sq_tail = 0.0_f32;
            let mut p_sq_tail = 0.0_f32;

            for local_i in 0..chunk_len {
                let idx = chunk_start + local_i;
                let offset = idx as i32;
                let p_init = match &view.master_param {
                    Some(mp) if has_master => mp[idx],
                    _ => view.param[idx],
                };
                let g = sanitize_gradient(view.grad[idx]);
                let m = view.exp_avg[idx];

                let m_new = beta1.mul_add(m, one_minus_beta1 * g);
                let clm_m = beta1_sq.mul_add(m, one_minus_beta1_sq * g);
                let diff_gm = (g - m).clamp(-1.0e18_f32, 1.0e18_f32);
                let diff = (diff_gm - diff_mean).clamp(-1.0e18_f32, 1.0e18_f32);

                let env0 = beta1_sq.mul_add(m.abs(), effective_eps);
                let env1 = one_minus_beta1_sq.mul_add(g.abs(), env0);
                let clm_env = config.lam.mul_add(diff.abs(), env1);
                let u = clm_m / clm_env;
                let mask_active = (u * g) > 0.0_f32;
                let m_mask = if mask_active { 1.0_f32 } else { 0.0_f32 };

                view.exp_avg[idx] = quantize_value(
                    m_new,
                    seed_m,
                    offset,
                    state_dtype_mode,
                    config.stochastic_rounding,
                );

                let delta_theta = if config.cautious {
                    (u * m_mask) * cautious_scale
                } else {
                    u
                };
                let mut step_val = config.lr * delta_theta;

                match bound_mode {
                    BoundType::Radial => {
                        step_val *= bound_scale;
                    }
                    BoundType::Coordinate => {
                        step_val = apply_coordinate_bound(step_val, p_init, config.bound_ratio);
                    }
                    BoundType::None => {}
                }

                if view.dtype_mode == DTypeMode::Float16 {
                    step_val = apply_fp16_safety_bound(step_val, p_init);
                }

                let p_decayed = if config.weight_decay != 0.0_f32 {
                    (-lr_wd).mul_add(p_init, p_init)
                } else {
                    p_init
                };
                let p_updated = p_decayed - step_val;

                if let Some(mp) = &mut view.master_param
                    && has_master
                {
                    mp[idx] = p_updated;
                    view.param[idx] =
                        quantize_value(p_updated, seed_p, offset, view.dtype_mode, false);
                } else {
                    view.param[idx] = quantize_value(
                        p_updated,
                        seed_p,
                        offset,
                        view.dtype_mode,
                        config.stochastic_rounding,
                    );
                }

                if config.cautious && mask_active {
                    mask_count_total += 1;
                }
                if bound_mode == BoundType::Radial {
                    let u_eff = if config.cautious { u * m_mask } else { u };
                    let u_sq = u_eff * u_eff;
                    let p_sq = p_init * p_init;
                    if local_i < simd_len {
                        let lane = local_i & 15;
                        u_sq_lanes[lane] += u_sq;
                        p_sq_lanes[lane] += p_sq;
                    } else {
                        u_sq_tail += u_sq;
                        p_sq_tail += p_sq;
                    }
                }
            }

            if bound_mode == BoundType::Radial {
                u_sq_sum_total += reduce_sum_16(&u_sq_lanes) + u_sq_tail;
                p_sq_sum_total += reduce_sum_16(&p_sq_lanes) + p_sq_tail;
            }
            chunk_start = chunk_end;
        }

        view.scratch.set_mask_sum(mask_count_total as f32);
        view.scratch.set_u_sq_sum(u_sq_sum_total);
        view.scratch.set_p_sq_sum(p_sq_sum_total);

        if config.cautious || bound_mode == BoundType::Radial {
            update_post_step_ema(view.scratch, total_elements, config);
        } else {
            view.scratch.reset_accumulators();
        }
    } else {
        view.scratch.set_mask_sum(0.0);
        view.scratch.set_u_sq_sum(0.0);
        view.scratch.set_p_sq_sum(0.0);

        let mut mask_count_total: u64 = 0;
        let mut u_sq_sum_total = 0.0_f32;
        let mut p_sq_sum_total = 0.0_f32;

        let mut chunk_start = 0;
        while chunk_start < n_elements {
            let chunk_end = (chunk_start + ELEM_CHUNK_SIZE).min(n_elements);
            let chunk_len = chunk_end - chunk_start;
            let simd_len = chunk_len & !15;
            let mut u_sq_lanes = [0.0_f32; 16];
            let mut p_sq_lanes = [0.0_f32; 16];
            let mut u_sq_tail = 0.0_f32;
            let mut p_sq_tail = 0.0_f32;

            for local_i in 0..chunk_len {
                let idx = chunk_start + local_i;
                let g = sanitize_gradient(view.grad[idx]);
                let m = view.exp_avg[idx];
                let diff_gm = (g - m).clamp(-1.0e18_f32, 1.0e18_f32);
                let diff = (diff_gm - diff_mean).clamp(-1.0e18_f32, 1.0e18_f32);

                let clm_m = beta1_sq.mul_add(m, one_minus_beta1_sq * g);
                let env0 = beta1_sq.mul_add(m.abs(), effective_eps);
                let env1 = one_minus_beta1_sq.mul_add(g.abs(), env0);
                let clm_env = config.lam.mul_add(diff.abs(), env1);
                let u = clm_m / clm_env;
                let mask_active = (u * g) > 0.0_f32;
                let m_mask = if mask_active { 1.0_f32 } else { 0.0_f32 };

                if config.cautious && mask_active {
                    mask_count_total += 1;
                }
                if bound_mode == BoundType::Radial {
                    let u_eff = if config.cautious { u * m_mask } else { u };
                    let p_val = match &view.master_param {
                        Some(mp) if has_master => mp[idx],
                        _ => view.param[idx],
                    };
                    let u_sq = u_eff * u_eff;
                    let p_sq = p_val * p_val;
                    if local_i < simd_len {
                        let lane = local_i & 15;
                        u_sq_lanes[lane] += u_sq;
                        p_sq_lanes[lane] += p_sq;
                    } else {
                        u_sq_tail += u_sq;
                        p_sq_tail += p_sq;
                    }
                }
            }

            if bound_mode == BoundType::Radial {
                u_sq_sum_total += reduce_sum_16(&u_sq_lanes) + u_sq_tail;
                p_sq_sum_total += reduce_sum_16(&p_sq_lanes) + p_sq_tail;
            }
            chunk_start = chunk_end;
        }

        let mask_sum_acc = mask_count_total as f32;
        view.scratch.set_mask_sum(mask_sum_acc);
        view.scratch.set_u_sq_sum(u_sq_sum_total);
        view.scratch.set_p_sq_sum(p_sq_sum_total);

        let cautious_scale = if config.cautious {
            compute_cautious_scale(mask_sum_acc, total_elements, config.cautious_clamp_min)
        } else {
            1.0_f32
        };

        let bound_scale = if bound_mode == BoundType::Radial {
            compute_radial_bound_scale(
                u_sq_sum_total,
                p_sq_sum_total,
                cautious_scale,
                config.lr,
                config.bound_ratio,
            )
        } else {
            1.0_f32
        };

        for idx in 0..n_elements {
            let offset = idx as i32;
            let p_init = match &view.master_param {
                Some(mp) if has_master => mp[idx],
                _ => view.param[idx],
            };
            let g = sanitize_gradient(view.grad[idx]);
            let m = view.exp_avg[idx];
            let diff_gm = (g - m).clamp(-1.0e18_f32, 1.0e18_f32);
            let diff = (diff_gm - diff_mean).clamp(-1.0e18_f32, 1.0e18_f32);

            let m_new = beta1.mul_add(m, one_minus_beta1 * g);
            let clm_m = beta1_sq.mul_add(m, one_minus_beta1_sq * g);
            let env0 = beta1_sq.mul_add(m.abs(), effective_eps);
            let env1 = one_minus_beta1_sq.mul_add(g.abs(), env0);
            let clm_env = config.lam.mul_add(diff.abs(), env1);
            let u = clm_m / clm_env;
            let m_mask = if (u * g) > 0.0_f32 { 1.0_f32 } else { 0.0_f32 };

            view.exp_avg[idx] = quantize_value(
                m_new,
                seed_m,
                offset,
                state_dtype_mode,
                config.stochastic_rounding,
            );

            let delta_theta = if config.cautious {
                (u * m_mask) * cautious_scale
            } else {
                u
            };
            let mut step_val = config.lr * delta_theta;

            match bound_mode {
                BoundType::Radial => {
                    step_val *= bound_scale;
                }
                BoundType::Coordinate => {
                    step_val = apply_coordinate_bound(step_val, p_init, config.bound_ratio);
                }
                BoundType::None => {}
            }

            if view.dtype_mode == DTypeMode::Float16 {
                step_val = apply_fp16_safety_bound(step_val, p_init);
            }

            let p_decayed = if config.weight_decay != 0.0_f32 {
                (-lr_wd).mul_add(p_init, p_init)
            } else {
                p_init
            };
            let p_updated = p_decayed - step_val;

            if let Some(mp) = &mut view.master_param
                && has_master
            {
                mp[idx] = p_updated;
                view.param[idx] = quantize_value(p_updated, seed_p, offset, view.dtype_mode, false);
            } else {
                view.param[idx] = quantize_value(
                    p_updated,
                    seed_p,
                    offset,
                    view.dtype_mode,
                    config.stochastic_rounding,
                );
            }
        }

        view.scratch.reset_accumulators();
        view.scratch.set_cautious_scale(cautious_scale);
        view.scratch.set_bound_scale(bound_scale);
    }

    Ok(())
}
