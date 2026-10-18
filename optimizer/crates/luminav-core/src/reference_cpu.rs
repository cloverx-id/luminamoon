use crate::LuminaError;
use crate::config::{BoundType, BufferMode, DTypeMode, LuminaVConfig};
use crate::math::{
    apply_coordinate_bound, apply_fp16_safety_bound, compute_bias_correction,
    compute_cautious_scale, compute_radial_bound_scale, reduce_sum_16, sanitize_gradient,
    tanh_pade_f32, update_post_step_ema,
};
use crate::prng::{derive_kernel_seeds, quantize_value};
use crate::state::TensorViewMut;

pub const ELEM_CHUNK_SIZE: usize = 2_097_152;

#[inline]
fn compute_diff_sum_scalar(grad: &[f32], exp_avg: &[f32]) -> f32 {
    let n = grad.len();
    let mut total_diff = 0.0_f32;
    let mut chunk_start = 0;
    while chunk_start < n {
        let chunk_end = (chunk_start + ELEM_CHUNK_SIZE).min(n);
        let chunk_len = chunk_end - chunk_start;
        let simd_len = chunk_len & !15;
        let mut lanes = [0.0_f32; 16];
        let mut i = 0;
        while i < simd_len {
            for (lane, acc) in lanes.iter_mut().enumerate() {
                let idx = chunk_start + i + lane;
                let g = sanitize_gradient(grad[idx]);
                let m = exp_avg[idx];
                let diff_gm = (g - m).clamp(-1.0e18_f32, 1.0e18_f32);
                *acc += diff_gm;
            }
            i += 16;
        }
        let mut chunk_sum = reduce_sum_16(&lanes);
        for idx in (chunk_start + simd_len)..chunk_end {
            let g = sanitize_gradient(grad[idx]);
            let m = exp_avg[idx];
            let diff_gm = (g - m).clamp(-1.0e18_f32, 1.0e18_f32);
            chunk_sum += diff_gm;
        }
        total_diff += chunk_sum;
        chunk_start = chunk_end;
    }
    total_diff
}

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
    let effective_eps = if view.dtype_mode == DTypeMode::Float16 {
        config.eps.max(1.0e-4_f32)
    } else {
        config.eps
    };
    let bound_mode = config.effective_bound_type();
    let (seed_m, seed_p) = derive_kernel_seeds(config.seed, step, param_index);
    let lr_wd = config.lr * config.weight_decay;

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
                    view.state_dtype_mode,
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
                view.state_dtype_mode,
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

pub fn single_step_dual_scalar(
    view: &mut TensorViewMut<'_>,
    config: &LuminaVConfig,
    step: u64,
    param_index: usize,
) -> Result<(), LuminaError> {
    config.validate()?;
    let has_master = config.master_weights.is_enabled();
    let n_elements = view.validate(BufferMode::Dual, has_master)?;
    if n_elements == 0 {
        return Ok(());
    }

    let total_elements = n_elements as f32;
    let (beta1, beta2) = config.betas;
    let one_minus_beta1 = 1.0_f32 - beta1;
    let one_minus_beta2 = 1.0_f32 - beta2;
    let coeffs =
        compute_bias_correction(step, beta1, beta2, config.eps, config.tau, view.dtype_mode);
    let bound_mode = config.effective_bound_type();
    let (seed_m, seed_p) = derive_kernel_seeds(config.seed, step, param_index);
    let lr_wd = config.lr * config.weight_decay;

    let use_cd = config.centered_defect && n_elements > 1;
    let diff_mean_scaled = if use_cd {
        let diff_sum = compute_diff_sum_scalar(view.grad, view.exp_avg);
        view.scratch.set_diff_sum(diff_sum);
        (diff_sum / total_elements) * beta1
    } else {
        0.0_f32
    };

    let exp_avg_sq = view
        .exp_avg_sq
        .as_deref_mut()
        .ok_or(LuminaError::MissingSecondaryStateBuffer)?;

    if config.fused_single_pass && step > 1 {
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
                let v = exp_avg_sq[idx];

                let m_new = beta1.mul_add(m, one_minus_beta1 * g);
                let clm_m = beta1.mul_add(m_new, one_minus_beta1 * g);
                let diff_gm = (g - m_new).clamp(-1.0e18_f32, 1.0e18_f32);
                let diff = (diff_gm - diff_mean_scaled).clamp(-1.0e18_f32, 1.0e18_f32);

                let mut v_new = beta2.mul_add(v, one_minus_beta2 * (diff * diff));
                if view.dtype_mode == DTypeMode::Float16
                    || view.state_dtype_mode == DTypeMode::Float16
                {
                    v_new = v_new.min(65_000.0_f32);
                }

                let sigma = v_new.max(0.0_f32).sqrt().mul_add(coeffs.c1, coeffs.c2);
                let u = tanh_pade_f32(clm_m / sigma.max(1.0e-12_f32));
                let mask_active = (u * g) > 0.0_f32;
                let m_mask = if mask_active { 1.0_f32 } else { 0.0_f32 };

                view.exp_avg[idx] = quantize_value(
                    m_new,
                    seed_m,
                    offset,
                    view.state_dtype_mode,
                    config.stochastic_rounding,
                );
                exp_avg_sq[idx] =
                    quantize_value(v_new, seed_m, offset, view.state_dtype_mode, false);

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

        update_post_step_ema(view.scratch, total_elements, config);
    } else if !config.cautious && bound_mode != BoundType::Radial {
        for idx in 0..n_elements {
            let offset = idx as i32;
            let p_init = match &view.master_param {
                Some(mp) if has_master => mp[idx],
                _ => view.param[idx],
            };
            let g = sanitize_gradient(view.grad[idx]);
            let m = view.exp_avg[idx];
            let v = exp_avg_sq[idx];

            let m_new = beta1.mul_add(m, one_minus_beta1 * g);
            let clm_m = beta1.mul_add(m_new, one_minus_beta1 * g);
            let diff_gm = (g - m_new).clamp(-1.0e18_f32, 1.0e18_f32);
            let diff = (diff_gm - diff_mean_scaled).clamp(-1.0e18_f32, 1.0e18_f32);

            let mut v_new = beta2.mul_add(v, one_minus_beta2 * (diff * diff));
            if view.dtype_mode == DTypeMode::Float16 || view.state_dtype_mode == DTypeMode::Float16
            {
                v_new = v_new.min(65_000.0_f32);
            }

            let sigma = v_new.max(0.0_f32).sqrt().mul_add(coeffs.c1, coeffs.c2);
            let u = tanh_pade_f32(clm_m / sigma.max(1.0e-12_f32));

            view.exp_avg[idx] = quantize_value(
                m_new,
                seed_m,
                offset,
                view.state_dtype_mode,
                config.stochastic_rounding,
            );
            exp_avg_sq[idx] = quantize_value(v_new, seed_m, offset, view.state_dtype_mode, false);

            let mut step_val = config.lr * u;
            if bound_mode == BoundType::Coordinate {
                step_val = apply_coordinate_bound(step_val, p_init, config.bound_ratio);
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
                let v = exp_avg_sq[idx];

                let m_new = beta1.mul_add(m, one_minus_beta1 * g);
                let clm_m = beta1.mul_add(m_new, one_minus_beta1 * g);
                let diff_gm = (g - m_new).clamp(-1.0e18_f32, 1.0e18_f32);
                let diff = (diff_gm - diff_mean_scaled).clamp(-1.0e18_f32, 1.0e18_f32);

                let mut v_new = beta2.mul_add(v, one_minus_beta2 * (diff * diff));
                if view.dtype_mode == DTypeMode::Float16
                    || view.state_dtype_mode == DTypeMode::Float16
                {
                    v_new = v_new.min(65_000.0_f32);
                }

                let sigma = v_new.max(0.0_f32).sqrt().mul_add(coeffs.c1, coeffs.c2);
                let u = tanh_pade_f32(clm_m / sigma.max(1.0e-12_f32));
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
            let v = exp_avg_sq[idx];

            let m_new = beta1.mul_add(m, one_minus_beta1 * g);
            let clm_m = beta1.mul_add(m_new, one_minus_beta1 * g);
            let diff_gm = (g - m_new).clamp(-1.0e18_f32, 1.0e18_f32);
            let diff = (diff_gm - diff_mean_scaled).clamp(-1.0e18_f32, 1.0e18_f32);

            let mut v_new = beta2.mul_add(v, one_minus_beta2 * (diff * diff));
            if view.dtype_mode == DTypeMode::Float16 || view.state_dtype_mode == DTypeMode::Float16
            {
                v_new = v_new.min(65_000.0_f32);
            }

            let sigma = v_new.max(0.0_f32).sqrt().mul_add(coeffs.c1, coeffs.c2);
            let u = tanh_pade_f32(clm_m / sigma.max(1.0e-12_f32));
            let m_mask = if (u * g) > 0.0_f32 { 1.0_f32 } else { 0.0_f32 };

            view.exp_avg[idx] = quantize_value(
                m_new,
                seed_m,
                offset,
                view.state_dtype_mode,
                config.stochastic_rounding,
            );
            exp_avg_sq[idx] = quantize_value(v_new, seed_m, offset, view.state_dtype_mode, false);

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

#[cfg(test)]
mod tests {
    use crate::LuminaError;
    use crate::config::{
        BoundType, BufferMode, DTypeMode, ExecutionMode, LuminaVConfig, MasterWeightMode,
    };
    use crate::math::{
        compute_bias_correction, sanitize_gradient, tanh_pade_f32, update_post_step_ema,
    };
    use crate::prng::{
        _prng, derive_kernel_seeds, f16_bits_to_f32, f32_to_f16_bits_rne, stochastic_round_bf16,
        stochastic_round_fp16,
    };
    use crate::reference_cpu::{single_step_dual_scalar, single_step_tdclm_scalar};
    use crate::state::{ParamState, ScratchBuffer, TensorViewMut};

    #[test]
    fn test_prng_determinism_and_signed_shift_invariants() {
        let s0 = _prng(1337, 0);
        let s0_repeat = _prng(1337, 0);
        assert_eq!(s0, s0_repeat);

        let s1 = _prng(1337, 1);
        let s2 = _prng(1338, 0);
        assert_ne!(s0, s1);
        assert_ne!(s0, s2);

        let mut expected = -1_640_531_527_i32;
        expected = (expected ^ (expected >> 16)).wrapping_mul(-2_048_173_461_i32);
        expected = (expected ^ (expected >> 13)).wrapping_mul(-1_028_477_387_i32);
        expected ^= expected >> 16;
        assert_eq!(_prng(0, 0), expected);

        let (seed_m, seed_p) = derive_kernel_seeds(1337, 1, 0);
        assert_eq!(seed_m, ((1337_u64 + 0x9E37_79B9_u64) & 0x7FFF_FFFF) as i32);
        assert_eq!(
            seed_p,
            (((seed_m as u64) + 0x5555_5555_u64) & 0x7FFF_FFFF) as i32
        );
    }

    #[test]
    fn test_stochastic_rounding_bf16_and_subnormal_fp16() {
        let val = 1.000_123_f32;
        for idx in 0..64 {
            let sr_bf16 = stochastic_round_bf16(val, 1337, idx);
            assert!(sr_bf16.is_finite());
            assert_eq!(sr_bf16.to_bits() & 0xFFFF, 0);
            assert!((sr_bf16 - val).abs() < 1.0e-2);

            let sr_fp16 = stochastic_round_fp16(val, 1337, idx);
            assert!(sr_fp16.is_finite());
            assert!((sr_fp16 - val).abs() < 2.0e-3);
        }

        let tiny = f32::from_bits(102 << 23);
        assert_eq!(stochastic_round_fp16(tiny, 1337, 0), 0.0_f32);
        assert_eq!(stochastic_round_fp16(-tiny, 1337, 0), -0.0_f32);

        let half_bits = f32_to_f16_bits_rne(0.5_f32);
        assert_eq!(f16_bits_to_f32(half_bits), 0.5_f32);
    }

    #[test]
    fn test_tanh_pade_f32_precision_against_std_tanh() {
        let mut max_abs_err = 0.0_f32;
        for i in -10_000..=10_000 {
            let x = (i as f32) * 0.001_f32;
            let approx = tanh_pade_f32(x);
            let exact = x.tanh();
            let err = (approx - exact).abs();
            if err > max_abs_err {
                max_abs_err = err;
            }
        }
        assert!(
            max_abs_err <= 3.0e-7_f32,
            "max_abs_err {max_abs_err} exceeded threshold"
        );
        assert_eq!(tanh_pade_f32(0.0_f32), 0.0_f32);
        assert_eq!(tanh_pade_f32(25.0_f32), 1.0_f32);
        assert_eq!(tanh_pade_f32(-25.0_f32), -1.0_f32);
        assert!(tanh_pade_f32(f32::NAN).is_nan());
    }

    #[test]
    fn test_scratch_buffer_ema_smoothing_and_reset() {
        let mut scratch = ScratchBuffer::new();
        assert_eq!(scratch.cautious_scale(), 1.0_f32);
        assert_eq!(scratch.bound_scale(), 1.0_f32);

        scratch.set_mask_sum(50.0_f32);
        scratch.set_u_sq_sum(25.0_f32);
        scratch.set_p_sq_sum(100.0_f32);
        scratch.set_diff_sum(12.5_f32);

        let cfg = LuminaVConfig::builder()
            .lr(1.0e-3_f32)
            .cautious_clamp_min(0.5_f32)
            .ema_decay(0.8_f32)
            .bound_type(BoundType::Radial)
            .bound_ratio(0.05_f32)
            .cautious(true)
            .build()
            .expect("valid test config");

        update_post_step_ema(&mut scratch, 100.0_f32, &cfg);

        assert!((scratch.cautious_scale() - 1.2_f32).abs() < 1.0e-6_f32);
        assert!(scratch.bound_scale() > 0.0_f32 && scratch.bound_scale() <= 1.0_f32);
        assert_eq!(scratch.mask_sum(), 0.0_f32);
        assert_eq!(scratch.u_sq_sum(), 0.0_f32);
        assert_eq!(scratch.p_sq_sum(), 0.0_f32);
        assert_eq!(scratch.diff_sum(), 0.0_f32);
    }

    #[test]
    fn test_config_validation_and_typo_suggestions() {
        assert!(LuminaVConfig::builder().build().is_ok());
        assert_eq!(
            LuminaVConfig::builder().lr(-1.0e-4).build(),
            Err(LuminaError::InvalidLearningRate(-1.0e-4))
        );
        assert_eq!(
            LuminaVConfig::builder().betas(1.0, 0.999).build(),
            Err(LuminaError::InvalidBeta1(1.0))
        );
        assert_eq!(
            BufferMode::try_from_count(3).unwrap_err().to_string(),
            "Invalid buffer count: 3. LuminaV only supports buffer=1 (Single-Buffer TD-CLM) or buffer=2 (Dual-Buffer CLM)."
        );
        assert_eq!(
            BufferMode::try_from_str("duel").unwrap_err().to_string(),
            "Invalid buffer count: 'duel'. LuminaV only supports buffer=1 (Single-Buffer TD-CLM) or buffer=2 (Dual-Buffer CLM). Did you mean 'dual'?"
        );
        assert_eq!(
            BoundType::try_from_str("radal").unwrap_err().to_string(),
            "Invalid bound_type: 'radal'. Expected 'coordinate' or 'radial'. Did you mean 'radial'?"
        );
        assert_eq!(
            ExecutionMode::try_from_str("vulkn")
                .unwrap_err()
                .to_string(),
            "Invalid execution mode: 'vulkn'. Expected 'auto', 'vulkan', 'simd', or 'scalar'. Did you mean 'vulkan'?"
        );
        assert_eq!(
            MasterWeightMode::try_from_str("sem")
                .unwrap_err()
                .to_string(),
            "Invalid master_weights: 'sem'. Expected False, 'none', 'semi', or 'full'. Did you mean 'semi'?"
        );

        let tdclm_cfg = LuminaVConfig::builder()
            .buffer(BufferMode::TdClm)
            .build()
            .expect("valid TD-CLM config");
        assert!((tdclm_cfg.ema_decay - 0.95_f32).abs() < 1.0e-6);

        let coeffs = compute_bias_correction(1, 0.9, 0.999, 1.0e-8, 0.8, DTypeMode::Float32);
        assert!((coeffs.bc1_nes - 0.19_f32).abs() < 1.0e-6);
        assert!(coeffs.c1 > 0.0 && coeffs.c2 > 0.0);
    }

    #[test]
    fn test_single_step_tdclm_and_dual_scalar_golden_oracle() {
        let n = 32;
        let initial_params: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let mut grads: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { 0.05_f32 } else { -0.03_f32 })
            .collect();
        grads[0] = f32::NAN;
        assert_eq!(sanitize_gradient(grads[0]), 0.0_f32);

        let tdclm_cfg = LuminaVConfig::builder()
            .buffer(BufferMode::TdClm)
            .bound_type(BoundType::Radial)
            .centered_defect(true)
            .master_weights(MasterWeightMode::None)
            .build()
            .expect("valid config");

        let mut p_tdclm = initial_params.clone();
        let mut state_tdclm = ParamState::new(&p_tdclm, BufferMode::TdClm, false);

        for _ in 0..2 {
            let step = state_tdclm.increment_step();
            let mut view = TensorViewMut {
                param: &mut p_tdclm,
                master_param: None,
                grad: &grads,
                exp_avg: &mut state_tdclm.exp_avg,
                exp_avg_sq: None,
                scratch: &mut state_tdclm.scratch,
                dtype_mode: DTypeMode::Float32,
                state_dtype_mode: DTypeMode::Float32,
            };
            single_step_tdclm_scalar(&mut view, &tdclm_cfg, step, 0)
                .expect("TD-CLM scalar step should succeed");
        }

        assert!(p_tdclm.iter().all(|v| v.is_finite()));
        assert!(state_tdclm.exp_avg.iter().all(|v| v.is_finite()));
        assert_ne!(p_tdclm, initial_params);

        let dual_cfg = LuminaVConfig::builder()
            .buffer(BufferMode::Dual)
            .bound_type(BoundType::Coordinate)
            .centered_defect(true)
            .master_weights(MasterWeightMode::Full)
            .build()
            .expect("valid config");

        let mut p_dual = initial_params.clone();
        let mut state_dual = ParamState::new(&p_dual, BufferMode::Dual, true);

        for _ in 0..2 {
            let step = state_dual.increment_step();
            let mut view = TensorViewMut {
                param: &mut p_dual,
                master_param: state_dual.master_param.as_deref_mut(),
                grad: &grads,
                exp_avg: &mut state_dual.exp_avg,
                exp_avg_sq: state_dual.exp_avg_sq.as_deref_mut(),
                scratch: &mut state_dual.scratch,
                dtype_mode: DTypeMode::BFloat16,
                state_dtype_mode: DTypeMode::Float32,
            };
            single_step_dual_scalar(&mut view, &dual_cfg, step, 0)
                .expect("Dual scalar step should succeed");
        }

        assert!(p_dual.iter().all(|v| v.is_finite()));
        assert!(state_dual.exp_avg.iter().all(|v| v.is_finite()));
        assert!(
            state_dual
                .exp_avg_sq
                .as_ref()
                .expect("v buffer")
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0)
        );
        assert_ne!(p_dual, initial_params);
    }
}
