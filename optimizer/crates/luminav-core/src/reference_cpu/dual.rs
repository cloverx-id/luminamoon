use crate::LuminaError;
use crate::config::{BoundType, BufferMode, DTypeMode, LuminaVConfig};
use crate::math::{
    apply_decoupled_weight_decay, apply_rpm_pair, apply_step_bounds, compute_bias_correction,
    compute_cautious_scale, compute_inv_sqrt_bc2_prev, compute_radial_bound_scale, reduce_sum_16,
    sanitize_gradient, tanh_pade_f32, temper_belief_defect, update_post_step_ema,
};
use crate::prng::{
    derive_spatial_seeds, derive_variance_seed, derive_weyl_step_offsets, quantize_value_weyl,
};
use crate::state::TensorViewMut;

use super::ELEM_CHUNK_SIZE;
use super::t2::single_step_t2_scalar;

pub fn single_step_dual_scalar(
    view: &mut TensorViewMut<'_>,
    config: &LuminaVConfig,
    step: u64,
    param_index: usize,
) -> Result<(), LuminaError> {
    if config.buffer == BufferMode::T2 {
        return single_step_t2_scalar(view, config, step, param_index);
    }
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
    let (seed_m, seed_p) = derive_spatial_seeds(config.seed, param_index);
    let seed_v = derive_variance_seed(seed_m);
    let (weyl_m, weyl_v, weyl_p) = derive_weyl_step_offsets(step);
    let lr_wd = config.lr * config.weight_decay;
    let use_rtv = config.uses_rtv() && step > 1;
    let use_rpm = config.uses_rpm();
    let inv_sqrt_bc2_prev = if use_rtv {
        compute_inv_sqrt_bc2_prev(step, beta2)
    } else {
        0.0_f32
    };
    // Promote FP16 state buffers to bitwise-BF16 quantization to prevent exp_avg_sq underflow.
    let state_dtype_mode = if view.state_dtype_mode == DTypeMode::Float16 {
        DTypeMode::BFloat16
    } else {
        view.state_dtype_mode
    };

    let exp_avg_sq = view
        .exp_avg_sq
        .as_deref_mut()
        .ok_or(LuminaError::MissingSecondaryStateBuffer)?;

    if !config.cautious && bound_mode != BoundType::Radial {
        let mut rpm_odd_m = 0.0_f32;
        for idx in 0..n_elements {
            let offset = idx as i32;
            let p_init = match &view.master_param {
                Some(mp) if has_master => mp[idx],
                _ => view.param[idx],
            };
            let g = sanitize_gradient(view.grad[idx]);
            let m = if use_rpm {
                if (idx & 1) == 0 {
                    if idx + 1 < n_elements {
                        let (m0_r, m1_r) = apply_rpm_pair(
                            view.exp_avg[idx],
                            view.exp_avg[idx + 1],
                            g,
                            sanitize_gradient(view.grad[idx + 1]),
                            one_minus_beta1,
                            config.eps,
                        );
                        rpm_odd_m = m1_r;
                        m0_r
                    } else {
                        view.exp_avg[idx]
                    }
                } else {
                    rpm_odd_m
                }
            } else {
                view.exp_avg[idx]
            };
            let v = exp_avg_sq[idx];

            let m_new = beta1.mul_add(m, one_minus_beta1 * g);
            let clm_m = beta1.mul_add(m_new, one_minus_beta1 * g);
            let diff = (g - m_new).clamp(-1.0e18_f32, 1.0e18_f32);

            let d_mem = if use_rtv {
                temper_belief_defect(
                    diff.abs(),
                    v,
                    m,
                    inv_sqrt_bc2_prev,
                    config.variance_temper,
                    config.eps,
                )
            } else {
                diff.abs()
            };
            let v_new = beta2.mul_add(v, one_minus_beta2 * (d_mem * d_mem));

            let sigma = v_new.max(0.0_f32).sqrt().mul_add(coeffs.c1, coeffs.c2);
            let u = tanh_pade_f32(clm_m / sigma.max(1.0e-12_f32));

            view.exp_avg[idx] = quantize_value_weyl(
                m_new,
                seed_m,
                offset,
                weyl_m,
                state_dtype_mode,
                config.stochastic_rounding,
            );
            exp_avg_sq[idx] = quantize_value_weyl(
                v_new,
                seed_v,
                offset,
                weyl_v,
                state_dtype_mode,
                config.stochastic_rounding,
            );

            let step_val = apply_step_bounds(
                config.lr * u,
                p_init,
                bound_mode,
                config.bound_ratio,
                config.bound_floor,
                1.0_f32,
                view.dtype_mode,
                false,
            );

            let p_decayed =
                apply_decoupled_weight_decay(p_init, step_val, lr_wd, config.cautious_weight_decay);
            let p_updated = p_decayed - step_val;

            if let Some(mp) = &mut view.master_param
                && has_master
            {
                mp[idx] = p_updated;
                view.param[idx] =
                    quantize_value_weyl(p_updated, seed_p, offset, weyl_p, view.dtype_mode, false);
            } else {
                view.param[idx] = quantize_value_weyl(
                    p_updated,
                    seed_p,
                    offset,
                    weyl_p,
                    view.dtype_mode,
                    config.stochastic_rounding,
                );
            }
        }
        view.scratch.reset_accumulators();
    } else if config.fused_single_pass && step > 1 {
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

            let mut rpm_odd_m = 0.0_f32;
            for local_i in 0..chunk_len {
                let idx = chunk_start + local_i;
                let offset = idx as i32;
                let p_init = match &view.master_param {
                    Some(mp) if has_master => mp[idx],
                    _ => view.param[idx],
                };
                let g = sanitize_gradient(view.grad[idx]);
                let m = if use_rpm {
                    if (idx & 1) == 0 {
                        if idx + 1 < n_elements {
                            let (m0_r, m1_r) = apply_rpm_pair(
                                view.exp_avg[idx],
                                view.exp_avg[idx + 1],
                                g,
                                sanitize_gradient(view.grad[idx + 1]),
                                one_minus_beta1,
                                config.eps,
                            );
                            rpm_odd_m = m1_r;
                            m0_r
                        } else {
                            view.exp_avg[idx]
                        }
                    } else {
                        rpm_odd_m
                    }
                } else {
                    view.exp_avg[idx]
                };
                let v = exp_avg_sq[idx];

                let m_new = beta1.mul_add(m, one_minus_beta1 * g);
                let clm_m = beta1.mul_add(m_new, one_minus_beta1 * g);
                let diff = (g - m_new).clamp(-1.0e18_f32, 1.0e18_f32);

                let d_mem = if use_rtv {
                    temper_belief_defect(
                        diff.abs(),
                        v,
                        m,
                        inv_sqrt_bc2_prev,
                        config.variance_temper,
                        config.eps,
                    )
                } else {
                    diff.abs()
                };
                let v_new = beta2.mul_add(v, one_minus_beta2 * (d_mem * d_mem));

                let sigma = v_new.max(0.0_f32).sqrt().mul_add(coeffs.c1, coeffs.c2);
                let u = tanh_pade_f32(clm_m / sigma.max(1.0e-12_f32));
                let mask_active = (u * g) > 0.0_f32;
                let m_mask = if mask_active { 1.0_f32 } else { 0.0_f32 };

                view.exp_avg[idx] = quantize_value_weyl(
                    m_new,
                    seed_m,
                    offset,
                    weyl_m,
                    state_dtype_mode,
                    config.stochastic_rounding,
                );
                exp_avg_sq[idx] = quantize_value_weyl(
                    v_new,
                    seed_v,
                    offset,
                    weyl_v,
                    state_dtype_mode,
                    config.stochastic_rounding,
                );

                let delta_theta = if config.cautious {
                    (u * m_mask) * cautious_scale
                } else {
                    u
                };
                let step_val = apply_step_bounds(
                    config.lr * delta_theta,
                    p_init,
                    bound_mode,
                    config.bound_ratio,
                    config.bound_floor,
                    bound_scale,
                    view.dtype_mode,
                    false,
                );

                let p_decayed = apply_decoupled_weight_decay(
                    p_init,
                    step_val,
                    lr_wd,
                    config.cautious_weight_decay,
                );
                let p_updated = p_decayed - step_val;

                if let Some(mp) = &mut view.master_param
                    && has_master
                {
                    mp[idx] = p_updated;
                    view.param[idx] = quantize_value_weyl(
                        p_updated,
                        seed_p,
                        offset,
                        weyl_p,
                        view.dtype_mode,
                        false,
                    );
                } else {
                    view.param[idx] = quantize_value_weyl(
                        p_updated,
                        seed_p,
                        offset,
                        weyl_p,
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

            let mut rpm_odd_m = 0.0_f32;
            for local_i in 0..chunk_len {
                let idx = chunk_start + local_i;
                let g = sanitize_gradient(view.grad[idx]);
                let m = if use_rpm {
                    if (idx & 1) == 0 {
                        if idx + 1 < n_elements {
                            let (m0_r, m1_r) = apply_rpm_pair(
                                view.exp_avg[idx],
                                view.exp_avg[idx + 1],
                                g,
                                sanitize_gradient(view.grad[idx + 1]),
                                one_minus_beta1,
                                config.eps,
                            );
                            rpm_odd_m = m1_r;
                            m0_r
                        } else {
                            view.exp_avg[idx]
                        }
                    } else {
                        rpm_odd_m
                    }
                } else {
                    view.exp_avg[idx]
                };
                let v = exp_avg_sq[idx];

                let m_new = beta1.mul_add(m, one_minus_beta1 * g);
                let clm_m = beta1.mul_add(m_new, one_minus_beta1 * g);
                let diff = (g - m_new).clamp(-1.0e18_f32, 1.0e18_f32);

                let d_mem = if use_rtv {
                    temper_belief_defect(
                        diff.abs(),
                        v,
                        m,
                        inv_sqrt_bc2_prev,
                        config.variance_temper,
                        config.eps,
                    )
                } else {
                    diff.abs()
                };
                let v_new = beta2.mul_add(v, one_minus_beta2 * (d_mem * d_mem));

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

        let mut rpm_odd_m = 0.0_f32;
        for idx in 0..n_elements {
            let offset = idx as i32;
            let p_init = match &view.master_param {
                Some(mp) if has_master => mp[idx],
                _ => view.param[idx],
            };
            let g = sanitize_gradient(view.grad[idx]);
            let m = if use_rpm {
                if (idx & 1) == 0 {
                    if idx + 1 < n_elements {
                        let (m0_r, m1_r) = apply_rpm_pair(
                            view.exp_avg[idx],
                            view.exp_avg[idx + 1],
                            g,
                            sanitize_gradient(view.grad[idx + 1]),
                            one_minus_beta1,
                            config.eps,
                        );
                        rpm_odd_m = m1_r;
                        m0_r
                    } else {
                        view.exp_avg[idx]
                    }
                } else {
                    rpm_odd_m
                }
            } else {
                view.exp_avg[idx]
            };
            let v = exp_avg_sq[idx];

            let m_new = beta1.mul_add(m, one_minus_beta1 * g);
            let clm_m = beta1.mul_add(m_new, one_minus_beta1 * g);
            let diff = (g - m_new).clamp(-1.0e18_f32, 1.0e18_f32);

            let d_mem = if use_rtv {
                temper_belief_defect(
                    diff.abs(),
                    v,
                    m,
                    inv_sqrt_bc2_prev,
                    config.variance_temper,
                    config.eps,
                )
            } else {
                diff.abs()
            };
            let v_new = beta2.mul_add(v, one_minus_beta2 * (d_mem * d_mem));

            let sigma = v_new.max(0.0_f32).sqrt().mul_add(coeffs.c1, coeffs.c2);
            let u = tanh_pade_f32(clm_m / sigma.max(1.0e-12_f32));
            let m_mask = if (u * g) > 0.0_f32 { 1.0_f32 } else { 0.0_f32 };

            view.exp_avg[idx] = quantize_value_weyl(
                m_new,
                seed_m,
                offset,
                weyl_m,
                state_dtype_mode,
                config.stochastic_rounding,
            );
            exp_avg_sq[idx] = quantize_value_weyl(
                v_new,
                seed_v,
                offset,
                weyl_v,
                state_dtype_mode,
                config.stochastic_rounding,
            );

            let delta_theta = if config.cautious {
                (u * m_mask) * cautious_scale
            } else {
                u
            };
            let step_val = apply_step_bounds(
                config.lr * delta_theta,
                p_init,
                bound_mode,
                config.bound_ratio,
                config.bound_floor,
                bound_scale,
                view.dtype_mode,
                false,
            );

            let p_decayed =
                apply_decoupled_weight_decay(p_init, step_val, lr_wd, config.cautious_weight_decay);
            let p_updated = p_decayed - step_val;

            if let Some(mp) = &mut view.master_param
                && has_master
            {
                mp[idx] = p_updated;
                view.param[idx] =
                    quantize_value_weyl(p_updated, seed_p, offset, weyl_p, view.dtype_mode, false);
            } else {
                view.param[idx] = quantize_value_weyl(
                    p_updated,
                    seed_p,
                    offset,
                    weyl_p,
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
