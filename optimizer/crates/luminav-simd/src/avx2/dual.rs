#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _CMP_GT_OQ, _mm256_add_ps, _mm256_and_ps, _mm256_cmp_ps, _mm256_div_ps, _mm256_fmadd_ps,
    _mm256_fnmadd_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_min_ps, _mm256_movemask_ps,
    _mm256_mul_ps, _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_ps, _mm256_sqrt_ps,
    _mm256_storeu_ps, _mm256_sub_ps,
};

#[cfg(target_arch = "x86_64")]
use crate::avx512::ChunkMetrics;

#[cfg(target_arch = "x86_64")]
use luminav_core::{
    BiasCorrectionCoeffs, BoundType, BufferMode, DTypeMode, LuminaError, LuminaVConfig,
    TensorViewMut, apply_coordinate_bound, apply_fp16_safety_bound, compute_bias_correction,
    compute_cautious_scale, compute_radial_bound_scale, derive_kernel_seeds, quantize_value,
    sanitize_gradient, tanh_pade_f32, update_post_step_ema,
};

#[cfg(target_arch = "x86_64")]
use super::{
    apply_coordinate_bound_avx2, apply_fp16_safety_bound_avx2, clamp_diff_avx2,
    compute_diff_sum_chunk_avx2, offset_vector_avx2, quantize_avx2, reduce_add_ps_avx2_pair,
    sanitize_gradient_avx2, tanh_pade_avx2,
};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dual_single_pass_chunk_avx2(
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
    diff_mean_scaled: f32,
    cautious_scale: f32,
    bound_scale: f32,
    seed_m: i32,
    seed_p: i32,
    accumulate_metrics: bool,
) -> ChunkMetrics {
    unsafe {
        let n = param.len();
        let simd_len = n & !15;
        let has_master = config.master_weights.is_enabled() && master_param.is_some();
        let bound_mode = config.effective_bound_type();
        let (beta1, beta2) = config.betas;
        let one_minus_beta1 = 1.0_f32 - beta1;
        let one_minus_beta2 = 1.0_f32 - beta2;
        let lr_wd = config.lr * config.weight_decay;
        let clamp_v = dtype_mode == DTypeMode::Float16 || state_dtype_mode == DTypeMode::Float16;
        let state_dtype_mode = if state_dtype_mode == DTypeMode::Float16 {
            DTypeMode::BFloat16
        } else {
            state_dtype_mode
        };

        let v_beta1 = _mm256_set1_ps(beta1);
        let v_one_minus_beta1 = _mm256_set1_ps(one_minus_beta1);
        let v_beta2 = _mm256_set1_ps(beta2);
        let v_one_minus_beta2 = _mm256_set1_ps(one_minus_beta2);
        let v_c1 = _mm256_set1_ps(coeffs.c1);
        let v_c2 = _mm256_set1_ps(coeffs.c2);
        let v_diff_mean_scaled = _mm256_set1_ps(diff_mean_scaled);
        let v_cautious_scale = _mm256_set1_ps(cautious_scale);
        let v_bound_scale = _mm256_set1_ps(bound_scale);
        let v_bound_ratio = _mm256_set1_ps(config.bound_ratio);
        let v_lr = _mm256_set1_ps(config.lr);
        let v_lr_wd = _mm256_set1_ps(lr_wd);
        let v_zero = _mm256_setzero_ps();
        let v_sigma_min = _mm256_set1_ps(1.0e-12_f32);
        let v_vmax = _mm256_set1_ps(65_000.0_f32);
        let v_seed_m = _mm256_set1_epi32(seed_m);
        let v_seed_p = _mm256_set1_epi32(seed_p);

        let mut mask_count: u64 = 0;
        let mut v_u_sq_lo = _mm256_setzero_ps();
        let mut v_u_sq_hi = _mm256_setzero_ps();
        let mut v_p_sq_lo = _mm256_setzero_ps();
        let mut v_p_sq_hi = _mm256_setzero_ps();

        let mut mp_slice = master_param;

        let mut i = 0;
        while i < simd_len {
            for half in 0..2 {
                let sub_i = i + half * 8;
                let global_offset = (chunk_offset + sub_i) as i32;
                let v_offsets = offset_vector_avx2(global_offset);

                let v_p_init = match &mp_slice {
                    Some(mp) if has_master => _mm256_loadu_ps(mp.as_ptr().add(sub_i)),
                    _ => _mm256_loadu_ps(param.as_ptr().add(sub_i)),
                };
                let v_g = sanitize_gradient_avx2(_mm256_loadu_ps(grad.as_ptr().add(sub_i)));
                let v_m = _mm256_loadu_ps(exp_avg.as_ptr().add(sub_i));
                let v_v = _mm256_loadu_ps(exp_avg_sq.as_ptr().add(sub_i));

                let v_g_scaled = _mm256_mul_ps(v_one_minus_beta1, v_g);
                let v_m_new = _mm256_fmadd_ps(v_beta1, v_m, v_g_scaled);
                let v_clm_m = _mm256_fmadd_ps(v_beta1, v_m_new, v_g_scaled);
                let v_diff_gm = clamp_diff_avx2(_mm256_sub_ps(v_g, v_m_new));
                let v_diff = clamp_diff_avx2(_mm256_sub_ps(v_diff_gm, v_diff_mean_scaled));

                let v_diff_sq = _mm256_mul_ps(v_diff, v_diff);
                let mut v_v_new =
                    _mm256_fmadd_ps(v_beta2, v_v, _mm256_mul_ps(v_one_minus_beta2, v_diff_sq));
                if clamp_v {
                    v_v_new = _mm256_min_ps(v_v_new, v_vmax);
                }

                let v_sigma =
                    _mm256_fmadd_ps(_mm256_sqrt_ps(_mm256_max_ps(v_v_new, v_zero)), v_c1, v_c2);
                let v_u =
                    tanh_pade_avx2(_mm256_div_ps(v_clm_m, _mm256_max_ps(v_sigma, v_sigma_min)));

                let v_mask_ps = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_mul_ps(v_u, v_g), v_zero);
                let v_u_masked = _mm256_and_ps(v_mask_ps, v_u);

                let v_m_stored = quantize_avx2(
                    v_m_new,
                    v_seed_m,
                    v_offsets,
                    state_dtype_mode,
                    config.stochastic_rounding,
                );
                _mm256_storeu_ps(exp_avg.as_mut_ptr().add(sub_i), v_m_stored);

                let v_v_stored =
                    quantize_avx2(v_v_new, v_seed_m, v_offsets, state_dtype_mode, false);
                _mm256_storeu_ps(exp_avg_sq.as_mut_ptr().add(sub_i), v_v_stored);

                let v_delta_theta = if config.cautious && accumulate_metrics {
                    _mm256_mul_ps(v_u_masked, v_cautious_scale)
                } else {
                    v_u
                };
                let mut v_step = _mm256_mul_ps(v_lr, v_delta_theta);

                match bound_mode {
                    BoundType::Radial => {
                        v_step = _mm256_mul_ps(v_step, v_bound_scale);
                    }
                    BoundType::Coordinate => {
                        v_step = apply_coordinate_bound_avx2(v_step, v_p_init, v_bound_ratio);
                    }
                    BoundType::None => {}
                }

                if dtype_mode == DTypeMode::Float16 {
                    v_step = apply_fp16_safety_bound_avx2(v_step, v_p_init);
                }

                let v_p_decayed = if config.weight_decay != 0.0_f32 {
                    _mm256_fnmadd_ps(v_lr_wd, v_p_init, v_p_init)
                } else {
                    v_p_init
                };
                let v_p_updated = _mm256_sub_ps(v_p_decayed, v_step);

                if let Some(mp) = &mut mp_slice
                    && has_master
                {
                    _mm256_storeu_ps(mp.as_mut_ptr().add(sub_i), v_p_updated);
                    let v_p_q = quantize_avx2(v_p_updated, v_seed_p, v_offsets, dtype_mode, false);
                    _mm256_storeu_ps(param.as_mut_ptr().add(sub_i), v_p_q);
                } else {
                    let v_p_q = quantize_avx2(
                        v_p_updated,
                        v_seed_p,
                        v_offsets,
                        dtype_mode,
                        config.stochastic_rounding,
                    );
                    _mm256_storeu_ps(param.as_mut_ptr().add(sub_i), v_p_q);
                }

                if accumulate_metrics {
                    if config.cautious {
                        mask_count +=
                            u64::from((_mm256_movemask_ps(v_mask_ps) as u32).count_ones());
                    }
                    if bound_mode == BoundType::Radial {
                        let v_u_eff = if config.cautious { v_u_masked } else { v_u };
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

        let mut u_sq_sum = if accumulate_metrics && bound_mode == BoundType::Radial {
            reduce_add_ps_avx2_pair(v_u_sq_lo, v_u_sq_hi)
        } else {
            0.0_f32
        };
        let mut p_sq_sum = if accumulate_metrics && bound_mode == BoundType::Radial {
            reduce_add_ps_avx2_pair(v_p_sq_lo, v_p_sq_hi)
        } else {
            0.0_f32
        };

        for idx in simd_len..n {
            let offset = (chunk_offset + idx) as i32;
            let p_init = match &mp_slice {
                Some(mp) if has_master => mp[idx],
                _ => param[idx],
            };
            let g = sanitize_gradient(grad[idx]);
            let m = exp_avg[idx];
            let v = exp_avg_sq[idx];

            let m_new = beta1.mul_add(m, one_minus_beta1 * g);
            let clm_m = beta1.mul_add(m_new, one_minus_beta1 * g);
            let diff_gm = (g - m_new).clamp(-1.0e18_f32, 1.0e18_f32);
            let diff = (diff_gm - diff_mean_scaled).clamp(-1.0e18_f32, 1.0e18_f32);

            let mut v_new = beta2.mul_add(v, one_minus_beta2 * (diff * diff));
            if clamp_v {
                v_new = v_new.min(65_000.0_f32);
            }

            let sigma = v_new.max(0.0_f32).sqrt().mul_add(coeffs.c1, coeffs.c2);
            let u = tanh_pade_f32(clm_m / sigma.max(1.0e-12_f32));
            let mask_active = (u * g) > 0.0_f32;
            let m_mask = if mask_active { 1.0_f32 } else { 0.0_f32 };

            exp_avg[idx] = quantize_value(
                m_new,
                seed_m,
                offset,
                state_dtype_mode,
                config.stochastic_rounding,
            );
            exp_avg_sq[idx] = quantize_value(v_new, seed_m, offset, state_dtype_mode, false);

            let delta_theta = if config.cautious && accumulate_metrics {
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

            if dtype_mode == DTypeMode::Float16 {
                step_val = apply_fp16_safety_bound(step_val, p_init);
            }

            let p_decayed = if config.weight_decay != 0.0_f32 {
                (-lr_wd).mul_add(p_init, p_init)
            } else {
                p_init
            };
            let p_updated = p_decayed - step_val;

            if let Some(mp) = &mut mp_slice
                && has_master
            {
                mp[idx] = p_updated;
                param[idx] = quantize_value(p_updated, seed_p, offset, dtype_mode, false);
            } else {
                param[idx] = quantize_value(
                    p_updated,
                    seed_p,
                    offset,
                    dtype_mode,
                    config.stochastic_rounding,
                );
            }

            if accumulate_metrics {
                if config.cautious && mask_active {
                    mask_count += 1;
                }
                if bound_mode == BoundType::Radial {
                    let u_eff = if config.cautious { u * m_mask } else { u };
                    u_sq_sum += u_eff * u_eff;
                    p_sq_sum += p_init * p_init;
                }
            }
        }

        ChunkMetrics {
            mask_count,
            u_sq_sum,
            p_sq_sum,
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dual_pass1_chunk_avx2(
    param: &[f32],
    master_param: Option<&[f32]>,
    grad: &[f32],
    exp_avg: &[f32],
    exp_avg_sq: &[f32],
    config: &LuminaVConfig,
    coeffs: &BiasCorrectionCoeffs,
    dtype_mode: DTypeMode,
    state_dtype_mode: DTypeMode,
    diff_mean_scaled: f32,
) -> ChunkMetrics {
    unsafe {
        let n = param.len();
        let simd_len = n & !15;
        let has_master = config.master_weights.is_enabled() && master_param.is_some();
        let bound_mode = config.effective_bound_type();
        let (beta1, beta2) = config.betas;
        let one_minus_beta1 = 1.0_f32 - beta1;
        let one_minus_beta2 = 1.0_f32 - beta2;
        let clamp_v = dtype_mode == DTypeMode::Float16 || state_dtype_mode == DTypeMode::Float16;

        let v_beta1 = _mm256_set1_ps(beta1);
        let v_one_minus_beta1 = _mm256_set1_ps(one_minus_beta1);
        let v_beta2 = _mm256_set1_ps(beta2);
        let v_one_minus_beta2 = _mm256_set1_ps(one_minus_beta2);
        let v_c1 = _mm256_set1_ps(coeffs.c1);
        let v_c2 = _mm256_set1_ps(coeffs.c2);
        let v_diff_mean_scaled = _mm256_set1_ps(diff_mean_scaled);
        let v_zero = _mm256_setzero_ps();
        let v_sigma_min = _mm256_set1_ps(1.0e-12_f32);
        let v_vmax = _mm256_set1_ps(65_000.0_f32);

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
                let v_v = _mm256_loadu_ps(exp_avg_sq.as_ptr().add(sub_i));

                let v_g_scaled = _mm256_mul_ps(v_one_minus_beta1, v_g);
                let v_m_new = _mm256_fmadd_ps(v_beta1, v_m, v_g_scaled);
                let v_clm_m = _mm256_fmadd_ps(v_beta1, v_m_new, v_g_scaled);
                let v_diff_gm = clamp_diff_avx2(_mm256_sub_ps(v_g, v_m_new));
                let v_diff = clamp_diff_avx2(_mm256_sub_ps(v_diff_gm, v_diff_mean_scaled));

                let v_diff_sq = _mm256_mul_ps(v_diff, v_diff);
                let mut v_v_new =
                    _mm256_fmadd_ps(v_beta2, v_v, _mm256_mul_ps(v_one_minus_beta2, v_diff_sq));
                if clamp_v {
                    v_v_new = _mm256_min_ps(v_v_new, v_vmax);
                }

                let v_sigma =
                    _mm256_fmadd_ps(_mm256_sqrt_ps(_mm256_max_ps(v_v_new, v_zero)), v_c1, v_c2);
                let v_u =
                    tanh_pade_avx2(_mm256_div_ps(v_clm_m, _mm256_max_ps(v_sigma, v_sigma_min)));

                let v_mask_ps = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_mul_ps(v_u, v_g), v_zero);

                if config.cautious {
                    mask_count += u64::from((_mm256_movemask_ps(v_mask_ps) as u32).count_ones());
                }
                if bound_mode == BoundType::Radial {
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

        let mut u_sq_sum = if bound_mode == BoundType::Radial {
            reduce_add_ps_avx2_pair(v_u_sq_lo, v_u_sq_hi)
        } else {
            0.0_f32
        };
        let mut p_sq_sum = if bound_mode == BoundType::Radial {
            reduce_add_ps_avx2_pair(v_p_sq_lo, v_p_sq_hi)
        } else {
            0.0_f32
        };

        for idx in simd_len..n {
            let g = sanitize_gradient(grad[idx]);
            let m = exp_avg[idx];
            let v = exp_avg_sq[idx];

            let m_new = beta1.mul_add(m, one_minus_beta1 * g);
            let clm_m = beta1.mul_add(m_new, one_minus_beta1 * g);
            let diff_gm = (g - m_new).clamp(-1.0e18_f32, 1.0e18_f32);
            let diff = (diff_gm - diff_mean_scaled).clamp(-1.0e18_f32, 1.0e18_f32);

            let mut v_new = beta2.mul_add(v, one_minus_beta2 * (diff * diff));
            if clamp_v {
                v_new = v_new.min(65_000.0_f32);
            }

            let sigma = v_new.max(0.0_f32).sqrt().mul_add(coeffs.c1, coeffs.c2);
            let u = tanh_pade_f32(clm_m / sigma.max(1.0e-12_f32));
            let mask_active = (u * g) > 0.0_f32;
            let m_mask = if mask_active { 1.0_f32 } else { 0.0_f32 };

            if config.cautious && mask_active {
                mask_count += 1;
            }
            if bound_mode == BoundType::Radial {
                let u_eff = if config.cautious { u * m_mask } else { u };
                let p_val = match master_param {
                    Some(mp) if has_master => mp[idx],
                    _ => param[idx],
                };
                u_sq_sum += u_eff * u_eff;
                p_sq_sum += p_val * p_val;
            }
        }

        ChunkMetrics {
            mask_count,
            u_sq_sum,
            p_sq_sum,
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dual_pass2_chunk_avx2(
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
    diff_mean_scaled: f32,
    cautious_scale: f32,
    bound_scale: f32,
    seed_m: i32,
    seed_p: i32,
) {
    unsafe {
        let n = param.len();
        let simd_len = n & !7;
        let has_master = config.master_weights.is_enabled() && master_param.is_some();
        let bound_mode = config.effective_bound_type();
        let (beta1, beta2) = config.betas;
        let one_minus_beta1 = 1.0_f32 - beta1;
        let one_minus_beta2 = 1.0_f32 - beta2;
        let lr_wd = config.lr * config.weight_decay;
        let clamp_v = dtype_mode == DTypeMode::Float16 || state_dtype_mode == DTypeMode::Float16;
        let state_dtype_mode = if state_dtype_mode == DTypeMode::Float16 {
            DTypeMode::BFloat16
        } else {
            state_dtype_mode
        };

        let v_beta1 = _mm256_set1_ps(beta1);
        let v_one_minus_beta1 = _mm256_set1_ps(one_minus_beta1);
        let v_beta2 = _mm256_set1_ps(beta2);
        let v_one_minus_beta2 = _mm256_set1_ps(one_minus_beta2);
        let v_c1 = _mm256_set1_ps(coeffs.c1);
        let v_c2 = _mm256_set1_ps(coeffs.c2);
        let v_diff_mean_scaled = _mm256_set1_ps(diff_mean_scaled);
        let v_cautious_scale = _mm256_set1_ps(cautious_scale);
        let v_bound_scale = _mm256_set1_ps(bound_scale);
        let v_bound_ratio = _mm256_set1_ps(config.bound_ratio);
        let v_lr = _mm256_set1_ps(config.lr);
        let v_lr_wd = _mm256_set1_ps(lr_wd);
        let v_zero = _mm256_setzero_ps();
        let v_sigma_min = _mm256_set1_ps(1.0e-12_f32);
        let v_vmax = _mm256_set1_ps(65_000.0_f32);
        let v_seed_m = _mm256_set1_epi32(seed_m);
        let v_seed_p = _mm256_set1_epi32(seed_p);

        let mut mp_slice = master_param;

        let mut i = 0;
        while i < simd_len {
            let global_offset = (chunk_offset + i) as i32;
            let v_offsets = offset_vector_avx2(global_offset);

            let v_p_init = match &mp_slice {
                Some(mp) if has_master => _mm256_loadu_ps(mp.as_ptr().add(i)),
                _ => _mm256_loadu_ps(param.as_ptr().add(i)),
            };
            let v_g = sanitize_gradient_avx2(_mm256_loadu_ps(grad.as_ptr().add(i)));
            let v_m = _mm256_loadu_ps(exp_avg.as_ptr().add(i));
            let v_v = _mm256_loadu_ps(exp_avg_sq.as_ptr().add(i));

            let v_g_scaled = _mm256_mul_ps(v_one_minus_beta1, v_g);
            let v_m_new = _mm256_fmadd_ps(v_beta1, v_m, v_g_scaled);
            let v_clm_m = _mm256_fmadd_ps(v_beta1, v_m_new, v_g_scaled);
            let v_diff_gm = clamp_diff_avx2(_mm256_sub_ps(v_g, v_m_new));
            let v_diff = clamp_diff_avx2(_mm256_sub_ps(v_diff_gm, v_diff_mean_scaled));

            let v_diff_sq = _mm256_mul_ps(v_diff, v_diff);
            let mut v_v_new =
                _mm256_fmadd_ps(v_beta2, v_v, _mm256_mul_ps(v_one_minus_beta2, v_diff_sq));
            if clamp_v {
                v_v_new = _mm256_min_ps(v_v_new, v_vmax);
            }

            let v_sigma =
                _mm256_fmadd_ps(_mm256_sqrt_ps(_mm256_max_ps(v_v_new, v_zero)), v_c1, v_c2);
            let v_u = tanh_pade_avx2(_mm256_div_ps(v_clm_m, _mm256_max_ps(v_sigma, v_sigma_min)));

            let v_mask_ps = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_mul_ps(v_u, v_g), v_zero);
            let v_u_masked = _mm256_and_ps(v_mask_ps, v_u);

            let v_m_stored = quantize_avx2(
                v_m_new,
                v_seed_m,
                v_offsets,
                state_dtype_mode,
                config.stochastic_rounding,
            );
            _mm256_storeu_ps(exp_avg.as_mut_ptr().add(i), v_m_stored);

            let v_v_stored = quantize_avx2(v_v_new, v_seed_m, v_offsets, state_dtype_mode, false);
            _mm256_storeu_ps(exp_avg_sq.as_mut_ptr().add(i), v_v_stored);

            let v_delta_theta = if config.cautious {
                _mm256_mul_ps(v_u_masked, v_cautious_scale)
            } else {
                v_u
            };
            let mut v_step = _mm256_mul_ps(v_lr, v_delta_theta);

            match bound_mode {
                BoundType::Radial => {
                    v_step = _mm256_mul_ps(v_step, v_bound_scale);
                }
                BoundType::Coordinate => {
                    v_step = apply_coordinate_bound_avx2(v_step, v_p_init, v_bound_ratio);
                }
                BoundType::None => {}
            }

            if dtype_mode == DTypeMode::Float16 {
                v_step = apply_fp16_safety_bound_avx2(v_step, v_p_init);
            }

            let v_p_decayed = if config.weight_decay != 0.0_f32 {
                _mm256_fnmadd_ps(v_lr_wd, v_p_init, v_p_init)
            } else {
                v_p_init
            };
            let v_p_updated = _mm256_sub_ps(v_p_decayed, v_step);

            if let Some(mp) = &mut mp_slice
                && has_master
            {
                _mm256_storeu_ps(mp.as_mut_ptr().add(i), v_p_updated);
                let v_p_q = quantize_avx2(v_p_updated, v_seed_p, v_offsets, dtype_mode, false);
                _mm256_storeu_ps(param.as_mut_ptr().add(i), v_p_q);
            } else {
                let v_p_q = quantize_avx2(
                    v_p_updated,
                    v_seed_p,
                    v_offsets,
                    dtype_mode,
                    config.stochastic_rounding,
                );
                _mm256_storeu_ps(param.as_mut_ptr().add(i), v_p_q);
            }

            i += 8;
        }

        for idx in simd_len..n {
            let offset = (chunk_offset + idx) as i32;
            let p_init = match &mp_slice {
                Some(mp) if has_master => mp[idx],
                _ => param[idx],
            };
            let g = sanitize_gradient(grad[idx]);
            let m = exp_avg[idx];
            let v = exp_avg_sq[idx];

            let m_new = beta1.mul_add(m, one_minus_beta1 * g);
            let clm_m = beta1.mul_add(m_new, one_minus_beta1 * g);
            let diff_gm = (g - m_new).clamp(-1.0e18_f32, 1.0e18_f32);
            let diff = (diff_gm - diff_mean_scaled).clamp(-1.0e18_f32, 1.0e18_f32);

            let mut v_new = beta2.mul_add(v, one_minus_beta2 * (diff * diff));
            if clamp_v {
                v_new = v_new.min(65_000.0_f32);
            }

            let sigma = v_new.max(0.0_f32).sqrt().mul_add(coeffs.c1, coeffs.c2);
            let u = tanh_pade_f32(clm_m / sigma.max(1.0e-12_f32));
            let m_mask = if (u * g) > 0.0_f32 { 1.0_f32 } else { 0.0_f32 };

            exp_avg[idx] = quantize_value(
                m_new,
                seed_m,
                offset,
                state_dtype_mode,
                config.stochastic_rounding,
            );
            exp_avg_sq[idx] = quantize_value(v_new, seed_m, offset, state_dtype_mode, false);

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

            if dtype_mode == DTypeMode::Float16 {
                step_val = apply_fp16_safety_bound(step_val, p_init);
            }

            let p_decayed = if config.weight_decay != 0.0_f32 {
                (-lr_wd).mul_add(p_init, p_init)
            } else {
                p_init
            };
            let p_updated = p_decayed - step_val;

            if let Some(mp) = &mut mp_slice
                && has_master
            {
                mp[idx] = p_updated;
                param[idx] = quantize_value(p_updated, seed_p, offset, dtype_mode, false);
            } else {
                param[idx] = quantize_value(
                    p_updated,
                    seed_p,
                    offset,
                    dtype_mode,
                    config.stochastic_rounding,
                );
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dual_step_avx2(
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
    let coeffs =
        compute_bias_correction(step, beta1, beta2, config.eps, config.tau, view.dtype_mode);
    let bound_mode = config.effective_bound_type();
    let (seed_m, seed_p) = derive_kernel_seeds(config.seed, step, param_index);
    let state_dtype_mode = if view.state_dtype_mode == DTypeMode::Float16 {
        DTypeMode::BFloat16
    } else {
        view.state_dtype_mode
    };

    let use_cd = config.centered_defect && n_elements > 1;
    let diff_mean_scaled = if use_cd {
        let mut diff_sum = 0.0_f32;
        let mut start = 0;
        while start < n_elements {
            let end = (start + luminav_core::reference_cpu::ELEM_CHUNK_SIZE).min(n_elements);
            diff_sum += unsafe {
                compute_diff_sum_chunk_avx2(&view.grad[start..end], &view.exp_avg[start..end])
            };
            start = end;
        }
        view.scratch.set_diff_sum(diff_sum);
        (diff_sum / total_elements) * beta1
    } else {
        0.0_f32
    };

    let exp_avg_sq = view
        .exp_avg_sq
        .as_deref_mut()
        .ok_or(LuminaError::MissingSecondaryStateBuffer)?;

    if !config.cautious && bound_mode != BoundType::Radial {
        let mut start = 0;
        while start < n_elements {
            let end = (start + luminav_core::reference_cpu::ELEM_CHUNK_SIZE).min(n_elements);
            let mp_chunk = view
                .master_param
                .as_deref_mut()
                .map(|mp| &mut mp[start..end]);
            unsafe {
                dual_single_pass_chunk_avx2(
                    &mut view.param[start..end],
                    mp_chunk,
                    &view.grad[start..end],
                    &mut view.exp_avg[start..end],
                    &mut exp_avg_sq[start..end],
                    config,
                    &coeffs,
                    view.dtype_mode,
                    state_dtype_mode,
                    start,
                    diff_mean_scaled,
                    1.0_f32,
                    1.0_f32,
                    seed_m,
                    seed_p,
                    false,
                );
            }
            start = end;
        }
        view.scratch.reset_accumulators();
    } else if config.fused_single_pass && step > 1 {
        view.scratch.set_mask_sum(0.0);
        view.scratch.set_u_sq_sum(0.0);
        view.scratch.set_p_sq_sum(0.0);

        let cautious_scale = view.scratch.cautious_scale();
        let bound_scale = view.scratch.bound_scale();

        let mut total_mask: u64 = 0;
        let mut total_u_sq = 0.0_f32;
        let mut total_p_sq = 0.0_f32;

        let mut start = 0;
        while start < n_elements {
            let end = (start + luminav_core::reference_cpu::ELEM_CHUNK_SIZE).min(n_elements);
            let mp_chunk = view
                .master_param
                .as_deref_mut()
                .map(|mp| &mut mp[start..end]);
            let metrics = unsafe {
                dual_single_pass_chunk_avx2(
                    &mut view.param[start..end],
                    mp_chunk,
                    &view.grad[start..end],
                    &mut view.exp_avg[start..end],
                    &mut exp_avg_sq[start..end],
                    config,
                    &coeffs,
                    view.dtype_mode,
                    state_dtype_mode,
                    start,
                    diff_mean_scaled,
                    cautious_scale,
                    bound_scale,
                    seed_m,
                    seed_p,
                    true,
                )
            };
            total_mask += metrics.mask_count;
            total_u_sq += metrics.u_sq_sum;
            total_p_sq += metrics.p_sq_sum;
            start = end;
        }

        view.scratch.set_mask_sum(total_mask as f32);
        view.scratch.set_u_sq_sum(total_u_sq);
        view.scratch.set_p_sq_sum(total_p_sq);

        update_post_step_ema(view.scratch, total_elements, config);
    } else {
        view.scratch.set_mask_sum(0.0);
        view.scratch.set_u_sq_sum(0.0);
        view.scratch.set_p_sq_sum(0.0);

        let mut total_mask: u64 = 0;
        let mut total_u_sq = 0.0_f32;
        let mut total_p_sq = 0.0_f32;

        let mut start = 0;
        while start < n_elements {
            let end = (start + luminav_core::reference_cpu::ELEM_CHUNK_SIZE).min(n_elements);
            let mp_chunk = view.master_param.as_deref().map(|mp| &mp[start..end]);
            let metrics = unsafe {
                dual_pass1_chunk_avx2(
                    &view.param[start..end],
                    mp_chunk,
                    &view.grad[start..end],
                    &view.exp_avg[start..end],
                    &exp_avg_sq[start..end],
                    config,
                    &coeffs,
                    view.dtype_mode,
                    state_dtype_mode,
                    diff_mean_scaled,
                )
            };
            total_mask += metrics.mask_count;
            total_u_sq += metrics.u_sq_sum;
            total_p_sq += metrics.p_sq_sum;
            start = end;
        }

        let mask_sum = total_mask as f32;
        view.scratch.set_mask_sum(mask_sum);
        view.scratch.set_u_sq_sum(total_u_sq);
        view.scratch.set_p_sq_sum(total_p_sq);

        let cautious_scale = if config.cautious {
            compute_cautious_scale(mask_sum, total_elements, config.cautious_clamp_min)
        } else {
            1.0_f32
        };

        let bound_scale = if bound_mode == BoundType::Radial {
            compute_radial_bound_scale(
                total_u_sq,
                total_p_sq,
                cautious_scale,
                config.lr,
                config.bound_ratio,
            )
        } else {
            1.0_f32
        };

        let mut start = 0;
        while start < n_elements {
            let end = (start + luminav_core::reference_cpu::ELEM_CHUNK_SIZE).min(n_elements);
            let mp_chunk = view
                .master_param
                .as_deref_mut()
                .map(|mp| &mut mp[start..end]);
            unsafe {
                dual_pass2_chunk_avx2(
                    &mut view.param[start..end],
                    mp_chunk,
                    &view.grad[start..end],
                    &mut view.exp_avg[start..end],
                    &mut exp_avg_sq[start..end],
                    config,
                    &coeffs,
                    view.dtype_mode,
                    state_dtype_mode,
                    start,
                    diff_mean_scaled,
                    cautious_scale,
                    bound_scale,
                    seed_m,
                    seed_p,
                );
            }
            start = end;
        }

        view.scratch.reset_accumulators();
        view.scratch.set_cautious_scale(cautious_scale);
        view.scratch.set_bound_scale(bound_scale);
    }

    Ok(())
}
