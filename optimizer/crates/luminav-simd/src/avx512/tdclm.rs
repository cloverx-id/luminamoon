#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __mmask16, _CMP_GT_OQ, _mm512_abs_ps, _mm512_add_ps, _mm512_cmp_ps_mask, _mm512_div_ps,
    _mm512_fmadd_ps, _mm512_fnmadd_ps, _mm512_loadu_ps, _mm512_maskz_mov_ps, _mm512_mul_ps,
    _mm512_set1_epi32, _mm512_set1_ps, _mm512_setzero_ps, _mm512_storeu_ps, _mm512_sub_ps,
};

#[cfg(target_arch = "x86_64")]
use luminav_core::{
    BoundType, BufferMode, DTypeMode, LuminaError, LuminaVConfig, TensorViewMut,
    apply_coordinate_bound, apply_fp16_safety_bound, compute_cautious_scale,
    compute_radial_bound_scale, derive_kernel_seeds, quantize_value, sanitize_gradient,
    update_post_step_ema,
};

#[cfg(target_arch = "x86_64")]
use super::{
    ChunkMetrics, apply_coordinate_bound_avx512, apply_fp16_safety_bound_avx512, clamp_diff_avx512,
    offset_vector_avx512, quantize_avx512, reduce_add_ps_avx512, sanitize_gradient_avx512,
};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn tdclm_single_pass_chunk_avx512(
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
    let n = param.len();
    let simd_len = n & !15;
    let has_master = config.master_weights.is_enabled() && master_param.is_some();
    let bound_mode = config.effective_bound_type();

    let beta1 = config.betas.0;
    let one_minus_beta1 = 1.0_f32 - beta1;
    let beta1_sq = beta1 * beta1;
    let one_minus_beta1_sq = 1.0_f32 - beta1_sq;
    let effective_eps = config.eps;
    let lr_wd = config.lr * config.weight_decay;
    let state_dtype_mode = if state_dtype_mode == DTypeMode::Float16 {
        DTypeMode::BFloat16
    } else {
        state_dtype_mode
    };

    let v_beta1 = _mm512_set1_ps(beta1);
    let v_one_minus_beta1 = _mm512_set1_ps(one_minus_beta1);
    let v_beta1_sq = _mm512_set1_ps(beta1_sq);
    let v_one_minus_beta1_sq = _mm512_set1_ps(one_minus_beta1_sq);
    let v_lam = _mm512_set1_ps(config.lam);
    let v_eps = _mm512_set1_ps(effective_eps);
    let v_cautious_scale = _mm512_set1_ps(cautious_scale);
    let v_bound_scale = _mm512_set1_ps(bound_scale);
    let v_bound_ratio = _mm512_set1_ps(config.bound_ratio);
    let v_lr = _mm512_set1_ps(config.lr);
    let v_lr_wd = _mm512_set1_ps(lr_wd);
    let v_zero = _mm512_setzero_ps();
    let v_seed_m = _mm512_set1_epi32(seed_m);
    let v_seed_p = _mm512_set1_epi32(seed_p);

    let mut mask_count: u64 = 0;
    let mut v_u_sq_acc = _mm512_setzero_ps();
    let mut v_p_sq_acc = _mm512_setzero_ps();

    let mut mp_slice = master_param;

    let mut i = 0;
    while i < simd_len {
        let global_offset = (chunk_offset + i) as i32;
        let v_offsets = offset_vector_avx512(global_offset);

        let v_p_init = match &mp_slice {
            Some(mp) if has_master => _mm512_loadu_ps(mp.as_ptr().add(i)),
            _ => _mm512_loadu_ps(param.as_ptr().add(i)),
        };
        let v_g = sanitize_gradient_avx512(_mm512_loadu_ps(grad.as_ptr().add(i)));
        let v_m = _mm512_loadu_ps(exp_avg.as_ptr().add(i));

        let v_g_scaled = _mm512_mul_ps(v_one_minus_beta1, v_g);
        let v_m_new = _mm512_fmadd_ps(v_beta1, v_m, v_g_scaled);
        let v_clm_m = _mm512_fmadd_ps(v_beta1_sq, v_m, _mm512_mul_ps(v_one_minus_beta1_sq, v_g));
        let v_diff = clamp_diff_avx512(_mm512_sub_ps(v_g, v_m));

        let v_env0 = _mm512_fmadd_ps(v_beta1_sq, _mm512_abs_ps(v_m), v_eps);
        let v_env1 = _mm512_fmadd_ps(v_one_minus_beta1_sq, _mm512_abs_ps(v_g), v_env0);
        let v_clm_env = _mm512_fmadd_ps(v_lam, _mm512_abs_ps(v_diff), v_env1);
        let v_u = _mm512_div_ps(v_clm_m, v_clm_env);

        let k_mask: __mmask16 = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(_mm512_mul_ps(v_u, v_g), v_zero);
        let v_u_masked = _mm512_maskz_mov_ps(k_mask, v_u);

        let v_m_stored = quantize_avx512(
            v_m_new,
            v_seed_m,
            v_offsets,
            state_dtype_mode,
            config.stochastic_rounding,
        );
        _mm512_storeu_ps(exp_avg.as_mut_ptr().add(i), v_m_stored);

        let v_delta_theta = if config.cautious {
            _mm512_mul_ps(v_u_masked, v_cautious_scale)
        } else {
            v_u
        };
        let mut v_step = _mm512_mul_ps(v_lr, v_delta_theta);

        match bound_mode {
            BoundType::Radial => {
                v_step = _mm512_mul_ps(v_step, v_bound_scale);
            }
            BoundType::Coordinate => {
                v_step = apply_coordinate_bound_avx512(v_step, v_p_init, v_bound_ratio);
            }
            BoundType::None => {}
        }

        if dtype_mode == DTypeMode::Float16 {
            v_step = apply_fp16_safety_bound_avx512(v_step, v_p_init);
        }

        let v_p_decayed = if config.weight_decay != 0.0_f32 {
            _mm512_fnmadd_ps(v_lr_wd, v_p_init, v_p_init)
        } else {
            v_p_init
        };
        let v_p_updated = _mm512_sub_ps(v_p_decayed, v_step);

        if let Some(mp) = &mut mp_slice
            && has_master
        {
            _mm512_storeu_ps(mp.as_mut_ptr().add(i), v_p_updated);
            let v_p_q = quantize_avx512(v_p_updated, v_seed_p, v_offsets, dtype_mode, false);
            _mm512_storeu_ps(param.as_mut_ptr().add(i), v_p_q);
        } else {
            let v_p_q = quantize_avx512(
                v_p_updated,
                v_seed_p,
                v_offsets,
                dtype_mode,
                config.stochastic_rounding,
            );
            _mm512_storeu_ps(param.as_mut_ptr().add(i), v_p_q);
        }

        if config.cautious {
            mask_count += u64::from(k_mask.count_ones());
        }
        if bound_mode == BoundType::Radial {
            let v_u_eff = if config.cautious { v_u_masked } else { v_u };
            v_u_sq_acc = _mm512_add_ps(v_u_sq_acc, _mm512_mul_ps(v_u_eff, v_u_eff));
            v_p_sq_acc = _mm512_add_ps(v_p_sq_acc, _mm512_mul_ps(v_p_init, v_p_init));
        }

        i += 16;
    }

    let mut u_sq_sum = if bound_mode == BoundType::Radial {
        reduce_add_ps_avx512(v_u_sq_acc)
    } else {
        0.0_f32
    };
    let mut p_sq_sum = if bound_mode == BoundType::Radial {
        reduce_add_ps_avx512(v_p_sq_acc)
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

        let m_new = beta1.mul_add(m, one_minus_beta1 * g);
        let clm_m = beta1_sq.mul_add(m, one_minus_beta1_sq * g);
        let diff = (g - m).clamp(-1.0e18_f32, 1.0e18_f32);

        let env0 = beta1_sq.mul_add(m.abs(), effective_eps);
        let env1 = one_minus_beta1_sq.mul_add(g.abs(), env0);
        let clm_env = config.lam.mul_add(diff.abs(), env1);
        let u = clm_m / clm_env;
        let mask_active = (u * g) > 0.0_f32;
        let m_mask = if mask_active { 1.0_f32 } else { 0.0_f32 };

        exp_avg[idx] = quantize_value(
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

        if config.cautious && mask_active {
            mask_count += 1;
        }
        if bound_mode == BoundType::Radial {
            let u_eff = if config.cautious { u * m_mask } else { u };
            u_sq_sum += u_eff * u_eff;
            p_sq_sum += p_init * p_init;
        }
    }

    ChunkMetrics {
        mask_count,
        u_sq_sum,
        p_sq_sum,
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
pub unsafe fn tdclm_pass1_chunk_avx512(
    param: &[f32],
    master_param: Option<&[f32]>,
    grad: &[f32],
    exp_avg: &[f32],
    config: &LuminaVConfig,
    dtype_mode: DTypeMode,
) -> ChunkMetrics {
    let n = param.len();
    let simd_len = n & !15;
    let has_master = config.master_weights.is_enabled() && master_param.is_some();
    let bound_mode = config.effective_bound_type();

    let _ = dtype_mode;
    let beta1 = config.betas.0;
    let beta1_sq = beta1 * beta1;
    let one_minus_beta1_sq = 1.0_f32 - beta1_sq;
    let effective_eps = config.eps;

    let v_beta1_sq = _mm512_set1_ps(beta1_sq);
    let v_one_minus_beta1_sq = _mm512_set1_ps(one_minus_beta1_sq);
    let v_lam = _mm512_set1_ps(config.lam);
    let v_eps = _mm512_set1_ps(effective_eps);
    let v_zero = _mm512_setzero_ps();

    let mut mask_count: u64 = 0;
    let mut v_u_sq_acc = _mm512_setzero_ps();
    let mut v_p_sq_acc = _mm512_setzero_ps();

    let mut i = 0;
    while i < simd_len {
        let v_g = sanitize_gradient_avx512(_mm512_loadu_ps(grad.as_ptr().add(i)));
        let v_m = _mm512_loadu_ps(exp_avg.as_ptr().add(i));
        let v_diff = clamp_diff_avx512(_mm512_sub_ps(v_g, v_m));

        let v_clm_m = _mm512_fmadd_ps(v_beta1_sq, v_m, _mm512_mul_ps(v_one_minus_beta1_sq, v_g));
        let v_env0 = _mm512_fmadd_ps(v_beta1_sq, _mm512_abs_ps(v_m), v_eps);
        let v_env1 = _mm512_fmadd_ps(v_one_minus_beta1_sq, _mm512_abs_ps(v_g), v_env0);
        let v_clm_env = _mm512_fmadd_ps(v_lam, _mm512_abs_ps(v_diff), v_env1);
        let v_u = _mm512_div_ps(v_clm_m, v_clm_env);

        let k_mask: __mmask16 = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(_mm512_mul_ps(v_u, v_g), v_zero);

        if config.cautious {
            mask_count += u64::from(k_mask.count_ones());
        }
        if bound_mode == BoundType::Radial {
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

    let mut u_sq_sum = if bound_mode == BoundType::Radial {
        reduce_add_ps_avx512(v_u_sq_acc)
    } else {
        0.0_f32
    };
    let mut p_sq_sum = if bound_mode == BoundType::Radial {
        reduce_add_ps_avx512(v_p_sq_acc)
    } else {
        0.0_f32
    };

    for idx in simd_len..n {
        let g = sanitize_gradient(grad[idx]);
        let m = exp_avg[idx];
        let diff = (g - m).clamp(-1.0e18_f32, 1.0e18_f32);

        let clm_m = beta1_sq.mul_add(m, one_minus_beta1_sq * g);
        let env0 = beta1_sq.mul_add(m.abs(), effective_eps);
        let env1 = one_minus_beta1_sq.mul_add(g.abs(), env0);
        let clm_env = config.lam.mul_add(diff.abs(), env1);
        let u = clm_m / clm_env;
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn tdclm_pass2_chunk_avx512(
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
    let n = param.len();
    let simd_len = n & !15;
    let has_master = config.master_weights.is_enabled() && master_param.is_some();
    let bound_mode = config.effective_bound_type();

    let beta1 = config.betas.0;
    let one_minus_beta1 = 1.0_f32 - beta1;
    let beta1_sq = beta1 * beta1;
    let one_minus_beta1_sq = 1.0_f32 - beta1_sq;
    let effective_eps = config.eps;
    let lr_wd = config.lr * config.weight_decay;
    let state_dtype_mode = if state_dtype_mode == DTypeMode::Float16 {
        DTypeMode::BFloat16
    } else {
        state_dtype_mode
    };

    let v_beta1 = _mm512_set1_ps(beta1);
    let v_one_minus_beta1 = _mm512_set1_ps(one_minus_beta1);
    let v_beta1_sq = _mm512_set1_ps(beta1_sq);
    let v_one_minus_beta1_sq = _mm512_set1_ps(one_minus_beta1_sq);
    let v_lam = _mm512_set1_ps(config.lam);
    let v_eps = _mm512_set1_ps(effective_eps);
    let v_cautious_scale = _mm512_set1_ps(cautious_scale);
    let v_bound_scale = _mm512_set1_ps(bound_scale);
    let v_bound_ratio = _mm512_set1_ps(config.bound_ratio);
    let v_lr = _mm512_set1_ps(config.lr);
    let v_lr_wd = _mm512_set1_ps(lr_wd);
    let v_zero = _mm512_setzero_ps();
    let v_seed_m = _mm512_set1_epi32(seed_m);
    let v_seed_p = _mm512_set1_epi32(seed_p);

    let mut mp_slice = master_param;

    let mut i = 0;
    while i < simd_len {
        let global_offset = (chunk_offset + i) as i32;
        let v_offsets = offset_vector_avx512(global_offset);

        let v_p_init = match &mp_slice {
            Some(mp) if has_master => _mm512_loadu_ps(mp.as_ptr().add(i)),
            _ => _mm512_loadu_ps(param.as_ptr().add(i)),
        };
        let v_g = sanitize_gradient_avx512(_mm512_loadu_ps(grad.as_ptr().add(i)));
        let v_m = _mm512_loadu_ps(exp_avg.as_ptr().add(i));
        let v_diff = clamp_diff_avx512(_mm512_sub_ps(v_g, v_m));

        let v_m_new = _mm512_fmadd_ps(v_beta1, v_m, _mm512_mul_ps(v_one_minus_beta1, v_g));
        let v_clm_m = _mm512_fmadd_ps(v_beta1_sq, v_m, _mm512_mul_ps(v_one_minus_beta1_sq, v_g));
        let v_env0 = _mm512_fmadd_ps(v_beta1_sq, _mm512_abs_ps(v_m), v_eps);
        let v_env1 = _mm512_fmadd_ps(v_one_minus_beta1_sq, _mm512_abs_ps(v_g), v_env0);
        let v_clm_env = _mm512_fmadd_ps(v_lam, _mm512_abs_ps(v_diff), v_env1);
        let v_u = _mm512_div_ps(v_clm_m, v_clm_env);

        let k_mask: __mmask16 = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(_mm512_mul_ps(v_u, v_g), v_zero);
        let v_u_masked = _mm512_maskz_mov_ps(k_mask, v_u);

        let v_m_stored = quantize_avx512(
            v_m_new,
            v_seed_m,
            v_offsets,
            state_dtype_mode,
            config.stochastic_rounding,
        );
        _mm512_storeu_ps(exp_avg.as_mut_ptr().add(i), v_m_stored);

        let v_delta_theta = if config.cautious {
            _mm512_mul_ps(v_u_masked, v_cautious_scale)
        } else {
            v_u
        };
        let mut v_step = _mm512_mul_ps(v_lr, v_delta_theta);

        match bound_mode {
            BoundType::Radial => {
                v_step = _mm512_mul_ps(v_step, v_bound_scale);
            }
            BoundType::Coordinate => {
                v_step = apply_coordinate_bound_avx512(v_step, v_p_init, v_bound_ratio);
            }
            BoundType::None => {}
        }

        if dtype_mode == DTypeMode::Float16 {
            v_step = apply_fp16_safety_bound_avx512(v_step, v_p_init);
        }

        let v_p_decayed = if config.weight_decay != 0.0_f32 {
            _mm512_fnmadd_ps(v_lr_wd, v_p_init, v_p_init)
        } else {
            v_p_init
        };
        let v_p_updated = _mm512_sub_ps(v_p_decayed, v_step);

        if let Some(mp) = &mut mp_slice
            && has_master
        {
            _mm512_storeu_ps(mp.as_mut_ptr().add(i), v_p_updated);
            let v_p_q = quantize_avx512(v_p_updated, v_seed_p, v_offsets, dtype_mode, false);
            _mm512_storeu_ps(param.as_mut_ptr().add(i), v_p_q);
        } else {
            let v_p_q = quantize_avx512(
                v_p_updated,
                v_seed_p,
                v_offsets,
                dtype_mode,
                config.stochastic_rounding,
            );
            _mm512_storeu_ps(param.as_mut_ptr().add(i), v_p_q);
        }

        i += 16;
    }

    for idx in simd_len..n {
        let offset = (chunk_offset + idx) as i32;
        let p_init = match &mp_slice {
            Some(mp) if has_master => mp[idx],
            _ => param[idx],
        };
        let g = sanitize_gradient(grad[idx]);
        let m = exp_avg[idx];
        let diff = (g - m).clamp(-1.0e18_f32, 1.0e18_f32);

        let m_new = beta1.mul_add(m, one_minus_beta1 * g);
        let clm_m = beta1_sq.mul_add(m, one_minus_beta1_sq * g);
        let env0 = beta1_sq.mul_add(m.abs(), effective_eps);
        let env1 = one_minus_beta1_sq.mul_add(g.abs(), env0);
        let clm_env = config.lam.mul_add(diff.abs(), env1);
        let u = clm_m / clm_env;
        let m_mask = if (u * g) > 0.0_f32 { 1.0_f32 } else { 0.0_f32 };

        exp_avg[idx] = quantize_value(
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma")]
pub unsafe fn tdclm_step_avx512(
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
    let bound_mode = config.effective_bound_type();
    let (seed_m, seed_p) = derive_kernel_seeds(config.seed, step, param_index);
    let state_dtype_mode = if view.state_dtype_mode == DTypeMode::Float16 {
        DTypeMode::BFloat16
    } else {
        view.state_dtype_mode
    };

    let use_single_pass = (config.fused_single_pass && step > 1)
        || (!config.cautious && bound_mode != BoundType::Radial);

    if use_single_pass {
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
            let metrics = tdclm_single_pass_chunk_avx512(
                &mut view.param[start..end],
                mp_chunk,
                &view.grad[start..end],
                &mut view.exp_avg[start..end],
                config,
                view.dtype_mode,
                state_dtype_mode,
                start,
                cautious_scale,
                bound_scale,
                seed_m,
                seed_p,
            );
            total_mask += metrics.mask_count;
            total_u_sq += metrics.u_sq_sum;
            total_p_sq += metrics.p_sq_sum;
            start = end;
        }

        view.scratch.set_mask_sum(total_mask as f32);
        view.scratch.set_u_sq_sum(total_u_sq);
        view.scratch.set_p_sq_sum(total_p_sq);

        if config.cautious || bound_mode == BoundType::Radial {
            update_post_step_ema(view.scratch, total_elements, config);
        } else {
            view.scratch.reset_accumulators();
        }
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
            let metrics = tdclm_pass1_chunk_avx512(
                &view.param[start..end],
                mp_chunk,
                &view.grad[start..end],
                &view.exp_avg[start..end],
                config,
                view.dtype_mode,
            );
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
            tdclm_pass2_chunk_avx512(
                &mut view.param[start..end],
                mp_chunk,
                &view.grad[start..end],
                &mut view.exp_avg[start..end],
                config,
                view.dtype_mode,
                state_dtype_mode,
                start,
                cautious_scale,
                bound_scale,
                seed_m,
                seed_p,
            );
            start = end;
        }

        view.scratch.reset_accumulators();
        view.scratch.set_cautious_scale(cautious_scale);
        view.scratch.set_bound_scale(bound_scale);
    }

    Ok(())
}
