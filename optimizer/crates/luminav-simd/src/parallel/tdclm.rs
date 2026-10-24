use crate::SimdBackend;
use crate::avx512::ChunkMetrics;
use luminav_core::{
    BoundType, BufferMode, DTypeMode, LuminaError, LuminaVConfig, TensorViewMut,
    apply_coordinate_bound, apply_fp16_safety_bound, compute_cautious_scale,
    compute_radial_bound_scale, derive_kernel_seeds, quantize_value, reduce_sum_16,
    sanitize_gradient, update_post_step_ema,
};
use rayon::prelude::*;

use super::{ELEM_CHUNK_SIZE, compute_diff_sum_chunk_dispatch};

#[allow(clippy::too_many_arguments)]
fn tdclm_single_pass_chunk_scalar(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    config: &LuminaVConfig,
    dtype_mode: DTypeMode,
    state_dtype_mode: DTypeMode,
    chunk_offset: usize,
    diff_mean: f32,
    cautious_scale: f32,
    bound_scale: f32,
    seed_m: i32,
    seed_p: i32,
) -> ChunkMetrics {
    let n = param.len();
    let has_master = config.master_weights.is_enabled() && master_param.is_some();
    let bound_mode = config.effective_bound_type();

    let beta1 = config.betas.0;
    let one_minus_beta1 = 1.0_f32 - beta1;
    let beta1_sq = beta1 * beta1;
    let one_minus_beta1_sq = 1.0_f32 - beta1_sq;
    let effective_eps = if dtype_mode == DTypeMode::Float16 {
        config.eps.max(1.0e-4_f32)
    } else {
        config.eps
    };
    let lr_wd = config.lr * config.weight_decay;

    let simd_len = n & !15;
    let mut mask_count: u64 = 0;
    let mut u_sq_lanes = [0.0_f32; 16];
    let mut p_sq_lanes = [0.0_f32; 16];
    let mut u_sq_tail = 0.0_f32;
    let mut p_sq_tail = 0.0_f32;

    let mut mp_slice = master_param;

    for idx in 0..n {
        let offset = (chunk_offset + idx) as i32;
        let p_init = match &mp_slice {
            Some(mp) if has_master => mp[idx],
            _ => param[idx],
        };
        let g = sanitize_gradient(grad[idx]);
        let m = exp_avg[idx];

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

        match &mut mp_slice {
            Some(mp) if has_master => {
                mp[idx] = p_updated;
                param[idx] = quantize_value(p_updated, seed_p, offset, dtype_mode, false);
            }
            _ => {
                param[idx] = quantize_value(
                    p_updated,
                    seed_p,
                    offset,
                    dtype_mode,
                    config.stochastic_rounding,
                );
            }
        }

        if config.cautious && mask_active {
            mask_count += 1;
        }
        if bound_mode == BoundType::Radial {
            let u_eff = if config.cautious { u * m_mask } else { u };
            let u_sq = u_eff * u_eff;
            let p_sq = p_init * p_init;
            if idx < simd_len {
                let lane = idx & 15;
                u_sq_lanes[lane] += u_sq;
                p_sq_lanes[lane] += p_sq;
            } else {
                u_sq_tail += u_sq;
                p_sq_tail += p_sq;
            }
        }
    }

    let (u_sq_sum, p_sq_sum) = if bound_mode == BoundType::Radial {
        (
            reduce_sum_16(&u_sq_lanes) + u_sq_tail,
            reduce_sum_16(&p_sq_lanes) + p_sq_tail,
        )
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
fn tdclm_pass1_chunk_scalar(
    param: &[f32],
    master_param: Option<&[f32]>,
    grad: &[f32],
    exp_avg: &[f32],
    config: &LuminaVConfig,
    dtype_mode: DTypeMode,
    diff_mean: f32,
) -> ChunkMetrics {
    let n = param.len();
    let has_master = config.master_weights.is_enabled() && master_param.is_some();
    let bound_mode = config.effective_bound_type();

    let beta1 = config.betas.0;
    let beta1_sq = beta1 * beta1;
    let one_minus_beta1_sq = 1.0_f32 - beta1_sq;
    let effective_eps = if dtype_mode == DTypeMode::Float16 {
        config.eps.max(1.0e-4_f32)
    } else {
        config.eps
    };

    let simd_len = n & !15;
    let mut mask_count: u64 = 0;
    let mut u_sq_lanes = [0.0_f32; 16];
    let mut p_sq_lanes = [0.0_f32; 16];
    let mut u_sq_tail = 0.0_f32;
    let mut p_sq_tail = 0.0_f32;

    for idx in 0..n {
        let g = sanitize_gradient(grad[idx]);
        let m = exp_avg[idx];
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
            mask_count += 1;
        }
        if bound_mode == BoundType::Radial {
            let u_eff = if config.cautious { u * m_mask } else { u };
            let p_val = match master_param {
                Some(mp) if has_master => mp[idx],
                _ => param[idx],
            };
            let u_sq = u_eff * u_eff;
            let p_sq = p_val * p_val;
            if idx < simd_len {
                let lane = idx & 15;
                u_sq_lanes[lane] += u_sq;
                p_sq_lanes[lane] += p_sq;
            } else {
                u_sq_tail += u_sq;
                p_sq_tail += p_sq;
            }
        }
    }

    let (u_sq_sum, p_sq_sum) = if bound_mode == BoundType::Radial {
        (
            reduce_sum_16(&u_sq_lanes) + u_sq_tail,
            reduce_sum_16(&p_sq_lanes) + p_sq_tail,
        )
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
fn tdclm_pass2_chunk_scalar(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    config: &LuminaVConfig,
    dtype_mode: DTypeMode,
    state_dtype_mode: DTypeMode,
    chunk_offset: usize,
    diff_mean: f32,
    cautious_scale: f32,
    bound_scale: f32,
    seed_m: i32,
    seed_p: i32,
) {
    let n = param.len();
    let has_master = config.master_weights.is_enabled() && master_param.is_some();
    let bound_mode = config.effective_bound_type();

    let beta1 = config.betas.0;
    let one_minus_beta1 = 1.0_f32 - beta1;
    let beta1_sq = beta1 * beta1;
    let one_minus_beta1_sq = 1.0_f32 - beta1_sq;
    let effective_eps = if dtype_mode == DTypeMode::Float16 {
        config.eps.max(1.0e-4_f32)
    } else {
        config.eps
    };
    let lr_wd = config.lr * config.weight_decay;

    let mut mp_slice = master_param;

    for idx in 0..n {
        let offset = (chunk_offset + idx) as i32;
        let p_init = match &mp_slice {
            Some(mp) if has_master => mp[idx],
            _ => param[idx],
        };
        let g = sanitize_gradient(grad[idx]);
        let m = exp_avg[idx];
        let diff_gm = (g - m).clamp(-1.0e18_f32, 1.0e18_f32);
        let diff = (diff_gm - diff_mean).clamp(-1.0e18_f32, 1.0e18_f32);

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

        match &mut mp_slice {
            Some(mp) if has_master => {
                mp[idx] = p_updated;
                param[idx] = quantize_value(p_updated, seed_p, offset, dtype_mode, false);
            }
            _ => {
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

#[allow(clippy::too_many_arguments)]
#[inline]
fn tdclm_single_pass_chunk_dispatch(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    config: &LuminaVConfig,
    dtype_mode: DTypeMode,
    state_dtype_mode: DTypeMode,
    chunk_offset: usize,
    diff_mean: f32,
    cautious_scale: f32,
    bound_scale: f32,
    seed_m: i32,
    seed_p: i32,
    backend: SimdBackend,
) -> ChunkMetrics {
    #[cfg(target_arch = "x86_64")]
    match backend {
        SimdBackend::Avx512 => unsafe {
            crate::avx512::tdclm_single_pass_chunk_avx512(
                param,
                master_param,
                grad,
                exp_avg,
                config,
                dtype_mode,
                state_dtype_mode,
                chunk_offset,
                diff_mean,
                cautious_scale,
                bound_scale,
                seed_m,
                seed_p,
            )
        },
        SimdBackend::Avx2 => unsafe {
            crate::avx2::tdclm_single_pass_chunk_avx2(
                param,
                master_param,
                grad,
                exp_avg,
                config,
                dtype_mode,
                state_dtype_mode,
                chunk_offset,
                diff_mean,
                cautious_scale,
                bound_scale,
                seed_m,
                seed_p,
            )
        },
        SimdBackend::Scalar => tdclm_single_pass_chunk_scalar(
            param,
            master_param,
            grad,
            exp_avg,
            config,
            dtype_mode,
            state_dtype_mode,
            chunk_offset,
            diff_mean,
            cautious_scale,
            bound_scale,
            seed_m,
            seed_p,
        ),
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = backend;
        tdclm_single_pass_chunk_scalar(
            param,
            master_param,
            grad,
            exp_avg,
            config,
            dtype_mode,
            state_dtype_mode,
            chunk_offset,
            diff_mean,
            cautious_scale,
            bound_scale,
            seed_m,
            seed_p,
        )
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn tdclm_pass1_chunk_dispatch(
    param: &[f32],
    master_param: Option<&[f32]>,
    grad: &[f32],
    exp_avg: &[f32],
    config: &LuminaVConfig,
    dtype_mode: DTypeMode,
    diff_mean: f32,
    backend: SimdBackend,
) -> ChunkMetrics {
    #[cfg(target_arch = "x86_64")]
    match backend {
        SimdBackend::Avx512 => unsafe {
            crate::avx512::tdclm_pass1_chunk_avx512(
                param,
                master_param,
                grad,
                exp_avg,
                config,
                dtype_mode,
                diff_mean,
            )
        },
        SimdBackend::Avx2 => unsafe {
            crate::avx2::tdclm_pass1_chunk_avx2(
                param,
                master_param,
                grad,
                exp_avg,
                config,
                dtype_mode,
                diff_mean,
            )
        },
        SimdBackend::Scalar => tdclm_pass1_chunk_scalar(
            param,
            master_param,
            grad,
            exp_avg,
            config,
            dtype_mode,
            diff_mean,
        ),
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = backend;
        tdclm_pass1_chunk_scalar(
            param,
            master_param,
            grad,
            exp_avg,
            config,
            dtype_mode,
            diff_mean,
        )
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn tdclm_pass2_chunk_dispatch(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    config: &LuminaVConfig,
    dtype_mode: DTypeMode,
    state_dtype_mode: DTypeMode,
    chunk_offset: usize,
    diff_mean: f32,
    cautious_scale: f32,
    bound_scale: f32,
    seed_m: i32,
    seed_p: i32,
    backend: SimdBackend,
) {
    #[cfg(target_arch = "x86_64")]
    match backend {
        SimdBackend::Avx512 => unsafe {
            crate::avx512::tdclm_pass2_chunk_avx512(
                param,
                master_param,
                grad,
                exp_avg,
                config,
                dtype_mode,
                state_dtype_mode,
                chunk_offset,
                diff_mean,
                cautious_scale,
                bound_scale,
                seed_m,
                seed_p,
            );
        },
        SimdBackend::Avx2 => unsafe {
            crate::avx2::tdclm_pass2_chunk_avx2(
                param,
                master_param,
                grad,
                exp_avg,
                config,
                dtype_mode,
                state_dtype_mode,
                chunk_offset,
                diff_mean,
                cautious_scale,
                bound_scale,
                seed_m,
                seed_p,
            );
        },
        SimdBackend::Scalar => {
            tdclm_pass2_chunk_scalar(
                param,
                master_param,
                grad,
                exp_avg,
                config,
                dtype_mode,
                state_dtype_mode,
                chunk_offset,
                diff_mean,
                cautious_scale,
                bound_scale,
                seed_m,
                seed_p,
            );
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = backend;
        tdclm_pass2_chunk_scalar(
            param,
            master_param,
            grad,
            exp_avg,
            config,
            dtype_mode,
            state_dtype_mode,
            chunk_offset,
            diff_mean,
            cautious_scale,
            bound_scale,
            seed_m,
            seed_p,
        );
    }
}

pub fn tdclm_step_parallel(
    view: &mut TensorViewMut<'_>,
    config: &LuminaVConfig,
    step: u64,
    param_index: usize,
    backend: SimdBackend,
) -> Result<(), LuminaError> {
    tdclm_step_parallel_chunked(view, config, step, param_index, backend, ELEM_CHUNK_SIZE)
}

pub fn tdclm_step_parallel_chunked(
    view: &mut TensorViewMut<'_>,
    config: &LuminaVConfig,
    step: u64,
    param_index: usize,
    backend: SimdBackend,
    chunk_size: usize,
) -> Result<(), LuminaError> {
    config.validate()?;
    let has_master = config.master_weights.is_enabled();
    let n_elements = view.validate(BufferMode::TdClm, has_master)?;
    if n_elements == 0 {
        return Ok(());
    }

    let eff_chunk = chunk_size.max(16) & !15;
    let total_elements = n_elements as f32;
    let bound_mode = config.effective_bound_type();
    let (seed_m, seed_p) = derive_kernel_seeds(config.seed, step, param_index);
    let dtype_mode = view.dtype_mode;
    let state_dtype_mode = view.state_dtype_mode;

    let use_cd = config.centered_defect && n_elements > 1;
    let diff_mean = if use_cd {
        let chunk_sums: Vec<f32> = view
            .grad
            .par_chunks(eff_chunk)
            .zip(view.exp_avg.par_chunks(eff_chunk))
            .map(|(g_chunk, m_chunk)| compute_diff_sum_chunk_dispatch(g_chunk, m_chunk, backend))
            .collect();
        let diff_sum: f32 = chunk_sums.into_iter().sum();
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

        let chunk_metrics: Vec<ChunkMetrics> = match view.master_param.as_deref_mut() {
            Some(mp) if has_master => (
                view.param.par_chunks_mut(eff_chunk),
                mp.par_chunks_mut(eff_chunk),
                view.grad.par_chunks(eff_chunk),
                view.exp_avg.par_chunks_mut(eff_chunk),
            )
                .into_par_iter()
                .enumerate()
                .map(|(chunk_idx, (p_c, mp_c, g_c, m_c))| {
                    tdclm_single_pass_chunk_dispatch(
                        p_c,
                        Some(mp_c),
                        g_c,
                        m_c,
                        config,
                        dtype_mode,
                        state_dtype_mode,
                        chunk_idx * eff_chunk,
                        diff_mean,
                        cautious_scale,
                        bound_scale,
                        seed_m,
                        seed_p,
                        backend,
                    )
                })
                .collect(),
            _ => (
                view.param.par_chunks_mut(eff_chunk),
                view.grad.par_chunks(eff_chunk),
                view.exp_avg.par_chunks_mut(eff_chunk),
            )
                .into_par_iter()
                .enumerate()
                .map(|(chunk_idx, (p_c, g_c, m_c))| {
                    tdclm_single_pass_chunk_dispatch(
                        p_c,
                        None,
                        g_c,
                        m_c,
                        config,
                        dtype_mode,
                        state_dtype_mode,
                        chunk_idx * eff_chunk,
                        diff_mean,
                        cautious_scale,
                        bound_scale,
                        seed_m,
                        seed_p,
                        backend,
                    )
                })
                .collect(),
        };

        let reduced = chunk_metrics
            .into_iter()
            .fold(ChunkMetrics::default(), ChunkMetrics::combine);

        view.scratch.set_mask_sum(reduced.mask_count as f32);
        view.scratch.set_u_sq_sum(reduced.u_sq_sum);
        view.scratch.set_p_sq_sum(reduced.p_sq_sum);

        if config.cautious || bound_mode == BoundType::Radial {
            update_post_step_ema(view.scratch, total_elements, config);
        } else {
            view.scratch.reset_accumulators();
        }
    } else {
        view.scratch.set_mask_sum(0.0);
        view.scratch.set_u_sq_sum(0.0);
        view.scratch.set_p_sq_sum(0.0);

        let chunk_metrics: Vec<ChunkMetrics> = match view.master_param.as_deref() {
            Some(mp) if has_master => (
                view.param.par_chunks(eff_chunk),
                mp.par_chunks(eff_chunk),
                view.grad.par_chunks(eff_chunk),
                view.exp_avg.par_chunks(eff_chunk),
            )
                .into_par_iter()
                .map(|(p_c, mp_c, g_c, m_c)| {
                    tdclm_pass1_chunk_dispatch(
                        p_c,
                        Some(mp_c),
                        g_c,
                        m_c,
                        config,
                        dtype_mode,
                        diff_mean,
                        backend,
                    )
                })
                .collect(),
            _ => (
                view.param.par_chunks(eff_chunk),
                view.grad.par_chunks(eff_chunk),
                view.exp_avg.par_chunks(eff_chunk),
            )
                .into_par_iter()
                .map(|(p_c, g_c, m_c)| {
                    tdclm_pass1_chunk_dispatch(
                        p_c, None, g_c, m_c, config, dtype_mode, diff_mean, backend,
                    )
                })
                .collect(),
        };

        let reduced = chunk_metrics
            .into_iter()
            .fold(ChunkMetrics::default(), ChunkMetrics::combine);
        let mask_sum = reduced.mask_count as f32;

        view.scratch.set_mask_sum(mask_sum);
        view.scratch.set_u_sq_sum(reduced.u_sq_sum);
        view.scratch.set_p_sq_sum(reduced.p_sq_sum);

        let cautious_scale = if config.cautious {
            compute_cautious_scale(mask_sum, total_elements, config.cautious_clamp_min)
        } else {
            1.0_f32
        };

        let bound_scale = if bound_mode == BoundType::Radial {
            compute_radial_bound_scale(
                reduced.u_sq_sum,
                reduced.p_sq_sum,
                cautious_scale,
                config.lr,
                config.bound_ratio,
            )
        } else {
            1.0_f32
        };

        match view.master_param.as_deref_mut() {
            Some(mp) if has_master => {
                (
                    view.param.par_chunks_mut(eff_chunk),
                    mp.par_chunks_mut(eff_chunk),
                    view.grad.par_chunks(eff_chunk),
                    view.exp_avg.par_chunks_mut(eff_chunk),
                )
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(chunk_idx, (p_c, mp_c, g_c, m_c))| {
                        tdclm_pass2_chunk_dispatch(
                            p_c,
                            Some(mp_c),
                            g_c,
                            m_c,
                            config,
                            dtype_mode,
                            state_dtype_mode,
                            chunk_idx * eff_chunk,
                            diff_mean,
                            cautious_scale,
                            bound_scale,
                            seed_m,
                            seed_p,
                            backend,
                        );
                    });
            }
            _ => {
                (
                    view.param.par_chunks_mut(eff_chunk),
                    view.grad.par_chunks(eff_chunk),
                    view.exp_avg.par_chunks_mut(eff_chunk),
                )
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(chunk_idx, (p_c, g_c, m_c))| {
                        tdclm_pass2_chunk_dispatch(
                            p_c,
                            None,
                            g_c,
                            m_c,
                            config,
                            dtype_mode,
                            state_dtype_mode,
                            chunk_idx * eff_chunk,
                            diff_mean,
                            cautious_scale,
                            bound_scale,
                            seed_m,
                            seed_p,
                            backend,
                        );
                    });
            }
        }

        view.scratch.reset_accumulators();
        view.scratch.set_cautious_scale(cautious_scale);
        view.scratch.set_bound_scale(bound_scale);
    }

    Ok(())
}
