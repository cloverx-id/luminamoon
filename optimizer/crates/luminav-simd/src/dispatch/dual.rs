use luminav_core::{
    BiasCorrectionCoeffs, BoundType, BufferMode, DTypeMode, LuminaError, LuminaVConfig,
    TensorViewMut, compute_bias_correction, derive_spatial_seeds, derive_weyl_step_offsets,
    update_post_step_ema,
};

use crate::SimdBackend;
use crate::common::{ChunkMetrics, DualChunkCtx};
use crate::dispatch::resolve_pass2_scales;
use crate::scalar::dual::{dual_pass1_range, dual_update_range};

#[inline]
pub(crate) fn dual_single_pass_chunk_dispatch(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    exp_avg_sq: &mut [f32],
    ctx: &DualChunkCtx<'_>,
    backend: SimdBackend,
) -> ChunkMetrics {
    #[cfg(target_arch = "x86_64")]
    match backend {
        SimdBackend::Avx512 => {
            return unsafe {
                crate::avx512::dual::dual_single_pass_ctx_avx512(
                    param,
                    master_param,
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    ctx,
                )
            };
        }
        SimdBackend::Avx2 => {
            return unsafe {
                crate::avx2::dual::dual_single_pass_ctx_avx2(
                    param,
                    master_param,
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    ctx,
                )
            };
        }
        SimdBackend::Neon | SimdBackend::Scalar => {}
    }

    #[cfg(target_arch = "aarch64")]
    if backend == SimdBackend::Neon {
        return unsafe {
            crate::neon::dual::dual_single_pass_ctx_neon(
                param,
                master_param,
                grad,
                exp_avg,
                exp_avg_sq,
                ctx,
            )
        };
    }

    let _ = backend;
    dual_update_range::<true, true, false>(param, master_param, grad, exp_avg, exp_avg_sq, ctx, 0)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn dual_pass1_chunk_dispatch(
    param: &[f32],
    master_param: Option<&[f32]>,
    grad: &[f32],
    exp_avg: &[f32],
    exp_avg_sq: &[f32],
    config: &LuminaVConfig,
    coeffs: &BiasCorrectionCoeffs,
    dtype_mode: DTypeMode,
    state_dtype_mode: DTypeMode,
    backend: SimdBackend,
) -> ChunkMetrics {
    #[cfg(target_arch = "x86_64")]
    match backend {
        SimdBackend::Avx512 => {
            return unsafe {
                crate::avx512::dual_pass1_chunk_avx512(
                    param,
                    master_param,
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    config,
                    coeffs,
                    dtype_mode,
                    state_dtype_mode,
                )
            };
        }
        SimdBackend::Avx2 => {
            return unsafe {
                crate::avx2::dual_pass1_chunk_avx2(
                    param,
                    master_param,
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    config,
                    coeffs,
                    dtype_mode,
                    state_dtype_mode,
                )
            };
        }
        SimdBackend::Neon | SimdBackend::Scalar => {}
    }

    #[cfg(target_arch = "aarch64")]
    if backend == SimdBackend::Neon {
        return unsafe {
            crate::neon::dual_pass1_chunk_neon(
                param,
                master_param,
                grad,
                exp_avg,
                exp_avg_sq,
                config,
                coeffs,
                dtype_mode,
                state_dtype_mode,
            )
        };
    }

    let _ = backend;
    dual_pass1_range::<true>(
        param,
        master_param,
        grad,
        exp_avg,
        exp_avg_sq,
        config,
        coeffs,
        dtype_mode,
        state_dtype_mode,
        0,
    )
}

#[inline]
pub(crate) fn dual_pass2_chunk_dispatch(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    exp_avg_sq: &mut [f32],
    ctx: &DualChunkCtx<'_>,
    backend: SimdBackend,
) {
    #[cfg(target_arch = "x86_64")]
    match backend {
        SimdBackend::Avx512 => {
            unsafe {
                crate::avx512::dual::dual_pass2_ctx_avx512(
                    param,
                    master_param,
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    ctx,
                );
            }
            return;
        }
        SimdBackend::Avx2 => {
            unsafe {
                crate::avx2::dual::dual_pass2_ctx_avx2(
                    param,
                    master_param,
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    ctx,
                );
            }
            return;
        }
        SimdBackend::Neon | SimdBackend::Scalar => {}
    }

    #[cfg(target_arch = "aarch64")]
    if backend == SimdBackend::Neon {
        unsafe {
            crate::neon::dual::dual_pass2_ctx_neon(
                param,
                master_param,
                grad,
                exp_avg,
                exp_avg_sq,
                ctx,
            );
        }
        return;
    }

    let _ = backend;
    dual_update_range::<false, false, true>(param, master_param, grad, exp_avg, exp_avg_sq, ctx, 0);
}

pub(crate) fn dual_step_sequential(
    view: &mut TensorViewMut<'_>,
    config: &LuminaVConfig,
    step: u64,
    param_index: usize,
    backend: SimdBackend,
) -> Result<(), LuminaError> {
    config.validate()?;
    let has_master = config.master_weights.is_enabled();
    let target_mode = if config.buffer.is_triangle_family() {
        BufferMode::T2
    } else {
        BufferMode::Dual
    };
    let n_elements = view.validate(target_mode, has_master)?;
    if n_elements == 0 {
        return Ok(());
    }

    let total_elements = n_elements as f32;
    let (beta1, beta2) = config.betas;
    let coeffs =
        compute_bias_correction(step, beta1, beta2, config.eps, config.tau, view.dtype_mode);
    let bound_mode = config.effective_bound_type();
    let (seed_m, seed_p) = derive_spatial_seeds(config.seed, param_index);
    let (weyl_m, weyl_v, weyl_p) = derive_weyl_step_offsets(step);
    let state_dtype_mode = if view.state_dtype_mode == DTypeMode::Float16 {
        DTypeMode::BFloat16
    } else {
        view.state_dtype_mode
    };
    let chunk_size = luminav_core::reference_cpu::ELEM_CHUNK_SIZE;
    let exp_avg_sq = view
        .exp_avg_sq
        .as_deref_mut()
        .ok_or(LuminaError::MissingSecondaryStateBuffer)?;

    if !config.cautious && bound_mode != BoundType::Radial {
        let mut start = 0;
        while start < n_elements {
            let end = (start + chunk_size).min(n_elements);
            let ctx = DualChunkCtx {
                config,
                coeffs: &coeffs,
                dtype_mode: view.dtype_mode,
                state_dtype_mode,
                chunk_offset: start,
                cautious_scale: 1.0_f32,
                bound_scale: 1.0_f32,
                seed_m,
                seed_p,
                weyl_m,
                weyl_v,
                weyl_p,
                accumulate_metrics: false,
            };
            let mp_chunk = view
                .master_param
                .as_deref_mut()
                .map(|mp| &mut mp[start..end]);
            dual_single_pass_chunk_dispatch(
                &mut view.param[start..end],
                mp_chunk,
                &view.grad[start..end],
                &mut view.exp_avg[start..end],
                &mut exp_avg_sq[start..end],
                &ctx,
                backend,
            );
            start = end;
        }
        view.scratch.reset_accumulators();
    } else if config.fused_single_pass && step > 1 {
        view.scratch.reset_accumulators();
        let cautious_scale = view.scratch.cautious_scale();
        let bound_scale = view.scratch.bound_scale();
        let mut total = ChunkMetrics::default();
        let mut start = 0;
        while start < n_elements {
            let end = (start + chunk_size).min(n_elements);
            let ctx = DualChunkCtx {
                config,
                coeffs: &coeffs,
                dtype_mode: view.dtype_mode,
                state_dtype_mode,
                chunk_offset: start,
                cautious_scale,
                bound_scale,
                seed_m,
                seed_p,
                weyl_m,
                weyl_v,
                weyl_p,
                accumulate_metrics: true,
            };
            let mp_chunk = view
                .master_param
                .as_deref_mut()
                .map(|mp| &mut mp[start..end]);
            let metrics = dual_single_pass_chunk_dispatch(
                &mut view.param[start..end],
                mp_chunk,
                &view.grad[start..end],
                &mut view.exp_avg[start..end],
                &mut exp_avg_sq[start..end],
                &ctx,
                backend,
            );
            total = total.combine(metrics);
            start = end;
        }
        view.scratch.set_mask_sum(total.mask_count as f32);
        view.scratch.set_u_sq_sum(total.u_sq_sum);
        view.scratch.set_p_sq_sum(total.p_sq_sum);
        update_post_step_ema(view.scratch, total_elements, config);
    } else {
        view.scratch.reset_accumulators();
        let mut total = ChunkMetrics::default();
        let mut start = 0;
        while start < n_elements {
            let end = (start + chunk_size).min(n_elements);
            let mp_chunk = view.master_param.as_deref().map(|mp| &mp[start..end]);
            let metrics = dual_pass1_chunk_dispatch(
                &view.param[start..end],
                mp_chunk,
                &view.grad[start..end],
                &view.exp_avg[start..end],
                &exp_avg_sq[start..end],
                config,
                &coeffs,
                view.dtype_mode,
                state_dtype_mode,
                backend,
            );
            total = total.combine(metrics);
            start = end;
        }

        view.scratch.set_mask_sum(total.mask_count as f32);
        view.scratch.set_u_sq_sum(total.u_sq_sum);
        view.scratch.set_p_sq_sum(total.p_sq_sum);
        let (cautious_scale, bound_scale) =
            resolve_pass2_scales(total, total_elements, bound_mode, config);

        let mut start = 0;
        while start < n_elements {
            let end = (start + chunk_size).min(n_elements);
            let ctx = DualChunkCtx {
                config,
                coeffs: &coeffs,
                dtype_mode: view.dtype_mode,
                state_dtype_mode,
                chunk_offset: start,
                cautious_scale,
                bound_scale,
                seed_m,
                seed_p,
                weyl_m,
                weyl_v,
                weyl_p,
                accumulate_metrics: false,
            };
            let mp_chunk = view
                .master_param
                .as_deref_mut()
                .map(|mp| &mut mp[start..end]);
            dual_pass2_chunk_dispatch(
                &mut view.param[start..end],
                mp_chunk,
                &view.grad[start..end],
                &mut view.exp_avg[start..end],
                &mut exp_avg_sq[start..end],
                &ctx,
                backend,
            );
            start = end;
        }

        view.scratch.reset_accumulators();
        view.scratch.set_cautious_scale(cautious_scale);
        view.scratch.set_bound_scale(bound_scale);
    }

    Ok(())
}
