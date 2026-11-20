use luminav_core::{
    BoundType, BufferMode, DTypeMode, LuminaError, LuminaVConfig, TensorViewMut,
    derive_spatial_seeds, derive_weyl_step_offsets, update_post_step_ema,
};

use crate::SimdBackend;
use crate::common::{ChunkMetrics, TdclmChunkCtx};
use crate::dispatch::resolve_pass2_scales;
use crate::scalar::tdclm::{tdclm_pass1_range, tdclm_update_range};

#[inline]
pub(crate) fn tdclm_single_pass_chunk_dispatch(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    ctx: &TdclmChunkCtx<'_>,
    backend: SimdBackend,
) -> ChunkMetrics {
    #[cfg(target_arch = "x86_64")]
    match backend {
        SimdBackend::Avx512 => {
            return unsafe {
                crate::avx512::tdclm::tdclm_single_pass_ctx_avx512(
                    param,
                    master_param,
                    grad,
                    exp_avg,
                    ctx,
                )
            };
        }
        SimdBackend::Avx2 => {
            return unsafe {
                crate::avx2::tdclm::tdclm_single_pass_ctx_avx2(
                    param,
                    master_param,
                    grad,
                    exp_avg,
                    ctx,
                )
            };
        }
        SimdBackend::Neon | SimdBackend::Scalar => {}
    }

    #[cfg(target_arch = "aarch64")]
    if backend == SimdBackend::Neon {
        return unsafe {
            crate::neon::tdclm::tdclm_single_pass_ctx_neon(param, master_param, grad, exp_avg, ctx)
        };
    }

    let _ = backend;
    tdclm_update_range::<true, true>(param, master_param, grad, exp_avg, ctx, 0)
}

#[inline]
pub(crate) fn tdclm_pass1_chunk_dispatch(
    param: &[f32],
    master_param: Option<&[f32]>,
    grad: &[f32],
    exp_avg: &[f32],
    config: &LuminaVConfig,
    dtype_mode: DTypeMode,
    backend: SimdBackend,
) -> ChunkMetrics {
    #[cfg(target_arch = "x86_64")]
    match backend {
        SimdBackend::Avx512 => {
            return unsafe {
                crate::avx512::tdclm_pass1_chunk_avx512(
                    param,
                    master_param,
                    grad,
                    exp_avg,
                    config,
                    dtype_mode,
                )
            };
        }
        SimdBackend::Avx2 => {
            return unsafe {
                crate::avx2::tdclm_pass1_chunk_avx2(
                    param,
                    master_param,
                    grad,
                    exp_avg,
                    config,
                    dtype_mode,
                )
            };
        }
        SimdBackend::Neon | SimdBackend::Scalar => {}
    }

    #[cfg(target_arch = "aarch64")]
    if backend == SimdBackend::Neon {
        return unsafe {
            crate::neon::tdclm_pass1_chunk_neon(
                param,
                master_param,
                grad,
                exp_avg,
                config,
                dtype_mode,
            )
        };
    }

    let _ = (dtype_mode, backend);
    tdclm_pass1_range::<true>(param, master_param, grad, exp_avg, config, 0)
}

#[inline]
pub(crate) fn tdclm_pass2_chunk_dispatch(
    param: &mut [f32],
    master_param: Option<&mut [f32]>,
    grad: &[f32],
    exp_avg: &mut [f32],
    ctx: &TdclmChunkCtx<'_>,
    backend: SimdBackend,
) {
    #[cfg(target_arch = "x86_64")]
    match backend {
        SimdBackend::Avx512 => {
            unsafe {
                crate::avx512::tdclm::tdclm_pass2_ctx_avx512(
                    param,
                    master_param,
                    grad,
                    exp_avg,
                    ctx,
                );
            }
            return;
        }
        SimdBackend::Avx2 => {
            unsafe {
                crate::avx2::tdclm::tdclm_pass2_ctx_avx2(param, master_param, grad, exp_avg, ctx);
            }
            return;
        }
        SimdBackend::Neon | SimdBackend::Scalar => {}
    }

    #[cfg(target_arch = "aarch64")]
    if backend == SimdBackend::Neon {
        unsafe {
            crate::neon::tdclm::tdclm_pass2_ctx_neon(param, master_param, grad, exp_avg, ctx);
        }
        return;
    }

    let _ = backend;
    tdclm_update_range::<false, false>(param, master_param, grad, exp_avg, ctx, 0);
}

pub(crate) fn tdclm_step_sequential(
    view: &mut TensorViewMut<'_>,
    config: &LuminaVConfig,
    step: u64,
    param_index: usize,
    backend: SimdBackend,
) -> Result<(), LuminaError> {
    config.validate()?;
    let has_master = config.master_weights.is_enabled();
    let target_mode = if config.buffer.is_triangle_family() {
        BufferMode::T1
    } else {
        BufferMode::TdClm
    };
    let n_elements = view.validate(target_mode, has_master)?;
    if n_elements == 0 {
        return Ok(());
    }

    let total_elements = n_elements as f32;
    let bound_mode = config.effective_bound_type();
    let (seed_m, seed_p) = derive_spatial_seeds(config.seed, param_index);
    let (weyl_m, _, weyl_p) = derive_weyl_step_offsets(step);
    let state_dtype_mode = if view.state_dtype_mode == DTypeMode::Float16 {
        DTypeMode::BFloat16
    } else {
        view.state_dtype_mode
    };
    let chunk_size = luminav_core::reference_cpu::ELEM_CHUNK_SIZE;
    let use_single_pass = (config.fused_single_pass && step > 1)
        || (!config.cautious && bound_mode != BoundType::Radial);

    if use_single_pass {
        view.scratch.reset_accumulators();
        let cautious_scale = view.scratch.cautious_scale();
        let bound_scale = view.scratch.bound_scale();
        let mut total = ChunkMetrics::default();
        let mut start = 0;
        while start < n_elements {
            let end = (start + chunk_size).min(n_elements);
            let ctx = TdclmChunkCtx {
                config,
                dtype_mode: view.dtype_mode,
                state_dtype_mode,
                chunk_offset: start,
                cautious_scale,
                bound_scale,
                seed_m,
                seed_p,
                weyl_m,
                weyl_p,
            };
            let mp_chunk = view
                .master_param
                .as_deref_mut()
                .map(|mp| &mut mp[start..end]);
            let metrics = tdclm_single_pass_chunk_dispatch(
                &mut view.param[start..end],
                mp_chunk,
                &view.grad[start..end],
                &mut view.exp_avg[start..end],
                &ctx,
                backend,
            );
            total = total.combine(metrics);
            start = end;
        }
        view.scratch.set_mask_sum(total.mask_count as f32);
        view.scratch.set_u_sq_sum(total.u_sq_sum);
        view.scratch.set_p_sq_sum(total.p_sq_sum);
        if config.cautious || bound_mode == BoundType::Radial {
            update_post_step_ema(view.scratch, total_elements, config);
        } else {
            view.scratch.reset_accumulators();
        }
    } else {
        view.scratch.reset_accumulators();
        let mut total = ChunkMetrics::default();
        let mut start = 0;
        while start < n_elements {
            let end = (start + chunk_size).min(n_elements);
            let mp_chunk = view.master_param.as_deref().map(|mp| &mp[start..end]);
            let metrics = tdclm_pass1_chunk_dispatch(
                &view.param[start..end],
                mp_chunk,
                &view.grad[start..end],
                &view.exp_avg[start..end],
                config,
                view.dtype_mode,
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
            let ctx = TdclmChunkCtx {
                config,
                dtype_mode: view.dtype_mode,
                state_dtype_mode,
                chunk_offset: start,
                cautious_scale,
                bound_scale,
                seed_m,
                seed_p,
                weyl_m,
                weyl_p,
            };
            let mp_chunk = view
                .master_param
                .as_deref_mut()
                .map(|mp| &mut mp[start..end]);
            tdclm_pass2_chunk_dispatch(
                &mut view.param[start..end],
                mp_chunk,
                &view.grad[start..end],
                &mut view.exp_avg[start..end],
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
