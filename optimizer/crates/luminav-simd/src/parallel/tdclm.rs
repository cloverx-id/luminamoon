use luminav_core::{
    BoundType, BufferMode, DTypeMode, LuminaError, LuminaVConfig, TensorViewMut,
    derive_kernel_seeds, update_post_step_ema,
};
use rayon::prelude::*;

use crate::SimdBackend;
use crate::common::{ChunkMetrics, TdclmChunkCtx};
use crate::dispatch::resolve_pass2_scales;
use crate::dispatch::tdclm::{
    tdclm_pass1_chunk_dispatch, tdclm_pass2_chunk_dispatch, tdclm_single_pass_chunk_dispatch,
};

use super::ELEM_CHUNK_SIZE;

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
    let target_mode = if config.buffer.is_triangle_family() {
        BufferMode::T1
    } else {
        BufferMode::TdClm
    };
    let n_elements = view.validate(target_mode, has_master)?;
    if n_elements == 0 {
        return Ok(());
    }

    let eff_chunk = chunk_size.max(16) & !15;
    let total_elements = n_elements as f32;
    let bound_mode = config.effective_bound_type();
    let (seed_m, seed_p) = derive_kernel_seeds(config.seed, step, param_index);
    let dtype_mode = view.dtype_mode;
    let state_dtype_mode = if view.state_dtype_mode == DTypeMode::Float16 {
        DTypeMode::BFloat16
    } else {
        view.state_dtype_mode
    };

    let make_ctx = |chunk_idx: usize, cautious_scale: f32, bound_scale: f32| TdclmChunkCtx {
        config,
        dtype_mode,
        state_dtype_mode,
        chunk_offset: chunk_idx * eff_chunk,
        cautious_scale,
        bound_scale,
        seed_m,
        seed_p,
    };

    let use_single_pass = (config.fused_single_pass && step > 1)
        || (!config.cautious && bound_mode != BoundType::Radial);

    if use_single_pass {
        view.scratch.reset_accumulators();
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
                .map(|(ci, (p, mp, g, m))| {
                    let ctx = make_ctx(ci, cautious_scale, bound_scale);
                    tdclm_single_pass_chunk_dispatch(p, Some(mp), g, m, &ctx, backend)
                })
                .collect(),
            _ => (
                view.param.par_chunks_mut(eff_chunk),
                view.grad.par_chunks(eff_chunk),
                view.exp_avg.par_chunks_mut(eff_chunk),
            )
                .into_par_iter()
                .enumerate()
                .map(|(ci, (p, g, m))| {
                    let ctx = make_ctx(ci, cautious_scale, bound_scale);
                    tdclm_single_pass_chunk_dispatch(p, None, g, m, &ctx, backend)
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
        view.scratch.reset_accumulators();
        let chunk_metrics: Vec<ChunkMetrics> = match view.master_param.as_deref() {
            Some(mp) if has_master => (
                view.param.par_chunks(eff_chunk),
                mp.par_chunks(eff_chunk),
                view.grad.par_chunks(eff_chunk),
                view.exp_avg.par_chunks(eff_chunk),
            )
                .into_par_iter()
                .map(|(p, mp, g, m)| {
                    tdclm_pass1_chunk_dispatch(p, Some(mp), g, m, config, dtype_mode, backend)
                })
                .collect(),
            _ => (
                view.param.par_chunks(eff_chunk),
                view.grad.par_chunks(eff_chunk),
                view.exp_avg.par_chunks(eff_chunk),
            )
                .into_par_iter()
                .map(|(p, g, m)| {
                    tdclm_pass1_chunk_dispatch(p, None, g, m, config, dtype_mode, backend)
                })
                .collect(),
        };

        let reduced = chunk_metrics
            .into_iter()
            .fold(ChunkMetrics::default(), ChunkMetrics::combine);
        view.scratch.set_mask_sum(reduced.mask_count as f32);
        view.scratch.set_u_sq_sum(reduced.u_sq_sum);
        view.scratch.set_p_sq_sum(reduced.p_sq_sum);
        let (cautious_scale, bound_scale) =
            resolve_pass2_scales(reduced, total_elements, bound_mode, config);

        match view.master_param.as_deref_mut() {
            Some(mp) if has_master => (
                view.param.par_chunks_mut(eff_chunk),
                mp.par_chunks_mut(eff_chunk),
                view.grad.par_chunks(eff_chunk),
                view.exp_avg.par_chunks_mut(eff_chunk),
            )
                .into_par_iter()
                .enumerate()
                .for_each(|(ci, (p, mp, g, m))| {
                    let ctx = make_ctx(ci, cautious_scale, bound_scale);
                    tdclm_pass2_chunk_dispatch(p, Some(mp), g, m, &ctx, backend);
                }),
            _ => (
                view.param.par_chunks_mut(eff_chunk),
                view.grad.par_chunks(eff_chunk),
                view.exp_avg.par_chunks_mut(eff_chunk),
            )
                .into_par_iter()
                .enumerate()
                .for_each(|(ci, (p, g, m))| {
                    let ctx = make_ctx(ci, cautious_scale, bound_scale);
                    tdclm_pass2_chunk_dispatch(p, None, g, m, &ctx, backend);
                }),
        }

        view.scratch.reset_accumulators();
        view.scratch.set_cautious_scale(cautious_scale);
        view.scratch.set_bound_scale(bound_scale);
    }

    Ok(())
}
