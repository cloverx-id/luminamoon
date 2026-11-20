use luminav_core::{
    BoundType, BufferMode, DTypeMode, LuminaError, LuminaVConfig, TensorViewMut,
    compute_bias_correction, derive_kernel_seeds, update_post_step_ema,
};
use rayon::prelude::*;

use crate::SimdBackend;
use crate::common::{ChunkMetrics, DualChunkCtx};
use crate::dispatch::dual::{
    dual_pass1_chunk_dispatch, dual_pass2_chunk_dispatch, dual_single_pass_chunk_dispatch,
};
use crate::dispatch::resolve_pass2_scales;

use super::ELEM_CHUNK_SIZE;

pub fn dual_step_parallel(
    view: &mut TensorViewMut<'_>,
    config: &LuminaVConfig,
    step: u64,
    param_index: usize,
    backend: SimdBackend,
) -> Result<(), LuminaError> {
    dual_step_parallel_chunked(view, config, step, param_index, backend, ELEM_CHUNK_SIZE)
}

pub fn dual_step_parallel_chunked(
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
        BufferMode::T2
    } else {
        BufferMode::Dual
    };
    let n_elements = view.validate(target_mode, has_master)?;
    if n_elements == 0 {
        return Ok(());
    }

    let eff_chunk = chunk_size.max(16) & !15;
    let total_elements = n_elements as f32;
    let (beta1, beta2) = config.betas;
    let coeffs =
        compute_bias_correction(step, beta1, beta2, config.eps, config.tau, view.dtype_mode);
    let bound_mode = config.effective_bound_type();
    let (seed_m, seed_p) = derive_kernel_seeds(config.seed, step, param_index);
    let dtype_mode = view.dtype_mode;
    let state_dtype_mode = if view.state_dtype_mode == DTypeMode::Float16 {
        DTypeMode::BFloat16
    } else {
        view.state_dtype_mode
    };
    let exp_avg_sq = view
        .exp_avg_sq
        .as_deref_mut()
        .ok_or(LuminaError::MissingSecondaryStateBuffer)?;

    let make_ctx =
        |chunk_idx: usize, cautious_scale: f32, bound_scale: f32, accumulate: bool| DualChunkCtx {
            config,
            coeffs: &coeffs,
            dtype_mode,
            state_dtype_mode,
            chunk_offset: chunk_idx * eff_chunk,
            cautious_scale,
            bound_scale,
            seed_m,
            seed_p,
            accumulate_metrics: accumulate,
        };

    if !config.cautious && bound_mode != BoundType::Radial {
        match view.master_param.as_deref_mut() {
            Some(mp) if has_master => (
                view.param.par_chunks_mut(eff_chunk),
                mp.par_chunks_mut(eff_chunk),
                view.grad.par_chunks(eff_chunk),
                view.exp_avg.par_chunks_mut(eff_chunk),
                exp_avg_sq.par_chunks_mut(eff_chunk),
            )
                .into_par_iter()
                .enumerate()
                .for_each(|(ci, (p, mp, g, m, v))| {
                    let ctx = make_ctx(ci, 1.0_f32, 1.0_f32, false);
                    dual_single_pass_chunk_dispatch(p, Some(mp), g, m, v, &ctx, backend);
                }),
            _ => (
                view.param.par_chunks_mut(eff_chunk),
                view.grad.par_chunks(eff_chunk),
                view.exp_avg.par_chunks_mut(eff_chunk),
                exp_avg_sq.par_chunks_mut(eff_chunk),
            )
                .into_par_iter()
                .enumerate()
                .for_each(|(ci, (p, g, m, v))| {
                    let ctx = make_ctx(ci, 1.0_f32, 1.0_f32, false);
                    dual_single_pass_chunk_dispatch(p, None, g, m, v, &ctx, backend);
                }),
        }
        view.scratch.reset_accumulators();
    } else if config.fused_single_pass && step > 1 {
        view.scratch.reset_accumulators();
        let cautious_scale = view.scratch.cautious_scale();
        let bound_scale = view.scratch.bound_scale();

        let chunk_metrics: Vec<ChunkMetrics> = match view.master_param.as_deref_mut() {
            Some(mp) if has_master => (
                view.param.par_chunks_mut(eff_chunk),
                mp.par_chunks_mut(eff_chunk),
                view.grad.par_chunks(eff_chunk),
                view.exp_avg.par_chunks_mut(eff_chunk),
                exp_avg_sq.par_chunks_mut(eff_chunk),
            )
                .into_par_iter()
                .enumerate()
                .map(|(ci, (p, mp, g, m, v))| {
                    let ctx = make_ctx(ci, cautious_scale, bound_scale, true);
                    dual_single_pass_chunk_dispatch(p, Some(mp), g, m, v, &ctx, backend)
                })
                .collect(),
            _ => (
                view.param.par_chunks_mut(eff_chunk),
                view.grad.par_chunks(eff_chunk),
                view.exp_avg.par_chunks_mut(eff_chunk),
                exp_avg_sq.par_chunks_mut(eff_chunk),
            )
                .into_par_iter()
                .enumerate()
                .map(|(ci, (p, g, m, v))| {
                    let ctx = make_ctx(ci, cautious_scale, bound_scale, true);
                    dual_single_pass_chunk_dispatch(p, None, g, m, v, &ctx, backend)
                })
                .collect(),
        };

        let reduced = chunk_metrics
            .into_iter()
            .fold(ChunkMetrics::default(), ChunkMetrics::combine);
        view.scratch.set_mask_sum(reduced.mask_count as f32);
        view.scratch.set_u_sq_sum(reduced.u_sq_sum);
        view.scratch.set_p_sq_sum(reduced.p_sq_sum);
        update_post_step_ema(view.scratch, total_elements, config);
    } else {
        view.scratch.reset_accumulators();
        let chunk_metrics: Vec<ChunkMetrics> = match view.master_param.as_deref() {
            Some(mp) if has_master => (
                view.param.par_chunks(eff_chunk),
                mp.par_chunks(eff_chunk),
                view.grad.par_chunks(eff_chunk),
                view.exp_avg.par_chunks(eff_chunk),
                exp_avg_sq.par_chunks(eff_chunk),
            )
                .into_par_iter()
                .map(|(p, mp, g, m, v)| {
                    dual_pass1_chunk_dispatch(
                        p,
                        Some(mp),
                        g,
                        m,
                        v,
                        config,
                        &coeffs,
                        dtype_mode,
                        state_dtype_mode,
                        backend,
                    )
                })
                .collect(),
            _ => (
                view.param.par_chunks(eff_chunk),
                view.grad.par_chunks(eff_chunk),
                view.exp_avg.par_chunks(eff_chunk),
                exp_avg_sq.par_chunks(eff_chunk),
            )
                .into_par_iter()
                .map(|(p, g, m, v)| {
                    dual_pass1_chunk_dispatch(
                        p,
                        None,
                        g,
                        m,
                        v,
                        config,
                        &coeffs,
                        dtype_mode,
                        state_dtype_mode,
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
        let (cautious_scale, bound_scale) =
            resolve_pass2_scales(reduced, total_elements, bound_mode, config);

        match view.master_param.as_deref_mut() {
            Some(mp) if has_master => (
                view.param.par_chunks_mut(eff_chunk),
                mp.par_chunks_mut(eff_chunk),
                view.grad.par_chunks(eff_chunk),
                view.exp_avg.par_chunks_mut(eff_chunk),
                exp_avg_sq.par_chunks_mut(eff_chunk),
            )
                .into_par_iter()
                .enumerate()
                .for_each(|(ci, (p, mp, g, m, v))| {
                    let ctx = make_ctx(ci, cautious_scale, bound_scale, false);
                    dual_pass2_chunk_dispatch(p, Some(mp), g, m, v, &ctx, backend);
                }),
            _ => (
                view.param.par_chunks_mut(eff_chunk),
                view.grad.par_chunks(eff_chunk),
                view.exp_avg.par_chunks_mut(eff_chunk),
                exp_avg_sq.par_chunks_mut(eff_chunk),
            )
                .into_par_iter()
                .enumerate()
                .for_each(|(ci, (p, g, m, v))| {
                    let ctx = make_ctx(ci, cautious_scale, bound_scale, false);
                    dual_pass2_chunk_dispatch(p, None, g, m, v, &ctx, backend);
                }),
        }

        view.scratch.reset_accumulators();
        view.scratch.set_cautious_scale(cautious_scale);
        view.scratch.set_bound_scale(bound_scale);
    }

    Ok(())
}
