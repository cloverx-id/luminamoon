use cubecl::prelude::*;
use luminav_core::{
    BoundType, BufferMode, DTypeMode, LuminaVConfig, TensorViewMut, compute_bias_correction,
    compute_cautious_scale, compute_inv_sqrt_bc2_prev, compute_radial_bound_scale,
    compute_radial_bound_scale_t2, compute_t2_sigma_coeff, derive_spatial_seeds,
    derive_variance_seed, derive_weyl_step_offsets, update_post_step_ema,
};

use crate::engine::CubeClLuminaEngine;
use crate::error::CubeClError;
use crate::kernels::optimizer::{
    lumina_dual_buffer_stats_kernel, lumina_dual_buffer_step_kernel,
    lumina_single_buffer_stats_kernel, lumina_single_buffer_step_kernel,
};

impl CubeClLuminaEngine {
    pub fn step(
        &mut self,
        view: &mut TensorViewMut<'_>,
        config: &LuminaVConfig,
        step: u64,
        param_index: usize,
    ) -> Result<(), CubeClError> {
        config.validate()?;
        let has_master = config.master_weights.is_enabled() && view.master_param.is_some();
        let n = view.validate(config.buffer, has_master)?;
        if n == 0 {
            return Ok(());
        }

        const MAX_GRID_X: usize = 65_535;
        let cube_dim = self.cube_dim_1d;
        let num_cubes = n.div_ceil(cube_dim);
        if num_cubes > MAX_GRID_X {
            return Err(CubeClError::LaunchFailed(format!(
                "optimizer step grid width {num_cubes} exceeds portable 1D workgroup limit ({MAX_GRID_X})"
            )));
        }
        let bound_mode = config.effective_bound_type();
        let bound_mode_u32 = match bound_mode {
            BoundType::None => 0_u32,
            BoundType::Radial => 1_u32,
            BoundType::Coordinate => 2_u32,
        };
        let dtype_mode_u32 = match view.dtype_mode {
            DTypeMode::Float32 => 0_u32,
            DTypeMode::BFloat16 => 1_u32,
            DTypeMode::Float16 => 2_u32,
        };
        let state_dtype_u32 = match view.state_dtype_mode {
            DTypeMode::Float32 => 0_u32,
            DTypeMode::BFloat16 | DTypeMode::Float16 => 1_u32,
        };
        let stochastic_u32 = u32::from(config.stochastic_rounding);
        let use_rtv = config.uses_rtv() && step > 1;
        let use_rpm = config.uses_rpm();
        let inv_sqrt_bc2_prev = compute_inv_sqrt_bc2_prev(step, config.betas.1);
        let needs_ema = config.cautious || bound_mode == BoundType::Radial;
        let two_pass = (!config.fused_single_pass || step <= 1) && needs_ema;

        let (seed_m_i, seed_p_i) = derive_spatial_seeds(config.seed, param_index);
        let seed_v_i = derive_variance_seed(seed_m_i);
        let (weyl_m, weyl_v, weyl_p) = derive_weyl_step_offsets(step);
        let (beta1, beta2) = config.betas;
        let one_minus_beta1 = 1.0_f32 - beta1;
        let one_minus_beta2 = 1.0_f32 - beta2;
        let beta1_sq = beta1 * beta1;
        let one_minus_beta1_sq = 1.0_f32 - beta1_sq;
        let lr_wd = config.lr * config.weight_decay;

        let h_param = self.client.create_from_slice(f32::as_bytes(view.param));
        let h_master = match &view.master_param {
            Some(mp) if has_master => self.client.create_from_slice(f32::as_bytes(mp)),
            _ => self.client.empty(core::mem::size_of::<f32>()),
        };
        let h_grad = self.client.create_from_slice(f32::as_bytes(view.grad));
        let h_exp_avg = self.client.create_from_slice(f32::as_bytes(view.exp_avg));
        let partial_bytes = num_cubes.max(1) * core::mem::size_of::<f32>();
        let h_pmask = self.client.empty(partial_bytes);
        let h_pusq = self.client.empty(partial_bytes);
        let h_ppsq = self.client.empty(partial_bytes);

        let master_len = if has_master { n } else { 1 };
        let grid = CubeCount::Static(num_cubes as u32, 1, 1);
        let dim = CubeDim::new_1d(cube_dim as u32);

        match config.buffer {
            BufferMode::TdClm | BufferMode::T1 => {
                let is_t1 = config.buffer == BufferMode::T1;
                let (mut c_scale, mut b_scale) =
                    (view.scratch.cautious_scale(), view.scratch.bound_scale());

                if two_pass {
                    unsafe {
                        lumina_single_buffer_stats_kernel::launch_unchecked(
                            &self.client,
                            grid.clone(),
                            dim,
                            BufferArg::from_raw_parts(h_param.clone(), n),
                            BufferArg::from_raw_parts(h_master.clone(), master_len),
                            BufferArg::from_raw_parts(h_grad.clone(), n),
                            BufferArg::from_raw_parts(h_exp_avg.clone(), n),
                            BufferArg::from_raw_parts(h_pmask.clone(), num_cubes),
                            BufferArg::from_raw_parts(h_pusq.clone(), num_cubes),
                            BufferArg::from_raw_parts(h_ppsq.clone(), num_cubes),
                            n,
                            one_minus_beta1,
                            beta1_sq,
                            one_minus_beta1_sq,
                            config.lam,
                            config.eps,
                            use_rpm,
                            has_master,
                            config.cautious,
                            bound_mode == BoundType::Radial,
                            cube_dim,
                        );
                    }
                    let mask_sum = self.read_sum_f32(h_pmask.clone());
                    let u_sq_sum = self.read_sum_f32(h_pusq.clone());
                    let p_sq_sum = self.read_sum_f32(h_ppsq.clone());
                    if config.cautious {
                        c_scale =
                            compute_cautious_scale(mask_sum, n as f32, config.cautious_clamp_min);
                        view.scratch.set_cautious_scale(c_scale);
                    }
                    if bound_mode == BoundType::Radial {
                        let eff_c = if config.cautious { c_scale } else { 1.0_f32 };
                        b_scale = if is_t1 {
                            compute_radial_bound_scale_t2(
                                u_sq_sum,
                                p_sq_sum,
                                eff_c,
                                config.lr,
                                config.bound_ratio,
                            )
                        } else {
                            compute_radial_bound_scale(
                                u_sq_sum,
                                p_sq_sum,
                                eff_c,
                                config.lr,
                                config.bound_ratio,
                            )
                        };
                        view.scratch.set_bound_scale(b_scale);
                    }
                }

                let collect_stats = !two_pass && needs_ema;
                unsafe {
                    lumina_single_buffer_step_kernel::launch_unchecked(
                        &self.client,
                        grid,
                        dim,
                        BufferArg::from_raw_parts(h_param.clone(), n),
                        BufferArg::from_raw_parts(h_master.clone(), master_len),
                        BufferArg::from_raw_parts(h_grad, n),
                        BufferArg::from_raw_parts(h_exp_avg.clone(), n),
                        BufferArg::from_raw_parts(h_pmask.clone(), num_cubes),
                        BufferArg::from_raw_parts(h_pusq.clone(), num_cubes),
                        BufferArg::from_raw_parts(h_ppsq.clone(), num_cubes),
                        n,
                        config.lr,
                        beta1,
                        one_minus_beta1,
                        beta1_sq,
                        one_minus_beta1_sq,
                        config.lam,
                        config.eps,
                        lr_wd,
                        config.bound_ratio,
                        config.bound_floor,
                        c_scale,
                        b_scale,
                        seed_m_i as u32,
                        seed_p_i as u32,
                        weyl_m,
                        weyl_p,
                        dtype_mode_u32,
                        state_dtype_u32,
                        stochastic_u32,
                        is_t1,
                        use_rpm,
                        has_master,
                        config.cautious,
                        config.cautious_weight_decay,
                        bound_mode_u32,
                        collect_stats,
                        cube_dim,
                    );
                }

                if collect_stats {
                    view.scratch.set_mask_sum(self.read_sum_f32(h_pmask));
                    view.scratch.set_u_sq_sum(self.read_sum_f32(h_pusq));
                    view.scratch.set_p_sq_sum(self.read_sum_f32(h_ppsq));
                    update_post_step_ema(view.scratch, n as f32, config);
                } else {
                    view.scratch.reset_accumulators();
                }
            }
            BufferMode::Dual | BufferMode::T2 => {
                let is_t2 = config.buffer == BufferMode::T2;
                let exp_avg_sq =
                    view.exp_avg_sq
                        .as_deref_mut()
                        .ok_or(CubeClError::InvalidShape(
                            "Missing exp_avg_sq in Dual/T2 mode".to_string(),
                        ))?;
                let h_exp_avg_sq = self.client.create_from_slice(f32::as_bytes(exp_avg_sq));

                let (c1, c2) = if is_t2 {
                    (
                        compute_t2_sigma_coeff(step, beta2, config.lam),
                        0.5_f32 * config.lam,
                    )
                } else {
                    let coeffs = compute_bias_correction(
                        step,
                        beta1,
                        beta2,
                        config.eps,
                        config.tau,
                        view.dtype_mode,
                    );
                    (coeffs.c1, coeffs.c2)
                };

                let (mut c_scale, mut b_scale) =
                    (view.scratch.cautious_scale(), view.scratch.bound_scale());

                if two_pass {
                    unsafe {
                        lumina_dual_buffer_stats_kernel::launch_unchecked(
                            &self.client,
                            grid.clone(),
                            dim,
                            BufferArg::from_raw_parts(h_param.clone(), n),
                            BufferArg::from_raw_parts(h_master.clone(), master_len),
                            BufferArg::from_raw_parts(h_grad.clone(), n),
                            BufferArg::from_raw_parts(h_exp_avg.clone(), n),
                            BufferArg::from_raw_parts(h_exp_avg_sq.clone(), n),
                            BufferArg::from_raw_parts(h_pmask.clone(), num_cubes),
                            BufferArg::from_raw_parts(h_pusq.clone(), num_cubes),
                            BufferArg::from_raw_parts(h_ppsq.clone(), num_cubes),
                            n,
                            beta1,
                            one_minus_beta1,
                            beta2,
                            one_minus_beta2,
                            beta1_sq,
                            one_minus_beta1_sq,
                            c1,
                            c2,
                            config.eps,
                            config.variance_temper,
                            inv_sqrt_bc2_prev,
                            is_t2,
                            use_rtv,
                            use_rpm,
                            has_master,
                            config.cautious,
                            bound_mode == BoundType::Radial,
                            cube_dim,
                        );
                    }
                    let mask_sum = self.read_sum_f32(h_pmask.clone());
                    let u_sq_sum = self.read_sum_f32(h_pusq.clone());
                    let p_sq_sum = self.read_sum_f32(h_ppsq.clone());
                    if config.cautious {
                        c_scale =
                            compute_cautious_scale(mask_sum, n as f32, config.cautious_clamp_min);
                        view.scratch.set_cautious_scale(c_scale);
                    }
                    if bound_mode == BoundType::Radial {
                        let eff_c = if config.cautious { c_scale } else { 1.0_f32 };
                        b_scale = if is_t2 {
                            compute_radial_bound_scale_t2(
                                u_sq_sum,
                                p_sq_sum,
                                eff_c,
                                config.lr,
                                config.bound_ratio,
                            )
                        } else {
                            compute_radial_bound_scale(
                                u_sq_sum,
                                p_sq_sum,
                                eff_c,
                                config.lr,
                                config.bound_ratio,
                            )
                        };
                        view.scratch.set_bound_scale(b_scale);
                    }
                }

                let collect_stats = !two_pass && needs_ema;
                unsafe {
                    lumina_dual_buffer_step_kernel::launch_unchecked(
                        &self.client,
                        grid,
                        dim,
                        BufferArg::from_raw_parts(h_param.clone(), n),
                        BufferArg::from_raw_parts(h_master.clone(), master_len),
                        BufferArg::from_raw_parts(h_grad, n),
                        BufferArg::from_raw_parts(h_exp_avg.clone(), n),
                        BufferArg::from_raw_parts(h_exp_avg_sq.clone(), n),
                        BufferArg::from_raw_parts(h_pmask.clone(), num_cubes),
                        BufferArg::from_raw_parts(h_pusq.clone(), num_cubes),
                        BufferArg::from_raw_parts(h_ppsq.clone(), num_cubes),
                        n,
                        config.lr,
                        beta1,
                        one_minus_beta1,
                        beta2,
                        one_minus_beta2,
                        beta1_sq,
                        one_minus_beta1_sq,
                        c1,
                        c2,
                        config.eps,
                        lr_wd,
                        config.bound_ratio,
                        config.bound_floor,
                        config.variance_temper,
                        inv_sqrt_bc2_prev,
                        c_scale,
                        b_scale,
                        seed_m_i as u32,
                        seed_v_i as u32,
                        seed_p_i as u32,
                        weyl_m,
                        weyl_v,
                        weyl_p,
                        dtype_mode_u32,
                        state_dtype_u32,
                        stochastic_u32,
                        is_t2,
                        use_rtv,
                        use_rpm,
                        has_master,
                        config.cautious,
                        config.cautious_weight_decay,
                        bound_mode_u32,
                        collect_stats,
                        cube_dim,
                    );
                }

                self.read_into_slice(h_exp_avg_sq, exp_avg_sq);
                if collect_stats {
                    view.scratch.set_mask_sum(self.read_sum_f32(h_pmask));
                    view.scratch.set_u_sq_sum(self.read_sum_f32(h_pusq));
                    view.scratch.set_p_sq_sum(self.read_sum_f32(h_ppsq));
                    update_post_step_ema(view.scratch, n as f32, config);
                } else {
                    view.scratch.reset_accumulators();
                }
            }
        }

        self.read_into_slice(h_param, view.param);
        if let Some(mp) = &mut view.master_param
            && has_master
        {
            self.read_into_slice(h_master, mp);
        }
        self.read_into_slice(h_exp_avg, view.exp_avg);
        Ok(())
    }
}
