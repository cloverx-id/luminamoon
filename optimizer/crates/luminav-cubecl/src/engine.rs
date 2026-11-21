use cubecl::Device;
use cubecl::prelude::*;
use luminav_core::{
    BoundType, BufferMode, DTypeMode, LuminaVConfig, TensorViewMut, compute_bias_correction,
    compute_cautious_scale, compute_radial_bound_scale, compute_radial_bound_scale_t2,
    compute_t2_sigma_coeff, derive_spatial_seeds, derive_variance_seed, derive_weyl_step_offsets,
    update_post_step_ema,
};

use crate::error::CubeClError;
use crate::kernels::nn::{
    pos_ra_glu_fwd_kernel, ra_norm_fwd_kernel, ra_softmax_fwd_kernel, ra_tanh_fwd_kernel,
};
use crate::kernels::optimizer::{
    lumina_dual_buffer_stats_kernel, lumina_dual_buffer_step_kernel,
    lumina_single_buffer_stats_kernel, lumina_single_buffer_step_kernel,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeClBackend {
    Cuda,
    Hip,
    Wgpu,
    Cpu,
}

impl CubeClBackend {
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Hip => "hip",
            Self::Wgpu => "wgpu",
            Self::Cpu => "cpu",
        }
    }
}

pub struct CubeClLuminaEngine {
    client: Client,
    backend: CubeClBackend,
    cube_dim_1d: usize,
}

impl CubeClLuminaEngine {
    pub fn new() -> Result<Self, CubeClError> {
        #[cfg(feature = "cuda")]
        if let Ok(eng) = Self::with_cuda() {
            return Ok(eng);
        }

        #[cfg(feature = "hip")]
        if let Ok(eng) = Self::with_hip() {
            return Ok(eng);
        }

        #[cfg(feature = "wgpu")]
        if let Ok(eng) = Self::with_wgpu() {
            return Ok(eng);
        }

        #[cfg(feature = "cpu")]
        if let Ok(eng) = Self::with_cpu() {
            return Ok(eng);
        }

        Err(CubeClError::RuntimeUnavailable(
            "No CubeCL runtime feature enabled or available on this host".to_string(),
        ))
    }

    #[cfg(feature = "wgpu")]
    pub fn with_wgpu() -> Result<Self, CubeClError> {
        let client = std::panic::catch_unwind(|| Device::Wgpu(Default::default()).client())
            .map_err(|_| CubeClError::RuntimeUnavailable("WGPU adapter unavailable".to_string()))?;
        Ok(Self::from_client(client, CubeClBackend::Wgpu))
    }

    #[cfg(feature = "cuda")]
    pub fn with_cuda() -> Result<Self, CubeClError> {
        let client = std::panic::catch_unwind(|| Device::Cuda(Default::default()).client())
            .map_err(|_| CubeClError::RuntimeUnavailable("CUDA runtime unavailable".to_string()))?;
        Ok(Self::from_client(client, CubeClBackend::Cuda))
    }

    #[cfg(feature = "hip")]
    pub fn with_hip() -> Result<Self, CubeClError> {
        let client = std::panic::catch_unwind(|| Device::Hip(Default::default()).client())
            .map_err(|_| CubeClError::RuntimeUnavailable("HIP/ROCm runtime unavailable".to_string()))?;
        Ok(Self::from_client(client, CubeClBackend::Hip))
    }

    #[cfg(feature = "cpu")]
    pub fn with_cpu() -> Result<Self, CubeClError> {
        let client = std::panic::catch_unwind(|| Device::Cpu(Default::default()).client())
            .map_err(|_| CubeClError::RuntimeUnavailable("CubeCL CPU runtime unavailable".to_string()))?;
        Ok(Self::from_client(client, CubeClBackend::Cpu))
    }

    fn from_client(client: Client, backend: CubeClBackend) -> Self {
        let max_units = client.properties().hardware.max_units_per_cube as usize;
        // Pick largest power-of-two cube size up to 256 supported by the active hardware.
        let cube_dim_1d = max_units.clamp(32, 256).next_power_of_two().min(max_units).max(1);
        let cube_dim_1d = if cube_dim_1d.is_power_of_two() {
            cube_dim_1d
        } else {
            (cube_dim_1d / 2).next_power_of_two().max(1)
        };
        Self {
            client,
            backend,
            cube_dim_1d,
        }
    }

    #[inline]
    pub fn backend(&self) -> CubeClBackend {
        self.backend
    }

    #[inline]
    pub fn cube_dim_1d(&self) -> usize {
        self.cube_dim_1d
    }

    fn read_sum_f32(&self, handle: cubecl::server::Handle) -> f32 {
        let bytes = self.client.read_one_unchecked(handle);
        f32::from_bytes(&bytes).iter().copied().sum()
    }

    fn read_into_slice(&self, handle: cubecl::server::Handle, dst: &mut [f32]) {
        let bytes = self.client.read_one_unchecked(handle);
        dst.copy_from_slice(f32::from_bytes(&bytes));
    }

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

        let cube_dim = self.cube_dim_1d;
        let num_cubes = n.div_ceil(cube_dim);
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
        let clamp_v_fp16 = u32::from(
            view.dtype_mode == DTypeMode::Float16 || view.state_dtype_mode == DTypeMode::Float16,
        );
        let stochastic_u32 = u32::from(config.stochastic_rounding);
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
                            beta1_sq,
                            one_minus_beta1_sq,
                            config.lam,
                            config.eps,
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
                        c_scale = compute_cautious_scale(mask_sum, n as f32, config.cautious_clamp_min);
                        view.scratch.set_cautious_scale(c_scale);
                    }
                    if bound_mode == BoundType::Radial {
                        let eff_c = if config.cautious { c_scale } else { 1.0_f32 };
                        b_scale = if is_t1 {
                            compute_radial_bound_scale_t2(u_sq_sum, p_sq_sum, eff_c, config.lr, config.bound_ratio)
                        } else {
                            compute_radial_bound_scale(u_sq_sum, p_sq_sum, eff_c, config.lr, config.bound_ratio)
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
                        has_master,
                        config.cautious,
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
                let exp_avg_sq = view
                    .exp_avg_sq
                    .as_deref_mut()
                    .ok_or(CubeClError::InvalidShape("Missing exp_avg_sq in Dual/T2 mode".to_string()))?;
                let h_exp_avg_sq = self.client.create_from_slice(f32::as_bytes(exp_avg_sq));

                let (c1, c2) = if is_t2 {
                    (compute_t2_sigma_coeff(step, beta2, config.lam), 0.5_f32 * config.lam)
                } else {
                    let coeffs = compute_bias_correction(step, beta1, beta2, config.eps, config.tau, view.dtype_mode);
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
                            clamp_v_fp16,
                            is_t2,
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
                        c_scale = compute_cautious_scale(mask_sum, n as f32, config.cautious_clamp_min);
                        view.scratch.set_cautious_scale(c_scale);
                    }
                    if bound_mode == BoundType::Radial {
                        let eff_c = if config.cautious { c_scale } else { 1.0_f32 };
                        b_scale = if is_t2 {
                            compute_radial_bound_scale_t2(u_sq_sum, p_sq_sum, eff_c, config.lr, config.bound_ratio)
                        } else {
                            compute_radial_bound_scale(u_sq_sum, p_sq_sum, eff_c, config.lr, config.bound_ratio)
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
                        clamp_v_fp16,
                        stochastic_u32,
                        is_t2,
                        has_master,
                        config.cautious,
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

    pub fn ra_tanh_f32(&self, x: &[f32], b: f32) -> Result<Vec<f32>, CubeClError> {
        let n = x.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let h_in = self.client.create_from_slice(f32::as_bytes(x));
        let h_out = self.client.empty(n * core::mem::size_of::<f32>());
        let num_cubes = n.div_ceil(self.cube_dim_1d);
        unsafe {
            ra_tanh_fwd_kernel::launch_unchecked(
                &self.client,
                CubeCount::Static(num_cubes as u32, 1, 1),
                CubeDim::new_1d(self.cube_dim_1d as u32),
                BufferArg::from_raw_parts(h_in, n),
                BufferArg::from_raw_parts(h_out.clone(), n),
                n,
                b,
            );
        }
        let mut out = vec![0.0_f32; n];
        self.read_into_slice(h_out, &mut out);
        Ok(out)
    }

    pub fn pos_ra_glu_f32(&self, gate: &[f32], up: &[f32], b: f32) -> Result<Vec<f32>, CubeClError> {
        if gate.len() != up.len() {
            return Err(CubeClError::InvalidShape(format!(
                "gate len {} != up len {}",
                gate.len(),
                up.len()
            )));
        }
        let n = gate.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let h_gate = self.client.create_from_slice(f32::as_bytes(gate));
        let h_up = self.client.create_from_slice(f32::as_bytes(up));
        let h_out = self.client.empty(n * core::mem::size_of::<f32>());
        let num_cubes = n.div_ceil(self.cube_dim_1d);
        unsafe {
            pos_ra_glu_fwd_kernel::launch_unchecked(
                &self.client,
                CubeCount::Static(num_cubes as u32, 1, 1),
                CubeDim::new_1d(self.cube_dim_1d as u32),
                BufferArg::from_raw_parts(h_gate, n),
                BufferArg::from_raw_parts(h_up, n),
                BufferArg::from_raw_parts(h_out.clone(), n),
                n,
                b,
            );
        }
        let mut out = vec![0.0_f32; n];
        self.read_into_slice(h_out, &mut out);
        Ok(out)
    }

    pub fn ra_norm_f32(
        &self,
        x: &[f32],
        weight: Option<&[f32]>,
        bias: Option<&[f32]>,
        cols: usize,
        eps: f32,
    ) -> Result<Vec<f32>, CubeClError> {
        if cols == 0 || !x.len().is_multiple_of(cols) {
            return Err(CubeClError::InvalidShape(format!(
                "x len {} not divisible by cols {}",
                x.len(),
                cols
            )));
        }
        let rows = x.len() / cols;
        if rows == 0 {
            return Ok(Vec::new());
        }
        let h_x = self.client.create_from_slice(f32::as_bytes(x));
        let h_w = match weight {
            Some(w) => self.client.create_from_slice(f32::as_bytes(w)),
            None => self.client.empty(core::mem::size_of::<f32>()),
        };
        let h_b = match bias {
            Some(b) => self.client.create_from_slice(f32::as_bytes(b)),
            None => self.client.empty(core::mem::size_of::<f32>()),
        };
        let h_out = self.client.empty(x.len() * core::mem::size_of::<f32>());
        let row_cube = cols.next_power_of_two().min(self.cube_dim_1d).max(1);
        unsafe {
            ra_norm_fwd_kernel::launch_unchecked(
                &self.client,
                CubeCount::Static(rows as u32, 1, 1),
                CubeDim::new_1d(row_cube as u32),
                BufferArg::from_raw_parts(h_x, x.len()),
                BufferArg::from_raw_parts(h_w, weight.map_or(1, |w| w.len())),
                BufferArg::from_raw_parts(h_b, bias.map_or(1, |b| b.len())),
                BufferArg::from_raw_parts(h_out.clone(), x.len()),
                cols,
                eps,
                weight.is_some(),
                bias.is_some(),
                row_cube,
            );
        }
        let mut out = vec![0.0_f32; x.len()];
        self.read_into_slice(h_out, &mut out);
        Ok(out)
    }

    pub fn ra_softmax_f32(&self, scores: &[f32], cols: usize, scale: f32) -> Result<Vec<f32>, CubeClError> {
        if cols == 0 || !scores.len().is_multiple_of(cols) {
            return Err(CubeClError::InvalidShape(format!(
                "scores len {} not divisible by cols {}",
                scores.len(),
                cols
            )));
        }
        let rows = scores.len() / cols;
        if rows == 0 {
            return Ok(Vec::new());
        }
        let h_s = self.client.create_from_slice(f32::as_bytes(scores));
        let h_out = self.client.empty(scores.len() * core::mem::size_of::<f32>());
        let row_cube = cols.next_power_of_two().min(self.cube_dim_1d).max(1);
        unsafe {
            ra_softmax_fwd_kernel::launch_unchecked(
                &self.client,
                CubeCount::Static(rows as u32, 1, 1),
                CubeDim::new_1d(row_cube as u32),
                BufferArg::from_raw_parts(h_s, scores.len()),
                BufferArg::from_raw_parts(h_out.clone(), scores.len()),
                cols,
                scale,
                row_cube,
            );
        }
        let mut out = vec![0.0_f32; scores.len()];
        self.read_into_slice(h_out, &mut out);
        Ok(out)
    }
}
