use std::ffi::c_void;

use luminav_core::{
    BoundType, BufferMode, DTypeMode, LuminaError, LuminaVConfig, TensorViewMut,
    compute_inv_sqrt_bc2_prev, derive_spatial_seeds,
};

use crate::buffer_pool::{dual_coeffs, kernel_names_for_step, validate_kernel_config_raw};
use crate::{DeviceStepPtrs, LoadedKernel, TritonAotEngine, TritonAotError};

#[inline]
fn ptr<T>(v: &mut T) -> *mut c_void {
    (v as *mut T).cast::<c_void>()
}

#[allow(clippy::too_many_arguments)]
fn push_update_scalar_args(
    args: &mut Vec<*mut c_void>,
    buffer: BufferMode,
    beta1: &mut f32,
    beta2: &mut f32,
    c1: &mut f32,
    c2: &mut f32,
    lr: &mut f32,
    wd: &mut f32,
    eps: &mut f32,
    lam: &mut f32,
) {
    match buffer {
        BufferMode::Dual | BufferMode::T2 => args.extend_from_slice(&[
            ptr(beta1),
            ptr(beta2),
            ptr(c1),
            ptr(c2),
            ptr(lr),
            ptr(wd),
            ptr(eps),
        ]),
        BufferMode::TdClm | BufferMode::T1 => {
            args.extend_from_slice(&[ptr(beta1), ptr(lr), ptr(wd), ptr(eps), ptr(lam)])
        }
    }
}

impl TritonAotEngine {
    pub fn supports_config(
        &self,
        config: &LuminaVConfig,
        dtype_mode: DTypeMode,
        state_dtype_mode: DTypeMode,
        has_master: bool,
    ) -> bool {
        if config.validate().is_err() {
            return false;
        }
        let bound_mode = config.effective_bound_type();
        let needs_reduction = config.cautious || bound_mode == BoundType::Radial;
        for use_two_pass in [needs_reduction, false] {
            if use_two_pass && !needs_reduction {
                continue;
            }
            let (k1_name, k2_opt) = kernel_names_for_step(config, needs_reduction, use_two_pass);
            let Ok(m1) = self.manifest.get_kernel(k1_name) else {
                return false;
            };
            if validate_kernel_config_raw(m1, dtype_mode, state_dtype_mode, has_master, config)
                .is_err()
            {
                return false;
            }
            if let Some(k2_name) = k2_opt {
                let Ok(m2) = self.manifest.get_kernel(k2_name) else {
                    return false;
                };
                if validate_kernel_config_raw(m2, dtype_mode, state_dtype_mode, has_master, config)
                    .is_err()
                {
                    return false;
                }
            }
        }
        true
    }

    pub fn step(
        &mut self,
        view: &mut TensorViewMut<'_>,
        config: &LuminaVConfig,
        step: u64,
        param_index: usize,
    ) -> Result<(), TritonAotError> {
        config.validate()?;
        let has_master = config.master_weights.is_enabled() && view.master_param.is_some();
        let n = view.validate(config.buffer, has_master)?;
        if n == 0 {
            return Ok(());
        }

        let needs_v = matches!(config.buffer, BufferMode::Dual | BufferMode::T2);
        let bound_mode = config.effective_bound_type();
        let needs_reduction = config.cautious || bound_mode == BoundType::Radial;
        let use_two_pass = needs_reduction && (!config.fused_single_pass || step <= 1);

        let (primary_k, secondary_k) = self.load_step_kernels(
            view.dtype_mode,
            view.state_dtype_mode,
            has_master,
            config,
            needs_reduction,
            use_two_pass,
        )?;

        if use_two_pass || (needs_reduction && config.fused_single_pass && step > 1) {
            view.scratch.reset_accumulators();
        }

        let d_param = self.staging_pool.param.upload(&self.driver, view.param)?;
        let d_master = match &view.master_param {
            Some(mp) if has_master => self.staging_pool.master.upload(&self.driver, mp)?,
            _ => d_param,
        };
        let d_grad = self.staging_pool.grad.upload(&self.driver, view.grad)?;
        let d_exp_avg = self
            .staging_pool
            .exp_avg
            .upload(&self.driver, view.exp_avg)?;
        let d_exp_avg_sq = if needs_v {
            let v = view
                .exp_avg_sq
                .as_deref()
                .ok_or(LuminaError::MissingSecondaryStateBuffer)?;
            self.staging_pool.exp_avg_sq.upload(&self.driver, v)?
        } else {
            0
        };
        let d_scratch = self
            .staging_pool
            .scratch
            .upload(&self.driver, view.scratch.as_slice())?;

        let bufs = DeviceStepPtrs {
            d_param,
            d_master,
            d_grad,
            d_exp_avg,
            d_exp_avg_sq,
            d_scratch,
        };
        self.launch_step_kernels(
            bufs,
            n,
            config,
            view.dtype_mode,
            step,
            param_index,
            needs_v,
            use_two_pass,
            &primary_k,
            secondary_k.as_ref(),
        )?;

        self.driver.download_from(d_param, view.param)?;
        if let Some(mp) = &mut view.master_param
            && has_master
        {
            self.driver.download_from(d_master, mp)?;
        }
        self.driver.download_from(d_exp_avg, view.exp_avg)?;
        if let Some(v) = &mut view.exp_avg_sq
            && needs_v
        {
            self.driver.download_from(d_exp_avg_sq, v)?;
        }
        self.driver
            .download_from(d_scratch, view.scratch.as_mut_slice())?;
        if use_two_pass || !needs_reduction {
            view.scratch.reset_accumulators();
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_device_ptrs(
        &mut self,
        mut bufs: DeviceStepPtrs,
        n: usize,
        config: &LuminaVConfig,
        dtype_mode: DTypeMode,
        state_dtype_mode: DTypeMode,
        has_master: bool,
        step: u64,
        param_index: usize,
    ) -> Result<(), TritonAotError> {
        config.validate()?;
        if n == 0 {
            return Ok(());
        }
        let needs_v = matches!(config.buffer, BufferMode::Dual | BufferMode::T2);
        if needs_v && bufs.d_exp_avg_sq == 0 {
            return Err(TritonAotError::Core(
                LuminaError::MissingSecondaryStateBuffer,
            ));
        }
        if !has_master || bufs.d_master == 0 {
            bufs.d_master = bufs.d_param;
        }
        let bound_mode = config.effective_bound_type();
        let needs_reduction = config.cautious || bound_mode == BoundType::Radial;
        let use_two_pass = needs_reduction && (!config.fused_single_pass || step <= 1);
        let (primary_k, secondary_k) = self.load_step_kernels(
            dtype_mode,
            state_dtype_mode,
            has_master,
            config,
            needs_reduction,
            use_two_pass,
        )?;
        if use_two_pass || (needs_reduction && config.fused_single_pass && step > 1) {
            self.driver.upload_into(bufs.d_scratch, &[0.0_f32; 3])?;
        }
        self.launch_step_kernels(
            bufs,
            n,
            config,
            dtype_mode,
            step,
            param_index,
            needs_v,
            use_two_pass,
            &primary_k,
            secondary_k.as_ref(),
        )?;
        if use_two_pass {
            self.driver.upload_into(bufs.d_scratch, &[0.0_f32; 3])?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_step_kernels(
        &self,
        mut bufs: DeviceStepPtrs,
        n: usize,
        config: &LuminaVConfig,
        dtype_mode: DTypeMode,
        step: u64,
        param_index: usize,
        needs_v: bool,
        use_two_pass: bool,
        primary_k: &LoadedKernel,
        secondary_k: Option<&LoadedKernel>,
    ) -> Result<(), TritonAotError> {
        let mut n_i32 = n as i32;
        let mut total_f32 = n as f32;
        let mut beta1 = config.betas.0;
        let mut beta2 = config.betas.1;
        let (mut c1, mut c2) = if needs_v {
            dual_coeffs(config, dtype_mode, step)
        } else {
            (0.0_f32, 0.0_f32)
        };
        let mut lr = config.lr;
        let mut wd = config.weight_decay;
        let mut eps = config.eps;
        let mut lam = config.lam;
        let mut clamp_min = config.cautious_clamp_min;
        let mut ema_decay = config.ema_decay;
        let mut bound_ratio = config.bound_ratio;
        let mut bound_floor = config.bound_floor;
        let mut variance_temper = config.variance_temper;
        let mut inv_sqrt_bc2_prev = compute_inv_sqrt_bc2_prev(step, beta2);
        let (seed_m, _) = derive_spatial_seeds(config.seed, param_index);
        let mut seed_i32 = seed_m;
        let mut step_i32 = step as i32;
        let grid_x = (n as u32).div_ceil(primary_k.1.block_size.max(1));

        if use_two_pass {
            let pass2_k = secondary_k.expect("two-pass requires pass2 kernel");
            let mut args1 = bufs.base_args(needs_v, &mut n_i32);
            match config.buffer {
                BufferMode::Dual | BufferMode::T2 => args1.extend_from_slice(&[
                    ptr(&mut beta1),
                    ptr(&mut beta2),
                    ptr(&mut c1),
                    ptr(&mut c2),
                    ptr(&mut eps),
                    ptr(&mut variance_temper),
                    ptr(&mut inv_sqrt_bc2_prev),
                ]),
                BufferMode::TdClm | BufferMode::T1 => {
                    args1.extend_from_slice(&[ptr(&mut beta1), ptr(&mut eps), ptr(&mut lam)])
                }
            }
            self.launch_raw(primary_k, grid_x, &args1)?;

            let mut args2 = bufs.base_args(needs_v, &mut n_i32);
            args2.push(ptr(&mut total_f32));
            push_update_scalar_args(
                &mut args2,
                config.buffer,
                &mut beta1,
                &mut beta2,
                &mut c1,
                &mut c2,
                &mut lr,
                &mut wd,
                &mut eps,
                &mut lam,
            );
            args2.extend_from_slice(&[
                ptr(&mut clamp_min),
                ptr(&mut bound_ratio),
                ptr(&mut bound_floor),
            ]);
            if needs_v {
                args2.extend_from_slice(&[ptr(&mut variance_temper), ptr(&mut inv_sqrt_bc2_prev)]);
            }
            args2.extend_from_slice(&[ptr(&mut seed_i32), ptr(&mut step_i32)]);
            self.launch_raw(pass2_k, grid_x, &args2)
        } else {
            let mut args = bufs.base_args(needs_v, &mut n_i32);
            push_update_scalar_args(
                &mut args,
                config.buffer,
                &mut beta1,
                &mut beta2,
                &mut c1,
                &mut c2,
                &mut lr,
                &mut wd,
                &mut eps,
                &mut lam,
            );
            args.extend_from_slice(&[ptr(&mut bound_ratio), ptr(&mut bound_floor)]);
            if needs_v {
                args.extend_from_slice(&[ptr(&mut variance_temper), ptr(&mut inv_sqrt_bc2_prev)]);
            }
            args.extend_from_slice(&[ptr(&mut seed_i32), ptr(&mut step_i32)]);
            self.launch_raw(primary_k, grid_x, &args)?;

            if let Some(ema_k) = secondary_k {
                let ema_args = [
                    ptr(&mut bufs.d_scratch),
                    ptr(&mut total_f32),
                    ptr(&mut lr),
                    ptr(&mut clamp_min),
                    ptr(&mut ema_decay),
                    ptr(&mut bound_ratio),
                ];
                self.launch_raw(ema_k, 1, &ema_args)?;
            }
            Ok(())
        }
    }

    fn load_step_kernels(
        &mut self,
        dtype_mode: DTypeMode,
        state_dtype_mode: DTypeMode,
        has_master: bool,
        config: &LuminaVConfig,
        needs_reduction: bool,
        use_two_pass: bool,
    ) -> Result<(LoadedKernel, Option<LoadedKernel>), TritonAotError> {
        let (k1_name, k2_opt) = kernel_names_for_step(config, needs_reduction, use_two_pass);
        let k1 = self.load_kernel_binary(k1_name)?;
        validate_kernel_config_raw(&k1.1, dtype_mode, state_dtype_mode, has_master, config)?;
        let k2 = match k2_opt {
            Some(name) => {
                let k = self.load_kernel_binary(name)?;
                validate_kernel_config_raw(&k.1, dtype_mode, state_dtype_mode, has_master, config)?;
                Some(k)
            }
            None => None,
        };
        Ok((k1, k2))
    }
}
