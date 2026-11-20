use std::ffi::c_void;

use luminav_core::{
    BoundType, BufferMode, DTypeMode, LuminaError, LuminaVConfig, TensorViewMut,
    compute_bias_correction, compute_t2_sigma_coeff, derive_spatial_seeds,
};

use crate::{LoadedKernel, TritonAotEngine, TritonAotError};

struct DeviceStepBuffers<'a> {
    engine: &'a TritonAotEngine,
    d_param: u64,
    d_master: u64,
    d_grad: u64,
    d_exp_avg: u64,
    d_exp_avg_sq: u64,
    d_scratch: u64,
}

impl<'a> DeviceStepBuffers<'a> {
    fn upload(
        engine: &'a TritonAotEngine,
        view: &TensorViewMut<'_>,
        needs_v: bool,
    ) -> Result<Self, TritonAotError> {
        let mut bufs = Self {
            engine,
            d_param: 0,
            d_master: 0,
            d_grad: 0,
            d_exp_avg: 0,
            d_exp_avg_sq: 0,
            d_scratch: 0,
        };
        bufs.d_param = engine.alloc_upload(view.param)?;
        bufs.d_master = match &view.master_param {
            Some(mp) => engine.alloc_upload(mp)?,
            None => bufs.d_param,
        };
        bufs.d_grad = engine.alloc_upload(view.grad)?;
        bufs.d_exp_avg = engine.alloc_upload(view.exp_avg)?;
        if needs_v {
            let v = view
                .exp_avg_sq
                .as_deref()
                .ok_or(LuminaError::MissingSecondaryStateBuffer)?;
            bufs.d_exp_avg_sq = engine.alloc_upload(v)?;
        }
        bufs.d_scratch = engine.alloc_upload(view.scratch.as_slice())?;
        Ok(bufs)
    }

    fn base_args(&mut self, needs_v: bool, n_i32: &mut i32) -> Vec<*mut c_void> {
        let mut args = Vec::with_capacity(20);
        args.push(ptr(&mut self.d_param));
        args.push(ptr(&mut self.d_master));
        args.push(ptr(&mut self.d_grad));
        args.push(ptr(&mut self.d_exp_avg));
        if needs_v {
            args.push(ptr(&mut self.d_exp_avg_sq));
        }
        args.push(ptr(&mut self.d_scratch));
        args.push(ptr(n_i32));
        args
    }

    fn download_and_release(mut self, view: &mut TensorViewMut<'_>) -> Result<(), TritonAotError> {
        let engine = self.engine;
        let has_distinct_master = self.d_master != 0 && self.d_master != self.d_param;
        let p = std::mem::take(&mut self.d_param);
        let m = if has_distinct_master {
            std::mem::take(&mut self.d_master)
        } else {
            self.d_master = 0;
            0
        };
        let g = std::mem::take(&mut self.d_grad);
        let ea = std::mem::take(&mut self.d_exp_avg);
        let easq = std::mem::take(&mut self.d_exp_avg_sq);
        let sc = std::mem::take(&mut self.d_scratch);

        engine.free_ptr(g);
        let r_p = engine.download_free(p, view.param);
        let r_m = match &mut view.master_param {
            Some(mp) if m != 0 => engine.download_free(m, mp),
            _ => Ok(()),
        };
        let r_ea = engine.download_free(ea, view.exp_avg);
        let r_easq = match &mut view.exp_avg_sq {
            Some(v) if easq != 0 => engine.download_free(easq, v),
            _ => {
                if easq != 0 {
                    engine.free_ptr(easq);
                }
                Ok(())
            }
        };
        let r_sc = engine.download_free(sc, view.scratch.as_mut_slice());

        r_p?;
        r_m?;
        r_ea?;
        r_easq?;
        r_sc?;
        Ok(())
    }
}

impl Drop for DeviceStepBuffers<'_> {
    fn drop(&mut self) {
        if self.d_scratch != 0 {
            self.engine.free_ptr(self.d_scratch);
        }
        if self.d_exp_avg_sq != 0 {
            self.engine.free_ptr(self.d_exp_avg_sq);
        }
        if self.d_exp_avg != 0 {
            self.engine.free_ptr(self.d_exp_avg);
        }
        if self.d_grad != 0 {
            self.engine.free_ptr(self.d_grad);
        }
        if self.d_master != 0 && self.d_master != self.d_param {
            self.engine.free_ptr(self.d_master);
        }
        if self.d_param != 0 {
            self.engine.free_ptr(self.d_param);
        }
    }
}

#[inline]
fn ptr<T>(v: &mut T) -> *mut c_void {
    (v as *mut T).cast::<c_void>()
}

fn dtype_code(dt: DTypeMode) -> i64 {
    match dt {
        DTypeMode::Float32 => 0,
        DTypeMode::BFloat16 => 1,
        DTypeMode::Float16 => 2,
    }
}

fn bound_code(b: BoundType) -> i64 {
    match b {
        BoundType::None => 0,
        BoundType::Radial => 1,
        BoundType::Coordinate => 2,
    }
}

fn validate_kernel_config(
    kernel: &LoadedKernel,
    view: &TensorViewMut<'_>,
    config: &LuminaVConfig,
) -> Result<(), TritonAotError> {
    let (_, meta, _) = kernel;
    let effective_sr = config.stochastic_rounding && view.dtype_mode != DTypeMode::Float32;
    let checks = [
        ("bound_mode", bound_code(config.effective_bound_type())),
        ("dtype_mode", dtype_code(view.dtype_mode)),
        ("state_dtype_mode", dtype_code(view.state_dtype_mode)),
        ("use_sr", i64::from(effective_sr)),
        ("cautious", i64::from(config.cautious)),
        ("has_master", i64::from(view.master_param.is_some())),
        ("is_triangle", i64::from(config.buffer.is_triangle_family())),
    ];
    meta.validate_constexprs(&checks)
        .map_err(|detail| TritonAotError::UnsupportedRuntimeConfig {
            kernel: meta.kernel_name.clone(),
            detail,
        })
}

fn dual_coeffs(config: &LuminaVConfig, view: &TensorViewMut<'_>, step: u64) -> (f32, f32) {
    if config.buffer == BufferMode::T2 {
        (
            compute_t2_sigma_coeff(step, config.betas.1, config.lam),
            0.5_f32 * config.lam,
        )
    } else {
        let c = compute_bias_correction(
            step,
            config.betas.0,
            config.betas.1,
            config.eps,
            config.tau,
            view.dtype_mode,
        );
        (c.c1, c.c2)
    }
}

impl TritonAotEngine {
    pub fn step(
        &mut self,
        view: &mut TensorViewMut<'_>,
        config: &LuminaVConfig,
        step: u64,
        param_index: usize,
    ) -> Result<(), TritonAotError> {
        config.validate()?;
        let n = view.param.len();
        if n == 0 {
            return Ok(());
        }

        let needs_v = matches!(config.buffer, BufferMode::Dual | BufferMode::T2);
        if needs_v && view.exp_avg_sq.is_none() {
            return Err(TritonAotError::Core(
                LuminaError::MissingSecondaryStateBuffer,
            ));
        }

        let bound_mode = config.effective_bound_type();
        let needs_reduction = config.cautious || bound_mode == BoundType::Radial;
        let use_two_pass = needs_reduction && (!config.fused_single_pass || step <= 1);

        let (primary_k, secondary_k) =
            self.load_step_kernels(view, config, needs_reduction, use_two_pass)?;

        if use_two_pass || (needs_reduction && config.fused_single_pass && step > 1) {
            view.scratch.reset_accumulators();
        }

        let mut bufs = DeviceStepBuffers::upload(self, view, needs_v)?;

        let mut n_i32 = n as i32;
        let mut total_f32 = n as f32;
        let mut beta1 = config.betas.0;
        let mut beta2 = config.betas.1;
        let (mut c1, mut c2) = if needs_v {
            dual_coeffs(config, view, step)
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
        let (seed_m, _) = derive_spatial_seeds(config.seed, param_index);
        let mut seed_i32 = seed_m;
        let mut step_i32 = step as i32;

        let grid_x = (n as u32).div_ceil(primary_k.1.block_size.max(1));

        if use_two_pass {
            let pass2_k = secondary_k
                .as_ref()
                .expect("two-pass requires pass2 kernel");
            let mut args1 = bufs.base_args(needs_v, &mut n_i32);
            match config.buffer {
                BufferMode::Dual => args1.extend_from_slice(&[
                    ptr(&mut beta1),
                    ptr(&mut beta2),
                    ptr(&mut c1),
                    ptr(&mut c2),
                ]),
                BufferMode::T2 => args1.extend_from_slice(&[
                    ptr(&mut beta1),
                    ptr(&mut beta2),
                    ptr(&mut c1),
                    ptr(&mut c2),
                    ptr(&mut eps),
                ]),
                BufferMode::TdClm | BufferMode::T1 => {
                    args1.extend_from_slice(&[ptr(&mut beta1), ptr(&mut eps), ptr(&mut lam)])
                }
            }
            self.launch_raw(&primary_k, grid_x, &args1)?;

            let mut args2 = bufs.base_args(needs_v, &mut n_i32);
            args2.push(ptr(&mut total_f32));
            match config.buffer {
                BufferMode::Dual => args2.extend_from_slice(&[
                    ptr(&mut beta1),
                    ptr(&mut beta2),
                    ptr(&mut c1),
                    ptr(&mut c2),
                    ptr(&mut lr),
                    ptr(&mut wd),
                ]),
                BufferMode::T2 => args2.extend_from_slice(&[
                    ptr(&mut beta1),
                    ptr(&mut beta2),
                    ptr(&mut c1),
                    ptr(&mut c2),
                    ptr(&mut lr),
                    ptr(&mut wd),
                    ptr(&mut eps),
                ]),
                BufferMode::TdClm | BufferMode::T1 => args2.extend_from_slice(&[
                    ptr(&mut beta1),
                    ptr(&mut lr),
                    ptr(&mut wd),
                    ptr(&mut eps),
                    ptr(&mut lam),
                ]),
            }
            args2.extend_from_slice(&[
                ptr(&mut clamp_min),
                ptr(&mut bound_ratio),
                ptr(&mut seed_i32),
                ptr(&mut step_i32),
            ]);
            self.launch_raw(pass2_k, grid_x, &args2)?;
        } else {
            let mut args = bufs.base_args(needs_v, &mut n_i32);
            match config.buffer {
                BufferMode::Dual => args.extend_from_slice(&[
                    ptr(&mut beta1),
                    ptr(&mut beta2),
                    ptr(&mut c1),
                    ptr(&mut c2),
                    ptr(&mut lr),
                    ptr(&mut wd),
                ]),
                BufferMode::T2 => args.extend_from_slice(&[
                    ptr(&mut beta1),
                    ptr(&mut beta2),
                    ptr(&mut c1),
                    ptr(&mut c2),
                    ptr(&mut lr),
                    ptr(&mut wd),
                    ptr(&mut eps),
                ]),
                BufferMode::TdClm | BufferMode::T1 => args.extend_from_slice(&[
                    ptr(&mut beta1),
                    ptr(&mut lr),
                    ptr(&mut wd),
                    ptr(&mut eps),
                    ptr(&mut lam),
                ]),
            }
            args.extend_from_slice(&[
                ptr(&mut bound_ratio),
                ptr(&mut seed_i32),
                ptr(&mut step_i32),
            ]);
            self.launch_raw(&primary_k, grid_x, &args)?;

            if let Some(ema_k) = secondary_k.as_ref() {
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
        }

        bufs.download_and_release(view)
    }

    fn load_step_kernels(
        &mut self,
        view: &TensorViewMut<'_>,
        config: &LuminaVConfig,
        needs_reduction: bool,
        use_two_pass: bool,
    ) -> Result<(LoadedKernel, Option<LoadedKernel>), TritonAotError> {
        if use_two_pass {
            let (p1_name, p2_name) = match config.buffer {
                BufferMode::Dual => ("_lumina_dual_pass1_kernel", "_lumina_dual_pass2_kernel"),
                BufferMode::T2 => ("_lumina_t2_pass1_kernel", "_lumina_t2_pass2_kernel"),
                BufferMode::TdClm | BufferMode::T1 => {
                    ("_lumina_tdclm_pass1_kernel", "_lumina_tdclm_pass2_kernel")
                }
            };
            let k1 = self.load_kernel_binary(p1_name)?;
            validate_kernel_config(&k1, view, config)?;
            let k2 = self.load_kernel_binary(p2_name)?;
            validate_kernel_config(&k2, view, config)?;
            return Ok((k1, Some(k2)));
        }

        let main_name = match (config.buffer, needs_reduction) {
            (BufferMode::Dual, true) => "_lumina_dual_fused_single_pass_kernel",
            (BufferMode::Dual, false) => "_lumina_dual_single_pass_kernel",
            (BufferMode::T2, true) => "_lumina_t2_fused_single_pass_kernel",
            (BufferMode::T2, false) => "_lumina_t2_single_pass_kernel",
            (BufferMode::TdClm, _) => "_lumina_tdclm_single_pass_kernel",
            (BufferMode::T1, _) => "_lumina_t1_single_pass_kernel",
        };
        let main_k = self.load_kernel_binary(main_name)?;
        validate_kernel_config(&main_k, view, config)?;
        let ema_k = if needs_reduction {
            let k = self.load_kernel_binary("_lumina_post_step_ema_kernel")?;
            validate_kernel_config(&k, view, config)?;
            Some(k)
        } else {
            None
        };
        Ok((main_k, ema_k))
    }
}
