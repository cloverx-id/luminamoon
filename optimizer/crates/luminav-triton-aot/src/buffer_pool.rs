use std::ffi::c_void;

use luminav_core::{
    BoundType, BufferMode, DTypeMode, LuminaVConfig, compute_bias_correction,
    compute_t2_sigma_coeff,
};

use crate::{ActiveDriver, KernelMetadata, TritonAotError};

pub(crate) fn dtype_code(dt: DTypeMode) -> i64 {
    match dt {
        DTypeMode::Float32 => 0,
        DTypeMode::BFloat16 => 1,
        DTypeMode::Float16 => 2,
    }
}

pub(crate) fn bound_code(b: BoundType) -> i64 {
    match b {
        BoundType::None => 0,
        BoundType::Radial => 1,
        BoundType::Coordinate => 2,
    }
}

pub(crate) fn validate_kernel_config_raw(
    meta: &KernelMetadata,
    dtype_mode: DTypeMode,
    state_dtype_mode: DTypeMode,
    has_master: bool,
    config: &LuminaVConfig,
) -> Result<(), TritonAotError> {
    if config.uses_rtv() {
        return Err(TritonAotError::UnsupportedRuntimeConfig {
            kernel: meta.kernel_name.clone(),
            detail: "AOT binaries only target variance_type='standard'".to_string(),
        });
    }
    if config.cautious_weight_decay {
        return Err(TritonAotError::UnsupportedRuntimeConfig {
            kernel: meta.kernel_name.clone(),
            detail: "AOT binaries only target cautious_weight_decay=false".to_string(),
        });
    }
    let bound = config.effective_bound_type();
    let effective_sr = if state_dtype_mode == DTypeMode::Float32
        && (has_master || dtype_mode == DTypeMode::Float32)
    {
        false
    } else {
        config.stochastic_rounding
    };
    let mut checks = vec![
        ("bound_mode", bound_code(bound)),
        ("dtype_mode", dtype_code(dtype_mode)),
        ("state_dtype_mode", dtype_code(state_dtype_mode)),
        ("use_sr", i64::from(effective_sr)),
        ("cautious", i64::from(config.cautious)),
        ("has_master", i64::from(has_master)),
    ];
    if meta.kernel_name != "_lumina_post_step_ema_kernel" || bound == BoundType::Radial {
        checks.push(("is_triangle", i64::from(config.buffer.is_triangle_family())));
    }
    meta.validate_constexprs(&checks)
        .map_err(|detail| TritonAotError::UnsupportedRuntimeConfig {
            kernel: meta.kernel_name.clone(),
            detail,
        })
}

pub(crate) fn dual_coeffs(config: &LuminaVConfig, dtype_mode: DTypeMode, step: u64) -> (f32, f32) {
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
            dtype_mode,
        );
        (c.c1, c.c2)
    }
}

pub(crate) fn kernel_names_for_step(
    config: &LuminaVConfig,
    needs_reduction: bool,
    use_two_pass: bool,
) -> (&'static str, Option<&'static str>) {
    if use_two_pass {
        let (p1, p2) = match config.buffer {
            BufferMode::Dual => ("_lumina_dual_pass1_kernel", "_lumina_dual_pass2_kernel"),
            BufferMode::T2 => ("_lumina_t2_pass1_kernel", "_lumina_t2_pass2_kernel"),
            BufferMode::TdClm => ("_lumina_tdclm_pass1_kernel", "_lumina_tdclm_pass2_kernel"),
            BufferMode::T1 => ("_lumina_tdclm_pass1_kernel", "_lumina_t1_pass2_kernel"),
        };
        return (p1, Some(p2));
    }
    let main_name = match (config.buffer, needs_reduction) {
        (BufferMode::Dual, true) => "_lumina_dual_fused_single_pass_kernel",
        (BufferMode::Dual, false) => "_lumina_dual_single_pass_kernel",
        (BufferMode::T2, true) => "_lumina_t2_fused_single_pass_kernel",
        (BufferMode::T2, false) => "_lumina_t2_single_pass_kernel",
        (BufferMode::TdClm, _) => "_lumina_tdclm_single_pass_kernel",
        (BufferMode::T1, _) => "_lumina_t1_single_pass_kernel",
    };
    let ema_name = if needs_reduction {
        Some("_lumina_post_step_ema_kernel")
    } else {
        None
    };
    (main_name, ema_name)
}

#[derive(Debug, Clone, Copy)]
pub struct DeviceStepPtrs {
    pub d_param: u64,
    pub d_master: u64,
    pub d_grad: u64,
    pub d_exp_avg: u64,
    pub d_exp_avg_sq: u64,
    pub d_scratch: u64,
}

impl DeviceStepPtrs {
    pub(crate) fn base_args(&mut self, needs_v: bool, n_i32: &mut i32) -> Vec<*mut c_void> {
        let mut args = Vec::with_capacity(20);
        args.push((&mut self.d_param as *mut u64).cast());
        args.push((&mut self.d_master as *mut u64).cast());
        args.push((&mut self.d_grad as *mut u64).cast());
        args.push((&mut self.d_exp_avg as *mut u64).cast());
        if needs_v {
            args.push((&mut self.d_exp_avg_sq as *mut u64).cast());
        }
        args.push((&mut self.d_scratch as *mut u64).cast());
        args.push((n_i32 as *mut i32).cast());
        args
    }
}

#[derive(Default)]
pub(crate) struct DeviceBufferSlot {
    pub(crate) dptr: u64,
    cap_bytes: usize,
}

impl DeviceBufferSlot {
    pub(crate) fn ensure_capacity(
        &mut self,
        driver: &ActiveDriver,
        min_bytes: usize,
    ) -> Result<u64, TritonAotError> {
        let req = min_bytes.max(16);
        if self.dptr == 0 || self.cap_bytes < req {
            let new_ptr = driver.alloc_bytes(req)?;
            if self.dptr != 0 {
                driver.free_ptr(self.dptr);
            }
            self.dptr = new_ptr;
            self.cap_bytes = req;
        }
        Ok(self.dptr)
    }

    pub(crate) fn upload(
        &mut self,
        driver: &ActiveDriver,
        slice: &[f32],
    ) -> Result<u64, TritonAotError> {
        let dptr = self.ensure_capacity(driver, std::mem::size_of_val(slice))?;
        driver.upload_into(dptr, slice)?;
        Ok(dptr)
    }

    pub(crate) fn release(&mut self, driver: &ActiveDriver) {
        if self.dptr != 0 {
            driver.free_ptr(self.dptr);
            self.dptr = 0;
            self.cap_bytes = 0;
        }
    }
}

#[derive(Default)]
pub(crate) struct DeviceStagingPool {
    pub(crate) param: DeviceBufferSlot,
    pub(crate) master: DeviceBufferSlot,
    pub(crate) grad: DeviceBufferSlot,
    pub(crate) exp_avg: DeviceBufferSlot,
    pub(crate) exp_avg_sq: DeviceBufferSlot,
    pub(crate) scratch: DeviceBufferSlot,
}

impl DeviceStagingPool {
    pub(crate) fn release_all(&mut self, driver: &ActiveDriver) {
        self.scratch.release(driver);
        self.exp_avg_sq.release(driver);
        self.exp_avg.release(driver);
        self.grad.release(driver);
        self.master.release(driver);
        self.param.release(driver);
    }
}
