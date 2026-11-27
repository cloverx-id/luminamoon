use std::ffi::{c_char, c_int, c_uint};
use std::ptr;

use crate::{DeviceStepPtrs, TritonAotEngine, TritonAotError};
use luminav_core::{
    BoundType, BufferMode, DTypeMode, LuminaVConfig, MasterWeightMode, ScratchBuffer,
    TensorViewMut, single_step_dual_scalar, single_step_t1_scalar, single_step_t2_scalar,
    single_step_tdclm_scalar,
};

/// # Safety
/// The returned pointer must be freed via `luminav_triton_engine_destroy` if non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luminav_triton_engine_create() -> *mut TritonAotEngine {
    match TritonAotEngine::new() {
        Ok(engine) => Box::into_raw(Box::new(engine)),
        Err(_) => ptr::null_mut(),
    }
}

/// # Safety
/// `engine` must be null or a valid pointer returned by `luminav_triton_engine_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luminav_triton_engine_destroy(engine: *mut TritonAotEngine) {
    if !engine.is_null() {
        unsafe { drop(Box::from_raw(engine)) };
    }
}

/// # Safety
/// If `engine` is non-null, it must point to a valid `TritonAotEngine`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luminav_triton_get_platform(
    engine: *const TritonAotEngine,
) -> *const c_char {
    if engine.is_null() {
        return c"none".as_ptr();
    }
    match unsafe { (*engine).platform_tag() } {
        "cuda" => c"cuda".as_ptr(),
        "rocm" => c"rocm".as_ptr(),
        _ => c"none".as_ptr(),
    }
}

/// # Safety
/// If `engine` is non-null, it must point to a valid `TritonAotEngine`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luminav_triton_get_arch(engine: *const TritonAotEngine) -> *const c_char {
    if engine.is_null() {
        return c"unknown".as_ptr();
    }
    match unsafe { (*engine).arch_tag() } {
        "sm_75" => c"sm_75".as_ptr(),
        "sm_80" => c"sm_80".as_ptr(),
        "sm_86" => c"sm_86".as_ptr(),
        "sm_89" => c"sm_89".as_ptr(),
        "sm_90" => c"sm_90".as_ptr(),
        "sm_100" => c"sm_100".as_ptr(),
        "sm_103" => c"sm_103".as_ptr(),
        "sm_120" => c"sm_120".as_ptr(),
        "gfx908" => c"gfx908".as_ptr(),
        "gfx90a" => c"gfx90a".as_ptr(),
        "gfx942" => c"gfx942".as_ptr(),
        "gfx1100" => c"gfx1100".as_ptr(),
        "gfx1103" => c"gfx1103".as_ptr(),
        _ => c"unknown".as_ptr(),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_c_api_config(
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    tau: f32,
    lam: f32,
    buffer_mode: c_int,
    cautious: c_int,
    bound_mode: c_int,
    bound_ratio: f32,
    has_master: bool,
    has_exp_avg_sq: bool,
    seed: c_uint,
) -> Result<LuminaVConfig, c_int> {
    let buf_mode = BufferMode::try_from_code(i64::from(buffer_mode)).map_err(|_| -3)?;
    let bound_type = BoundType::try_from_code(i64::from(bound_mode)).map_err(|_| -3)?;
    if buf_mode.is_dual_state() && !has_exp_avg_sq {
        return Err(-1);
    }
    let config = LuminaVConfig {
        lr,
        betas: (beta1, beta2),
        eps,
        weight_decay,
        tau,
        lam,
        cautious: cautious != 0,
        cautious_clamp_min: 0.5,
        buffer: buf_mode,
        stochastic_rounding: false,
        bound: bound_type != BoundType::None,
        bound_type,
        bound_ratio,
        master_weights: if has_master {
            MasterWeightMode::Full
        } else {
            MasterWeightMode::None
        },
        ema_decay: buf_mode.default_ema_decay(),
        seed,
        ..Default::default()
    };
    config.validate().map_err(|_| -3)?;
    Ok(config)
}

fn step_scalar_fallback(
    view: &mut TensorViewMut<'_>,
    config: &LuminaVConfig,
    step: u64,
    param_index: usize,
) -> c_int {
    let r = match config.buffer {
        BufferMode::TdClm => single_step_tdclm_scalar(view, config, step, param_index),
        BufferMode::T1 => single_step_t1_scalar(view, config, step, param_index),
        BufferMode::Dual => single_step_dual_scalar(view, config, step, param_index),
        BufferMode::T2 => single_step_t2_scalar(view, config, step, param_index),
    };
    if r.is_ok() { 0 } else { -4 }
}

/// Returns 1 if the requested FP32 configuration matches an embedded GPU AOT kernel profile, else 0.
///
/// # Safety
/// If `engine` is non-null, it must point to a valid `TritonAotEngine`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luminav_triton_supports_config(
    engine: *const TritonAotEngine,
    buffer_mode: c_int,
    cautious: c_int,
    bound_mode: c_int,
    has_master: c_int,
) -> c_int {
    if engine.is_null() {
        return 0;
    }
    let Ok(cfg) = build_c_api_config(
        1e-3,
        0.9,
        0.999,
        1e-8,
        0.0,
        0.8,
        0.3,
        buffer_mode,
        cautious,
        bound_mode,
        0.05,
        has_master != 0,
        true,
        1337,
    ) else {
        return 0;
    };
    let ok = unsafe {
        (*engine).supports_config(
            &cfg,
            DTypeMode::Float32,
            DTypeMode::Float32,
            has_master != 0,
        )
    };
    c_int::from(ok)
}

/// Executes a host-memory FP32 optimization step with explicit `param_index` and optional CPU fallback.
///
/// # Safety
/// Host pointers `param`, `grad`, `exp_avg`, and `scratch` (6 floats) must be valid and non-null.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn luminav_triton_step_f32_ex(
    engine: *mut TritonAotEngine,
    param: *mut f32,
    grad: *const f32,
    exp_avg: *mut f32,
    exp_avg_sq: *mut f32,
    master_param: *mut f32,
    scratch: *mut f32,
    num_elements: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    tau: f32,
    lam: f32,
    buffer_mode: c_int,
    cautious: c_int,
    bound_mode: c_int,
    bound_ratio: f32,
    step: u64,
    seed: c_uint,
    param_index: usize,
    allow_cpu_fallback: c_int,
) -> c_int {
    if engine.is_null()
        || param.is_null()
        || grad.is_null()
        || exp_avg.is_null()
        || scratch.is_null()
    {
        return -1;
    }
    let config = match build_c_api_config(
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        tau,
        lam,
        buffer_mode,
        cautious,
        bound_mode,
        bound_ratio,
        !master_param.is_null(),
        !exp_avg_sq.is_null(),
        seed,
    ) {
        Ok(cfg) => cfg,
        Err(code) => return code,
    };

    let engine_ref = unsafe { &mut *engine };
    let param_slice = unsafe { std::slice::from_raw_parts_mut(param, num_elements) };
    let grad_slice = unsafe { std::slice::from_raw_parts(grad, num_elements) };
    let exp_avg_slice = unsafe { std::slice::from_raw_parts_mut(exp_avg, num_elements) };
    let exp_avg_sq_slice = if config.buffer.is_dual_state() {
        Some(unsafe { std::slice::from_raw_parts_mut(exp_avg_sq, num_elements) })
    } else {
        None
    };
    let master_param_slice = if !master_param.is_null() {
        Some(unsafe { std::slice::from_raw_parts_mut(master_param, num_elements) })
    } else {
        None
    };
    let scratch_slice = unsafe { std::slice::from_raw_parts_mut(scratch, 6) };
    let mut scratch_buf = ScratchBuffer::from_array([
        scratch_slice[0],
        scratch_slice[1],
        scratch_slice[2],
        scratch_slice[3],
        scratch_slice[4],
        scratch_slice[5],
    ]);

    let mut view = TensorViewMut {
        param: param_slice,
        grad: grad_slice,
        exp_avg: exp_avg_slice,
        exp_avg_sq: exp_avg_sq_slice,
        master_param: master_param_slice,
        scratch: &mut scratch_buf,
        dtype_mode: DTypeMode::Float32,
        state_dtype_mode: DTypeMode::Float32,
    };

    let res = match engine_ref.step(&mut view, &config, step, param_index) {
        Ok(()) => 0,
        Err(TritonAotError::UnsupportedRuntimeConfig { .. }) if allow_cpu_fallback != 0 => {
            step_scalar_fallback(&mut view, &config, step, param_index)
        }
        Err(TritonAotError::UnsupportedRuntimeConfig { .. }) => -5,
        Err(_) => -4,
    };

    scratch_slice[0] = scratch_buf.mask_sum();
    scratch_slice[1] = scratch_buf.u_sq_sum();
    scratch_slice[2] = scratch_buf.p_sq_sum();
    scratch_slice[3] = scratch_buf.cautious_scale();
    scratch_slice[4] = scratch_buf.bound_scale();
    scratch_slice[5] = 0.0;
    res
}

/// Executes a host-memory FP32 optimization step (`param_index = 0`, auto CPU fallback on non-AOT flags).
///
/// # Safety
/// Host pointers `param`, `grad`, `exp_avg`, and `scratch` (6 floats) must be valid and non-null.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn luminav_triton_step_f32(
    engine: *mut TritonAotEngine,
    param: *mut f32,
    grad: *const f32,
    exp_avg: *mut f32,
    exp_avg_sq: *mut f32,
    master_param: *mut f32,
    scratch: *mut f32,
    num_elements: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    tau: f32,
    lam: f32,
    buffer_mode: c_int,
    cautious: c_int,
    bound_mode: c_int,
    bound_ratio: f32,
    step: u64,
    seed: c_uint,
) -> c_int {
    unsafe {
        luminav_triton_step_f32_ex(
            engine,
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            master_param,
            scratch,
            num_elements,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            tau,
            lam,
            buffer_mode,
            cautious,
            bound_mode,
            bound_ratio,
            step,
            seed,
            0,
            1,
        )
    }
}

/// Executes a zero-copy GPU-resident FP32 optimization step directly on device pointers (zero PCIe transfer).
///
/// # Safety
/// Device pointers must be valid allocations on the active CUDA/ROCm device.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn luminav_triton_step_f32_device(
    engine: *mut TritonAotEngine,
    d_param: u64,
    d_grad: u64,
    d_exp_avg: u64,
    d_exp_avg_sq: u64,
    d_master_param: u64,
    d_scratch: u64,
    num_elements: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    tau: f32,
    lam: f32,
    buffer_mode: c_int,
    cautious: c_int,
    bound_mode: c_int,
    bound_ratio: f32,
    step: u64,
    seed: c_uint,
    param_index: usize,
) -> c_int {
    if engine.is_null() || d_param == 0 || d_grad == 0 || d_exp_avg == 0 || d_scratch == 0 {
        return -1;
    }
    let config = match build_c_api_config(
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        tau,
        lam,
        buffer_mode,
        cautious,
        bound_mode,
        bound_ratio,
        d_master_param != 0,
        d_exp_avg_sq != 0,
        seed,
    ) {
        Ok(cfg) => cfg,
        Err(code) => return code,
    };
    let bufs = DeviceStepPtrs {
        d_param,
        d_master: d_master_param,
        d_grad,
        d_exp_avg,
        d_exp_avg_sq,
        d_scratch,
    };
    let engine_ref = unsafe { &mut *engine };
    match engine_ref.step_device_ptrs(
        bufs,
        num_elements,
        &config,
        DTypeMode::Float32,
        DTypeMode::Float32,
        d_master_param != 0,
        step,
        param_index,
    ) {
        Ok(()) => 0,
        Err(TritonAotError::UnsupportedRuntimeConfig { .. }) => -5,
        Err(_) => -4,
    }
}
