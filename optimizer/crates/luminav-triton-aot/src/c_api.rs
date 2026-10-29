use std::ffi::{c_char, c_int, c_uint};
use std::ptr;

use crate::TritonAotEngine;
use luminav_core::{
    BoundType, BufferMode, DTypeMode, LuminaVConfig, MasterWeightMode, ScratchBuffer, TensorViewMut,
};

/// Creates a new standalone Triton AOT engine instance.
///
/// # Safety
/// The caller must ensure that the returned pointer is eventually freed using
/// `luminav_triton_engine_destroy`. Returns null if driver probing fails.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luminav_triton_engine_create() -> *mut TritonAotEngine {
    match TritonAotEngine::new() {
        Ok(engine) => Box::into_raw(Box::new(engine)),
        Err(_) => ptr::null_mut(),
    }
}

/// Destroys a standalone Triton AOT engine instance.
///
/// # Safety
/// `engine` must be a valid pointer obtained from `luminav_triton_engine_create` or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luminav_triton_engine_destroy(engine: *mut TritonAotEngine) {
    if !engine.is_null() {
        unsafe {
            drop(Box::from_raw(engine));
        }
    }
}

/// Returns the active platform tag: "cuda", "rocm", or "none".
///
/// # Safety
/// If `engine` is non-null, it must point to an active `TritonAotEngine`.
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

/// Returns the active GPU architecture tag.
///
/// # Safety
/// If `engine` is non-null, it must point to an active `TritonAotEngine`.
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
        _ => c"unknown".as_ptr(),
    }
}

/// Executes a single optimization step on FP32 parameter buffer using GPU Triton AOT kernel.
///
/// # Safety
/// Pointers `param`, `grad`, `exp_avg`, and `scratch` must be valid, non-null, and point to
/// memory buffers with at least `num_elements` elements (or 6 elements for `scratch`).
#[unsafe(no_mangle)]
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
    if engine.is_null()
        || param.is_null()
        || grad.is_null()
        || exp_avg.is_null()
        || scratch.is_null()
    {
        return -1;
    }

    let engine_ref = unsafe { &mut *engine };
    let param_slice = unsafe { std::slice::from_raw_parts_mut(param, num_elements) };
    let grad_slice = unsafe { std::slice::from_raw_parts(grad, num_elements) };
    let exp_avg_slice = unsafe { std::slice::from_raw_parts_mut(exp_avg, num_elements) };
    let exp_avg_sq_slice = if !exp_avg_sq.is_null() && buffer_mode == 2 {
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

    let bound_type = match bound_mode {
        1 => BoundType::Radial,
        2 => BoundType::Coordinate,
        _ => BoundType::None,
    };

    let buf_mode = match buffer_mode {
        1 => BufferMode::TdClm,
        _ => BufferMode::Dual,
    };

    let master_weights = if !master_param.is_null() {
        MasterWeightMode::Full
    } else {
        MasterWeightMode::None
    };

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
        bound: bound_mode > 0,
        bound_type,
        bound_ratio,
        master_weights,
        ema_decay: buf_mode.default_ema_decay(),
        seed,
        ..Default::default()
    };

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

    let res = match engine_ref.step(&mut view, &config, step, 0) {
        Ok(_) => 0,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c_api_null_pointer_checks() {
        unsafe {
            assert_eq!(
                luminav_triton_step_f32(
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    0,
                    1e-3,
                    0.9,
                    0.999,
                    1e-8,
                    0.0,
                    0.8,
                    0.3,
                    1,
                    1,
                    2,
                    0.05,
                    1,
                    1337
                ),
                -1
            );
            let platform = luminav_triton_get_platform(ptr::null());
            assert_eq!(std::ffi::CStr::from_ptr(platform).to_str().unwrap(), "none");
            let arch = luminav_triton_get_arch(ptr::null());
            assert_eq!(std::ffi::CStr::from_ptr(arch).to_str().unwrap(), "unknown");
            luminav_triton_engine_destroy(ptr::null_mut());
        }
    }
}
