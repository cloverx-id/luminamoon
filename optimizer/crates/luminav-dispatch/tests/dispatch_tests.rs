use luminav_core::{
    BufferMode, DTypeMode, ExecutionMode, LuminaVConfig, ScratchBuffer, TensorViewMut,
};
use luminav_dispatch::{
    GPU_DISPATCH_THRESHOLD, LuminaDispatcher, LuminaEngine, detect_hardware, step_auto,
};

#[test]
fn test_hardware_detector() {
    let caps = detect_hardware();
    println!(
        "[Dispatch] CPU Backend: {:?} | Vulkan Available: {} (Device: {:?}, Wave32: {})",
        caps.simd_backend, caps.vulkan_available, caps.gpu_device_name, caps.wave32_supported
    );
    assert!(
        matches!(
            caps.simd_backend,
            luminav_simd::SimdBackend::Avx512
                | luminav_simd::SimdBackend::Avx2
                | luminav_simd::SimdBackend::Scalar
        ),
        "A valid CPU SIMD/Scalar backend must be detected"
    );
}

#[test]
fn test_router_threshold_gating_small_tensor_simd() {
    let mut dispatcher = LuminaDispatcher::new();
    let n = 1024;
    let mut param = vec![1.0_f32; n];
    let grad = vec![0.1_f32; n];
    let mut exp_avg = vec![0.0_f32; n];
    let mut exp_avg_sq = vec![0.0_f32; n];
    let mut scratch = ScratchBuffer::new();

    let config = LuminaVConfig {
        execution: ExecutionMode::Auto,
        buffer: BufferMode::Dual,
        ..Default::default()
    };

    let mut view = TensorViewMut {
        param: &mut param,
        master_param: None,
        grad: &grad,
        exp_avg: &mut exp_avg,
        exp_avg_sq: Some(&mut exp_avg_sq),
        scratch: &mut scratch,
        dtype_mode: DTypeMode::Float32,
        state_dtype_mode: DTypeMode::Float32,
    };

    let res = dispatcher.step(&mut view, &config, 1, 0);
    assert!(res.is_ok(), "Small tensor step failed: {:?}", res.err());
    assert_ne!(param[0], 1.0, "Param should have updated");
}

#[test]
fn test_router_threshold_gating_large_tensor_vulkan() {
    let mut dispatcher = LuminaDispatcher::new();
    let n = GPU_DISPATCH_THRESHOLD;
    let mut param = vec![1.0_f32; n];
    let grad = vec![0.1_f32; n];
    let mut exp_avg = vec![0.0_f32; n];
    let mut exp_avg_sq = vec![0.0_f32; n];
    let mut scratch = ScratchBuffer::new();

    let config = LuminaVConfig {
        execution: ExecutionMode::Auto,
        buffer: BufferMode::Dual,
        ..Default::default()
    };

    let mut view = TensorViewMut {
        param: &mut param,
        master_param: None,
        grad: &grad,
        exp_avg: &mut exp_avg,
        exp_avg_sq: Some(&mut exp_avg_sq),
        scratch: &mut scratch,
        dtype_mode: DTypeMode::Float32,
        state_dtype_mode: DTypeMode::Float32,
    };

    let res = dispatcher.step(&mut view, &config, 1, 0);
    assert!(res.is_ok(), "Large tensor step failed: {:?}", res.err());
    assert_ne!(param[0], 1.0, "Param should have updated");
}

#[test]
fn test_global_step_auto() {
    let n = 512;
    let mut param = vec![0.5_f32; n];
    let grad = vec![0.05_f32; n];
    let mut exp_avg = vec![0.0_f32; n];
    let mut scratch = ScratchBuffer::new();

    let config = LuminaVConfig {
        execution: ExecutionMode::Auto,
        buffer: BufferMode::TdClm,
        ..Default::default()
    };

    let mut view = TensorViewMut {
        param: &mut param,
        master_param: None,
        grad: &grad,
        exp_avg: &mut exp_avg,
        exp_avg_sq: None,
        scratch: &mut scratch,
        dtype_mode: DTypeMode::Float32,
        state_dtype_mode: DTypeMode::Float32,
    };

    let res = step_auto(&mut view, &config, 1, 0);
    assert!(res.is_ok(), "step_auto failed: {:?}", res.err());
    assert_ne!(param[0], 0.5, "Param should have updated");
}
