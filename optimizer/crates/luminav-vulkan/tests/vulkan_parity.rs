use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use luminav_core::{
    BoundType, BufferMode, DTypeMode, LuminaVConfig, MasterWeightMode, ScratchBuffer,
    TensorViewMut, single_step_dual_scalar, single_step_tdclm_scalar,
};
use luminav_vulkan::VulkanLuminaEngine;

const N_ELEMENTS: usize = 131_072;
const STEPS: u64 = 5;
const PARITY_TOLERANCE: f32 = 1.5e-6;

static SHARED_VULKAN_ENGINE: OnceLock<Option<Mutex<VulkanLuminaEngine>>> = OnceLock::new();

fn shared_vulkan_engine() -> Option<&'static Mutex<VulkanLuminaEngine>> {
    SHARED_VULKAN_ENGINE
        .get_or_init(|| VulkanLuminaEngine::new().ok().map(Mutex::new))
        .as_ref()
}

macro_rules! get_vulkan_engine_or_skip {
    () => {
        match shared_vulkan_engine() {
            Some(m) => m.lock().unwrap_or_else(|e| e.into_inner()),
            None => {
                eprintln!("[Skip] Vulkan device unavailable on this runner");
                return;
            }
        }
    };
}

fn init_test_tensors(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut param = vec![0.0_f32; n];
    let mut grad = vec![0.0_f32; n];
    for i in 0..n {
        let x = (i as f32) * 0.001;
        param[i] = (x.sin() * 0.5) + ((x * 0.3).cos() * 0.2);
        grad[i] = ((x * 1.7).sin() * 0.02) - ((x * 0.5).cos() * 0.01);
    }
    (param, grad)
}

#[test]
fn test_vulkan_device_init_and_uma_allocation() {
    let engine = get_vulkan_engine_or_skip!();
    println!(
        "[Vulkan] Target Device: {} | Wave32 Supported: {}",
        engine.context.device_name, engine.context.wave32_supported
    );

    let mut buf = engine
        .allocate_buffer::<f32>(1024)
        .expect("UMA allocation failed");
    assert_eq!(buf.len(), 1024);
    assert!(buf.device_address() > 0);

    for i in 0..1024 {
        buf.as_slice_mut()[i] = (i as f32) * std::f32::consts::PI;
    }
    for i in 0..1024 {
        assert_eq!(buf.as_slice()[i], (i as f32) * std::f32::consts::PI);
    }
}

#[test]
fn test_vulkan_parity_tdclm_fp32_131072_5steps() {
    let mut engine = get_vulkan_engine_or_skip!();
    let (init_p, init_g) = init_test_tensors(N_ELEMENTS);

    let config = LuminaVConfig {
        lr: 1e-3,
        betas: (0.9, 0.999),
        eps: 1e-8,
        weight_decay: 0.01,
        tau: 0.8,
        lam: 0.3,
        cautious: true,
        cautious_clamp_min: 0.5,
        buffer: BufferMode::TdClm,
        centered_defect: true,
        stochastic_rounding: false,
        bound: true,
        bound_type: BoundType::Coordinate,
        bound_ratio: 0.05,
        master_weights: MasterWeightMode::None,
        execution: luminav_core::ExecutionMode::Auto,
        fused_single_pass: true,
        ema_decay: 0.95,
        fsdp_sync: false,
        seed: 42,
    };

    let mut ref_p = init_p.clone();
    let ref_g = init_g.clone();
    let mut ref_exp_avg = vec![0.0_f32; N_ELEMENTS];
    let mut ref_scratch = ScratchBuffer::new();

    let t_cpu = Instant::now();
    for step in 1..=STEPS {
        let mut view = TensorViewMut {
            param: &mut ref_p,
            master_param: None,
            grad: &ref_g,
            exp_avg: &mut ref_exp_avg,
            exp_avg_sq: None,
            scratch: &mut ref_scratch,
            dtype_mode: DTypeMode::Float32,
            state_dtype_mode: DTypeMode::Float32,
        };
        single_step_tdclm_scalar(&mut view, &config, step, 0).expect("CPU TD-CLM failed");
    }
    let cpu_dur = t_cpu.elapsed();

    let mut gpu_p = engine
        .allocate_buffer::<f32>(N_ELEMENTS)
        .expect("Allocate param");
    let mut gpu_g = engine
        .allocate_buffer::<f32>(N_ELEMENTS)
        .expect("Allocate grad");
    let mut gpu_exp_avg = engine
        .allocate_buffer::<f32>(N_ELEMENTS)
        .expect("Allocate exp_avg");
    let mut gpu_scratch = engine.allocate_buffer::<f32>(6).expect("Allocate scratch");

    gpu_p.copy_from_slice(&init_p);
    gpu_g.copy_from_slice(&init_g);
    gpu_exp_avg.as_slice_mut().fill(0.0);
    gpu_scratch.copy_from_slice(ScratchBuffer::new().as_slice());

    let t_gpu = Instant::now();
    for step in 1..=STEPS {
        engine
            .step_tdclm_uma(
                &mut gpu_p,
                None,
                &gpu_g,
                &mut gpu_exp_avg,
                &mut gpu_scratch,
                &config,
                DTypeMode::Float32,
                step,
                0,
            )
            .expect("GPU TD-CLM step failed");
    }
    let gpu_dur = t_gpu.elapsed();

    let mut max_err_p = 0.0_f32;
    let mut max_err_m = 0.0_f32;
    let p_slice = gpu_p.as_slice();
    let m_slice = gpu_exp_avg.as_slice();

    for i in 0..N_ELEMENTS {
        let err_p = (p_slice[i] - ref_p[i]).abs();
        let err_m = (m_slice[i] - ref_exp_avg[i]).abs();
        if err_p > max_err_p {
            max_err_p = err_p;
        }
        if err_m > max_err_m {
            max_err_m = err_m;
        }
    }

    let speedup = (cpu_dur.as_secs_f64() / gpu_dur.as_secs_f64().max(1e-9)) as f32;
    println!(
        "[Vulkan TD-CLM FP32] max_err(p): {:e}, max_err(m): {:e} | CPU: {:?}, GPU: {:?} (speedup: {:.2}x)",
        max_err_p, max_err_m, cpu_dur, gpu_dur, speedup
    );

    assert!(
        max_err_p <= PARITY_TOLERANCE,
        "TD-CLM FP32 param error {max_err_p:e} exceeds tolerance {PARITY_TOLERANCE:e}"
    );
    assert!(
        max_err_m <= PARITY_TOLERANCE,
        "TD-CLM FP32 exp_avg error {max_err_m:e} exceeds tolerance {PARITY_TOLERANCE:e}"
    );
}

#[test]
fn test_vulkan_parity_dual_fp32_131072_5steps() {
    let mut engine = get_vulkan_engine_or_skip!();
    let (init_p, init_g) = init_test_tensors(N_ELEMENTS);

    let config = LuminaVConfig {
        lr: 1e-3,
        betas: (0.9, 0.999),
        eps: 1e-8,
        weight_decay: 0.01,
        tau: 0.8,
        lam: 0.3,
        cautious: true,
        cautious_clamp_min: 0.5,
        buffer: BufferMode::Dual,
        centered_defect: true,
        stochastic_rounding: false,
        bound: true,
        bound_type: BoundType::Coordinate,
        bound_ratio: 0.05,
        master_weights: MasterWeightMode::None,
        execution: luminav_core::ExecutionMode::Auto,
        fused_single_pass: true,
        ema_decay: 0.80,
        fsdp_sync: false,
        seed: 42,
    };

    let mut ref_p = init_p.clone();
    let ref_g = init_g.clone();
    let mut ref_exp_avg = vec![0.0_f32; N_ELEMENTS];
    let mut ref_exp_avg_sq = vec![0.0_f32; N_ELEMENTS];
    let mut ref_scratch = ScratchBuffer::new();

    let t_cpu = Instant::now();
    for step in 1..=STEPS {
        let mut view = TensorViewMut {
            param: &mut ref_p,
            master_param: None,
            grad: &ref_g,
            exp_avg: &mut ref_exp_avg,
            exp_avg_sq: Some(&mut ref_exp_avg_sq),
            scratch: &mut ref_scratch,
            dtype_mode: DTypeMode::Float32,
            state_dtype_mode: DTypeMode::Float32,
        };
        single_step_dual_scalar(&mut view, &config, step, 0).expect("CPU Dual failed");
    }
    let cpu_dur = t_cpu.elapsed();

    let mut gpu_p = engine
        .allocate_buffer::<f32>(N_ELEMENTS)
        .expect("Allocate param");
    let mut gpu_g = engine
        .allocate_buffer::<f32>(N_ELEMENTS)
        .expect("Allocate grad");
    let mut gpu_exp_avg = engine
        .allocate_buffer::<f32>(N_ELEMENTS)
        .expect("Allocate exp_avg");
    let mut gpu_exp_avg_sq = engine
        .allocate_buffer::<f32>(N_ELEMENTS)
        .expect("Allocate exp_avg_sq");
    let mut gpu_scratch = engine.allocate_buffer::<f32>(6).expect("Allocate scratch");

    gpu_p.copy_from_slice(&init_p);
    gpu_g.copy_from_slice(&init_g);
    gpu_exp_avg.as_slice_mut().fill(0.0);
    gpu_exp_avg_sq.as_slice_mut().fill(0.0);
    gpu_scratch.copy_from_slice(ScratchBuffer::new().as_slice());

    let t_gpu = Instant::now();
    for step in 1..=STEPS {
        engine
            .step_dual_uma(
                &mut gpu_p,
                None,
                &gpu_g,
                &mut gpu_exp_avg,
                &mut gpu_exp_avg_sq,
                &mut gpu_scratch,
                &config,
                DTypeMode::Float32,
                step,
                0,
            )
            .expect("GPU Dual step failed");
    }
    let gpu_dur = t_gpu.elapsed();

    let mut max_err_p = 0.0_f32;
    let mut max_err_m = 0.0_f32;
    let mut max_err_v = 0.0_f32;
    let p_slice = gpu_p.as_slice();
    let m_slice = gpu_exp_avg.as_slice();
    let v_slice = gpu_exp_avg_sq.as_slice();

    for i in 0..N_ELEMENTS {
        let err_p = (p_slice[i] - ref_p[i]).abs();
        let err_m = (m_slice[i] - ref_exp_avg[i]).abs();
        let err_v = (v_slice[i] - ref_exp_avg_sq[i]).abs();
        if err_p > max_err_p {
            max_err_p = err_p;
        }
        if err_m > max_err_m {
            max_err_m = err_m;
        }
        if err_v > max_err_v {
            max_err_v = err_v;
        }
    }

    let speedup = (cpu_dur.as_secs_f64() / gpu_dur.as_secs_f64().max(1e-9)) as f32;
    println!(
        "[Vulkan Dual FP32] max_err(p): {:e}, max_err(m): {:e}, max_err(v): {:e} | CPU: {:?}, GPU: {:?} (speedup: {:.2}x)",
        max_err_p, max_err_m, max_err_v, cpu_dur, gpu_dur, speedup
    );

    assert!(
        max_err_p <= PARITY_TOLERANCE,
        "Dual FP32 param error {max_err_p:e} exceeds tolerance {PARITY_TOLERANCE:e}"
    );
    assert!(
        max_err_m <= PARITY_TOLERANCE,
        "Dual FP32 exp_avg error {max_err_m:e} exceeds tolerance {PARITY_TOLERANCE:e}"
    );
    assert!(
        max_err_v <= PARITY_TOLERANCE,
        "Dual FP32 exp_avg_sq error {max_err_v:e} exceeds tolerance {PARITY_TOLERANCE:e}"
    );
}

#[test]
fn test_vulkan_parity_tdclm_bf16_131072_5steps() {
    let mut engine = get_vulkan_engine_or_skip!();
    let (init_p, init_g) = init_test_tensors(N_ELEMENTS);

    let config = LuminaVConfig {
        lr: 1e-3,
        betas: (0.9, 0.999),
        eps: 1e-8,
        weight_decay: 0.01,
        tau: 0.8,
        lam: 0.3,
        cautious: true,
        cautious_clamp_min: 0.5,
        buffer: BufferMode::TdClm,
        centered_defect: false,
        stochastic_rounding: false,
        bound: true,
        bound_type: BoundType::Coordinate,
        bound_ratio: 0.05,
        master_weights: MasterWeightMode::None,
        execution: luminav_core::ExecutionMode::Auto,
        fused_single_pass: true,
        ema_decay: 0.95,
        fsdp_sync: false,
        seed: 42,
    };

    let mut ref_p = init_p.clone();
    let ref_g = init_g.clone();
    let mut ref_exp_avg = vec![0.0_f32; N_ELEMENTS];
    let mut ref_scratch = ScratchBuffer::new();

    for step in 1..=STEPS {
        let mut view = TensorViewMut {
            param: &mut ref_p,
            master_param: None,
            grad: &ref_g,
            exp_avg: &mut ref_exp_avg,
            exp_avg_sq: None,
            scratch: &mut ref_scratch,
            dtype_mode: DTypeMode::BFloat16,
            state_dtype_mode: DTypeMode::Float32,
        };
        single_step_tdclm_scalar(&mut view, &config, step, 0).expect("CPU TD-CLM BF16 failed");
    }

    let mut gpu_p = engine
        .allocate_buffer::<f32>(N_ELEMENTS)
        .expect("Allocate param");
    let mut gpu_g = engine
        .allocate_buffer::<f32>(N_ELEMENTS)
        .expect("Allocate grad");
    let mut gpu_exp_avg = engine
        .allocate_buffer::<f32>(N_ELEMENTS)
        .expect("Allocate exp_avg");
    let mut gpu_scratch = engine.allocate_buffer::<f32>(6).expect("Allocate scratch");

    gpu_p.copy_from_slice(&init_p);
    gpu_g.copy_from_slice(&init_g);
    gpu_exp_avg.as_slice_mut().fill(0.0);
    gpu_scratch.copy_from_slice(ScratchBuffer::new().as_slice());

    for step in 1..=STEPS {
        engine
            .step_tdclm_uma(
                &mut gpu_p,
                None,
                &gpu_g,
                &mut gpu_exp_avg,
                &mut gpu_scratch,
                &config,
                DTypeMode::BFloat16,
                step,
                0,
            )
            .expect("GPU TD-CLM BF16 step failed");
    }

    let mut max_err_p = 0.0_f32;
    let p_slice = gpu_p.as_slice();

    for i in 0..N_ELEMENTS {
        let err_p = (p_slice[i] - ref_p[i]).abs();
        if err_p > max_err_p {
            max_err_p = err_p;
        }
    }

    let bf16_tolerance = 2.0e-4;
    assert!(
        max_err_p <= bf16_tolerance,
        "TD-CLM BF16 error {max_err_p:e} exceeds tolerance {bf16_tolerance:e}"
    );
}

#[test]
fn test_vulkan_high_level_step_api() {
    let mut engine = get_vulkan_engine_or_skip!();
    let (mut param, grad) = init_test_tensors(1024);
    let mut exp_avg = vec![0.0_f32; 1024];
    let mut exp_avg_sq = vec![0.0_f32; 1024];
    let mut scratch = ScratchBuffer::new();

    let config = LuminaVConfig::default();

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

    let res = engine.step(&mut view, &config, 1, 0);
    assert!(
        res.is_ok(),
        "Engine high-level step failed: {:?}",
        res.err()
    );
}

#[test]
fn test_vulkan_parity_dual_fp16_native_bf16_state_131072_5steps() {
    let mut engine = get_vulkan_engine_or_skip!();
    let (init_p, mut init_g) = init_test_tensors(N_ELEMENTS);
    for g in init_g.iter_mut().step_by(2) {
        *g = 1.0e-4_f32;
    }

    let config = LuminaVConfig {
        lr: 1e-3,
        betas: (0.9, 0.999),
        eps: 1e-8,
        weight_decay: 0.01,
        tau: 0.8,
        lam: 0.3,
        cautious: true,
        cautious_clamp_min: 0.5,
        buffer: BufferMode::Dual,
        centered_defect: false,
        stochastic_rounding: false,
        bound: true,
        bound_type: BoundType::Coordinate,
        bound_ratio: 0.05,
        master_weights: MasterWeightMode::None,
        execution: luminav_core::ExecutionMode::Auto,
        fused_single_pass: true,
        ema_decay: 0.80,
        fsdp_sync: false,
        seed: 42,
    };

    let mut ref_p = init_p.clone();
    let ref_g = init_g.clone();
    let mut ref_exp_avg = vec![0.0_f32; N_ELEMENTS];
    let mut ref_exp_avg_sq = vec![0.0_f32; N_ELEMENTS];
    let mut ref_scratch = ScratchBuffer::new();

    for step in 1..=STEPS {
        let mut view = TensorViewMut {
            param: &mut ref_p,
            master_param: None,
            grad: &ref_g,
            exp_avg: &mut ref_exp_avg,
            exp_avg_sq: Some(&mut ref_exp_avg_sq),
            scratch: &mut ref_scratch,
            dtype_mode: DTypeMode::Float16,
            state_dtype_mode: DTypeMode::Float16,
        };
        single_step_dual_scalar(&mut view, &config, step, 0).expect("CPU Dual FP16 failed");
    }

    let mut gpu_p = engine
        .allocate_buffer::<f32>(N_ELEMENTS)
        .expect("Allocate param");
    let mut gpu_g = engine
        .allocate_buffer::<f32>(N_ELEMENTS)
        .expect("Allocate grad");
    let mut gpu_exp_avg = engine
        .allocate_buffer::<f32>(N_ELEMENTS)
        .expect("Allocate exp_avg");
    let mut gpu_exp_avg_sq = engine
        .allocate_buffer::<f32>(N_ELEMENTS)
        .expect("Allocate exp_avg_sq");
    let mut gpu_scratch = engine.allocate_buffer::<f32>(6).expect("Allocate scratch");

    gpu_p.copy_from_slice(&init_p);
    gpu_g.copy_from_slice(&init_g);
    gpu_exp_avg.as_slice_mut().fill(0.0);
    gpu_exp_avg_sq.as_slice_mut().fill(0.0);
    gpu_scratch.copy_from_slice(ScratchBuffer::new().as_slice());

    for step in 1..=STEPS {
        engine
            .step_dual_uma_with_state_dtype(
                &mut gpu_p,
                None,
                &gpu_g,
                &mut gpu_exp_avg,
                &mut gpu_exp_avg_sq,
                &mut gpu_scratch,
                &config,
                DTypeMode::Float16,
                DTypeMode::Float16,
                step,
                0,
            )
            .expect("GPU Dual FP16 step failed");
    }

    let p_slice = gpu_p.as_slice();
    let m_slice = gpu_exp_avg.as_slice();
    let v_slice = gpu_exp_avg_sq.as_slice();

    let mut max_err_p = 0.0_f32;
    let mut max_err_m = 0.0_f32;
    let mut max_err_v = 0.0_f32;

    for i in 0..N_ELEMENTS {
        if i % 2 == 0 {
            assert!(
                v_slice[i] > 0.0,
                "Vulkan FP16 exp_avg_sq underflowed at index {i} despite native BF16 promotion"
            );
        }
        let err_p = (p_slice[i] - ref_p[i]).abs();
        let err_m = (m_slice[i] - ref_exp_avg[i]).abs();
        let err_v = (v_slice[i] - ref_exp_avg_sq[i]).abs();
        if err_p > max_err_p {
            max_err_p = err_p;
        }
        if err_m > max_err_m {
            max_err_m = err_m;
        }
        if err_v > max_err_v {
            max_err_v = err_v;
        }
    }

    println!(
        "[Vulkan Dual FP16 + Native BF16 State] max_err(p): {:e}, max_err(m): {:e}, max_err(v): {:e}",
        max_err_p, max_err_m, max_err_v
    );

    let fp16_tolerance = 5.0e-4;
    assert!(
        max_err_p <= fp16_tolerance,
        "Dual FP16 param error {max_err_p:e} exceeds tolerance {fp16_tolerance:e}"
    );
    assert!(
        max_err_m <= PARITY_TOLERANCE,
        "Dual FP16 exp_avg error {max_err_m:e} exceeds tolerance {PARITY_TOLERANCE:e}"
    );
    assert!(
        max_err_v <= PARITY_TOLERANCE,
        "Dual FP16 exp_avg_sq error {max_err_v:e} exceeds tolerance {PARITY_TOLERANCE:e}"
    );
}

#[test]
fn test_vulkan_parity_tdclm_fp16_native_bf16_state_131072_5steps() {
    let mut engine = get_vulkan_engine_or_skip!();
    let (init_p, mut init_g) = init_test_tensors(N_ELEMENTS);
    for g in init_g.iter_mut().step_by(2) {
        *g = 1.0e-4_f32;
    }

    let config = LuminaVConfig {
        lr: 1e-3,
        betas: (0.9, 0.999),
        eps: 1e-8,
        weight_decay: 0.01,
        tau: 0.8,
        lam: 0.3,
        cautious: true,
        cautious_clamp_min: 0.5,
        buffer: BufferMode::TdClm,
        centered_defect: false,
        stochastic_rounding: false,
        bound: true,
        bound_type: BoundType::Coordinate,
        bound_ratio: 0.05,
        master_weights: MasterWeightMode::None,
        execution: luminav_core::ExecutionMode::Auto,
        fused_single_pass: true,
        ema_decay: 0.95,
        fsdp_sync: false,
        seed: 42,
    };

    let mut ref_p = init_p.clone();
    let ref_g = init_g.clone();
    let mut ref_exp_avg = vec![0.0_f32; N_ELEMENTS];
    let mut ref_scratch = ScratchBuffer::new();

    for step in 1..=STEPS {
        let mut view = TensorViewMut {
            param: &mut ref_p,
            master_param: None,
            grad: &ref_g,
            exp_avg: &mut ref_exp_avg,
            exp_avg_sq: None,
            scratch: &mut ref_scratch,
            dtype_mode: DTypeMode::Float16,
            state_dtype_mode: DTypeMode::Float16,
        };
        single_step_tdclm_scalar(&mut view, &config, step, 0).expect("CPU TD-CLM FP16 failed");
    }

    let mut gpu_p = engine
        .allocate_buffer::<f32>(N_ELEMENTS)
        .expect("Allocate param");
    let mut gpu_g = engine
        .allocate_buffer::<f32>(N_ELEMENTS)
        .expect("Allocate grad");
    let mut gpu_exp_avg = engine
        .allocate_buffer::<f32>(N_ELEMENTS)
        .expect("Allocate exp_avg");
    let mut gpu_scratch = engine.allocate_buffer::<f32>(6).expect("Allocate scratch");

    gpu_p.copy_from_slice(&init_p);
    gpu_g.copy_from_slice(&init_g);
    gpu_exp_avg.as_slice_mut().fill(0.0);
    gpu_scratch.copy_from_slice(ScratchBuffer::new().as_slice());

    for step in 1..=STEPS {
        engine
            .step_tdclm_uma_with_state_dtype(
                &mut gpu_p,
                None,
                &gpu_g,
                &mut gpu_exp_avg,
                &mut gpu_scratch,
                &config,
                DTypeMode::Float16,
                DTypeMode::Float16,
                step,
                0,
            )
            .expect("GPU TD-CLM FP16 step failed");
    }

    let p_slice = gpu_p.as_slice();
    let m_slice = gpu_exp_avg.as_slice();
    let mut max_err_p = 0.0_f32;
    let mut max_err_m = 0.0_f32;

    for i in 0..N_ELEMENTS {
        if i % 2 == 0 {
            assert!(
                m_slice[i] > 0.0,
                "Vulkan TD-CLM FP16 exp_avg underflowed at index {i}"
            );
        }
        let err_p = (p_slice[i] - ref_p[i]).abs();
        let err_m = (m_slice[i] - ref_exp_avg[i]).abs();
        if err_p > max_err_p {
            max_err_p = err_p;
        }
        if err_m > max_err_m {
            max_err_m = err_m;
        }
    }

    println!(
        "[Vulkan TD-CLM FP16 + Native BF16 State] max_err(p): {:e}, max_err(m): {:e}",
        max_err_p, max_err_m
    );

    let fp16_tolerance = 5.0e-4;
    assert!(
        max_err_p <= fp16_tolerance,
        "TD-CLM FP16 param error {max_err_p:e} exceeds tolerance {fp16_tolerance:e}"
    );
    assert!(
        max_err_m <= PARITY_TOLERANCE,
        "TD-CLM FP16 exp_avg error {max_err_m:e} exceeds tolerance {PARITY_TOLERANCE:e}"
    );
}
