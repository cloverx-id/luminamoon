use std::time::Instant;

use luminav_core::{
    _prng, BoundType, BufferMode, DTypeMode, LuminaVConfig, MasterWeightMode, ParamState,
    ScratchBuffer, TensorViewMut, quantize_value, single_step_dual_scalar,
    single_step_tdclm_scalar, stochastic_round_bf16, stochastic_round_fp16,
};
use luminav_simd::{
    SimdBackend, detect_simd_backend, dual_step_parallel, dual_step_parallel_chunked, step_dual,
    step_tdclm, tdclm_step_parallel, tdclm_step_parallel_chunked,
};

const NUM_ELEMENTS: usize = 131_072;
const NUM_STEPS: usize = 5;
const TOLERANCE: f32 = 1.5e-6_f32;

fn generate_pseudo_random_tensor(n: usize, seed: i32, scale: f32, dtype: DTypeMode) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let r = _prng(seed, i as i32);
            let unit = (r as f64 / f64::from(i32::MAX)) as f32;
            let val = unit * scale;
            quantize_value(val, seed, i as i32, dtype, false)
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut max_err = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        assert!(
            x.is_finite() && y.is_finite(),
            "Non-finite value encountered: {x} vs {y}"
        );
        let diff = (x - y).abs();
        if diff > max_err {
            max_err = diff;
        }
    }
    max_err
}

fn assert_scratch_parity(ref_s: &ScratchBuffer, simd_s: &ScratchBuffer, label: &str) -> f32 {
    let mut max_err = 0.0_f32;
    for (idx, (r, s)) in ref_s
        .as_slice()
        .iter()
        .zip(simd_s.as_slice().iter())
        .enumerate()
    {
        let diff = (r - s).abs();
        if diff > max_err {
            max_err = diff;
        }
        let tol = if idx == 1 || idx == 2 || idx == 5 {
            (r.abs() * 1.0e-5).max(TOLERANCE)
        } else {
            TOLERANCE
        };
        assert!(
            diff <= tol,
            "[{label}] ScratchBuffer[{idx}] mismatch: ref={r}, simd={s}, diff={diff} > {tol}"
        );
    }
    max_err
}

#[test]
fn test_differential_parity_tdclm_fp32_131072_5steps() {
    let backend = detect_simd_backend();
    println!(
        "[TD-CLM FP32] Active CPU SIMD Backend: {} (N = {}, Steps = {})",
        backend.name(),
        NUM_ELEMENTS,
        NUM_STEPS
    );

    let config = LuminaVConfig::builder()
        .lr(1.0e-3)
        .betas(0.9, 0.999)
        .weight_decay(0.01)
        .buffer(BufferMode::TdClm)
        .bound_type(BoundType::Radial)
        .centered_defect(true)
        .cautious(true)
        .master_weights(MasterWeightMode::None)
        .stochastic_rounding(true)
        .seed(1337)
        .build()
        .expect("valid TD-CLM FP32 config");

    let init_params = generate_pseudo_random_tensor(NUM_ELEMENTS, 42, 0.5, DTypeMode::Float32);
    let grads_per_step: Vec<Vec<f32>> = (0..NUM_STEPS)
        .map(|s| {
            generate_pseudo_random_tensor(NUM_ELEMENTS, 1000 + s as i32, 0.05, DTypeMode::Float32)
        })
        .collect();

    let mut p_ref = init_params.clone();
    let mut state_ref = ParamState::new(&p_ref, BufferMode::TdClm, false);

    let mut p_simd = init_params;
    let mut state_simd = ParamState::new(&p_simd, BufferMode::TdClm, false);

    let t0_ref = Instant::now();
    for grad in &grads_per_step {
        let step = state_ref.increment_step();
        let mut view = TensorViewMut {
            param: &mut p_ref,
            master_param: None,
            grad,
            exp_avg: &mut state_ref.exp_avg,
            exp_avg_sq: None,
            scratch: &mut state_ref.scratch,
            dtype_mode: DTypeMode::Float32,
            state_dtype_mode: DTypeMode::Float32,
        };
        single_step_tdclm_scalar(&mut view, &config, step, 0).expect("scalar TD-CLM step");
    }
    let elapsed_ref = t0_ref.elapsed();

    let t0_simd = Instant::now();
    for grad in &grads_per_step {
        let step = state_simd.increment_step();
        let mut view = TensorViewMut {
            param: &mut p_simd,
            master_param: None,
            grad,
            exp_avg: &mut state_simd.exp_avg,
            exp_avg_sq: None,
            scratch: &mut state_simd.scratch,
            dtype_mode: DTypeMode::Float32,
            state_dtype_mode: DTypeMode::Float32,
        };
        step_tdclm(&mut view, &config, step, 0).expect("SIMD TD-CLM step");
    }
    let elapsed_simd = t0_simd.elapsed();

    let param_err = max_abs_diff(&p_ref, &p_simd);
    let m_err = max_abs_diff(&state_ref.exp_avg, &state_simd.exp_avg);
    let scratch_err = assert_scratch_parity(&state_ref.scratch, &state_simd.scratch, "TD-CLM FP32");

    println!(
        "[TD-CLM FP32] max_err(param) = {:.3e}, max_err(exp_avg) = {:.3e}, max_err(scratch) = {:.3e} | ref = {:?}, simd = {:?} (speedup = {:.2}x)",
        param_err,
        m_err,
        scratch_err,
        elapsed_ref,
        elapsed_simd,
        elapsed_ref.as_secs_f64() / elapsed_simd.as_secs_f64().max(1.0e-9)
    );

    assert!(
        param_err <= TOLERANCE,
        "TD-CLM FP32 param max_abs_err {param_err} > {TOLERANCE}"
    );
    assert!(
        m_err <= TOLERANCE,
        "TD-CLM FP32 exp_avg max_abs_err {m_err} > {TOLERANCE}"
    );
}

#[test]
fn test_differential_parity_tdclm_bf16_131072_5steps() {
    let config = LuminaVConfig::builder()
        .lr(1.0e-3)
        .betas(0.9, 0.999)
        .weight_decay(0.01)
        .buffer(BufferMode::TdClm)
        .bound_type(BoundType::Radial)
        .centered_defect(true)
        .cautious(true)
        .master_weights(MasterWeightMode::None)
        .stochastic_rounding(true)
        .seed(2026)
        .build()
        .expect("valid TD-CLM BF16 config");

    let init_params = generate_pseudo_random_tensor(NUM_ELEMENTS, 77, 0.5, DTypeMode::BFloat16);
    let grads_per_step: Vec<Vec<f32>> = (0..NUM_STEPS)
        .map(|s| {
            generate_pseudo_random_tensor(NUM_ELEMENTS, 2000 + s as i32, 0.05, DTypeMode::BFloat16)
        })
        .collect();

    let mut p_ref = init_params.clone();
    let mut state_ref = ParamState::new(&p_ref, BufferMode::TdClm, false);

    let mut p_simd = init_params;
    let mut state_simd = ParamState::new(&p_simd, BufferMode::TdClm, false);

    let t0_ref = Instant::now();
    for grad in &grads_per_step {
        let step = state_ref.increment_step();
        let mut view = TensorViewMut {
            param: &mut p_ref,
            master_param: None,
            grad,
            exp_avg: &mut state_ref.exp_avg,
            exp_avg_sq: None,
            scratch: &mut state_ref.scratch,
            dtype_mode: DTypeMode::BFloat16,
            state_dtype_mode: DTypeMode::BFloat16,
        };
        single_step_tdclm_scalar(&mut view, &config, step, 0).expect("scalar TD-CLM BF16 step");
    }
    let elapsed_ref = t0_ref.elapsed();

    let t0_simd = Instant::now();
    for grad in &grads_per_step {
        let step = state_simd.increment_step();
        let mut view = TensorViewMut {
            param: &mut p_simd,
            master_param: None,
            grad,
            exp_avg: &mut state_simd.exp_avg,
            exp_avg_sq: None,
            scratch: &mut state_simd.scratch,
            dtype_mode: DTypeMode::BFloat16,
            state_dtype_mode: DTypeMode::BFloat16,
        };
        step_tdclm(&mut view, &config, step, 0).expect("SIMD TD-CLM BF16 step");
    }
    let elapsed_simd = t0_simd.elapsed();

    let param_err = max_abs_diff(&p_ref, &p_simd);
    let m_err = max_abs_diff(&state_ref.exp_avg, &state_simd.exp_avg);
    let scratch_err = assert_scratch_parity(&state_ref.scratch, &state_simd.scratch, "TD-CLM BF16");

    println!(
        "[TD-CLM BF16] max_err(param) = {:.3e}, max_err(exp_avg) = {:.3e}, max_err(scratch) = {:.3e} | ref = {:?}, simd = {:?} (speedup = {:.2}x)",
        param_err,
        m_err,
        scratch_err,
        elapsed_ref,
        elapsed_simd,
        elapsed_ref.as_secs_f64() / elapsed_simd.as_secs_f64().max(1.0e-9)
    );

    assert!(
        param_err <= TOLERANCE,
        "TD-CLM BF16 param max_abs_err {param_err} > {TOLERANCE}"
    );
    assert!(
        m_err <= TOLERANCE,
        "TD-CLM BF16 exp_avg max_abs_err {m_err} > {TOLERANCE}"
    );
}

#[test]
fn test_differential_parity_dual_fp32_131072_5steps() {
    let config = LuminaVConfig::builder()
        .lr(8.0e-4)
        .betas(0.9, 0.999)
        .weight_decay(0.01)
        .buffer(BufferMode::Dual)
        .bound_type(BoundType::Coordinate)
        .centered_defect(true)
        .cautious(true)
        .master_weights(MasterWeightMode::None)
        .stochastic_rounding(true)
        .seed(9001)
        .build()
        .expect("valid Dual FP32 config");

    let init_params = generate_pseudo_random_tensor(NUM_ELEMENTS, 123, 0.5, DTypeMode::Float32);
    let grads_per_step: Vec<Vec<f32>> = (0..NUM_STEPS)
        .map(|s| {
            generate_pseudo_random_tensor(NUM_ELEMENTS, 3000 + s as i32, 0.05, DTypeMode::Float32)
        })
        .collect();

    let mut p_ref = init_params.clone();
    let mut state_ref = ParamState::new(&p_ref, BufferMode::Dual, false);

    let mut p_simd = init_params;
    let mut state_simd = ParamState::new(&p_simd, BufferMode::Dual, false);

    let t0_ref = Instant::now();
    for grad in &grads_per_step {
        let step = state_ref.increment_step();
        let mut view = TensorViewMut {
            param: &mut p_ref,
            master_param: None,
            grad,
            exp_avg: &mut state_ref.exp_avg,
            exp_avg_sq: state_ref.exp_avg_sq.as_deref_mut(),
            scratch: &mut state_ref.scratch,
            dtype_mode: DTypeMode::Float32,
            state_dtype_mode: DTypeMode::Float32,
        };
        single_step_dual_scalar(&mut view, &config, step, 0).expect("scalar Dual FP32 step");
    }
    let elapsed_ref = t0_ref.elapsed();

    let t0_simd = Instant::now();
    for grad in &grads_per_step {
        let step = state_simd.increment_step();
        let mut view = TensorViewMut {
            param: &mut p_simd,
            master_param: None,
            grad,
            exp_avg: &mut state_simd.exp_avg,
            exp_avg_sq: state_simd.exp_avg_sq.as_deref_mut(),
            scratch: &mut state_simd.scratch,
            dtype_mode: DTypeMode::Float32,
            state_dtype_mode: DTypeMode::Float32,
        };
        step_dual(&mut view, &config, step, 0).expect("SIMD Dual FP32 step");
    }
    let elapsed_simd = t0_simd.elapsed();

    let param_err = max_abs_diff(&p_ref, &p_simd);
    let m_err = max_abs_diff(&state_ref.exp_avg, &state_simd.exp_avg);
    let v_err = max_abs_diff(
        state_ref.exp_avg_sq.as_ref().expect("v_ref"),
        state_simd.exp_avg_sq.as_ref().expect("v_simd"),
    );
    let scratch_err = assert_scratch_parity(&state_ref.scratch, &state_simd.scratch, "Dual FP32");

    println!(
        "[Dual FP32]   max_err(param) = {:.3e}, max_err(exp_avg) = {:.3e}, max_err(exp_avg_sq) = {:.3e}, max_err(scratch) = {:.3e} | ref = {:?}, simd = {:?} (speedup = {:.2}x)",
        param_err,
        m_err,
        v_err,
        scratch_err,
        elapsed_ref,
        elapsed_simd,
        elapsed_ref.as_secs_f64() / elapsed_simd.as_secs_f64().max(1.0e-9)
    );

    assert!(
        param_err <= TOLERANCE,
        "Dual FP32 param max_abs_err {param_err} > {TOLERANCE}"
    );
    assert!(
        m_err <= TOLERANCE,
        "Dual FP32 exp_avg max_abs_err {m_err} > {TOLERANCE}"
    );
    assert!(
        v_err <= TOLERANCE,
        "Dual FP32 exp_avg_sq max_abs_err {v_err} > {TOLERANCE}"
    );
}

#[test]
fn test_differential_parity_dual_bf16_131072_5steps() {
    let config = LuminaVConfig::builder()
        .lr(8.0e-4)
        .betas(0.9, 0.999)
        .weight_decay(0.01)
        .buffer(BufferMode::Dual)
        .bound_type(BoundType::Radial)
        .centered_defect(true)
        .cautious(true)
        .master_weights(MasterWeightMode::None)
        .stochastic_rounding(true)
        .seed(31415)
        .build()
        .expect("valid Dual BF16 config");

    let init_params = generate_pseudo_random_tensor(NUM_ELEMENTS, 555, 0.5, DTypeMode::BFloat16);
    let grads_per_step: Vec<Vec<f32>> = (0..NUM_STEPS)
        .map(|s| {
            generate_pseudo_random_tensor(NUM_ELEMENTS, 4000 + s as i32, 0.05, DTypeMode::BFloat16)
        })
        .collect();

    let mut p_ref = init_params.clone();
    let mut state_ref = ParamState::new(&p_ref, BufferMode::Dual, false);

    let mut p_simd = init_params;
    let mut state_simd = ParamState::new(&p_simd, BufferMode::Dual, false);

    let t0_ref = Instant::now();
    for grad in &grads_per_step {
        let step = state_ref.increment_step();
        let mut view = TensorViewMut {
            param: &mut p_ref,
            master_param: None,
            grad,
            exp_avg: &mut state_ref.exp_avg,
            exp_avg_sq: state_ref.exp_avg_sq.as_deref_mut(),
            scratch: &mut state_ref.scratch,
            dtype_mode: DTypeMode::BFloat16,
            state_dtype_mode: DTypeMode::BFloat16,
        };
        single_step_dual_scalar(&mut view, &config, step, 0).expect("scalar Dual BF16 step");
    }
    let elapsed_ref = t0_ref.elapsed();

    let t0_simd = Instant::now();
    for grad in &grads_per_step {
        let step = state_simd.increment_step();
        let mut view = TensorViewMut {
            param: &mut p_simd,
            master_param: None,
            grad,
            exp_avg: &mut state_simd.exp_avg,
            exp_avg_sq: state_simd.exp_avg_sq.as_deref_mut(),
            scratch: &mut state_simd.scratch,
            dtype_mode: DTypeMode::BFloat16,
            state_dtype_mode: DTypeMode::BFloat16,
        };
        step_dual(&mut view, &config, step, 0).expect("SIMD Dual BF16 step");
    }
    let elapsed_simd = t0_simd.elapsed();

    let param_err = max_abs_diff(&p_ref, &p_simd);
    let m_err = max_abs_diff(&state_ref.exp_avg, &state_simd.exp_avg);
    let v_err = max_abs_diff(
        state_ref.exp_avg_sq.as_ref().expect("v_ref"),
        state_simd.exp_avg_sq.as_ref().expect("v_simd"),
    );
    let scratch_err = assert_scratch_parity(&state_ref.scratch, &state_simd.scratch, "Dual BF16");

    println!(
        "[Dual BF16]   max_err(param) = {:.3e}, max_err(exp_avg) = {:.3e}, max_err(exp_avg_sq) = {:.3e}, max_err(scratch) = {:.3e} | ref = {:?}, simd = {:?} (speedup = {:.2}x)",
        param_err,
        m_err,
        v_err,
        scratch_err,
        elapsed_ref,
        elapsed_simd,
        elapsed_ref.as_secs_f64() / elapsed_simd.as_secs_f64().max(1.0e-9)
    );

    assert!(
        param_err <= TOLERANCE,
        "Dual BF16 param max_abs_err {param_err} > {TOLERANCE}"
    );
    assert!(
        m_err <= TOLERANCE,
        "Dual BF16 exp_avg max_abs_err {m_err} > {TOLERANCE}"
    );
    assert!(
        v_err <= TOLERANCE,
        "Dual BF16 exp_avg_sq max_abs_err {v_err} > {TOLERANCE}"
    );
}

#[test]
fn test_differential_parity_avx2_and_rayon_parallel_131072_5steps() {
    let config = LuminaVConfig::builder()
        .lr(1.0e-3)
        .buffer(BufferMode::Dual)
        .bound_type(BoundType::Radial)
        .centered_defect(true)
        .cautious(true)
        .master_weights(MasterWeightMode::Full)
        .stochastic_rounding(true)
        .seed(777)
        .build()
        .expect("valid config");

    let init_params = generate_pseudo_random_tensor(NUM_ELEMENTS, 888, 0.5, DTypeMode::BFloat16);
    let grads_per_step: Vec<Vec<f32>> = (0..NUM_STEPS)
        .map(|s| {
            generate_pseudo_random_tensor(NUM_ELEMENTS, 5000 + s as i32, 0.05, DTypeMode::Float32)
        })
        .collect();

    let mut p_ref = init_params.clone();
    let mut state_ref = ParamState::new(&p_ref, BufferMode::Dual, true);

    let mut p_avx2 = init_params.clone();
    let mut state_avx2 = ParamState::new(&p_avx2, BufferMode::Dual, true);

    let mut p_par = init_params;
    let mut state_par = ParamState::new(&p_par, BufferMode::Dual, true);

    for grad in &grads_per_step {
        let step_r = state_ref.increment_step();
        let mut view_r = TensorViewMut {
            param: &mut p_ref,
            master_param: state_ref.master_param.as_deref_mut(),
            grad,
            exp_avg: &mut state_ref.exp_avg,
            exp_avg_sq: state_ref.exp_avg_sq.as_deref_mut(),
            scratch: &mut state_ref.scratch,
            dtype_mode: DTypeMode::BFloat16,
            state_dtype_mode: DTypeMode::Float32,
        };
        single_step_dual_scalar(&mut view_r, &config, step_r, 0).expect("ref step");

        let step_a2 = state_avx2.increment_step();
        let mut view_a2 = TensorViewMut {
            param: &mut p_avx2,
            master_param: state_avx2.master_param.as_deref_mut(),
            grad,
            exp_avg: &mut state_avx2.exp_avg,
            exp_avg_sq: state_avx2.exp_avg_sq.as_deref_mut(),
            scratch: &mut state_avx2.scratch,
            dtype_mode: DTypeMode::BFloat16,
            state_dtype_mode: DTypeMode::Float32,
        };
        dual_step_parallel(&mut view_a2, &config, step_a2, 0, SimdBackend::Avx2)
            .expect("avx2 parallel step");

        let step_p = state_par.increment_step();
        let mut view_p = TensorViewMut {
            param: &mut p_par,
            master_param: state_par.master_param.as_deref_mut(),
            grad,
            exp_avg: &mut state_par.exp_avg,
            exp_avg_sq: state_par.exp_avg_sq.as_deref_mut(),
            scratch: &mut state_par.scratch,
            dtype_mode: DTypeMode::BFloat16,
            state_dtype_mode: DTypeMode::Float32,
        };
        dual_step_parallel(&mut view_p, &config, step_p, 0, detect_simd_backend())
            .expect("rayon auto step");
    }

    assert!(max_abs_diff(&p_ref, &p_avx2) <= TOLERANCE);
    assert!(max_abs_diff(&p_ref, &p_par) <= TOLERANCE);
    assert!(max_abs_diff(&state_ref.exp_avg, &state_avx2.exp_avg) <= TOLERANCE);
    assert!(max_abs_diff(&state_ref.exp_avg, &state_par.exp_avg) <= TOLERANCE);
    assert_scratch_parity(&state_ref.scratch, &state_avx2.scratch, "AVX2 Parallel");
    assert_scratch_parity(&state_ref.scratch, &state_par.scratch, "AVX-512 Parallel");

    let tdclm_cfg = LuminaVConfig::builder()
        .buffer(BufferMode::TdClm)
        .build()
        .expect("tdclm cfg");
    let mut p_td = vec![0.25_f32; 1024];
    let g_td = vec![0.01_f32; 1024];
    let mut st_td = ParamState::new(&p_td, BufferMode::TdClm, false);
    let step_td = st_td.increment_step();
    let mut view_td = TensorViewMut {
        param: &mut p_td,
        master_param: None,
        grad: &g_td,
        exp_avg: &mut st_td.exp_avg,
        exp_avg_sq: None,
        scratch: &mut st_td.scratch,
        dtype_mode: DTypeMode::Float32,
        state_dtype_mode: DTypeMode::Float32,
    };
    tdclm_step_parallel(&mut view_td, &tdclm_cfg, step_td, 0, detect_simd_backend())
        .expect("tdclm parallel step");
}

#[test]
fn test_scalar_parallel_chunked_exact_parity() {
    let n = 2048;
    let chunk_size = 128;

    // 1. TD-CLM (2-pass: cautious = true, radial bound)
    let config_tdclm = LuminaVConfig::builder()
        .lr(1.0e-3)
        .betas(0.9, 0.999)
        .buffer(BufferMode::TdClm)
        .bound_type(BoundType::Radial)
        .centered_defect(true)
        .cautious(true)
        .build()
        .expect("config tdclm");

    let init_p = generate_pseudo_random_tensor(n, 123, 0.5, DTypeMode::Float32);
    let grad = generate_pseudo_random_tensor(n, 456, 0.05, DTypeMode::Float32);

    let mut p_ref = init_p.clone();
    let mut state_ref = ParamState::new(&p_ref, BufferMode::TdClm, false);
    let mut view_ref = TensorViewMut {
        param: &mut p_ref,
        master_param: None,
        grad: &grad,
        exp_avg: &mut state_ref.exp_avg,
        exp_avg_sq: None,
        scratch: &mut state_ref.scratch,
        dtype_mode: DTypeMode::Float32,
        state_dtype_mode: DTypeMode::Float32,
    };
    single_step_tdclm_scalar(&mut view_ref, &config_tdclm, 1, 0).expect("scalar ref");

    let mut p_par = init_p.clone();
    let mut state_par = ParamState::new(&p_par, BufferMode::TdClm, false);
    let mut view_par = TensorViewMut {
        param: &mut p_par,
        master_param: None,
        grad: &grad,
        exp_avg: &mut state_par.exp_avg,
        exp_avg_sq: None,
        scratch: &mut state_par.scratch,
        dtype_mode: DTypeMode::Float32,
        state_dtype_mode: DTypeMode::Float32,
    };
    tdclm_step_parallel(&mut view_par, &config_tdclm, 1, 0, SimdBackend::Scalar)
        .expect("scalar parallel");

    assert!(max_abs_diff(&p_ref, &p_par) <= 1e-6);
    assert!(max_abs_diff(&state_ref.exp_avg, &state_par.exp_avg) <= 1e-6);
    assert_scratch_parity(
        &state_ref.scratch,
        &state_par.scratch,
        "Scalar Parallel TD-CLM",
    );

    // Also test multi-chunk Rayon partitioning
    let mut p_chunked = init_p;
    let mut state_chunked = ParamState::new(&p_chunked, BufferMode::TdClm, false);
    let mut view_chunked = TensorViewMut {
        param: &mut p_chunked,
        master_param: None,
        grad: &grad,
        exp_avg: &mut state_chunked.exp_avg,
        exp_avg_sq: None,
        scratch: &mut state_chunked.scratch,
        dtype_mode: DTypeMode::Float32,
        state_dtype_mode: DTypeMode::Float32,
    };
    tdclm_step_parallel_chunked(
        &mut view_chunked,
        &config_tdclm,
        1,
        0,
        SimdBackend::Scalar,
        chunk_size,
    )
    .expect("scalar chunked parallel");

    assert!(max_abs_diff(&p_ref, &p_chunked) <= 1e-6);
    assert!(max_abs_diff(&state_ref.exp_avg, &state_chunked.exp_avg) <= 1e-6);
    assert!((state_ref.scratch.u_sq_sum() - state_chunked.scratch.u_sq_sum()).abs() <= 1e-3);

    // 2. Dual-CLM (2-pass: cautious = true, radial bound)
    let config_dual = LuminaVConfig::builder()
        .lr(1.0e-3)
        .betas(0.9, 0.999)
        .buffer(BufferMode::Dual)
        .bound_type(BoundType::Radial)
        .centered_defect(true)
        .cautious(true)
        .build()
        .expect("config dual");

    let init_p_dual = generate_pseudo_random_tensor(n, 789, 0.5, DTypeMode::Float32);
    let grad_dual = generate_pseudo_random_tensor(n, 101, 0.05, DTypeMode::Float32);

    let mut p_ref_dual = init_p_dual.clone();
    let mut state_ref_dual = ParamState::new(&p_ref_dual, BufferMode::Dual, false);
    let mut view_ref_dual = TensorViewMut {
        param: &mut p_ref_dual,
        master_param: None,
        grad: &grad_dual,
        exp_avg: &mut state_ref_dual.exp_avg,
        exp_avg_sq: state_ref_dual.exp_avg_sq.as_deref_mut(),
        scratch: &mut state_ref_dual.scratch,
        dtype_mode: DTypeMode::Float32,
        state_dtype_mode: DTypeMode::Float32,
    };
    single_step_dual_scalar(&mut view_ref_dual, &config_dual, 1, 0).expect("scalar dual ref");

    let mut p_par_dual = init_p_dual.clone();
    let mut state_par_dual = ParamState::new(&p_par_dual, BufferMode::Dual, false);
    let mut view_par_dual = TensorViewMut {
        param: &mut p_par_dual,
        master_param: None,
        grad: &grad_dual,
        exp_avg: &mut state_par_dual.exp_avg,
        exp_avg_sq: state_par_dual.exp_avg_sq.as_deref_mut(),
        scratch: &mut state_par_dual.scratch,
        dtype_mode: DTypeMode::Float32,
        state_dtype_mode: DTypeMode::Float32,
    };
    dual_step_parallel(&mut view_par_dual, &config_dual, 1, 0, SimdBackend::Scalar)
        .expect("scalar dual parallel");

    assert!(max_abs_diff(&p_ref_dual, &p_par_dual) <= 1e-6);
    assert!(max_abs_diff(&state_ref_dual.exp_avg, &state_par_dual.exp_avg) <= 1e-6);
    assert_scratch_parity(
        &state_ref_dual.scratch,
        &state_par_dual.scratch,
        "Scalar Parallel Dual-CLM",
    );

    // Also test multi-chunk Rayon partitioning for Dual-CLM
    let mut p_chunked_dual = init_p_dual;
    let mut state_chunked_dual = ParamState::new(&p_chunked_dual, BufferMode::Dual, false);
    let mut view_chunked_dual = TensorViewMut {
        param: &mut p_chunked_dual,
        master_param: None,
        grad: &grad_dual,
        exp_avg: &mut state_chunked_dual.exp_avg,
        exp_avg_sq: state_chunked_dual.exp_avg_sq.as_deref_mut(),
        scratch: &mut state_chunked_dual.scratch,
        dtype_mode: DTypeMode::Float32,
        state_dtype_mode: DTypeMode::Float32,
    };
    dual_step_parallel_chunked(
        &mut view_chunked_dual,
        &config_dual,
        1,
        0,
        SimdBackend::Scalar,
        chunk_size,
    )
    .expect("scalar dual chunked parallel");

    assert!(max_abs_diff(&p_ref_dual, &p_chunked_dual) <= 1e-6);
    assert!(max_abs_diff(&state_ref_dual.exp_avg, &state_chunked_dual.exp_avg) <= 1e-6);
    assert!(
        (state_ref_dual.scratch.u_sq_sum() - state_chunked_dual.scratch.u_sq_sum()).abs() <= 1e-3
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_bitwise_prng_and_sr_fp16_bf16_avx512_exact() {
    if detect_simd_backend() != SimdBackend::Avx512 {
        return;
    }
    unsafe {
        use std::arch::x86_64::{
            _mm512_loadu_ps, _mm512_set_epi32, _mm512_set1_epi32, _mm512_storeu_ps,
            _mm512_storeu_si512,
        };
        let seed = 1337_i32;
        let v_seed = _mm512_set1_epi32(seed);
        let v_offsets = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

        let v_prng = luminav_simd::avx512::prng_avx512(v_seed, v_offsets);
        let mut prng_lanes = [0_i32; 16];
        _mm512_storeu_si512(prng_lanes.as_mut_ptr().cast(), v_prng);
        for (idx, &val) in prng_lanes.iter().enumerate() {
            assert_eq!(val, _prng(seed, idx as i32));
        }

        let mut inputs = [0.0_f32; 16];
        for (i, item) in inputs.iter_mut().enumerate() {
            *item = match i {
                0 => 0.0_f32,
                1 => -0.0_f32,
                2 => f32::from_bits(102 << 23),
                3 => f32::from_bits((105 << 23) | 0x123456),
                4 => -f32::from_bits((110 << 23) | 0x654321),
                _ => (i as f32) * 0.123_456_7_f32 - 0.5_f32,
            };
        }

        let v_in = _mm512_loadu_ps(inputs.as_ptr());
        let v_bf16 = luminav_simd::avx512::stochastic_round_bf16_avx512(v_in, v_seed, v_offsets);
        let v_fp16 = luminav_simd::avx512::stochastic_round_fp16_avx512(v_in, v_seed, v_offsets);
        let v_rne_fp16 = luminav_simd::avx512::round_to_nearest_fp16_avx512(v_in);

        let mut bf16_out = [0.0_f32; 16];
        let mut fp16_out = [0.0_f32; 16];
        let mut rne_fp16_out = [0.0_f32; 16];
        _mm512_storeu_ps(bf16_out.as_mut_ptr(), v_bf16);
        _mm512_storeu_ps(fp16_out.as_mut_ptr(), v_fp16);
        _mm512_storeu_ps(rne_fp16_out.as_mut_ptr(), v_rne_fp16);

        for i in 0..16 {
            assert_eq!(
                bf16_out[i].to_bits(),
                stochastic_round_bf16(inputs[i], seed, i as i32).to_bits()
            );
            assert_eq!(
                fp16_out[i].to_bits(),
                stochastic_round_fp16(inputs[i], seed, i as i32).to_bits()
            );
            assert_eq!(
                rne_fp16_out[i].to_bits(),
                luminav_core::round_to_nearest_fp16(inputs[i]).to_bits()
            );
        }

        let normal_inputs: [f32; 16] = std::array::from_fn(|i| match i {
            0 => 0.0_f32,
            1 => -0.0_f32,
            _ => (i as f32) * 0.137_931_f32 - 1.25_f32,
        });
        let v_norm = _mm512_loadu_ps(normal_inputs.as_ptr());
        let v_norm_rne = luminav_simd::avx512::round_to_nearest_fp16_avx512(v_norm);
        let mut norm_out = [0.0_f32; 16];
        _mm512_storeu_ps(norm_out.as_mut_ptr(), v_norm_rne);
        for i in 0..16 {
            assert_eq!(
                norm_out[i].to_bits(),
                luminav_core::round_to_nearest_fp16(normal_inputs[i]).to_bits()
            );
        }
    }
}
