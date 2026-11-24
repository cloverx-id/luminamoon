#![allow(unused_imports)]
use std::time::Instant;

use luminav_core::{
    _prng, BoundType, BufferMode, DTypeMode, LuminaVConfig, MasterWeightMode, ParamState,
    ScratchBuffer, TensorViewMut, VarianceType, quantize_value, single_step_dual_scalar,
    single_step_t1_scalar, single_step_t2_scalar, single_step_tdclm_scalar,
};
use luminav_simd::{
    SimdBackend, detect_simd_backend, dual_step_parallel, dual_step_parallel_chunked, step_dual,
    step_t1, step_t2, step_tdclm, tdclm_step_parallel, tdclm_step_parallel_chunked,
};

const NUM_ELEMENTS: usize = 131_072;
const NUM_STEPS: usize = 5;
const TOLERANCE: f32 = 1.5e-6_f32;

fn generate_tensor(n: usize, seed: i32, scale: f32, dtype: DTypeMode) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let r = _prng(seed, i as i32);
            let unit = (r as f64 / f64::from(i32::MAX)) as f32;
            quantize_value(unit * scale, seed, i as i32, dtype, false)
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
        max_err = max_err.max((x - y).abs());
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
        max_err = max_err.max(diff);
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

fn run_multi_step_parity(
    label: &str,
    config: &LuminaVConfig,
    dtype: DTypeMode,
    param_seed: i32,
    grad_seed_base: i32,
) {
    let init_params = generate_tensor(NUM_ELEMENTS, param_seed, 0.5, dtype);
    let grads: Vec<Vec<f32>> = (0..NUM_STEPS)
        .map(|s| generate_tensor(NUM_ELEMENTS, grad_seed_base + s as i32, 0.05, dtype))
        .collect();

    let is_dual = matches!(config.buffer, BufferMode::Dual | BufferMode::T2);
    let mut p_ref = init_params.clone();
    let mut st_ref = ParamState::new(&p_ref, config.buffer, false);
    let mut p_simd = init_params;
    let mut st_simd = ParamState::new(&p_simd, config.buffer, false);

    let t0_ref = Instant::now();
    for grad in &grads {
        let step = st_ref.increment_step();
        let mut view = TensorViewMut {
            param: &mut p_ref,
            master_param: None,
            grad,
            exp_avg: &mut st_ref.exp_avg,
            exp_avg_sq: st_ref.exp_avg_sq.as_deref_mut(),
            scratch: &mut st_ref.scratch,
            dtype_mode: dtype,
            state_dtype_mode: dtype,
        };
        match config.buffer {
            BufferMode::Dual => {
                single_step_dual_scalar(&mut view, config, step, 0).expect("scalar dual")
            }
            BufferMode::T2 => single_step_t2_scalar(&mut view, config, step, 0).expect("scalar t2"),
            BufferMode::TdClm => {
                single_step_tdclm_scalar(&mut view, config, step, 0).expect("scalar tdclm")
            }
            BufferMode::T1 => single_step_t1_scalar(&mut view, config, step, 0).expect("scalar t1"),
        }
    }
    let elapsed_ref = t0_ref.elapsed();

    let t0_simd = Instant::now();
    for grad in &grads {
        let step = st_simd.increment_step();
        let mut view = TensorViewMut {
            param: &mut p_simd,
            master_param: None,
            grad,
            exp_avg: &mut st_simd.exp_avg,
            exp_avg_sq: st_simd.exp_avg_sq.as_deref_mut(),
            scratch: &mut st_simd.scratch,
            dtype_mode: dtype,
            state_dtype_mode: dtype,
        };
        match config.buffer {
            BufferMode::Dual => step_dual(&mut view, config, step, 0).expect("SIMD dual"),
            BufferMode::T2 => step_t2(&mut view, config, step, 0).expect("SIMD t2"),
            BufferMode::TdClm => step_tdclm(&mut view, config, step, 0).expect("SIMD tdclm"),
            BufferMode::T1 => step_t1(&mut view, config, step, 0).expect("SIMD t1"),
        }
    }
    let elapsed_simd = t0_simd.elapsed();

    let param_err = max_abs_diff(&p_ref, &p_simd);
    let m_err = max_abs_diff(&st_ref.exp_avg, &st_simd.exp_avg);
    let scratch_err = assert_scratch_parity(&st_ref.scratch, &st_simd.scratch, label);

    println!(
        "[{label}] backend={} max_err(p)={:.3e} max_err(m)={:.3e} max_err(s)={:.3e} speedup={:.2}x",
        detect_simd_backend().name(),
        param_err,
        m_err,
        scratch_err,
        elapsed_ref.as_secs_f64() / elapsed_simd.as_secs_f64().max(1.0e-9)
    );
    assert!(param_err <= TOLERANCE, "{label} param err {param_err}");
    assert!(m_err <= TOLERANCE, "{label} exp_avg err {m_err}");
    if is_dual {
        let v_err = max_abs_diff(
            st_ref.exp_avg_sq.as_ref().expect("v_ref"),
            st_simd.exp_avg_sq.as_ref().expect("v_simd"),
        );
        assert!(v_err <= TOLERANCE, "{label} exp_avg_sq err {v_err}");
    }
}

#[test]
fn test_differential_parity_tdclm_fp32_131072_5steps() {
    let config = LuminaVConfig::builder()
        .lr(1.0e-3)
        .betas(0.9, 0.999)
        .weight_decay(0.01)
        .buffer(BufferMode::TdClm)
        .bound_type(BoundType::Radial)
        .cautious(true)
        .master_weights(MasterWeightMode::None)
        .stochastic_rounding(true)
        .seed(1337)
        .build()
        .expect("valid TD-CLM FP32 config");
    run_multi_step_parity("TD-CLM FP32", &config, DTypeMode::Float32, 42, 1000);
}

#[test]
fn test_differential_parity_tdclm_bf16_131072_5steps() {
    let config = LuminaVConfig::builder()
        .lr(1.0e-3)
        .betas(0.9, 0.999)
        .weight_decay(0.01)
        .buffer(BufferMode::TdClm)
        .bound_type(BoundType::Radial)
        .cautious(true)
        .master_weights(MasterWeightMode::None)
        .stochastic_rounding(true)
        .seed(2026)
        .build()
        .expect("valid TD-CLM BF16 config");
    run_multi_step_parity("TD-CLM BF16", &config, DTypeMode::BFloat16, 77, 2000);
}

#[test]
fn test_differential_parity_dual_fp32_131072_5steps() {
    let config = LuminaVConfig::builder()
        .lr(8.0e-4)
        .betas(0.9, 0.999)
        .weight_decay(0.01)
        .buffer(BufferMode::Dual)
        .bound_type(BoundType::Coordinate)
        .cautious(true)
        .master_weights(MasterWeightMode::None)
        .stochastic_rounding(true)
        .seed(9001)
        .build()
        .expect("valid Dual FP32 config");
    run_multi_step_parity("Dual FP32", &config, DTypeMode::Float32, 123, 3000);
}

#[test]
fn test_differential_parity_dual_bf16_131072_5steps() {
    let config = LuminaVConfig::builder()
        .lr(8.0e-4)
        .betas(0.9, 0.999)
        .weight_decay(0.01)
        .buffer(BufferMode::Dual)
        .bound_type(BoundType::Radial)
        .cautious(true)
        .master_weights(MasterWeightMode::None)
        .stochastic_rounding(true)
        .seed(31415)
        .build()
        .expect("valid Dual BF16 config");
    run_multi_step_parity("Dual BF16", &config, DTypeMode::BFloat16, 555, 4000);
}

#[test]
fn test_differential_parity_avx2_and_rayon_parallel_131072_5steps() {
    let config = LuminaVConfig::builder()
        .lr(1.0e-3)
        .buffer(BufferMode::Dual)
        .bound_type(BoundType::Radial)
        .cautious(true)
        .master_weights(MasterWeightMode::Full)
        .stochastic_rounding(true)
        .seed(777)
        .build()
        .expect("valid config");

    let init_params = generate_tensor(NUM_ELEMENTS, 888, 0.5, DTypeMode::BFloat16);
    let grads: Vec<Vec<f32>> = (0..NUM_STEPS)
        .map(|s| generate_tensor(NUM_ELEMENTS, 5000 + s as i32, 0.05, DTypeMode::Float32))
        .collect();

    let mut p_ref = init_params.clone();
    let mut st_ref = ParamState::new(&p_ref, BufferMode::Dual, true);
    let mut p_alt = init_params.clone();
    let mut st_alt = ParamState::new(&p_alt, BufferMode::Dual, true);
    let mut p_par = init_params;
    let mut st_par = ParamState::new(&p_par, BufferMode::Dual, true);

    #[cfg(target_arch = "x86_64")]
    let alt_backend = SimdBackend::Avx2;
    #[cfg(not(target_arch = "x86_64"))]
    let alt_backend = detect_simd_backend();

    for grad in &grads {
        let s_r = st_ref.increment_step();
        let mut v_r = TensorViewMut {
            param: &mut p_ref,
            master_param: st_ref.master_param.as_deref_mut(),
            grad,
            exp_avg: &mut st_ref.exp_avg,
            exp_avg_sq: st_ref.exp_avg_sq.as_deref_mut(),
            scratch: &mut st_ref.scratch,
            dtype_mode: DTypeMode::BFloat16,
            state_dtype_mode: DTypeMode::Float32,
        };
        single_step_dual_scalar(&mut v_r, &config, s_r, 0).expect("ref step");

        let s_a = st_alt.increment_step();
        let mut v_a = TensorViewMut {
            param: &mut p_alt,
            master_param: st_alt.master_param.as_deref_mut(),
            grad,
            exp_avg: &mut st_alt.exp_avg,
            exp_avg_sq: st_alt.exp_avg_sq.as_deref_mut(),
            scratch: &mut st_alt.scratch,
            dtype_mode: DTypeMode::BFloat16,
            state_dtype_mode: DTypeMode::Float32,
        };
        dual_step_parallel(&mut v_a, &config, s_a, 0, alt_backend).expect("alt parallel step");

        let s_p = st_par.increment_step();
        let mut v_p = TensorViewMut {
            param: &mut p_par,
            master_param: st_par.master_param.as_deref_mut(),
            grad,
            exp_avg: &mut st_par.exp_avg,
            exp_avg_sq: st_par.exp_avg_sq.as_deref_mut(),
            scratch: &mut st_par.scratch,
            dtype_mode: DTypeMode::BFloat16,
            state_dtype_mode: DTypeMode::Float32,
        };
        dual_step_parallel(&mut v_p, &config, s_p, 0, detect_simd_backend())
            .expect("auto parallel step");
    }

    assert!(max_abs_diff(&p_ref, &p_alt) <= TOLERANCE);
    assert!(max_abs_diff(&p_ref, &p_par) <= TOLERANCE);
    assert!(max_abs_diff(&st_ref.exp_avg, &st_alt.exp_avg) <= TOLERANCE);
    assert!(max_abs_diff(&st_ref.exp_avg, &st_par.exp_avg) <= TOLERANCE);
    assert_scratch_parity(&st_ref.scratch, &st_alt.scratch, "Alt Parallel");
    assert_scratch_parity(&st_ref.scratch, &st_par.scratch, "Auto Parallel");
}

#[test]
fn test_scalar_parallel_chunked_exact_parity() {
    let n = 2048;
    let chunk_size = 128;

    for buffer in [BufferMode::TdClm, BufferMode::Dual] {
        let cfg = LuminaVConfig::builder()
            .lr(1.0e-3)
            .betas(0.9, 0.999)
            .buffer(buffer)
            .bound_type(BoundType::Radial)
            .cautious(true)
            .build()
            .expect("valid config");

        let init_p = generate_tensor(n, 123, 0.5, DTypeMode::Float32);
        let grad = generate_tensor(n, 456, 0.05, DTypeMode::Float32);

        let mut p_ref = init_p.clone();
        let mut st_ref = ParamState::new(&p_ref, buffer, false);
        let mut p_par = init_p.clone();
        let mut st_par = ParamState::new(&p_par, buffer, false);
        let mut p_chk = init_p;
        let mut st_chk = ParamState::new(&p_chk, buffer, false);

        let mut v_ref = TensorViewMut {
            param: &mut p_ref,
            master_param: None,
            grad: &grad,
            exp_avg: &mut st_ref.exp_avg,
            exp_avg_sq: st_ref.exp_avg_sq.as_deref_mut(),
            scratch: &mut st_ref.scratch,
            dtype_mode: DTypeMode::Float32,
            state_dtype_mode: DTypeMode::Float32,
        };
        let mut v_par = TensorViewMut {
            param: &mut p_par,
            master_param: None,
            grad: &grad,
            exp_avg: &mut st_par.exp_avg,
            exp_avg_sq: st_par.exp_avg_sq.as_deref_mut(),
            scratch: &mut st_par.scratch,
            dtype_mode: DTypeMode::Float32,
            state_dtype_mode: DTypeMode::Float32,
        };
        let mut v_chk = TensorViewMut {
            param: &mut p_chk,
            master_param: None,
            grad: &grad,
            exp_avg: &mut st_chk.exp_avg,
            exp_avg_sq: st_chk.exp_avg_sq.as_deref_mut(),
            scratch: &mut st_chk.scratch,
            dtype_mode: DTypeMode::Float32,
            state_dtype_mode: DTypeMode::Float32,
        };

        if buffer == BufferMode::TdClm {
            single_step_tdclm_scalar(&mut v_ref, &cfg, 1, 0).expect("scalar ref");
            tdclm_step_parallel(&mut v_par, &cfg, 1, 0, SimdBackend::Scalar).expect("par");
            tdclm_step_parallel_chunked(&mut v_chk, &cfg, 1, 0, SimdBackend::Scalar, chunk_size)
                .expect("chunked");
        } else {
            single_step_dual_scalar(&mut v_ref, &cfg, 1, 0).expect("scalar ref");
            dual_step_parallel(&mut v_par, &cfg, 1, 0, SimdBackend::Scalar).expect("par");
            dual_step_parallel_chunked(&mut v_chk, &cfg, 1, 0, SimdBackend::Scalar, chunk_size)
                .expect("chunked");
        }

        assert!(max_abs_diff(&p_ref, &p_par) <= 1e-6);
        assert!(max_abs_diff(&p_ref, &p_chk) <= 1e-6);
        assert!(max_abs_diff(&st_ref.exp_avg, &st_par.exp_avg) <= 1e-6);
        assert!(max_abs_diff(&st_ref.exp_avg, &st_chk.exp_avg) <= 1e-6);
        assert_scratch_parity(&st_ref.scratch, &st_par.scratch, "Scalar Parallel");
        assert!((st_ref.scratch.u_sq_sum() - st_chk.scratch.u_sq_sum()).abs() <= 1e-3);
    }
}

#[test]
fn test_differential_parity_t1_and_t2_simd_and_parallel() {
    let n = 32_768;
    let init_params = generate_tensor(n, 77, 0.5, DTypeMode::Float32);
    let grads: Vec<Vec<f32>> = (0..NUM_STEPS)
        .map(|s| generate_tensor(n, 2000 + s as i32, 0.05, DTypeMode::Float32))
        .collect();

    for bound_type in [BoundType::Coordinate, BoundType::Radial] {
        for buffer in [BufferMode::T1, BufferMode::T2] {
            let cfg = LuminaVConfig::builder()
                .lr(1.0e-3)
                .betas(0.9, 0.999)
                .weight_decay(0.01)
                .buffer(buffer)
                .bound_type(bound_type)
                .cautious(true)
                .master_weights(MasterWeightMode::None)
                .stochastic_rounding(true)
                .seed(1337)
                .build()
                .expect("valid triangle config");

            let mut p_ref = init_params.clone();
            let mut st_ref = ParamState::new(&p_ref, buffer, false);
            let mut p_simd = init_params.clone();
            let mut st_simd = ParamState::new(&p_simd, buffer, false);
            let mut p_par = init_params.clone();
            let mut st_par = ParamState::new(&p_par, buffer, false);

            for grad in &grads {
                let s1 = st_ref.increment_step();
                let mut v_ref = TensorViewMut {
                    param: &mut p_ref,
                    master_param: None,
                    grad,
                    exp_avg: &mut st_ref.exp_avg,
                    exp_avg_sq: st_ref.exp_avg_sq.as_deref_mut(),
                    scratch: &mut st_ref.scratch,
                    dtype_mode: DTypeMode::Float32,
                    state_dtype_mode: DTypeMode::Float32,
                };
                let s2 = st_simd.increment_step();
                let mut v_simd = TensorViewMut {
                    param: &mut p_simd,
                    master_param: None,
                    grad,
                    exp_avg: &mut st_simd.exp_avg,
                    exp_avg_sq: st_simd.exp_avg_sq.as_deref_mut(),
                    scratch: &mut st_simd.scratch,
                    dtype_mode: DTypeMode::Float32,
                    state_dtype_mode: DTypeMode::Float32,
                };
                let s3 = st_par.increment_step();
                let mut v_par = TensorViewMut {
                    param: &mut p_par,
                    master_param: None,
                    grad,
                    exp_avg: &mut st_par.exp_avg,
                    exp_avg_sq: st_par.exp_avg_sq.as_deref_mut(),
                    scratch: &mut st_par.scratch,
                    dtype_mode: DTypeMode::Float32,
                    state_dtype_mode: DTypeMode::Float32,
                };

                if buffer == BufferMode::T1 {
                    single_step_t1_scalar(&mut v_ref, &cfg, s1, 0).expect("scalar T1");
                    step_t1(&mut v_simd, &cfg, s2, 0).expect("simd T1");
                    tdclm_step_parallel(&mut v_par, &cfg, s3, 0, detect_simd_backend())
                        .expect("parallel T1");
                } else {
                    single_step_t2_scalar(&mut v_ref, &cfg, s1, 0).expect("scalar T2");
                    step_t2(&mut v_simd, &cfg, s2, 0).expect("simd T2");
                    dual_step_parallel(&mut v_par, &cfg, s3, 0, detect_simd_backend())
                        .expect("parallel T2");
                }
            }

            assert!(max_abs_diff(&p_ref, &p_simd) <= TOLERANCE);
            assert!(max_abs_diff(&p_ref, &p_par) <= TOLERANCE);
            assert!(max_abs_diff(&st_ref.exp_avg, &st_simd.exp_avg) <= TOLERANCE);
            assert_scratch_parity(&st_ref.scratch, &st_simd.scratch, "Triangle SIMD");
            assert_scratch_parity(&st_ref.scratch, &st_par.scratch, "Triangle Parallel");
        }
    }
}

#[test]
fn test_differential_parity_rtv_cwd_and_custom_bound_floor() {
    for buffer in [
        BufferMode::Dual,
        BufferMode::T2,
        BufferMode::TdClm,
        BufferMode::T1,
    ] {
        let config = LuminaVConfig::builder()
            .lr(2.0e-3)
            .betas(0.9, 0.999)
            .weight_decay(0.05)
            .cautious_weight_decay(true)
            .variance_type(VarianceType::Rtv)
            .variance_temper(3.0)
            .buffer(buffer)
            .bound_type(BoundType::Coordinate)
            .bound_ratio(0.05)
            .bound_floor(0.1)
            .cautious(true)
            .master_weights(MasterWeightMode::None)
            .stochastic_rounding(true)
            .seed(4242)
            .build()
            .expect("valid RTV+CWD+bound_floor config");
        let label = format!("{buffer:?} RTV+CWD+Floor");
        run_multi_step_parity(&label, &config, DTypeMode::Float32, 321, 7000);
    }
}
