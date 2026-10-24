use luminav_core::LuminaError;
use luminav_core::config::{
    BoundType, BufferMode, DTypeMode, ExecutionMode, LuminaVConfig, MasterWeightMode,
};
use luminav_core::math::{
    compute_bias_correction, sanitize_gradient, tanh_pade_f32, update_post_step_ema,
};
use luminav_core::prng::{
    _prng, derive_kernel_seeds, f16_bits_to_f32, f32_to_f16_bits_rne, stochastic_round_bf16,
    stochastic_round_fp16,
};
use luminav_core::reference_cpu::{single_step_dual_scalar, single_step_tdclm_scalar};
use luminav_core::state::{ParamState, ScratchBuffer, TensorViewMut};

#[test]
fn test_prng_determinism_and_signed_shift_invariants() {
    let s0 = _prng(1337, 0);
    let s0_repeat = _prng(1337, 0);
    assert_eq!(s0, s0_repeat);

    let s1 = _prng(1337, 1);
    let s2 = _prng(1338, 0);
    assert_ne!(s0, s1);
    assert_ne!(s0, s2);

    let mut expected = -1_640_531_527_i32;
    expected = (expected ^ (expected >> 16)).wrapping_mul(-2_048_173_461_i32);
    expected = (expected ^ (expected >> 13)).wrapping_mul(-1_028_477_387_i32);
    expected ^= expected >> 16;
    assert_eq!(_prng(0, 0), expected);

    let (seed_m, seed_p) = derive_kernel_seeds(1337, 1, 0);
    assert_eq!(seed_m, ((1337_u64 + 0x9E37_79B9_u64) & 0x7FFF_FFFF) as i32);
    assert_eq!(
        seed_p,
        (((seed_m as u64) + 0x5555_5555_u64) & 0x7FFF_FFFF) as i32
    );
}

#[test]
fn test_stochastic_rounding_bf16_and_subnormal_fp16() {
    let val = 1.000_123_f32;
    for idx in 0..64 {
        let sr_bf16 = stochastic_round_bf16(val, 1337, idx);
        assert!(sr_bf16.is_finite());
        assert_eq!(sr_bf16.to_bits() & 0xFFFF, 0);
        assert!((sr_bf16 - val).abs() < 1.0e-2);

        let sr_fp16 = stochastic_round_fp16(val, 1337, idx);
        assert!(sr_fp16.is_finite());
        assert!((sr_fp16 - val).abs() < 2.0e-3);
    }

    let tiny = f32::from_bits(102 << 23);
    assert_eq!(stochastic_round_fp16(tiny, 1337, 0), 0.0_f32);
    assert_eq!(stochastic_round_fp16(-tiny, 1337, 0), -0.0_f32);

    let half_bits = f32_to_f16_bits_rne(0.5_f32);
    assert_eq!(f16_bits_to_f32(half_bits), 0.5_f32);
}

#[test]
fn test_tanh_pade_f32_precision_against_std_tanh() {
    let mut max_abs_err = 0.0_f32;
    for i in -10_000..=10_000 {
        let x = (i as f32) * 0.001_f32;
        let approx = tanh_pade_f32(x);
        let exact = x.tanh();
        let err = (approx - exact).abs();
        if err > max_abs_err {
            max_abs_err = err;
        }
    }
    assert!(
        max_abs_err <= 3.0e-7_f32,
        "max_abs_err {max_abs_err} exceeded threshold"
    );
    assert_eq!(tanh_pade_f32(0.0_f32), 0.0_f32);
    assert_eq!(tanh_pade_f32(25.0_f32), 1.0_f32);
    assert_eq!(tanh_pade_f32(-25.0_f32), -1.0_f32);
    assert!(tanh_pade_f32(f32::NAN).is_nan());
}

#[test]
fn test_scratch_buffer_ema_smoothing_and_reset() {
    let mut scratch = ScratchBuffer::new();
    assert_eq!(scratch.cautious_scale(), 1.0_f32);
    assert_eq!(scratch.bound_scale(), 1.0_f32);

    scratch.set_mask_sum(50.0_f32);
    scratch.set_u_sq_sum(25.0_f32);
    scratch.set_p_sq_sum(100.0_f32);
    scratch.set_diff_sum(12.5_f32);

    let cfg = LuminaVConfig::builder()
        .lr(1.0e-3_f32)
        .cautious_clamp_min(0.5_f32)
        .ema_decay(0.8_f32)
        .bound_type(BoundType::Radial)
        .bound_ratio(0.05_f32)
        .cautious(true)
        .build()
        .expect("valid test config");

    update_post_step_ema(&mut scratch, 100.0_f32, &cfg);

    assert!((scratch.cautious_scale() - 1.2_f32).abs() < 1.0e-6_f32);
    assert!(scratch.bound_scale() > 0.0_f32 && scratch.bound_scale() <= 1.0_f32);
    assert_eq!(scratch.mask_sum(), 0.0_f32);
    assert_eq!(scratch.u_sq_sum(), 0.0_f32);
    assert_eq!(scratch.p_sq_sum(), 0.0_f32);
    assert_eq!(scratch.diff_sum(), 0.0_f32);
}

#[test]
fn test_config_validation_and_typo_suggestions() {
    assert!(LuminaVConfig::builder().build().is_ok());
    assert_eq!(
        LuminaVConfig::builder().lr(-1.0e-4).build(),
        Err(LuminaError::InvalidLearningRate(-1.0e-4))
    );
    assert_eq!(
        LuminaVConfig::builder().betas(1.0, 0.999).build(),
        Err(LuminaError::InvalidBeta1(1.0))
    );
    assert_eq!(
        BufferMode::try_from_count(3).unwrap_err().to_string(),
        "Invalid buffer count: 3. LuminaV only supports buffer=1 (Single-Buffer TD-CLM) or buffer=2 (Dual-Buffer CLM)."
    );
    assert_eq!(
        BufferMode::try_from_str("duel").unwrap_err().to_string(),
        "Invalid buffer count: 'duel'. LuminaV only supports buffer=1 (Single-Buffer TD-CLM) or buffer=2 (Dual-Buffer CLM). Did you mean 'dual'?"
    );
    assert_eq!(
        BoundType::try_from_str("radal").unwrap_err().to_string(),
        "Invalid bound_type: 'radal'. Expected 'coordinate' or 'radial'. Did you mean 'radial'?"
    );
    assert_eq!(
        ExecutionMode::try_from_str("vulkn")
            .unwrap_err()
            .to_string(),
        "Invalid execution mode: 'vulkn'. Expected 'auto', 'vulkan', 'simd', or 'scalar'. Did you mean 'vulkan'?"
    );
    assert_eq!(
        MasterWeightMode::try_from_str("sem")
            .unwrap_err()
            .to_string(),
        "Invalid master_weights: 'sem'. Expected False, 'none', 'semi', or 'full'. Did you mean 'semi'?"
    );

    let tdclm_cfg = LuminaVConfig::builder()
        .buffer(BufferMode::TdClm)
        .build()
        .expect("valid TD-CLM config");
    assert!((tdclm_cfg.ema_decay - 0.95_f32).abs() < 1.0e-6);

    let coeffs = compute_bias_correction(1, 0.9, 0.999, 1.0e-8, 0.8, DTypeMode::Float32);
    assert!((coeffs.bc1_nes - 0.19_f32).abs() < 1.0e-6);
    assert!(coeffs.c1 > 0.0 && coeffs.c2 > 0.0);
}

#[test]
fn test_single_step_tdclm_and_dual_scalar_golden_oracle() {
    let n = 32;
    let initial_params: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let mut grads: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { 0.05_f32 } else { -0.03_f32 })
        .collect();
    grads[0] = f32::NAN;
    assert_eq!(sanitize_gradient(grads[0]), 0.0_f32);

    let tdclm_cfg = LuminaVConfig::builder()
        .buffer(BufferMode::TdClm)
        .bound_type(BoundType::Radial)
        .centered_defect(true)
        .master_weights(MasterWeightMode::None)
        .build()
        .expect("valid config");

    let mut p_tdclm = initial_params.clone();
    let mut state_tdclm = ParamState::new(&p_tdclm, BufferMode::TdClm, false);

    for _ in 0..2 {
        let step = state_tdclm.increment_step();
        let mut view = TensorViewMut {
            param: &mut p_tdclm,
            master_param: None,
            grad: &grads,
            exp_avg: &mut state_tdclm.exp_avg,
            exp_avg_sq: None,
            scratch: &mut state_tdclm.scratch,
            dtype_mode: DTypeMode::Float32,
            state_dtype_mode: DTypeMode::Float32,
        };
        single_step_tdclm_scalar(&mut view, &tdclm_cfg, step, 0)
            .expect("TD-CLM scalar step should succeed");
    }

    assert!(p_tdclm.iter().all(|v| v.is_finite()));
    assert!(state_tdclm.exp_avg.iter().all(|v| v.is_finite()));
    assert_ne!(p_tdclm, initial_params);

    let dual_cfg = LuminaVConfig::builder()
        .buffer(BufferMode::Dual)
        .bound_type(BoundType::Coordinate)
        .centered_defect(true)
        .master_weights(MasterWeightMode::Full)
        .build()
        .expect("valid config");

    let mut p_dual = initial_params.clone();
    let mut state_dual = ParamState::new(&p_dual, BufferMode::Dual, true);

    for _ in 0..2 {
        let step = state_dual.increment_step();
        let mut view = TensorViewMut {
            param: &mut p_dual,
            master_param: state_dual.master_param.as_deref_mut(),
            grad: &grads,
            exp_avg: &mut state_dual.exp_avg,
            exp_avg_sq: state_dual.exp_avg_sq.as_deref_mut(),
            scratch: &mut state_dual.scratch,
            dtype_mode: DTypeMode::BFloat16,
            state_dtype_mode: DTypeMode::Float32,
        };
        single_step_dual_scalar(&mut view, &dual_cfg, step, 0)
            .expect("Dual scalar step should succeed");
    }

    assert!(p_dual.iter().all(|v| v.is_finite()));
    assert!(state_dual.exp_avg.iter().all(|v| v.is_finite()));
    assert!(
        state_dual
            .exp_avg_sq
            .as_ref()
            .expect("v buffer")
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0)
    );
    assert_ne!(p_dual, initial_params);
}
