use luminav_core::LuminaError;
use luminav_core::config::{
    BoundType, BufferMode, DTypeMode, ExecutionMode, LuminaVConfig, MasterWeightMode, VarianceType,
};
use luminav_core::math::{
    apply_decoupled_weight_decay, apply_rational_bound, apply_step_bounds, compute_bias_correction,
    compute_inv_sqrt_bc2_prev, sanitize_gradient, tanh_pade_f32, temper_belief_defect,
    update_post_step_ema,
};
use luminav_core::prng::{
    _prng, derive_kernel_seeds, derive_spatial_seeds, derive_variance_seed,
    derive_weyl_step_offsets, f16_bits_to_f32, f32_to_f16_bits_rne, stochastic_round_bf16,
    stochastic_round_bf16_weyl, stochastic_round_fp16, weyl_top24,
};
use luminav_core::reference_cpu::{single_step_dual_scalar, single_step_tdclm_scalar};
use luminav_core::state::{ParamState, ScratchBuffer, TensorViewMut};

#[test]
fn test_prng_determinism_and_logical_shift_invariants() {
    let s0 = _prng(1337, 0);
    let s0_repeat = _prng(1337, 0);
    assert_eq!(s0, s0_repeat);

    let s1 = _prng(1337, 1);
    let s2 = _prng(1338, 0);
    assert_ne!(s0, s1);
    assert_ne!(s0, s2);

    let mut expected = 0_u32.wrapping_sub(1_640_531_527_u32);
    expected = (expected ^ (expected >> 16)).wrapping_mul(0x85EB_5A6B_u32);
    expected = (expected ^ (expected >> 13)).wrapping_mul(0xC2B2_AE35_u32);
    expected ^= expected >> 16;
    assert_eq!(_prng(0, 0), expected as i32);

    // Logical right shift ensures full 32-bit uniform coverage (bit 31 can be 1)
    let neg_count = (0..1024).filter(|&i| _prng(1337, i) < 0).count();
    assert!(
        neg_count > 400 && neg_count < 624,
        "expected ~50% negative i32 outputs from full 32-bit logical shift, got {neg_count}/1024"
    );

    let (seed_m, seed_p) = derive_kernel_seeds(1337, 1, 0);
    assert_eq!(seed_m, ((1337_u64 + 0x9E37_79B9_u64) & 0x7FFF_FFFF) as i32);
    assert_eq!(
        seed_p,
        (((seed_m as u64) + 0x5555_5555_u64) & 0x7FFF_FFFF) as i32
    );
}

#[test]
fn test_stochastic_rounding_preserves_nan() {
    assert!(stochastic_round_bf16(f32::NAN, 1337, 0).is_nan());
    assert!(stochastic_round_fp16(f32::NAN, 1337, 0).is_nan());
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
        "Invalid buffer: 3. Expected 1 (TD-CLM), 2 (Dual CLM), 't1' (Triangle 1-Buffer), or 't2' (Triangle 2-Buffer)."
    );
    assert_eq!(
        BufferMode::try_from_str("dual").unwrap_err().to_string(),
        "Legacy buffer alias 'dual' was removed in LuminaV v2.0.0. Use buffer=2 (Dual-Buffer CLM) instead."
    );
    assert_eq!(
        BufferMode::try_from_str("t3").unwrap_err().to_string(),
        "Invalid buffer: 't3'. Expected 1 (TD-CLM), 2 (Dual CLM), 't1' (Triangle 1-Buffer), or 't2' (Triangle 2-Buffer). Did you mean 't1'?"
    );
    assert_eq!(
        BoundType::try_from_str("radal").unwrap_err().to_string(),
        "Invalid bound_type: 'radal'. Expected 'coordinate' or 'radial'. Did you mean 'radial'?"
    );
    assert_eq!(
        ExecutionMode::try_from_str("vulkn")
            .unwrap_err()
            .to_string(),
        "Invalid execution mode: 'vulkn'. Expected 'auto', 'triton', 'cubecl', 'vulkan', 'simd', 'eager', or 'scalar'. Did you mean 'vulkan'?"
    );
    assert_eq!(
        ExecutionMode::try_from_str("foreach")
            .unwrap_err()
            .to_string(),
        "Legacy execution mode 'foreach' was removed in LuminaV v2.0.0. Use 'auto' instead."
    );
    assert_eq!(
        MasterWeightMode::try_from_str("sem")
            .unwrap_err()
            .to_string(),
        "Invalid master_weights: 'sem'. Expected 'none', 'semi', or 'full'. Did you mean 'semi'?"
    );
    assert_eq!(
        MasterWeightMode::try_from_str("fp32")
            .unwrap_err()
            .to_string(),
        "Legacy master_weights alias 'fp32' was removed in LuminaV v2.0.0. Use 'full' instead."
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

#[test]
fn test_bf16_exp_avg_sq_stochastic_decay_post_spike() {
    let (seed_m, seed_p) = derive_kernel_seeds(1337, 1, 0);
    let seed_v = derive_variance_seed(seed_m);
    assert_ne!(seed_v, seed_m);
    assert_ne!(seed_v, seed_p);

    let n = 256;
    let spike_grads = vec![100.0_f32; n];
    let zero_grads = vec![0.0_f32; n];

    for buf_mode in [BufferMode::Dual, BufferMode::T2] {
        for mw_mode in [MasterWeightMode::None, MasterWeightMode::Semi] {
            let has_master = mw_mode.is_enabled();

            let cfg_rne = LuminaVConfig::builder()
                .buffer(buf_mode)
                .master_weights(mw_mode)
                .stochastic_rounding(false)
                .build()
                .expect("valid RNE config");

            let cfg_sr = LuminaVConfig::builder()
                .buffer(buf_mode)
                .master_weights(mw_mode)
                .stochastic_rounding(true)
                .build()
                .expect("valid SR config");

            let mut p_rne = vec![1.0_f32; n];
            let mut st_rne = ParamState::new(&p_rne, buf_mode, has_master);
            let mut p_sr = vec![1.0_f32; n];
            let mut st_sr = ParamState::new(&p_sr, buf_mode, has_master);

            for step_idx in 0..600 {
                let g = if step_idx == 0 {
                    &spike_grads
                } else {
                    &zero_grads
                };

                let s_rne = st_rne.increment_step();
                let mut view_rne = TensorViewMut {
                    param: &mut p_rne,
                    master_param: st_rne.master_param.as_deref_mut(),
                    grad: g,
                    exp_avg: &mut st_rne.exp_avg,
                    exp_avg_sq: st_rne.exp_avg_sq.as_deref_mut(),
                    scratch: &mut st_rne.scratch,
                    dtype_mode: DTypeMode::BFloat16,
                    state_dtype_mode: DTypeMode::BFloat16,
                };
                single_step_dual_scalar(&mut view_rne, &cfg_rne, s_rne, 0)
                    .expect("RNE step should succeed");

                let s_sr = st_sr.increment_step();
                let mut view_sr = TensorViewMut {
                    param: &mut p_sr,
                    master_param: st_sr.master_param.as_deref_mut(),
                    grad: g,
                    exp_avg: &mut st_sr.exp_avg,
                    exp_avg_sq: st_sr.exp_avg_sq.as_deref_mut(),
                    scratch: &mut st_sr.scratch,
                    dtype_mode: DTypeMode::BFloat16,
                    state_dtype_mode: DTypeMode::BFloat16,
                };
                single_step_dual_scalar(&mut view_sr, &cfg_sr, s_sr, 0)
                    .expect("SR step should succeed");
            }

            let v_rne = st_rne.exp_avg_sq.as_ref().expect("RNE v");
            let v_sr = st_sr.exp_avg_sq.as_ref().expect("SR v");
            assert!(v_sr.iter().all(|v| v.is_finite() && *v >= 0.0));

            let mean_rne: f32 = v_rne.iter().sum::<f32>() / (n as f32);
            let mean_sr: f32 = v_sr.iter().sum::<f32>() / (n as f32);

            assert!(
                mean_rne > 8.0,
                "Expected RNE BF16 variance to freeze above 8.0, got {mean_rne}"
            );
            assert!(
                mean_sr < 6.5,
                "Expected SR BF16 variance to decay smoothly below 6.5, got {mean_sr}"
            );
        }
    }
}

fn pearson_corr(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut num = 0.0_f64;
    let mut dx2 = 0.0_f64;
    let mut dy2 = 0.0_f64;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let dx = x - mx;
        let dy = y - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    num / (dx2 * dy2).sqrt()
}

#[test]
fn test_weyl_kronecker_3d_sub_ulp_precision_and_zero_correlation() {
    let (seed_m, seed_p) = derive_spatial_seeds(1337, 0);
    let seed_v = derive_variance_seed(seed_m);

    let steps = 4096_u64;
    let mut us_m = Vec::with_capacity(steps as usize);
    let mut us_v = Vec::with_capacity(steps as usize);
    let mut us_p = Vec::with_capacity(steps as usize);
    for t in 1..=steps {
        let (wm, wv, wp) = derive_weyl_step_offsets(t);
        us_m.push(weyl_top24(seed_m, 0, wm) as f64 / 16_777_216.0);
        us_v.push(weyl_top24(seed_v, 0, wv) as f64 / 16_777_216.0);
        us_p.push(weyl_top24(seed_p, 0, wp) as f64 / 16_777_216.0);
    }

    let corr_mp = pearson_corr(&us_m, &us_p).abs();
    let corr_mv = pearson_corr(&us_m, &us_v).abs();
    let corr_vp = pearson_corr(&us_v, &us_p).abs();
    assert!(corr_mp < 0.05, "m-p correlation too high: {corr_mp}");
    assert!(corr_mv < 0.05, "m-v correlation too high: {corr_mv}");
    assert!(corr_vp < 0.05, "v-p correlation too high: {corr_vp}");

    let mut acc_weyl = 1.0_f32;
    let delta = 1.0e-5_f32;
    let n_steps = 10_000_u64;
    for t in 1..=n_steps {
        let (_, _, wp) = derive_weyl_step_offsets(t);
        acc_weyl = stochastic_round_bf16_weyl(acc_weyl + delta, seed_p, 0, wp);
    }
    let exact = 1.0_f32 + delta * (n_steps as f32);
    let err = (acc_weyl - exact).abs();
    assert!(
        err < 0.015_f32,
        "Weyl BF16 sub-ULP drift too high: got {acc_weyl}, exact {exact}, err {err}"
    );
}

#[test]
fn test_rational_bound_normalized_extreme_ranges_and_fp16_single_bound() {
    let cases: &[(f32, f32)] = &[
        (0.0, 0.0),
        (0.0, 1.0e-3),
        (1.0e-3, 0.0),
        (-1.0e-25, 1.0e-25),
        (1.0e25, 1.0e25),
        (-1.0e25, 2.0e24),
        (f32::MAX, f32::MAX),
        (3.0e-4, 5.0e-4),
        (-5.0e-2, 1.0e-3),
    ];
    for &(s, b) in cases {
        let got = apply_rational_bound(s, b);
        assert!(got.is_finite(), "non-finite for s={s}, b={b}: got {got}");
        let sd = s as f64;
        let bd = (b.max(0.0)) as f64;
        let expected = if bd == 0.0 || sd == 0.0 {
            0.0_f64
        } else {
            let abs_s = sd.abs();
            let m = bd.min(abs_s);
            let big_m = bd.max(abs_s);
            let t = m / big_m;
            (m * (1.0 + t) / (1.0 + t + t * t)).copysign(sd)
        };
        let tol = expected.abs() * 1.0e-6 + 1.0e-38;
        assert!(
            ((got as f64) - expected).abs() <= tol,
            "s={s}, b={b}: got={got}, expected={expected}"
        );
    }

    // Verify FP16 + Coordinate uses single effective radius min(b_coord, b_safe) without double damping.
    let p_init = 1.0_f32;
    let step_val = 0.02_f32;
    let b_safe = (p_init.abs() * 0.01_f32).max(1.5e-4_f32);
    for is_triangle in [false, true] {
        let got = apply_step_bounds(
            step_val,
            p_init,
            BoundType::Coordinate,
            0.05_f32,
            1.0e-2_f32,
            1.0_f32,
            DTypeMode::Float16,
            is_triangle,
        );
        let expected = if is_triangle {
            apply_rational_bound(step_val, b_safe)
        } else {
            b_safe * tanh_pade_f32(step_val / b_safe)
        };
        assert!(
            (got - expected).abs() < 1.0e-7,
            "is_triangle={is_triangle}: got={got}, expected={expected}"
        );
    }
}

#[test]
fn test_v2_knobs_rtv_cwd_bound_floor_and_fast_brake_radial_ema() {
    assert_eq!(VarianceType::try_from_str("standard").unwrap(), VarianceType::Standard);
    assert_eq!(VarianceType::try_from_str("RTV").unwrap(), VarianceType::Rtv);
    assert!(VarianceType::try_from_str("unknown").is_err());

    assert!(LuminaVConfig::builder().bound_floor(0.0).build().is_err());
    assert!(LuminaVConfig::builder().bound_floor(-0.01).build().is_err());
    assert!(LuminaVConfig::builder().variance_temper(0.0).build().is_err());
    assert!(LuminaVConfig::builder().variance_temper(f32::NAN).build().is_err());

    // Asymmetric fast-brake radial EMA: drops fast with min(ema_decay, 0.10), releases slowly with ema_decay.
    let cfg = LuminaVConfig::builder()
        .buffer(BufferMode::T2)
        .bound_type(BoundType::Radial)
        .cautious(false)
        .ema_decay(0.80)
        .lr(1.0)
        .bound_ratio(0.05)
        .build()
        .expect("valid radial config");
    let mut sc = ScratchBuffer::default();
    sc.set_bound_scale(1.0);
    sc.set_u_sq_sum(100.0);
    sc.set_p_sq_sum(1.0);
    update_post_step_ema(&mut sc, 16.0, &cfg);
    assert!(sc.bound_scale() < 0.15, "fast-brake should clamp rapidly, got {}", sc.bound_scale());

    let prev = sc.bound_scale();
    sc.set_u_sq_sum(0.0);
    sc.set_p_sq_sum(1.0);
    update_post_step_ema(&mut sc, 16.0, &cfg);
    let expected_release = 0.80 * prev + 0.20 * 1.0;
    assert!((sc.bound_scale() - expected_release).abs() < 1.0e-6);

    // RTV tempering bounds spike defect relative to max(sigma_prev, |m_prev|).
    let inv_bc2 = compute_inv_sqrt_bc2_prev(10, 0.999);
    let tempered = temper_belief_defect(50.0, 0.01, 0.5, inv_bc2, 3.0, 1.0e-8);
    assert!(tempered > 0.0 && tempered < 5.0, "RTV should bound spike defect, got {tempered}");

    // Cautious weight decay decays only when step_val * p_init >= 0.0.
    let p_aligned = apply_decoupled_weight_decay(2.0, 0.1, 0.05, true);
    let p_opposed = apply_decoupled_weight_decay(2.0, -0.1, 0.05, true);
    let p_zero = apply_decoupled_weight_decay(2.0, 0.0, 0.05, true);
    assert!((p_aligned - 1.9).abs() < 1.0e-6);
    assert!((p_opposed - 2.0).abs() < 1.0e-6);
    assert!((p_zero - 1.9).abs() < 1.0e-6);
}
