#![deny(warnings)]

pub mod engine;
mod engine_nn;
mod engine_step;
pub mod error;
pub mod kernels;

pub use engine::{CubeClBackend, CubeClLuminaEngine};
pub use error::CubeClError;

#[cfg(test)]
mod tests {
    use super::*;
    use luminav_core::{
        BoundType, BufferMode, DTypeMode, LuminaVConfig, ScratchBuffer, TensorViewMut,
        single_step_dual_scalar, single_step_t1_scalar, single_step_t2_scalar,
        single_step_tdclm_scalar,
    };

    fn run_mode_parity(buffer: BufferMode, bound: BoundType, cautious: bool, dtype: DTypeMode) {
        let mut eng = match CubeClLuminaEngine::new() {
            Ok(e) => e,
            Err(_) => return,
        };
        let n = 1024_usize;
        let base_p: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.13).sin()) * 0.5).collect();
        let grad: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.37).cos()) * 0.25).collect();
        let base_m: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.07).sin()) * 0.1).collect();
        let base_v: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.11).cos().abs()) * 0.05 + 1e-4)
            .collect();

        let cfg = LuminaVConfig {
            buffer,
            bound_type: bound,
            cautious,
            lr: 1e-2,
            weight_decay: 0.01,
            stochastic_rounding: dtype != DTypeMode::Float32,
            ..Default::default()
        };

        let mut p_ref = base_p.clone();
        let mut m_ref = base_m.clone();
        let mut v_ref = base_v.clone();
        let mut s_ref = ScratchBuffer::default();

        let mut p_gpu = base_p;
        let mut m_gpu = base_m;
        let mut v_gpu = base_v;
        let mut s_gpu = ScratchBuffer::default();

        let is_dual = matches!(buffer, BufferMode::Dual | BufferMode::T2);
        for step in 1..=3_u64 {
            let mut view_ref = TensorViewMut {
                param: &mut p_ref,
                master_param: None,
                grad: &grad,
                exp_avg: &mut m_ref,
                exp_avg_sq: if is_dual { Some(&mut v_ref) } else { None },
                scratch: &mut s_ref,
                dtype_mode: dtype,
                state_dtype_mode: DTypeMode::Float32,
            };
            match buffer {
                BufferMode::TdClm => {
                    single_step_tdclm_scalar(&mut view_ref, &cfg, step, 0).unwrap()
                }
                BufferMode::T1 => single_step_t1_scalar(&mut view_ref, &cfg, step, 0).unwrap(),
                BufferMode::Dual => single_step_dual_scalar(&mut view_ref, &cfg, step, 0).unwrap(),
                BufferMode::T2 => single_step_t2_scalar(&mut view_ref, &cfg, step, 0).unwrap(),
            }

            let mut view_gpu = TensorViewMut {
                param: &mut p_gpu,
                master_param: None,
                grad: &grad,
                exp_avg: &mut m_gpu,
                exp_avg_sq: if is_dual { Some(&mut v_gpu) } else { None },
                scratch: &mut s_gpu,
                dtype_mode: dtype,
                state_dtype_mode: DTypeMode::Float32,
            };
            eng.step(&mut view_gpu, &cfg, step, 0).unwrap();
        }

        let tol = if dtype == DTypeMode::Float32 {
            1e-4_f32
        } else {
            5e-3_f32
        };
        for i in 0..n {
            assert!(
                (p_ref[i] - p_gpu[i]).abs() <= tol,
                "param mismatch at {i} ({buffer:?}, {bound:?}, cautious={cautious}, {dtype:?}): ref={} gpu={}",
                p_ref[i],
                p_gpu[i]
            );
            assert!(
                (m_ref[i] - m_gpu[i]).abs() <= tol,
                "m mismatch at {i} ({buffer:?}): ref={} gpu={}",
                m_ref[i],
                m_gpu[i]
            );
            if is_dual {
                assert!(
                    (v_ref[i] - v_gpu[i]).abs() <= tol,
                    "v mismatch at {i} ({buffer:?}): ref={} gpu={}",
                    v_ref[i],
                    v_gpu[i]
                );
            }
        }
    }

    #[test]
    fn test_cubecl_all_luminav_modes_parity() {
        for buffer in [
            BufferMode::TdClm,
            BufferMode::T1,
            BufferMode::Dual,
            BufferMode::T2,
        ] {
            for bound in [BoundType::None, BoundType::Radial, BoundType::Coordinate] {
                run_mode_parity(buffer, bound, true, DTypeMode::Float32);
                run_mode_parity(buffer, bound, false, DTypeMode::BFloat16);
            }
        }
        run_mode_parity(
            BufferMode::T1,
            BoundType::Coordinate,
            true,
            DTypeMode::Float16,
        );
        run_mode_parity(BufferMode::T2, BoundType::Radial, true, DTypeMode::Float16);
    }

    #[test]
    fn test_cubecl_master_weights_parity() {
        let mut eng = match CubeClLuminaEngine::new() {
            Ok(e) => e,
            Err(_) => return,
        };
        let n = 512_usize;
        let base_p: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.17).sin()) * 0.4).collect();
        let grad: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.29).cos()) * 0.2).collect();
        let base_m: Vec<f32> = vec![0.0; n];
        let base_v: Vec<f32> = vec![0.0; n];

        let cfg = LuminaVConfig {
            buffer: BufferMode::T2,
            master_weights: luminav_core::MasterWeightMode::Full,
            lr: 1e-2,
            weight_decay: 0.01,
            ..Default::default()
        };

        let (mut p_ref, mut mp_ref, mut m_ref, mut v_ref, mut s_ref) = (
            base_p.clone(),
            base_p.clone(),
            base_m.clone(),
            base_v.clone(),
            ScratchBuffer::default(),
        );
        let (mut p_gpu, mut mp_gpu, mut m_gpu, mut v_gpu, mut s_gpu) = (
            base_p.clone(),
            base_p,
            base_m,
            base_v,
            ScratchBuffer::default(),
        );

        for step in 1..=2_u64 {
            let mut vr = TensorViewMut {
                param: &mut p_ref,
                master_param: Some(&mut mp_ref),
                grad: &grad,
                exp_avg: &mut m_ref,
                exp_avg_sq: Some(&mut v_ref),
                scratch: &mut s_ref,
                dtype_mode: DTypeMode::BFloat16,
                state_dtype_mode: DTypeMode::Float32,
            };
            single_step_t2_scalar(&mut vr, &cfg, step, 3).unwrap();

            let mut vg = TensorViewMut {
                param: &mut p_gpu,
                master_param: Some(&mut mp_gpu),
                grad: &grad,
                exp_avg: &mut m_gpu,
                exp_avg_sq: Some(&mut v_gpu),
                scratch: &mut s_gpu,
                dtype_mode: DTypeMode::BFloat16,
                state_dtype_mode: DTypeMode::Float32,
            };
            eng.step(&mut vg, &cfg, step, 3).unwrap();
        }

        for i in 0..n {
            assert!(
                (mp_ref[i] - mp_gpu[i]).abs() < 1e-4,
                "master mismatch at {i}"
            );
            assert!((p_ref[i] - p_gpu[i]).abs() < 1e-3, "param mismatch at {i}");
        }
    }

    #[test]
    fn test_cubecl_nn_kernels_parity() {
        let eng = match CubeClLuminaEngine::new() {
            Ok(e) => e,
            Err(_) => return,
        };

        // 1. RaTanh
        let x: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) * 0.03).collect();
        let b = 1.5_f32;
        let y_tanh = eng.ra_tanh_f32(&x, b).unwrap();
        for (&xi, &yi) in x.iter().zip(y_tanh.iter()) {
            let bt = b * (b + xi.abs());
            let expected = (bt * xi) / (bt + xi * xi);
            assert!(
                (yi - expected).abs() < 1e-5,
                "ra_tanh: got {yi} exp {expected}"
            );
        }

        // 2. RaSwiGLU & RaGLU
        let up: Vec<f32> = (0..512).map(|i| (i as f32 * 0.17).cos()).collect();
        let y_swiglu = eng.ra_swiglu_f32(&x, &up, b).unwrap();
        let y_glu = eng.ra_glu_f32(&x, &up, b).unwrap();
        for (((&zi, &ui), &ys), &yg) in x
            .iter()
            .zip(up.iter())
            .zip(y_swiglu.iter())
            .zip(y_glu.iter())
        {
            let d = b * (b + zi.abs()) + zi * zi;
            let q = (0.5 * b * b) / d;
            let sigma = if zi >= 0.0 { 1.0 - q } else { q };
            let exp_swi = (zi * sigma) * ui;
            let exp_glu = ((b * (b + zi.abs()) * zi) / d) * ui;
            assert!(
                (ys - exp_swi).abs() < 1e-5,
                "ra_swiglu: got {ys} exp {exp_swi}"
            );
            assert!(
                (yg - exp_glu).abs() < 1e-5,
                "ra_glu: got {yg} exp {exp_glu}"
            );
        }

        // 3. RaNorm
        let rows = 8_usize;
        let cols = 128_usize;
        let x_norm: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.19).sin() * 2.0)
            .collect();
        let w: Vec<f32> = (0..cols).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let bias: Vec<f32> = (0..cols).map(|i| -0.2 + (i as f32) * 0.005).collect();
        let y_norm = eng
            .ra_norm_f32(&x_norm, Some(&w), Some(&bias), cols, 1e-6)
            .unwrap();
        for r in 0..rows {
            let row = &x_norm[r * cols..(r + 1) * cols];
            let l1 = row.iter().map(|v| v.abs()).sum::<f32>() / (cols as f32);
            let inv = 1.0 / (l1 + 1e-6);
            for c in 0..cols {
                let expected = row[c] * inv * w[c] + bias[c];
                let got = y_norm[r * cols + c];
                assert!(
                    (got - expected).abs() < 1e-4,
                    "ra_norm ({r},{c}): got {got} exp {expected}"
                );
            }
        }

        // 4. ra_softmax
        let s: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.23).cos() * 10.0)
            .collect();
        let p_gpu = eng.ra_softmax_f32(&s, cols, 0.0625).unwrap();
        for r in 0..rows {
            let row = &s[r * cols..(r + 1) * cols];
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let phis: Vec<f32> = row
                .iter()
                .map(|&v| {
                    let u = (m - v) * 0.0625;
                    if u < 1.0 {
                        ((1.0 - u) / (1.0 + u)).powi(8)
                    } else {
                        0.0
                    }
                })
                .collect();
            let sum_phi = phis.iter().sum::<f32>().max(1e-38);
            for c in 0..cols {
                let expected = phis[c] / sum_phi;
                let got = p_gpu[r * cols + c];
                assert!(
                    (got - expected).abs() < 1e-5,
                    "ra_softmax ({r},{c}): got {got} exp {expected}"
                );
            }
        }
    }
}
