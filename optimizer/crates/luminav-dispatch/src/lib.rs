#![deny(warnings)]

pub mod detector;
pub mod error;
pub mod router;

pub use detector::{HardwareCapabilities, detect_hardware};
pub use error::DispatchError;
#[cfg(feature = "cubecl")]
pub use luminav_cubecl;
pub use router::{
    GPU_DISPATCH_THRESHOLD, LuminaDispatcher, LuminaEngine, get_global_dispatcher, step_auto,
};

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "cubecl")]
    use luminav_core::single_step_t2_scalar;
    use luminav_core::{
        BufferMode, DTypeMode, ExecutionMode, LuminaVConfig, ScratchBuffer, TensorViewMut,
    };

    #[test]
    fn test_cubecl_dispatch_mode_contract() {
        let cfg = LuminaVConfig {
            buffer: BufferMode::T2,
            execution: ExecutionMode::CubeCl,
            ..Default::default()
        };

        let mut p = vec![0.5_f32; 256];
        let g = vec![0.1_f32; 256];
        let mut m = vec![0.0_f32; 256];
        let mut v = vec![0.0_f32; 256];
        let mut scratch = ScratchBuffer::default();

        let mut view = TensorViewMut {
            param: &mut p,
            master_param: None,
            grad: &g,
            exp_avg: &mut m,
            exp_avg_sq: Some(&mut v),
            scratch: &mut scratch,
            dtype_mode: DTypeMode::Float32,
            state_dtype_mode: DTypeMode::Float32,
        };

        let res = step_auto(&mut view, &cfg, 1, 0);
        #[cfg(not(feature = "cubecl"))]
        {
            assert!(matches!(
                res,
                Err(DispatchError::BackendUnavailable(ExecutionMode::CubeCl))
            ));
            assert!(!detect_hardware().cubecl_available);
        }
        #[cfg(feature = "cubecl")]
        {
            let caps = detect_hardware();
            if caps.cubecl_available {
                res.unwrap();
                let mut p_ref = vec![0.5_f32; 256];
                let mut m_ref = vec![0.0_f32; 256];
                let mut v_ref = vec![0.0_f32; 256];
                let mut s_ref = ScratchBuffer::default();
                let mut vr = TensorViewMut {
                    param: &mut p_ref,
                    master_param: None,
                    grad: &g,
                    exp_avg: &mut m_ref,
                    exp_avg_sq: Some(&mut v_ref),
                    scratch: &mut s_ref,
                    dtype_mode: DTypeMode::Float32,
                    state_dtype_mode: DTypeMode::Float32,
                };
                single_step_t2_scalar(&mut vr, &cfg, 1, 0).unwrap();
                for i in 0..256 {
                    assert!((p[i] - p_ref[i]).abs() < 1e-4, "mismatch at {i}");
                }
            }
        }
    }
}
