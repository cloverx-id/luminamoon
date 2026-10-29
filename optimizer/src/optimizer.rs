pub use luminav_core::{
    BoundType, BufferMode, DTypeMode, ExecutionMode, LuminaError, LuminaVConfig,
    LuminaVConfigBuilder, MasterWeightMode, ParamState, ScratchBuffer, TensorViewMut,
};
pub use luminav_dispatch::{
    DispatchError, GPU_DISPATCH_THRESHOLD, HardwareCapabilities, LuminaDispatcher, LuminaEngine,
    detect_hardware, get_global_dispatcher, step_auto,
};
pub use luminav_simd::{SimdBackend, detect_simd_backend};
pub use luminav_triton_aot::{AotManifest, KernelMetadata, TritonAotEngine, TritonDriverBackend};
pub use luminav_vulkan::{UmaBuffer, VulkanContext, VulkanLuminaEngine};

pub struct LuminaV {
    config: LuminaVConfig,
    states: Vec<Option<ParamState>>,
}

impl LuminaV {
    pub fn new(config: LuminaVConfig) -> Result<Self, LuminaError> {
        config.validate()?;
        Ok(Self {
            config,
            states: Vec::new(),
        })
    }

    #[inline]
    pub fn builder() -> LuminaVConfigBuilder {
        LuminaVConfig::builder()
    }

    #[inline]
    pub const fn config(&self) -> &LuminaVConfig {
        &self.config
    }

    #[inline]
    pub fn config_mut(&mut self) -> &mut LuminaVConfig {
        &mut self.config
    }

    #[inline]
    pub fn state(&self, param_index: usize) -> Option<&ParamState> {
        self.states.get(param_index).and_then(|s| s.as_ref())
    }

    pub fn step_param(
        &mut self,
        param_index: usize,
        param: &mut [f32],
        grad: &[f32],
        dtype_mode: DTypeMode,
    ) -> Result<(), DispatchError> {
        if param.len() != grad.len() {
            return Err(DispatchError::Core(LuminaError::ShapeMismatch {
                expected: param.len(),
                actual: grad.len(),
            }));
        }
        if param.is_empty() {
            return Ok(());
        }

        if self.states.len() <= param_index {
            self.states.resize_with(param_index + 1, || None);
        }

        let has_master =
            self.config.master_weights.is_enabled() && dtype_mode != DTypeMode::Float32;
        let state = self.states[param_index]
            .get_or_insert_with(|| ParamState::new(param, self.config.buffer, has_master));
        state.ensure_buffer_mode(self.config.buffer);
        if has_master && state.master_param.is_none() {
            state.master_param = Some(param.to_vec());
        }
        let step = state.increment_step();

        let state_dtype_mode = if self.config.master_weights == MasterWeightMode::Full
            || dtype_mode == DTypeMode::Float32
        {
            DTypeMode::Float32
        } else {
            DTypeMode::BFloat16
        };

        let mut effective_config = self.config.clone();
        if !has_master {
            effective_config.master_weights = MasterWeightMode::None;
        }

        let mut view = TensorViewMut {
            param,
            master_param: state.master_param.as_deref_mut(),
            grad,
            exp_avg: &mut state.exp_avg,
            exp_avg_sq: state.exp_avg_sq.as_deref_mut(),
            scratch: &mut state.scratch,
            dtype_mode,
            state_dtype_mode,
        };

        step_auto(&mut view, &effective_config, step, param_index)
    }

    pub fn step_batch(
        &mut self,
        params: &mut [&mut [f32]],
        grads: &[&[f32]],
        dtype_mode: DTypeMode,
    ) -> Result<(), DispatchError> {
        if params.len() != grads.len() {
            return Err(DispatchError::Core(LuminaError::ShapeMismatch {
                expected: params.len(),
                actual: grads.len(),
            }));
        }
        for (idx, (p, g)) in params.iter_mut().zip(grads.iter()).enumerate() {
            self.step_param(idx, p, g, dtype_mode)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_facade_luminav_step_batch() {
        let cfg = LuminaV::builder()
            .lr(1.0e-2)
            .buffer(BufferMode::Dual)
            .master_weights(MasterWeightMode::Full)
            .build()
            .expect("valid config");
        let mut opt = LuminaV::new(cfg).expect("optimizer init");

        let mut p0 = vec![1.0_f32; 64];
        let mut p1 = vec![-0.5_f32; 64];
        let g0 = vec![0.1_f32; 64];
        let g1 = vec![-0.2_f32; 64];

        opt.step_batch(&mut [&mut p0, &mut p1], &[&g0, &g1], DTypeMode::Float32)
            .expect("step 1");
        opt.step_batch(&mut [&mut p0, &mut p1], &[&g0, &g1], DTypeMode::BFloat16)
            .expect("step 2");

        assert!(p0.iter().all(|v| *v < 1.0_f32 && v.is_finite()));
        assert!(p1.iter().all(|v| *v > -0.5_f32 && v.is_finite()));
        let st0 = opt.state(0).expect("state 0");
        assert_eq!(st0.step, 2);
        let mp0 = st0.master_param.as_ref().expect("master_param 0");
        assert!(mp0.iter().all(|v| *v < 1.0_f32 && v.is_finite()));
    }
}
