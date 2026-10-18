use std::sync::{Mutex, OnceLock};

use luminav_core::{BufferMode, ExecutionMode, LuminaVConfig, TensorViewMut};
use luminav_triton_aot::TritonAotEngine;
use luminav_vulkan::VulkanLuminaEngine;

use crate::error::DispatchError;

pub const GPU_DISPATCH_THRESHOLD: usize = 16_384;

pub trait LuminaEngine {
    fn step(
        &mut self,
        view: &mut TensorViewMut<'_>,
        config: &LuminaVConfig,
        step: u64,
        param_index: usize,
    ) -> Result<(), DispatchError>;
}

pub struct LuminaDispatcher {
    pub triton_engine: Option<TritonAotEngine>,
    pub vulkan_engine: Option<VulkanLuminaEngine>,
}

impl Default for LuminaDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl LuminaDispatcher {
    pub fn new() -> Self {
        let triton_engine = TritonAotEngine::new().ok();
        let vulkan_engine = VulkanLuminaEngine::new().ok();
        Self {
            triton_engine,
            vulkan_engine,
        }
    }

    pub fn with_vulkan(engine: VulkanLuminaEngine) -> Self {
        Self {
            triton_engine: None,
            vulkan_engine: Some(engine),
        }
    }

    #[inline]
    pub fn is_triton_ready(&self) -> bool {
        self.triton_engine.is_some()
    }

    #[inline]
    pub fn is_vulkan_ready(&self) -> bool {
        self.vulkan_engine.is_some()
    }
}

impl LuminaEngine for LuminaDispatcher {
    fn step(
        &mut self,
        view: &mut TensorViewMut<'_>,
        config: &LuminaVConfig,
        step: u64,
        param_index: usize,
    ) -> Result<(), DispatchError> {
        let n = view.param.len();
        if n == 0 {
            return Ok(());
        }

        match config.execution {
            ExecutionMode::Auto => {
                if n >= GPU_DISPATCH_THRESHOLD {
                    if let Some(triton_engine) = &mut self.triton_engine
                        && triton_engine.step(view, config, step, param_index).is_ok()
                    {
                        return Ok(());
                    }

                    if let Some(vk_engine) = &mut self.vulkan_engine
                        && vk_engine.step(view, config, step, param_index).is_ok()
                    {
                        return Ok(());
                    }
                }

                luminav_simd::step(view, config, step, param_index)?;
                Ok(())
            }
            ExecutionMode::Triton => {
                if let Some(triton_engine) = &mut self.triton_engine {
                    match triton_engine.step(view, config, step, param_index) {
                        Ok(()) => return Ok(()),
                        Err(luminav_triton_aot::TritonAotError::StubBinaryNotExecutable {
                            ..
                        }) => {}
                        Err(e) => return Err(DispatchError::TritonAot(e)),
                    }
                }
                if n >= GPU_DISPATCH_THRESHOLD
                    && let Some(vk_engine) = &mut self.vulkan_engine
                {
                    vk_engine.step(view, config, step, param_index)?;
                } else {
                    luminav_simd::step(view, config, step, param_index)?;
                }
                Ok(())
            }
            ExecutionMode::Vulkan => {
                let vk_engine = self
                    .vulkan_engine
                    .as_mut()
                    .ok_or(DispatchError::BackendUnavailable(ExecutionMode::Vulkan))?;
                vk_engine.step(view, config, step, param_index)?;
                Ok(())
            }
            ExecutionMode::Simd => {
                luminav_simd::step(view, config, step, param_index)?;
                Ok(())
            }
            ExecutionMode::ScalarCore | ExecutionMode::Single => {
                match config.buffer {
                    BufferMode::TdClm => {
                        luminav_core::single_step_tdclm_scalar(view, config, step, param_index)?;
                    }
                    BufferMode::Dual => {
                        luminav_core::single_step_dual_scalar(view, config, step, param_index)?;
                    }
                }
                Ok(())
            }
        }
    }
}

static GLOBAL_DISPATCHER: OnceLock<Mutex<LuminaDispatcher>> = OnceLock::new();

pub fn get_global_dispatcher() -> &'static Mutex<LuminaDispatcher> {
    GLOBAL_DISPATCHER.get_or_init(|| Mutex::new(LuminaDispatcher::new()))
}

pub fn step_auto(
    view: &mut TensorViewMut<'_>,
    config: &LuminaVConfig,
    step: u64,
    param_index: usize,
) -> Result<(), DispatchError> {
    let n = view.param.len();
    if n == 0 {
        return Ok(());
    }

    match config.execution {
        ExecutionMode::Simd => {
            luminav_simd::step(view, config, step, param_index)?;
            return Ok(());
        }
        ExecutionMode::ScalarCore | ExecutionMode::Single => {
            match config.buffer {
                BufferMode::TdClm => {
                    luminav_core::single_step_tdclm_scalar(view, config, step, param_index)?;
                }
                BufferMode::Dual => {
                    luminav_core::single_step_dual_scalar(view, config, step, param_index)?;
                }
            }
            return Ok(());
        }
        ExecutionMode::Auto if n < GPU_DISPATCH_THRESHOLD => {
            luminav_simd::step(view, config, step, param_index)?;
            return Ok(());
        }
        _ => {}
    }

    let mut dispatcher = get_global_dispatcher()
        .lock()
        .map_err(|_| DispatchError::LockPoisoned("Global dispatcher mutex poisoned".to_string()))?;
    dispatcher.step(view, config, step, param_index)
}
