use std::sync::{Mutex, OnceLock};

use luminav_core::{BufferMode, ExecutionMode, LuminaVConfig, TensorViewMut};
#[cfg(feature = "cubecl")]
use luminav_cubecl::CubeClLuminaEngine;
use luminav_triton_aot::TritonAotEngine;

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
    #[cfg(feature = "cubecl")]
    pub cubecl_engine: Option<CubeClLuminaEngine>,
}

impl Default for LuminaDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl LuminaDispatcher {
    pub fn new() -> Self {
        let triton_engine = TritonAotEngine::new().ok();
        #[cfg(feature = "cubecl")]
        let cubecl_engine = CubeClLuminaEngine::new().ok();
        Self {
            triton_engine,
            #[cfg(feature = "cubecl")]
            cubecl_engine,
        }
    }

    #[cfg(feature = "cubecl")]
    pub fn with_cubecl(engine: CubeClLuminaEngine) -> Self {
        Self {
            triton_engine: None,
            cubecl_engine: Some(engine),
        }
    }

    #[inline]
    pub fn is_triton_ready(&self) -> bool {
        self.triton_engine.is_some()
    }

    #[inline]
    pub fn is_cubecl_ready(&self) -> bool {
        #[cfg(feature = "cubecl")]
        {
            self.cubecl_engine.is_some()
        }
        #[cfg(not(feature = "cubecl"))]
        {
            false
        }
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

                    #[cfg(feature = "cubecl")]
                    if let Some(cubecl_engine) = &mut self.cubecl_engine
                        && cubecl_engine.step(view, config, step, param_index).is_ok()
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
                        Err(e) if e.is_fallback_eligible() => {}
                        Err(e) => return Err(DispatchError::TritonAot(e)),
                    }
                }
                #[cfg(feature = "cubecl")]
                if n >= GPU_DISPATCH_THRESHOLD
                    && let Some(cubecl_engine) = &mut self.cubecl_engine
                    && cubecl_engine.step(view, config, step, param_index).is_ok()
                {
                    return Ok(());
                }
                luminav_simd::step(view, config, step, param_index)?;
                Ok(())
            }
            ExecutionMode::CubeCl => {
                #[cfg(feature = "cubecl")]
                {
                    let cubecl_engine = self
                        .cubecl_engine
                        .as_mut()
                        .ok_or(DispatchError::BackendUnavailable(config.execution))?;
                    cubecl_engine.step(view, config, step, param_index)?;
                    Ok(())
                }
                #[cfg(not(feature = "cubecl"))]
                {
                    Err(DispatchError::BackendUnavailable(config.execution))
                }
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
                    BufferMode::T1 => {
                        luminav_core::single_step_t1_scalar(view, config, step, param_index)?;
                    }
                    BufferMode::T2 => {
                        luminav_core::single_step_t2_scalar(view, config, step, param_index)?;
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
                BufferMode::T1 => {
                    luminav_core::single_step_t1_scalar(view, config, step, param_index)?;
                }
                BufferMode::T2 => {
                    luminav_core::single_step_t2_scalar(view, config, step, param_index)?;
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
