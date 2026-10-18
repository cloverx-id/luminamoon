use thiserror::Error;

#[derive(Error, Debug)]
pub enum DispatchError {
    #[error("Lumina Core error: {0}")]
    Core(#[from] luminav_core::LuminaError),

    #[error("Vulkan GPU error: {0}")]
    Vulkan(#[from] luminav_vulkan::VulkanError),

    #[error("Triton AOT GPU error: {0}")]
    TritonAot(#[from] luminav_triton_aot::TritonAotError),

    #[error("Requested backend {0:?} is not available")]
    BackendUnavailable(luminav_core::ExecutionMode),

    #[error("Concurrency lock error: {0}")]
    LockPoisoned(String),
}
