use thiserror::Error;

#[derive(Error, Debug)]
pub enum VulkanError {
    #[error("Vulkan loading error: {0}")]
    Loading(#[from] ash::LoadingError),

    #[error("Vulkan API error: {0:?}")]
    Vk(#[from] ash::vk::Result),

    #[error("No suitable Vulkan physical device found")]
    NoPhysicalDevice,

    #[error("Buffer device address feature is not supported on this device")]
    BdaNotSupported,

    #[error("No compute queue family found")]
    NoComputeQueue,

    #[error("No suitable UMA host-visible memory type found (required flags: {0:?})")]
    NoSuitableMemoryType(ash::vk::MemoryPropertyFlags),

    #[error("Failed to map memory: {0:?}")]
    MapFailed(ash::vk::Result),

    #[error("Lumina core error: {0}")]
    Core(#[from] luminav_core::LuminaError),

    #[error("Invalid tensor shape or size: {0}")]
    InvalidShape(String),
}
