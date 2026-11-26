use luminav_core::LuminaError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CubeClError {
    #[error("LuminaV core validation failed: {0}")]
    Core(#[from] LuminaError),

    #[error("CubeCL runtime unavailable: {0}")]
    RuntimeUnavailable(String),

    #[error("Invalid buffer shape or configuration: {0}")]
    InvalidShape(String),

    #[error("CubeCL kernel launch failed: {0}")]
    LaunchFailed(String),
}
