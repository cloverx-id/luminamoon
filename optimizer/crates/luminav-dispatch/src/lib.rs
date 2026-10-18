#![deny(warnings)]

pub mod detector;
pub mod error;
pub mod router;

pub use detector::{HardwareCapabilities, detect_hardware};
pub use error::DispatchError;
pub use router::{
    GPU_DISPATCH_THRESHOLD, LuminaDispatcher, LuminaEngine, get_global_dispatcher, step_auto,
};
