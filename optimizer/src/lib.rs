#![deny(warnings)]

pub mod optimizer;
pub mod python_ffi;

pub use optimizer::*;

use pyo3::prelude::*;

#[pymodule]
mod _luminamoon_optimizer_rs {
    #[pymodule_export]
    use crate::python_ffi::{detect_hardware_pyo3, step_pyo3};

    #[allow(non_upper_case_globals)]
    #[pymodule_export]
    const __version__: &str = "1.0.0";
}
