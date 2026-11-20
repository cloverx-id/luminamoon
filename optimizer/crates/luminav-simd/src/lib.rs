#![cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    allow(unused_imports)
)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]

pub mod avx2;
pub mod avx512;
pub mod common;
pub(crate) mod dispatch;
pub mod neon;
pub mod parallel;
pub mod ra;
pub(crate) mod scalar;

use luminav_core::{BufferMode, LuminaError, LuminaVConfig, TensorViewMut};

pub use common::ChunkMetrics;
pub use parallel::{
    ELEM_CHUNK_SIZE, dual_step_parallel, dual_step_parallel_chunked, tdclm_step_parallel,
    tdclm_step_parallel_chunked,
};
pub use ra::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    Avx512,
    Avx2,
    Neon,
    Scalar,
}

impl SimdBackend {
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Avx512 => "x86_64 AVX-512 (avx512f+dq+bw+vl+fma)",
            Self::Avx2 => "x86_64 AVX2 (avx2+fma)",
            Self::Neon => "aarch64 NEON (asimd)",
            Self::Scalar => "Scalar Reference Fallback",
        }
    }
}

#[inline]
pub fn detect_simd_backend() -> SimdBackend {
    std::cfg_select! {
        target_arch = "x86_64" => {
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512dq")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vl")
                && std::arch::is_x86_feature_detected!("fma")
            {
                SimdBackend::Avx512
            } else if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                SimdBackend::Avx2
            } else {
                SimdBackend::Scalar
            }
        }
        target_arch = "aarch64" => {
            if std::arch::is_aarch64_feature_detected!("neon") {
                SimdBackend::Neon
            } else {
                SimdBackend::Scalar
            }
        }
        _ => {
            SimdBackend::Scalar
        }
    }
}

pub fn step_tdclm(
    view: &mut TensorViewMut<'_>,
    config: &LuminaVConfig,
    step: u64,
    param_index: usize,
) -> Result<(), LuminaError> {
    let backend = detect_simd_backend();
    if view.param.len() > ELEM_CHUNK_SIZE && rayon::current_thread_index().is_none() {
        return tdclm_step_parallel(view, config, step, param_index, backend);
    }

    match backend {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx512 => unsafe {
            avx512::tdclm_step_avx512(view, config, step, param_index)
        },
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 => unsafe { avx2::tdclm_step_avx2(view, config, step, param_index) },
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => unsafe { neon::tdclm_step_neon(view, config, step, param_index) },
        _ => luminav_core::single_step_tdclm_scalar(view, config, step, param_index),
    }
}

pub fn step_t1(
    view: &mut TensorViewMut<'_>,
    config: &LuminaVConfig,
    step: u64,
    param_index: usize,
) -> Result<(), LuminaError> {
    if config.buffer != BufferMode::T1 {
        let mut cfg = config.clone();
        cfg.buffer = BufferMode::T1;
        return step_tdclm(view, &cfg, step, param_index);
    }
    step_tdclm(view, config, step, param_index)
}

pub fn step_dual(
    view: &mut TensorViewMut<'_>,
    config: &LuminaVConfig,
    step: u64,
    param_index: usize,
) -> Result<(), LuminaError> {
    let backend = detect_simd_backend();
    if view.param.len() > ELEM_CHUNK_SIZE && rayon::current_thread_index().is_none() {
        return dual_step_parallel(view, config, step, param_index, backend);
    }

    match backend {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx512 => unsafe { avx512::dual_step_avx512(view, config, step, param_index) },
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 => unsafe { avx2::dual_step_avx2(view, config, step, param_index) },
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => unsafe { neon::dual_step_neon(view, config, step, param_index) },
        _ => luminav_core::single_step_dual_scalar(view, config, step, param_index),
    }
}

pub fn step_t2(
    view: &mut TensorViewMut<'_>,
    config: &LuminaVConfig,
    step: u64,
    param_index: usize,
) -> Result<(), LuminaError> {
    if config.buffer != BufferMode::T2 {
        let mut cfg = config.clone();
        cfg.buffer = BufferMode::T2;
        return step_dual(view, &cfg, step, param_index);
    }
    step_dual(view, config, step, param_index)
}

pub fn step(
    view: &mut TensorViewMut<'_>,
    config: &LuminaVConfig,
    step: u64,
    param_index: usize,
) -> Result<(), LuminaError> {
    match config.buffer {
        BufferMode::TdClm | BufferMode::T1 => step_tdclm(view, config, step, param_index),
        BufferMode::Dual | BufferMode::T2 => step_dual(view, config, step, param_index),
    }
}
