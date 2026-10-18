#![deny(warnings)]

pub mod config;
pub mod math;
pub mod prng;
pub mod reference_cpu;
pub mod state;

use thiserror::Error;

pub use config::{
    BoundType, BufferMode, DTypeMode, ExecutionMode, LuminaVConfig, LuminaVConfigBuilder,
    MasterWeightMode, levenshtein_distance, suggest_closest,
};
pub use math::{
    BiasCorrectionCoeffs, apply_coordinate_bound, apply_fp16_safety_bound, compute_bias_correction,
    compute_cautious_scale, compute_radial_bound_scale, reduce_sum_16, sanitize_gradient,
    tanh_pade_f32, update_post_step_ema,
};
pub use prng::{
    _prng, derive_kernel_seeds, f16_bits_to_f32, f32_to_f16_bits_rne, quantize_value,
    round_to_nearest_bf16, round_to_nearest_fp16, stochastic_round_bf16,
    stochastic_round_bf16_bits, stochastic_round_fp16, stochastic_round_fp16_bits,
};
pub use reference_cpu::{single_step_dual_scalar, single_step_tdclm_scalar};
pub use state::{ParamState, ScratchBuffer, TensorViewMut};

#[derive(Debug, Error, Clone, PartialEq)]
pub enum LuminaError {
    #[error("Invalid learning rate: {0}. Expected a finite value >= 0.0.")]
    InvalidLearningRate(f32),

    #[error("Invalid beta1 parameter: {0}. Expected a value in the range [0.0, 1.0).")]
    InvalidBeta1(f32),

    #[error("Invalid beta2 parameter: {0}. Expected a value in the range [0.0, 1.0).")]
    InvalidBeta2(f32),

    #[error("Invalid epsilon value: {0}. Expected a finite value > 0.0.")]
    InvalidEpsilon(f32),

    #[error("Invalid weight_decay value: {0}. Expected a finite value >= 0.0.")]
    InvalidWeightDecay(f32),

    #[error("Invalid tau parameter: {0}. Expected a finite value > 0.0.")]
    InvalidTau(f32),

    #[error("Invalid lam (triangle defect ratio) parameter: {0}. Expected a finite value >= 0.0.")]
    InvalidLambda(f32),

    #[error("Invalid cautious_clamp_min: {0}. Expected a value in the range (0.0, 1.0].")]
    InvalidCautiousClampMin(f32),

    #[error("Invalid bound_ratio: {0}. Expected a finite value > 0.0.")]
    InvalidBoundRatio(f32),

    #[error("Invalid ema_decay: {0}. Expected 'auto' or a float in the range (0.0, 1.0).")]
    InvalidEmaDecay(f32),

    #[error(
        "Invalid buffer count: {0}. LuminaV only supports buffer=1 (Single-Buffer TD-CLM) or buffer=2 (Dual-Buffer CLM)."
    )]
    InvalidBufferCount(i64),

    #[error("{0}")]
    InvalidBufferMode(String),

    #[error("{0}")]
    InvalidBoundType(String),

    #[error("{0}")]
    InvalidMasterWeights(String),

    #[error("{0}")]
    InvalidExecutionMode(String),

    #[error("Buffer length mismatch: expected {expected} elements, got {actual}")]
    ShapeMismatch { expected: usize, actual: usize },

    #[error("Dual-buffer mode requires an initialized exp_avg_sq buffer")]
    MissingSecondaryStateBuffer,

    #[error("Master weight mode requires an initialized master_param buffer")]
    MissingMasterParameterBuffer,
}
