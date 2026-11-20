use luminav_core::{BiasCorrectionCoeffs, BoundType, DTypeMode, LuminaVConfig};

#[derive(Debug, Clone, Copy, Default)]
pub struct ChunkMetrics {
    pub mask_count: u64,
    pub u_sq_sum: f32,
    pub p_sq_sum: f32,
}

impl ChunkMetrics {
    #[inline]
    pub fn combine(self, other: Self) -> Self {
        Self {
            mask_count: self.mask_count + other.mask_count,
            u_sq_sum: self.u_sq_sum + other.u_sq_sum,
            p_sq_sum: self.p_sq_sum + other.p_sq_sum,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DualChunkCtx<'a> {
    pub config: &'a LuminaVConfig,
    pub coeffs: &'a BiasCorrectionCoeffs,
    pub dtype_mode: DTypeMode,
    pub state_dtype_mode: DTypeMode,
    pub chunk_offset: usize,
    pub cautious_scale: f32,
    pub bound_scale: f32,
    pub seed_m: i32,
    pub seed_p: i32,
    pub weyl_m: u32,
    pub weyl_v: u32,
    pub weyl_p: u32,
    pub accumulate_metrics: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TdclmChunkCtx<'a> {
    pub config: &'a LuminaVConfig,
    pub dtype_mode: DTypeMode,
    pub state_dtype_mode: DTypeMode,
    pub chunk_offset: usize,
    pub cautious_scale: f32,
    pub bound_scale: f32,
    pub seed_m: i32,
    pub seed_p: i32,
    pub weyl_m: u32,
    pub weyl_p: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DualParams {
    pub bound_mode: BoundType,
    pub is_t2: bool,
    pub beta1: f32,
    pub one_minus_beta1: f32,
    pub beta2: f32,
    pub one_minus_beta2: f32,
    pub beta1_sq: f32,
    pub one_minus_beta1_sq: f32,
    pub lam1: f32,
    pub c_sigma: f32,
    pub effective_eps: f32,
    pub lr_wd: f32,
    pub clamp_v: bool,
    pub state_dtype_mode: DTypeMode,
}

impl DualParams {
    #[inline]
    pub fn new(
        config: &LuminaVConfig,
        coeffs: &BiasCorrectionCoeffs,
        dtype_mode: DTypeMode,
        state_dtype_mode: DTypeMode,
    ) -> Self {
        let (beta1, beta2) = config.betas;
        let beta1_sq = beta1 * beta1;
        let clamp_v = dtype_mode == DTypeMode::Float16 || state_dtype_mode == DTypeMode::Float16;
        let effective_state_dtype = if state_dtype_mode == DTypeMode::Float16 {
            DTypeMode::BFloat16
        } else {
            state_dtype_mode
        };
        Self {
            bound_mode: config.effective_bound_type(),
            is_t2: config.buffer.is_triangle_family(),
            beta1,
            one_minus_beta1: 1.0_f32 - beta1,
            beta2,
            one_minus_beta2: 1.0_f32 - beta2,
            beta1_sq,
            one_minus_beta1_sq: 1.0_f32 - beta1_sq,
            lam1: 0.5_f32 * config.lam,
            c_sigma: (0.5_f32 * config.lam) / coeffs.bc2.max(1.0e-15_f32).sqrt(),
            effective_eps: config.eps,
            lr_wd: config.lr * config.weight_decay,
            clamp_v,
            state_dtype_mode: effective_state_dtype,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TdclmParams {
    pub bound_mode: BoundType,
    pub is_t1: bool,
    pub beta1: f32,
    pub one_minus_beta1: f32,
    pub beta1_sq: f32,
    pub one_minus_beta1_sq: f32,
    pub effective_eps: f32,
    pub lr_wd: f32,
    pub state_dtype_mode: DTypeMode,
}

impl TdclmParams {
    #[inline]
    pub fn new(config: &LuminaVConfig, state_dtype_mode: DTypeMode) -> Self {
        let beta1 = config.betas.0;
        let beta1_sq = beta1 * beta1;
        let effective_state_dtype = if state_dtype_mode == DTypeMode::Float16 {
            DTypeMode::BFloat16
        } else {
            state_dtype_mode
        };
        Self {
            bound_mode: config.effective_bound_type(),
            is_t1: config.buffer.is_triangle_family(),
            beta1,
            one_minus_beta1: 1.0_f32 - beta1,
            beta1_sq,
            one_minus_beta1_sq: 1.0_f32 - beta1_sq,
            effective_eps: config.eps,
            lr_wd: config.lr * config.weight_decay,
            state_dtype_mode: effective_state_dtype,
        }
    }
}
