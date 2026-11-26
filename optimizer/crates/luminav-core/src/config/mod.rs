mod modes;

use crate::LuminaError;

pub use modes::{
    BoundType, BufferMode, DTypeMode, ExecutionMode, MasterWeightMode, MomentumType, VarianceType,
};

pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = usize::from(a_chars[i - 1] != b_chars[j - 1]);
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

pub fn suggest_closest<'a>(
    input: &str,
    candidates: &[&'a str],
    max_dist: usize,
) -> Option<&'a str> {
    let normalized = input.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }
    let mut best: Option<(&'a str, usize)> = None;
    for &cand in candidates {
        let dist = levenshtein_distance(&normalized, cand);
        if dist <= max_dist {
            match best {
                Some((_, best_dist)) if dist < best_dist => best = Some((cand, dist)),
                None => best = Some((cand, dist)),
                _ => {}
            }
        }
    }
    best.map(|(cand, _)| cand)
}

#[derive(Debug, Clone, PartialEq)]
pub struct LuminaVConfig {
    pub lr: f32,
    pub betas: (f32, f32),
    pub eps: f32,
    pub weight_decay: f32,
    pub tau: f32,
    pub lam: f32,
    pub cautious: bool,
    pub cautious_clamp_min: f32,
    pub cautious_weight_decay: bool,
    pub buffer: BufferMode,
    pub variance_type: VarianceType,
    pub variance_temper: f32,
    pub momentum_type: MomentumType,
    pub stochastic_rounding: bool,
    pub bound: bool,
    pub bound_type: BoundType,
    pub bound_ratio: f32,
    pub bound_floor: f32,
    pub master_weights: MasterWeightMode,
    pub execution: ExecutionMode,
    pub fused_single_pass: bool,
    pub ema_decay: f32,
    pub fsdp_sync: bool,
    pub seed: u32,
}

impl Default for LuminaVConfig {
    fn default() -> Self {
        Self {
            lr: 3.0e-4,
            betas: (0.9, 0.999),
            eps: 1.0e-8,
            weight_decay: 8.0e-2,
            tau: 0.8,
            lam: 0.3,
            cautious: true,
            cautious_clamp_min: 0.5,
            cautious_weight_decay: false,
            buffer: BufferMode::Dual,
            variance_type: VarianceType::Standard,
            variance_temper: 3.0,
            momentum_type: MomentumType::Standard,
            stochastic_rounding: true,
            bound: true,
            bound_type: BoundType::Coordinate,
            bound_ratio: 0.05,
            bound_floor: 1.0e-2,
            master_weights: MasterWeightMode::None,
            execution: ExecutionMode::Auto,
            fused_single_pass: true,
            ema_decay: BufferMode::Dual.default_ema_decay(),
            fsdp_sync: false,
            seed: 1337,
        }
    }
}

impl LuminaVConfig {
    #[inline]
    pub fn builder() -> LuminaVConfigBuilder {
        LuminaVConfigBuilder::default()
    }

    pub fn validate(&self) -> Result<(), LuminaError> {
        if !self.lr.is_finite() || self.lr < 0.0 {
            return Err(LuminaError::InvalidLearningRate(self.lr));
        }
        if !self.betas.0.is_finite() || !(0.0..1.0).contains(&self.betas.0) {
            return Err(LuminaError::InvalidBeta1(self.betas.0));
        }
        if !self.betas.1.is_finite() || !(0.0..1.0).contains(&self.betas.1) {
            return Err(LuminaError::InvalidBeta2(self.betas.1));
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err(LuminaError::InvalidEpsilon(self.eps));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(LuminaError::InvalidWeightDecay(self.weight_decay));
        }
        if !self.tau.is_finite() || self.tau <= 0.0 {
            return Err(LuminaError::InvalidTau(self.tau));
        }
        if !self.lam.is_finite() || self.lam < 0.0 {
            return Err(LuminaError::InvalidLambda(self.lam));
        }
        if !self.cautious_clamp_min.is_finite()
            || self.cautious_clamp_min <= 0.0
            || self.cautious_clamp_min > 1.0
        {
            return Err(LuminaError::InvalidCautiousClampMin(
                self.cautious_clamp_min,
            ));
        }
        if !self.bound_ratio.is_finite() || self.bound_ratio <= 0.0 {
            return Err(LuminaError::InvalidBoundRatio(self.bound_ratio));
        }
        if !self.bound_floor.is_finite() || self.bound_floor <= 0.0 {
            return Err(LuminaError::InvalidBoundFloor(self.bound_floor));
        }
        if !self.variance_temper.is_finite() || self.variance_temper <= 0.0 {
            return Err(LuminaError::InvalidVarianceTemper(self.variance_temper));
        }
        if !self.ema_decay.is_finite() || self.ema_decay <= 0.0 || self.ema_decay >= 1.0 {
            return Err(LuminaError::InvalidEmaDecay(self.ema_decay));
        }
        Ok(())
    }

    #[inline]
    pub const fn uses_rtv(&self) -> bool {
        matches!(self.variance_type, VarianceType::Rtv) && self.buffer.is_dual_state()
    }

    #[inline]
    pub const fn uses_rpm(&self) -> bool {
        matches!(self.momentum_type, MomentumType::Rpm)
    }

    #[inline]
    pub const fn effective_bound_type(&self) -> BoundType {
        if self.bound && self.bound_ratio > 0.0 {
            self.bound_type
        } else {
            BoundType::None
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LuminaVConfigBuilder {
    config: LuminaVConfig,
    explicit_ema_decay: Option<f32>,
}

impl LuminaVConfigBuilder {
    #[inline]
    pub fn lr(mut self, lr: f32) -> Self {
        self.config.lr = lr;
        self
    }

    #[inline]
    pub fn betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.config.betas = (beta1, beta2);
        self
    }

    #[inline]
    pub fn eps(mut self, eps: f32) -> Self {
        self.config.eps = eps;
        self
    }

    #[inline]
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.config.weight_decay = weight_decay;
        self
    }

    #[inline]
    pub fn tau(mut self, tau: f32) -> Self {
        self.config.tau = tau;
        self
    }

    #[inline]
    pub fn lam(mut self, lam: f32) -> Self {
        self.config.lam = lam;
        self
    }

    #[inline]
    pub fn cautious(mut self, cautious: bool) -> Self {
        self.config.cautious = cautious;
        self
    }

    #[inline]
    pub fn cautious_clamp_min(mut self, clamp_min: f32) -> Self {
        self.config.cautious_clamp_min = clamp_min;
        self
    }

    #[inline]
    pub fn cautious_weight_decay(mut self, cautious_weight_decay: bool) -> Self {
        self.config.cautious_weight_decay = cautious_weight_decay;
        self
    }

    #[inline]
    pub fn buffer(mut self, buffer: BufferMode) -> Self {
        self.config.buffer = buffer;
        self
    }

    #[inline]
    pub fn variance_type(mut self, variance_type: VarianceType) -> Self {
        self.config.variance_type = variance_type;
        self
    }

    #[inline]
    pub fn variance_temper(mut self, variance_temper: f32) -> Self {
        self.config.variance_temper = variance_temper;
        self
    }

    #[inline]
    pub fn momentum_type(mut self, momentum_type: MomentumType) -> Self {
        self.config.momentum_type = momentum_type;
        self
    }

    #[inline]
    pub fn stochastic_rounding(mut self, stochastic_rounding: bool) -> Self {
        self.config.stochastic_rounding = stochastic_rounding;
        self
    }

    #[inline]
    pub fn bound(mut self, bound: bool) -> Self {
        self.config.bound = bound;
        self
    }

    #[inline]
    pub fn bound_type(mut self, bound_type: BoundType) -> Self {
        self.config.bound_type = bound_type;
        self
    }

    #[inline]
    pub fn bound_ratio(mut self, bound_ratio: f32) -> Self {
        self.config.bound_ratio = bound_ratio;
        self
    }

    #[inline]
    pub fn bound_floor(mut self, bound_floor: f32) -> Self {
        self.config.bound_floor = bound_floor;
        self
    }

    #[inline]
    pub fn master_weights(mut self, master_weights: MasterWeightMode) -> Self {
        self.config.master_weights = master_weights;
        self
    }

    #[inline]
    pub fn execution(mut self, execution: ExecutionMode) -> Self {
        self.config.execution = execution;
        self
    }

    #[inline]
    pub fn fused_single_pass(mut self, fused_single_pass: bool) -> Self {
        self.config.fused_single_pass = fused_single_pass;
        self
    }

    #[inline]
    pub fn ema_decay(mut self, ema_decay: f32) -> Self {
        self.explicit_ema_decay = Some(ema_decay);
        self
    }

    #[inline]
    pub fn fsdp_sync(mut self, fsdp_sync: bool) -> Self {
        self.config.fsdp_sync = fsdp_sync;
        self
    }

    #[inline]
    pub fn seed(mut self, seed: u32) -> Self {
        self.config.seed = seed;
        self
    }

    pub fn build(mut self) -> Result<LuminaVConfig, LuminaError> {
        self.config.ema_decay = self
            .explicit_ema_decay
            .unwrap_or_else(|| self.config.buffer.default_ema_decay());
        self.config.validate()?;
        Ok(self.config)
    }
}
