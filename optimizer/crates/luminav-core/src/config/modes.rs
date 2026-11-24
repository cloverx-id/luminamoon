use crate::LuminaError;

use super::suggest_closest;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BufferMode {
    TdClm = 1,
    Dual = 2,
    T1 = 3,
    T2 = 4,
}

impl BufferMode {
    #[inline]
    pub const fn is_dual_state(self) -> bool {
        matches!(self, Self::Dual | Self::T2)
    }

    #[inline]
    pub const fn is_triangle_family(self) -> bool {
        matches!(self, Self::T1 | Self::T2)
    }

    #[inline]
    pub const fn default_ema_decay(self) -> f32 {
        match self {
            Self::TdClm | Self::T1 => 0.95,
            Self::Dual | Self::T2 => 0.80,
        }
    }

    pub fn try_from_count(count: i64) -> Result<Self, LuminaError> {
        match count {
            1 => Ok(Self::TdClm),
            2 => Ok(Self::Dual),
            _ => Err(LuminaError::InvalidBufferCount(count)),
        }
    }

    pub fn try_from_code(code: i64) -> Result<Self, LuminaError> {
        match code {
            1 => Ok(Self::TdClm),
            2 => Ok(Self::Dual),
            3 => Ok(Self::T1),
            4 => Ok(Self::T2),
            _ => Err(LuminaError::InvalidBufferCount(code)),
        }
    }

    pub fn try_from_str(s: &str) -> Result<Self, LuminaError> {
        let lower = s.trim().to_ascii_lowercase();
        match lower.as_str() {
            "1" => Ok(Self::TdClm),
            "2" => Ok(Self::Dual),
            "t1" => Ok(Self::T1),
            "t2" => Ok(Self::T2),
            "tdclm" | "td-clm" | "single" => Err(LuminaError::InvalidBufferMode(format!(
                "Legacy buffer alias '{s}' was removed in LuminaV v2.0.0. Use buffer=1 (Single-Buffer TD-CLM) instead."
            ))),
            "dual" | "clm" => Err(LuminaError::InvalidBufferMode(format!(
                "Legacy buffer alias '{s}' was removed in LuminaV v2.0.0. Use buffer=2 (Dual-Buffer CLM) instead."
            ))),
            "td-t1" | "t1-clm" | "triangle1" | "td-clm-t1" | "tdclm-t1" => {
                Err(LuminaError::InvalidBufferMode(format!(
                    "Legacy buffer alias '{s}' was removed in LuminaV v2.0.0. Use buffer='t1' (Single-Buffer Triangle) instead."
                )))
            }
            "td-t2" | "t2-clm" | "td-dual" | "triangle2" | "clm-t2" | "dual-t2" => {
                Err(LuminaError::InvalidBufferMode(format!(
                    "Legacy buffer alias '{s}' was removed in LuminaV v2.0.0. Use buffer='t2' (Dual-Buffer Triangle) instead."
                )))
            }
            _ => {
                if let Ok(parsed) = lower.parse::<i64>() {
                    return Self::try_from_count(parsed);
                }
                let candidates = ["1", "2", "t1", "t2"];
                if let Some(suggestion) = suggest_closest(&lower, &candidates, 2) {
                    Err(LuminaError::InvalidBufferMode(format!(
                        "Invalid buffer: '{s}'. Expected 1 (TD-CLM), 2 (Dual CLM), 't1' (Triangle 1-Buffer), or 't2' (Triangle 2-Buffer). Did you mean '{suggestion}'?"
                    )))
                } else {
                    Err(LuminaError::InvalidBufferMode(format!(
                        "Invalid buffer: '{s}'. Expected 1 (TD-CLM), 2 (Dual CLM), 't1' (Triangle 1-Buffer), or 't2' (Triangle 2-Buffer)."
                    )))
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BoundType {
    None = 0,
    Radial = 1,
    Coordinate = 2,
}

impl BoundType {
    pub fn try_from_code(code: i64) -> Result<Self, LuminaError> {
        match code {
            0 => Ok(Self::None),
            1 => Ok(Self::Radial),
            2 => Ok(Self::Coordinate),
            _ => Err(LuminaError::InvalidBoundType(format!(
                "Invalid bound_type: {code}. Expected 'coordinate' or 'radial'."
            ))),
        }
    }

    pub fn try_from_str(s: &str) -> Result<Self, LuminaError> {
        let lower = s.trim().to_ascii_lowercase();
        match lower.as_str() {
            "radial" => Ok(Self::Radial),
            "coordinate" => Ok(Self::Coordinate),
            "none" => Ok(Self::None),
            "0" => Err(LuminaError::InvalidBoundType(format!(
                "Legacy bound_type '{s}' was removed in LuminaV v2.0.0. Use bound=false (or 'none' in Rust) instead."
            ))),
            "1" => Err(LuminaError::InvalidBoundType(format!(
                "Legacy bound_type '{s}' was removed in LuminaV v2.0.0. Use 'radial' instead."
            ))),
            "2" => Err(LuminaError::InvalidBoundType(format!(
                "Legacy bound_type '{s}' was removed in LuminaV v2.0.0. Use 'coordinate' instead."
            ))),
            _ => {
                let candidates = ["coordinate", "radial"];
                if let Some(suggestion) = suggest_closest(&lower, &candidates, 4) {
                    Err(LuminaError::InvalidBoundType(format!(
                        "Invalid bound_type: '{s}'. Expected 'coordinate' or 'radial'. Did you mean '{suggestion}'?"
                    )))
                } else {
                    Err(LuminaError::InvalidBoundType(format!(
                        "Invalid bound_type: '{s}'. Expected 'coordinate' or 'radial'."
                    )))
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MasterWeightMode {
    None = 0,
    Semi = 1,
    Full = 2,
}

impl MasterWeightMode {
    #[inline]
    pub const fn is_enabled(self) -> bool {
        !matches!(self, Self::None)
    }

    pub fn try_from_code(code: i64) -> Result<Self, LuminaError> {
        match code {
            0 => Ok(Self::None),
            1 => Ok(Self::Semi),
            2 => Ok(Self::Full),
            _ => Err(LuminaError::InvalidMasterWeights(format!(
                "Invalid master_weights: {code}. Expected 'none', 'semi', or 'full'."
            ))),
        }
    }

    pub fn try_from_str(s: &str) -> Result<Self, LuminaError> {
        let lower = s.trim().to_ascii_lowercase();
        match lower.as_str() {
            "none" => Ok(Self::None),
            "semi" => Ok(Self::Semi),
            "full" => Ok(Self::Full),
            "false" | "off" | "0" => Err(LuminaError::InvalidMasterWeights(format!(
                "Legacy master_weights alias '{s}' was removed in LuminaV v2.0.0. Use 'none' instead."
            ))),
            "hybrid" | "half" | "1" => Err(LuminaError::InvalidMasterWeights(format!(
                "Legacy master_weights alias '{s}' was removed in LuminaV v2.0.0. Use 'semi' instead."
            ))),
            "true" | "fp32" | "masterweight" | "2" => {
                Err(LuminaError::InvalidMasterWeights(format!(
                    "Legacy master_weights alias '{s}' was removed in LuminaV v2.0.0. Use 'full' instead."
                )))
            }
            _ => {
                let candidates = ["none", "semi", "full"];
                if let Some(suggestion) = suggest_closest(&lower, &candidates, 3) {
                    Err(LuminaError::InvalidMasterWeights(format!(
                        "Invalid master_weights: '{s}'. Expected 'none', 'semi', or 'full'. Did you mean '{suggestion}'?"
                    )))
                } else {
                    Err(LuminaError::InvalidMasterWeights(format!(
                        "Invalid master_weights: '{s}'. Expected 'none', 'semi', or 'full'."
                    )))
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DTypeMode {
    Float32 = 0,
    BFloat16 = 1,
    Float16 = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ExecutionMode {
    Auto = 0,
    Triton = 1,
    Single = 2,
    Simd = 4,
    ScalarCore = 5,
    CubeCl = 6,
}

impl ExecutionMode {
    pub fn try_from_code(code: i64) -> Result<Self, LuminaError> {
        match code {
            0 => Ok(Self::Auto),
            1 => Ok(Self::Triton),
            2 => Ok(Self::Single),
            4 => Ok(Self::Simd),
            5 => Ok(Self::ScalarCore),
            6 => Ok(Self::CubeCl),
            _ => Err(LuminaError::InvalidExecutionMode(format!(
                "Invalid execution mode: {code}. Expected 'auto', 'triton', 'cubecl', 'simd', 'eager', or 'scalar'."
            ))),
        }
    }

    pub fn try_from_str(s: &str) -> Result<Self, LuminaError> {
        let lower = s.trim().to_ascii_lowercase();
        match lower.as_str() {
            "auto" => Ok(Self::Auto),
            "triton" => Ok(Self::Triton),
            "eager" => Ok(Self::Single),
            "simd" => Ok(Self::Simd),
            "scalar" => Ok(Self::ScalarCore),
            "cubecl" => Ok(Self::CubeCl),
            "foreach" | "0" => Err(LuminaError::InvalidExecutionMode(format!(
                "Legacy execution mode '{s}' was removed in LuminaV v2.0.0. Use 'auto' instead."
            ))),
            "1" => Err(LuminaError::InvalidExecutionMode(format!(
                "Legacy execution mode '{s}' was removed in LuminaV v2.0.0. Use 'triton' instead."
            ))),
            "single" | "2" => Err(LuminaError::InvalidExecutionMode(format!(
                "Legacy execution mode '{s}' was removed in LuminaV v2.0.0. Use 'eager' instead."
            ))),
            "vulkan" | "3" => Err(LuminaError::InvalidExecutionMode(format!(
                "Legacy execution mode '{s}' was removed in LuminaV v2.0.0. Use 'cubecl' instead."
            ))),
            "4" => Err(LuminaError::InvalidExecutionMode(format!(
                "Legacy execution mode '{s}' was removed in LuminaV v2.0.0. Use 'simd' instead."
            ))),
            "scalar_core" | "5" => Err(LuminaError::InvalidExecutionMode(format!(
                "Legacy execution mode '{s}' was removed in LuminaV v2.0.0. Use 'scalar' instead."
            ))),
            "6" => Err(LuminaError::InvalidExecutionMode(format!(
                "Legacy execution mode '{s}' was removed in LuminaV v2.0.0. Use 'cubecl' instead."
            ))),
            _ => {
                let candidates = ["auto", "triton", "cubecl", "simd", "eager", "scalar"];
                if let Some(suggestion) = suggest_closest(&lower, &candidates, 3) {
                    Err(LuminaError::InvalidExecutionMode(format!(
                        "Invalid execution mode: '{s}'. Expected 'auto', 'triton', 'cubecl', 'simd', 'eager', or 'scalar'. Did you mean '{suggestion}'?"
                    )))
                } else {
                    Err(LuminaError::InvalidExecutionMode(format!(
                        "Invalid execution mode: '{s}'. Expected 'auto', 'triton', 'cubecl', 'simd', 'eager', or 'scalar'."
                    )))
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum VarianceType {
    #[default]
    Standard = 0,
    Rtv = 1,
}

impl VarianceType {
    pub fn try_from_code(code: i64) -> Result<Self, LuminaError> {
        match code {
            0 => Ok(Self::Standard),
            1 => Ok(Self::Rtv),
            _ => Err(LuminaError::InvalidVarianceType(format!(
                "Invalid variance_type: {code}. Expected 'standard' or 'rtv'."
            ))),
        }
    }

    pub fn try_from_str(s: &str) -> Result<Self, LuminaError> {
        let lower = s.trim().to_ascii_lowercase();
        match lower.as_str() {
            "standard" => Ok(Self::Standard),
            "rtv" => Ok(Self::Rtv),
            _ => {
                let candidates = ["standard", "rtv"];
                if let Some(suggestion) = suggest_closest(&lower, &candidates, 3) {
                    Err(LuminaError::InvalidVarianceType(format!(
                        "Invalid variance_type: '{s}'. Expected 'standard' or 'rtv'. Did you mean '{suggestion}'?"
                    )))
                } else {
                    Err(LuminaError::InvalidVarianceType(format!(
                        "Invalid variance_type: '{s}'. Expected 'standard' or 'rtv'."
                    )))
                }
            }
        }
    }
}
