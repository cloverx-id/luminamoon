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
            "1" | "tdclm" | "td-clm" | "single" => Ok(Self::TdClm),
            "2" | "dual" | "clm" => Ok(Self::Dual),
            "t1" | "td-t1" | "t1-clm" | "triangle1" | "td-clm-t1" | "tdclm-t1" => Ok(Self::T1),
            "t2" | "td-t2" | "t2-clm" | "td-dual" | "triangle2" | "clm-t2" | "dual-t2" => {
                Ok(Self::T2)
            }
            _ => {
                if let Ok(parsed) = lower.parse::<i64>() {
                    return Self::try_from_count(parsed);
                }
                let candidates = ["1", "2", "t1", "t2", "single", "dual", "tdclm", "td-clm"];
                if let Some(suggestion) = suggest_closest(&lower, &candidates, 2) {
                    Err(LuminaError::InvalidBufferMode(format!(
                        "Invalid buffer count: '{s}'. LuminaV only supports buffer=1 (Single-Buffer TD-CLM) or buffer=2 (Dual-Buffer CLM). Did you mean '{suggestion}'?"
                    )))
                } else {
                    Err(LuminaError::InvalidBufferMode(format!(
                        "Invalid buffer count: '{s}'. LuminaV only supports buffer=1 (Single-Buffer TD-CLM) or buffer=2 (Dual-Buffer CLM)."
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
            "radial" | "1" => Ok(Self::Radial),
            "coordinate" | "2" => Ok(Self::Coordinate),
            "none" | "0" => Ok(Self::None),
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
                "Invalid master_weights: {code}. Expected False, 'none', 'semi', or 'full'."
            ))),
        }
    }

    pub fn try_from_str(s: &str) -> Result<Self, LuminaError> {
        let lower = s.trim().to_ascii_lowercase();
        match lower.as_str() {
            "false" | "none" | "off" | "0" => Ok(Self::None),
            "semi" | "hybrid" | "half" | "1" => Ok(Self::Semi),
            "full" | "true" | "fp32" | "masterweight" | "2" => Ok(Self::Full),
            _ => {
                let candidates = ["none", "semi", "full"];
                if let Some(suggestion) = suggest_closest(&lower, &candidates, 3) {
                    Err(LuminaError::InvalidMasterWeights(format!(
                        "Invalid master_weights: '{s}'. Expected False, 'none', 'semi', or 'full'. Did you mean '{suggestion}'?"
                    )))
                } else {
                    Err(LuminaError::InvalidMasterWeights(format!(
                        "Invalid master_weights: '{s}'. Expected False, 'none', 'semi', or 'full'."
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
    Vulkan = 3,
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
            3 => Ok(Self::Vulkan),
            4 => Ok(Self::Simd),
            5 => Ok(Self::ScalarCore),
            6 => Ok(Self::CubeCl),
            _ => Err(LuminaError::InvalidExecutionMode(format!(
                "Invalid execution mode: {code}. Expected 'auto', 'vulkan', 'simd', or 'scalar'."
            ))),
        }
    }

    pub fn try_from_str(s: &str) -> Result<Self, LuminaError> {
        let lower = s.trim().to_ascii_lowercase();
        match lower.as_str() {
            "auto" | "foreach" | "0" => Ok(Self::Auto),
            "triton" | "1" => Ok(Self::Triton),
            "single" | "2" => Ok(Self::Single),
            "vulkan" | "3" => Ok(Self::Vulkan),
            "simd" | "4" => Ok(Self::Simd),
            "scalar" | "scalar_core" | "5" => Ok(Self::ScalarCore),
            "cubecl" | "6" => Ok(Self::CubeCl),
            _ => {
                let candidates = [
                    "auto", "cubecl", "vulkan", "simd", "scalar", "triton", "single",
                ];
                if let Some(suggestion) = suggest_closest(&lower, &candidates, 3) {
                    Err(LuminaError::InvalidExecutionMode(format!(
                        "Invalid execution mode: '{s}'. Expected 'auto', 'vulkan', 'simd', or 'scalar'. Did you mean '{suggestion}'?"
                    )))
                } else {
                    Err(LuminaError::InvalidExecutionMode(format!(
                        "Invalid execution mode: '{s}'. Expected 'auto', 'vulkan', 'simd', or 'scalar'."
                    )))
                }
            }
        }
    }
}
