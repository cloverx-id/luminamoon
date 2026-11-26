use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};

use luminav_core::{
    BoundType, BufferMode, ExecutionMode, LuminaVConfig, LuminaVConfigBuilder, MasterWeightMode,
    MomentumType, VarianceType,
};

pub(crate) fn parse_config(dict: &Bound<'_, PyDict>) -> PyResult<LuminaVConfig> {
    let lr: f32 = dict
        .get_item("lr")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(3.0e-4);
    let beta1: f32 = dict
        .get_item("beta1")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0.9);
    let beta2: f32 = dict
        .get_item("beta2")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0.999);
    let eps: f32 = dict
        .get_item("eps")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(1.0e-8);
    let weight_decay: f32 = dict
        .get_item("weight_decay")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(8.0e-2);
    let cautious_weight_decay: bool = dict
        .get_item("cautious_weight_decay")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(false);
    let variance_type = if let Some(val) = dict.get_item("variance_type")? {
        if val.is_instance_of::<PyBool>() {
            return Err(PyValueError::new_err(format!(
                "Invalid variance_type: {}. Expected 'standard' or 'rtv'.",
                val
            )));
        } else if let Ok(s) = val.extract::<String>() {
            VarianceType::try_from_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(code) = val.extract::<i64>() {
            VarianceType::try_from_code(code).map_err(|e| PyValueError::new_err(e.to_string()))?
        } else {
            return Err(PyValueError::new_err(format!(
                "Invalid variance_type: {}. Expected 'standard' or 'rtv'.",
                val
            )));
        }
    } else {
        VarianceType::Standard
    };
    let variance_temper: f32 = dict
        .get_item("variance_temper")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(3.0);
    let momentum_type = if let Some(val) = dict.get_item("momentum_type")? {
        if val.is_instance_of::<PyBool>() {
            return Err(PyValueError::new_err(format!(
                "Invalid momentum_type: {}. Expected 'standard' or 'rpm'.",
                val
            )));
        } else if let Ok(s) = val.extract::<String>() {
            MomentumType::try_from_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(code) = val.extract::<i64>() {
            MomentumType::try_from_code(code).map_err(|e| PyValueError::new_err(e.to_string()))?
        } else {
            return Err(PyValueError::new_err(format!(
                "Invalid momentum_type: {}. Expected 'standard' or 'rpm'.",
                val
            )));
        }
    } else {
        MomentumType::Standard
    };
    let tau: f32 = dict
        .get_item("tau")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0.8);
    let lam: f32 = dict
        .get_item("lam")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0.3);
    let cautious: bool = dict
        .get_item("cautious")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(true);
    let cautious_clamp_min: f32 = dict
        .get_item("cautious_clamp_min")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0.5);

    let buffer = if let Some(val) = dict.get_item("buffer")? {
        if val.is_instance_of::<PyBool>() {
            return Err(PyValueError::new_err(format!(
                "Invalid buffer count: {}. LuminaV only supports buffer=1 (Single-Buffer TD-CLM), buffer=2 (Dual-Buffer CLM), buffer='T1' (Single-Buffer Triangle), or buffer='T2' (Dual-Buffer Triangle).",
                val
            )));
        } else if let Ok(s) = val.extract::<String>() {
            BufferMode::try_from_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(code) = val.extract::<i64>() {
            BufferMode::try_from_count(code).map_err(|e| PyValueError::new_err(e.to_string()))?
        } else {
            return Err(PyValueError::new_err(format!(
                "Invalid buffer count: {}. LuminaV only supports buffer=1 (Single-Buffer TD-CLM), buffer=2 (Dual-Buffer CLM), buffer='T1' (Single-Buffer Triangle), or buffer='T2' (Dual-Buffer Triangle).",
                val
            )));
        }
    } else {
        BufferMode::Dual
    };

    let stochastic_rounding: bool = dict
        .get_item("stochastic_rounding")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(true);
    let bound: bool = dict
        .get_item("bound")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(true);

    let bound_type = if let Some(val) = dict.get_item("bound_type")? {
        if val.is_instance_of::<PyBool>() {
            return Err(PyValueError::new_err(format!(
                "Invalid bound_type: {}. Expected 'coordinate' or 'radial'.",
                val
            )));
        } else if let Ok(s) = val.extract::<String>() {
            BoundType::try_from_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(code) = val.extract::<i64>() {
            BoundType::try_from_code(code).map_err(|e| PyValueError::new_err(e.to_string()))?
        } else {
            return Err(PyValueError::new_err(format!(
                "Invalid bound_type: {}. Expected 'coordinate' or 'radial'.",
                val
            )));
        }
    } else {
        BoundType::Coordinate
    };

    let bound_ratio: f32 = dict
        .get_item("bound_ratio")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0.05);
    let bound_floor: f32 = dict
        .get_item("bound_floor")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(1.0e-2);

    let master_weights = if let Some(val) = dict.get_item("master_weights")? {
        if val.is_instance_of::<PyBool>() {
            let b: bool = val.extract()?;
            let replacement = if b { "'full'" } else { "'none'" };
            return Err(PyValueError::new_err(format!(
                "Legacy boolean master_weights={val} was removed in LuminaV v2.0.0. Use master_weights={replacement} instead."
            )));
        } else if let Ok(s) = val.extract::<String>() {
            MasterWeightMode::try_from_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(code) = val.extract::<i64>() {
            MasterWeightMode::try_from_code(code)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else {
            return Err(PyValueError::new_err(format!(
                "Invalid master_weights: {}. Expected 'none', 'semi', or 'full'.",
                val
            )));
        }
    } else {
        MasterWeightMode::None
    };

    let execution = if let Some(val) = dict.get_item("execution")? {
        if val.is_instance_of::<PyBool>() {
            return Err(PyValueError::new_err(format!(
                "Invalid execution mode: {}. Expected 'auto', 'triton', 'cubecl', 'simd', 'eager', or 'scalar'.",
                val
            )));
        } else if let Ok(s) = val.extract::<String>() {
            ExecutionMode::try_from_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(code) = val.extract::<i64>() {
            ExecutionMode::try_from_code(code).map_err(|e| PyValueError::new_err(e.to_string()))?
        } else {
            return Err(PyValueError::new_err(format!(
                "Invalid execution mode: {}. Expected 'auto', 'triton', 'cubecl', 'simd', 'eager', or 'scalar'.",
                val
            )));
        }
    } else {
        ExecutionMode::Auto
    };

    let fused_single_pass: bool = dict
        .get_item("fused_single_pass")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(true);
    let ema_decay: f32 = if let Some(val) = dict.get_item("ema_decay")? {
        if val.is_none() {
            buffer.default_ema_decay()
        } else if let Ok(s) = val.extract::<String>() {
            if s.trim().eq_ignore_ascii_case("auto") {
                buffer.default_ema_decay()
            } else {
                return Err(PyValueError::new_err(format!(
                    "Invalid ema_decay: '{s}'. Expected 'auto', None, or a float in (0.0, 1.0)."
                )));
            }
        } else {
            val.extract()?
        }
    } else {
        buffer.default_ema_decay()
    };
    let fsdp_sync: bool = dict
        .get_item("fsdp_sync")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(false);
    let seed: u32 = dict
        .get_item("seed")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(1337);

    LuminaVConfigBuilder::default()
        .lr(lr)
        .betas(beta1, beta2)
        .eps(eps)
        .weight_decay(weight_decay)
        .cautious_weight_decay(cautious_weight_decay)
        .variance_type(variance_type)
        .variance_temper(variance_temper)
        .momentum_type(momentum_type)
        .tau(tau)
        .lam(lam)
        .cautious(cautious)
        .cautious_clamp_min(cautious_clamp_min)
        .buffer(buffer)
        .stochastic_rounding(stochastic_rounding)
        .bound(bound)
        .bound_type(bound_type)
        .bound_ratio(bound_ratio)
        .bound_floor(bound_floor)
        .master_weights(master_weights)
        .execution(execution)
        .fused_single_pass(fused_single_pass)
        .ema_decay(ema_decay)
        .fsdp_sync(fsdp_sync)
        .seed(seed)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}
