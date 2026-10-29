use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};

use luminav_core::{
    BoundType, BufferMode, ExecutionMode, LuminaVConfig, LuminaVConfigBuilder, MasterWeightMode,
};

pub(crate) fn parse_config(dict: &Bound<'_, PyDict>) -> PyResult<LuminaVConfig> {
    let lr: f32 = dict
        .get_item("lr")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(8.0e-4);
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
                "Invalid buffer count: {}. LuminaV only supports buffer=1 (Single-Buffer TD-CLM) or buffer=2 (Dual-Buffer CLM).",
                val
            )));
        } else if let Ok(s) = val.extract::<String>() {
            BufferMode::try_from_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(code) = val.extract::<i64>() {
            BufferMode::try_from_count(code).map_err(|e| PyValueError::new_err(e.to_string()))?
        } else {
            return Err(PyValueError::new_err(format!(
                "Invalid buffer count: {}. LuminaV only supports buffer=1 (Single-Buffer TD-CLM) or buffer=2 (Dual-Buffer CLM).",
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

    let master_weights = if let Some(val) = dict.get_item("master_weights")? {
        if val.is_instance_of::<PyBool>() {
            let b: bool = val.extract()?;
            if b {
                MasterWeightMode::Full
            } else {
                MasterWeightMode::None
            }
        } else if let Ok(s) = val.extract::<String>() {
            MasterWeightMode::try_from_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(code) = val.extract::<i64>() {
            MasterWeightMode::try_from_code(code)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else {
            return Err(PyValueError::new_err(format!(
                "Invalid master_weights: {}. Expected False, 'none', 'semi', or 'full'.",
                val
            )));
        }
    } else {
        MasterWeightMode::None
    };

    let execution = if let Some(val) = dict.get_item("execution")? {
        if val.is_instance_of::<PyBool>() {
            return Err(PyValueError::new_err(format!(
                "Invalid execution mode: {}. Expected 'auto', 'vulkan', 'simd', or 'scalar'.",
                val
            )));
        } else if let Ok(s) = val.extract::<String>() {
            ExecutionMode::try_from_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(code) = val.extract::<i64>() {
            ExecutionMode::try_from_code(code).map_err(|e| PyValueError::new_err(e.to_string()))?
        } else {
            return Err(PyValueError::new_err(format!(
                "Invalid execution mode: {}. Expected 'auto', 'vulkan', 'simd', or 'scalar'.",
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
    let ema_decay: f32 = dict
        .get_item("ema_decay")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_else(|| buffer.default_ema_decay());
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
        .tau(tau)
        .lam(lam)
        .cautious(cautious)
        .cautious_clamp_min(cautious_clamp_min)
        .buffer(buffer)
        .stochastic_rounding(stochastic_rounding)
        .bound(bound)
        .bound_type(bound_type)
        .bound_ratio(bound_ratio)
        .master_weights(master_weights)
        .execution(execution)
        .fused_single_pass(fused_single_pass)
        .ema_decay(ema_decay)
        .fsdp_sync(fsdp_sync)
        .seed(seed)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}
