pub mod config_parser;
pub mod extract;
pub mod worker;

use std::collections::HashSet;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use luminav_core::{DTypeMode, ExecutionMode};
use luminav_dispatch::GPU_DISPATCH_THRESHOLD;

use config_parser::parse_config;
pub use extract::detect_hardware_pyo3;
use extract::{extract_scratch_ptr, extract_tensor_info};
use worker::{WorkItem, process_work_item};

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn step_pyo3<'py>(
    py: Python<'py>,
    params: Vec<Bound<'py, PyAny>>,
    grads: Vec<Bound<'py, PyAny>>,
    exp_avgs: Vec<Bound<'py, PyAny>>,
    exp_avg_sqs: Vec<Option<Bound<'py, PyAny>>>,
    master_params: Vec<Option<Bound<'py, PyAny>>>,
    scratches: Vec<Bound<'py, PyAny>>,
    steps: Vec<u64>,
    param_indices: Vec<usize>,
    config_dict: Bound<'py, PyDict>,
) -> PyResult<()> {
    let config = parse_config(&config_dict)?;
    let count = params.len();
    if count == 0 {
        return Ok(());
    }

    if grads.len() != count
        || exp_avgs.len() != count
        || exp_avg_sqs.len() != count
        || master_params.len() != count
        || scratches.len() != count
        || steps.len() != count
        || param_indices.len() != count
    {
        return Err(PyValueError::new_err(
            "Mismatched input slice lengths for step_pyo3.",
        ));
    }

    let mut items = Vec::with_capacity(count);

    for i in 0..count {
        let (param_ptr, numel, dtype_mode) = extract_tensor_info(&params[i], "param")?;
        let (grad_ptr, grad_numel, grad_dtype_mode) = extract_tensor_info(&grads[i], "grad")?;
        let (exp_avg_ptr, exp_numel, state_dtype_mode) =
            extract_tensor_info(&exp_avgs[i], "exp_avg")?;
        if numel == 0 && grad_numel == 0 && exp_numel == 0 {
            continue;
        }
        let scratch_ptr = extract_scratch_ptr(&scratches[i])?;

        if grad_numel != numel || exp_numel != numel {
            return Err(PyValueError::new_err(format!(
                "Shape mismatch for param index {}: param={}, grad={}, exp_avg={}.",
                param_indices[i], numel, grad_numel, exp_numel
            )));
        }

        let exp_avg_sq_ptr = if let Some(ref sq) = exp_avg_sqs[i] {
            let (ptr, sq_numel, sq_dtype) = extract_tensor_info(sq, "exp_avg_sq")?;
            if sq_numel != numel {
                return Err(PyValueError::new_err(format!(
                    "Shape mismatch for exp_avg_sq at param index {}: expected {}, got {}.",
                    param_indices[i], numel, sq_numel
                )));
            }
            if sq_dtype != state_dtype_mode {
                return Err(PyValueError::new_err(format!(
                    "DType mismatch between exp_avg ({:?}) and exp_avg_sq ({:?}) at param index {}.",
                    state_dtype_mode, sq_dtype, param_indices[i]
                )));
            }
            Some(ptr)
        } else {
            None
        };

        let master_ptr = if let Some(ref mp) = master_params[i] {
            let (ptr, mp_numel, mp_dtype) = extract_tensor_info(mp, "master_param")?;
            if mp_numel != numel {
                return Err(PyValueError::new_err(format!(
                    "Shape mismatch for master_param at param index {}: expected {}, got {}.",
                    param_indices[i], numel, mp_numel
                )));
            }
            if mp_dtype != DTypeMode::Float32 {
                return Err(PyValueError::new_err(
                    "Tensor/buffer 'master_param' must have dtype float32.",
                ));
            }
            Some(ptr)
        } else {
            None
        };

        items.push(WorkItem {
            param_ptr,
            master_ptr,
            grad_ptr,
            exp_avg_ptr,
            exp_avg_sq_ptr,
            scratch_ptr,
            numel,
            dtype_mode,
            grad_dtype_mode,
            state_dtype_mode,
            step: steps[i],
            param_index: param_indices[i],
        });
    }

    if items.is_empty() {
        return Ok(());
    }

    // Pointer anti-aliasing validation across concurrent tasks
    let mut seen_mut_ptrs = HashSet::with_capacity(count * 4);
    for it in &items {
        if it.grad_ptr == it.param_ptr || it.grad_ptr == it.exp_avg_ptr {
            return Err(PyValueError::new_err(format!(
                "Pointer aliasing detected: grad_ptr 0x{:x} aliases mutable state at param index {}.",
                it.grad_ptr, it.param_index
            )));
        }
        for &ptr in &[it.param_ptr, it.exp_avg_ptr, it.scratch_ptr] {
            if !seen_mut_ptrs.insert(ptr) {
                return Err(PyValueError::new_err(format!(
                    "Pointer aliasing detected in step_pyo3: mutable pointer 0x{:x} is referenced multiple times.",
                    ptr
                )));
            }
        }
        if let Some(sq_ptr) = it.exp_avg_sq_ptr
            && !seen_mut_ptrs.insert(sq_ptr)
        {
            return Err(PyValueError::new_err(format!(
                "Pointer aliasing detected in step_pyo3: mutable exp_avg_sq pointer 0x{:x} is referenced multiple times.",
                sq_ptr
            )));
        }
        if let Some(mp_ptr) = it.master_ptr
            && mp_ptr != it.param_ptr
            && !seen_mut_ptrs.insert(mp_ptr)
        {
            return Err(PyValueError::new_err(format!(
                "Pointer aliasing detected in step_pyo3: mutable master_param pointer 0x{:x} is referenced multiple times.",
                mp_ptr
            )));
        }
    }
    for it in &items {
        if seen_mut_ptrs.contains(&it.grad_ptr) {
            return Err(PyValueError::new_err(format!(
                "Pointer aliasing detected: grad_ptr 0x{:x} aliases mutable state at param index {}.",
                it.grad_ptr, it.param_index
            )));
        }
    }

    let result = py.detach(|| {
        if items.len() > 1 {
            match config.execution {
                ExecutionMode::Simd | ExecutionMode::ScalarCore | ExecutionMode::Single => items
                    .par_iter()
                    .try_for_each(|item| process_work_item(item, &config)),
                ExecutionMode::Auto => {
                    let (cpu_items, gpu_items): (Vec<_>, Vec<_>) = items
                        .iter()
                        .partition(|it| it.numel < GPU_DISPATCH_THRESHOLD);

                    if !cpu_items.is_empty() {
                        cpu_items
                            .par_iter()
                            .try_for_each(|item| process_work_item(item, &config))?;
                    }
                    for item in gpu_items {
                        process_work_item(&item, &config)?;
                    }
                    Ok(())
                }
                _ => {
                    for item in &items {
                        process_work_item(item, &config)?;
                    }
                    Ok(())
                }
            }
        } else {
            for item in &items {
                process_work_item(item, &config)?;
            }
            Ok(())
        }
    });

    result.map_err(PyRuntimeError::new_err)
}
