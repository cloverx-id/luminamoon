use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use luminav_core::DTypeMode;
use luminav_simd::{pos_ra_glu_slice, ra_norm_slice, ra_softmax_row, ra_tanh_slice};

use super::extract::extract_tensor_info;

#[pyfunction]
pub fn ra_norm_f32_pyo3<'py>(
    _py: Python<'py>,
    input: Bound<'py, PyAny>,
    output: Bound<'py, PyAny>,
    weight: Option<Bound<'py, PyAny>>,
    bias: Option<Bound<'py, PyAny>>,
    eps: f32,
) -> PyResult<()> {
    let (in_ptr, in_len, in_dtype) = extract_tensor_info(&input, "input")?;
    let (out_ptr, out_len, out_dtype) = extract_tensor_info(&output, "output")?;

    if in_dtype != DTypeMode::Float32 || out_dtype != DTypeMode::Float32 {
        return Err(PyValueError::new_err(
            "ra_norm_f32_pyo3 currently requires Float32 tensors.",
        ));
    }
    if in_len != out_len {
        return Err(PyValueError::new_err(
            "Input and output tensor lengths must match.",
        ));
    }

    let in_slice = unsafe { std::slice::from_raw_parts(in_ptr as *const f32, in_len) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len) };

    let weight_vec;
    let weight_slice = if let Some(w) = weight {
        let (w_ptr, w_len, _) = extract_tensor_info(&w, "weight")?;
        if w_len != in_len {
            return Err(PyValueError::new_err(
                "Weight length must match input length.",
            ));
        }
        weight_vec = unsafe { std::slice::from_raw_parts(w_ptr as *const f32, w_len) };
        Some(weight_vec)
    } else {
        None
    };

    let bias_vec;
    let bias_slice = if let Some(b) = bias {
        let (b_ptr, b_len, _) = extract_tensor_info(&b, "bias")?;
        if b_len != in_len {
            return Err(PyValueError::new_err(
                "Bias length must match input length.",
            ));
        }
        bias_vec = unsafe { std::slice::from_raw_parts(b_ptr as *const f32, b_len) };
        Some(bias_vec)
    } else {
        None
    };

    ra_norm_slice(in_slice, out_slice, weight_slice, bias_slice, eps);
    Ok(())
}

#[pyfunction]
pub fn ra_tanh_f32_pyo3<'py>(
    _py: Python<'py>,
    input: Bound<'py, PyAny>,
    output: Bound<'py, PyAny>,
    b: f32,
) -> PyResult<()> {
    let (in_ptr, in_len, in_dtype) = extract_tensor_info(&input, "input")?;
    let (out_ptr, out_len, out_dtype) = extract_tensor_info(&output, "output")?;

    if in_dtype != DTypeMode::Float32 || out_dtype != DTypeMode::Float32 {
        return Err(PyValueError::new_err(
            "ra_tanh_f32_pyo3 currently requires Float32 tensors.",
        ));
    }
    if in_len != out_len {
        return Err(PyValueError::new_err(
            "Input and output tensor lengths must match.",
        ));
    }

    let in_slice = unsafe { std::slice::from_raw_parts(in_ptr as *const f32, in_len) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len) };

    ra_tanh_slice(in_slice, out_slice, b);
    Ok(())
}

#[pyfunction]
pub fn pos_ra_glu_f32_pyo3<'py>(
    _py: Python<'py>,
    gate: Bound<'py, PyAny>,
    up: Bound<'py, PyAny>,
    output: Bound<'py, PyAny>,
    b: f32,
) -> PyResult<()> {
    let (g_ptr, g_len, g_dtype) = extract_tensor_info(&gate, "gate")?;
    let (u_ptr, u_len, u_dtype) = extract_tensor_info(&up, "up")?;
    let (out_ptr, out_len, out_dtype) = extract_tensor_info(&output, "output")?;

    if g_dtype != DTypeMode::Float32 || u_dtype != DTypeMode::Float32 || out_dtype != DTypeMode::Float32 {
        return Err(PyValueError::new_err(
            "pos_ra_glu_f32_pyo3 requires Float32 tensors.",
        ));
    }
    if g_len != u_len || g_len != out_len {
        return Err(PyValueError::new_err(
            "Gate, up, and output tensor lengths must match.",
        ));
    }

    let g_slice = unsafe { std::slice::from_raw_parts(g_ptr as *const f32, g_len) };
    let u_slice = unsafe { std::slice::from_raw_parts(u_ptr as *const f32, u_len) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len) };

    pos_ra_glu_slice(g_slice, u_slice, out_slice, b);
    Ok(())
}

#[pyfunction]
pub fn ra_softmax_f32_pyo3<'py>(
    _py: Python<'py>,
    scores: Bound<'py, PyAny>,
    output: Bound<'py, PyAny>,
    rows: usize,
    cols: usize,
    scale: f32,
) -> PyResult<()> {
    let (s_ptr, s_len, s_dtype) = extract_tensor_info(&scores, "scores")?;
    let (out_ptr, out_len, out_dtype) = extract_tensor_info(&output, "output")?;

    if s_dtype != DTypeMode::Float32 || out_dtype != DTypeMode::Float32 {
        return Err(PyValueError::new_err(
            "ra_softmax_f32_pyo3 requires Float32 tensors.",
        ));
    }
    if s_len != rows * cols || out_len != rows * cols {
        return Err(PyValueError::new_err(
            "Tensor dimensions do not match rows * cols.",
        ));
    }

    let s_slice = unsafe { std::slice::from_raw_parts(s_ptr as *const f32, s_len) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len) };

    for r in 0..rows {
        let start = r * cols;
        let end = start + cols;
        ra_softmax_row(&s_slice[start..end], &mut out_slice[start..end], scale);
    }
    Ok(())
}
