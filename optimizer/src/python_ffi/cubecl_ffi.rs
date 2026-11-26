use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use luminav_core::DTypeMode;

use super::extract::{extract_tensor_info, extract_tensor_info_mut};
use super::ra_ffi::{check_disjoint_bytes, check_non_null};

#[pyfunction]
pub fn is_cubecl_available_pyo3() -> bool {
    luminav_dispatch::detect_hardware().cubecl_available
}

#[cfg(feature = "cubecl")]
fn with_cubecl_engine<T, F>(f: F) -> Result<T, String>
where
    F: FnOnce(
        &luminav_dispatch::luminav_cubecl::CubeClLuminaEngine,
    ) -> Result<T, luminav_dispatch::luminav_cubecl::CubeClError>,
{
    let guard = luminav_dispatch::get_global_dispatcher()
        .lock()
        .map_err(|_| "Global dispatcher mutex poisoned".to_string())?;
    let eng = guard
        .cubecl_engine
        .as_ref()
        .ok_or_else(|| "CubeCL GPU backend is unavailable on this host.".to_string())?;
    f(eng).map_err(|e| e.to_string())
}

#[pyfunction]
#[pyo3(signature = (input, output, weight, bias, eps, cols=None))]
pub fn ra_norm_cubecl_f32_pyo3<'py>(
    py: Python<'py>,
    input: Bound<'py, PyAny>,
    output: Bound<'py, PyAny>,
    weight: Option<Bound<'py, PyAny>>,
    bias: Option<Bound<'py, PyAny>>,
    eps: f32,
    cols: Option<usize>,
) -> PyResult<()> {
    let (in_ptr, in_len, in_dtype) = extract_tensor_info(&input, "input")?;
    let (out_ptr, out_len, out_dtype) = extract_tensor_info_mut(&output, "output")?;
    if in_dtype != DTypeMode::Float32 || out_dtype != DTypeMode::Float32 {
        return Err(PyValueError::new_err(
            "ra_norm_cubecl_f32_pyo3 requires Float32 tensors.",
        ));
    }
    if in_len != out_len {
        return Err(PyValueError::new_err(
            "Input and output tensor lengths must match.",
        ));
    }
    if in_len == 0 {
        return Ok(());
    }

    let w_info = match weight {
        Some(w) => {
            let (ptr, len, dt) = extract_tensor_info(&w, "weight")?;
            if dt != DTypeMode::Float32 {
                return Err(PyValueError::new_err("Weight tensor must be Float32."));
            }
            Some((ptr, len))
        }
        None => None,
    };
    let b_info = match bias {
        Some(b) => {
            let (ptr, len, dt) = extract_tensor_info(&b, "bias")?;
            if dt != DTypeMode::Float32 {
                return Err(PyValueError::new_err("Bias tensor must be Float32."));
            }
            Some((ptr, len))
        }
        None => None,
    };
    let eff_cols = cols
        .or_else(|| w_info.map(|(_, len)| len))
        .or_else(|| b_info.map(|(_, len)| len))
        .unwrap_or(in_len);
    if eff_cols == 0 || !in_len.is_multiple_of(eff_cols) {
        return Err(PyValueError::new_err(
            "Input length must be a positive multiple of cols.",
        ));
    }
    if let Some((_, w_len)) = w_info
        && w_len != eff_cols
    {
        return Err(PyValueError::new_err("Weight length must match cols."));
    }
    if let Some((_, b_len)) = b_info
        && b_len != eff_cols
    {
        return Err(PyValueError::new_err("Bias length must match cols."));
    }
    let out_bytes = out_len.saturating_mul(std::mem::size_of::<f32>());
    check_non_null(in_ptr, in_len, "input")?;
    check_non_null(out_ptr, out_len, "output")?;
    check_disjoint_bytes(in_ptr, out_bytes, "input", out_ptr, out_bytes, "output")?;
    if let Some((w_ptr, w_len)) = w_info {
        check_non_null(w_ptr, w_len, "weight")?;
        check_disjoint_bytes(
            w_ptr,
            w_len.saturating_mul(std::mem::size_of::<f32>()),
            "weight",
            out_ptr,
            out_bytes,
            "output",
        )?;
    }
    if let Some((b_ptr, b_len)) = b_info {
        check_non_null(b_ptr, b_len, "bias")?;
        check_disjoint_bytes(
            b_ptr,
            b_len.saturating_mul(std::mem::size_of::<f32>()),
            "bias",
            out_ptr,
            out_bytes,
            "output",
        )?;
    }

    #[cfg(feature = "cubecl")]
    {
        let res = py.detach(|| {
            let in_slice = unsafe { std::slice::from_raw_parts(in_ptr as *const f32, in_len) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len) };
            let w_slice = w_info
                .map(|(ptr, len)| unsafe { std::slice::from_raw_parts(ptr as *const f32, len) });
            let b_slice = b_info
                .map(|(ptr, len)| unsafe { std::slice::from_raw_parts(ptr as *const f32, len) });
            let out_vec = with_cubecl_engine(|eng| {
                eng.ra_norm_f32(in_slice, w_slice, b_slice, eff_cols, eps)
            })?;
            out_slice.copy_from_slice(&out_vec);
            Ok::<(), String>(())
        });
        res.map_err(PyRuntimeError::new_err)
    }
    #[cfg(not(feature = "cubecl"))]
    {
        let _ = (py, in_ptr, out_ptr, eps);
        Err(PyRuntimeError::new_err(
            "CubeCL backend was not enabled in this build.",
        ))
    }
}

#[pyfunction]
pub fn ra_tanh_cubecl_f32_pyo3<'py>(
    py: Python<'py>,
    input: Bound<'py, PyAny>,
    output: Bound<'py, PyAny>,
    b: f32,
) -> PyResult<()> {
    let (in_ptr, in_len, in_dtype) = extract_tensor_info(&input, "input")?;
    let (out_ptr, out_len, out_dtype) = extract_tensor_info_mut(&output, "output")?;
    if in_dtype != DTypeMode::Float32 || out_dtype != DTypeMode::Float32 {
        return Err(PyValueError::new_err(
            "ra_tanh_cubecl_f32_pyo3 requires Float32 tensors.",
        ));
    }
    if in_len != out_len {
        return Err(PyValueError::new_err(
            "Input and output tensor lengths must match.",
        ));
    }
    if in_len == 0 {
        return Ok(());
    }
    let bytes = in_len.saturating_mul(std::mem::size_of::<f32>());
    check_non_null(in_ptr, in_len, "input")?;
    check_non_null(out_ptr, out_len, "output")?;
    check_disjoint_bytes(in_ptr, bytes, "input", out_ptr, bytes, "output")?;

    #[cfg(feature = "cubecl")]
    {
        let res = py.detach(|| {
            let in_slice = unsafe { std::slice::from_raw_parts(in_ptr as *const f32, in_len) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len) };
            let out_vec = with_cubecl_engine(|eng| eng.ra_tanh_f32(in_slice, b))?;
            out_slice.copy_from_slice(&out_vec);
            Ok::<(), String>(())
        });
        res.map_err(PyRuntimeError::new_err)
    }
    #[cfg(not(feature = "cubecl"))]
    {
        let _ = (py, in_ptr, out_ptr, b);
        Err(PyRuntimeError::new_err(
            "CubeCL backend was not enabled in this build.",
        ))
    }
}

fn run_ra_gated_cubecl_f32_pyo3<'py>(
    py: Python<'py>,
    gate: Bound<'py, PyAny>,
    up: Bound<'py, PyAny>,
    output: Bound<'py, PyAny>,
    b: f32,
    fn_name: &str,
    is_swiglu: bool,
) -> PyResult<()> {
    let (g_ptr, g_len, g_dtype) = extract_tensor_info(&gate, "gate")?;
    let (u_ptr, u_len, u_dtype) = extract_tensor_info(&up, "up")?;
    let (out_ptr, out_len, out_dtype) = extract_tensor_info_mut(&output, "output")?;
    if g_dtype != DTypeMode::Float32
        || u_dtype != DTypeMode::Float32
        || out_dtype != DTypeMode::Float32
    {
        return Err(PyValueError::new_err(format!(
            "{fn_name} requires Float32 tensors."
        )));
    }
    if g_len != u_len || g_len != out_len {
        return Err(PyValueError::new_err(
            "Gate, up, and output tensor lengths must match.",
        ));
    }
    if g_len == 0 {
        return Ok(());
    }
    let bytes = g_len.saturating_mul(std::mem::size_of::<f32>());
    check_non_null(g_ptr, g_len, "gate")?;
    check_non_null(u_ptr, u_len, "up")?;
    check_non_null(out_ptr, out_len, "output")?;
    check_disjoint_bytes(g_ptr, bytes, "gate", out_ptr, bytes, "output")?;
    check_disjoint_bytes(u_ptr, bytes, "up", out_ptr, bytes, "output")?;

    #[cfg(feature = "cubecl")]
    {
        let res = py.detach(|| {
            let g_slice = unsafe { std::slice::from_raw_parts(g_ptr as *const f32, g_len) };
            let u_slice = unsafe { std::slice::from_raw_parts(u_ptr as *const f32, u_len) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len) };
            let out_vec = with_cubecl_engine(|eng| {
                if is_swiglu {
                    eng.ra_swiglu_f32(g_slice, u_slice, b)
                } else {
                    eng.ra_glu_f32(g_slice, u_slice, b)
                }
            })?;
            out_slice.copy_from_slice(&out_vec);
            Ok::<(), String>(())
        });
        res.map_err(PyRuntimeError::new_err)
    }
    #[cfg(not(feature = "cubecl"))]
    {
        let _ = (py, g_ptr, u_ptr, out_ptr, b, is_swiglu);
        Err(PyRuntimeError::new_err(
            "CubeCL backend was not enabled in this build.",
        ))
    }
}

#[pyfunction]
pub fn ra_swiglu_cubecl_f32_pyo3<'py>(
    py: Python<'py>,
    gate: Bound<'py, PyAny>,
    up: Bound<'py, PyAny>,
    output: Bound<'py, PyAny>,
    b: f32,
) -> PyResult<()> {
    run_ra_gated_cubecl_f32_pyo3(py, gate, up, output, b, "ra_swiglu_cubecl_f32_pyo3", true)
}

#[pyfunction]
pub fn ra_glu_cubecl_f32_pyo3<'py>(
    py: Python<'py>,
    gate: Bound<'py, PyAny>,
    up: Bound<'py, PyAny>,
    output: Bound<'py, PyAny>,
    b: f32,
) -> PyResult<()> {
    run_ra_gated_cubecl_f32_pyo3(py, gate, up, output, b, "ra_glu_cubecl_f32_pyo3", false)
}

#[pyfunction]
pub fn ra_softmax_cubecl_f32_pyo3<'py>(
    py: Python<'py>,
    scores: Bound<'py, PyAny>,
    output: Bound<'py, PyAny>,
    cols: usize,
    scale: f32,
) -> PyResult<()> {
    let (s_ptr, s_len, s_dtype) = extract_tensor_info(&scores, "scores")?;
    let (out_ptr, out_len, out_dtype) = extract_tensor_info_mut(&output, "output")?;
    if s_dtype != DTypeMode::Float32 || out_dtype != DTypeMode::Float32 {
        return Err(PyValueError::new_err(
            "ra_softmax_cubecl_f32_pyo3 requires Float32 tensors.",
        ));
    }
    if s_len != out_len {
        return Err(PyValueError::new_err(
            "Scores and output tensor lengths must match.",
        ));
    }
    if s_len == 0 {
        return Ok(());
    }
    if cols == 0 || !s_len.is_multiple_of(cols) {
        return Err(PyValueError::new_err(
            "Scores length must be a positive multiple of cols.",
        ));
    }
    let bytes = s_len.saturating_mul(std::mem::size_of::<f32>());
    check_non_null(s_ptr, s_len, "scores")?;
    check_non_null(out_ptr, out_len, "output")?;
    check_disjoint_bytes(s_ptr, bytes, "scores", out_ptr, bytes, "output")?;

    #[cfg(feature = "cubecl")]
    {
        let res = py.detach(|| {
            let s_slice = unsafe { std::slice::from_raw_parts(s_ptr as *const f32, s_len) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len) };
            let out_vec = with_cubecl_engine(|eng| eng.ra_softmax_f32(s_slice, cols, scale))?;
            out_slice.copy_from_slice(&out_vec);
            Ok::<(), String>(())
        });
        res.map_err(PyRuntimeError::new_err)
    }
    #[cfg(not(feature = "cubecl"))]
    {
        let _ = (py, s_ptr, out_ptr, scale);
        Err(PyRuntimeError::new_err(
            "CubeCL backend was not enabled in this build.",
        ))
    }
}
