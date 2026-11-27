use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use luminav_core::DTypeMode;
use luminav_simd::{
    ra_attn_scores_bwd_batch_f32, ra_attn_scores_bwd_inplace_batch_f32,
    ra_attn_weights_fwd_batch_f32, ra_attn_weights_fwd_inplace_batch_f32, ra_glu_slice,
    ra_norm_batch_f32, ra_softmax_batch_f32, ra_softmax_bwd_batch_f32, ra_swiglu_slice,
    ra_tanh_slice,
};

use super::extract::{
    extract_i64_tensor_info, extract_i64_tensor_info_mut, extract_tensor_info,
    extract_tensor_info_mut,
};

pub(crate) fn ranges_overlap(a_ptr: usize, a_bytes: usize, b_ptr: usize, b_bytes: usize) -> bool {
    if a_bytes == 0 || b_bytes == 0 {
        return false;
    }
    let a_end = a_ptr.saturating_add(a_bytes);
    let b_end = b_ptr.saturating_add(b_bytes);
    a_ptr < b_end && b_ptr < a_end
}

pub(crate) fn check_non_null(ptr: usize, len: usize, name: &str) -> PyResult<()> {
    if len > 0 && ptr == 0 {
        return Err(PyValueError::new_err(format!(
            "Tensor '{name}' has a null data pointer for non-empty length {len}."
        )));
    }
    Ok(())
}

pub(crate) fn check_disjoint_bytes(
    a_ptr: usize,
    a_bytes: usize,
    a_name: &str,
    b_ptr: usize,
    b_bytes: usize,
    b_name: &str,
) -> PyResult<()> {
    if ranges_overlap(a_ptr, a_bytes, b_ptr, b_bytes) {
        return Err(PyValueError::new_err(format!(
            "Aliased memory detected between '{a_name}' and '{b_name}'."
        )));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (input, output, weight, bias, eps, cols=None))]
pub fn ra_norm_f32_pyo3<'py>(
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
            "ra_norm_f32_pyo3 currently requires Float32 tensors.",
        ));
    }
    if in_len != out_len {
        return Err(PyValueError::new_err(
            "Input and output tensor lengths must match.",
        ));
    }

    let w_info = if let Some(w) = weight {
        let (w_ptr, w_len, w_dtype) = extract_tensor_info(&w, "weight")?;
        if w_dtype != DTypeMode::Float32 {
            return Err(PyValueError::new_err(
                "ra_norm_f32_pyo3 requires Float32 weight tensor.",
            ));
        }
        Some((w_ptr, w_len))
    } else {
        None
    };

    let b_info = if let Some(b) = bias {
        let (b_ptr, b_len, b_dtype) = extract_tensor_info(&b, "bias")?;
        if b_dtype != DTypeMode::Float32 {
            return Err(PyValueError::new_err(
                "ra_norm_f32_pyo3 requires Float32 bias tensor.",
            ));
        }
        Some((b_ptr, b_len))
    } else {
        None
    };

    if in_len == 0 {
        return Ok(());
    }

    let eff_cols = cols
        .or_else(|| w_info.map(|(_, len)| len))
        .or_else(|| b_info.map(|(_, len)| len))
        .unwrap_or(in_len);

    if eff_cols == 0 || !in_len.is_multiple_of(eff_cols) {
        return Err(PyValueError::new_err(
            "Input length must be a positive multiple of normalized dimension cols.",
        ));
    }
    if let Some((_, w_len)) = w_info
        && w_len != eff_cols
    {
        return Err(PyValueError::new_err(
            "Weight length must match normalized dimension cols.",
        ));
    }
    if let Some((_, b_len)) = b_info
        && b_len != eff_cols
    {
        return Err(PyValueError::new_err(
            "Bias length must match normalized dimension cols.",
        ));
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

    let rows = in_len / eff_cols;
    py.detach(|| {
        let in_slice = unsafe { std::slice::from_raw_parts(in_ptr as *const f32, in_len) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len) };
        let weight_slice =
            w_info.map(|(ptr, len)| unsafe { std::slice::from_raw_parts(ptr as *const f32, len) });
        let bias_slice =
            b_info.map(|(ptr, len)| unsafe { std::slice::from_raw_parts(ptr as *const f32, len) });
        ra_norm_batch_f32(
            in_slice,
            out_slice,
            weight_slice,
            bias_slice,
            rows,
            eff_cols,
            eps,
        );
    });
    Ok(())
}

#[pyfunction]
pub fn ra_tanh_f32_pyo3<'py>(
    py: Python<'py>,
    input: Bound<'py, PyAny>,
    output: Bound<'py, PyAny>,
    b: f32,
) -> PyResult<()> {
    let (in_ptr, in_len, in_dtype) = extract_tensor_info(&input, "input")?;
    let (out_ptr, out_len, out_dtype) = extract_tensor_info_mut(&output, "output")?;

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
    if in_len == 0 {
        return Ok(());
    }
    let bytes = in_len.saturating_mul(std::mem::size_of::<f32>());
    check_non_null(in_ptr, in_len, "input")?;
    check_non_null(out_ptr, out_len, "output")?;
    check_disjoint_bytes(in_ptr, bytes, "input", out_ptr, bytes, "output")?;

    py.detach(|| {
        let in_slice = unsafe { std::slice::from_raw_parts(in_ptr as *const f32, in_len) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len) };
        ra_tanh_slice(in_slice, out_slice, b);
    });
    Ok(())
}

fn run_ra_gated_f32_pyo3<'py>(
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

    py.detach(|| {
        let g_slice = unsafe { std::slice::from_raw_parts(g_ptr as *const f32, g_len) };
        let u_slice = unsafe { std::slice::from_raw_parts(u_ptr as *const f32, u_len) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len) };

        if is_swiglu {
            ra_swiglu_slice(g_slice, u_slice, out_slice, b);
        } else {
            ra_glu_slice(g_slice, u_slice, out_slice, b);
        }
    });
    Ok(())
}

#[pyfunction]
pub fn ra_swiglu_f32_pyo3<'py>(
    py: Python<'py>,
    gate: Bound<'py, PyAny>,
    up: Bound<'py, PyAny>,
    output: Bound<'py, PyAny>,
    b: f32,
) -> PyResult<()> {
    run_ra_gated_f32_pyo3(py, gate, up, output, b, "ra_swiglu_f32_pyo3", true)
}

#[pyfunction]
pub fn ra_glu_f32_pyo3<'py>(
    py: Python<'py>,
    gate: Bound<'py, PyAny>,
    up: Bound<'py, PyAny>,
    output: Bound<'py, PyAny>,
    b: f32,
) -> PyResult<()> {
    run_ra_gated_f32_pyo3(py, gate, up, output, b, "ra_glu_f32_pyo3", false)
}

#[pyfunction]
pub fn ra_softmax_f32_pyo3<'py>(
    py: Python<'py>,
    scores: Bound<'py, PyAny>,
    output: Bound<'py, PyAny>,
    rows: usize,
    cols: usize,
    scale: f32,
) -> PyResult<()> {
    let (s_ptr, s_len, s_dtype) = extract_tensor_info(&scores, "scores")?;
    let (out_ptr, out_len, out_dtype) = extract_tensor_info_mut(&output, "output")?;

    if s_dtype != DTypeMode::Float32 || out_dtype != DTypeMode::Float32 {
        return Err(PyValueError::new_err(
            "ra_softmax_f32_pyo3 requires Float32 tensors.",
        ));
    }
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows * cols overflowed usize."))?;
    if s_len != expected || out_len != expected {
        return Err(PyValueError::new_err(
            "Tensor dimensions do not match rows * cols.",
        ));
    }
    if expected == 0 {
        return Ok(());
    }
    let bytes = expected.saturating_mul(std::mem::size_of::<f32>());
    check_non_null(s_ptr, s_len, "scores")?;
    check_non_null(out_ptr, out_len, "output")?;
    check_disjoint_bytes(s_ptr, bytes, "scores", out_ptr, bytes, "output")?;

    py.detach(|| {
        let s_slice = unsafe { std::slice::from_raw_parts(s_ptr as *const f32, s_len) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len) };
        ra_softmax_batch_f32(s_slice, out_slice, rows, cols, scale);
    });
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ra_softmax_bwd_f32_pyo3<'py>(
    py: Python<'py>,
    scores: Bound<'py, PyAny>,
    p_in: Bound<'py, PyAny>,
    grad_out: Bound<'py, PyAny>,
    out_ds: Bound<'py, PyAny>,
    rows: usize,
    cols: usize,
    scale: f32,
) -> PyResult<()> {
    let (s_ptr, s_len, s_dt) = extract_tensor_info(&scores, "scores")?;
    let (p_ptr, p_len, p_dt) = extract_tensor_info(&p_in, "p_in")?;
    let (g_ptr, g_len, g_dt) = extract_tensor_info(&grad_out, "grad_out")?;
    let (ds_ptr, ds_len, ds_dt) = extract_tensor_info_mut(&out_ds, "out_ds")?;
    if s_dt != DTypeMode::Float32
        || p_dt != DTypeMode::Float32
        || g_dt != DTypeMode::Float32
        || ds_dt != DTypeMode::Float32
    {
        return Err(PyValueError::new_err(
            "ra_softmax_bwd_f32_pyo3 requires Float32 tensors.",
        ));
    }
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows * cols overflowed usize."))?;
    if s_len != expected || p_len != expected || g_len != expected || ds_len != expected {
        return Err(PyValueError::new_err(
            "Tensor dimensions do not match rows * cols.",
        ));
    }
    if expected == 0 {
        return Ok(());
    }
    let bytes = expected.saturating_mul(std::mem::size_of::<f32>());
    check_non_null(s_ptr, s_len, "scores")?;
    check_non_null(p_ptr, p_len, "p_in")?;
    check_non_null(g_ptr, g_len, "grad_out")?;
    check_non_null(ds_ptr, ds_len, "out_ds")?;
    check_disjoint_bytes(s_ptr, bytes, "scores", ds_ptr, bytes, "out_ds")?;
    check_disjoint_bytes(p_ptr, bytes, "p_in", ds_ptr, bytes, "out_ds")?;
    check_disjoint_bytes(g_ptr, bytes, "grad_out", ds_ptr, bytes, "out_ds")?;
    py.detach(|| {
        let s_slice = unsafe { std::slice::from_raw_parts(s_ptr as *const f32, s_len) };
        let p_slice = unsafe { std::slice::from_raw_parts(p_ptr as *const f32, p_len) };
        let g_slice = unsafe { std::slice::from_raw_parts(g_ptr as *const f32, g_len) };
        let ds_slice = unsafe { std::slice::from_raw_parts_mut(ds_ptr as *mut f32, ds_len) };
        ra_softmax_bwd_batch_f32(s_slice, p_slice, g_slice, ds_slice, rows, cols, scale);
    });
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ra_attn_weights_fwd_f32_pyo3<'py>(
    py: Python<'py>,
    scores: Bound<'py, PyAny>,
    out_p: Bound<'py, PyAny>,
    out_m: Bound<'py, PyAny>,
    out_l: Bound<'py, PyAny>,
    out_m_idx: Bound<'py, PyAny>,
    rows: usize,
    cols: usize,
    n_q: usize,
    q_start: usize,
    is_causal: bool,
    scale: f32,
) -> PyResult<()> {
    let (s_ptr, s_len, s_dt) = extract_tensor_info(&scores, "scores")?;
    let (p_ptr, p_len, p_dt) = extract_tensor_info_mut(&out_p, "out_p")?;
    let (m_ptr, m_len, m_dt) = extract_tensor_info_mut(&out_m, "out_m")?;
    let (l_ptr, l_len, l_dt) = extract_tensor_info_mut(&out_l, "out_l")?;
    let (idx_ptr, idx_len) = extract_i64_tensor_info_mut(&out_m_idx, "out_m_idx")?;

    if s_dt != DTypeMode::Float32
        || p_dt != DTypeMode::Float32
        || m_dt != DTypeMode::Float32
        || l_dt != DTypeMode::Float32
    {
        return Err(PyValueError::new_err(
            "ra_attn_weights_fwd_f32_pyo3 requires Float32 tensors.",
        ));
    }
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows * cols overflowed usize."))?;
    if s_len != expected || p_len != expected || m_len != rows || l_len != rows || idx_len != rows {
        return Err(PyValueError::new_err(
            "Tensor dimensions mismatch in ra_attn_weights_fwd_f32_pyo3.",
        ));
    }
    if expected == 0 {
        return Ok(());
    }
    let mat_bytes = expected.saturating_mul(std::mem::size_of::<f32>());
    let row_f32_bytes = rows.saturating_mul(std::mem::size_of::<f32>());
    let row_i64_bytes = rows.saturating_mul(std::mem::size_of::<i64>());
    check_non_null(s_ptr, s_len, "scores")?;
    check_non_null(p_ptr, p_len, "out_p")?;
    check_non_null(m_ptr, m_len, "out_m")?;
    check_non_null(l_ptr, l_len, "out_l")?;
    check_non_null(idx_ptr, idx_len, "out_m_idx")?;
    check_disjoint_bytes(m_ptr, row_f32_bytes, "out_m", l_ptr, row_f32_bytes, "out_l")?;
    check_disjoint_bytes(
        m_ptr,
        row_f32_bytes,
        "out_m",
        idx_ptr,
        row_i64_bytes,
        "out_m_idx",
    )?;
    check_disjoint_bytes(
        l_ptr,
        row_f32_bytes,
        "out_l",
        idx_ptr,
        row_i64_bytes,
        "out_m_idx",
    )?;
    check_disjoint_bytes(p_ptr, mat_bytes, "out_p", m_ptr, row_f32_bytes, "out_m")?;
    check_disjoint_bytes(p_ptr, mat_bytes, "out_p", l_ptr, row_f32_bytes, "out_l")?;
    check_disjoint_bytes(
        p_ptr,
        mat_bytes,
        "out_p",
        idx_ptr,
        row_i64_bytes,
        "out_m_idx",
    )?;
    if s_ptr != p_ptr {
        check_disjoint_bytes(s_ptr, mat_bytes, "scores", p_ptr, mat_bytes, "out_p")?;
        check_disjoint_bytes(s_ptr, mat_bytes, "scores", m_ptr, row_f32_bytes, "out_m")?;
        check_disjoint_bytes(s_ptr, mat_bytes, "scores", l_ptr, row_f32_bytes, "out_l")?;
        check_disjoint_bytes(
            s_ptr,
            mat_bytes,
            "scores",
            idx_ptr,
            row_i64_bytes,
            "out_m_idx",
        )?;
    }

    py.detach(|| {
        let m_slice = unsafe { std::slice::from_raw_parts_mut(m_ptr as *mut f32, m_len) };
        let l_slice = unsafe { std::slice::from_raw_parts_mut(l_ptr as *mut f32, l_len) };
        let idx_slice = unsafe { std::slice::from_raw_parts_mut(idx_ptr as *mut i64, idx_len) };
        if s_ptr == p_ptr {
            let buf = unsafe { std::slice::from_raw_parts_mut(p_ptr as *mut f32, p_len) };
            ra_attn_weights_fwd_inplace_batch_f32(
                buf, m_slice, l_slice, idx_slice, rows, cols, n_q, q_start, is_causal, scale,
            );
        } else {
            let s_slice = unsafe { std::slice::from_raw_parts(s_ptr as *const f32, s_len) };
            let p_slice = unsafe { std::slice::from_raw_parts_mut(p_ptr as *mut f32, p_len) };
            ra_attn_weights_fwd_batch_f32(
                s_slice, p_slice, m_slice, l_slice, idx_slice, rows, cols, n_q, q_start, is_causal,
                scale,
            );
        }
    });
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ra_attn_scores_bwd_f32_pyo3<'py>(
    py: Python<'py>,
    scores: Bound<'py, PyAny>,
    dp: Bound<'py, PyAny>,
    m_vals: Bound<'py, PyAny>,
    l_vals: Bound<'py, PyAny>,
    m_idxs: Bound<'py, PyAny>,
    d_vals: Bound<'py, PyAny>,
    out_p: Bound<'py, PyAny>,
    out_ds: Bound<'py, PyAny>,
    rows: usize,
    cols: usize,
    n_q: usize,
    q_start: usize,
    is_causal: bool,
    scale: f32,
) -> PyResult<()> {
    let (s_ptr, s_len, s_dt) = extract_tensor_info(&scores, "scores")?;
    let (dp_ptr, dp_len, dp_dt) = extract_tensor_info(&dp, "dp")?;
    let (m_ptr, m_len, m_dt) = extract_tensor_info(&m_vals, "m_vals")?;
    let (l_ptr, l_len, l_dt) = extract_tensor_info(&l_vals, "l_vals")?;
    let (idx_ptr, idx_len) = extract_i64_tensor_info(&m_idxs, "m_idxs")?;
    let (d_ptr, d_len, d_dt) = extract_tensor_info(&d_vals, "d_vals")?;
    let (p_ptr, p_len, p_dt) = extract_tensor_info_mut(&out_p, "out_p")?;
    let (ds_ptr, ds_len, ds_dt) = extract_tensor_info_mut(&out_ds, "out_ds")?;

    if s_dt != DTypeMode::Float32
        || dp_dt != DTypeMode::Float32
        || m_dt != DTypeMode::Float32
        || l_dt != DTypeMode::Float32
        || d_dt != DTypeMode::Float32
        || p_dt != DTypeMode::Float32
        || ds_dt != DTypeMode::Float32
    {
        return Err(PyValueError::new_err(
            "ra_attn_scores_bwd_f32_pyo3 requires Float32 tensors.",
        ));
    }
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows * cols overflowed usize."))?;
    if s_len != expected
        || dp_len != expected
        || p_len != expected
        || ds_len != expected
        || m_len != rows
        || l_len != rows
        || idx_len != rows
        || d_len != rows
    {
        return Err(PyValueError::new_err(
            "Tensor dimensions mismatch in ra_attn_scores_bwd_f32_pyo3.",
        ));
    }
    if expected == 0 {
        return Ok(());
    }
    let mat_bytes = expected.saturating_mul(std::mem::size_of::<f32>());
    let row_f32_bytes = rows.saturating_mul(std::mem::size_of::<f32>());
    let row_i64_bytes = rows.saturating_mul(std::mem::size_of::<i64>());
    check_non_null(s_ptr, s_len, "scores")?;
    check_non_null(dp_ptr, dp_len, "dp")?;
    check_non_null(m_ptr, m_len, "m_vals")?;
    check_non_null(l_ptr, l_len, "l_vals")?;
    check_non_null(idx_ptr, idx_len, "m_idxs")?;
    check_non_null(d_ptr, d_len, "d_vals")?;
    check_non_null(p_ptr, p_len, "out_p")?;
    check_non_null(ds_ptr, ds_len, "out_ds")?;
    check_disjoint_bytes(p_ptr, mat_bytes, "out_p", ds_ptr, mat_bytes, "out_ds")?;
    for (r_ptr, r_bytes, r_name) in [
        (m_ptr, row_f32_bytes, "m_vals"),
        (l_ptr, row_f32_bytes, "l_vals"),
        (idx_ptr, row_i64_bytes, "m_idxs"),
        (d_ptr, row_f32_bytes, "d_vals"),
    ] {
        check_disjoint_bytes(r_ptr, r_bytes, r_name, p_ptr, mat_bytes, "out_p")?;
        check_disjoint_bytes(r_ptr, r_bytes, r_name, ds_ptr, mat_bytes, "out_ds")?;
    }
    let is_inplace =
        s_ptr == p_ptr && dp_ptr == ds_ptr && !ranges_overlap(s_ptr, mat_bytes, dp_ptr, mat_bytes);
    if !is_inplace {
        check_disjoint_bytes(s_ptr, mat_bytes, "scores", p_ptr, mat_bytes, "out_p")?;
        check_disjoint_bytes(s_ptr, mat_bytes, "scores", ds_ptr, mat_bytes, "out_ds")?;
        check_disjoint_bytes(dp_ptr, mat_bytes, "dp", p_ptr, mat_bytes, "out_p")?;
        check_disjoint_bytes(dp_ptr, mat_bytes, "dp", ds_ptr, mat_bytes, "out_ds")?;
    }

    py.detach(|| {
        let m_slice = unsafe { std::slice::from_raw_parts(m_ptr as *const f32, m_len) };
        let l_slice = unsafe { std::slice::from_raw_parts(l_ptr as *const f32, l_len) };
        let idx_slice = unsafe { std::slice::from_raw_parts(idx_ptr as *const i64, idx_len) };
        let d_slice = unsafe { std::slice::from_raw_parts(d_ptr as *const f32, d_len) };
        if is_inplace {
            let sp_slice = unsafe { std::slice::from_raw_parts_mut(p_ptr as *mut f32, p_len) };
            let dd_slice = unsafe { std::slice::from_raw_parts_mut(ds_ptr as *mut f32, ds_len) };
            ra_attn_scores_bwd_inplace_batch_f32(
                sp_slice, dd_slice, m_slice, l_slice, idx_slice, d_slice, rows, cols, n_q, q_start,
                is_causal, scale,
            );
        } else {
            let s_slice = unsafe { std::slice::from_raw_parts(s_ptr as *const f32, s_len) };
            let dp_slice = unsafe { std::slice::from_raw_parts(dp_ptr as *const f32, dp_len) };
            let p_slice = unsafe { std::slice::from_raw_parts_mut(p_ptr as *mut f32, p_len) };
            let ds_slice = unsafe { std::slice::from_raw_parts_mut(ds_ptr as *mut f32, ds_len) };
            ra_attn_scores_bwd_batch_f32(
                s_slice, dp_slice, m_slice, l_slice, idx_slice, d_slice, p_slice, ds_slice, rows,
                cols, n_q, q_start, is_causal, scale,
            );
        }
    });
    Ok(())
}
