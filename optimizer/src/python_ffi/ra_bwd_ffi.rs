use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use luminav_core::DTypeMode;
use luminav_simd::{
    qrope_batch_f32, ra_gated_packed_bwd_batch_f32, ra_gated_packed_fwd_batch_f32,
    ra_glu_bwd_slice, ra_norm_bwd_batch_f32, ra_swiglu_bwd_slice, ra_tanh_bwd_slice,
};

use super::extract::{extract_tensor_info, extract_tensor_info_mut};
use super::ra_ffi::{check_disjoint_bytes, check_non_null};

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ra_norm_bwd_f32_pyo3<'py>(
    py: Python<'py>,
    grad_y: Bound<'py, PyAny>,
    input: Bound<'py, PyAny>,
    weight: Option<Bound<'py, PyAny>>,
    dx: Bound<'py, PyAny>,
    dw: Option<Bound<'py, PyAny>>,
    db: Option<Bound<'py, PyAny>>,
    eps: f32,
    cols: usize,
) -> PyResult<()> {
    let (gy_ptr, gy_len, gy_dt) = extract_tensor_info(&grad_y, "grad_y")?;
    let (x_ptr, x_len, x_dt) = extract_tensor_info(&input, "input")?;
    let (dx_ptr, dx_len, dx_dt) = extract_tensor_info_mut(&dx, "dx")?;
    if gy_dt != DTypeMode::Float32 || x_dt != DTypeMode::Float32 || dx_dt != DTypeMode::Float32 {
        return Err(PyValueError::new_err(
            "ra_norm_bwd_f32_pyo3 requires Float32 tensors.",
        ));
    }
    if gy_len != x_len || dx_len != x_len {
        return Err(PyValueError::new_err(
            "grad_y, input, and dx lengths must match.",
        ));
    }
    if cols == 0 || !x_len.is_multiple_of(cols) {
        return Err(PyValueError::new_err(
            "input length must be a non-negative multiple of positive cols.",
        ));
    }

    let w_info = if let Some(ref w) = weight {
        let (ptr, len, dt) = extract_tensor_info(w, "weight")?;
        if dt != DTypeMode::Float32 || len != cols {
            return Err(PyValueError::new_err(
                "weight must be Float32 with length == cols.",
            ));
        }
        Some((ptr, len))
    } else {
        None
    };

    let dw_info = if let Some(ref dw_t) = dw {
        let (ptr, len, dt) = extract_tensor_info_mut(dw_t, "dw")?;
        if dt != DTypeMode::Float32 || len != cols {
            return Err(PyValueError::new_err(
                "dw must be Float32 with length == cols.",
            ));
        }
        Some((ptr, len))
    } else {
        None
    };

    let db_info = if let Some(ref db_t) = db {
        let (ptr, len, dt) = extract_tensor_info_mut(db_t, "db")?;
        if dt != DTypeMode::Float32 || len != cols {
            return Err(PyValueError::new_err(
                "db must be Float32 with length == cols.",
            ));
        }
        Some((ptr, len))
    } else {
        None
    };

    let mat_bytes = x_len.saturating_mul(std::mem::size_of::<f32>());
    let col_bytes = cols.saturating_mul(std::mem::size_of::<f32>());
    if let Some((w_ptr, w_len)) = w_info {
        check_non_null(w_ptr, w_len, "weight")?;
        if x_len > 0 {
            check_disjoint_bytes(w_ptr, col_bytes, "weight", dx_ptr, mat_bytes, "dx")?;
        }
    }
    if let Some((dw_ptr, dw_len)) = dw_info {
        check_non_null(dw_ptr, dw_len, "dw")?;
        if x_len > 0 {
            check_disjoint_bytes(dw_ptr, col_bytes, "dw", dx_ptr, mat_bytes, "dx")?;
            check_disjoint_bytes(dw_ptr, col_bytes, "dw", gy_ptr, mat_bytes, "grad_y")?;
            check_disjoint_bytes(dw_ptr, col_bytes, "dw", x_ptr, mat_bytes, "input")?;
        }
        if let Some((w_ptr, _)) = w_info {
            check_disjoint_bytes(dw_ptr, col_bytes, "dw", w_ptr, col_bytes, "weight")?;
        }
    }
    if let Some((db_ptr, db_len)) = db_info {
        check_non_null(db_ptr, db_len, "db")?;
        if x_len > 0 {
            check_disjoint_bytes(db_ptr, col_bytes, "db", dx_ptr, mat_bytes, "dx")?;
            check_disjoint_bytes(db_ptr, col_bytes, "db", gy_ptr, mat_bytes, "grad_y")?;
            check_disjoint_bytes(db_ptr, col_bytes, "db", x_ptr, mat_bytes, "input")?;
        }
        if let Some((w_ptr, _)) = w_info {
            check_disjoint_bytes(db_ptr, col_bytes, "db", w_ptr, col_bytes, "weight")?;
        }
        if let Some((dw_ptr, _)) = dw_info {
            check_disjoint_bytes(db_ptr, col_bytes, "db", dw_ptr, col_bytes, "dw")?;
        }
    }
    if x_len == 0 {
        if let Some((dw_ptr, dw_len)) = dw_info {
            unsafe { std::slice::from_raw_parts_mut(dw_ptr as *mut f32, dw_len) }.fill(0.0);
        }
        if let Some((db_ptr, db_len)) = db_info {
            unsafe { std::slice::from_raw_parts_mut(db_ptr as *mut f32, db_len) }.fill(0.0);
        }
        return Ok(());
    }
    check_non_null(gy_ptr, gy_len, "grad_y")?;
    check_non_null(x_ptr, x_len, "input")?;
    check_non_null(dx_ptr, dx_len, "dx")?;
    check_disjoint_bytes(gy_ptr, mat_bytes, "grad_y", dx_ptr, mat_bytes, "dx")?;
    check_disjoint_bytes(x_ptr, mat_bytes, "input", dx_ptr, mat_bytes, "dx")?;

    let rows = x_len / cols;
    py.detach(|| {
        let gy_s = unsafe { std::slice::from_raw_parts(gy_ptr as *const f32, gy_len) };
        let x_s = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, x_len) };
        let dx_s = unsafe { std::slice::from_raw_parts_mut(dx_ptr as *mut f32, dx_len) };
        let w_s = w_info.map(|(p, l)| unsafe { std::slice::from_raw_parts(p as *const f32, l) });
        let dw_s =
            dw_info.map(|(p, l)| unsafe { std::slice::from_raw_parts_mut(p as *mut f32, l) });
        let db_s =
            db_info.map(|(p, l)| unsafe { std::slice::from_raw_parts_mut(p as *mut f32, l) });
        ra_norm_bwd_batch_f32(gy_s, x_s, w_s, dx_s, dw_s, db_s, rows, cols, eps);
    });
    Ok(())
}

#[pyfunction]
pub fn ra_tanh_bwd_f32_pyo3<'py>(
    py: Python<'py>,
    grad_out: Bound<'py, PyAny>,
    input: Bound<'py, PyAny>,
    dx: Bound<'py, PyAny>,
    b: f32,
) -> PyResult<()> {
    let (g_ptr, g_len, g_dt) = extract_tensor_info(&grad_out, "grad_out")?;
    let (x_ptr, x_len, x_dt) = extract_tensor_info(&input, "input")?;
    let (dx_ptr, dx_len, dx_dt) = extract_tensor_info_mut(&dx, "dx")?;
    if g_dt != DTypeMode::Float32 || x_dt != DTypeMode::Float32 || dx_dt != DTypeMode::Float32 {
        return Err(PyValueError::new_err(
            "ra_tanh_bwd_f32_pyo3 requires Float32 tensors.",
        ));
    }
    if g_len != x_len || dx_len != x_len {
        return Err(PyValueError::new_err(
            "grad_out, input, and dx lengths must match.",
        ));
    }
    if x_len == 0 {
        return Ok(());
    }
    let bytes = x_len.saturating_mul(std::mem::size_of::<f32>());
    check_non_null(g_ptr, g_len, "grad_out")?;
    check_non_null(x_ptr, x_len, "input")?;
    check_non_null(dx_ptr, dx_len, "dx")?;
    check_disjoint_bytes(g_ptr, bytes, "grad_out", dx_ptr, bytes, "dx")?;
    check_disjoint_bytes(x_ptr, bytes, "input", dx_ptr, bytes, "dx")?;

    py.detach(|| {
        let g_s = unsafe { std::slice::from_raw_parts(g_ptr as *const f32, g_len) };
        let x_s = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, x_len) };
        let dx_s = unsafe { std::slice::from_raw_parts_mut(dx_ptr as *mut f32, dx_len) };
        ra_tanh_bwd_slice(g_s, x_s, dx_s, b);
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_ra_gated_bwd_f32_pyo3<'py>(
    py: Python<'py>,
    grad_out: Bound<'py, PyAny>,
    gate: Bound<'py, PyAny>,
    up: Bound<'py, PyAny>,
    dgate: Bound<'py, PyAny>,
    dup: Bound<'py, PyAny>,
    b: f32,
    fn_name: &str,
    is_swiglu: bool,
) -> PyResult<()> {
    let (g_ptr, g_len, g_dt) = extract_tensor_info(&grad_out, "grad_out")?;
    let (z_ptr, z_len, z_dt) = extract_tensor_info(&gate, "gate")?;
    let (u_ptr, u_len, u_dt) = extract_tensor_info(&up, "up")?;
    let (dg_ptr, dg_len, dg_dt) = extract_tensor_info_mut(&dgate, "dgate")?;
    let (du_ptr, du_len, du_dt) = extract_tensor_info_mut(&dup, "dup")?;
    if g_dt != DTypeMode::Float32
        || z_dt != DTypeMode::Float32
        || u_dt != DTypeMode::Float32
        || dg_dt != DTypeMode::Float32
        || du_dt != DTypeMode::Float32
    {
        return Err(PyValueError::new_err(format!(
            "{fn_name} requires Float32 tensors."
        )));
    }
    if g_len != z_len || u_len != z_len || dg_len != z_len || du_len != z_len {
        return Err(PyValueError::new_err(
            "All tensor lengths must match in gated backward.",
        ));
    }
    if z_len == 0 {
        return Ok(());
    }
    let bytes = z_len.saturating_mul(std::mem::size_of::<f32>());
    check_non_null(g_ptr, g_len, "grad_out")?;
    check_non_null(z_ptr, z_len, "gate")?;
    check_non_null(u_ptr, u_len, "up")?;
    check_non_null(dg_ptr, dg_len, "dgate")?;
    check_non_null(du_ptr, du_len, "dup")?;
    check_disjoint_bytes(dg_ptr, bytes, "dgate", du_ptr, bytes, "dup")?;
    for (p, n) in [(g_ptr, "grad_out"), (z_ptr, "gate"), (u_ptr, "up")] {
        check_disjoint_bytes(p, bytes, n, dg_ptr, bytes, "dgate")?;
        check_disjoint_bytes(p, bytes, n, du_ptr, bytes, "dup")?;
    }

    py.detach(|| {
        let g_s = unsafe { std::slice::from_raw_parts(g_ptr as *const f32, g_len) };
        let z_s = unsafe { std::slice::from_raw_parts(z_ptr as *const f32, z_len) };
        let u_s = unsafe { std::slice::from_raw_parts(u_ptr as *const f32, u_len) };
        let dg_s = unsafe { std::slice::from_raw_parts_mut(dg_ptr as *mut f32, dg_len) };
        let du_s = unsafe { std::slice::from_raw_parts_mut(du_ptr as *mut f32, du_len) };
        if is_swiglu {
            ra_swiglu_bwd_slice(g_s, z_s, u_s, dg_s, du_s, b);
        } else {
            ra_glu_bwd_slice(g_s, z_s, u_s, dg_s, du_s, b);
        }
    });
    Ok(())
}

#[pyfunction]
pub fn ra_swiglu_bwd_f32_pyo3<'py>(
    py: Python<'py>,
    grad_out: Bound<'py, PyAny>,
    gate: Bound<'py, PyAny>,
    up: Bound<'py, PyAny>,
    dgate: Bound<'py, PyAny>,
    dup: Bound<'py, PyAny>,
    b: f32,
) -> PyResult<()> {
    run_ra_gated_bwd_f32_pyo3(
        py,
        grad_out,
        gate,
        up,
        dgate,
        dup,
        b,
        "ra_swiglu_bwd_f32_pyo3",
        true,
    )
}

#[pyfunction]
pub fn ra_glu_bwd_f32_pyo3<'py>(
    py: Python<'py>,
    grad_out: Bound<'py, PyAny>,
    gate: Bound<'py, PyAny>,
    up: Bound<'py, PyAny>,
    dgate: Bound<'py, PyAny>,
    dup: Bound<'py, PyAny>,
    b: f32,
) -> PyResult<()> {
    run_ra_gated_bwd_f32_pyo3(
        py,
        grad_out,
        gate,
        up,
        dgate,
        dup,
        b,
        "ra_glu_bwd_f32_pyo3",
        false,
    )
}

#[pyfunction]
pub fn ra_gated_packed_fwd_f32_pyo3<'py>(
    py: Python<'py>,
    gu: Bound<'py, PyAny>,
    output: Bound<'py, PyAny>,
    rows: usize,
    hidden: usize,
    b: f32,
    is_swiglu: bool,
) -> PyResult<()> {
    let (gu_ptr, gu_len, gu_dt) = extract_tensor_info(&gu, "gu")?;
    let (out_ptr, out_len, out_dt) = extract_tensor_info_mut(&output, "output")?;
    if gu_dt != DTypeMode::Float32 || out_dt != DTypeMode::Float32 {
        return Err(PyValueError::new_err(
            "ra_gated_packed_fwd_f32_pyo3 requires Float32 tensors.",
        ));
    }
    let exp_out = rows
        .checked_mul(hidden)
        .ok_or_else(|| PyValueError::new_err("rows * hidden overflowed usize."))?;
    let exp_gu = exp_out
        .checked_mul(2)
        .ok_or_else(|| PyValueError::new_err("rows * 2 * hidden overflowed usize."))?;
    if gu_len != exp_gu || out_len != exp_out {
        return Err(PyValueError::new_err(
            "Packed gate-up / output lengths do not match rows * hidden.",
        ));
    }
    if exp_out == 0 {
        return Ok(());
    }
    check_non_null(gu_ptr, gu_len, "gu")?;
    check_non_null(out_ptr, out_len, "output")?;
    check_disjoint_bytes(
        gu_ptr,
        gu_len * std::mem::size_of::<f32>(),
        "gu",
        out_ptr,
        out_len * std::mem::size_of::<f32>(),
        "output",
    )?;
    py.detach(|| {
        let gu_s = unsafe { std::slice::from_raw_parts(gu_ptr as *const f32, gu_len) };
        let out_s = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len) };
        ra_gated_packed_fwd_batch_f32(gu_s, out_s, rows, hidden, b, is_swiglu);
    });
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ra_gated_packed_bwd_f32_pyo3<'py>(
    py: Python<'py>,
    grad_out: Bound<'py, PyAny>,
    gu: Bound<'py, PyAny>,
    dgu: Bound<'py, PyAny>,
    rows: usize,
    hidden: usize,
    b: f32,
    is_swiglu: bool,
) -> PyResult<()> {
    let (g_ptr, g_len, g_dt) = extract_tensor_info(&grad_out, "grad_out")?;
    let (gu_ptr, gu_len, gu_dt) = extract_tensor_info(&gu, "gu")?;
    let (dgu_ptr, dgu_len, dgu_dt) = extract_tensor_info_mut(&dgu, "dgu")?;
    if g_dt != DTypeMode::Float32 || gu_dt != DTypeMode::Float32 || dgu_dt != DTypeMode::Float32 {
        return Err(PyValueError::new_err(
            "ra_gated_packed_bwd_f32_pyo3 requires Float32 tensors.",
        ));
    }
    let exp_out = rows
        .checked_mul(hidden)
        .ok_or_else(|| PyValueError::new_err("rows * hidden overflowed usize."))?;
    let exp_gu = exp_out
        .checked_mul(2)
        .ok_or_else(|| PyValueError::new_err("rows * 2 * hidden overflowed usize."))?;
    if g_len != exp_out || gu_len != exp_gu || dgu_len != exp_gu {
        return Err(PyValueError::new_err(
            "Packed backward tensor lengths do not match rows * hidden.",
        ));
    }
    if exp_out == 0 {
        return Ok(());
    }
    let g_bytes = g_len * std::mem::size_of::<f32>();
    let gu_bytes = gu_len * std::mem::size_of::<f32>();
    check_non_null(g_ptr, g_len, "grad_out")?;
    check_non_null(gu_ptr, gu_len, "gu")?;
    check_non_null(dgu_ptr, dgu_len, "dgu")?;
    check_disjoint_bytes(g_ptr, g_bytes, "grad_out", dgu_ptr, gu_bytes, "dgu")?;
    check_disjoint_bytes(gu_ptr, gu_bytes, "gu", dgu_ptr, gu_bytes, "dgu")?;

    py.detach(|| {
        let g_s = unsafe { std::slice::from_raw_parts(g_ptr as *const f32, g_len) };
        let gu_s = unsafe { std::slice::from_raw_parts(gu_ptr as *const f32, gu_len) };
        let dgu_s = unsafe { std::slice::from_raw_parts_mut(dgu_ptr as *mut f32, dgu_len) };
        ra_gated_packed_bwd_batch_f32(g_s, gu_s, dgu_s, rows, hidden, b, is_swiglu);
    });
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn qrope_f32_pyo3<'py>(
    py: Python<'py>,
    x: Bound<'py, PyAny>,
    output: Bound<'py, PyAny>,
    cos_l: Bound<'py, PyAny>,
    sin_l: Bound<'py, PyAny>,
    cos_r: Bound<'py, PyAny>,
    sin_r: Bound<'py, PyAny>,
    batch_heads: usize,
    seq_len: usize,
    num_quaternions: usize,
    sign: f32,
) -> PyResult<()> {
    let (x_ptr, x_len, x_dt) = extract_tensor_info(&x, "x")?;
    let (o_ptr, o_len, o_dt) = extract_tensor_info_mut(&output, "output")?;
    let (cl_ptr, cl_len, cl_dt) = extract_tensor_info(&cos_l, "cos_l")?;
    let (sl_ptr, sl_len, sl_dt) = extract_tensor_info(&sin_l, "sin_l")?;
    let (cr_ptr, cr_len, cr_dt) = extract_tensor_info(&cos_r, "cos_r")?;
    let (sr_ptr, sr_len, sr_dt) = extract_tensor_info(&sin_r, "sin_r")?;

    if x_dt != DTypeMode::Float32
        || o_dt != DTypeMode::Float32
        || cl_dt != DTypeMode::Float32
        || sl_dt != DTypeMode::Float32
        || cr_dt != DTypeMode::Float32
        || sr_dt != DTypeMode::Float32
    {
        return Err(PyValueError::new_err(
            "qrope_f32_pyo3 requires Float32 tensors.",
        ));
    }
    let trig_len = seq_len
        .checked_mul(num_quaternions)
        .ok_or_else(|| PyValueError::new_err("seq_len * num_quaternions overflowed usize."))?;
    let expected_x = batch_heads
        .checked_mul(trig_len)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| PyValueError::new_err("qrope tensor size overflowed usize."))?;
    if x_len != expected_x
        || o_len != expected_x
        || cl_len != trig_len
        || sl_len != trig_len
        || cr_len != trig_len
        || sr_len != trig_len
    {
        return Err(PyValueError::new_err(
            "Tensor dimensions mismatch in qrope_f32_pyo3.",
        ));
    }
    if expected_x == 0 {
        return Ok(());
    }
    let x_bytes = expected_x * std::mem::size_of::<f32>();
    let t_bytes = trig_len * std::mem::size_of::<f32>();
    check_non_null(x_ptr, x_len, "x")?;
    check_non_null(o_ptr, o_len, "output")?;
    check_non_null(cl_ptr, cl_len, "cos_l")?;
    check_non_null(sl_ptr, sl_len, "sin_l")?;
    check_non_null(cr_ptr, cr_len, "cos_r")?;
    check_non_null(sr_ptr, sr_len, "sin_r")?;
    check_disjoint_bytes(x_ptr, x_bytes, "x", o_ptr, x_bytes, "output")?;
    for (p, n) in [
        (cl_ptr, "cos_l"),
        (sl_ptr, "sin_l"),
        (cr_ptr, "cos_r"),
        (sr_ptr, "sin_r"),
    ] {
        check_disjoint_bytes(p, t_bytes, n, o_ptr, x_bytes, "output")?;
    }

    py.detach(|| {
        let x_s = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, x_len) };
        let o_s = unsafe { std::slice::from_raw_parts_mut(o_ptr as *mut f32, o_len) };
        let cl_s = unsafe { std::slice::from_raw_parts(cl_ptr as *const f32, cl_len) };
        let sl_s = unsafe { std::slice::from_raw_parts(sl_ptr as *const f32, sl_len) };
        let cr_s = unsafe { std::slice::from_raw_parts(cr_ptr as *const f32, cr_len) };
        let sr_s = unsafe { std::slice::from_raw_parts(sr_ptr as *const f32, sr_len) };
        qrope_batch_f32(
            x_s,
            o_s,
            cl_s,
            sl_s,
            cr_s,
            sr_s,
            batch_heads,
            seq_len,
            num_quaternions,
            sign,
        );
    });
    Ok(())
}
