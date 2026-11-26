use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use luminav_core::DTypeMode;

fn check_cpu_device(tensor: &Bound<'_, PyAny>, name: &str) -> PyResult<()> {
    if let Ok(dev) = tensor.getattr("device")
        && let Ok(dev_type) = dev.getattr("type")
    {
        let dt: String = dev_type.extract()?;
        if dt != "cpu" {
            return Err(PyValueError::new_err(format!(
                "Tensor '{}' resides on device '{}'. Expected 'cpu' tensor.",
                name, dt
            )));
        }
    }
    Ok(())
}

fn check_contiguous_and_writable(
    tensor: &Bound<'_, PyAny>,
    name: &str,
    require_writeable: bool,
) -> PyResult<()> {
    let is_contiguous = if let Ok(is_contig_method) = tensor.getattr("is_contiguous")
        && is_contig_method.is_callable()
    {
        is_contig_method.call0()?.extract::<bool>()?
    } else if let Ok(flags) = tensor.getattr("flags")
        && let Ok(c_contig) = flags.getattr("c_contiguous")
    {
        c_contig.extract::<bool>()?
    } else {
        return Err(PyValueError::new_err(format!(
            "Unable to verify memory contiguity for '{}'. Expected a PyTorch tensor or NumPy ndarray.",
            name
        )));
    };

    if !is_contiguous {
        return Err(PyValueError::new_err(format!(
            "Tensor/array '{}' is non-contiguous. Please call .contiguous() or np.ascontiguousarray() before passing it to LuminaV.",
            name
        )));
    }

    if require_writeable
        && let Ok(flags) = tensor.getattr("flags")
        && let Ok(writeable) = flags.getattr("writeable")
        && !writeable.extract::<bool>()?
    {
        return Err(PyValueError::new_err(format!(
            "Mutable buffer '{}' is read-only (flags.writeable == False).",
            name
        )));
    }

    Ok(())
}

fn extract_raw_ptr(tensor: &Bound<'_, PyAny>, name: &str) -> PyResult<usize> {
    if let Ok(method) = tensor.getattr("data_ptr")
        && method.is_callable()
    {
        return method.call0()?.extract();
    }
    if let Ok(array_iface) = tensor.getattr("__array_interface__") {
        let data_tuple = array_iface.get_item("data")?;
        return data_tuple.get_item(0)?.extract();
    }
    if let Ok(ctypes) = tensor.getattr("ctypes") {
        return ctypes.getattr("data")?.extract();
    }
    Err(PyValueError::new_err(format!(
        "Unable to extract raw memory pointer from object '{}'. Expected an object supporting data_ptr(), __array_interface__, or ctypes.data.",
        name
    )))
}

fn extract_numel(tensor: &Bound<'_, PyAny>, name: &str) -> PyResult<usize> {
    if let Ok(method) = tensor.getattr("numel")
        && method.is_callable()
    {
        return method.call0()?.extract();
    }
    if let Ok(size) = tensor.getattr("size")
        && !size.is_callable()
    {
        return size.extract();
    }
    Err(PyValueError::new_err(format!(
        "Unable to determine element count for object '{}'.",
        name
    )))
}

fn validate_scalar_dtype_layout(
    tensor: &Bound<'_, PyAny>,
    dtype_obj: &Bound<'_, PyAny>,
    expected_itemsize: usize,
    name: &str,
) -> PyResult<()> {
    if let Ok(fields) = dtype_obj.getattr("fields")
        && !fields.is_none()
    {
        return Err(PyValueError::new_err(format!(
            "Structured dtype is not supported for '{}'.",
            name
        )));
    }
    if let Ok(subdtype) = dtype_obj.getattr("subdtype")
        && !subdtype.is_none()
    {
        return Err(PyValueError::new_err(format!(
            "Subarray dtype is not supported for '{}'.",
            name
        )));
    }
    if let Ok(byteorder_obj) = dtype_obj.getattr("byteorder")
        && let Ok(bo) = byteorder_obj.extract::<String>()
    {
        let native_bo = if cfg!(target_endian = "little") {
            "<"
        } else {
            ">"
        };
        if bo != "=" && bo != "|" && bo != native_bo {
            return Err(PyValueError::new_err(format!(
                "Non-native byteorder '{}' is not supported for '{}'.",
                bo, name
            )));
        }
    }
    let actual_itemsize: Option<usize> = if let Ok(method) = tensor.getattr("element_size")
        && method.is_callable()
    {
        Some(method.call0()?.extract()?)
    } else if let Ok(itemsize) = tensor.getattr("itemsize") {
        Some(itemsize.extract()?)
    } else if let Ok(itemsize) = dtype_obj.getattr("itemsize") {
        Some(itemsize.extract()?)
    } else {
        None
    };

    match actual_itemsize {
        Some(sz) if sz == expected_itemsize => Ok(()),
        Some(sz) => Err(PyValueError::new_err(format!(
            "Element size mismatch for '{}': expected {} bytes, got {} bytes.",
            name, expected_itemsize, sz
        ))),
        None => Err(PyValueError::new_err(format!(
            "Unable to verify element byte size for '{}'.",
            name
        ))),
    }
}

fn validate_ptr_bounds(ptr: usize, numel: usize, elem_bytes: usize, name: &str) -> PyResult<()> {
    let total_bytes = numel
        .checked_mul(elem_bytes)
        .filter(|&b| b <= isize::MAX as usize)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "Tensor '{}' byte size overflows isize::MAX (numel={}, elem_bytes={}).",
                name, numel, elem_bytes
            ))
        })?;
    if total_bytes == 0 {
        return Ok(());
    }
    if ptr == 0 {
        return Err(PyValueError::new_err(format!(
            "Tensor '{}' has a null data pointer for non-empty length {}.",
            name, numel
        )));
    }
    if !ptr.is_multiple_of(elem_bytes) {
        return Err(PyValueError::new_err(format!(
            "Tensor '{}' pointer 0x{:x} is not {}-byte aligned.",
            name, ptr, elem_bytes
        )));
    }
    Ok(())
}

fn parse_float_dtype_exact(dtype_str: &str) -> Option<(DTypeMode, usize)> {
    let native_f4 = if cfg!(target_endian = "little") {
        "<f4"
    } else {
        ">f4"
    };
    let native_f2 = if cfg!(target_endian = "little") {
        "<f2"
    } else {
        ">f2"
    };
    if dtype_str == "torch.float32"
        || dtype_str == "float32"
        || dtype_str == "f4"
        || dtype_str == "=f4"
        || dtype_str == native_f4
    {
        Some((DTypeMode::Float32, 4))
    } else if dtype_str == "torch.bfloat16" || dtype_str == "bfloat16" {
        Some((DTypeMode::BFloat16, 2))
    } else if dtype_str == "torch.float16"
        || dtype_str == "float16"
        || dtype_str == "f2"
        || dtype_str == "=f2"
        || dtype_str == native_f2
    {
        Some((DTypeMode::Float16, 2))
    } else {
        None
    }
}

fn extract_tensor_info_impl(
    tensor: &Bound<'_, PyAny>,
    name: &str,
    require_writeable: bool,
) -> PyResult<(usize, usize, DTypeMode)> {
    check_cpu_device(tensor, name)?;
    check_contiguous_and_writable(tensor, name, require_writeable)?;

    let dtype_obj = tensor.getattr("dtype")?;
    let dtype_str = dtype_obj.str()?.to_string().to_lowercase();
    let (dtype_mode, elem_bytes) = parse_float_dtype_exact(&dtype_str).ok_or_else(|| {
        PyValueError::new_err(format!(
            "Unsupported dtype '{}' for '{}'. Supported dtypes are torch.float32 (float32), torch.bfloat16 (bfloat16), and torch.float16 (float16).",
            dtype_str, name
        ))
    })?;
    validate_scalar_dtype_layout(tensor, &dtype_obj, elem_bytes, name)?;

    let data_ptr = extract_raw_ptr(tensor, name)?;
    let numel = extract_numel(tensor, name)?;
    validate_ptr_bounds(data_ptr, numel, elem_bytes, name)?;
    Ok((data_ptr, numel, dtype_mode))
}

pub(crate) fn extract_tensor_info(
    tensor: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<(usize, usize, DTypeMode)> {
    extract_tensor_info_impl(tensor, name, false)
}

pub(crate) fn extract_tensor_info_mut(
    tensor: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<(usize, usize, DTypeMode)> {
    extract_tensor_info_impl(tensor, name, true)
}

pub(crate) fn extract_scratch_ptr(scratch: &Bound<'_, PyAny>) -> PyResult<usize> {
    let (data_ptr, numel, dtype_mode) = extract_tensor_info_impl(scratch, "scratch", true)?;
    if dtype_mode != DTypeMode::Float32 {
        return Err(PyValueError::new_err(format!(
            "Tensor 'scratch' must have dtype float32, got {:?}.",
            dtype_mode
        )));
    }
    if numel < 6 {
        return Err(PyValueError::new_err(format!(
            "Scratch buffer must have at least 6 elements, got {}.",
            numel
        )));
    }
    Ok(data_ptr)
}

fn extract_i64_tensor_info_impl(
    tensor: &Bound<'_, PyAny>,
    name: &str,
    require_writeable: bool,
) -> PyResult<(usize, usize)> {
    check_cpu_device(tensor, name)?;
    check_contiguous_and_writable(tensor, name, require_writeable)?;

    let dtype_obj = tensor.getattr("dtype")?;
    let dtype_str = dtype_obj.str()?.to_string().to_lowercase();
    let native_i8 = if cfg!(target_endian = "little") {
        "<i8"
    } else {
        ">i8"
    };
    if dtype_str != "torch.int64"
        && dtype_str != "int64"
        && dtype_str != "i8"
        && dtype_str != "=i8"
        && dtype_str != native_i8
    {
        return Err(PyValueError::new_err(format!(
            "Tensor '{}' must have dtype int64, got '{}'.",
            name, dtype_str
        )));
    }
    validate_scalar_dtype_layout(tensor, &dtype_obj, 8, name)?;

    let data_ptr = extract_raw_ptr(tensor, name)?;
    let numel = extract_numel(tensor, name)?;
    validate_ptr_bounds(data_ptr, numel, 8, name)?;
    Ok((data_ptr, numel))
}

pub(crate) fn extract_i64_tensor_info(
    tensor: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<(usize, usize)> {
    extract_i64_tensor_info_impl(tensor, name, false)
}

pub(crate) fn extract_i64_tensor_info_mut(
    tensor: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<(usize, usize)> {
    extract_i64_tensor_info_impl(tensor, name, true)
}

#[pyfunction]
pub fn detect_hardware_pyo3(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let caps = luminav_dispatch::detect_hardware();
    let dict = PyDict::new(py);
    dict.set_item("simd_backend", caps.simd_backend.name())?;
    dict.set_item("triton_aot_available", caps.triton_aot_available)?;
    dict.set_item("triton_arch", caps.triton_arch.as_deref())?;
    dict.set_item("cubecl_available", caps.cubecl_available)?;
    dict.set_item("cubecl_backend", caps.cubecl_backend.as_deref())?;
    dict.set_item("gpu_device_name", caps.gpu_device_name.as_deref())?;
    dict.set_item("wave32_supported", caps.wave32_supported)?;
    dict.set_item(
        "gpu_dispatch_threshold",
        luminav_dispatch::GPU_DISPATCH_THRESHOLD,
    )?;
    Ok(dict)
}
