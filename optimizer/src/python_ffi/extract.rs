use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use luminav_core::DTypeMode;

pub(crate) fn extract_tensor_info(
    tensor: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<(usize, usize, DTypeMode)> {
    if let Ok(dev) = tensor.getattr("device")
        && let Ok(dev_type) = dev.getattr("type")
    {
        let dt: String = dev_type.extract()?;
        if dt != "cpu" {
            return Err(PyValueError::new_err(format!(
                "Tensor '{}' resides on device '{}' which is not directly accessible as host memory in step_pyo3. Expected 'cpu' tensor.",
                name, dt
            )));
        }
    }

    let is_contiguous = if let Ok(is_contig_method) = tensor.getattr("is_contiguous") {
        if is_contig_method.is_callable() {
            is_contig_method.call0()?.extract::<bool>()?
        } else {
            true
        }
    } else if let Ok(flags) = tensor.getattr("flags") {
        if let Ok(c_contig) = flags.getattr("c_contiguous") {
            c_contig.extract::<bool>()?
        } else {
            true
        }
    } else {
        true
    };

    if !is_contiguous {
        return Err(PyValueError::new_err(format!(
            "Tensor/array '{}' is non-contiguous. Please call .contiguous() or np.ascontiguousarray() before passing it to LuminaV.",
            name
        )));
    }

    let data_ptr: usize = if let Ok(method) = tensor.getattr("data_ptr") {
        method.call0()?.extract()?
    } else if let Ok(array_iface) = tensor.getattr("__array_interface__") {
        let data_tuple = array_iface.get_item("data")?;
        let ptr_val = data_tuple.get_item(0)?;
        ptr_val.extract()?
    } else if let Ok(ctypes) = tensor.getattr("ctypes") {
        ctypes.getattr("data")?.extract()?
    } else {
        return Err(PyValueError::new_err(format!(
            "Unable to extract raw memory pointer from object '{}'. Expected an object supporting __array_interface__, ctypes.data, or data_ptr (e.g. NumPy ndarray or PyTorch tensor).",
            name
        )));
    };

    let numel: usize = if let Ok(method) = tensor.getattr("numel") {
        method.call0()?.extract()?
    } else if let Ok(size) = tensor.getattr("size") {
        if size.is_callable() {
            if let Ok(n) = tensor.getattr("numel") {
                n.call0()?.extract()?
            } else {
                size.call0()?.extract()?
            }
        } else {
            size.extract()?
        }
    } else if let Ok(len) = tensor.len() {
        len
    } else {
        return Err(PyValueError::new_err(format!(
            "Unable to determine element count for object '{}'.",
            name
        )));
    };

    let dtype_str: String = tensor.getattr("dtype")?.str()?.to_string().to_lowercase();
    let dtype_mode = if dtype_str.contains("float32")
        || dtype_str == "f4"
        || dtype_str == "<f4"
        || dtype_str == "=f4"
    {
        DTypeMode::Float32
    } else if dtype_str.contains("bfloat16") {
        DTypeMode::BFloat16
    } else if dtype_str.contains("float16")
        || dtype_str == "f2"
        || dtype_str == "<f2"
        || dtype_str == "=f2"
    {
        DTypeMode::Float16
    } else {
        return Err(PyValueError::new_err(format!(
            "Unsupported dtype '{}' for '{}'. Supported dtypes are torch.float32 (float32), torch.bfloat16 (bfloat16), and torch.float16 (float16).",
            dtype_str, name
        )));
    };
    Ok((data_ptr, numel, dtype_mode))
}

pub(crate) fn extract_scratch_ptr(scratch: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(dev) = scratch.getattr("device")
        && let Ok(dev_type) = dev.getattr("type")
    {
        let dt: String = dev_type.extract()?;
        if dt != "cpu" {
            return Err(PyValueError::new_err(format!(
                "Tensor 'scratch' resides on device '{}'. Expected 'cpu' tensor.",
                dt
            )));
        }
    }

    if let Ok(dtype_attr) = scratch.getattr("dtype") {
        let dtype_str = dtype_attr.to_string();
        if !dtype_str.contains("float32") && dtype_str != "<f4" && dtype_str != "=f4" {
            return Err(PyValueError::new_err(format!(
                "Tensor 'scratch' must have dtype float32, got '{}'.",
                dtype_str
            )));
        }
    }

    let numel: usize = if let Ok(method) = scratch.getattr("numel") {
        method.call0()?.extract()?
    } else if let Ok(size) = scratch.getattr("size") {
        if size.is_callable() {
            size.call0()?.extract()?
        } else {
            size.extract()?
        }
    } else {
        scratch.len().unwrap_or(6)
    };

    if numel < 6 {
        return Err(PyValueError::new_err(format!(
            "Scratch buffer must have at least 6 elements, got {}.",
            numel
        )));
    }

    let data_ptr: usize = if let Ok(method) = scratch.getattr("data_ptr") {
        method.call0()?.extract()?
    } else if let Ok(array_iface) = scratch.getattr("__array_interface__") {
        let data_tuple = array_iface.get_item("data")?;
        let ptr_val = data_tuple.get_item(0)?;
        ptr_val.extract()?
    } else if let Ok(ctypes) = scratch.getattr("ctypes") {
        ctypes.getattr("data")?.extract()?
    } else {
        return Err(PyValueError::new_err(
            "Unable to extract raw memory pointer from scratch buffer. Expected an object supporting __array_interface__, ctypes.data, or data_ptr (e.g. NumPy ndarray or PyTorch tensor).",
        ));
    };

    Ok(data_ptr)
}

#[pyfunction]
pub fn detect_hardware_pyo3(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let caps = luminav_dispatch::detect_hardware();
    let dict = PyDict::new(py);
    dict.set_item("simd_backend", caps.simd_backend.name())?;
    dict.set_item("triton_aot_available", caps.triton_aot_available)?;
    dict.set_item("triton_arch", caps.triton_arch.as_deref())?;
    dict.set_item("vulkan_available", caps.vulkan_available)?;
    dict.set_item("gpu_device_name", caps.gpu_device_name.as_deref())?;
    dict.set_item("wave32_supported", caps.wave32_supported)?;
    dict.set_item(
        "gpu_dispatch_threshold",
        luminav_dispatch::GPU_DISPATCH_THRESHOLD,
    )?;
    Ok(dict)
}
