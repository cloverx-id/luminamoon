use std::cell::RefCell;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};
use rayon::prelude::*;

use luminav_core::prng::{f16_bits_to_f32, f32_to_f16_bits_rne};
use luminav_core::{
    BoundType, BufferMode, DTypeMode, ExecutionMode, LuminaVConfig, LuminaVConfigBuilder,
    MasterWeightMode, ScratchBuffer, TensorViewMut,
};
use luminav_dispatch::{GPU_DISPATCH_THRESHOLD, step_auto};

fn parse_config(dict: &Bound<'_, PyDict>) -> PyResult<LuminaVConfig> {
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

    let centered_defect: bool = dict
        .get_item("centered_defect")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(false);
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
        .centered_defect(centered_defect)
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

fn extract_tensor_info(
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

fn extract_scratch_ptr(scratch: &Bound<'_, PyAny>) -> PyResult<usize> {
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

struct WorkItem {
    param_ptr: usize,
    master_ptr: Option<usize>,
    grad_ptr: usize,
    exp_avg_ptr: usize,
    exp_avg_sq_ptr: Option<usize>,
    scratch_ptr: usize,
    numel: usize,
    dtype_mode: DTypeMode,
    grad_dtype_mode: DTypeMode,
    state_dtype_mode: DTypeMode,
    step: u64,
    param_index: usize,
}

#[derive(Default)]
struct FfiDecodeWorkspace {
    param_buf: Vec<f32>,
    grad_buf: Vec<f32>,
    exp_avg_buf: Vec<f32>,
    exp_avg_sq_buf: Vec<f32>,
}

thread_local! {
    static FFI_WORKSPACE: RefCell<FfiDecodeWorkspace> = RefCell::new(FfiDecodeWorkspace::default());
}

#[inline]
unsafe fn decode_into_f32_buf(ptr: usize, numel: usize, mode: DTypeMode, dst: &mut Vec<f32>) {
    dst.clear();
    dst.reserve(numel);
    match mode {
        DTypeMode::BFloat16 => {
            let raw = unsafe { std::slice::from_raw_parts(ptr as *const u16, numel) };
            dst.extend(raw.iter().map(|&bits| f32::from_bits((bits as u32) << 16)));
        }
        DTypeMode::Float16 => {
            let raw = unsafe { std::slice::from_raw_parts(ptr as *const u16, numel) };
            dst.extend(raw.iter().map(|&bits| f16_bits_to_f32(bits)));
        }
        DTypeMode::Float32 => {
            let f32_slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, numel) };
            dst.extend_from_slice(f32_slice);
        }
    }
}

#[inline]
unsafe fn encode_from_f32_slice(ptr: usize, src: &[f32], mode: DTypeMode) {
    match mode {
        DTypeMode::BFloat16 => {
            let dst = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u16, src.len()) };
            for (d, &val) in dst.iter_mut().zip(src.iter()) {
                *d = (val.to_bits() >> 16) as u16;
            }
        }
        DTypeMode::Float16 => {
            let dst = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u16, src.len()) };
            for (d, &val) in dst.iter_mut().zip(src.iter()) {
                *d = f32_to_f16_bits_rne(val);
            }
        }
        DTypeMode::Float32 => {
            let dst = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, src.len()) };
            dst.copy_from_slice(src);
        }
    }
}

fn process_work_item(item: &WorkItem, config: &LuminaVConfig) -> Result<(), String> {
    FFI_WORKSPACE.with(|cell| {
        let mut ws = cell.borrow_mut();
        let FfiDecodeWorkspace {
            param_buf,
            grad_buf,
            exp_avg_buf,
            exp_avg_sq_buf,
        } = &mut *ws;

        unsafe {
            let param_slice: &mut [f32] = if item.dtype_mode != DTypeMode::Float32 {
                decode_into_f32_buf(item.param_ptr, item.numel, item.dtype_mode, param_buf);
                param_buf.as_mut_slice()
            } else {
                std::slice::from_raw_parts_mut(item.param_ptr as *mut f32, item.numel)
            };

            let master_slice = item
                .master_ptr
                .map(|p| std::slice::from_raw_parts_mut(p as *mut f32, item.numel));

            let grad_slice: &[f32] = if item.grad_dtype_mode != DTypeMode::Float32 {
                decode_into_f32_buf(item.grad_ptr, item.numel, item.grad_dtype_mode, grad_buf);
                grad_buf.as_slice()
            } else {
                std::slice::from_raw_parts(item.grad_ptr as *const f32, item.numel)
            };

            let exp_avg_slice: &mut [f32] = if item.state_dtype_mode != DTypeMode::Float32 {
                decode_into_f32_buf(
                    item.exp_avg_ptr,
                    item.numel,
                    item.state_dtype_mode,
                    exp_avg_buf,
                );
                exp_avg_buf.as_mut_slice()
            } else {
                std::slice::from_raw_parts_mut(item.exp_avg_ptr as *mut f32, item.numel)
            };

            let exp_avg_sq_slice: Option<&mut [f32]> = match item.exp_avg_sq_ptr {
                Some(ptr) => {
                    if item.state_dtype_mode != DTypeMode::Float32 {
                        decode_into_f32_buf(ptr, item.numel, item.state_dtype_mode, exp_avg_sq_buf);
                        Some(exp_avg_sq_buf.as_mut_slice())
                    } else {
                        Some(std::slice::from_raw_parts_mut(ptr as *mut f32, item.numel))
                    }
                }
                None => None,
            };

            let mut scratch_copy =
                ScratchBuffer::from_array(*(item.scratch_ptr as *const [f32; 6]));

            let mut view = TensorViewMut {
                param: param_slice,
                master_param: master_slice,
                grad: grad_slice,
                exp_avg: exp_avg_slice,
                exp_avg_sq: exp_avg_sq_slice,
                scratch: &mut scratch_copy,
                dtype_mode: item.dtype_mode,
                state_dtype_mode: item.state_dtype_mode,
            };

            let mut item_config = config.clone();
            if item.dtype_mode == DTypeMode::Float32 && item.master_ptr.is_none() {
                item_config.master_weights = MasterWeightMode::None;
            }

            step_auto(&mut view, &item_config, item.step, item.param_index).map_err(|e| {
                format!(
                    "Dispatch execution failed on param {}: {}",
                    item.param_index, e
                )
            })?;

            if item.dtype_mode != DTypeMode::Float32 {
                encode_from_f32_slice(item.param_ptr, param_buf, item.dtype_mode);
            }
            if item.state_dtype_mode != DTypeMode::Float32 {
                encode_from_f32_slice(item.exp_avg_ptr, exp_avg_buf, item.state_dtype_mode);
                if let Some(ptr) = item.exp_avg_sq_ptr {
                    encode_from_f32_slice(ptr, exp_avg_sq_buf, item.state_dtype_mode);
                }
            }

            *(item.scratch_ptr as *mut [f32; 6]) = *scratch_copy.as_slice();
        }
        Ok(())
    })
}

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
                        process_work_item(item, &config)?;
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
