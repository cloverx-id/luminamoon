use std::cell::RefCell;

use luminav_core::prng::{f16_bits_to_f32, f32_to_f16_bits_rne};
use luminav_core::{DTypeMode, LuminaVConfig, MasterWeightMode, ScratchBuffer, TensorViewMut};
use luminav_dispatch::step_auto;

#[derive(Clone, Copy)]
pub(crate) struct WorkItem {
    pub(crate) param_ptr: usize,
    pub(crate) master_ptr: Option<usize>,
    pub(crate) grad_ptr: usize,
    pub(crate) exp_avg_ptr: usize,
    pub(crate) exp_avg_sq_ptr: Option<usize>,
    pub(crate) scratch_ptr: usize,
    pub(crate) numel: usize,
    pub(crate) dtype_mode: DTypeMode,
    pub(crate) grad_dtype_mode: DTypeMode,
    pub(crate) state_dtype_mode: DTypeMode,
    pub(crate) step: u64,
    pub(crate) param_index: usize,
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

pub(crate) fn process_work_item(item: &WorkItem, config: &LuminaVConfig) -> Result<(), String> {
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
                .filter(|&p| !(item.dtype_mode == DTypeMode::Float32 && p == item.param_ptr))
                .map(|p| std::slice::from_raw_parts_mut(p as *mut f32, item.numel));

            let grad_slice: &[f32] = if item.grad_dtype_mode != DTypeMode::Float32 {
                decode_into_f32_buf(item.grad_ptr, item.numel, item.grad_dtype_mode, grad_buf);
                grad_buf.as_slice()
            } else {
                std::slice::from_raw_parts(item.grad_ptr as *const f32, item.numel)
            };

            // Promote 16-bit FP16 state buffers to bitwise-BF16 encoding in-place so
            // exp_avg and exp_avg_sq retain an 8-bit exponent range without extra memory.
            let eff_state_dtype_mode = if item.state_dtype_mode == DTypeMode::Float16 {
                DTypeMode::BFloat16
            } else {
                item.state_dtype_mode
            };

            let exp_avg_slice: &mut [f32] = if eff_state_dtype_mode != DTypeMode::Float32 {
                decode_into_f32_buf(
                    item.exp_avg_ptr,
                    item.numel,
                    eff_state_dtype_mode,
                    exp_avg_buf,
                );
                exp_avg_buf.as_mut_slice()
            } else {
                std::slice::from_raw_parts_mut(item.exp_avg_ptr as *mut f32, item.numel)
            };

            let exp_avg_sq_slice: Option<&mut [f32]> = match item.exp_avg_sq_ptr {
                Some(ptr) => {
                    if eff_state_dtype_mode != DTypeMode::Float32 {
                        decode_into_f32_buf(ptr, item.numel, eff_state_dtype_mode, exp_avg_sq_buf);
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
                state_dtype_mode: eff_state_dtype_mode,
            };

            let mut item_config = config.clone();
            if item.dtype_mode == DTypeMode::Float32 && view.master_param.is_none() {
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
            if eff_state_dtype_mode != DTypeMode::Float32 {
                encode_from_f32_slice(item.exp_avg_ptr, exp_avg_buf, eff_state_dtype_mode);
                if let Some(ptr) = item.exp_avg_sq_ptr {
                    encode_from_f32_slice(ptr, exp_avg_sq_buf, eff_state_dtype_mode);
                }
            }

            std::ptr::write_unaligned(item.scratch_ptr as *mut [f32; 6], *scratch_copy.as_slice());
        }
        Ok(())
    })
}
