use std::collections::HashMap;
use std::ffi::{CString, c_char, c_int, c_uint, c_void};
use std::sync::Mutex;

use libloading::Library;
use thiserror::Error;

use crate::manifest::KernelMetadata;

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUdeviceptr = u64;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

pub static TRITON_NULL_SCRATCH: u64 = 0;

pub const SUPPORTED_CUDA_ARCHS: &[&str] = &[
    "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_103", "sm_120",
];

#[derive(Debug, Error)]
pub enum CudaDriverError {
    #[error("CUDA Driver library (nvcuda.dll / libcuda.so.1) not found: {0}")]
    LibraryLoad(#[from] libloading::Error),

    #[error("CUDA Driver API call '{api}' failed with CUresult = {code}")]
    DriverError { api: &'static str, code: CUresult },

    #[error("No CUDA-capable GPU detected")]
    NoDeviceFound,

    #[error("Unsupported NVIDIA SM architecture sm_{0}{1}")]
    UnsupportedArch(i32, i32),

    #[error("Invalid C symbol name: {0}")]
    InvalidSymbol(#[from] std::ffi::NulError),
}

type FnCuInit = unsafe extern "C" fn(flags: c_uint) -> CUresult;
type FnCuDeviceGetCount = unsafe extern "C" fn(count: *mut c_int) -> CUresult;
type FnCuDeviceGet = unsafe extern "C" fn(device: *mut CUdevice, ordinal: c_int) -> CUresult;
type FnCuDeviceGetAttribute =
    unsafe extern "C" fn(pi: *mut c_int, attrib: c_int, dev: CUdevice) -> CUresult;
type FnCuDevicePrimaryCtxRetain =
    unsafe extern "C" fn(pctx: *mut CUcontext, dev: CUdevice) -> CUresult;
type FnCuCtxSetCurrent = unsafe extern "C" fn(ctx: CUcontext) -> CUresult;
type FnCuModuleLoadData =
    unsafe extern "C" fn(module: *mut CUmodule, image: *const c_void) -> CUresult;
type FnCuModuleUnload = unsafe extern "C" fn(hmod: CUmodule) -> CUresult;
type FnCuModuleGetFunction =
    unsafe extern "C" fn(hfunc: *mut CUfunction, hmod: CUmodule, name: *const c_char) -> CUresult;
type FnCuLaunchKernel = unsafe extern "C" fn(
    f: CUfunction,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem_bytes: c_uint,
    h_stream: CUstream,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult;
type FnCuMemAllocV2 = unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
type FnCuMemFreeV2 = unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult;
type FnCuMemcpyHtoDV2 = unsafe extern "C" fn(
    dst_device: CUdeviceptr,
    src_host: *const c_void,
    byte_count: usize,
) -> CUresult;
type FnCuMemcpyDtoHV2 = unsafe extern "C" fn(
    dst_host: *mut c_void,
    src_device: CUdeviceptr,
    byte_count: usize,
) -> CUresult;
type FnCuStreamSynchronize = unsafe extern "C" fn(h_stream: CUstream) -> CUresult;

pub fn resolve_cuda_cubin_arch(major: i32, minor: i32, available: &[&str]) -> Option<String> {
    let exact = format!("sm_{major}{minor}");
    if available.contains(&exact.as_str()) {
        return Some(exact);
    }

    let mut best_minor = -1;
    let mut best_arch: Option<String> = None;

    for &cand in available {
        if let Some(num_str) = cand.strip_prefix("sm_")
            && num_str.len() >= 2
        {
            let (maj_s, min_s) = num_str.split_at(num_str.len() - 1);
            if let (Ok(c_maj), Ok(c_min)) = (maj_s.parse::<i32>(), min_s.parse::<i32>())
                && c_maj == major
                && c_min <= minor
                && c_min > best_minor
            {
                best_minor = c_min;
                best_arch = Some(cand.to_string());
            }
        }
    }

    best_arch
}

pub struct CudaDriver {
    pub device: CUdevice,
    pub context: CUcontext,
    pub major: i32,
    pub minor: i32,
    pub arch_tag: String,
    cu_module_load_data: FnCuModuleLoadData,
    cu_module_unload: FnCuModuleUnload,
    cu_module_get_function: FnCuModuleGetFunction,
    cu_launch_kernel: FnCuLaunchKernel,
    cu_mem_alloc: FnCuMemAllocV2,
    cu_mem_free: FnCuMemFreeV2,
    cu_memcpy_htod: FnCuMemcpyHtoDV2,
    cu_memcpy_dtoh: FnCuMemcpyDtoHV2,
    cu_stream_sync: FnCuStreamSynchronize,
    module_cache: Mutex<HashMap<String, (usize, usize)>>,
    _lib: Library,
}

unsafe impl Send for CudaDriver {}
unsafe impl Sync for CudaDriver {}

impl Drop for CudaDriver {
    fn drop(&mut self) {
        if let Ok(cache) = self.module_cache.get_mut() {
            for &(mod_addr, _) in cache.values() {
                if mod_addr != 0 {
                    unsafe {
                        let _ = (self.cu_module_unload)(mod_addr as CUmodule);
                    }
                }
            }
            cache.clear();
        }
    }
}

impl CudaDriver {
    pub fn probe() -> Result<Self, CudaDriverError> {
        let lib = unsafe {
            #[cfg(target_os = "windows")]
            {
                Library::new("nvcuda.dll")?
            }
            #[cfg(not(target_os = "windows"))]
            {
                Library::new("libcuda.so.1").or_else(|_| Library::new("libcuda.so"))?
            }
        };

        unsafe {
            let cu_init: FnCuInit = *lib.get(b"cuInit\0")?;
            let cu_device_get_count: FnCuDeviceGetCount = *lib.get(b"cuDeviceGetCount\0")?;
            let cu_device_get: FnCuDeviceGet = *lib.get(b"cuDeviceGet\0")?;
            let cu_device_get_attribute: FnCuDeviceGetAttribute =
                *lib.get(b"cuDeviceGetAttribute\0")?;
            let cu_primary_ctx_retain: FnCuDevicePrimaryCtxRetain =
                *lib.get(b"cuDevicePrimaryCtxRetain\0")?;
            let cu_ctx_set_current: FnCuCtxSetCurrent = *lib.get(b"cuCtxSetCurrent\0")?;
            let cu_module_load_data: FnCuModuleLoadData = *lib.get(b"cuModuleLoadData\0")?;
            let cu_module_unload: FnCuModuleUnload = *lib.get(b"cuModuleUnload\0")?;
            let cu_module_get_function: FnCuModuleGetFunction =
                *lib.get(b"cuModuleGetFunction\0")?;
            let cu_launch_kernel: FnCuLaunchKernel = *lib.get(b"cuLaunchKernel\0")?;
            let cu_mem_alloc: FnCuMemAllocV2 = *lib.get(b"cuMemAlloc_v2\0")?;
            let cu_mem_free: FnCuMemFreeV2 = *lib.get(b"cuMemFree_v2\0")?;
            let cu_memcpy_htod: FnCuMemcpyHtoDV2 = *lib.get(b"cuMemcpyHtoD_v2\0")?;
            let cu_memcpy_dtoh: FnCuMemcpyDtoHV2 = *lib.get(b"cuMemcpyDtoH_v2\0")?;
            let cu_stream_sync: FnCuStreamSynchronize = *lib.get(b"cuStreamSynchronize\0")?;

            let res = cu_init(0);
            if res != CUDA_SUCCESS {
                return Err(CudaDriverError::DriverError {
                    api: "cuInit",
                    code: res,
                });
            }

            let mut count: c_int = 0;
            let res = cu_device_get_count(&mut count);
            if res != CUDA_SUCCESS || count <= 0 {
                return Err(CudaDriverError::NoDeviceFound);
            }

            let mut dev: CUdevice = 0;
            let res = cu_device_get(&mut dev, 0);
            if res != CUDA_SUCCESS {
                return Err(CudaDriverError::DriverError {
                    api: "cuDeviceGet",
                    code: res,
                });
            }

            let mut major: c_int = 0;
            let mut minor: c_int = 0;
            let res_maj = cu_device_get_attribute(
                &mut major,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                dev,
            );
            let res_min = cu_device_get_attribute(
                &mut minor,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                dev,
            );
            if res_maj != CUDA_SUCCESS || res_min != CUDA_SUCCESS {
                return Err(CudaDriverError::DriverError {
                    api: "cuDeviceGetAttribute",
                    code: if res_maj != CUDA_SUCCESS {
                        res_maj
                    } else {
                        res_min
                    },
                });
            }

            let arch_tag = resolve_cuda_cubin_arch(major, minor, SUPPORTED_CUDA_ARCHS)
                .ok_or(CudaDriverError::UnsupportedArch(major, minor))?;

            let mut context: CUcontext = std::ptr::null_mut();
            let res = cu_primary_ctx_retain(&mut context, dev);
            if res != CUDA_SUCCESS {
                return Err(CudaDriverError::DriverError {
                    api: "cuDevicePrimaryCtxRetain",
                    code: res,
                });
            }

            let res = cu_ctx_set_current(context);
            if res != CUDA_SUCCESS {
                return Err(CudaDriverError::DriverError {
                    api: "cuCtxSetCurrent",
                    code: res,
                });
            }

            Ok(Self {
                device: dev,
                context,
                major,
                minor,
                arch_tag,
                cu_module_load_data,
                cu_module_unload,
                cu_module_get_function,
                cu_launch_kernel,
                cu_mem_alloc,
                cu_mem_free,
                cu_memcpy_htod,
                cu_memcpy_dtoh,
                cu_stream_sync,
                module_cache: Mutex::new(HashMap::new()),
                _lib: lib,
            })
        }
    }

    pub fn alloc_and_upload(&self, host_slice: &[f32]) -> Result<CUdeviceptr, CudaDriverError> {
        let byte_size = std::mem::size_of_val(host_slice);
        let mut dptr: CUdeviceptr = 0;
        unsafe {
            let res = (self.cu_mem_alloc)(&mut dptr, byte_size.max(16));
            if res != CUDA_SUCCESS {
                return Err(CudaDriverError::DriverError {
                    api: "cuMemAlloc_v2",
                    code: res,
                });
            }
            if byte_size > 0 {
                let res =
                    (self.cu_memcpy_htod)(dptr, host_slice.as_ptr() as *const c_void, byte_size);
                if res != CUDA_SUCCESS {
                    let _ = (self.cu_mem_free)(dptr);
                    return Err(CudaDriverError::DriverError {
                        api: "cuMemcpyHtoD_v2",
                        code: res,
                    });
                }
            }
        }
        Ok(dptr)
    }

    pub fn download_and_free(
        &self,
        dptr: CUdeviceptr,
        host_slice: &mut [f32],
    ) -> Result<(), CudaDriverError> {
        let byte_size = std::mem::size_of_val(host_slice);
        unsafe {
            if byte_size > 0 {
                let res =
                    (self.cu_memcpy_dtoh)(host_slice.as_mut_ptr() as *mut c_void, dptr, byte_size);
                if res != CUDA_SUCCESS {
                    let _ = (self.cu_mem_free)(dptr);
                    return Err(CudaDriverError::DriverError {
                        api: "cuMemcpyDtoH_v2",
                        code: res,
                    });
                }
            }
            let res = (self.cu_mem_free)(dptr);
            if res != CUDA_SUCCESS {
                return Err(CudaDriverError::DriverError {
                    api: "cuMemFree_v2",
                    code: res,
                });
            }
        }
        Ok(())
    }

    pub fn free_device_ptr(&self, dptr: CUdeviceptr) {
        if dptr != 0 {
            unsafe {
                let _ = (self.cu_mem_free)(dptr);
            }
        }
    }

    pub fn launch_cubin_kernel(
        &self,
        cubin_bytes: &[u8],
        meta: &KernelMetadata,
        grid_x: u32,
        user_arg_ptrs: &[*mut c_void],
    ) -> Result<(), CudaDriverError> {
        let mut launch_params = pack_triton_abi_args(user_arg_ptrs, meta.num_trailing_scratch_ptrs);

        let func: CUfunction = {
            let mut cache = self
                .module_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(&(_, func_addr)) = cache.get(&meta.symbol_name) {
                func_addr as CUfunction
            } else {
                let mut module: CUmodule = std::ptr::null_mut();
                let mut loaded_func: CUfunction = std::ptr::null_mut();
                let sym = CString::new(meta.symbol_name.as_str())?;
                unsafe {
                    let res = (self.cu_module_load_data)(
                        &mut module,
                        cubin_bytes.as_ptr() as *const c_void,
                    );
                    if res != CUDA_SUCCESS {
                        return Err(CudaDriverError::DriverError {
                            api: "cuModuleLoadData",
                            code: res,
                        });
                    }

                    let res = (self.cu_module_get_function)(&mut loaded_func, module, sym.as_ptr());
                    if res != CUDA_SUCCESS {
                        let _ = (self.cu_module_unload)(module);
                        return Err(CudaDriverError::DriverError {
                            api: "cuModuleGetFunction",
                            code: res,
                        });
                    }
                }
                cache.insert(
                    meta.symbol_name.clone(),
                    (module as usize, loaded_func as usize),
                );
                loaded_func
            }
        };

        unsafe {
            let block_x = meta.num_warps * 32;
            let res = (self.cu_launch_kernel)(
                func,
                grid_x.max(1),
                1,
                1,
                block_x.max(32),
                1,
                1,
                meta.shared_mem_bytes,
                std::ptr::null_mut(),
                launch_params.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if res != CUDA_SUCCESS {
                return Err(CudaDriverError::DriverError {
                    api: "cuLaunchKernel",
                    code: res,
                });
            }

            let res = (self.cu_stream_sync)(std::ptr::null_mut());
            if res != CUDA_SUCCESS {
                return Err(CudaDriverError::DriverError {
                    api: "cuStreamSynchronize",
                    code: res,
                });
            }
        }

        Ok(())
    }
}

pub fn pack_triton_abi_args(
    user_arg_ptrs: &[*mut c_void],
    num_trailing_scratch_ptrs: usize,
) -> Vec<*mut c_void> {
    let mut args = Vec::with_capacity(user_arg_ptrs.len() + num_trailing_scratch_ptrs);
    args.extend_from_slice(user_arg_ptrs);
    let scratch_ptr = (&TRITON_NULL_SCRATCH as *const u64) as *mut c_void;
    for _ in 0..num_trailing_scratch_ptrs {
        args.push(scratch_ptr);
    }
    args
}
