use std::collections::HashMap;
use std::ffi::{CString, c_char, c_int, c_uint, c_void};
use std::sync::Mutex;

use libloading::Library;
use thiserror::Error;

use crate::cuda_driver::pack_triton_abi_args;
use crate::manifest::KernelMetadata;

pub type HipError = c_int;
pub type HipDevice = c_int;
pub type HipModule = *mut c_void;
pub type HipFunction = *mut c_void;
pub type HipStream = *mut c_void;
pub type HipDeviceptr = *mut c_void;

pub const HIP_SUCCESS: HipError = 0;
pub const HIP_MEMCPY_HOST_TO_DEVICE: c_uint = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: c_uint = 2;

pub const SUPPORTED_ROCM_ARCHS: &[&str] = &["gfx908", "gfx90a", "gfx942", "gfx1100"];

#[repr(C, align(16))]
pub struct HipDevicePropBuffer {
    pub raw: [u8; 4096],
}

impl Default for HipDevicePropBuffer {
    fn default() -> Self {
        Self { raw: [0u8; 4096] }
    }
}

#[derive(Debug, Error)]
pub enum HipDriverError {
    #[error("ROCm HIP Driver library (amdhip64.dll / libamdhip64.so) not found: {0}")]
    LibraryLoad(#[from] libloading::Error),

    #[error("ROCm HIP API call '{api}' failed with hipError_t = {code}")]
    DriverError { api: &'static str, code: HipError },

    #[error("No ROCm-capable AMD GPU detected")]
    NoDeviceFound,

    #[error("Unsupported AMD GCN/CDNA/RDNA architecture '{0}'")]
    UnsupportedArch(String),

    #[error("Invalid C symbol name: {0}")]
    InvalidSymbol(#[from] std::ffi::NulError),
}

type FnHipInit = unsafe extern "C" fn(flags: c_uint) -> HipError;
type FnHipGetDeviceCount = unsafe extern "C" fn(count: *mut c_int) -> HipError;
type FnHipGetDevice = unsafe extern "C" fn(device: *mut HipDevice) -> HipError;
type FnHipGetDeviceProperties =
    unsafe extern "C" fn(prop: *mut HipDevicePropBuffer, device: HipDevice) -> HipError;
type FnHipModuleLoadData =
    unsafe extern "C" fn(module: *mut HipModule, image: *const c_void) -> HipError;
type FnHipModuleUnload = unsafe extern "C" fn(module: HipModule) -> HipError;
type FnHipModuleGetFunction = unsafe extern "C" fn(
    function: *mut HipFunction,
    module: HipModule,
    kname: *const c_char,
) -> HipError;
type FnHipModuleLaunchKernel = unsafe extern "C" fn(
    f: HipFunction,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem_bytes: c_uint,
    stream: HipStream,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> HipError;
type FnHipMalloc = unsafe extern "C" fn(ptr: *mut HipDeviceptr, size: usize) -> HipError;
type FnHipFree = unsafe extern "C" fn(ptr: HipDeviceptr) -> HipError;
type FnHipMemcpyAsync = unsafe extern "C" fn(
    dst: *mut c_void,
    src: *const c_void,
    size_bytes: usize,
    kind: c_uint,
    stream: HipStream,
) -> HipError;
type FnHipStreamSynchronize = unsafe extern "C" fn(stream: HipStream) -> HipError;

pub const HIP_DLL_CANDIDATES_WINDOWS: &[&str] =
    &["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"];

pub const HIP_SO_CANDIDATES_UNIX: &[&str] = &[
    "libamdhip64.so.7",
    "libamdhip64.so.6",
    "libamdhip64.so",
    "libamdhip64.so.5",
];

pub fn candidate_hip_lib_names() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        HIP_DLL_CANDIDATES_WINDOWS
    } else {
        HIP_SO_CANDIDATES_UNIX
    }
}

pub fn resolve_amd_gfx_arch(gcn_arch_raw: &str, available: &[&str]) -> Option<String> {
    let base = gcn_arch_raw
        .trim_matches(char::from(0))
        .trim()
        .split(':')
        .next()
        .unwrap_or("")
        .trim();
    if available.contains(&base) {
        return Some(base.to_string());
    }
    if matches!(base, "gfx940" | "gfx941" | "gfx950") && available.contains(&"gfx942") {
        return Some("gfx942".to_string());
    }
    if (base.starts_with("gfx11") || base.starts_with("gfx12")) && available.contains(&"gfx1100") {
        return Some("gfx1100".to_string());
    }
    None
}

#[inline]
pub fn wavefront_size_for_arch(arch: &str) -> u32 {
    if arch.starts_with("gfx10") || arch.starts_with("gfx11") || arch.starts_with("gfx12") {
        32
    } else {
        64
    }
}

pub fn extract_gcn_arch_from_prop_bytes(bytes: &[u8]) -> Option<String> {
    for i in 0..bytes.len().saturating_sub(6) {
        if &bytes[i..i + 3] == b"gfx"
            && bytes[i + 3].is_ascii_alphanumeric()
            && bytes[i + 4].is_ascii_alphanumeric()
            && bytes[i + 5].is_ascii_alphanumeric()
        {
            let end = bytes[i..]
                .iter()
                .position(|&b| b == 0)
                .map(|pos| i + pos)
                .unwrap_or(bytes.len());
            if let Ok(s) = std::str::from_utf8(&bytes[i..end]) {
                return Some(s.to_string());
            }
        }
    }
    None
}

pub struct HipDriver {
    pub device: HipDevice,
    pub gcn_arch_name: String,
    pub arch_tag: String,
    pub warp_size: u32,
    hip_module_load_data: FnHipModuleLoadData,
    hip_module_unload: FnHipModuleUnload,
    hip_module_get_function: FnHipModuleGetFunction,
    hip_module_launch_kernel: FnHipModuleLaunchKernel,
    hip_malloc: FnHipMalloc,
    hip_free: FnHipFree,
    hip_memcpy_async: FnHipMemcpyAsync,
    hip_stream_sync: FnHipStreamSynchronize,
    module_cache: Mutex<HashMap<String, (usize, usize)>>,
    _lib: Library,
}

unsafe impl Send for HipDriver {}
unsafe impl Sync for HipDriver {}

impl Drop for HipDriver {
    fn drop(&mut self) {
        if let Ok(cache) = self.module_cache.get_mut() {
            for &(mod_addr, _) in cache.values() {
                if mod_addr != 0 {
                    unsafe {
                        let _ = (self.hip_module_unload)(mod_addr as HipModule);
                    }
                }
            }
            cache.clear();
        }
    }
}

impl HipDriver {
    pub fn probe() -> Result<Self, HipDriverError> {
        let candidates = candidate_hip_lib_names();
        let mut last_err = None;
        let mut loaded_lib = None;
        for &cand in candidates {
            match unsafe { Library::new(cand) } {
                Ok(l) => {
                    loaded_lib = Some(l);
                    break;
                }
                Err(e) => last_err = Some(e),
            }
        }
        let lib = match loaded_lib {
            Some(l) => l,
            None => return Err(HipDriverError::LibraryLoad(last_err.unwrap())),
        };

        unsafe {
            let hip_init: FnHipInit = *lib.get(b"hipInit\0")?;
            let hip_get_device_count: FnHipGetDeviceCount = *lib.get(b"hipGetDeviceCount\0")?;
            let hip_get_device: FnHipGetDevice = *lib.get(b"hipGetDevice\0")?;
            let hip_get_device_props: FnHipGetDeviceProperties = *lib
                .get(b"hipGetDevicePropertiesR0600\0")
                .or_else(|_| lib.get(b"hipGetDeviceProperties\0"))?;
            let hip_module_load_data: FnHipModuleLoadData = *lib.get(b"hipModuleLoadData\0")?;
            let hip_module_unload: FnHipModuleUnload = *lib.get(b"hipModuleUnload\0")?;
            let hip_module_get_function: FnHipModuleGetFunction =
                *lib.get(b"hipModuleGetFunction\0")?;
            let hip_module_launch_kernel: FnHipModuleLaunchKernel =
                *lib.get(b"hipModuleLaunchKernel\0")?;
            let hip_malloc: FnHipMalloc = *lib.get(b"hipMalloc\0")?;
            let hip_free: FnHipFree = *lib.get(b"hipFree\0")?;
            let hip_memcpy_async: FnHipMemcpyAsync = *lib.get(b"hipMemcpyAsync\0")?;
            let hip_stream_sync: FnHipStreamSynchronize = *lib.get(b"hipStreamSynchronize\0")?;

            let res = hip_init(0);
            if res != HIP_SUCCESS {
                return Err(HipDriverError::DriverError {
                    api: "hipInit",
                    code: res,
                });
            }

            let mut count: c_int = 0;
            let res = hip_get_device_count(&mut count);
            if res != HIP_SUCCESS || count <= 0 {
                return Err(HipDriverError::NoDeviceFound);
            }

            let mut dev: HipDevice = 0;
            let res = hip_get_device(&mut dev);
            if res != HIP_SUCCESS {
                return Err(HipDriverError::DriverError {
                    api: "hipGetDevice",
                    code: res,
                });
            }

            let mut prop_buf = HipDevicePropBuffer::default();
            let res = hip_get_device_props(&mut prop_buf, dev);
            if res != HIP_SUCCESS {
                return Err(HipDriverError::DriverError {
                    api: "hipGetDeviceProperties",
                    code: res,
                });
            }

            let gcn_arch_name = extract_gcn_arch_from_prop_bytes(&prop_buf.raw)
                .or_else(|| std::env::var("LUMINAV_ROCM_ARCH").ok())
                .unwrap_or_else(|| "gfx942".to_string());

            let arch_tag = resolve_amd_gfx_arch(&gcn_arch_name, SUPPORTED_ROCM_ARCHS)
                .ok_or_else(|| HipDriverError::UnsupportedArch(gcn_arch_name.clone()))?;
            let warp_size = wavefront_size_for_arch(&arch_tag);

            Ok(Self {
                device: dev,
                gcn_arch_name,
                arch_tag,
                warp_size,
                hip_module_load_data,
                hip_module_unload,
                hip_module_get_function,
                hip_module_launch_kernel,
                hip_malloc,
                hip_free,
                hip_memcpy_async,
                hip_stream_sync,
                module_cache: Mutex::new(HashMap::new()),
                _lib: lib,
            })
        }
    }

    pub fn alloc_and_upload(&self, host_slice: &[f32]) -> Result<u64, HipDriverError> {
        let byte_size = std::mem::size_of_val(host_slice);
        let mut dptr: HipDeviceptr = std::ptr::null_mut();
        unsafe {
            let res = (self.hip_malloc)(&mut dptr, byte_size.max(16));
            if res != HIP_SUCCESS {
                return Err(HipDriverError::DriverError {
                    api: "hipMalloc",
                    code: res,
                });
            }
            if byte_size > 0 {
                let res = (self.hip_memcpy_async)(
                    dptr,
                    host_slice.as_ptr() as *const c_void,
                    byte_size,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    std::ptr::null_mut(),
                );
                if res != HIP_SUCCESS {
                    let _ = (self.hip_free)(dptr);
                    return Err(HipDriverError::DriverError {
                        api: "hipMemcpyAsync(HtoD)",
                        code: res,
                    });
                }
            }
        }
        Ok(dptr as u64)
    }

    pub fn download_and_free(
        &self,
        dptr: u64,
        host_slice: &mut [f32],
    ) -> Result<(), HipDriverError> {
        let byte_size = std::mem::size_of_val(host_slice);
        let raw_ptr = dptr as HipDeviceptr;
        unsafe {
            if byte_size > 0 {
                let res = (self.hip_memcpy_async)(
                    host_slice.as_mut_ptr() as *mut c_void,
                    raw_ptr as *const c_void,
                    byte_size,
                    HIP_MEMCPY_DEVICE_TO_HOST,
                    std::ptr::null_mut(),
                );
                if res != HIP_SUCCESS {
                    let _ = (self.hip_free)(raw_ptr);
                    return Err(HipDriverError::DriverError {
                        api: "hipMemcpyAsync(DtoH)",
                        code: res,
                    });
                }
                let sync_res = (self.hip_stream_sync)(std::ptr::null_mut());
                if sync_res != HIP_SUCCESS {
                    let _ = (self.hip_free)(raw_ptr);
                    return Err(HipDriverError::DriverError {
                        api: "hipStreamSynchronize",
                        code: sync_res,
                    });
                }
            }
            let res = (self.hip_free)(raw_ptr);
            if res != HIP_SUCCESS {
                return Err(HipDriverError::DriverError {
                    api: "hipFree",
                    code: res,
                });
            }
        }
        Ok(())
    }

    pub fn free_device_ptr(&self, dptr: u64) {
        if dptr != 0 {
            unsafe {
                let _ = (self.hip_free)(dptr as HipDeviceptr);
            }
        }
    }

    pub fn launch_hsaco_kernel(
        &self,
        hsaco_bytes: &[u8],
        meta: &KernelMetadata,
        grid_x: u32,
        user_arg_ptrs: &[*mut c_void],
    ) -> Result<(), HipDriverError> {
        let mut launch_params = pack_triton_abi_args(user_arg_ptrs, meta.num_trailing_scratch_ptrs);

        let func: HipFunction = {
            let mut cache = self
                .module_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(&(_, func_addr)) = cache.get(&meta.symbol_name) {
                func_addr as HipFunction
            } else {
                let mut module: HipModule = std::ptr::null_mut();
                let mut loaded_func: HipFunction = std::ptr::null_mut();
                let sym = CString::new(meta.symbol_name.as_str())?;
                unsafe {
                    let res = (self.hip_module_load_data)(
                        &mut module,
                        hsaco_bytes.as_ptr() as *const c_void,
                    );
                    if res != HIP_SUCCESS {
                        return Err(HipDriverError::DriverError {
                            api: "hipModuleLoadData",
                            code: res,
                        });
                    }

                    let res =
                        (self.hip_module_get_function)(&mut loaded_func, module, sym.as_ptr());
                    if res != HIP_SUCCESS {
                        let _ = (self.hip_module_unload)(module);
                        return Err(HipDriverError::DriverError {
                            api: "hipModuleGetFunction",
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
            let block_x = meta.num_warps * self.warp_size;
            let res = (self.hip_module_launch_kernel)(
                func,
                grid_x.max(1),
                1,
                1,
                block_x.max(self.warp_size),
                1,
                1,
                meta.shared_mem_bytes,
                std::ptr::null_mut(),
                launch_params.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if res != HIP_SUCCESS {
                return Err(HipDriverError::DriverError {
                    api: "hipModuleLaunchKernel",
                    code: res,
                });
            }

            let res = (self.hip_stream_sync)(std::ptr::null_mut());
            if res != HIP_SUCCESS {
                return Err(HipDriverError::DriverError {
                    api: "hipStreamSynchronize",
                    code: res,
                });
            }
        }

        Ok(())
    }
}
