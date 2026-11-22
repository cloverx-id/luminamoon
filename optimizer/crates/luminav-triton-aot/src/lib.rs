#![deny(warnings)]

mod buffer_pool;
pub mod c_api;
pub mod cuda_driver;
pub mod hip_driver;
pub mod manifest;
mod step;

use std::collections::HashMap;
use std::ffi::c_void;
use std::path::PathBuf;

use thiserror::Error;

use buffer_pool::DeviceStagingPool;
pub use cuda_driver::{
    CudaDriver, CudaDriverError, SUPPORTED_CUDA_ARCHS, TRITON_NULL_SCRATCH, pack_triton_abi_args,
    resolve_cuda_cubin_arch,
};
pub use hip_driver::{
    HipDriver, HipDriverError, SUPPORTED_ROCM_ARCHS, extract_gcn_arch_from_prop_bytes,
    resolve_amd_gfx_arch, wavefront_size_for_arch,
};
pub use manifest::{
    AbiSpec, AotManifest, KernelMetadata, ManifestError, RuntimeParamSpec, TargetBinaryConfig,
};
pub use buffer_pool::DeviceStepPtrs;

use luminav_core::LuminaError;

pub type LoadedKernel = (Vec<u8>, KernelMetadata, TargetBinaryConfig);

pub fn is_synthetic_stub_binary(bytes: &[u8]) -> bool {
    const ELF64_TEXT_OFFSET: usize = 64;
    const CUDA_STUB_MAGIC: &[u8] = b"// LUMINAV_TRITON_AOT_CUBIN";
    const ROCM_STUB_MAGIC: &[u8] = b"; LUMINAV_TRITON_AOT_HSACO";

    if bytes.len() > ELF64_TEXT_OFFSET {
        let payload = &bytes[ELF64_TEXT_OFFSET..];
        if payload.starts_with(CUDA_STUB_MAGIC) || payload.starts_with(ROCM_STUB_MAGIC) {
            return true;
        }
    }
    false
}

#[derive(Debug, Error)]
pub enum TritonAotError {
    #[error("Lumina Core error: {0}")]
    Core(#[from] LuminaError),

    #[error("AOT Manifest error: {0}")]
    Manifest(#[from] ManifestError),

    #[error("CUDA Driver error: {0}")]
    Cuda(#[from] CudaDriverError),

    #[error("ROCm HIP Driver error: {0}")]
    Hip(#[from] HipDriverError),

    #[error("Failed to read AOT binary artifact '{path}': {source}")]
    ArtifactIo {
        path: String,
        source: std::io::Error,
    },

    #[error("AOT artifact '{path}' is a synthetic stub container without compiled GPU ISA")]
    StubBinaryNotExecutable { path: String },

    #[error("AOT kernel '{kernel}' does not match runtime configuration: {detail}")]
    UnsupportedRuntimeConfig { kernel: String, detail: String },

    #[error("Neither NVIDIA CUDA (nvcuda) nor AMD ROCm (amdhip64) discrete GPU driver is active")]
    NoDiscreteDriverAvailable,
}

impl TritonAotError {
    pub fn is_fallback_eligible(&self) -> bool {
        matches!(
            self,
            Self::StubBinaryNotExecutable { .. }
                | Self::UnsupportedRuntimeConfig { .. }
                | Self::Hip(HipDriverError::DriverError {
                    api: "hipModuleLoadData",
                    code: 209,
                })
                | Self::Cuda(CudaDriverError::DriverError {
                    api: "cuModuleLoadData",
                    code: 209 | 222,
                })
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TritonDriverBackend {
    Cuda,
    RocmHip,
}

pub enum ActiveDriver {
    Cuda(CudaDriver),
    Hip(HipDriver),
}

impl ActiveDriver {
    pub(crate) fn alloc_bytes(&self, byte_size: usize) -> Result<u64, TritonAotError> {
        match self {
            Self::Cuda(c) => Ok(c.alloc_bytes(byte_size)?),
            Self::Hip(h) => Ok(h.alloc_bytes(byte_size)?),
        }
    }

    pub(crate) fn upload_into(&self, dptr: u64, slice: &[f32]) -> Result<(), TritonAotError> {
        match self {
            Self::Cuda(c) => Ok(c.upload_into(dptr, slice)?),
            Self::Hip(h) => Ok(h.upload_into(dptr, slice)?),
        }
    }

    pub(crate) fn download_from(&self, dptr: u64, slice: &mut [f32]) -> Result<(), TritonAotError> {
        match self {
            Self::Cuda(c) => Ok(c.download_from(dptr, slice)?),
            Self::Hip(h) => Ok(h.download_from(dptr, slice)?),
        }
    }

    pub(crate) fn free_ptr(&self, dptr: u64) {
        match self {
            Self::Cuda(c) => c.free_device_ptr(dptr),
            Self::Hip(h) => h.free_device_ptr(dptr),
        }
    }
}

pub struct TritonAotEngine {
    pub manifest: AotManifest,
    pub driver: ActiveDriver,
    pub artifact_root: PathBuf,
    binary_cache: HashMap<String, LoadedKernel>,
    pub(crate) staging_pool: DeviceStagingPool,
}

impl Drop for TritonAotEngine {
    fn drop(&mut self) {
        self.staging_pool.release_all(&self.driver);
    }
}

impl TritonAotEngine {
    pub fn new() -> Result<Self, TritonAotError> {
        let manifest = AotManifest::load_embedded()?;
        let artifact_root = std::env::var("LUMINAV_AOT_ARTIFACT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));

        if let Ok(cuda) = CudaDriver::probe() {
            return Ok(Self {
                manifest,
                driver: ActiveDriver::Cuda(cuda),
                artifact_root,
                binary_cache: HashMap::new(),
                staging_pool: DeviceStagingPool::default(),
            });
        }

        if let Ok(hip) = HipDriver::probe() {
            return Ok(Self {
                manifest,
                driver: ActiveDriver::Hip(hip),
                artifact_root,
                binary_cache: HashMap::new(),
                staging_pool: DeviceStagingPool::default(),
            });
        }

        Err(TritonAotError::NoDiscreteDriverAvailable)
    }

    #[inline]
    pub fn backend_kind(&self) -> TritonDriverBackend {
        match &self.driver {
            ActiveDriver::Cuda(_) => TritonDriverBackend::Cuda,
            ActiveDriver::Hip(_) => TritonDriverBackend::RocmHip,
        }
    }

    #[inline]
    pub fn arch_tag(&self) -> &str {
        match &self.driver {
            ActiveDriver::Cuda(c) => &c.arch_tag,
            ActiveDriver::Hip(h) => &h.arch_tag,
        }
    }

    #[inline]
    pub fn device_name(&self) -> &str {
        match &self.driver {
            ActiveDriver::Cuda(c) => &c.device_name,
            ActiveDriver::Hip(h) => &h.device_name,
        }
    }

    #[inline]
    pub fn platform_tag(&self) -> &str {
        match &self.driver {
            ActiveDriver::Cuda(_) => "cuda",
            ActiveDriver::Hip(_) => "rocm",
        }
    }

    pub(crate) fn load_kernel_binary(
        &mut self,
        kernel_name: &str,
    ) -> Result<LoadedKernel, TritonAotError> {
        if let Some(cached) = self.binary_cache.get(kernel_name) {
            return Ok(cached.clone());
        }
        let arch = self.arch_tag().to_string();
        let platform = self.platform_tag();
        let meta = self.manifest.get_kernel(kernel_name)?.clone();
        let target_cfg = self
            .manifest
            .resolve_target(platform, &arch, kernel_name)?
            .clone();
        let full_path = self.artifact_root.join(&target_cfg.binary_path);
        let bytes = std::fs::read(&full_path).map_err(|source| TritonAotError::ArtifactIo {
            path: full_path.display().to_string(),
            source,
        })?;
        if is_synthetic_stub_binary(&bytes) {
            return Err(TritonAotError::StubBinaryNotExecutable {
                path: full_path.display().to_string(),
            });
        }
        let entry = (bytes, meta, target_cfg);
        self.binary_cache
            .insert(kernel_name.to_string(), entry.clone());
        Ok(entry)
    }

    pub(crate) fn launch_raw(
        &self,
        kernel: &LoadedKernel,
        grid_x: u32,
        user_args: &[*mut c_void],
    ) -> Result<(), TritonAotError> {
        let (bytes, meta, target_cfg) = kernel;
        match &self.driver {
            ActiveDriver::Cuda(c) => {
                Ok(c.launch_cubin_kernel(bytes, meta, target_cfg, grid_x, user_args)?)
            }
            ActiveDriver::Hip(h) => {
                Ok(h.launch_hsaco_kernel(bytes, meta, target_cfg, grid_x, user_args)?)
            }
        }
    }
}
