use cubecl::Device;
use cubecl::prelude::*;

use crate::error::CubeClError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeClBackend {
    Cuda,
    Hip,
    Wgpu,
    Cpu,
}

impl CubeClBackend {
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Hip => "hip",
            Self::Wgpu => "wgpu",
            Self::Cpu => "cpu",
        }
    }
}

pub struct CubeClLuminaEngine {
    pub(crate) client: Client,
    backend: CubeClBackend,
    pub(crate) cube_dim_1d: usize,
}

impl CubeClLuminaEngine {
    pub fn new() -> Result<Self, CubeClError> {
        #[cfg(feature = "cuda")]
        if let Ok(eng) = Self::with_cuda() {
            return Ok(eng);
        }

        #[cfg(feature = "hip")]
        if let Ok(eng) = Self::with_hip() {
            return Ok(eng);
        }

        #[cfg(feature = "wgpu")]
        if let Ok(eng) = Self::with_wgpu() {
            return Ok(eng);
        }

        #[cfg(feature = "cpu")]
        if let Ok(eng) = Self::with_cpu() {
            return Ok(eng);
        }

        Err(CubeClError::RuntimeUnavailable(
            "No CubeCL runtime feature enabled or available on this host".to_string(),
        ))
    }

    #[cfg(feature = "wgpu")]
    pub fn with_wgpu() -> Result<Self, CubeClError> {
        let client = std::panic::catch_unwind(|| Device::Wgpu(Default::default()).client())
            .map_err(|_| CubeClError::RuntimeUnavailable("WGPU adapter unavailable".to_string()))?;
        Ok(Self::from_client(client, CubeClBackend::Wgpu))
    }

    #[cfg(feature = "cuda")]
    pub fn with_cuda() -> Result<Self, CubeClError> {
        let client = std::panic::catch_unwind(|| Device::Cuda(Default::default()).client())
            .map_err(|_| CubeClError::RuntimeUnavailable("CUDA runtime unavailable".to_string()))?;
        Ok(Self::from_client(client, CubeClBackend::Cuda))
    }

    #[cfg(feature = "hip")]
    pub fn with_hip() -> Result<Self, CubeClError> {
        let client = std::panic::catch_unwind(|| Device::Hip(Default::default()).client())
            .map_err(|_| {
                CubeClError::RuntimeUnavailable("HIP/ROCm runtime unavailable".to_string())
            })?;
        Ok(Self::from_client(client, CubeClBackend::Hip))
    }

    #[cfg(feature = "cpu")]
    pub fn with_cpu() -> Result<Self, CubeClError> {
        let client = std::panic::catch_unwind(|| Device::Cpu(Default::default()).client())
            .map_err(|_| {
                CubeClError::RuntimeUnavailable("CubeCL CPU runtime unavailable".to_string())
            })?;
        Ok(Self::from_client(client, CubeClBackend::Cpu))
    }

    fn from_client(client: Client, backend: CubeClBackend) -> Self {
        let max_units = client.properties().hardware.max_units_per_cube as usize;
        // Pick largest power-of-two cube size up to 256 supported by the active hardware.
        let cube_dim_1d = max_units
            .clamp(32, 256)
            .next_power_of_two()
            .min(max_units)
            .max(1);
        let cube_dim_1d = if cube_dim_1d.is_power_of_two() {
            cube_dim_1d
        } else {
            (cube_dim_1d / 2).next_power_of_two().max(1)
        };
        Self {
            client,
            backend,
            cube_dim_1d,
        }
    }

    #[inline]
    pub fn backend(&self) -> CubeClBackend {
        self.backend
    }

    #[inline]
    pub fn cube_dim_1d(&self) -> usize {
        self.cube_dim_1d
    }

    pub(crate) fn read_sum_f32(&self, handle: cubecl::server::Handle) -> f32 {
        let bytes = self.client.read_one_unchecked(handle);
        f32::from_bytes(&bytes).iter().copied().sum()
    }

    pub(crate) fn read_into_slice(&self, handle: cubecl::server::Handle, dst: &mut [f32]) {
        let bytes = self.client.read_one_unchecked(handle);
        dst.copy_from_slice(f32::from_bytes(&bytes));
    }
}
