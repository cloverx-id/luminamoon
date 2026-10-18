use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const EMBEDDED_MANIFEST_JSON: &str = include_str!("../artifacts/manifest.json");

#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("Failed to read manifest file: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to parse Triton AOT manifest JSON: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Kernel '{0}' not found in Triton AOT manifest")]
    KernelNotFound(String),

    #[error("Architecture '{arch}' (platform '{platform}') not supported for kernel '{kernel}'")]
    ArchNotFound {
        platform: String,
        kernel: String,
        arch: String,
    },

    #[error("Invalid Triton 3.x ABI trailing scratch count: expected >= 1, got {0}")]
    InvalidTrailingScratchCount(usize),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeParamSpec {
    pub name: String,
    pub ty: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbiSpec {
    pub triton_version: String,
    pub trailing_scratch_convention: String,
    pub ast_parameters: Vec<String>,
    pub runtime_signature: Vec<RuntimeParamSpec>,
    pub constexpr_constants: HashMap<String, i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetBinaryConfig {
    pub platform: String,
    pub arch: String,
    pub family: String,
    pub binary_path: String,
    pub shared_memory_bytes: u32,
    pub num_registers: u32,
    pub warp_size: u32,
    pub num_warps: u32,
    pub block_threads: u32,
    pub num_trailing_scratch_ptrs: usize,
    pub bf16_mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelMetadata {
    pub kernel_name: String,
    pub symbol_name: String,
    pub shared_mem_bytes: u32,
    pub num_regs: u32,
    pub num_warps: u32,
    pub num_stages: u32,
    pub block_size: u32,
    pub num_trailing_scratch_ptrs: usize,
    pub abi_spec: AbiSpec,
    pub binaries: HashMap<String, String>,
    pub targets: HashMap<String, TargetBinaryConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AotManifest {
    pub schema_version: String,
    pub generator: String,
    pub num_trailing_scratch_ptrs: usize,
    pub supported_cuda_archs: Vec<String>,
    pub supported_rocm_archs: Vec<String>,
    pub kernels: Vec<KernelMetadata>,
}

impl AotManifest {
    pub fn load_embedded() -> Result<Self, ManifestError> {
        Self::from_json_str(EMBEDDED_MANIFEST_JSON)
    }

    pub fn from_json_str(raw: &str) -> Result<Self, ManifestError> {
        let manifest: Self = serde_json::from_str(raw)?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ManifestError> {
        let content = std::fs::read_to_string(path)?;
        Self::from_json_str(&content)
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.num_trailing_scratch_ptrs == 0 {
            return Err(ManifestError::InvalidTrailingScratchCount(0));
        }
        for k in &self.kernels {
            if k.num_trailing_scratch_ptrs == 0 {
                return Err(ManifestError::InvalidTrailingScratchCount(0));
            }
            for target_cfg in k.targets.values() {
                if target_cfg.num_trailing_scratch_ptrs == 0 {
                    return Err(ManifestError::InvalidTrailingScratchCount(0));
                }
            }
        }
        Ok(())
    }

    pub fn get_kernel(&self, name: &str) -> Result<&KernelMetadata, ManifestError> {
        self.kernels
            .iter()
            .find(|k| k.kernel_name == name)
            .ok_or_else(|| ManifestError::KernelNotFound(name.to_string()))
    }

    pub fn resolve_target(
        &self,
        platform: &str,
        arch: &str,
        kernel_name: &str,
    ) -> Result<&TargetBinaryConfig, ManifestError> {
        let kernel = self.get_kernel(kernel_name)?;
        if let Some(cfg) = kernel.targets.get(arch)
            && cfg.platform == platform
        {
            return Ok(cfg);
        }
        Err(ManifestError::ArchNotFound {
            platform: platform.to_string(),
            kernel: kernel_name.to_string(),
            arch: arch.to_string(),
        })
    }

    pub fn get_binary_rel_path(
        &self,
        kernel_name: &str,
        arch: &str,
    ) -> Result<&str, ManifestError> {
        let kernel = self.get_kernel(kernel_name)?;
        kernel
            .binaries
            .get(arch)
            .map(String::as_str)
            .ok_or_else(|| ManifestError::ArchNotFound {
                platform: "any".to_string(),
                kernel: kernel_name.to_string(),
                arch: arch.to_string(),
            })
    }
}
