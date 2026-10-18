#![deny(warnings)]

pub mod c_api;
pub mod cuda_driver;
pub mod hip_driver;
pub mod manifest;

use std::collections::HashMap;
use std::ffi::c_void;
use std::path::PathBuf;

use thiserror::Error;

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

use luminav_core::{
    BoundType, BufferMode, LuminaError, LuminaVConfig, TensorViewMut, compute_bias_correction,
    derive_kernel_seeds,
};

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

    #[error("Neither NVIDIA CUDA (nvcuda) nor AMD ROCm (amdhip64) discrete GPU driver is active")]
    NoDiscreteDriverAvailable,
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

pub struct TritonAotEngine {
    pub manifest: AotManifest,
    pub driver: ActiveDriver,
    pub artifact_root: PathBuf,
    binary_cache: HashMap<String, (Vec<u8>, KernelMetadata)>,
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
            });
        }

        if let Ok(hip) = HipDriver::probe() {
            return Ok(Self {
                manifest,
                driver: ActiveDriver::Hip(hip),
                artifact_root,
                binary_cache: HashMap::new(),
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
    pub fn platform_tag(&self) -> &str {
        match &self.driver {
            ActiveDriver::Cuda(_) => "cuda",
            ActiveDriver::Hip(_) => "rocm",
        }
    }

    fn load_kernel_binary(
        &mut self,
        kernel_name: &str,
    ) -> Result<(Vec<u8>, KernelMetadata), TritonAotError> {
        if let Some(cached) = self.binary_cache.get(kernel_name) {
            return Ok(cached.clone());
        }
        let arch = self.arch_tag().to_string();
        let platform = self.platform_tag();
        let meta = self.manifest.get_kernel(kernel_name)?.clone();
        let target_cfg = self.manifest.resolve_target(platform, &arch, kernel_name)?;
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
        let entry = (bytes, meta);
        self.binary_cache
            .insert(kernel_name.to_string(), entry.clone());
        Ok(entry)
    }

    fn alloc_upload(&self, slice: &[f32]) -> Result<u64, TritonAotError> {
        match &self.driver {
            ActiveDriver::Cuda(c) => Ok(c.alloc_and_upload(slice)?),
            ActiveDriver::Hip(h) => Ok(h.alloc_and_upload(slice)?),
        }
    }

    fn download_free(&self, dptr: u64, slice: &mut [f32]) -> Result<(), TritonAotError> {
        match &self.driver {
            ActiveDriver::Cuda(c) => Ok(c.download_and_free(dptr, slice)?),
            ActiveDriver::Hip(h) => Ok(h.download_and_free(dptr, slice)?),
        }
    }

    fn free_ptr(&self, dptr: u64) {
        match &self.driver {
            ActiveDriver::Cuda(c) => c.free_device_ptr(dptr),
            ActiveDriver::Hip(h) => h.free_device_ptr(dptr),
        }
    }

    fn launch_raw(
        &self,
        kernel_bytes: &[u8],
        meta: &KernelMetadata,
        grid_x: u32,
        user_args: &[*mut c_void],
    ) -> Result<(), TritonAotError> {
        match &self.driver {
            ActiveDriver::Cuda(c) => {
                Ok(c.launch_cubin_kernel(kernel_bytes, meta, grid_x, user_args)?)
            }
            ActiveDriver::Hip(h) => {
                Ok(h.launch_hsaco_kernel(kernel_bytes, meta, grid_x, user_args)?)
            }
        }
    }

    pub fn step(
        &mut self,
        view: &mut TensorViewMut<'_>,
        config: &LuminaVConfig,
        step: u64,
        param_index: usize,
    ) -> Result<(), TritonAotError> {
        config.validate()?;
        let n = view.param.len();
        if n == 0 {
            return Ok(());
        }

        let diff_kernel = if config.centered_defect && n > 1 {
            Some(self.load_kernel_binary("_lumina_diff_sum_kernel")?)
        } else {
            None
        };

        let main_kernel = match config.buffer {
            BufferMode::TdClm => self.load_kernel_binary("_lumina_tdclm_single_pass_kernel")?,
            BufferMode::Dual => {
                if view.exp_avg_sq.is_none() {
                    return Err(TritonAotError::Core(
                        LuminaError::MissingSecondaryStateBuffer,
                    ));
                }
                self.load_kernel_binary("_lumina_dual_fused_single_pass_kernel")?
            }
        };

        let ema_kernel = if config.cautious || config.effective_bound_type() == BoundType::Radial {
            Some(self.load_kernel_binary("_lumina_post_step_ema_kernel")?)
        } else {
            None
        };

        let mut n_i32 = n as i32;
        let mut total_f32 = n as f32;
        let mut beta1 = config.betas.0;
        let mut beta2 = config.betas.1;
        let mut lr = config.lr;
        let mut wd = config.weight_decay;
        let mut eps = config.eps;
        let mut lam = config.lam;
        let mut bound_ratio = config.bound_ratio;
        let mut clamp_min = config.cautious_clamp_min;
        let mut ema_decay = config.ema_decay;
        let mut step_i32 = step as i32;

        let (seed_m, _) = derive_kernel_seeds(config.seed, step, param_index);
        let mut seed_i32 = seed_m;

        let d_param = self.alloc_upload(view.param)?;
        let d_master = match &view.master_param {
            Some(mp) => match self.alloc_upload(mp) {
                Ok(ptr) => ptr,
                Err(e) => {
                    self.free_ptr(d_param);
                    return Err(e);
                }
            },
            None => 0u64,
        };
        let d_grad = match self.alloc_upload(view.grad) {
            Ok(ptr) => ptr,
            Err(e) => {
                self.free_ptr(d_master);
                self.free_ptr(d_param);
                return Err(e);
            }
        };
        let d_exp_avg = match self.alloc_upload(view.exp_avg) {
            Ok(ptr) => ptr,
            Err(e) => {
                self.free_ptr(d_grad);
                self.free_ptr(d_master);
                self.free_ptr(d_param);
                return Err(e);
            }
        };
        let mut d_scratch = match self.alloc_upload(view.scratch.as_slice()) {
            Ok(ptr) => ptr,
            Err(e) => {
                self.free_ptr(d_exp_avg);
                self.free_ptr(d_grad);
                self.free_ptr(d_master);
                self.free_ptr(d_param);
                return Err(e);
            }
        };

        let free_all_base = |engine: &Self, p: u64, m: u64, g: u64, ea: u64, sc: u64| {
            engine.free_ptr(sc);
            engine.free_ptr(ea);
            engine.free_ptr(g);
            engine.free_ptr(m);
            engine.free_ptr(p);
        };

        let mut p_ptr = d_param;
        let mut master_p_ptr = d_master;
        let mut grad_ptr = d_grad;
        let mut exp_avg_ptr = d_exp_avg;

        if let Some((diff_bytes, diff_meta)) = diff_kernel {
            let grid_x = (n as u32).div_ceil(diff_meta.block_size.max(1));
            let args: [*mut c_void; 4] = [
                (&mut grad_ptr as *mut u64) as *mut c_void,
                (&mut exp_avg_ptr as *mut u64) as *mut c_void,
                (&mut d_scratch as *mut u64) as *mut c_void,
                (&mut n_i32 as *mut i32) as *mut c_void,
            ];
            if let Err(e) = self.launch_raw(&diff_bytes, &diff_meta, grid_x, &args) {
                free_all_base(self, d_param, d_master, d_grad, d_exp_avg, d_scratch);
                return Err(e);
            }
        }

        let (k_bytes, k_meta) = main_kernel;
        match config.buffer {
            BufferMode::TdClm => {
                let grid_x = (n as u32).div_ceil(k_meta.block_size.max(1));
                let args: [*mut c_void; 14] = [
                    (&mut p_ptr as *mut u64) as *mut c_void,
                    (&mut master_p_ptr as *mut u64) as *mut c_void,
                    (&mut grad_ptr as *mut u64) as *mut c_void,
                    (&mut exp_avg_ptr as *mut u64) as *mut c_void,
                    (&mut d_scratch as *mut u64) as *mut c_void,
                    (&mut n_i32 as *mut i32) as *mut c_void,
                    (&mut total_f32 as *mut f32) as *mut c_void,
                    (&mut beta1 as *mut f32) as *mut c_void,
                    (&mut lr as *mut f32) as *mut c_void,
                    (&mut wd as *mut f32) as *mut c_void,
                    (&mut eps as *mut f32) as *mut c_void,
                    (&mut lam as *mut f32) as *mut c_void,
                    (&mut bound_ratio as *mut f32) as *mut c_void,
                    (&mut seed_i32 as *mut i32) as *mut c_void,
                ];
                if let Err(e) = self.launch_raw(&k_bytes, &k_meta, grid_x, &args) {
                    free_all_base(self, d_param, d_master, d_grad, d_exp_avg, d_scratch);
                    return Err(e);
                }
            }
            BufferMode::Dual => {
                let exp_avg_sq = view
                    .exp_avg_sq
                    .as_mut()
                    .ok_or(LuminaError::MissingSecondaryStateBuffer)?;
                let mut exp_avg_sq_ptr = match self.alloc_upload(exp_avg_sq) {
                    Ok(ptr) => ptr,
                    Err(e) => {
                        free_all_base(self, d_param, d_master, d_grad, d_exp_avg, d_scratch);
                        return Err(e);
                    }
                };
                let coeffs = compute_bias_correction(
                    step,
                    config.betas.0,
                    config.betas.1,
                    config.eps,
                    config.tau,
                    view.dtype_mode,
                );
                let mut c1 = coeffs.c1;
                let mut c2 = coeffs.c2;

                let grid_x = (n as u32).div_ceil(k_meta.block_size.max(1));
                let args: [*mut c_void; 16] = [
                    (&mut p_ptr as *mut u64) as *mut c_void,
                    (&mut master_p_ptr as *mut u64) as *mut c_void,
                    (&mut grad_ptr as *mut u64) as *mut c_void,
                    (&mut exp_avg_ptr as *mut u64) as *mut c_void,
                    (&mut exp_avg_sq_ptr as *mut u64) as *mut c_void,
                    (&mut d_scratch as *mut u64) as *mut c_void,
                    (&mut n_i32 as *mut i32) as *mut c_void,
                    (&mut total_f32 as *mut f32) as *mut c_void,
                    (&mut beta1 as *mut f32) as *mut c_void,
                    (&mut beta2 as *mut f32) as *mut c_void,
                    (&mut c1 as *mut f32) as *mut c_void,
                    (&mut c2 as *mut f32) as *mut c_void,
                    (&mut lr as *mut f32) as *mut c_void,
                    (&mut wd as *mut f32) as *mut c_void,
                    (&mut bound_ratio as *mut f32) as *mut c_void,
                    (&mut seed_i32 as *mut i32) as *mut c_void,
                ];
                if let Err(e) = self.launch_raw(&k_bytes, &k_meta, grid_x, &args) {
                    self.free_ptr(exp_avg_sq_ptr);
                    free_all_base(self, d_param, d_master, d_grad, d_exp_avg, d_scratch);
                    return Err(e);
                }
                if let Err(e) = self.download_free(exp_avg_sq_ptr, exp_avg_sq) {
                    free_all_base(self, d_param, d_master, d_grad, d_exp_avg, d_scratch);
                    return Err(e);
                }
            }
        }

        if let Some((ema_bytes, ema_meta)) = ema_kernel {
            let ema_args: [*mut c_void; 7] = [
                (&mut d_scratch as *mut u64) as *mut c_void,
                (&mut total_f32 as *mut f32) as *mut c_void,
                (&mut lr as *mut f32) as *mut c_void,
                (&mut clamp_min as *mut f32) as *mut c_void,
                (&mut ema_decay as *mut f32) as *mut c_void,
                (&mut step_i32 as *mut i32) as *mut c_void,
                (&mut bound_ratio as *mut f32) as *mut c_void,
            ];
            if let Err(e) = self.launch_raw(&ema_bytes, &ema_meta, 1, &ema_args) {
                free_all_base(self, d_param, d_master, d_grad, d_exp_avg, d_scratch);
                return Err(e);
            }
        }

        self.free_ptr(d_grad);
        let res_param = self.download_free(d_param, view.param);
        let res_master = if let Some(mp) = &mut view.master_param {
            self.download_free(d_master, mp)
        } else {
            Ok(())
        };
        let res_exp_avg = self.download_free(d_exp_avg, view.exp_avg);
        let res_scratch = self.download_free(d_scratch, view.scratch.as_mut_slice());

        res_param?;
        res_master?;
        res_exp_avg?;
        res_scratch?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_parsing_all_84_binaries() {
        let manifest = AotManifest::load_embedded().expect("Embedded manifest must parse cleanly");
        assert_eq!(manifest.schema_version, "1.0.0");
        assert_eq!(manifest.num_trailing_scratch_ptrs, 1);
        assert_eq!(manifest.kernels.len(), 7);
        assert_eq!(
            manifest.supported_cuda_archs,
            vec![
                "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_103", "sm_120"
            ]
        );
        assert_eq!(
            manifest.supported_rocm_archs,
            vec!["gfx908", "gfx90a", "gfx942", "gfx1100"]
        );

        let expected_kernels = [
            "_lumina_diff_sum_kernel",
            "_lumina_dual_fused_single_pass_kernel",
            "_lumina_tdclm_single_pass_kernel",
            "_lumina_tdclm_pass1_kernel",
            "_lumina_tdclm_pass2_kernel",
            "_lumina_post_step_ema_kernel",
            "_lumina_dual_pass1_kernel",
        ];

        let mut total_verified = 0usize;
        for name in expected_kernels {
            let k = manifest
                .get_kernel(name)
                .unwrap_or_else(|_| panic!("Kernel {name} missing"));
            assert_eq!(k.num_trailing_scratch_ptrs, 1);

            for &arch in SUPPORTED_CUDA_ARCHS {
                let target_cfg = manifest
                    .resolve_target("cuda", arch, name)
                    .expect("CUDA target config must exist");
                assert_eq!(target_cfg.warp_size, 32);
                assert_eq!(target_cfg.block_threads, k.num_warps * 32);
                if arch == "sm_75" {
                    assert_eq!(target_cfg.bf16_mode, "fp32_upcast_emulation");
                } else {
                    assert!(target_cfg.bf16_mode.starts_with("native_"));
                }
                let full = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&target_cfg.binary_path);
                assert!(
                    full.exists(),
                    "CUDA artifact file {} must exist on disk",
                    full.display()
                );
                total_verified += 1;
            }

            for &arch in SUPPORTED_ROCM_ARCHS {
                let target_cfg = manifest
                    .resolve_target("rocm", arch, name)
                    .expect("ROCm target config must exist");
                let expected_warp = if arch == "gfx1100" { 32 } else { 64 };
                assert_eq!(target_cfg.warp_size, expected_warp);
                assert_eq!(target_cfg.block_threads, k.num_warps * expected_warp);
                let full = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&target_cfg.binary_path);
                assert!(
                    full.exists(),
                    "ROCm artifact file {} must exist on disk",
                    full.display()
                );
                total_verified += 1;
            }
        }
        assert_eq!(total_verified, 84);
    }

    #[test]
    fn test_cuda_sm_arch_matching_and_blackwell_isolation() {
        assert_eq!(
            resolve_cuda_cubin_arch(8, 0, SUPPORTED_CUDA_ARCHS).as_deref(),
            Some("sm_80")
        );
        assert_eq!(
            resolve_cuda_cubin_arch(8, 7, SUPPORTED_CUDA_ARCHS).as_deref(),
            Some("sm_86")
        );
        assert_eq!(
            resolve_cuda_cubin_arch(10, 0, SUPPORTED_CUDA_ARCHS).as_deref(),
            Some("sm_100")
        );
        assert_eq!(
            resolve_cuda_cubin_arch(10, 3, SUPPORTED_CUDA_ARCHS).as_deref(),
            Some("sm_103")
        );
        assert_eq!(
            resolve_cuda_cubin_arch(10, 7, SUPPORTED_CUDA_ARCHS).as_deref(),
            Some("sm_103")
        );
        assert_eq!(
            resolve_cuda_cubin_arch(12, 0, SUPPORTED_CUDA_ARCHS).as_deref(),
            Some("sm_120")
        );
        assert_eq!(
            resolve_cuda_cubin_arch(12, 1, SUPPORTED_CUDA_ARCHS).as_deref(),
            Some("sm_120")
        );
        let without_120 = &[
            "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_103",
        ];
        assert_eq!(resolve_cuda_cubin_arch(12, 0, without_120), None);
    }

    #[test]
    fn test_rocm_gcn_arch_parsing_and_wavefront_size() {
        assert_eq!(
            resolve_amd_gfx_arch("gfx942:sramecc+:xnack-", SUPPORTED_ROCM_ARCHS).as_deref(),
            Some("gfx942")
        );
        assert_eq!(
            resolve_amd_gfx_arch("gfx950:sramecc+:xnack-", SUPPORTED_ROCM_ARCHS).as_deref(),
            Some("gfx942")
        );
        assert_eq!(
            resolve_amd_gfx_arch("gfx90a:sramecc+:xnack+", SUPPORTED_ROCM_ARCHS).as_deref(),
            Some("gfx90a")
        );
        assert_eq!(
            resolve_amd_gfx_arch("gfx1100", SUPPORTED_ROCM_ARCHS).as_deref(),
            Some("gfx1100")
        );
        assert_eq!(
            resolve_amd_gfx_arch("gfx1151", SUPPORTED_ROCM_ARCHS).as_deref(),
            Some("gfx1100")
        );
        assert_eq!(
            resolve_amd_gfx_arch("gfx1201", SUPPORTED_ROCM_ARCHS).as_deref(),
            Some("gfx1100")
        );
        assert_eq!(
            resolve_amd_gfx_arch("gfx1250", SUPPORTED_ROCM_ARCHS).as_deref(),
            Some("gfx1100")
        );
        assert_eq!(wavefront_size_for_arch("gfx908"), 64);
        assert_eq!(wavefront_size_for_arch("gfx90a"), 64);
        assert_eq!(wavefront_size_for_arch("gfx942"), 64);
        assert_eq!(wavefront_size_for_arch("gfx950"), 64);
        assert_eq!(wavefront_size_for_arch("gfx1100"), 32);
        assert_eq!(wavefront_size_for_arch("gfx1151"), 32);
        assert_eq!(wavefront_size_for_arch("gfx1201"), 32);
        assert_eq!(wavefront_size_for_arch("gfx1250"), 32);
        let hip_libs = hip_driver::candidate_hip_lib_names();
        assert!(!hip_libs.is_empty());
    }

    #[test]
    fn test_triton3_abi_trailing_scratch_pointer() {
        let mut dummy_arg: u64 = 0xDEADBEEF;
        let user_ptrs = [(&mut dummy_arg as *mut u64) as *mut c_void];
        let packed = pack_triton_abi_args(&user_ptrs, 1);
        assert_eq!(packed.len(), 2);

        let trailing_val = unsafe { *(packed[1] as *const u64) };
        assert_eq!(trailing_val, 0u64);

        let packed_v38 = pack_triton_abi_args(&user_ptrs, 2);
        assert_eq!(packed_v38.len(), 3);
        let profile_scratch_val = unsafe { *(packed_v38[2] as *const u64) };
        assert_eq!(profile_scratch_val, 0u64);
    }

    #[test]
    fn test_driver_probe_graceful_handling() {
        let _ = TritonAotEngine::new();
    }
}
