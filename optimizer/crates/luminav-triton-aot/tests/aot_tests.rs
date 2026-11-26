use std::ffi::c_void;
use std::path::{Path, PathBuf};

use luminav_core::{
    BoundType, BufferMode, DTypeMode, LuminaVConfig, MomentumType, ParamState, TensorViewMut,
    VarianceType, single_step_dual_scalar, single_step_t1_scalar, single_step_t2_scalar,
    single_step_tdclm_scalar,
};
use luminav_triton_aot::{
    AotManifest, CudaDriverError, HipDriverError, SUPPORTED_CUDA_ARCHS, SUPPORTED_ROCM_ARCHS,
    TritonAotEngine, TritonAotError, hip_driver, is_synthetic_stub_binary, pack_triton_abi_args,
    resolve_amd_gfx_arch, resolve_cuda_cubin_arch, wavefront_size_for_arch,
};

const ALL_14_KERNELS: &[&str] = &[
    "_lumina_dual_fused_single_pass_kernel",
    "_lumina_dual_single_pass_kernel",
    "_lumina_dual_pass1_kernel",
    "_lumina_dual_pass2_kernel",
    "_lumina_tdclm_single_pass_kernel",
    "_lumina_tdclm_pass1_kernel",
    "_lumina_tdclm_pass2_kernel",
    "_lumina_t1_pass2_kernel",
    "_lumina_t1_single_pass_kernel",
    "_lumina_t2_fused_single_pass_kernel",
    "_lumina_t2_single_pass_kernel",
    "_lumina_t2_pass1_kernel",
    "_lumina_t2_pass2_kernel",
    "_lumina_post_step_ema_kernel",
];

const WEYL_STEP_KERNELS: &[&str] = &[
    "_lumina_dual_fused_single_pass_kernel",
    "_lumina_dual_single_pass_kernel",
    "_lumina_dual_pass2_kernel",
    "_lumina_tdclm_single_pass_kernel",
    "_lumina_tdclm_pass2_kernel",
    "_lumina_t1_pass2_kernel",
    "_lumina_t1_single_pass_kernel",
    "_lumina_t2_fused_single_pass_kernel",
    "_lumina_t2_single_pass_kernel",
    "_lumina_t2_pass2_kernel",
];

fn contains_windows_drive_path(bytes: &[u8]) -> bool {
    let forbidden: &[&[u8]] = &[b"Primary Data", b"\\Users\\", b"/Users/", b"/home/", b".py"];
    if forbidden
        .iter()
        .any(|needle| bytes.windows(needle.len()).any(|w| w == *needle))
    {
        return true;
    }
    bytes.windows(6).any(|w| {
        w[0].is_ascii_uppercase()
            && w[1] == b':'
            && (w[2] == b'\\' || w[2] == b'/')
            && w[3..6]
                .iter()
                .all(|&b| b.is_ascii_alphanumeric() || b == b'_' || b == b' ')
    })
}

#[test]
fn test_manifest_and_all_182_real_elf64_binaries() {
    let manifest = AotManifest::load_embedded().expect("Embedded manifest must parse cleanly");
    assert_eq!(manifest.schema_version, "2.0.0");
    assert_eq!(manifest.math_version, "2.0.0");
    assert_eq!(manifest.num_trailing_scratch_ptrs, 2);
    assert_eq!(manifest.kernels.len(), 14);
    assert_eq!(manifest.supported_cuda_archs, SUPPORTED_CUDA_ARCHS);
    assert_eq!(manifest.supported_rocm_archs, SUPPORTED_ROCM_ARCHS);

    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut total_verified = 0usize;

    for &name in ALL_14_KERNELS {
        let k = manifest
            .get_kernel(name)
            .unwrap_or_else(|_| panic!("Kernel {name} missing"));
        let expected_symbol = if name == "_lumina_t1_pass2_kernel" {
            "_lumina_tdclm_pass2_kernel"
        } else {
            name
        };
        assert_eq!(k.symbol_name, expected_symbol);
        assert_eq!(k.num_trailing_scratch_ptrs, 2);

        if WEYL_STEP_KERNELS.contains(&name) {
            assert!(
                k.abi_spec
                    .runtime_signature
                    .iter()
                    .any(|p| p.name == "step_i32" && p.ty == "i32"),
                "Weyl kernel {name} must include step_i32 in runtime_signature"
            );
        }

        for &arch in SUPPORTED_CUDA_ARCHS {
            let cfg = manifest
                .resolve_target("cuda", arch, name)
                .expect("CUDA target config must exist");
            assert_eq!(cfg.warp_size, 32);
            assert_eq!(cfg.block_threads, k.num_warps * 32);
            assert_eq!(cfg.num_trailing_scratch_ptrs, 2);
            let full = crate_root.join(&cfg.binary_path);
            let raw = std::fs::read(&full)
                .unwrap_or_else(|e| panic!("Failed to read {}: {e}", full.display()));
            assert!(
                raw.starts_with(b"\x7fELF") && raw.len() >= 1024,
                "CUDA binary {} must be real ELF64",
                full.display()
            );
            assert!(
                !is_synthetic_stub_binary(&raw),
                "CUDA binary {} must not be a synthetic stub",
                full.display()
            );
            assert!(
                !contains_windows_drive_path(&raw),
                "CUDA binary {} leaked a local drive path",
                full.display()
            );
            total_verified += 1;
        }

        for &arch in SUPPORTED_ROCM_ARCHS {
            let cfg = manifest
                .resolve_target("rocm", arch, name)
                .expect("ROCm target config must exist");
            let expected_warp = if arch.starts_with("gfx11") { 32 } else { 64 };
            assert_eq!(cfg.warp_size, expected_warp);
            assert_eq!(cfg.block_threads, k.num_warps * expected_warp);
            assert_eq!(cfg.num_trailing_scratch_ptrs, 2);
            let full = crate_root.join(&cfg.binary_path);
            let raw = std::fs::read(&full)
                .unwrap_or_else(|e| panic!("Failed to read {}: {e}", full.display()));
            assert!(
                raw.starts_with(b"\x7fELF") && raw.len() >= 1024,
                "ROCm binary {} must be real ELF64",
                full.display()
            );
            assert!(
                !is_synthetic_stub_binary(&raw),
                "ROCm binary {} must not be a synthetic stub",
                full.display()
            );
            assert!(
                !contains_windows_drive_path(&raw),
                "ROCm binary {} leaked a local drive path",
                full.display()
            );
            total_verified += 1;
        }
    }

    assert_eq!(total_verified, 182);
}

#[test]
fn test_live_triton_aot_gpu_parity_against_scalar_oracle() {
    let Ok(mut engine) = TritonAotEngine::new() else {
        return;
    };

    let n = 4096usize;
    let init_p: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin()).collect();
    let grads: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.029).cos() * 0.1).collect();

    for buf_mode in [
        BufferMode::Dual,
        BufferMode::TdClm,
        BufferMode::T1,
        BufferMode::T2,
    ] {
        let cfg = LuminaVConfig::builder()
            .buffer(buf_mode)
            .bound_type(BoundType::Coordinate)
            .cautious(true)
            .stochastic_rounding(false)
            .fused_single_pass(true)
            .build()
            .expect("valid config");

        let mut p_gpu = init_p.clone();
        let mut st_gpu = ParamState::new(&p_gpu, buf_mode, false);
        let mut p_ref = init_p.clone();
        let mut st_ref = ParamState::new(&p_ref, buf_mode, false);

        for _ in 0..5 {
            let s_gpu = st_gpu.increment_step();
            let mut v_gpu = TensorViewMut {
                param: &mut p_gpu,
                master_param: None,
                grad: &grads,
                exp_avg: &mut st_gpu.exp_avg,
                exp_avg_sq: st_gpu.exp_avg_sq.as_deref_mut(),
                scratch: &mut st_gpu.scratch,
                dtype_mode: DTypeMode::Float32,
                state_dtype_mode: DTypeMode::Float32,
            };
            match engine.step(&mut v_gpu, &cfg, s_gpu, 0) {
                Ok(()) => {
                    assert_eq!(v_gpu.scratch.mask_sum(), 0.0_f32);
                    assert_eq!(v_gpu.scratch.u_sq_sum(), 0.0_f32);
                    assert_eq!(v_gpu.scratch.p_sq_sum(), 0.0_f32);
                }
                Err(
                    TritonAotError::Hip(HipDriverError::DriverError {
                        api: "hipModuleLoadData",
                        code: 209,
                    })
                    | TritonAotError::Cuda(CudaDriverError::DriverError {
                        api: "cuModuleLoadData",
                        code: 209 | 222,
                    }),
                ) => return,
                Err(e) => {
                    panic!("Live GPU Triton AOT step failed for {buf_mode:?} step {s_gpu}: {e}")
                }
            }

            let s_ref = st_ref.increment_step();
            let mut v_ref = TensorViewMut {
                param: &mut p_ref,
                master_param: None,
                grad: &grads,
                exp_avg: &mut st_ref.exp_avg,
                exp_avg_sq: st_ref.exp_avg_sq.as_deref_mut(),
                scratch: &mut st_ref.scratch,
                dtype_mode: DTypeMode::Float32,
                state_dtype_mode: DTypeMode::Float32,
            };
            match buf_mode {
                BufferMode::Dual => single_step_dual_scalar(&mut v_ref, &cfg, s_ref, 0).unwrap(),
                BufferMode::T2 => single_step_t2_scalar(&mut v_ref, &cfg, s_ref, 0).unwrap(),
                BufferMode::T1 => single_step_t1_scalar(&mut v_ref, &cfg, s_ref, 0).unwrap(),
                BufferMode::TdClm => single_step_tdclm_scalar(&mut v_ref, &cfg, s_ref, 0).unwrap(),
            }
        }

        let max_err_p = p_gpu
            .iter()
            .zip(p_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_err_p < 1.0e-5,
            "Live Triton AOT GPU param parity drift for {buf_mode:?}: max_err={max_err_p}"
        );
    }
}

#[test]
fn test_constexpr_validation_guards_runtime_config() {
    let manifest = AotManifest::load_embedded().expect("manifest");
    let k = manifest
        .get_kernel("_lumina_dual_fused_single_pass_kernel")
        .expect("dual kernel");
    assert!(
        k.validate_constexprs(&[
            ("bound_mode", 2),
            ("dtype_mode", 0),
            ("state_dtype_mode", 0),
            ("use_sr", 0),
            ("cautious", 1),
            ("has_master", 0),
        ])
        .is_ok()
    );
    assert!(k.validate_constexprs(&[("dtype_mode", 1)]).is_err());
    assert!(k.validate_constexprs(&[("bound_mode", 1)]).is_err());
    assert!(k.validate_constexprs(&[("has_master", 1)]).is_err());
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
        resolve_cuda_cubin_arch(10, 7, SUPPORTED_CUDA_ARCHS).as_deref(),
        Some("sm_103")
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
        resolve_amd_gfx_arch("gfx1201", SUPPORTED_ROCM_ARCHS).as_deref(),
        Some("gfx1100")
    );
    assert_eq!(wavefront_size_for_arch("gfx942"), 64);
    assert_eq!(wavefront_size_for_arch("gfx1100"), 32);
    assert!(!hip_driver::candidate_hip_lib_names().is_empty());
}

#[test]
fn test_triton38_abi_two_trailing_scratch_pointers() {
    let mut dummy_arg: u64 = 0xDEADBEEF;
    let user_ptrs = [(&mut dummy_arg as *mut u64).cast::<c_void>()];
    let packed = pack_triton_abi_args(&user_ptrs, 2);
    assert_eq!(packed.len(), 3);
    assert_eq!(unsafe { *(packed[1] as *const u64) }, 0u64);
    assert_eq!(unsafe { *(packed[2] as *const u64) }, 0u64);
}

#[test]
fn test_driver_probe_graceful_handling() {
    let _ = TritonAotEngine::new();
}

#[test]
fn test_luminav_triton_aot_files_respect_line_budget() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for subdir in ["src", "tests"] {
        for entry in std::fs::read_dir(crate_root.join(subdir)).expect("read dir") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let lines = std::fs::read_to_string(&path)
                .expect("read rs")
                .lines()
                .count();
            let ceiling = if subdir == "tests" { 650 } else { 450 };
            assert!(
                lines <= ceiling,
                "{} has {lines} lines (ceiling {ceiling})",
                path.display()
            );
        }
    }
}

#[test]
fn test_c_api_extended_and_device_entry_points() {
    use luminav_triton_aot::c_api::{
        luminav_triton_engine_destroy, luminav_triton_get_arch, luminav_triton_get_platform,
        luminav_triton_step_f32, luminav_triton_step_f32_device, luminav_triton_step_f32_ex,
        luminav_triton_supports_config,
    };
    use std::ptr;

    unsafe {
        assert_eq!(
            luminav_triton_step_f32(
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                1e-3,
                0.9,
                0.999,
                1e-8,
                0.0,
                0.8,
                0.3,
                1,
                1,
                2,
                0.05,
                1,
                1337
            ),
            -1
        );
        assert_eq!(
            luminav_triton_step_f32_ex(
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                1e-3,
                0.9,
                0.999,
                1e-8,
                0.0,
                0.8,
                0.3,
                4,
                1,
                2,
                0.05,
                1,
                1337,
                3,
                1
            ),
            -1
        );
        assert_eq!(
            luminav_triton_step_f32_device(
                ptr::null_mut(),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1e-3,
                0.9,
                0.999,
                1e-8,
                0.0,
                0.8,
                0.3,
                2,
                1,
                2,
                0.05,
                1,
                1337,
                0
            ),
            -1
        );
        assert_eq!(luminav_triton_supports_config(ptr::null(), 2, 1, 2, 0), 0);
        let platform = luminav_triton_get_platform(ptr::null());
        assert_eq!(std::ffi::CStr::from_ptr(platform).to_str().unwrap(), "none");
        let arch = luminav_triton_get_arch(ptr::null());
        assert_eq!(std::ffi::CStr::from_ptr(arch).to_str().unwrap(), "unknown");
        luminav_triton_engine_destroy(ptr::null_mut());
    }

    if let Ok(mut engine) = TritonAotEngine::new() {
        let eng_ptr: *mut TritonAotEngine = &mut engine;
        unsafe {
            assert_eq!(luminav_triton_supports_config(eng_ptr, 2, 1, 2, 0), 1);
            assert_eq!(luminav_triton_supports_config(eng_ptr, 4, 0, 2, 0), 1);
            assert_eq!(luminav_triton_supports_config(eng_ptr, 2, 1, 1, 0), 0);

            // Non-AOT radial bound: strict mode returns -5, auto fallback succeeds (0).
            let mut p = vec![1.0_f32; 64];
            let g = vec![0.1_f32; 64];
            let mut m = vec![0.0_f32; 64];
            let mut v = vec![0.0_f32; 64];
            let mut sc = [0.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0];
            assert_eq!(
                luminav_triton_step_f32_ex(
                    eng_ptr,
                    p.as_mut_ptr(),
                    g.as_ptr(),
                    m.as_mut_ptr(),
                    v.as_mut_ptr(),
                    ptr::null_mut(),
                    sc.as_mut_ptr(),
                    p.len(),
                    1e-2,
                    0.9,
                    0.999,
                    1e-8,
                    0.0,
                    0.8,
                    0.3,
                    2,
                    1,
                    1, // Radial (outside AOT constexpr)
                    0.05,
                    1,
                    1337,
                    2,
                    0 // strict GPU-only
                ),
                -5
            );
            assert_eq!(
                luminav_triton_step_f32_ex(
                    eng_ptr,
                    p.as_mut_ptr(),
                    g.as_ptr(),
                    m.as_mut_ptr(),
                    v.as_mut_ptr(),
                    ptr::null_mut(),
                    sc.as_mut_ptr(),
                    p.len(),
                    1e-2,
                    0.9,
                    0.999,
                    1e-8,
                    0.0,
                    0.8,
                    0.3,
                    2,
                    1,
                    1,
                    0.05,
                    1,
                    1337,
                    2,
                    1 // allow CPU fallback
                ),
                0
            );
            assert!(p[0] < 1.0_f32);
        }

        let cfg_rtv = LuminaVConfig::builder()
            .buffer(BufferMode::T2)
            .variance_type(VarianceType::Rtv)
            .build()
            .expect("valid rtv");
        assert!(!engine.supports_config(&cfg_rtv, DTypeMode::Float32, DTypeMode::Float32, false));

        let cfg_cwd = LuminaVConfig::builder()
            .buffer(BufferMode::Dual)
            .cautious_weight_decay(true)
            .build()
            .expect("valid cwd");
        assert!(!engine.supports_config(&cfg_cwd, DTypeMode::Float32, DTypeMode::Float32, false));

        let cfg_rpm = LuminaVConfig::builder()
            .buffer(BufferMode::T1)
            .momentum_type(MomentumType::Rpm)
            .build()
            .expect("valid rpm");
        assert!(!engine.supports_config(&cfg_rpm, DTypeMode::Float32, DTypeMode::Float32, false));

        let cfg_floor = LuminaVConfig::builder()
            .buffer(BufferMode::T2)
            .bound_floor(0.1)
            .build()
            .expect("valid floor");
        assert!(engine.supports_config(&cfg_floor, DTypeMode::Float32, DTypeMode::Float32, false));

        // Verify view.validate() rejects mismatched slice lengths before GPU upload.
        let mut p_bad = vec![1.0_f32; 64];
        let g_short = vec![0.1_f32; 32];
        let mut m_bad = vec![0.0_f32; 64];
        let mut v_bad = vec![0.0_f32; 64];
        let mut sc_bad = luminav_core::ScratchBuffer::default();
        let mut bad_view = TensorViewMut {
            param: &mut p_bad,
            master_param: None,
            grad: &g_short,
            exp_avg: &mut m_bad,
            exp_avg_sq: Some(&mut v_bad),
            scratch: &mut sc_bad,
            dtype_mode: DTypeMode::Float32,
            state_dtype_mode: DTypeMode::Float32,
        };
        assert!(matches!(
            engine.step(&mut bad_view, &cfg_floor, 1, 0),
            Err(TritonAotError::Core(
                luminav_core::LuminaError::ShapeMismatch { .. }
            ))
        ));
    }
}
