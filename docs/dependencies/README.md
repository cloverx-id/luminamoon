# Workspace Dependency & Compatibility Notes

This directory documents version compatibility, API/ABI updates, security fixes, and source-code adaptations across the dependencies of the **LuminaMoon (`LuminaV v2.0.0` & `luminamoon v2.0.0rc1`)** workspace.

---

## Dependency Summary

| Component / Library | Tested Version Range | Target Version | Adaptation Notes | Detailed Document |
| :--- | :--- | :--- | :--- | :--- |
| **Rust Toolchain & SIMD** | `1.95.0` - `1.99.1` | `1.99.1` (`edition = "2024"`) | Adopted `std::cfg_select!`, isolated `Scalar` vs `Avx2` dispatch paths, ARM64 NEON | [01_rust_toolchain_and_simd.md](01_rust_toolchain_and_simd.md) |
| **Supporting Rust Crates** | `2024` - `2026` | `rayon 1.12.0`, `thiserror 2.0.21`, `libloading 0.9.0`, `serde 1.0.229`, `serde_json 1.0.154` | Removed `cfg-if` (replaced by `std::cfg_select!`) | [01_rust_toolchain_and_simd.md](01_rust_toolchain_and_simd.md) |
| **Python, PyO3 & Maturin** | `Py 3.10` - `3.15.0`, `pyo3 0.24` - `0.30.1` | `Python 3.15.0`, `pyo3 0.30.1`, `maturin 1.15.0` | Removed `extension-module`, enabled `abi3t`, migrated to `py.detach` & declarative `#[pymodule] mod` | [02_python_pyo3_and_maturin.md](02_python_pyo3_and_maturin.md) |
| **PyTorch & Triton JIT** | `torch 2.4` - `2.14.1`, `triton 3.0` - `3.8.0` | `torch 2.14.1`, `triton 3.8.0` | Integrated `torch.accelerator`, fixed `tl.sqrt` in JIT AST, supported 1-2 trailing scratch pointers | [03_pytorch_and_triton_jit.md](03_pytorch_and_triton_jit.md) |
| **CUDA & AMD ROCm / HIP** | `CUDA 12.0` - `13.x`, `ROCm 5.7` - `7.x / TheRock 10.1.0` | `ROCm 7.x / TheRock 10.1.0` | Probing `amdhip64_7.dll`, `R0600` ABI, Wave32/Wave64 resolution for CDNA4 & RDNA4, SM Rubin/Blackwell | [04_cuda_and_rocm_hip_drivers.md](04_cuda_and_rocm_hip_drivers.md) |
| **CubeCL Portable GPU** | `cubecl 0.11.x` | `cubecl 0.11.4` | Optional cross-vendor GPU backend (`wgpu`, Vulkan, Metal, CUDA, HIP, CPU) replacing `ash` Vulkan | [05_cubecl_portable_gpu.md](05_cubecl_portable_gpu.md) |
| **Data Engine Crates** | `arrow/parquet 60.2.0` | `arrow-* 60.2.0`, `parquet 60.2.0` (`arrow`, `zstd`), `xxhash-rust 0.8.19`, `hashbrown 0.17.1`, `crc32fast 1.5.2`, `parking_lot 0.12.5` | Mitigated `RUSTSEC-2026-0595` & `RUSTSEC-2026-0442`, trimmed unused Parquet codecs, enabled Arrow C-Data Interface (`ffi`), 64-bit `LargeUtf8`/`LargeBinary`, `XXH3-128` WAL | [06_dataloader_crates.md](06_dataloader_crates.md) |

---

## File Navigation Map

| File Path | Role Summary | Status |
| :--- | :--- | :--- |
| `docs/dependencies/README.md` | Index and version summary table across workspace dependencies. | Active |
| `docs/dependencies/01_rust_toolchain_and_simd.md` | Compatibility notes for Rust 1.95-1.99.1, SIMD intrinsics, and core crates. | Active |
| `docs/dependencies/02_python_pyo3_and_maturin.md` | Compatibility notes for Python 3.10-3.15, PyO3 0.24-0.30.1, `abi3t`, and Maturin. | Active |
| `docs/dependencies/03_pytorch_and_triton_jit.md` | Compatibility notes for PyTorch 2.4-2.14.1 and Triton compiler 3.0-3.8.0. | Active |
| `docs/dependencies/04_cuda_and_rocm_hip_drivers.md` | Compatibility notes for CUDA drivers and ROCm 5-7 / TheRock 10.1 on Windows and Linux. | Active |
| `docs/dependencies/05_cubecl_portable_gpu.md` | Compatibility notes for optional CubeCL `0.11.4` portable GPU compute kernels. | Active |
| `docs/dependencies/06_dataloader_crates.md` | Compatibility notes for `arrow-*`/`parquet 60.2.0`, `xxhash-rust`, `hashbrown`, `crc32fast`, and `parking_lot`. | Active |
