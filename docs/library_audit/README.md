# Library Version & Changelog Audit Documentation (October 2026)

This directory documents the version-by-version changelog audit, intermediate API/ABI deprecations, security advisory mitigations, and source-code adaptations to the latest stable releases across all dependencies of the **LuminaMoon (`LuminaV v1.0.0` & `luminamoon-dataloader v1.0.0`)** workspace.

---

## Ecosystem Summary Matrix

| Component / Library | Audited Version Range | Latest Stable Version | Adaptation Status | Detailed Document |
| :--- | :--- | :--- | :--- | :--- |
| **Rust Toolchain & SIMD** | `1.85.0` - `1.99.1` | `1.99.1` (`edition = "2024"`) | Adopted `std::cfg_select!`, isolated `Scalar` vs `Avx2` dispatch paths | [01_rust_toolchain_and_simd.md](01_rust_toolchain_and_simd.md) |
| **Supporting Rust Crates** | `2024` - `2026` | `rayon 1.12.0`, `thiserror 2.0.21`, `libloading 0.9.0`, `serde 1.0.229`, `serde_json 1.0.154` | Removed obsolete external crates (`cfg-if` replaced by `std::cfg_select!`) | [01_rust_toolchain_and_simd.md](01_rust_toolchain_and_simd.md) |
| **Python, PyO3 & Maturin** | `Py 3.10` - `3.15.0`, `pyo3 0.24` - `0.30.1` | `Python 3.15.0`, `pyo3 0.30.1`, `maturin 1.15.0` | Removed `extension-module`, enabled `abi3t`, migrated to `py.detach` & declarative `#[pymodule] mod` | [02_python_pyo3_and_maturin.md](02_python_pyo3_and_maturin.md) |
| **PyTorch & Triton JIT** | `torch 2.4` - `2.14.1`, `triton 3.0` - `3.8.0` | `torch 2.14.1`, `triton 3.8.0` | Integrated `torch.accelerator`, fixed `tl.sqrt` in JIT AST, supported 1-2 trailing scratch pointers | [03_pytorch_and_triton_jit.md](03_pytorch_and_triton_jit.md) |
| **CUDA & AMD ROCm / HIP** | `CUDA 12.0` - `13.x`, `ROCm 5.7` - `7.x / TheRock 10.1.0` | `ROCm 7.x / TheRock 10.1.0` | Probing `amdhip64_7.dll`, `R0600` ABI, Wave32/Wave64 resolution for CDNA4 & RDNA4, SM Rubin/Blackwell | [04_cuda_and_rocm_hip_drivers.md](04_cuda_and_rocm_hip_drivers.md) |
| **Vulkan Compute & `ash`** | `Vulkan 1.2` - `1.4`, `ash 0.37` - `0.38.0` | `Vulkan 1.4`, `ash 0.38.0` | Dynamic `1.4`/`1.3` instance negotiation, `VkPhysicalDeviceVulkan13Properties` subgroup bounds validation | [05_vulkan_and_ash.md](05_vulkan_and_ash.md) |
| **Universal DataLoader & Compression** | `2025` - `2026` | `arrow-ipc 60.0.0`, `half 2.7.1`, `flate2 1.1.10`, `zstd 0.14.3`, `lz4_flex 0.14.0` | Mitigated `CVE-2026-32829` (`lz4_flex 0.14.0`), `DCtx` soundness fixes (`zstd 0.14.3`), `runtime_detection` (`flate2 1.1.10`), Arrow IPC/Feather/WebDataset TAR | [06_universal_dataloader_formats_and_compression.md](06_universal_dataloader_formats_and_compression.md) |

---

## File Navigation Map

| File Path | Role Summary | Status |
| :--- | :--- | :--- |
| `docs/library_audit/README.md` | Master index and version summary matrix across all workspace libraries. | Active |
| `docs/library_audit/01_rust_toolchain_and_simd.md` | Changelog audit for Rust 1.85-1.99.1, x86_64 SIMD intrinsics, and core crates. | Active |
| `docs/library_audit/02_python_pyo3_and_maturin.md` | Changelog audit for Python 3.10-3.15, PyO3 0.24-0.30.1, ABI3t, and Maturin. | Active |
| `docs/library_audit/03_pytorch_and_triton_jit.md` | Changelog audit for PyTorch 2.4-2.14.1 and Triton compiler 3.0-3.8.0. | Active |
| `docs/library_audit/04_cuda_and_rocm_hip_drivers.md` | Changelog audit for CUDA drivers and ROCm 5-7 / TheRock 10.1 on Windows and Linux. | Active |
| `docs/library_audit/05_vulkan_and_ash.md` | Changelog audit for Vulkan 1.3/1.4 core promotions and `ash 0.38.0` bindings. | Active |
| `docs/library_audit/06_universal_dataloader_formats_and_compression.md` | Version-by-version changelog audit (2025-2026) for `arrow-ipc 60.0.0`, `half 2.7.1`, `flate2 1.1.10`, `zstd 0.14.3`, and `lz4_flex 0.14.0`. | Active |
