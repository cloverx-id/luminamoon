# 4. Discrete GPU Drivers: NVIDIA CUDA and AMD ROCm / HIP (TheRock)

## 4.1 Latest Stable Versions
- **NVIDIA CUDA Driver API**: `CUDA 12.x / 13.x` (`nvcuda.dll` / `libcuda.so.1`)
- **AMD ROCm / HIP Runtime**: `ROCm 7.x` / `ROCm Core SDK (TheRock) 10.1.0` (`therock-10.1`)

---

## 4.2 AMD ROCm & HIP Changelog (`ROCm 5.7` - `ROCm 7.x / TheRock 10.1.0`)

### A. Major-Versioned Dynamic Library Filenames (`.dll` / `.so`)
- **What changed**:
  - In **ROCm 5.x**, the Windows HIP runtime library was named `amdhip64.dll`.
  - Starting in **ROCm 6.1.2+**, AMD renamed the Windows DLL to `amdhip64_6.dll` so multiple ROCm runtime versions can coexist on a single machine.
  - In **ROCm 7.x** and the modular **TheRock 10.x** distribution (`_rocm_sdk_core`), the Windows DLL is named `amdhip64_7.dll`, while Linux uses `libamdhip64.so.7` (alongside `libamdhip64.so` and `libamdhip64.so.6` symlinks).
- **Adaptation in `optimizer/crates/luminav-triton-aot/src/hip_driver.rs`**:
  `candidate_hip_lib_names()` and `HipDriver::probe()` probe candidate dynamic libraries from newest to oldest:
  - **Windows**: `["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"]`
  - **Linux / Unix**: `["libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so", "libamdhip64.so.5"]`

### B. ABI Symbol Evolution: `hipGetDevicePropertiesR0600`
- **What changed**: Starting in ROCm 6.0, the C struct `hipDeviceProp_t` changed field layout, introducing the versioned symbol `hipGetDevicePropertiesR0600` (while retaining `hipGetDeviceProperties` for legacy compatibility).
- **Adaptation in `hip_driver.rs`**:
  - `HipDriver::probe()` resolves `hipGetDevicePropertiesR0600` first and falls back to `hipGetDeviceProperties`.
  - Because the byte offset of `gcnArchName` differs between ROCm 5 and `R0600` ABIs, `extract_gcn_arch_from_prop_bytes` scans the 4096-byte `HipDevicePropBuffer` directly for null-terminated `"gfx..."` strings, remaining resilient to struct offset shifts across ROCm 5, 6, 7, and 10.

### C. New AMD GPU Architectures & Wavefront Sizes (`Wave32` vs `Wave64`)
- **What changed**:
  - **CDNA 4 (`gfx950` - Instinct MI350X/MI355X)**: Uses a wavefront size of `64` (*Wave64*), matching CDNA 3 (`gfx942`).
  - **RDNA 3.5 (`gfx1150`, `gfx1151` - Ryzen AI 300 Strix Point / Ryzen AI Max+ Strix Halo)** and **RDNA 4 (`gfx1200`, `gfx1201`, `gfx1250` - Radeon RX 9070 XT)**: Use a wavefront size of `32` (*Wave32*).
- **Adaptation in `hip_driver.rs`**:
  - `wavefront_size_for_arch(arch)` returns `32` for all `gfx10*`, `gfx11*`, and `gfx12*` families, and `64` for CDNA (`gfx9*`).
  - `resolve_amd_gfx_arch(gcn_arch_raw, available)` maps `gfx940`/`gfx941`/`gfx950` variants to `gfx942` (Wave64) binaries and `gfx11xx`/`gfx12xx` variants to `gfx1100` (Wave32) binaries when dedicated sub-variant binaries are not bundled separately.

---

## 4.3 NVIDIA CUDA Architecture Changelog (`CUDA 12.x` - `13.x` & Triton `3.6` - `3.8`)

### A. Data Center vs. Client Blackwell & Rubin Compute Capabilities
- **What changed**:
  - **Blackwell Data Center**: `sm_100` (B100/B200/GB200) and `sm_103` (B300/GB300 Blackwell Ultra).
  - **NVIDIA Rubin**: `sm_107` (supported in Triton 3.7/3.8).
  - **Blackwell Client / Workstation & DGX Spark**: `sm_120` (RTX 5090 / RTX PRO 6000 Blackwell) and `sm_121` (GB10 Superchip).
  - Critical distinction: `sm_100`/`sm_103` binaries utilize Tensor Memory (`TMEM`) and Cluster Launch Control instructions that are hardware-incompatible with client `sm_120`/`sm_121` GPUs.
- **Adaptation in `optimizer/crates/luminav-triton-aot/src/cuda_driver.rs`**:
  `resolve_cuda_cubin_arch(major, minor, available)` matches compute capabilities strictly within the same `major` family (`c_maj == major && c_min <= minor`):
  - `sm_107` safely maps to `sm_103` (major 10).
  - `sm_121` safely maps to `sm_120` (major 12) and never erroneously loads an `sm_100`/`sm_103` binary.
