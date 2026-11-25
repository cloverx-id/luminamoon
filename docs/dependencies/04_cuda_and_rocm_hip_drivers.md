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

### C. New AMD GPU Architectures, Wavefront Sizes (`Wave32` vs `Wave64`), & Generic Targets
- **What changed**:
  - **CDNA 4 (`gfx950` - Instinct MI350X/MI355X/MI350P)**: Uses a wavefront size of `64` (*Wave64*), matching CDNA 3 (`gfx942`).
  - **RDNA 3.5 (`gfx1150`, `gfx1151` - Ryzen AI 300 Strix Point / Ryzen AI Max+ Strix Halo)** and **RDNA 4 (`gfx1200`, `gfx1201`, `gfx1250` - Radeon RX 9060/9070 XT, Radeon AI PRO R9700)**: Use a wavefront size of `32` (*Wave32*).
  - **ROCm Generic Targets**: LLVM/ROCm introduced portable ELF machine identifiers (`gfx9-4-generic`, `gfx11-generic`, `gfx12-generic`) so one `.hsaco` code object can load across sub-variants within a generation without strict exact-stepping rejection.
- **Adaptation & Current Boundary in `hip_driver.rs`**:
  - `wavefront_size_for_arch(arch)` returns `32` for all `gfx10*`, `gfx11*`, and `gfx12*` families, and `64` for CDNA (`gfx9*`).
  - `resolve_amd_gfx_arch(gcn_arch_raw, available)` maps `gfx940`/`gfx941`/`gfx950` to `gfx942` (Wave64) and `gfx11xx`/`gfx12xx` to `gfx1100` (Wave32) when dedicated sub-variant binaries are not bundled.
  - **ELF `EF_AMDGPU_MACH` Compatibility Note**: ROCm's `hipModuleLoadData` enforces strict `EF_AMDGPU_MACH` checks; loading a `gfx1100` `.hsaco` on RDNA 4 (`gfx1200`/`gfx1201`) or a `gfx942` `.hsaco` on CDNA 4 (`gfx950`) can fail with error `209`. `TritonAotError::Hip(209)` is marked `is_fallback_eligible()`, so `luminav-dispatch` falls back to CubeCL or CPU SIMD (while PyTorch Triton JIT compiles natively for the active GPU).

---

## 4.3 NVIDIA CUDA Architecture Changelog (`CUDA 12.x` - `13.5` & Triton `3.6` - `3.8`)

### A. CUDA 13 Floor, Blackwell Variants, Rubin (`sm_107`), & Family Suffixes
- **What changed**:
    - **CUDA 13.0+ Baseline Floor**: Offline compilation for Maxwell (`sm_5x`), Pascal (`sm_6x`), and Volta (`sm_70`) was dropped in CUDA 13.0; **`sm_75` (Turing)** is the minimum supported offline target, matching `SUPPORTED_CUDA_ARCHS` in `cuda_driver.rs`.
    - **Blackwell Data Center**: `sm_100` (B100/B200/GB200) and `sm_103` (B300/GB300 Blackwell Ultra).
    - **NVIDIA Rubin**: `sm_107` (Vera Rubin NVL72; preview in CUDA 13.4, GA in CUDA 13.5 with R615+ driver, PTX ISA 9.1+).
    - **Blackwell Embedded / Automotive**: `sm_110` (Jetson Thor T5000/T4000).
    - **Blackwell Client / Workstation & DGX Spark**: `sm_120` (RTX 50-series / RTX PRO Blackwell) and `sm_121` (GB10 Superchip).
    - **Family (`.f`) & Arch (`.a`) Suffixes (CUDA 12.9+ / 13.x, PTX ISA 9.0–9.5)**: `sm_100f` / `sm_120f` target family-compatible execution across minor steppings, while `sm_100a` / `sm_120a` enable architecture-specific non-forward-compatible instructions.
    - Critical distinction: `sm_100`/`sm_103` binaries use Tensor Memory (`TMEM`) and Cluster Launch Control instructions that are hardware-incompatible with client `sm_120`/`sm_121` GPUs.
- **Adaptation & Current Boundary in `optimizer/crates/luminav-triton-aot/src/cuda_driver.rs`**:
    - `resolve_cuda_cubin_arch(major, minor, available)` matches compute capabilities strictly within the same `major` family (`c_maj == major && c_min <= minor`):
    - `sm_107` maps to `sm_103` (major 10), and `sm_121` maps to `sm_120` (major 12) without ever loading an `sm_100`/`sm_103` binary on client silicon.
    - `sm_110` (major 11, Jetson Thor) currently returns `UnsupportedArch(11, 0)` and falls back to CubeCL/CPU SIMD in `ExecutionMode::Auto`.
