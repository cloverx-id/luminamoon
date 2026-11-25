# `luminav-triton-aot`: Ahead-of-Time Kernels & Dynamic GPU Drivers

`luminav-triton-aot` provides low-level dynamic GPU driver bindings (NVIDIA CUDA and AMD ROCm HIP) to execute Ahead-of-Time (AOT) compiled Triton kernels with a standalone C-FFI interface.

---

## 1. Dynamic Driver Probing & Architecture Resolution

To avoid mandatory linked GPU driver dependencies at build time, the crate probes driver shared libraries at runtime using `libloading`:

```mermaid
flowchart LR
    Probe["libloading Runtime Probe"] --> CUDA["NVIDIA CUDA Driver (nvcuda: sm_75..sm_120 .cubin)"]
    Probe --> ROCm["AMD ROCm HIP Driver (amdhip64: gfx908..gfx1103 .hsaco)"]
    Probe --> Fallback["No GPU Driver: Fallback to CubeCL / CPU SIMD"]
    CUDA --> Launch["ELF64 & Constexpr Check -> Kernel Launch"]
    ROCm --> Launch
```

---

## 2. AOT Manifest & Artifact Inspection (`src/manifest.rs` & `src/step.rs`)

182 GPU ELF64 binaries (14 kernels across 8 NVIDIA CUDA targets `sm_75..sm_120` and 5 AMD ROCm targets `gfx908`, `gfx90a`, `gfx942`, `gfx1100`, `gfx1103`) are pre-compiled by `optimizer/tools/aot_compiler/compile_triton_aot.py` and indexed in `artifacts/manifest.json` (`schema_version = "2.0.0"`, `math_version = "2.0.0"`):

- **ELF64 & Constexpr Validation**: Validates binary headers (`\x7fELF`), guards against synthetic stubs (`is_synthetic_stub_binary`), caches loaded modules per `"{kernel_name}::{symbol_name}::{arch}"` (preventing specialization cache collisions), and verifies runtime `constexpr` flags (`bound_mode`, `dtype_mode`, `state_dtype_mode`, `use_sr`, `cautious`, `has_master`, `use_rtv`, `cautious_wd`, `is_triangle`) via `supports_config` before launch. Unsupported runtime configurations or ISA load mismatches (`is_fallback_eligible()`) fall back cleanly to CubeCL or Rayon CPU SIMD.
- **Reusable Staging Pool & Direct Device-Pointer Execution (Zero Staging Copies)**: `DeviceStagingPool` (`src/buffer_pool.rs`) reuses 6 VRAM buffers across host-buffer steps (allocating only on capacity growth and freeing on `Drop`), while `TritonAotEngine::step_device_ptrs` executes directly on device-resident CUDA/HIP pointers with zero HtoD/DtoH staging copies.
- **Triton 3.9 ABI Packing**: Packs device pointers, scalar config fields (`step_i32`, `seed_i32`, `lr`, `beta1`, `beta2`, `eps`, `weight_decay`, `c1`/`c2` incorporating `tau`, `lam`, `clamp_min`, `ema_decay`, `bound_ratio`, `bound_floor`, `variance_temper`, `inv_sqrt_bc2_prev`), and `num_trailing_scratch_ptrs = 2` null trailing scratch pointers expected by Triton 3.9.0 compiled entry points across both 2-pass (`Dual`, `TdClm`, `T1`, `T2`) and fused single-pass + post-step EMA paths.

---

## 3. Standalone C-API Interface (`src/c_api.rs`)

The crate exposes a standalone C-API declared in [`include/luminav_triton_standalone.h`](https://github.com/cloverx-id/luminamoon/blob/main/optimizer/include/luminav_triton_standalone.h):

```c
typedef struct LuminaVTritonEngine LuminaVTritonEngine;

LuminaVTritonEngine* luminav_triton_engine_create(void);
void luminav_triton_engine_destroy(LuminaVTritonEngine* engine);
const char* luminav_triton_get_platform(const LuminaVTritonEngine* engine);
const char* luminav_triton_get_arch(const LuminaVTritonEngine* engine);
int luminav_triton_supports_config(
    const LuminaVTritonEngine* engine,
    int buffer_mode,
    int cautious,
    int bound_mode,
    int has_master
);

int luminav_triton_step_f32(
    LuminaVTritonEngine* engine,
    float* param, const float* grad, float* exp_avg, float* exp_avg_sq,
    float* master_param, float* scratch, size_t num_elements,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float tau, float lam, int buffer_mode, int cautious, int bound_mode,
    float bound_ratio, uint64_t step, uint32_t seed
);

int luminav_triton_step_f32_ex(
    LuminaVTritonEngine* engine,
    float* param, const float* grad, float* exp_avg, float* exp_avg_sq,
    float* master_param, float* scratch, size_t num_elements,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float tau, float lam, int buffer_mode, int cautious, int bound_mode,
    float bound_ratio, uint64_t step, uint32_t seed,
    size_t param_index, int allow_cpu_fallback
);

int luminav_triton_step_f32_device(
    LuminaVTritonEngine* engine,
    uint64_t d_param, uint64_t d_grad, uint64_t d_exp_avg, uint64_t d_exp_avg_sq,
    uint64_t d_master_param, uint64_t d_scratch, size_t num_elements,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float tau, float lam, int buffer_mode, int cautious, int bound_mode,
    float bound_ratio, uint64_t step, uint32_t seed, size_t param_index
);
```

This allows native C/C++ training engines or embedded inference harnesses to drive LuminaV Triton kernels on either host buffers (`DeviceStagingPool` + optional `luminav_core` fallback) or device-resident VRAM pointers without a Python interpreter.
