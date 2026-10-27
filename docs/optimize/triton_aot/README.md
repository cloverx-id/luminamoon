# `luminav-triton-aot`: Ahead-of-Time Kernels & Dynamic GPU Drivers

`luminav-triton-aot` provides low-level dynamic GPU driver bindings (NVIDIA CUDA and AMD ROCm HIP) to execute Ahead-of-Time (AOT) compiled Triton kernels with a standalone C-FFI interface.

---

## 1. Dynamic Driver Probing & Architecture Resolution

To maintain zero mandatory linked runtime dependencies, the crate probes driver shared libraries at runtime using `libloading`:

```mermaid
flowchart LR
    Probe["libloading Runtime Probe"] --> CUDA["NVIDIA CUDA Driver (nvcuda: SM75–SM120 .cubin)"]
    Probe --> ROCm["AMD ROCm HIP Driver (amdhip64: RDNA3/CDNA .hsaco)"]
    Probe --> Fallback["No GPU Driver: Graceful Fallback to CPU SIMD"]
    CUDA --> Launch["ELF64 Header Check & C-ABI Kernel Launch"]
    ROCm --> Launch
```

---

## 2. AOT Manifest & Artifact Inspection (`src/manifest.rs`)

AOT kernels are pre-compiled into target binary artifacts (.cubin or .hsaco) registered in `manifest.json`:

- **ELF64 Magic Header**: Pre-validates binary headers against synthetic stubs (`CUDA_STUB_MAGIC = b"// LUMINAV_TRITON_AOT_CUBIN"`, `ROCM_STUB_MAGIC = b"; LUMINAV_TRITON_AOT_HSACO"`).
- **ABI Argument Packing**: Automatically packs pointers and scalar config fields (`step`, `lr`, `beta1`, `beta2`, `eps`, `bound_ratio`) into strict C-compatible memory layouts expected by Triton-generated entry points.

---

## 3. Standalone C-API Interface (`src/c_api.rs`)

The crate exposes a standalone C-API declared in [`include/luminav_triton_standalone.h`](https://github.com/cloverx-id/luminamoon/blob/main/optimizer/include/luminav_triton_standalone.h):

```c
typedef struct LuminaVTritonEngine LuminaVTritonEngine;

LuminaVTritonEngine* luminav_triton_engine_create(void);
void luminav_triton_engine_destroy(LuminaVTritonEngine* engine);
const char* luminav_triton_get_platform(const LuminaVTritonEngine* engine);
const char* luminav_triton_get_arch(const LuminaVTritonEngine* engine);

int luminav_triton_step_f32(
    LuminaVTritonEngine* engine,
    float* param,
    const float* grad,
    float* exp_avg,
    float* exp_avg_sq,
    float* master_param,
    float* scratch,
    size_t num_elements,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float tau,
    float lam,
    int buffer_mode,
    int cautious,
    int bound_mode,
    float bound_ratio,
    uint64_t step,
    uint32_t seed
);
```

This allows native C/C++ high-performance training engines or embedded inference harnesses to drive LuminaV Triton kernels without a Python interpreter.
