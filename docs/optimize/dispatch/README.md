# `luminav-dispatch`: Hardware Capability Probing & Routing Heuristics

`luminav-dispatch` coordinates runtime hardware capability auditing and routes parameter update workloads to the optimal compute backend based on tensor size, architecture availability, and user preferences.

[Versi Bahasa Indonesia (README.id.md)](README.id.md)

---

## 1. Hardware Detection Engine (`src/detector.rs`)

The static probe method `LuminaV.detect_hardware()` queries host capabilities without throwing exceptions:

```mermaid
flowchart TD
    Detect["detect_hardware()"] --> CPU["x86_64 Feature Probe"]
    Detect --> VK["Vulkan Instance Query"]
    Detect --> TR["Triton AOT Driver Check"]

    CPU -->|"avx512f, dq, bw, vl, fma"| AVX512["SimdBackend::Avx512"]
    CPU -->|"avx2, fma"| AVX2["SimdBackend::Avx2"]
    CPU -->|"other / ARM"| Scalar["SimdBackend::Scalar"]

    VK -->|"Vulkan 1.3/1.4 + UMA"| VK_OK["vulkan_available = true<br/>wave32_supported = true/false"]
    TR -->|"libloading CUDA / HIP"| TR_OK["triton_aot_available = true/false"]
```

---

## 2. Dynamic Routing Heuristic (`src/router.rs`)

When `execution = "auto"`, execution paths are selected dynamically:

```mermaid
flowchart TD
    Input["Tensor Parameter (numel = N)"] --> CheckExplicit{"Explicit Execution Mode?"}
    
    CheckExplicit -->|"Vulkan / SIMD / Scalar"| ExecSelected["Dispatch to Selected Engine"]
    CheckExplicit -->|"Auto"| CheckGPU{"N >= 16,384 & Vulkan OK?"}

    CheckGPU -->|"Yes"| DispatchVK["Dispatch to luminav-vulkan (Wave32 UMA)"]
    CheckGPU -->|"No"| CheckSIMD{"x86_64 SIMD Available?"}

    CheckSIMD -->|"AVX-512 / AVX2"| DispatchSIMD["Dispatch to luminav-simd (Rayon Parallel)"]
    CheckSIMD -->|"Fallback"| DispatchScalar["Dispatch to luminav-core (Scalar Engine)"]
```

### The 16,384 Threshold Heuristic (GPU_DISPATCH_THRESHOLD)
For small and medium tensors ($N < 16,384$ elements, defined by `GPU_DISPATCH_THRESHOLD = 16_384`), GPU command buffer submission and pipeline synchronization latencies outweigh kernel compute speedups. In this regime, multi-threaded CPU SIMD executes with direct cache locality. For large tensors ($N \ge 16,384$), execution shifts to the GPU compute pipeline when available.
