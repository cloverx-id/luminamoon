# `luminav-dispatch`: Hardware Capability Probing & Routing Heuristics

`luminav-dispatch` coordinates runtime hardware capability auditing and routes parameter update workloads to the optimal compute backend based on tensor size, architecture availability, and user preferences.

---

## 1. Hardware Detection Engine (`src/detector.rs`)

The static probe method `LuminaV.detect_hardware()` queries host capabilities without throwing exceptions:

```mermaid
flowchart LR
    Probe["LuminaV.detect_hardware()"] --> CPU["CPU ISA Probe"]
    Probe --> VK["Vulkan 1.3/1.4 Probe"]
    Probe --> GPU["GPU Driver Probe (libloading)"]
    CPU --> ISA["AVX-512 / AVX2 / Scalar"]
    VK --> UMA["UMA Host-Visible Queue & Wave32"]
    GPU --> Driver["CUDA (nvcuda) / ROCm (amdhip64)"]
```

---

## 2. Dynamic Routing Heuristic (`src/router.rs`)

When `execution = "auto"`, execution paths are selected dynamically:

```mermaid
flowchart TD
    Input["Input Tensor (Length N)"] --> Router["Execution Mode Router"]
    Router --> Manual["Explicit Mode: Selected Target Backend"]
    Router --> Vulkan["Auto (N >= 16,384 & UMA): Vulkan 1.3/1.4 Backend"]
    Router --> CPU["Auto (N < 16,384): CPU Dispatch"]
    CPU --> SIMD["Rayon Parallel CPU SIMD (AVX-512 / AVX2)"]
    CPU --> Scalar["Reference Scalar Core (Portable Fallback)"]
```

### The 16,384 Threshold Heuristic (GPU_DISPATCH_THRESHOLD)
For small and medium tensors ($N < 16,384$ elements, defined by `GPU_DISPATCH_THRESHOLD = 16_384`), GPU command buffer submission and pipeline synchronization latencies outweigh kernel compute speedups. In this regime, multi-threaded CPU SIMD executes with direct cache locality. For large tensors ($N \ge 16,384$), execution shifts to the GPU compute pipeline when available.
