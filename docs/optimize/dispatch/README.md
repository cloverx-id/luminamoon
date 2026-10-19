# `luminav-dispatch`: Hardware Capability Probing & Routing Heuristics

`luminav-dispatch` coordinates runtime hardware capability auditing and routes parameter update workloads to the optimal compute backend based on tensor size, architecture availability, and user preferences.

[Versi Bahasa Indonesia (README.id.md)](README.id.md)

---

## 1. Hardware Detection Engine (`src/detector.rs`)

The static probe method `LuminaV.detect_hardware()` queries host capabilities without throwing exceptions:

```mermaid
flowchart TD
    Detect["Hardware Capability Probe"] --> CPU["CPU Feature Scan"]
    Detect --> Vulkan["Vulkan Instance Query"]
    Detect --> Triton["GPU Driver Dynamic Check"]

    CPU -->|"AVX-512"| AVX512["AVX-512 Backend"]
    CPU -->|"AVX2"| AVX2["AVX2 Backend"]
    CPU -->|"Other / ARM"| Scalar["Scalar Backend"]

    Vulkan -->|"Available & UMA"| VK_Ready["Vulkan UMA Enabled"]
    Vulkan -->|"Unavailable"| VK_Off["Vulkan Disabled"]

    Triton -->|"CUDA / HIP Found"| TR_Ready["Triton Driver Active"]
    Triton -->|"Not Found"| TR_Off["Triton Disabled"]
```

---

## 2. Dynamic Routing Heuristic (`src/router.rs`)

When `execution = "auto"`, execution paths are selected dynamically:

```mermaid
flowchart TD
    Tensor["Input Tensor (Length N)"] --> ModeCheck{"Explicit Mode Set?"}

    ModeCheck -->|"Yes"| ManualExec["Dispatch to Specified Backend"]
    ModeCheck -->|"Auto"| SizeCheck{"N &ge; 16,384 & Vulkan Available?"}

    SizeCheck -->|"Yes"| ExecVulkan["Vulkan UMA Backend"]
    SizeCheck -->|"No"| SIMDCheck{"SIMD Supported?"}

    SIMDCheck -->|"AVX-512 / AVX2"| ExecSIMD["Parallel CPU SIMD"]
    SIMDCheck -->|"Fallback"| ExecScalar["Reference Scalar Engine"]
```

### The 16,384 Threshold Heuristic (GPU_DISPATCH_THRESHOLD)
For small and medium tensors ($N < 16,384$ elements, defined by `GPU_DISPATCH_THRESHOLD = 16_384`), GPU command buffer submission and pipeline synchronization latencies outweigh kernel compute speedups. In this regime, multi-threaded CPU SIMD executes with direct cache locality. For large tensors ($N \ge 16,384$), execution shifts to the GPU compute pipeline when available.
