# `luminav-dispatch`: Hardware Capability Probing & Routing Heuristics

`luminav-dispatch` coordinates runtime hardware capability detection and routes parameter updates to the best-matching compute backend based on tensor size, architecture availability, and user preferences.

---

## 1. Hardware Detection Engine (`src/detector.rs`)

The static probe method `LuminaV.detect_hardware()` queries host capabilities without throwing exceptions:

```mermaid
flowchart LR
    Probe["LuminaV.detect_hardware()"] --> CPU["CPU ISA Probe"]
    Probe --> CubeCL["CubeCL Backend Probe (Optional Feature)"]
    Probe --> GPU["GPU Driver Probe (libloading)"]
    CPU --> ISA["AVX-512 / AVX2 / NEON / Scalar"]
    CubeCL --> Portable["WGPU / Vulkan / Metal / CUDA / HIP / CPU"]
    GPU --> Driver["CUDA (nvcuda) / ROCm (amdhip64)"]
```

---

## 2. Dynamic Routing Heuristic (`src/router.rs`)

When `execution = "auto"`, execution paths are selected dynamically:

```mermaid
flowchart TD
    Input["Input Tensor (Length N)"] --> Router["Execution Mode Router"]
    Router --> Manual["Explicit Mode: Selected Target Backend"]
    Router --> GPU["Auto (N >= 16,384): Triton AOT -> CubeCL (if enabled)"]
    Router --> CPU["Auto (N < 16,384): CPU Dispatch"]
    CPU --> SIMD["Rayon Parallel CPU SIMD (AVX-512 / AVX2 / NEON)"]
    CPU --> Scalar["Reference Scalar Core (Portable Fallback)"]
```

### The 16,384 Threshold Heuristic (`GPU_DISPATCH_THRESHOLD`)
For small and medium tensors ($N < 16{,}384$ elements, defined by `GPU_DISPATCH_THRESHOLD = 16_384`), GPU command submission and synchronization latencies outweigh compute speedups, so multi-threaded CPU SIMD runs directly. For large tensors ($N \ge 16{,}384$), `Auto` tries `TritonAotEngine` first for all four buffer modes (`Dual`, `TdClm`, `T1`, `T2`), then tries `CubeClLuminaEngine` if compiled with a `cubecl-*` backend feature (e.g., `cubecl-wgpu`), and falls back to multi-threaded CPU SIMD (`luminav_simd::step`) if no GPU backend matches.
