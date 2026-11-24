# `luminav-cubecl`: Portable GPU Compute Backend

`luminav-cubecl` provides optional cross-vendor GPU compute acceleration built on [CubeCL](https://github.com/tracel-ai/cubecl) (`0.11.4`), targeting `wgpu`, Vulkan, Metal, CUDA, ROCm HIP, and CPU runtimes from a single set of Rust kernels.

---

## 1. Feature Flags & Build Configuration

CubeCL is opt-in so default `luminamoon` wheels keep a minimal dependency footprint. Enable one or more runtime targets in `optimizer/Cargo.toml`:

| Feature | Underlying `luminav-cubecl` Target | Intended Platform |
| :--- | :--- | :--- |
| `cubecl-wgpu` | `cubecl/wgpu` | Cross-platform WebGPU / Vulkan / Metal / DX12 |
| `cubecl-vulkan` | `cubecl/vulkan` | Native Vulkan compute (AMD, Intel, NVIDIA) |
| `cubecl-metal` | `cubecl/metal` | Apple Silicon / macOS Metal |
| `cubecl-cuda` | `cubecl/cuda` | NVIDIA CUDA devices |
| `cubecl-hip` | `cubecl/hip` | AMD ROCm HIP devices |
| `cubecl-cpu` | `cubecl/cpu` | Portable CPU JIT runtime |

```bash
maturin develop --release --features cubecl-wgpu
```

---

## 2. Supported Kernels & Parity

1. **Optimizer Kernels (`src/engine_step.rs`, `src/kernels/`)**:
   - Executes all four `LuminaV` buffer modes: Dual-Buffer CLM (`buffer=2`), Single-Buffer TD-CLM (`buffer=1`), and Triangle Family (`buffer="T1"`, `"T2"`).
   - Supports Cautious Weight Decay (`cautious_weight_decay`), Rational-Tempered Variance (`variance_type="rtv"`), configurable `bound_floor`, and MurmurHash3 + 3D Weyl-Kronecker stochastic rounding (`FP32`, `BF16`, `FP16`).
2. **Ra-Family Neural Network Kernels (`src/engine_nn.rs`)**:
   - Accelerates FP32 forward ops for `RaNorm`, `ra_tanh`, `ra_gelu`, `ra_swiglu`, `ra_glu`, and `ra_softmax` when `luminamoon.nn.is_cubecl_available()` is `True`.
   - Validates affine `weight`/`bias` lengths and caps 1D dispatch grids at `MAX_GRID_X = 65_535` (falling back to CPU SIMD / eager when limits are exceeded).
3. **Dispatch Integration (`luminav-dispatch`)**:
   - Selected when `execution="cubecl"` is requested explicitly, or under `execution="auto"` for tensors with $N \ge 16{,}384$ when Triton AOT is unavailable and a CubeCL backend is compiled in.
