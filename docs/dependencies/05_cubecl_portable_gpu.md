# 5. Portable GPU Compute (`cubecl`)

## 5.1 Workspace Dependency Version
- **`cubecl`**: `0.11.4` (`default-features = false, features = ["std"]`, optional crate dependency in `optimizer/crates/luminav-cubecl/Cargo.toml`)

---

## 5.2 Design Role in LuminaMoon

In `v2.0.0`, `luminav-cubecl` replaced the legacy `ash` / GLSL / SPIR-V Vulkan crate (`luminav-vulkan`) to share one set of portable GPU kernels across vendors without maintaining separate shader toolchains:

- **Single Rust Kernel Source**: Optimizer steps (`Dual`, `TdClm`, `T1`, `T2`) and five `luminamoon.nn` ops (`RaNorm`, `ra_tanh`, `ra_swiglu`, `ra_glu`, `ra_softmax`) are written once as CubeCL Rust kernels and compiled for `wgpu`, `vulkan`, `metal`, `cuda`, `hip`, or `cpu` depending on enabled Cargo features.
- **Optional Compilation**: Default workspace builds omit `cubecl` to keep wheel sizes compact and build times fast on CPU-only or Triton-only environments.
- **Runtime Safety Guards**: `optimizer/crates/luminav-cubecl/src/engine_nn.rs` validates affine `weight`/`bias` buffer lengths against the feature dimension `D` and checks `MAX_GRID_X = 65_535` before kernel launch so oversized grids fall back cleanly instead of truncating dispatch ranges.
