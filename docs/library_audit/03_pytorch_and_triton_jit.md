# 3. PyTorch and Triton Compiler (JIT & AOT)

## 3.1 Latest Stable Versions
- **PyTorch (`torch`)**: `2.14.1` (`2.14.0` major release)
- **Triton (`triton-lang`)**: `3.8.0`

---

## 3.2 PyTorch Changelog (`2.4.0` - `2.14.1`)

### A. Stabilization of `torch.accelerator` API (`2.6.0` - `2.14.1`)
- **What changed**: PyTorch introduced and stabilized `torch.accelerator` as a unified hardware abstraction layer across CUDA, Intel XPU, Apple MPS, and custom accelerators.
- **Core APIs**:
  - `torch.accelerator.is_available()`
  - `torch.accelerator.device(dev)` (device-agnostic context manager)
  - `torch.accelerator.current_device_index()` (active device index resolution)
- **Adaptation in `python/luminamoon/optimizer/gpu_backend.py`**:
  `_get_device_context` and `_get_generator` check `torch.cuda`, `torch.xpu`, and `torch.accelerator.current_device_index()` in sequence so `torch.Generator` instantiation for Stochastic Rounding never defaults to device index `0` on multi-GPU XPU or non-CUDA accelerators.

---

## 3.3 Triton Changelog (`3.0.0` - `3.8.0`)

### A. Stricter `@triton.jit` AST Parser (`3.2.0` - `3.8.0`)
- **What changed**: The Triton AST compiler rejects `torch.*` function symbol resolution inside `@triton.jit`-decorated kernels. All mathematical operations inside kernels must use `triton.language` primitives (`tl.*` or `tl.math.*`).
- **Bug Discovery & Fix in `python/luminamoon/optimizer/gpu_backend.py`**:
  In `_lumina_tdclm_pass2_kernel`, a `torch.sqrt(p_init * p_init + 1e-4)` call inside the `bound_mode == 2` branch was replaced with `tl.sqrt(p_init * p_init + 1e-4)`, eliminating `CompilationError` failures on Triton `3.2`–`3.8`.

### B. `tl.math` vs `tl` Namespacing (`3.0.0` - `3.8.0`)
- **What changed**: Transcendental functions such as `tanh` reside under `tl.math.tanh` (superseding removed `tl.libdevice.tanh` and legacy `tl.tanh`).
- **Adaptation**: `_triton_tanh_fast` prioritizes `tl.math.tanh(x)` with a safe rational exponential fallback on minimal installations.

### C. Trailing Scratch Pointer Conventions in Triton AOT (`3.0.0` - `3.8.0`)
- **What changed**:
  - **Triton 3.0 - 3.2**: AOT-compiled kernels (`cubin` / `hsaco`) append 1 implicit pointer at the end of runtime argument lists: `CUdeviceptr global_scratch`.
  - **Triton 3.3 - 3.8**: The AOT kernel ABI supports up to 2 implicit trailing scratch pointers: `global_scratch` and `profile_scratch` (populated with pointers to `0u64` when global scratch or Proton profiling is unused).
- **Adaptation in `optimizer/crates/luminav-triton-aot/`**:
  - `KernelMetadata` and `AotManifest` store `num_trailing_scratch_ptrs`.
  - `pack_triton_abi_args(user_arg_ptrs, num_trailing_scratch_ptrs)` in `cuda_driver.rs` dynamically appends `num_trailing_scratch_ptrs` pointers to `&TRITON_NULL_SCRATCH` (supporting both 1-pointer and 2-pointer conventions in Triton `3.8.0`).
  - Unit test `test_triton3_abi_trailing_scratch_pointer` verifies ABI packing for both 1 and 2 trailing scratch pointers.
