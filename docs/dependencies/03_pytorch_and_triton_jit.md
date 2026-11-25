# 3. PyTorch and Triton Compiler (JIT & AOT)

## 3.1 Tested & Target Versions
- **PyTorch (`torch`)**: Tested on `2.14.1` (upstream `2.15` released October 28, 2026)
- **Triton (`triton-lang`)**: `3.9.0`

---

## 3.2 PyTorch Changelog (`2.4.0` - `2.15`)

### A. Stabilization of `torch.accelerator` API (`2.6.0` - `2.15`)
- **What changed**: PyTorch introduced and stabilized `torch.accelerator` as a shared hardware abstraction layer across CUDA, Intel XPU, Apple MPS, and custom accelerators.
- **Core APIs**:
    - `torch.accelerator.is_available()`
    - `torch.accelerator.device(dev)` (device-agnostic context manager)
    - `torch.accelerator.current_device_index()` (active device index resolution)
- **Adaptation in `python/luminamoon/optimizer/torch_ops.py`**:
    - `_get_device_context(device)` selects `torch.cuda.device(device)` on CUDA, `torch.xpu.device(device)` on Intel XPU, and `torch.accelerator.device(device)` on other PyTorch accelerators so Triton and device operations run on the active device cleanly across multi-GPU setups.

---

## 3.3 Triton Changelog (`3.0.0` - `3.9.0`)

### A. Stricter `@triton.jit` AST Parser (`3.2.0` - `3.9.0`)
- **What changed**: The Triton AST compiler rejects `torch.*` function symbol resolution inside `@triton.jit`-decorated kernels. All mathematical operations inside kernels must use `triton.language` primitives (`tl.*` or `tl.math.*`).
- **Fix in `python/luminamoon/optimizer/backends/triton/`**:
  In `_lumina_tdclm_pass2_kernel`, `torch.sqrt(...)` inside the `bound_mode == 2` branch was replaced with `tl.sqrt(...)`, fixing `CompilationError` failures on Triton `3.2`–`3.9`.

### B. Padé `[13/6]` Rational `tanh` & Transcendental Avoidance (`3.0.0` - `3.9.0`)
- **What changed**: Legacy `tl.libdevice.tanh` and `tl.tanh` were moved under `tl.math.tanh`, while hardware SFU `tanh` / `exp` units across GPU generations exhibit cross-vendor rounding drift and slower throughput scaling relative to Tensor Cores on Hopper/Blackwell (`tcgen05`, TMA, warp specialization).
- **Adaptation in `python/luminamoon/optimizer/backends/triton/common.py`**:
    - `_triton_tanh_fast(x)` does not invoke `tl.math.tanh`; it evaluates the deterministic Padé `[13/6]` rational polynomial via Horner FMA on `clamp(x, -7.9988117, 7.9988117)`, matching `luminav-core`, `luminav-simd`, `luminav-cubecl`, and PyTorch eager within $< 3.0 \times 10^{-7}$ without SFU divergence.
    - Triangle Family modes (`buffer="t1"`, `"t2"`) and `luminamoon.nn` (`RaNorm`, `RaSwiGLU`, `RaGLU`, `ra_softmax`, `RaAttention`) avoid `tanh`, `exp`, and `sqrt` in their core algebraic paths.
    - For MoE layers (`python/luminamoon/nn/moe.py`), PyTorch `torch._grouped_mm` is selected on `sm_90+` (Hopper, Blackwell, Rubin) with a persistent Triton grouped GEMM fallback (`moe_grouped_gemm.py`) on `sm_75+` / ROCm.

### C. Trailing Scratch Pointer Conventions in Triton AOT (`3.0.0` - `3.9.0`)
- **What changed**:
    - **Triton 3.0 - 3.2**: AOT-compiled kernels (`cubin` / `hsaco`) append 1 implicit pointer at the end of runtime argument lists: `CUdeviceptr global_scratch`.
    - **Triton 3.3 - 3.9**: The AOT kernel ABI supports up to 2 implicit trailing scratch pointers: `global_scratch` and `profile_scratch` (populated with pointers to `0u64` when global scratch or Proton profiling is unused).
- **Adaptation in `optimizer/crates/luminav-triton-aot/`**:
    - `KernelMetadata` and `AotManifest` store `num_trailing_scratch_ptrs`.
    - `pack_triton_abi_args(user_arg_ptrs, num_trailing_scratch_ptrs)` in `cuda_driver.rs` dynamically appends `num_trailing_scratch_ptrs` pointers to `&TRITON_NULL_SCRATCH` (supporting both 1-pointer and 2-pointer conventions in Triton `3.9.0`).
    - Unit test `test_triton38_abi_two_trailing_scratch_pointers` (`optimizer/crates/luminav-triton-aot/tests/aot_tests.rs`) verifies ABI packing for both 1 and 2 trailing scratch pointers.
