# Loomy (luminamoon) Documentation (English)

**Loomy (`luminamoon`)** is a deep learning optimization, rational neural architecture, and dataset text processing library built in Rust (Edition 2024) with a Python C-extension bridge (`PyO3 0.30.1` `abi3-py310` & `abi3t`). It includes:

1. **LuminaV Optimizer v2.0.0**: Low-precision numerical optimization library with host CPU SIMD (`AVX-512`, `AVX2`, `NEON`), optional CubeCL portable GPU compute (`wgpu`, Vulkan, Metal, CUDA, HIP, CPU), Triton JIT/AOT GPU kernels, CWD, RTV, configurable `bound_floor`, FSDP1/FSDP2 (`DTensor`)/HSDP sync, and MurmurHash3 + 3D Weyl-Kronecker stochastic rounding.
2. **LuminaMoon Data Engine v2.0.0**: `DataPipeline` with document-isolated `SequencePacker` / `TokenBatch`, `AVX-512BW` / `AVX2` / `NEON` CPU SIMD text normalization, `StringTensor` (16-byte `StringView16`, Arrow PyCapsule `__arrow_c_array__`, and contiguous GPU buffers), heuristic Shannon quality filtering, two-stage `XXH3-128` (`LMWL v2` WAL) and Kirsch-Mitzenmacher MinHash LSH deduplication, code-safe PII masking, `TokenRingBuffer`, and 256-byte `LMDL v2` Ping-Pong CRC32 checkpointing.
3. **Ra-Family Neural Architecture v2.0.0**: Rational, `tanh`-free and square-root-free neural components (`RaNorm`, `RaBatchNorm2d`, `RaTanh`, `RaGELU`, `RaSwiGLU`, `RaGLU`, `ra_softmax`, `RaAttention`, `ra_attention`, `LinearCrossEntropy`, `linear_cross_entropy`, `RaMoE`).

## Documentation Map

| Document | Description |
| :--- | :--- |
| [01. Getting Started](01_getting_started.md) | System requirements (Rust 1.95+), pip installation, optional extras, CLI (`luminamoon --help`), hardware & SIMD ISA probes, and quickstarts for NumPy, PyTorch, and `DataPipeline`. |
| [02. User Guide](02_user_guide.md) | Optimizer buffer modes (Dual-Buffer CLM, Single-Buffer TD-CLM, and Triangle Family T1/T2), CWD, RTV, `bound_floor`, mixed-precision & stochastic rounding, FSDP1/FSDP2/HSDP, plus `DataPipeline`, `SequencePacker`, `FusedPipeline`, `StringTensor`, 256B `LMDL v2` checkpoints, `StreamDeduplicator`, `MinHashDeduplicator`, and `PIIMasker`. |
| [03. Configuration & API Reference](03_configuration_api.md) | Constructor and function signatures, parameter tables, public methods, and validation rules for `LuminaV`, `luminamoon.data`, `luminamoon.nn`, and `luminamoon.cli`. |
| [04. Architecture & Hardware Acceleration](04_architecture_and_hardware.md) | Dispatch flow, Optimizer Rust crates (`luminav-core`, `luminav-simd`, `luminav-cubecl`, `luminav-dispatch`, `luminav-triton-aot`), Data Engine Rust modules, buffer memory protocols, and free-threaded Python support (PEP 703). |
| [Optimizer Developer Modular Docs](../optimize/README.md) | Internal developer documentation for the 7 Optimizer modules (`core`, `simd`, `cubecl`, `triton_aot`, `dispatch`, `python_ffi`, `python_runtime`). |
| [Data Engine Architecture & User Guide](../dataloader/standalone_guide.md) | Internal developer documentation for the 7 Data Engine modules (`coord_and_metadata`, `dedup`, `simd_and_postprocess`, `sources_and_streaming`, `preram`, `multimodal_and_hf_hub`, `python_ffi_and_runtime`). |
| [Ra-Family Neural Architecture](../nn/README.md) | Rational neural network components (`RaNorm`, `RaBatchNorm2d`, `RaTanh`, `RaGELU`, `RaSwiGLU`, `RaGLU`, `RaAttention`, `ra_softmax`, `LinearCrossEntropy`, `RaMoE`) and [Standalone Guide](../nn/standalone_guide.md). |
| [Theoretical References](../references.md) | Component mapping, mathematical derivations, and literature comparisons across the referenced papers. |
| [Workspace Dependency Notes](../dependencies/README.md) | Version and compatibility documentation for Rust toolchain ABI, SIMD intrinsics, GPU drivers (CUDA/ROCm/CubeCL), PyO3/Maturin bindings, Apache Arrow/Parquet/IPC `60.1.0`, `rayon 1.12.0`, `xxhash-rust 0.8.19`, `hashbrown 0.17.1`, and `crc32fast 1.5.2`. |

## Overview

- **Lazy Root Imports & Optional Dependencies**: The base wheel ships with `dependencies = []`, and `import luminamoon` only loads version metadata until a subpackage (`optimizer`, `nn`, or `data`) is accessed.
- **Dual-Buffer CLM, Single-Buffer TD-CLM & Triangle Family**: Convex Lookahead Momentum (CLM), Triangle-Defect Convex Lookahead Momentum (TD-CLM), and `tanh`-free Triangle Family (`T1`/`T2`) with the Triangle Inequality Envelope and optional 50% optimizer state memory reduction.
- **`DataPipeline`, Dual-Layout `StringTensor` & Resumable Checkpoints**: Document-isolated `SequencePacker`, 16-byte `StringView16` + Arrow PyCapsule & contiguous GPU buffers, parallel Rayon preprocessing (`FusedPipeline`), two-stage `XXH3-128` `LMWL v2` WAL deduplication, and 256-byte `LMDL v2` Ping-Pong CRC32 checkpoints.
- **Free-Threaded Python Support**: Compatible with Python 3.10 through 3.15+ via CPython Stable ABI (`abi3` & `abi3t` / PEP 703) with `py.detach()` GIL release during Rust execution.
