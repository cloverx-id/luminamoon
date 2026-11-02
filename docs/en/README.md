# Loomy (luminamoon) Documentation (English)

**Loomy (`luminamoon`)** (**PROJECT LOOMY**) is a deep learning optimization and dataset text processing library built in Rust (Edition 2024) with a Python C-extension bridge (`PyO3 0.30.1` `abi3-py310` & `abi3t`). It combines two subsystems:

1. **LuminaV Optimizer v1.5.5**: Low-precision numerical optimization library with host CPU SIMD (`AVX-512`, `AVX2`), Vulkan 1.3/1.4 UMA Wave32 compute shaders, Triton JIT/AOT GPU kernels, and direct buffer interoperability for NumPy and PyTorch.
2. **LuminaMoon Data Engine v1.0.9**: Modular text preprocessing library with `AVX-512BW` / `AVX2` / `NEON` CPU SIMD text normalization, dual-layout `StringTensor` (16-byte CPU `StringView16` & contiguous GPU buffers), heuristic Shannon quality filtering, two-stage `xxh3_64` and MinHash LSH deduplication, PII masking, and 128-byte Ping-Pong (`Slot A` / `Slot B`) CRC32 checkpointing.

## Documentation Map

| Document | Description |
| :--- | :--- |
| [01. Getting Started](01_getting_started.md) | System requirements, pip installation, source compilation via Maturin/Rust, hardware & SIMD ISA detection (`LuminaV.detect_hardware()`, `lm.data_cpu_isa()`), and quickstarts for NumPy, PyTorch, and DataLoader (`FusedPipeline`, `StringTensor`, `StateTracker`, `ResumablePipeline`). |
| [02. User Guide](02_user_guide.md) | Guide on Optimizer buffer modes (Dual-Buffer CLM vs Single-Buffer TD-CLM), mixed-precision & stochastic rounding, step bounding, FSDP distributed training, plus Data Engine `FusedPipeline`, `StringTensor`, 128B Ping-Pong `StateTracker` checkpoint recovery, `StreamDeduplicator`, `MinHashDeduplicator`, and `PIIMasker`. |
| [03. Configuration & API Reference](03_configuration_api.md) | Constructor and function signatures, parameter tables, public methods, and validation rules for `LuminaV`, `FusedPipeline`, `StringTensor`, `StateTracker`, `StreamCoordinate`, `TextCleaner`, `QualityFilter`, `StreamDeduplicator`, `MinHashDeduplicator`, `PIIMasker`, and `ResumablePipeline`. |
| [04. Architecture & Hardware Acceleration](04_architecture_and_hardware.md) | C4 dispatch flow, Optimizer Rust crates (`luminav-core`, `luminav-simd`, `luminav-vulkan`, `luminav-dispatch`, `luminav-triton-aot`), Data Engine Rust modules (`coord`, `metadata`, `dedup`, `minhash`, `simd_cpu`, `postprocess`, `string_tensor`, `quality`, `serialize`, `py_bridge`), buffer memory protocols, and free-threaded Python support (PEP 703). |
| [Optimizer Developer Modular Docs](../optimize/README.md) | Internal developer documentation for the 7 Optimizer modules (`core`, `simd`, `vulkan`, `triton_aot`, `dispatch`, `python_ffi`, `python_runtime`). |
| [Data Engine Architecture & User Guide](../dataloader/standalone_guide.md) | Internal developer documentation for the 7 Data Engine modules (`coord_and_metadata`, `dedup`, `simd_and_postprocess`, `sources_and_streaming`, `preram`, `multimodal_and_hf_hub`, `python_ffi_and_runtime`). |
| [Optimizer Library Audit Suite](../library_audit/README.md) | Version and compatibility documentation for Rust toolchain ABI, SIMD intrinsics, GPU drivers (CUDA/ROCm/Vulkan), and PyO3/Maturin bindings. |
| [Data Engine Library Audit Suite](../library_audit/README.md) | Version and compatibility documentation for Apache Arrow/Parquet/IPC `60.0.0`, `rayon 1.12.0`, `xxhash-rust 0.8.19`, `hashbrown 0.17.1`, `crc32fast 1.5.2`, `parking_lot 0.12.5`, and `serde 1.0.229`. |

## Overview

- **Optional Framework Dependencies**: The base wheel ships with `dependencies = []`, allowing NumPy, PyTorch, or Triton to be installed as needed.
- **Dual-Buffer CLM & Single-Buffer TD-CLM**: Convex Lookahead Momentum (CLM) and Triangle-Defect Convex Lookahead Momentum (TD-CLM) with the Triangle Inequality Envelope and optional 50% optimizer state memory reduction.
- **Dual-Layout `StringTensor` & Resumable Checkpoints**: 16-byte CPU `StringView16` & contiguous GPU offset/payload buffers, parallel Rayon preprocessing (`FusedPipeline`), two-stage `xxh3_64` WAL deduplication, and 128-byte Ping-Pong CRC32 checkpoints.
- **Free-Threaded Python Support**: Compatible with Python 3.10 through 3.15+ via CPython Stable ABI (`abi3` & `abi3t` / PEP 703) with `py.detach()` GIL release during Rust execution.
