# Loomy (luminamoon) Documentation (English)

**Loomy (`luminamoon`)** (**PROJECT LOOMY**) is a unified high-performance deep learning optimization and dataset streaming suite built in Rust (Edition 2024) with a GIL-free Python C-extension bridge (`PyO3 0.30.1` `abi3-py310` & `abi3t`). It unites two core subsystems:

1. **LuminaV Optimizer v1.5.5**: Low-precision mathematical optimization engine with host CPU SIMD (`AVX-512`, `AVX2`), Vulkan 1.3/1.4 UMA Wave32 compute shaders, Triton JIT/AOT GPU kernels, and universal zero-copy execution for NumPy and PyTorch.
2. **LuminaMoon Data Engine v1.0.8**: Modular preprocessing engine with self-contained `AVX-512BW` / `AVX2` / `NEON` CPU SIMD text normalization, dual-layout `StringTensor` (16-byte Umbra CPU `StringView16` & contiguous GPU buffers), heuristic Shannon quality filtering, two-stage `xxh3_64` and MinHash LSH deduplication, PII masking, and 128-byte Ping-Pong (`Slot A` / `Slot B`) CRC32 crash recovery.

## Documentation Map

| Document | Description |
| :--- | :--- |
| [01. Getting Started](01_getting_started.md) | System requirements, pip installation, source compilation via Maturin/Rust, hardware & SIMD ISA auditing (`LuminaV.detect_hardware()`, `lm.data_cpu_isa()`), and quickstarts for NumPy, PyTorch, and DataLoader (`FusedPipeline`, `StringTensor`, `StateTracker`, `ResumablePipeline`). |
| [02. User Guide](02_user_guide.md) | In-depth guide on Optimizer buffer modes (Dual-Buffer CLM vs Single-Buffer TD-CLM), mixed-precision & stochastic rounding, boundary dynamics, FSDP distributed training, plus Data Engine `FusedPipeline`, `StringTensor`, 128B Ping-Pong `StateTracker` crash recovery, `StreamDeduplicator`, `MinHashDeduplicator`, and `PIIMasker`. |
| [03. Configuration & API Reference](03_configuration_api.md) | Complete constructor and function signatures, parameter tables, public methods, and validation rules for `LuminaV`, `FusedPipeline`, `StringTensor`, `StateTracker`, `StreamCoordinate`, `TextCleaner`, `QualityFilter`, `StreamDeduplicator`, `MinHashDeduplicator`, `PIIMasker`, and `ResumablePipeline`. |
| [04. Architecture & Hardware Acceleration](04_architecture_and_hardware.md) | C4 dispatch flow, Optimizer Rust crates (`luminav-core`, `luminav-simd`, `luminav-vulkan`, `luminav-dispatch`, `luminav-triton-aot`), Data Engine Rust modules (`coord`, `metadata`, `dedup`, `minhash`, `simd_cpu`, `postprocess`, `string_tensor`, `quality`, `serialize`, `py_bridge`), zero-copy memory protocols, and free-threaded Python support (PEP 703). |
| [Optimizer Developer Modular Docs](../optimize/README.md) | Internal developer specifications for all 7 Optimizer modules (`core`, `simd`, `vulkan`, `triton_aot`, `dispatch`, `python_ffi`, `python_runtime`). |
| [Data Engine Architecture & User Guide](../dataloader/standalone_guide.md) | Internal developer specifications for all 7 Data Engine modules (`coord_and_metadata`, `dedup`, `simd_and_postprocess`, `sources_and_streaming`, `preram`, `multimodal_and_hf_hub`, `python_ffi_and_runtime`). |
| [Optimizer Library Audit Suite](../library_audit/README.md) | Version and compatibility documentation for Rust toolchain ABI, SIMD intrinsics, GPU drivers (CUDA/ROCm/Vulkan), and PyO3/Maturin bindings. |
| [Data Engine Library Audit Suite](../library_audit/README.md) | Version and compatibility documentation for Apache Arrow/Parquet/IPC `60.0.0`, `rayon 1.12.0`, `xxhash-rust 0.8.19`, `hashbrown 0.17.1`, `crc32fast 1.5.2`, `parking_lot 0.12.5`, and `serde 1.0.229`. |

## Core Architectural Highlights

- **Zero Heavy Mandatory Dependencies**: The core package ships with an empty `dependencies = []` footprint.
- **Dual-Buffer CLM & Single-Buffer TD-CLM**: Convex Lookahead Momentum (CLM) and Triangle-Defect Convex Lookahead Momentum (TD-CLM) with the Triangle Inequality Envelope and optional 50% optimizer state memory savings.
- **Dual-Layout `StringTensor` & $O(1)$ Crash Recovery**: 16-byte Umbra CPU `StringView16` & contiguous GPU offset/payload buffers, fused parallel Rayon preprocessing (`FusedPipeline`), two-stage `xxh3_64` WAL deduplication, and 128-byte Ping-Pong CRC32 checkpoints.
- **Cross-Platform Free-Threaded Ready**: Compatible with Python 3.10 through 3.15+ via CPython Stable ABI (`abi3` & `abi3t` / PEP 703) with full `py.detach()` GIL liberation.
