# LuminaMoon Documentation (English)

LuminaMoon (**PROJECT LUMI**) is a unified high-performance deep learning optimization and dataset streaming suite built in Rust (Edition 2024) with a GIL-free Python C-extension bridge (`PyO3 0.30.1` `abi3-py310` & `abi3t`). It unites two core subsystems:

1. **LuminaV Optimizer v1.5.3**: Low-precision mathematical optimization engine with host CPU SIMD (`AVX-512`, `AVX2`), Vulkan 1.3/1.4 UMA Wave32 compute shaders, Triton JIT/AOT GPU kernels, and universal zero-copy execution for NumPy and PyTorch.
2. **LuminaMoon Data Engine v1.0.5**: Zero-copy local `mmap` and concurrent HTTP 206 dataset streaming engine with a 100-batch Pre-RAM watermark ring buffer, 128-byte Ping-Pong (`Slot A` / `Slot B`) CRC32 crash recovery, two-stage `xxh3_64` deduplication, self-contained `AVX-512BW` / `AVX2` / `NEON` CPU SIMD byte scanning, multimodal Parquet/JSONL/JSON serialization, and official HuggingFace Hub integration.

## Documentation Map

| Document | Description |
| :--- | :--- |
| [01. Getting Started](01_getting_started.md) | System requirements, pip installation, source compilation via Maturin/Rust, hardware & SIMD ISA auditing (`LuminaV.detect_hardware()`, `lm.data_cpu_isa()`), and quickstarts for NumPy, PyTorch, and DataLoader (`FusedPipeline`, `StringTensor`, `StateTracker`, `ResumablePipeline`). |
| [02. User Guide](02_user_guide.md) | In-depth guide on Optimizer buffer modes (Dual-Buffer CLM vs Single-Buffer TD-CLM), mixed-precision & stochastic rounding, boundary dynamics, FSDP distributed training, plus DataLoader $O(1)$ local/HTTP-206 streaming, Pre-RAM watermark ring buffer, Ping-Pong crash recovery, two-stage XXH3-64 deduplication, and multimodal HuggingFace pipelines. |
| [03. Configuration & API Reference](03_configuration_api.md) | Complete constructor and function signatures, parameter tables, public methods, and validation rules for `LuminaV`, `lm.stream`, `DataStreamLoader`, `MultimodalParquetLoader`, `lm.from_hf_dataset`, `lm.push_to_hub`, `DataBatch`, `MultimodalBatch`, and `StreamCoordinate`. |
| [04. Architecture & Hardware Acceleration](04_architecture_and_hardware.md) | C4 dispatch flow, Optimizer Rust crates (`luminav-core`, `luminav-simd`, `luminav-vulkan`, `luminav-dispatch`, `luminav-triton-aot`), DataLoader Rust modules (`coord`, `dedup`, `metadata`, `simd_cpu`, `postprocess`, `source_local`, `source_hf`, `preram`, `multimodal`, `hf_hub`, `py_bridge`), zero-copy memory protocols, and free-threaded Python support (PEP 703). |
| [Optimizer Developer Modular Docs](../optimize/README.md) | Internal developer specifications for all 7 Optimizer modules (`core`, `simd`, `vulkan`, `triton_aot`, `dispatch`, `python_ffi`, `python_runtime`). |
| [Data Engine Architecture & User Guide](../../data/README.md) | Internal developer specifications for all 7 DataLoader modules (`coord_and_metadata`, `dedup`, `simd_and_postprocess`, `sources_and_streaming`, `preram`, `multimodal_and_hf_hub`, `python_ffi_and_runtime`). |
| [Optimizer Library Audit Suite](../library_audit/README.md) | Audit reports on Rust toolchain ABI, SIMD intrinsics, GPU drivers (CUDA/ROCm/Vulkan), and PyO3/Maturin bindings. |
| [Data Engine Library Audit Suite](../../data/docs/library_audit/README.md) | Audit reports on Arrow/Parquet 60, `reqwest 0.13.5`, `tokio 1.53.3`, `bytes 1.12.1` (`CVE-2026-25541`), `memmap2`, `xxhash-rust`, `hashbrown`, `crc32fast`, and HuggingFace Hub API protocols. |

## Core Architectural Highlights

- **Zero Heavy Mandatory Dependencies**: The core package ships with an empty `dependencies = []` footprint.
- **Dual-Buffer CLM & Single-Buffer TD-CLM**: Convex Lookahead Momentum (CLM) and Triangle-Defect Convex Lookahead Momentum (TD-CLM) with the Triangle Inequality Envelope and optional 50% optimizer state memory savings.
- **Zero-Copy Streaming & $O(1)$ Crash Recovery**: Local `mmap`, Arrow `RowSelection` page skipping, concurrent HTTP 206 multi-range workers, 100-batch Pre-RAM watermark prefetching, and 128-byte Ping-Pong CRC32 checkpoints.
- **Cross-Platform Free-Threaded Ready**: Compatible with Python 3.10 through 3.15+ via CPython Stable ABI (`abi3` & `abi3t` / PEP 703) with full `py.detach()` GIL liberation.
