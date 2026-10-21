# LuminaMoon Documentation Portal

Welcome to the documentation portal for **LuminaMoon** (**PROJECT LUMI**), the monorepo for numerical optimization, multimodal dataset streaming, and formal verification tools developed by **Lumina Moon ([@cloverx-id](https://github.com/cloverx-id))** and **Silver Moon ([@cloverxion](https://github.com/cloverxion))**, hosting the **LuminaV Optimizer**, **LuminaMoon Data Engine**, and verification suites.

---

## 1. User Documentation (`docs/en/`)

1. **[01. Getting Started](en/01_getting_started.md)**: System requirements, installation, source compilation via Maturin/Rust, hardware & SIMD ISA auditing (`LuminaV.detect_hardware()`, `lm.data_cpu_isa()`), and quickstarts for NumPy, PyTorch, and DataLoader (`FusedPipeline`, `StringTensor`, `StateTracker`, `ResumablePipeline`).
2. **[02. User Guide](en/02_user_guide.md)**: Optimizer buffer modes (Dual-Buffer CLM vs Single-Buffer TD-CLM), mixed-precision & stochastic rounding, boundary dynamics, FSDP distributed training, plus DataLoader $O(1)$ local/HTTP-206 streaming, 100-batch Pre-RAM watermark ring buffer, Ping-Pong crash recovery, two-stage XXH3-64 deduplication, and multimodal HuggingFace Hub pipelines.
3. **[03. Configuration & API Reference](en/03_configuration_api.md)**: Complete parameter reference tables, public methods, and validation rules for `LuminaV`, `TextCleaner`, `QualityFilter`, `StreamDeduplicator`, `MinHashDeduplicator`, `PIIMasker`, `StateTracker`, `StreamCoordinate`, `StringTensor`, `FusedPipeline`, and `ResumablePipeline`.
4. **[04. Architecture & Hardware Acceleration](en/04_architecture_and_hardware.md)**: C4 dispatch hierarchy, Optimizer Rust crates (`core`, `simd`, `vulkan`, `dispatch`, `triton-aot`), DataLoader Rust modules (`constants`, `coord`, `dedup`, `metadata`, `simd_cpu`, `postprocess`, `source_local`, `source_hf`, `preram`, `multimodal`, `multimodal_formats`, `multimodal_serialize`, `hf_hub`, `py_bridge`), zero-copy memory protocols, and free-threaded Python (`abi3t` / PEP 703).

---

## 2. Developer Modular Documentation (`docs/optimize/` & `data/docs/`)

Deep technical documentation for systems engineers, kernel writers, and library developers:

### A. LuminaV Optimizer (`docs/optimize/` & `optimizer/`)
- **[LuminaV Optimizer Overview & User Guide (`optimizer/README.md`)](../optimizer/README.md)**
- **[Developer Optimizer Index (`docs/optimize/README.md`)](optimize/README.md)**
- **[01. Core & Math](optimize/core/README.md)**: CLM, TD-CLM temporal difference, PRNG stochastic rounding, and bias correction.
- **[02. SIMD Vectorization](optimize/simd/README.md)**: AVX-512 and AVX2/FMA x86_64 intrinsics and Rayon parallel threadpool.
- **[03. Vulkan Compute](optimize/vulkan/README.md)**: Ash Vulkan 1.3/1.4 SPIR-V compute shaders and UMA zero-copy host buffers.
- **[04. Triton AOT & GPU Drivers](optimize/triton_aot/README.md)**: AOT compilation, CUDA/HIP driver dynamic loading, and standalone C-API.
- **[05. Dispatch & Routing](optimize/dispatch/README.md)**: Hardware capability detector and 16,384-element dispatch threshold heuristic.
- **[06. Python FFI & PyO3 Bridge](optimize/python_ffi/README.md)**: PyO3 GIL release (`py.detach()`), anti-aliasing safety, and NumPy buffer protocols.
- **[07. Python Runtime & GPU Mixin](optimize/python_runtime/README.md)**: `LuminaV` class, `constants.py`, Triton JIT kernels, FSDP synchronization, and state checkpointing.

### B. LuminaMoon Data Engine (`data/` & `data/docs/`)
- **[LuminaMoon Data Engine Overview & User Guide (`data/README.md`)](../data/README.md)**
- **[Data Engine Standalone Portal (`data/docs/README.md`)](../data/docs/README.md)**
- **[Developer Data Engine Index (`data/docs/README.md`)](dataloader/README.md)**
- **[01. TextCleaner & QualityFilter](dataloader/simd_and_postprocess/README.md)**: AVX-512/AVX2/NEON SIMD carriage-return stripping, JSON extraction, Shannon entropy, repetition loop, and symbol ratio heuristics.
- **[02. StreamDeduplicator & MinHashDeduplicator](dataloader/dedup/README.md)**: High-throughput exact streaming deduplication with Blake2b/XXH3-64, WAL persistence, and MinHash LSH fuzzy deduplication.
- **[03. PIIMasker & Data Sanitization](dataloader/simd_and_postprocess/README.md)**: Pattern-based and regex sanitizer for emails, IPv4 addresses, phone numbers, and custom secret tokens.
- **[04. StateTracker & StreamCoordinate](dataloader/coord_and_metadata/README.md)**: 32-byte `StreamCoordinate` navigation, dual-slot 128-byte crash-proof Ping-Pong persistence, and multi-epoch state tracking.
- **[05. StringTensor & Memory Layouts](dataloader/coord_and_metadata/README.md)**: CPU StringView (inlined <= 12 bytes) and GPU contiguous offset/payload buffers compatible with PyTorch, NumPy, Arrow, and Polars.
- **[06. FusedPipeline & ResumablePipeline](dataloader/python_ffi_and_runtime/README.md)**: Multi-threaded Rayon single-pass pipeline (>400k records/s) and resumable streaming pipeline with async background prefetcher.
- **[07. Multimodal Serialization](dataloader/multimodal_and_hf_hub/README.md)**: Zero-copy serialization to JSONL, Parquet with Zstd compression and embedded image payloads, Arrow IPC, and CSV.

---

## 3. Engineering & External Library Audits (`docs/library_audit/` & `data/docs/library_audit/`)

- **[Optimizer Library Audit Suite (`docs/library_audit/`)](library_audit/README.md)**: Technical verification and ABI compliance audits for Rust 1.99.1 toolchain, PyO3/Maturin, PyTorch 2.14.1, Triton JIT 3.8.0, CUDA/ROCm, and Vulkan Ash compute pipeline.
- **[Data Engine Standalone Library Audit Suite (`data/docs/library_audit/`)](../data/docs/library_audit/README.md)**: Technical verification and codebase adaptations for Rust 1.99.1 / Edition 2024, `pyo3 0.30.1` (`abi3t`), Apache Arrow/Parquet/IPC `60.0.0`, `half 2.7.1`, `flate2 1.1.10`, `zstd 0.14.3`, `lz4_flex 0.14.0` (`CVE-2026-32829`), `reqwest 0.13.5`, `tokio 1.53.3`, `bytes 1.12.1` (`CVE-2026-25541`), `memmap2 0.9.11` (`RUSTSEC-2026-0186`), `xxhash-rust`, `hashbrown`, `crc32fast`, and HuggingFace Hub API protocols.
