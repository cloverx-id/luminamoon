# LuminaMoon Documentation Portal

<div class="portal-badges">
  <span class="pill-badge">LuminaV Optimizer v1.5.3</span>
  <span class="pill-badge pill-badge-green">Data Engine v1.0.5.post1</span>
  <span class="pill-badge pill-badge-cyan">Rust 1.99.1 + Python ABI3T</span>
</div>

Welcome to the documentation portal for **LuminaMoon** (**PROJECT LUMI**), the monorepo for numerical optimization, multimodal dataset streaming, and formal verification tools developed by **Lumina Moon ([@cloverx-id](https://github.com/cloverx-id))** and **Silver Moon ([@cloverxion](https://github.com/cloverxion))**, hosting the **LuminaV Optimizer**, **LuminaMoon Data Engine**, and verification suites.

<div class="friendly-grid">
  <a class="friendly-card" href="en/01_getting_started/">
    <h3>Getting Started</h3>
    <p>System requirements, pip install, Rust compilation, and quickstart examples for NumPy, PyTorch, and Data Engine.</p>
  </a>
  <a class="friendly-card" href="en/02_user_guide/">
    <h3>LuminaV Optimizer</h3>
    <p>Dual-Buffer CLM vs Single-Buffer TD-CLM, mixed-precision, stochastic rounding, and FSDP distributed training.</p>
  </a>
  <a class="friendly-card" href="dataloader/standalone_guide/">
    <h3>Data Engine</h3>
    <p>Fused SIMD preprocessing, StringTensor zero-copy buffers, streaming datasets, and 128B Ping-Pong crash recovery.</p>
  </a>
  <a class="friendly-card" href="en/03_configuration_api/">
    <h3>API Reference</h3>
    <p>Complete signatures, parameter reference tables, validation constraints, and return types for all modules.</p>
  </a>
  <a class="friendly-card" href="security/">
    <h3>Security & Audits</h3>
    <p>Direct vulnerability disclosure policy to Silver Moon, CodeQL scanning, and third-party library safety audits.</p>
  </a>
  <a class="friendly-card" href="changelogs/">
    <h3>Changelogs</h3>
    <p>Version-by-version release history for Software package distributions and LuminaV algorithm specifications.</p>
  </a>
</div>

---

## 1. User Documentation

Source directory: `docs/en/`

1. **[01. Getting Started](en/01_getting_started.md)**: System requirements, installation, source compilation via Maturin/Rust, hardware & SIMD ISA auditing (`LuminaV.detect_hardware()`, `lm.data_cpu_isa()`), and quickstarts for NumPy, PyTorch, and DataLoader (`FusedPipeline`, `StringTensor`, `StateTracker`, `ResumablePipeline`).
2. **[02. User Guide](en/02_user_guide.md)**: Optimizer buffer modes (Dual-Buffer CLM vs Single-Buffer TD-CLM), mixed-precision & stochastic rounding, boundary dynamics, FSDP distributed training, plus DataLoader $O(1)$ local/HTTP-206 streaming, 100-batch Pre-RAM watermark ring buffer, Ping-Pong crash recovery, two-stage XXH3-64 deduplication, and multimodal HuggingFace Hub pipelines.
3. **[03. Configuration & API Reference](en/03_configuration_api.md)**: Complete parameter reference tables, public methods, and validation rules for `LuminaV`, `TextCleaner`, `QualityFilter`, `StreamDeduplicator`, `MinHashDeduplicator`, `PIIMasker`, `StateTracker`, `StreamCoordinate`, `StringTensor`, `FusedPipeline`, and `ResumablePipeline`.
4. **[04. Architecture & Hardware Acceleration](en/04_architecture_and_hardware.md)**: C4 dispatch hierarchy, Optimizer Rust crates (`core`, `simd`, `vulkan`, `dispatch`, `triton-aot`), DataLoader Rust modules (`constants`, `coord`, `dedup`, `metadata`, `simd_cpu`, `postprocess`, `source_local`, `source_hf`, `preram`, `multimodal`, `multimodal_formats`, `multimodal_serialize`, `hf_hub`, `py_bridge`), zero-copy memory protocols, and free-threaded Python (`abi3t` / PEP 703).

---

## 2. Developer Modular Documentation

Deep technical documentation for systems engineers, kernel writers, and library developers (`docs/optimize/` & `data/docs/`):

### A. LuminaV Optimizer
Source directories: `docs/optimize/` & `optimizer/`

- **[LuminaV Optimizer Overview & User Guide (`optimizer/README.md`)](optimize/standalone_guide.md)**
- **[Developer Optimizer Index (`docs/optimize/README.md`)](optimize/README.md)**
- **[01. Core & Math](optimize/core/README.md)**: CLM, TD-CLM temporal difference, PRNG stochastic rounding, and bias correction.
- **[02. SIMD Vectorization](optimize/simd/README.md)**: AVX-512 and AVX2/FMA x86_64 intrinsics and Rayon parallel threadpool.
- **[03. Vulkan Compute](optimize/vulkan/README.md)**: Ash Vulkan 1.3/1.4 SPIR-V compute shaders and UMA zero-copy host buffers.
- **[04. Triton AOT & GPU Drivers](optimize/triton_aot/README.md)**: AOT compilation, CUDA/HIP driver dynamic loading, and standalone C-API.
- **[05. Dispatch & Routing](optimize/dispatch/README.md)**: Hardware capability detector and 16,384-element dispatch threshold heuristic.
- **[06. Python FFI & PyO3 Bridge](optimize/python_ffi/README.md)**: PyO3 GIL release (`py.detach()`), anti-aliasing safety, and NumPy buffer protocols.
- **[07. Python Runtime & GPU Mixin](optimize/python_runtime/README.md)**: `LuminaV` class, `constants.py`, Triton JIT kernels, FSDP synchronization, and state checkpointing.

### B. LuminaMoon Data Engine
Source directories: `data/` & `data/docs/`

- **[LuminaMoon Data Engine Overview & User Guide (`data/README.md`)](dataloader/standalone_guide.md)**
- **[Data Engine Standalone Portal (`data/docs/README.md`)](dataloader/README.md)**
- **[Developer Data Engine Index (`data/docs/README.md`)](dataloader/README.md)**
- **[01. TextCleaner & QualityFilter](dataloader/simd_and_postprocess/README.md)**: AVX-512/AVX2/NEON SIMD carriage-return stripping, JSON extraction, Shannon entropy, repetition loop, and symbol ratio heuristics.
- **[02. StreamDeduplicator & MinHashDeduplicator](dataloader/dedup/README.md)**: High-throughput exact streaming deduplication with Blake2b/XXH3-64, WAL persistence, and MinHash LSH fuzzy deduplication.
- **[03. PIIMasker & Data Sanitization](dataloader/simd_and_postprocess/README.md)**: Pattern-based and regex sanitizer for emails, IPv4 addresses, phone numbers, and custom secret tokens.
- **[04. StateTracker & StreamCoordinate](dataloader/coord_and_metadata/README.md)**: 32-byte `StreamCoordinate` navigation, dual-slot 128-byte crash-proof Ping-Pong persistence, and multi-epoch state tracking.
- **[05. StringTensor & Memory Layouts](dataloader/coord_and_metadata/README.md)**: CPU StringView (inlined <= 12 bytes) and GPU contiguous offset/payload buffers compatible with PyTorch, NumPy, Arrow, and Polars.
- **[06. FusedPipeline & ResumablePipeline](dataloader/python_ffi_and_runtime/README.md)**: Multi-threaded Rayon single-pass pipeline (>400k records/s) and resumable streaming pipeline with async background prefetcher.
- **[07. Multimodal Serialization](dataloader/multimodal_and_hf_hub/README.md)**: Zero-copy serialization to JSONL, Parquet with Zstd compression and embedded image payloads, Arrow IPC, and CSV.

---

## 3. Engineering & External Library Audits

Source directories: `docs/library_audit/` & `data/docs/library_audit/`

- **[Optimizer Library Audit Suite (`docs/library_audit/`)](library_audit/README.md)**: Technical verification and ABI compliance audits for Rust 1.99.1 toolchain, PyO3/Maturin, PyTorch 2.14.1, Triton JIT 3.8.0, CUDA/ROCm, and Vulkan Ash compute pipeline.
- **[Data Engine Standalone Library Audit Suite (`data/docs/library_audit/`)](library_audit/README.md)**: Technical verification and codebase adaptations for Rust 1.99.1 / Edition 2024, `pyo3 0.30.1` (`abi3t`), Apache Arrow/Parquet/IPC `60.0.0`, `rayon 1.12.0`, `xxhash-rust 0.8.19` (`xxh3`), `crc32fast 1.5.2`, `hashbrown 0.17.1`, `parking_lot 0.12.5`, `serde 1.0.229`, and HuggingFace Hub API protocols.