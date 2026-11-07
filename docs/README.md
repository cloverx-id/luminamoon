# Loomy (luminamoon) Documentation

<div class="portal-badges">
  <span class="pill-badge">LuminaV Optimizer v1.6.0</span>
  <span class="pill-badge pill-badge-green">Data Engine v1.1.0</span>
  <span class="pill-badge pill-badge-cyan">LSO-Eval Verified</span>
  <span class="pill-badge pill-badge-cyan">Rust 1.99.1 + Python ABI3T</span>
</div>

**Loomy (`luminamoon`)** provides low-precision numerical optimization, rational neural network layers, and dataset text preprocessing utilities developed by **Silver Moon ([@cloverxion](https://github.com/cloverxion))** and **Lumina Moon ([@cloverx-id](https://github.com/cloverx-id)) contributors**.

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
    <p>Fused SIMD text preprocessing, StringTensor CPU/GPU buffers, deduplication, and 128B Ping-Pong checkpoint recovery.</p>
  </a>
  <a class="friendly-card" href="lso_eval/">
    <h3>LSO-Eval</h3>
    <p>Z3 SMT safety checks, 2D trajectory benchmarks, 1-Pass EMA parity, and Qwen3.5-4B & Vision Two-Class evaluation.</p>
  </a>
  <a class="friendly-card" href="en/03_configuration_api/">
    <h3>API Reference</h3>
    <p>Complete signatures, parameter reference tables, validation constraints, and return types for all modules.</p>
  </a>
  <a class="friendly-card" href="security/">
    <h3>Security & Notes</h3>
    <p>Direct vulnerability disclosure policy to Silver Moon, CodeQL scanning, and third-party library compatibility notes.</p>
  </a>
  <a class="friendly-card" href="changelogs/">
    <h3>Changelogs</h3>
    <p>Version-by-version release history for Software package distributions and LuminaV algorithm specifications.</p>
  </a>
</div>

---

## 1. User Documentation

Source directory: `docs/en/`

- **[01. Getting Started](en/01_getting_started.md)**: System requirements, installation, source compilation via Maturin/Rust, hardware & SIMD ISA detection (`LuminaV.detect_hardware()`, `lm.data_cpu_isa()`), and quickstarts for NumPy, PyTorch, and Data Engine (`FusedPipeline`, `StringTensor`, `StateTracker`, `ResumablePipeline`).
- **[02. User Guide](en/02_user_guide.md)**: Optimizer buffer modes (2-Buffer CLM, 1-Buffer TD-CLM, and Triangle Family T1/T2), mixed-precision & stochastic rounding, step bounding, FSDP distributed training, plus Data Engine SIMD text cleaning, quality filtering, two-stage XXH3-64 & MinHash deduplication, PII masking, StringTensor buffers, and Ping-Pong checkpoint recovery.
- **[03. Configuration & API Reference](en/03_configuration_api.md)**: Complete parameter reference tables, public methods, and validation rules for `LuminaV`, `TextCleaner`, `QualityFilter`, `StreamDeduplicator`, `MinHashDeduplicator`, `PIIMasker`, `StateTracker`, `StreamCoordinate`, `StringTensor`, `FusedPipeline`, and `ResumablePipeline`.
- **[04. Architecture & Hardware Acceleration](en/04_architecture_and_hardware.md)**: C4 dispatch hierarchy, Optimizer Rust crates (`core`, `simd`, `vulkan`, `dispatch`, `triton-aot`), Data Engine Rust modules (`constants`, `coord`, `dedup`, `error`, `metadata`, `minhash`, `postprocess`, `py_bridge`, `quality`, `serialize`, `simd_cpu`, `string_tensor`), buffer memory protocols, and free-threaded Python (`abi3t` / PEP 703).

---

## 2. Developer Modular Documentation

Technical documentation for contributors working on the Rust crates, GPU kernels, and Python bindings (`docs/optimize/` & `data/docs/`):

### A. LuminaV Optimizer
Source directories: `docs/optimize/` & `optimizer/`

- **[LuminaV Optimizer Overview & User Guide (`optimizer/README.md`)](optimize/standalone_guide.md)**
- **[Developer Optimizer Index (`docs/optimize/README.md`)](optimize/README.md)**
- **[01. Core & Math](optimize/core/README.md)**: CLM, TD-CLM Triangle Defect, PRNG stochastic rounding, and bias correction.
- **[02. SIMD Vectorization](optimize/simd/README.md)**: AVX-512 and AVX2/FMA x86_64 intrinsics and Rayon parallel threadpool.
- **[03. Vulkan Compute](optimize/vulkan/README.md)**: Ash Vulkan 1.3/1.4 SPIR-V compute shaders and UMA host buffers.
- **[04. Triton AOT & GPU Drivers](optimize/triton_aot/README.md)**: AOT compilation, CUDA/HIP driver dynamic loading, and standalone C-API.
- **[05. Dispatch & Routing](optimize/dispatch/README.md)**: Hardware capability detector and 16,384-element dispatch threshold heuristic.
- **[06. Python FFI & PyO3 Bridge](optimize/python_ffi/README.md)**: PyO3 GIL release (`py.detach()`), pointer aliasing checks, and NumPy buffer protocols.
- **[07. Python Runtime & GPU Mixin](optimize/python_runtime/README.md)**: `LuminaV` class, `constants.py`, Triton JIT kernels, FSDP synchronization, and state checkpointing.

### B. LuminaMoon Data Engine
Source directories: `data/` & `data/docs/`

- **[LuminaMoon Data Engine Overview & User Guide (`data/README.md`)](dataloader/standalone_guide.md)**
- **[Developer Data Engine Index (`data/docs/README.md`)](dataloader/README.md)**
- **[01. TextCleaner & QualityFilter](dataloader/simd_and_postprocess/README.md)**: AVX-512/AVX2/NEON SIMD carriage-return stripping, JSON extraction, Shannon entropy, repetition loop, and symbol ratio filters.
- **[02. StreamDeduplicator & MinHashDeduplicator](dataloader/dedup/README.md)**: Exact streaming deduplication with `xxh3_64` (`IdentityBuildHasher` SwissTable), WAL persistence, and 64-permutation MinHash LSH fuzzy deduplication.
- **[03. PIIMasker & Data Sanitization](dataloader/simd_and_postprocess/README.md)**: Pattern-based and regex masking for emails, IPv4 addresses, phone numbers, and secret tokens.
- **[04. StateTracker & StreamCoordinate](dataloader/coord_and_metadata/README.md)**: 32-byte `StreamCoordinate` navigation, dual-slot 128-byte Ping-Pong checkpoint persistence, and multi-epoch state tracking.
- **[05. StringTensor & Memory Layouts](dataloader/coord_and_metadata/README.md)**: CPU StringView (inlined <= 12 bytes) and GPU contiguous offset/payload buffers compatible with PyTorch, NumPy, Arrow, and Polars.
- **[06. FusedPipeline & ResumablePipeline](dataloader/python_ffi_and_runtime/README.md)**: Multi-threaded Rayon single-pass preprocessing pipeline and resumable batch pipeline with background prefetcher.
- **[07. Multimodal Serialization](dataloader/multimodal_and_hf_hub/README.md)**: Serialization to JSONL, Parquet with Zstd compression and embedded image payloads, Arrow IPC, and CSV.

### C. Ra-Family Neural Architecture & Theoretical References
Source directories: `docs/nn/` & `references/`

- **[Ra-Family Overview (`docs/nn/README.md`)](nn/README.md)**: Square-root-free `RaNorm` & `RaBatchNorm2d`, rational `RaTanh` & `RaGELU`, bounded `PosRaGLU` / `RaGLU` / `TriGLU`, and compact-support `ra_softmax` & `RaAttention`.
- **[Ra-Family Standalone Guide (`docs/nn/standalone_guide.md`)](nn/standalone_guide.md)**: Drop-in Transformer block replacement, `RaBatchNorm2d` `Conv2d` parameter folding, and backend introspection.
- **[Theoretical References (`docs/references.md`)](references.md)**: Component mapping, literature comparisons across all 29 referenced papers, and BibTeX database (`references/bibliography.bib`).

---

## 3. LSO-Eval Verification & Library Compatibility Notes

Source directories: `eval/`, `docs/library_audit/`, & `data/docs/library_audit/`

- **[LSO-Eval Optimizer Evaluation & Verification (`eval/README.md`)](lso_eval.md)**: Algorithm-agnostic Z3 SMT solver invariant checks, 2D trajectory benchmarks, numerical parity tests, and Two-Class 4B LLM & Vision evaluation (`LuminaV v1.5.3–v1.6.0`).
- **[Optimizer Library Notes (`docs/library_audit/`)](library_audit/README.md)**: Version and ABI compatibility notes for Rust 1.99.1 toolchain, PyO3/Maturin, PyTorch 2.14.1, Triton JIT 3.8.0, CUDA/ROCm, and Vulkan Ash compute pipeline.
- **[Data Engine Library Notes (`data/docs/library_audit/`)](library_audit/README.md)**: Version and compatibility notes for Rust 1.99.1 / Edition 2024, `pyo3 0.30.1` (`abi3t`), Apache Arrow/Parquet/IPC `60.0.0`, `rayon 1.12.0`, `xxhash-rust 0.8.19` (`xxh3`), `crc32fast 1.5.2`, `hashbrown 0.17.1`, `parking_lot 0.12.5`, `serde 1.0.229`, and HuggingFace Hub API protocols.
