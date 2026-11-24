# Loomy (luminamoon) Documentation

<div class="portal-badges">
  <span class="pill-badge">LuminaV Optimizer v2.0.0</span>
  <span class="pill-badge pill-badge-green">Data Engine v2.0.0</span>
  <span class="pill-badge pill-badge-cyan">LSO-Eval Verified</span>
  <span class="pill-badge pill-badge-cyan">Rust 1.99 + Python ABI3T</span>
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
    <p>DataPipeline, document-isolated SequencePacker, StringTensor Arrow PyCapsule, XXH3-128 WAL, and 256B LMDL v2 checkpoints.</p>
  </a>
  <a class="friendly-card" href="lso_eval/">
    <h3>LSO-Eval</h3>
    <p>Z3 SMT safety checks, 2D trajectory benchmarks, 1-Pass EMA parity, and Qwen3.5-4B & Vision Two-Class evaluation.</p>
  </a>
  <a class="friendly-card" href="en/03_configuration_api/">
    <h3>API Reference</h3>
    <p>Signatures, parameter reference tables, validation constraints, and return types for all modules.</p>
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

- **[01. Getting Started](en/01_getting_started.md)**: System requirements (Rust 1.95+ / Edition 2024), installation, optional extras (`torch`, `gpu`, `data`, `numpy`, `all`), CLI (`luminamoon --help`), hardware probes (`LuminaV.detect_hardware()`, `lm.data_cpu_isa()`), and quickstarts for NumPy, PyTorch, and `DataPipeline`.
- **[02. User Guide](en/02_user_guide.md)**: Optimizer buffer modes (2-Buffer CLM, 1-Buffer TD-CLM, and Triangle Family T1/T2), CWD, RTV, `bound_floor`, mixed-precision & stochastic rounding, FSDP1/FSDP2 (`DTensor`)/HSDP, plus `DataPipeline`, `SequencePacker`, `FusedPipeline`, `StringTensor`, XXH3-128 WAL & MinHash deduplication, PII masking, and 256B `LMDL v2` checkpoints.
- **[03. Configuration & API Reference](en/03_configuration_api.md)**: Parameter reference tables, public methods, and validation rules for `LuminaV`, `DataPipeline`, `TokenBatch`, `SequencePacker`, `FusedPipeline`, `StringTensor`, `StateTracker`, `luminamoon.nn`, and `luminamoon.cli`.
- **[04. Architecture & Hardware Acceleration](en/04_architecture_and_hardware.md)**: Dispatch hierarchy, Optimizer Rust crates (`core`, `simd`, `cubecl`, `dispatch`, `triton-aot`), Data Engine Rust modules, buffer memory protocols, and free-threaded Python (`abi3t` / PEP 703).

---

## 2. Developer Modular Documentation

Technical documentation for contributors working on the Rust crates, GPU kernels, and Python bindings (`docs/optimize/` & `docs/dataloader/`):

### A. LuminaV Optimizer
Source directories: `docs/optimize/` & `optimizer/`

- **[LuminaV Optimizer Overview & User Guide (`optimizer/README.md`)](optimize/standalone_guide.md)**
- **[Developer Optimizer Index (`docs/optimize/README.md`)](optimize/README.md)**
- **[01. Core & Math](optimize/core/README.md)**: CLM, TD-CLM Triangle Defect, Triangle Family (`T1`/`T2`), CWD, RTV, `bound_floor`, and MurmurHash3 + 3D Weyl-Kronecker stochastic rounding.
- **[02. SIMD Vectorization](optimize/simd/README.md)**: AVX-512, AVX2/FMA, and ARM64 NEON intrinsics with Rayon parallelism.
- **[03. CubeCL Portable GPU](optimize/cubecl/README.md)**: Optional CubeCL portable GPU backend (`wgpu`, Vulkan, Metal, CUDA, HIP, CPU) for optimizer and NN kernels.
- **[04. Triton AOT & GPU Drivers](optimize/triton_aot/README.md)**: 182 compiled CUDA/ROCm ELF64 binaries, dynamic driver loading, and standalone C-API.
- **[05. Dispatch & Routing](optimize/dispatch/README.md)**: Hardware capability detector and 16,384-element dispatch threshold heuristic.
- **[06. Python FFI & PyO3 Bridge](optimize/python_ffi/README.md)**: PyO3 GIL release (`py.detach()`), byte-range pointer anti-aliasing, and NumPy/PyTorch buffer protocols.
- **[07. Python Runtime & GPU Mixin](optimize/python_runtime/README.md)**: `LuminaV` class, modular backends (`cpu.py`, `gpu.py`, `eager.py`, `fsdp.py`, `triton/`), FSDP1/FSDP2 synchronization, and state checkpointing.

### B. LuminaMoon Data Engine
Source directories: `data/` & `docs/dataloader/`

- **[LuminaMoon Data Engine Overview & User Guide (`data/README.md`)](dataloader/standalone_guide.md)**
- **[Developer Data Engine Index (`docs/dataloader/README.md`)](dataloader/README.md)**
- **[01. TextCleaner & QualityFilter](dataloader/simd_and_postprocess/README.md)**: AVX-512/AVX2/NEON SIMD carriage-return stripping, JSON extraction, Shannon entropy, repetition loop, and symbol ratio filters.
- **[02. StreamDeduplicator & MinHashDeduplicator](dataloader/dedup/README.md)**: Exact streaming deduplication with `XXH3-128` (`IdentityBuildHasher` SwissTable), `O(delta)` `LMWL v2` WAL persistence, and Kirsch-Mitzenmacher MinHash LSH fuzzy deduplication.
- **[03. PIIMasker & Data Sanitization](dataloader/simd_and_postprocess/README.md)**: Code-safe pattern masking with strict boundaries and Luhn card validation.
- **[04. StateTracker & StreamCoordinate](dataloader/coord_and_metadata/README.md)**: 32-byte `StreamCoordinate` navigation, 256-byte `LMDL v2` Ping-Pong checkpoint persistence, and `consumed_tokens` tracking.
- **[05. StringTensor, Packing & Streaming Pipelines](dataloader/preram/README.md)**: `StringView16` + Arrow PyCapsule (`__arrow_c_array__`), `DataPipeline`, `SequencePacker`, `TokenBatch`, `ResumablePipeline`, and `TokenRingBuffer`.
- **[06. FusedPipeline & Modular Runtime](dataloader/python_ffi_and_runtime/README.md)**: Multi-threaded Rayon single-pass preprocessing pipeline, PyO3 FFI bridge, and pure-Python `fallbacks/`.
- **[07. Columnar Serialization](dataloader/multimodal_and_hf_hub/README.md)**: 64-bit columnar serialization to JSONL, ZSTD Parquet (`LargeUtf8`/`LargeBinary` + HuggingFace image structs), and Arrow IPC.

### C. Ra-Family Neural Architecture & Theoretical References
Source directories: `docs/nn/` & `references/`

- **[Ra-Family Overview (`docs/nn/README.md`)](nn/README.md)**: Square-root-free `RaNorm` & `RaBatchNorm2d`, rational `RaTanh` & `RaGELU`, `RaSwiGLU` / `RaGLU`, compact-support `ra_softmax` & `RaAttention`, `LinearCrossEntropy`, and `RaMoE`.
- **[Ra-Family Standalone Guide (`docs/nn/standalone_guide.md`)](nn/standalone_guide.md)**: Drop-in Transformer block replacement, `RaBatchNorm2d` `Conv2d` parameter folding, and backend introspection (`is_cubecl_available()`).
- **[Theoretical References (`docs/references.md`)](references.md)**: Component mapping, literature comparisons across the referenced papers, and BibTeX database (`references/bibliography.bib`).

---

## 3. LSO-Eval Verification & Dependency Compatibility Notes

Source directories: `eval/` & `docs/dependencies/`

- **[LSO-Eval Optimizer Evaluation & Verification (`eval/README.md`)](lso_eval.md)**: Algorithm-agnostic Z3 SMT solver invariant checks, 2D trajectory benchmarks, numerical parity tests, and Two-Class 4B LLM & Vision evaluation (`LuminaV v2.0.0`).
- **[Workspace Dependency Notes (`docs/dependencies/`)](dependencies/README.md)**: Version and ABI compatibility notes for Rust toolchain, PyO3/Maturin, PyTorch, Triton 3.8.0, CUDA/ROCm, CubeCL `0.11.4`, and Data Engine crates (`arrow-*`/`parquet 60.2.0`, `xxhash-rust 0.8.19`, `hashbrown 0.17.1`, `crc32fast 1.5.2`, `parking_lot 0.12.5`).
