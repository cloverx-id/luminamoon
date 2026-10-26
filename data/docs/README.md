# LuminaMoon Data Engine Documentation Portal

Welcome to the standalone documentation portal for **LuminaMoon Data Engine (`luminamoon.data` / `luminamoon-dataloader`)** under **PROJECT LOOMY**.

---

## 1. Architecture, User Guide & Developer Modular Specs

- **[LuminaMoon Data Engine Guide (`data/README.md`)](../README.md)**: Complete architecture blueprint, crate module breakdown, `FusedPipeline` single-pass Rayon execution, dual-layout `StringTensor` (CPU 16-byte `StringView16` & GPU contiguous buffers), 128-byte Ping-Pong crash recovery (`Slot A` / `Slot B`), two-stage `xxh3_64` deduplication, 64-permutation MinHash LSH, self-contained `AVX-512BW` / `AVX2` / `NEON` CPU SIMD engine, and PyTorch `ResumablePipeline` / `create_resumable_collate` integration.
- **[Developer Modular Documentation Index (`docs/dataloader/README.md`)](../../docs/dataloader/README.md)**:
  - **[01. Coordinates & Dual Metadata (`coord.rs`, `metadata.rs`, `constants.rs`, `error.rs`)](../../docs/dataloader/coord_and_metadata/README.md)**
  - **[02. Two-Stage XXH3-64 Deduplication (`dedup.rs`)](../../docs/dataloader/dedup/README.md)**
  - **[03. CPU SIMD & Parallel Post-Processing (`simd_cpu.rs`, `postprocess.rs`)](../../docs/dataloader/simd_and_postprocess/README.md)**
  - **[04. Heuristic Quality Filter & MinHash LSH (`quality.rs`, `minhash.rs`)](../../docs/dataloader/sources_and_streaming/README.md)**
  - **[05. Dual-Layout `StringTensor` & Resumable Adapters (`string_tensor.rs`, `adapters.py`)](../../docs/dataloader/preram/README.md)**
  - **[06. Columnar Table Serialization (`serialize.rs`, `serialization.py`)](../../docs/dataloader/multimodal_and_hf_hub/README.md)**
  - **[07. PyO3 `abi3t` FFI Bridge & Modular Runtime (`py_bridge.rs`, `python/luminamoon/data/`)](../../docs/dataloader/python_ffi_and_runtime/README.md)**

---

## 2. Standalone External Library Audits (`data/docs/library_audit/`)

Technical version verification, changelog tracking, CVE/RUSTSEC mitigations, and codebase adaptation audits for all external dependencies used by `luminamoon-dataloader`:

- **[Library Audit Index](library_audit/README.md)**
- **[01. Rust Toolchain, SIMD & Memory Alignment (`rayon`)](library_audit/01_rust_toolchain_simd_and_mmap.md)**: Rust 1.99.1, Edition 2024 `unsafe_op_in_unsafe_fn`, `OnceLock` ISA caching, and self-contained `AVX-512BW` / `AVX2` / `NEON` intrinsics.
- **[02. Python, PyO3 (`abi3t`), and Maturin](library_audit/02_python_pyo3_abi3t_and_maturin.md)**: Free-threaded Python (`abi3t`), `py.detach` GIL release, `Bound::cast::<T>()`, `PyBool` vs `PyInt` type ordering, and zero-copy table/postprocess FFI functions.
- **[03. Apache Arrow & Parquet (`arrow-array`, `arrow-schema`, `parquet`)](library_audit/03_parquet_arrow_and_http206_streaming.md)**: Columnar `RecordBatch` construction, ZSTD Parquet compression, and HuggingFace Dataset Viewer `Struct<bytes, path>` schema generation.
- **[04. `xxhash-rust`, `hashbrown`, `crc32fast`, and `parking_lot`](library_audit/04_xxh3_dedup_crc32_and_hf_hub_security.md)**: `xxh3_64` + `IdentityBuildHasher` SwissTable bypass, `crc32fast` Ping-Pong checkpoint slots, and low-overhead synchronization.
- **[05. Universal Formats (`arrow-ipc`, `serde_json`) and Typo Diagnostics](library_audit/05_universal_formats_compression_and_diagnostics.md)**: Arrow IPC (`ARROW1`), NDJSON serialization, and Damerau-Levenshtein developer diagnostics (`did_you_mean`).

---

## 3. Subsystem Architecture, Module Verification & Final Audit Report

### 3.1. Module Verification Matrix

| Module / File | Core Responsibility & Verified Invariants | Status |
| :--- | :--- | :--- |
| [`data/src/constants.rs`](../src/constants.rs) | Single source of truth for magic bytes (`LMDL`, `PAR1`, `ARROW1`), slot sizes, and default bounds. | Verified |
| [`data/src/coord.rs`](../src/coord.rs) | 32-byte little-endian `StreamCoordinate` (`shard_id`, `byte_offset`, `row_group_idx`, `page_idx`, `row_offset_in_page`, `global_row_idx`). | Verified |
| [`data/src/dedup.rs`](../src/dedup.rs) | `TwoStageDeduplicator` using `xxh3_64` with `IdentityBuildHasher` SwissTable. | Verified |
| [`data/src/error.rs`](../src/error.rs) | Unified `DataLoaderError` with Damerau-Levenshtein edit-distance suggestions (`did_you_mean`). | Verified |
| [`data/src/metadata.rs`](../src/metadata.rs) | 128-byte Dual-Slot Ping-Pong (`Slot A` / `Slot B`) atomic checkpoint file with IEEE CRC32 (`crc32fast`) and `.dedup` WAL. | Verified |
| [`data/src/minhash.rs`](../src/minhash.rs) | 64-permutation MinHash LSH fuzzy deduplicator and Jaccard similarity estimation. | Verified |
| [`data/src/quality.rs`](../src/quality.rs) | Shannon Character Entropy, loop repetition detector, and symbol-to-word ratio heuristic filters. | Verified |
| [`data/src/simd_cpu.rs`](../src/simd_cpu.rs) | Self-contained `AVX-512BW`, `AVX2`, `NEON`, and Scalar kernels (`find_byte_fast`, `rfind_byte_fast`) + 64-byte `CacheAlignedBuffer`. | Verified |
| [`data/src/string_tensor.rs`](../src/string_tensor.rs) | Dual-layout container for CPU `StringView16` (<=12B inline) and GPU contiguous offsets/payload buffers. | Verified |
| [`data/src/postprocess.rs`](../src/postprocess.rs) | Rayon parallel UTF-8 validation, SIMD `\r` detection, JSON dotted-path extraction, and whitespace normalization. | Verified |
| [`data/src/serialize.rs`](../src/serialize.rs) | GIL-free table serializers for Apache Arrow IPC, Parquet (ZSTD), and NDJSON (`DataValue`). | Verified |
| [`data/src/py_bridge.rs`](../src/py_bridge.rs) | `PyO3 0.30.1` (`abi3t`) bindings for `FusedPipeline`, `StringTensor`, `StateTracker`, `StreamDeduplicator`, `QualityFilter`, and `MinHashDeduplicator`. | Verified |
| [`python/luminamoon/data/`](../../python/luminamoon/data/__init__.py) | Modular Python package (`fused.py`, `cleaner.py`, `quality.py`, `dedup.py`, `minhash.py`, `masker.py`, `string_tensor.py`, `checkpoint.py`, `adapters.py`, `serialization.py`, `version.py`, and `fallbacks/`). | Verified |

### 3.2. Verification Status (Release & Test Suite)

| Verification Suite | Command | Result |
| :--- | :--- | :--- |
| Rust Workspace Linter | `cargo clippy --workspace --all-targets -- -D warnings` | **PASSED** (0 Warnings) |
| Rust Release Test Suite | `cargo test --workspace` | **PASSED** (All Data Engine & Optimizer Tests) |
| Python Static Type Check | `npx pyright python/luminamoon` | **PASSED** (0 Errors, 0 Warnings) |
| Python Modular Data Suite | `python -B tests/test_data_modular.py` | **PASSED** (End-to-End Verified) |
| Python StringTensor CPU/GPU Suite | `python -B tests/test_gpu_vs_cpu_strings.py` | **PASSED** (Zero-Copy Verified) |
