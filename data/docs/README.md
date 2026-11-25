# LuminaMoon Data Engine Documentation

Documentation index for the **LuminaMoon Data Engine (`luminamoon.data` / `luminamoon-dataloader`)**.

---

## 1. Architecture, User Guide & Developer Specs

- **[LuminaMoon Data Engine Guide (`data/README.md`)](../README.md)**: Architecture overview, `DataPipeline`, `TokenBatch`, `SequencePacker`, `FusedPipeline`, `StringTensor` (16-byte `StringView16` + Arrow PyCapsule & GPU buffers), 256-byte `LMDL v2` Ping-Pong checkpoint recovery, two-stage `XXH3-128` deduplication, Kirsch-Mitzenmacher MinHash LSH, `AVX-512BW` / `AVX2` / `NEON` CPU SIMD normalization, and PyTorch `DataLoader` integration.
- **[Developer Modular Documentation Index (`docs/dataloader/README.md`)](../../docs/dataloader/README.md)**:
  - **[01. Coordinates & Dual Metadata (`coord.rs`, `metadata.rs`, `constants.rs`, `error.rs`)](../../docs/dataloader/coord_and_metadata/README.md)**
  - **[02. Two-Stage XXH3-128 Deduplication (`dedup.rs`)](../../docs/dataloader/dedup/README.md)**
  - **[03. CPU SIMD & Parallel Post-Processing (`simd_cpu.rs`, `postprocess.rs`)](../../docs/dataloader/simd_and_postprocess/README.md)**
  - **[04. Heuristic Quality Filter & MinHash LSH (`quality.rs`, `minhash.rs`)](../../docs/dataloader/sources_and_streaming/README.md)**
  - **[05. StringTensor, Sequence Packing & Streaming Pipelines (`string_tensor.rs`, `loader.py`, `packing.py`, `pipeline.py`, `ring_buffer.py`)](../../docs/dataloader/preram/README.md)**
  - **[06. Columnar Table Serialization (`serialize.rs`, `serialization.py`)](../../docs/dataloader/multimodal_and_hf_hub/README.md)**
  - **[07. PyO3 `abi3t` FFI Bridge & Modular Runtime (`data/src/py_bridge/`, `python/luminamoon/data/`)](../../docs/dataloader/python_ffi_and_runtime/README.md)**

---

## 2. Dependency Compatibility Notes (`docs/dependencies/`)

Workspace-wide crate versions and compatibility notes (including `arrow-*` / `parquet 60.2.0`, `xxhash-rust 0.8.19`, `hashbrown 0.17.1`, `crc32fast 1.5.2`, `parking_lot 0.12.5`, `rayon 1.12.0`, and `pyo3 0.30.1`) are maintained centrally under [`docs/dependencies/`](../../docs/dependencies/README.md):

- **[Workspace Dependency Index](../../docs/dependencies/README.md)**
- **[06. Data Engine Crates](../../docs/dependencies/06_dataloader_crates.md)**

---

## 3. Module Summary & Verification

### 3.1. Module Matrix

| Module / File | Core Responsibility | Status |
| :--- | :--- | :--- |
| [`data/src/constants.rs`](../src/constants.rs) | Magic bytes (`LMDL`, `LMWL`, `PAR1`, `ARROW1`), slot sizes, and default bounds. | Verified |
| [`data/src/coord.rs`](../src/coord.rs) | 32-byte little-endian `StreamCoordinate` (`shard_id`, `byte_offset`, `row_group_idx`, `page_idx`, `row_offset_in_page`, `global_row_idx`). | Verified |
| [`data/src/dedup.rs`](../src/dedup.rs) | `TwoStageDeduplicator` using `xxh3_128` with `IdentityBuildHasher` SwissTable. | Verified |
| [`data/src/error.rs`](../src/error.rs) | `DataLoaderError` enum with Damerau-Levenshtein edit-distance suggestions (`did_you_mean`). | Verified |
| [`data/src/metadata.rs`](../src/metadata.rs) | 256-byte `LMDL v2` Dual-Slot Ping-Pong (`Slot A` / `Slot B`) checkpoint with IEEE CRC32 (`crc32fast`), `consumed_tokens`, and `LMWL v2` WAL. | Verified |
| [`data/src/minhash.rs`](../src/minhash.rs) | Double-hashed MinHash LSH fuzzy deduplicator (adapted from Kirsch-Mitzenmacher) with auto-tuned bands and Rayon `filter_batch`. | Verified |
| [`data/src/quality.rs`](../src/quality.rs) | Shannon Character Entropy, trigram repetition detector, and symbol-to-visible-character ratio (`symbols / visible`) filters. | Verified |
| [`data/src/simd_cpu.rs`](../src/simd_cpu.rs) | `AVX-512BW`, `AVX2`, `NEON`, and Scalar kernels (`find_byte_fast`, `rfind_byte_fast`, `contains_byte_fast`). | Verified |
| [`data/src/string_tensor.rs`](../src/string_tensor.rs) | Dual-layout container for Arrow `Utf8View`-aligned `StringView16`, `__arrow_c_array__` PyCapsule, and GPU contiguous offsets/payload buffers. | Verified |
| [`data/src/postprocess.rs`](../src/postprocess.rs) | Rayon parallel UTF-8 validation, SIMD `\r` detection, JSON dotted-path extraction, and whitespace normalization. | Verified |
| [`data/src/serialize.rs`](../src/serialize.rs) | 64-bit columnar serializers (`LargeUtf8`/`LargeBinary`) for Apache Arrow IPC, Parquet (ZSTD), and NDJSON (`DataValue`). | Verified |
| [`data/src/py_bridge/`](../src/py_bridge/mod.rs) | `PyO3 0.30.1` (`abi3t`) bindings split across `mod.rs`, `state.rs`, `text.rs`, and `pipeline.rs`. | Verified |
| [`python/luminamoon/data/`](../../python/luminamoon/data/__init__.py) | Modular Python package (`loader.py`, `_sources.py`, `packing.py`, `fused.py`, `cleaner.py`, `quality.py`, `dedup.py`, `minhash.py`, `masker.py`, `string_tensor.py`, `checkpoint.py`, `collate.py`, `pipeline.py`, `ring_buffer.py`, `serialization.py`, and `fallbacks/`). | Verified |

### 3.2. Verification Commands

| Check | Command | Scope |
| :--- | :--- | :--- |
| Rust Workspace Linter | `cargo clippy --workspace --all-targets -- -D warnings` | Zero-warning Rust static lint across workspace targets |
| Rust Release Tests | `cargo test --workspace` | Rust unit and integration tests across crates |
| Python Static Type Check | `npx pyright python/luminamoon` | Static type check across Python package modules |
| Python Modular Data Tests | `pytest -q tests/test_data_core_v2.py tests/test_data_fallbacks_modular.py tests/test_data_pipeline_v2.py tests/test_data_v2_unified.py` | Data Engine unit, fallback, pipeline, and checkpoint tests |
| Python StringTensor CPU/GPU Tests | `python -B tests/test_gpu_vs_cpu_strings.py` | `StringTensor` Arrow PyCapsule and CPU/GPU buffer parity |
