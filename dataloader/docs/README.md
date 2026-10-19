# LuminaMoon DataLoader Documentation Portal

Welcome to the standalone documentation portal for **LuminaMoon DataLoader (`luminamoon-dataloader`)** under **PROJECT LUMI**.

[Versi Bahasa Indonesia (README.id.md)](README.id.md)

---

## 1. Architecture, User Guide & Developer Modular Specs

- **[LuminaMoon DataLoader Guide (`dataloader/README.md`)](../README.md)**: Complete architecture blueprint, crate module breakdown, $O(1)$ local `mmap` and concurrent HTTP 206 Range streaming, 100-batch Pre-RAM watermark ring buffer, dual-metadata ping-pong crash recovery (`Slot A` / `Slot B`), two-stage `xxh3_64` deduplication, self-contained `AVX-512BW` / `AVX2` / `NEON` CPU SIMD engine, and official HuggingFace ecosystem integration.
- **[Developer Modular Documentation Index (`docs/dataloader/README.md`)](../../docs/dataloader/README.md)** / **[Versi Indonesia (`docs/dataloader/README.id.md`)](../../docs/dataloader/README.id.md)**:
  - **[01. Coordinates & Dual Metadata (`coord.rs`, `metadata.rs`)](../../docs/dataloader/coord_and_metadata/README.md)**
  - **[02. Two-Stage Deduplication (`dedup.rs`)](../../docs/dataloader/dedup/README.md)**
  - **[03. CPU SIMD & Parallel Post-Processing (`simd_cpu.rs`, `postprocess.rs`)](../../docs/dataloader/simd_and_postprocess/README.md)**
  - **[04. Local & HTTP 206 Shard Sources (`source_local.rs`, `source_hf.rs`)](../../docs/dataloader/sources_and_streaming/README.md)**
  - **[05. Pre-RAM Watermark Ring Buffer (`preram.rs`)](../../docs/dataloader/preram/README.md)**
  - **[06. Multimodal Engine & HuggingFace Hub Client (`multimodal.rs`, `hf_hub.rs`)](../../docs/dataloader/multimodal_and_hf_hub/README.md)**
  - **[07. Python FFI Bridge & Runtime (`py_bridge.rs`, `dataloader/__init__.py`)](../../docs/dataloader/python_ffi_and_runtime/README.md)**

---

## 2. Standalone External Library Audits (`dataloader/docs/library_audit/`)

Technical version verification, changelog tracking, CVE/RUSTSEC mitigations, and codebase adaptation audits for all external dependencies used by `luminamoon-dataloader`:

- **[Library Audit Index](library_audit/README.md)**
- **[01. Rust Toolchain, SIMD & Memory Mapping (`memmap2`, `rayon`)](library_audit/01_rust_toolchain_simd_and_mmap.md)**: Rust 1.99.1, Edition 2024 `unsafe_op_in_unsafe_fn`, `OnceLock` ISA caching, self-contained `AVX-512BW` / `AVX2` / `NEON` intrinsics, and `memmap2` 0.9.11 safety invariants.
- **[02. Python, PyO3 (`abi3t`), and Maturin](library_audit/02_python_pyo3_abi3t_and_maturin.md)**: Free-threaded Python (`abi3t`), `py.detach` GIL release, `Bound::cast::<T>()`, `PyBool` vs `PyInt` type ordering, and zero-copy table/postprocess FFI functions.
- **[03. Apache Arrow/Parquet, `reqwest`, `tokio`, `bytes`, and `futures`](library_audit/03_parquet_arrow_and_http206_streaming.md)**: `RowSelection` $O(1)$ page skipping, `AsyncFileReader::get_byte_ranges`, concurrent HTTP 206 multi-range downloads (`fetch_ranges_concurrent`), multi-threaded Tokio worker pool (`download_workers`), and `CVE-2026-25541` (`bytes 1.12.1`) mitigation.
- **[04. `xxhash-rust`, `hashbrown`, `crc32fast`, `parking_lot`, and HuggingFace Hub API](library_audit/04_xxh3_dedup_crc32_and_hf_hub_security.md)**: `xxh3_64` + `IdentityBuildHasher` SwissTable bypass, `crc32fast` Ping-Pong checkpoint slots, Colab/Kaggle secret auto-resolution, RFC 5988 `Link` pagination, and hybrid LFS/Xet + NDJSON upload safety.

---

## 3. Subsystem Architecture, Module Verification & Final Audit Report (Steps 1–9)

### 3.1. Module Verification Matrix

| Module / File | Core Responsibility & Verified Invariants | Status |
| :--- | :--- | :--- |
| [`dataloader/src/coord.rs`](../src/coord.rs) | 32-byte little-endian `StreamCoordinate` (`shard_id`, `byte_offset`, `row_group_idx`, `page_idx`, `row_offset_in_page`, `global_row_idx`) for $O(1)$ deterministic resume. | Verified |
| [`dataloader/src/dedup.rs`](../src/dedup.rs) | `TwoStageDeduplicator` separating `staged_lookup` (`HashSet<u64, IdentityBuildHasher>`) from `committed_hashes` (`HashSet<u64, IdentityBuildHasher>`) using `xxh3_64`. | Verified |
| [`dataloader/src/metadata.rs`](../src/metadata.rs) | 128-byte Ping-Pong (`Slot A` / `Slot B`) atomic checkpoint file with IEEE CRC32 (`crc32fast`) and `.dedup` write-ahead log. | Verified |
| [`dataloader/src/simd_cpu.rs`](../src/simd_cpu.rs) | Self-contained `AVX-512BW` (64B ZMM), `AVX2` (32B YMM), `NEON` (16B), and Scalar kernels (`find_byte_fast`, `rfind_byte_fast`, `contains_byte_fast`) + 64-byte `CacheAlignedBuffer`. | Verified |
| [`dataloader/src/postprocess.rs`](../src/postprocess.rs) | Rayon parallel UTF-8 validation, SIMD `\r` detection (`contains_byte_fast`), JSON field extraction, and byte-length filtering/truncation. | Verified |
| [`dataloader/src/source_local.rs`](../src/source_local.rs) | `LocalMmapLineSource` (`memmap2` + SIMD newline scan) and `LocalParquetSource` (`ArrowReaderMetadata` cache + `RowSelection` row-group skipping). | Verified |
| [`dataloader/src/source_hf.rs`](../src/source_hf.rs) | `HttpRangeClient` (`fetch_range`, `fetch_ranges_concurrent`), `HttpParquetReader` (`get_byte_ranges`), `RemoteLineSource` (`download_workers`), and `RemoteParquetSource`. | Verified |
| [`dataloader/src/preram.rs`](../src/preram.rs) | `PreRamDataLoader` 100-batch ring buffer, watermark refill (`capacity - refill_trigger`), multi-threaded Tokio worker pool (`download_workers`), and cross-shard batch coalescing. | Verified |
| [`dataloader/src/multimodal.rs`](../src/multimodal.rs) | `MultimodalValue` (`Text`, `Bytes`, `Json`, `Int`, `Float`, `Bool`), `LocalMultimodalParquetSource`, `RemoteMultimodalParquetSource`, and Parquet/JSONL/JSON serializers. | Verified |
| [`dataloader/src/hf_hub.rs`](../src/hf_hub.rs) | `HfCredentials` (redacted `Debug`, HTTPS token guard) and `HfHubClient` atomic NDJSON commit API (`upload_bytes`, `upload_table_auto_format`). | Verified |
| [`dataloader/src/py_bridge.rs`](../src/py_bridge.rs) | `PyO3 0.30.1` (`abi3t`) classes (`DataStreamLoader`, `MultimodalParquetLoader`, `HfHubUploader`) and functions (`serialize_table_to_bytes`, `postprocess_and_dedup_records`). | Verified |
| [`python/luminamoon/dataloader/__init__.py`](../../python/luminamoon/dataloader/__init__.py) | High-level API: `stream`, `from_hf_dataset`, `expand_hf_sources`, `push_to_hub`, `resolve_hf_secrets` (Colab/Kaggle/env), `tokenizer` hook, and deferred `auto_commit`. | Verified |

### 3.2. Key Security & Engineering Audit Fixes

1. **HTTPS Bearer Token Guard & Redacted Credentials**:
   - Both Rust (`HfCredentials::validate_url_security` in [`dataloader/src/hf_hub.rs`](../src/hf_hub.rs)) and Python (`_is_loopback_endpoint` in [`python/luminamoon/dataloader/__init__.py`](../../python/luminamoon/dataloader/__init__.py)) refuse to transmit `Authorization: Bearer <token>` over plaintext `http://` unless targeting loopback (`127.0.0.1`, `localhost`, `[::1]`).
2. **RFC 5988 `Link: <...>; rel="next"` Pagination on Repository Auto-Discovery**:
   - `_list_hf_repo_files` follows paginated `Link` headers up to 50 pages when listing large HuggingFace dataset repositories via the Tree REST API, verifying HTTPS transport on every next-page URL.
3. **Single-Batch GIL-Free Rayon + SIMD Execution (`postprocess_and_dedup_records`)**:
   - Supports `deduplicate: bool = true` directly in Rust so both deduplicated and non-deduplicated `from_hf_dataset` pipelines execute in one parallel Rayon + SIMD pass with the GIL detached (`py.detach`).
4. **Deferred `auto_commit` Crash Safety**:
   - `DataStreamLoader` Python iterators commit batch $N$ to the Ping-Pong `.meta` checkpoint only after the training loop body finishes processing batch $N$ and requests batch $N+1$.

### 3.3. Final Verification Status (Release & Test Suite)

| Verification Suite | Command | Result |
| :--- | :--- | :--- |
| Rust Workspace Linter | `cargo clippy --workspace --all-targets -- -D warnings` | **PASSED** (0 Warnings) |
| Rust Release Build | `cargo build --release --workspace` | **PASSED** (`_luminamoon_optimizer_rs.pyd` Built) |
| Rust Release Test Suite | `cargo test --release --workspace` | **PASSED** (15 DataLoader + 29 Optimizer Tests) |
| Python Static Type Check | `npx pyright python/luminamoon` | **PASSED** (0 Errors, 0 Warnings) |
| Python DataLoader Suite | `python -B dataloader/tests/test_dataloader_python.py` | **PASSED** (Steps 4–9 End-to-End Verified) |
| Python Optimizer Suite | `python -B optimizer/tests/test_standalone_numpy.py` | **PASSED** (CLM, TD-CLM, Resume & Audit Verified) |
