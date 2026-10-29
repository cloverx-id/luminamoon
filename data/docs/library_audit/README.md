# Library Version & Changelog Audit Documentation (October 2026)

This directory documents the version-by-version changelog audit, intermediate API/ABI deprecations, security advisory mitigations (`RUSTSEC` / `CVE`), and source-code adaptations to the latest stable releases across all external libraries used by the **LuminaMoon Data Engine (`luminamoon-data v1.0.7`)** crate.

---

## Ecosystem Summary Matrix

| Component / Library | Audited Version Range | Latest Stable Version | Adaptation Status | Detailed Document |
| :--- | :--- | :--- | :--- | :--- |
| **Rust Toolchain & `std::arch`** | `1.85.0` - `1.99.1` | `1.99.1` (`edition = "2024"`) | Adopted `std::cfg_select!`, `[T]::as_chunks::<3>()`, and `unsafe_op_in_unsafe_fn` compliance across `AVX-512BW` / `AVX2` / `NEON` intrinsics | [01_rust_toolchain_simd_and_mmap.md](01_rust_toolchain_simd_and_mmap.md) |
| **System & Parallelism Crates** | `2024` - `2026` | `memmap2 0.9.11`, `rayon 1.12.0`, `thiserror 2.0.21`, `serde 1.0.229`, `serde_json 1.0.154` | Mitigated `RUSTSEC-2026-0186` (`memmap2`), indexed parallel batch processing via `rayon`, and direct byte-buffer streaming via `serde_json::to_writer` | [01_rust_toolchain_simd_and_mmap.md](01_rust_toolchain_simd_and_mmap.md) |
| **Python, PyO3 & Maturin** | `Py 3.10` - `3.15.0`, `pyo3 0.24` - `0.30.1` | `Python 3.15.0`, `pyo3 0.30.1`, `maturin 1.15.0` | Removed `extension-module`, enabled `abi3t`, migrated to `py.detach`, `Bound::cast::<T>()`, and `#[pyclass(frozen, subclass)]` | [02_python_pyo3_abi3t_and_maturin.md](02_python_pyo3_abi3t_and_maturin.md) |
| **Apache Arrow & Parquet** | `53.0.0` - `60.0.0` | `parquet 60.0.0`, `arrow-array 60.0.0`, `arrow-schema 60.0.0` | Added `StringViewArray`/`BinaryViewArray` support, `ArrowReaderMetadata` caching, and `&mut *self` reborrowing in `ParquetMetaDataReader` | [03_parquet_arrow_and_http206_streaming.md](03_parquet_arrow_and_http206_streaming.md) |
| **Async Networking & HTTP Buffers** | `reqwest 0.12` - `0.13.5`, `tokio 1.38` - `1.53.3`, `bytes 1.6` - `1.12.1` | `reqwest 0.13.5`, `tokio 1.53.3`, `bytes 1.12.1`, `futures 0.3.34` | Mitigated `CVE-2026-25541` (`bytes 1.12.1`), explicit `default-tls` + `stream` + `json` feature selection on `reqwest 0.13.5` | [03_parquet_arrow_and_http206_streaming.md](03_parquet_arrow_and_http206_streaming.md) |
| **Hashing, Tables & Synchronization** | `2024` - `2026` | `xxhash-rust 0.8.19`, `hashbrown 0.17.1`, `crc32fast 1.5.2`, `parking_lot 0.12.5` | Bypassed default `foldhash` in `hashbrown 0.17` via `IdentityBuildHasher`, enabled `vpclmulqdq` hardware acceleration in `crc32fast`, and enforced HuggingFace Hub NDJSON security | [04_xxh3_dedup_crc32_and_hf_hub_security.md](04_xxh3_dedup_crc32_and_hf_hub_security.md) |
| **Universal Formats & Decompression** | `2025` - `2026` | `arrow-ipc 60.0.0`, `half 2.7.1`, `flate2 1.1.10`, `zstd 0.14.3`, `lz4_flex 0.14.0` | Mitigated `CVE-2026-32829` (`lz4_flex 0.14.0`), `DCtx` soundness fixes (`zstd 0.14.3`), `runtime_detection` (`flate2 1.1.10`), Arrow IPC/Feather/WebDataset TAR, and Damerau-Levenshtein typo diagnostics | [05_universal_formats_compression_and_diagnostics.md](05_universal_formats_compression_and_diagnostics.md) |

---

## File Navigation Map

| File Path | Role Summary | Status |
| :--- | :--- | :--- |
| `data/docs/library_audit/README.md` | Master index and version summary matrix for all external libraries in `luminamoon-dataloader`. | Active |
| `data/docs/library_audit/01_rust_toolchain_simd_and_mmap.md` | Changelog audit for Rust 1.85-1.99.1 (`std::arch`), `memmap2 0.9.11`, `rayon 1.12.0`, `thiserror 2.0.21`, and `serde`. | Active |
| `data/docs/library_audit/02_python_pyo3_abi3t_and_maturin.md` | Changelog audit for Python 3.10-3.15, PyO3 0.24-0.30.1 (`abi3t`, `py.detach`, `Bound::cast`), and Maturin 1.15. | Active |
| `data/docs/library_audit/03_parquet_arrow_and_http206_streaming.md` | Changelog audit for Apache Arrow/Parquet 53-60.0.0, `reqwest 0.13.5`, `tokio 1.53.3`, and `bytes 1.12.1`. | Active |
| `data/docs/library_audit/04_xxh3_dedup_crc32_and_hf_hub_security.md` | Changelog audit for `xxhash-rust 0.8.19`, `hashbrown 0.17.1`, `crc32fast 1.5.2`, `parking_lot 0.12.5`, and HuggingFace Hub API specs. | Active |
| `data/docs/library_audit/05_universal_formats_compression_and_diagnostics.md` | Version-by-version changelog audit (2025-2026) for `arrow-ipc 60.0.0`, `half 2.7.1`, `flate2 1.1.10`, `zstd 0.14.3`, `lz4_flex 0.14.0`, WebDataset `.tar` parser, and typo diagnostics. | Active |
