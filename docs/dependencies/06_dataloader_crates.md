# 6. Data Engine Dependencies (`arrow-*`, `parquet`, `xxhash-rust`, `hashbrown`, `crc32fast`, `parking_lot`)

## 6.1 Locked Workspace Versions (`data/Cargo.toml`)
- **`arrow-array`**: `60.2.0` (`features = ["ffi"]`)
- **`arrow-buffer`**: `60.2.0`
- **`arrow-schema`**: `60.2.0`
- **`arrow-ipc`**: `60.2.0`
- **`parquet`**: `60.2.0` (`default-features = false`, `features = ["arrow", "zstd"]`)
- **`xxhash-rust`**: `0.8.19` (`features = ["xxh3"]`)
- **`hashbrown`**: `0.17.1`
- **`crc32fast`**: `1.5.2`
- **`parking_lot`**: `0.12.5`
- **Shared Core Crates**: `pyo3 0.30.1` (`abi3-py310`, `abi3t`), `rayon 1.12.0`, `serde 1.0.229`, `serde_json 1.0.154`, `thiserror 2.0.21`

---

## 6.2 Role & Adaptation Notes

### A. Apache Arrow & Parquet (`60.2.0`)
- Synchronized across `arrow-array`, `arrow-buffer`, `arrow-schema`, `arrow-ipc`, and `parquet` at `60.2.0` (includes the fixes for **RUSTSEC-2026-0595** in `parquet` and **RUSTSEC-2026-0442 / CVE-2026-111959** in transitive `arrow-data`).
- `parquet` disables default features and enables only `["arrow", "zstd"]`, omitting unused codecs (`brotli`, `flate2`, `lz4_flex`, `snap`).
- `arrow-array` enables the `ffi` feature so `StringTensor` exposes the Apache Arrow C-Data Interface (`__arrow_c_array__` PyCapsule with `Utf8View` layout) directly to PyArrow and Polars.
- `data/src/serialize.rs` writes 64-bit `LargeUtf8` and `LargeBinary` columns (avoiding 2 GiB 32-bit offset limits) to ZSTD-compressed Parquet (`PAR1`) and Arrow IPC (`ARROW1`), with optional HuggingFace Dataset Viewer image struct encoding (`Struct<bytes: LargeBinary, path: LargeUtf8>`).

### B. Hashing, Indexing & Integrity (`xxhash-rust`, `hashbrown`, `crc32fast`, `parking_lot`)
- **`xxhash-rust 0.8.19` (`xxh3`)**: Provides 128-bit `xxh3_128` digests for `StreamDeduplicator` (`LMWL v2` WAL) and `StringTensor.hash_all()`, plus single-hash seeds for Kirsch-Mitzenmacher universal MinHash LSH (`data/src/minhash.rs`).
- **`hashbrown 0.17.1`**: Powers the SwissTable backing `TwoStageDeduplicator` with a zero-rehash `IdentityBuildHasher` over pre-mixed 128-bit digests.
- **`crc32fast 1.5.2`**: Hardware-accelerated IEEE 802.3 CRC32 verification for 256-byte dual-slot Ping-Pong `LMDL v2` checkpoints (`data/src/metadata.rs`).
- **`parking_lot 0.12.5`**: Compact synchronization primitives for concurrent Rust state wrappers (`PyStateTracker`, `PyStreamDeduplicator`, `PyMinHashDeduplicator`).
