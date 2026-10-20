# 3. Apache Arrow/Parquet, `reqwest`, `tokio`, `bytes`, and `futures`

## 3.1 Latest Stable Versions
- **`parquet`**: `60.0.0` (`features = ["async"]`)
- **`arrow-array`**: `60.0.0`
- **`arrow-schema`**: `60.0.0`
- **`reqwest`**: `0.13.5` (`default-features = false`, `features = ["default-tls", "http2", "system-proxy", "json", "stream", "gzip"]`)
- **`tokio`**: `1.53.3` (`features = ["rt-multi-thread", "fs", "io-util", "sync", "time", "macros"]`)
- **`bytes`**: `1.12.1`
- **`futures`**: `0.3.34`

---

## 3.2 Version Range Changelog (`arrow-rs 53.0.0` - `60.0.0`, `reqwest 0.12` - `0.13.5`, `bytes 1.6` - `1.12.1`)

### A. Apache Arrow & Parquet (`53.0.0` - `60.0.0`)
- **`Utf8View` (`StringViewArray`) and `BinaryView` (`BinaryViewArray`) Type Promotion**:
  Across `arrow 54.0`–`60.0.0`, modern Parquet readers can map string and binary columns into *ByteView* representations (12-byte inline prefix + offset buffer) alongside `StringArray`/`LargeStringArray` and `BinaryArray`/`LargeBinaryArray`.
- **Separation of `ArrowReaderMetadata` for Repeated Reads**:
  `ParquetRecordBatchReaderBuilder::try_new(file)` re-reads and re-parses the Parquet footer metadata on every invocation. The `ArrowReaderMetadata::load(&file, options)` API paired with `ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata)` allows sharing cached metadata in memory in $O(1)$ time.
- **`ParquetMetaDataReader::load_and_finish` Signature Change (`parquet 58.0` - `60.0.0`)**:
  The async method `ParquetMetaDataReader::load_and_finish` now takes a generic parameter `F: MetadataSuffixFetch` (blanket-implemented for `&mut T` where `T: AsyncFileReader`). Calling `.load_and_finish(self, file_size)` directly inside `fn get_metadata(&'a mut self, ...)` fails trait-bound resolution because `self` has type `&'a mut Self`, requiring an explicit reborrow `&mut *self`.

### B. `reqwest` (`0.12.0` - `0.13.5`) & `tokio` (`1.38.0` - `1.53.3`)
- **Default TLS Backend Change in `reqwest 0.13.x`**:
  Starting in `reqwest 0.13.0`, default features (`default-features = true`) switched from `native-tls` to `rustls` backed by `aws-lc-sys`. On Windows, compiling `aws-lc-sys` requires external CMake and NASM installations, causing build failures when absent.
- **Modularization of `json`, `stream`, and `gzip` Features**:
  `.json(...)`, `.bytes_stream()`, and automatic response decompression in `reqwest 0.13.5` are gated behind explicit feature flags.

### C. `bytes` (`1.6.0` - `1.12.1`): Mitigation of `RUSTSEC-2026-0007` / `CVE-2026-25541`
- **What changed**: In `bytes < 1.11.1`, an integer overflow vulnerability existed in capacity calculations inside `BytesMut::reserve` when reservation sizes approached `usize::MAX`, potentially allocating a smaller buffer than requested (*heap buffer overflow*).
- **Upstream fix (`1.11.1` - `1.12.1`)**: Added checked arithmetic across all `BytesMut` capacity expansion paths and zero-copy `Bytes::slice` operations.

---

## 3.3 Adaptations in LuminaMoon Source Code

1. **`StringViewArray` and `BinaryViewArray` Support (`dataloader/src/source_local.rs` & `dataloader/src/multimodal.rs`)**:
   - `extract_strings_from_batch` and `extract_array_values` handle all three Arrow string/binary type families (`StringArray` / `LargeStringArray` / `StringViewArray` and `BinaryArray` / `LargeBinaryArray` / `BinaryViewArray`).
2. **`ArrowReaderMetadata` Caching and `RowSelection` (`dataloader/src/source_local.rs` & `dataloader/src/multimodal.rs`)**:
   - `LocalParquetSource` and `LocalMultimodalParquetSource` load `ArrowReaderMetadata` once during `open()` and combine it with `ProjectionMask::roots(...)` and `RowSelection::from(vec![RowSelector::skip(...), RowSelector::select(...)])`.
3. **Explicit `&mut *self` Reborrow on `HttpParquetReader` (`dataloader/src/source_hf.rs`)**:
   - `AsyncFileReader::get_metadata` for `HttpParquetReader` calls `ParquetMetaDataReader::new().with_arrow_reader_options(options).load_and_finish(&mut *self, file_size).await?` and caches the result in `cached_metadata: Option<Arc<ParquetMetaData>>`.
4. **`reqwest 0.13.5` and `bytes 1.12.1` Configuration (`Cargo.toml`)**:
   - Configured `reqwest = { version = "0.13.5", default-features = false, features = ["default-tls", "http2", "system-proxy", "json", "stream", "gzip"] }` and `bytes = "1.12.1"` so cross-platform builds remain free of external `aws-lc-sys` CMake/NASM dependencies while mitigating `CVE-2026-25541`.
5. **Concurrent Multi-Range Downloads (`fetch_ranges_concurrent`) & Multi-Thread Tokio Pool (`dataloader/src/source_hf.rs` & `dataloader/src/preram.rs`)**:
   - `HttpRangeClient::fetch_ranges_concurrent` uses `futures::stream::iter(...).buffered(concurrency)` to download multiple HTTP 206 byte ranges in parallel across workers while preserving range ordering.
   - `HttpParquetReader::get_byte_ranges` forwards Arrow/Parquet column range lists to `fetch_ranges_concurrent`, while `RemoteLineSource` and `PreRamDataLoader` run a multi-threaded Tokio runtime (`Builder::new_multi_thread().worker_threads(download_workers)`) for parallel multi-chunk prefetching.
