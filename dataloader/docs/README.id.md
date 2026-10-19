# Portal Dokumentasi LuminaMoon DataLoader

Selamat datang di portal dokumentasi mandiri untuk **LuminaMoon DataLoader (`luminamoon-dataloader`)** di bawah ekosistem **PROJECT LUMI**.

[English Version (README.md)](README.md)

---

## 1. Arsitektur, Panduan Pengguna & Spesifikasi Modular Developer

- **[Panduan LuminaMoon DataLoader (`dataloader/README.md`)](../README.md)**: Cetak biru arsitektur lengkap, pembedahan modul crate, streaming `mmap` lokal $O(1)$ dan HTTP 206 Range konkuren, ring buffer watermark Pre-RAM 100-batch, pemulihan crash ping-pong dual-metadata (`Slot A` / `Slot B`), deduplikasi dua tahap `xxh3_64`, mesin SIMD CPU mandiri `AVX-512BW` / `AVX2` / `NEON`, serta integrasi resmi ekosistem HuggingFace.
- **[Indeks Dokumentasi Modular Developer (`docs/dataloader/README.id.md`)](../../docs/dataloader/README.id.md)** / **[English Version (`docs/dataloader/README.md`)](../../docs/dataloader/README.md)**:
  - **[01. Koordinat & Dual Metadata (`coord.rs`, `metadata.rs`)](../../docs/dataloader/coord_and_metadata/README.id.md)**
  - **[02. Deduplikasi Dua Tahap (`dedup.rs`)](../../docs/dataloader/dedup/README.id.md)**
  - **[03. SIMD CPU & Post-Processing Paralel (`simd_cpu.rs`, `postprocess.rs`)](../../docs/dataloader/simd_and_postprocess/README.id.md)**
  - **[04. Sumber Shard Lokal & HTTP 206 (`source_local.rs`, `source_hf.rs`)](../../docs/dataloader/sources_and_streaming/README.id.md)**
  - **[05. Ring Buffer Watermark Pre-RAM (`preram.rs`)](../../docs/dataloader/preram/README.id.md)**
  - **[06. Mesin Multimodal & Klien HuggingFace Hub (`multimodal.rs`, `hf_hub.rs`)](../../docs/dataloader/multimodal_and_hf_hub/README.id.md)**
  - **[07. Jembatan FFI & Runtime Python (`py_bridge.rs`, `dataloader/__init__.py`)](../../docs/dataloader/python_ffi_and_runtime/README.id.md)**

---

## 2. Audit Pustaka Eksternal Mandiri (`dataloader/docs/library_audit/`)

Verifikasi teknis versi, penelusuran riwayat perubahan (*changelog*), mitigasi kerentanan CVE/RUSTSEC, dan audit adaptasi kode untuk seluruh dependensi eksternal `luminamoon-dataloader`:

- **[Indeks Audit Pustaka](library_audit/README.md)**
- **[01. Rust Toolchain, SIMD & Memory Mapping (`memmap2`, `rayon`)](library_audit/01_rust_toolchain_simd_and_mmap.md)**: Rust 1.99.1, Edition 2024 `unsafe_op_in_unsafe_fn`, caching ISA `OnceLock`, intrinsik mandiri `AVX-512BW` / `AVX2` / `NEON`, dan invarian keamanan `memmap2` 0.9.11.
- **[02. Python, PyO3 (`abi3t`), dan Maturin](library_audit/02_python_pyo3_abi3t_and_maturin.md)**: Python *free-threaded* (`abi3t`), pelepasan GIL `py.detach`, `Bound::cast::<T>()`, urutan evaluasi tipe `PyBool` vs `PyInt`, dan fungsi FFI tabel/postprocess.
- **[03. Apache Arrow/Parquet, `reqwest`, `tokio`, `bytes`, dan `futures`](library_audit/03_parquet_arrow_and_http206_streaming.md)**: Lompatan halaman $O(1)$ `RowSelection`, `AsyncFileReader::get_byte_ranges`, unduhan multi-range HTTP 206 konkuren (`fetch_ranges_concurrent`), pool worker Tokio multi-thread (`download_workers`), serta mitigasi `CVE-2026-25541` (`bytes 1.12.1`).
- **[04. `xxhash-rust`, `hashbrown`, `crc32fast`, `parking_lot`, dan API HuggingFace Hub](library_audit/04_xxh3_dedup_crc32_and_hf_hub_security.md)**: Bypass SwissTable `xxh3_64` + `IdentityBuildHasher`, slot checkpoint Ping-Pong `crc32fast`, resolusi rahasia Colab/Kaggle otomatis, paginasi `Link` RFC 5988, dan keamanan unggahan hibrida LFS/Xet + NDJSON.

---

## 3. Arsitektur Subsistem, Matriks Verifikasi Modul & Laporan Audit Akhir (Langkah 1–9)

### 3.1. Matriks Verifikasi Modul

| Modul / Berkas | Tanggung Jawab Inti & Invarian Terverifikasi | Status |
| :--- | :--- | :--- |
| [`dataloader/src/coord.rs`](../src/coord.rs) | `StreamCoordinate` *little-endian* 32-byte (`shard_id`, `byte_offset`, `row_group_idx`, `page_idx`, `row_offset_in_page`, `global_row_idx`) untuk resume deterministik $O(1)$. | Terverifikasi |
| [`dataloader/src/dedup.rs`](../src/dedup.rs) | `TwoStageDeduplicator` yang memisahkan `staged_lookup` (`HashSet<u64, IdentityBuildHasher>`) dari `committed_hashes` (`HashSet<u64, IdentityBuildHasher>`) berbasis `xxh3_64`. | Terverifikasi |
| [`dataloader/src/metadata.rs`](../src/metadata.rs) | Berkas checkpoint atomik Ping-Pong 128-byte (`Slot A` / `Slot B`) dengan IEEE CRC32 (`crc32fast`) dan *write-ahead log* `.dedup`. | Terverifikasi |
| [`dataloader/src/simd_cpu.rs`](../src/simd_cpu.rs) | Kernel mandiri `AVX-512BW` (64B ZMM), `AVX2` (32B YMM), `NEON` (16B), dan Scalar (`find_byte_fast`, `rfind_byte_fast`, `contains_byte_fast`) + `CacheAlignedBuffer` 64-byte. | Terverifikasi |
| [`dataloader/src/postprocess.rs`](../src/postprocess.rs) | Validasi UTF-8 paralel Rayon, deteksi `\r` SIMD (`contains_byte_fast`), ekstraksi field JSON, dan penyaringan/pemotongan panjang byte. | Terverifikasi |
| [`dataloader/src/source_local.rs`](../src/source_local.rs) | `LocalMmapLineSource` (`memmap2` + pemindaian newline SIMD) dan `LocalParquetSource` (cache `ArrowReaderMetadata` + lompatan baris `RowSelection`). | Terverifikasi |
| [`dataloader/src/source_hf.rs`](../src/source_hf.rs) | `HttpRangeClient` (`fetch_range`, `fetch_ranges_concurrent`), `HttpParquetReader` (`get_byte_ranges`), `RemoteLineSource` (`download_workers`), dan `RemoteParquetSource`. | Terverifikasi |
| [`dataloader/src/preram.rs`](../src/preram.rs) | Ring buffer 100-batch `PreRamDataLoader`, isi ulang watermark (`capacity - refill_trigger`), pool worker Tokio multi-thread (`download_workers`), dan penggabungan batch lintas-shard. | Terverifikasi |
| [`dataloader/src/multimodal.rs`](../src/multimodal.rs) | `MultimodalValue` (`Text`, `Bytes`, `Json`, `Int`, `Float`, `Bool`), `LocalMultimodalParquetSource`, `RemoteMultimodalParquetSource`, serta serializer Parquet/JSONL/JSON. | Terverifikasi |
| [`dataloader/src/hf_hub.rs`](../src/hf_hub.rs) | `HfCredentials` (`Debug` tersensor, proteksi token HTTPS) dan klien komit NDJSON atomik `HfHubClient` (`upload_bytes`, `upload_table_auto_format`). | Terverifikasi |
| [`dataloader/src/py_bridge.rs`](../src/py_bridge.rs) | Kelas `PyO3 0.30.1` (`abi3t`) (`DataStreamLoader`, `MultimodalParquetLoader`, `HfHubUploader`) serta fungsi bebas-GIL (`serialize_table_to_bytes`, `postprocess_and_dedup_records`). | Terverifikasi |
| [`python/luminamoon/dataloader/__init__.py`](../../python/luminamoon/dataloader/__init__.py) | API tingkat tinggi: `stream`, `from_hf_dataset`, `expand_hf_sources`, `push_to_hub`, `resolve_hf_secrets` (Colab/Kaggle/env), hook `tokenizer`, dan `auto_commit` tertunda. | Terverifikasi |

### 3.2. Perbaikan Utama Audit Keamanan & Rekayasa

1. **Proteksi Token Bearer HTTPS & Kredensial Tersensor**:
   - Sisi Rust (`HfCredentials::validate_url_security` pada [`dataloader/src/hf_hub.rs`](../src/hf_hub.rs)) maupun Python (`_is_loopback_endpoint` pada [`python/luminamoon/dataloader/__init__.py`](../../python/luminamoon/dataloader/__init__.py)) menolak pengiriman `Authorization: Bearer <token>` melalui koneksi `http://` teks biasa kecuali menuju alamat loopback (`127.0.0.1`, `localhost`, `[::1]`).
2. **Paginasi RFC 5988 `Link: <...>; rel="next"` pada Penemuan Otomatis Repositori**:
   - `_list_hf_repo_files` menelusuri header `Link` hingga 50 halaman saat memindai repositori dataset HuggingFace berukuran besar melalui Tree REST API, sambil memverifikasi keamanan HTTPS pada setiap URL halaman berikutnya.
3. **Eksekusi Tunggal Bebas-GIL Rayon + SIMD (`postprocess_and_dedup_records`)**:
   - Mendukung argumen `deduplicate: bool = true` langsung di sisi Rust sehingga pipeline `from_hf_dataset` baik dengan maupun tanpa deduplikasi dieksekusi dalam satu lintasan paralel Rayon + SIMD dengan GIL dilepas (`py.detach`).
4. **Keamanan Crash `auto_commit` Tertunda**:
   - Iterator Python `DataStreamLoader` hanya menyimpan komit batch $N$ ke berkas checkpoint Ping-Pong `.meta` setelah tubuh loop pelatihan selesai memproses batch $N$ dan meminta batch $N+1$.

### 3.3. Status Verifikasi Akhir (Rilis & Suite Pengujian)

| Suite Verifikasi | Perintah | Hasil |
| :--- | :--- | :--- |
| Linter Workspace Rust | `cargo clippy --workspace --all-targets -- -D warnings` | **PASSED** (0 Warning) |
| Kompilasi Rilis Rust | `cargo build --release --workspace` | **PASSED** (`_luminamoon_optimizer_rs.pyd` Terbangun) |
| Suite Uji Rilis Rust | `cargo test --release --workspace` | **PASSED** (15 Tes DataLoader + 29 Tes Optimizer) |
| Pemeriksaan Tipe Python | `npx pyright python/luminamoon` | **PASSED** (0 Error, 0 Warning) |
| Suite Python DataLoader | `python -B dataloader/tests/test_dataloader_python.py` | **PASSED** (Langkah 4–9 Terverifikasi End-to-End) |
| Suite Python Optimizer | `python -B optimizer/tests/test_standalone_numpy.py` | **PASSED** (CLM, TD-CLM, Resume & Audit Terverifikasi) |
