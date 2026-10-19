# 3. Apache Arrow/Parquet, `reqwest`, `tokio`, `bytes`, dan `futures`

## 3.1 Versi Stabil Terbaru
- **`parquet`**: `60.0.0` (`features = ["async"]`)
- **`arrow-array`**: `60.0.0`
- **`arrow-schema`**: `60.0.0`
- **`reqwest`**: `0.13.5` (`default-features = false`, `features = ["default-tls", "http2", "system-proxy", "json", "stream", "gzip"]`)
- **`tokio`**: `1.53.3` (`features = ["rt-multi-thread", "fs", "io-util", "sync", "time", "macros"]`)
- **`bytes`**: `1.12.1`
- **`futures`**: `0.3.34`

---

## 3.2 Riwayat Perubahan pada Rentang Versi (`arrow-rs 53.0.0` - `60.0.0`, `reqwest 0.12` - `0.13.5`, `bytes 1.6` - `1.12.1`)

### A. Apache Arrow & Parquet (`53.0.0` - `60.0.0`)
- **Promosi Tipe `Utf8View` (`StringViewArray`) dan `BinaryView` (`BinaryViewArray`)**:
  Pada seri `arrow 54.0`–`60.0.0`, pembaca Parquet modern dapat memetakan kolom string dan biner ke dalam representasi *ByteView* (12-byte inline prefix + offset buffer) di samping `StringArray`/`LargeStringArray` dan `BinaryArray`/`LargeBinaryArray`.
- **Pemisahan `ArrowReaderMetadata` untuk Pembacaan Berulang**:
  `ParquetRecordBatchReaderBuilder::try_new(file)` membaca dan mem-parsing ulang *footer metadata* Parquet pada setiap pemanggilan. API `ArrowReaderMetadata::load(&file, options)` bersama `ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata)` disediakan untuk membagikan metadata ter-cache di memori secara $O(1)$.
- **Perubahan Signature `ParquetMetaDataReader::load_and_finish` (`parquet 58.0` - `60.0.0`)**:
  Metode asinkron `ParquetMetaDataReader::load_and_finish` kini menerima parameter generik `F: MetadataSuffixFetch` (yang diimplementasikan secara *blanket* untuk `&mut T` di mana `T: AsyncFileReader`). Memanggil `.load_and_finish(self, file_size)` secara langsung di dalam metode `fn get_metadata(&'a mut self, ...)` gagal melewati pemeriksa *trait bound* karena `self` bertipe `&'a mut Self`, sehingga memerlukan *reborrow* eksplisit `&mut *self`.

### B. `reqwest` (`0.12.0` - `0.13.5`) & `tokio` (`1.38.0` - `1.53.3`)
- **Perubahan Backend TLS Bawaan pada `reqwest 0.13.x`**:
  Mulai `reqwest 0.13.0`, fitur bawaan (`default-features = true`) beralih dari `native-tls` ke `rustls` dengan kriptografi `aws-lc-sys`. Pada Windows, kompilasi `aws-lc-sys` membutuhkan CMake dan NASM eksternal yang sering memicu kegagalan *build* apabila tidak terinstal.
- **Modularisasi Fitur `json`, `stream`, dan `gzip`**:
  Metode `.json(...)`, `.bytes_stream()`, serta dekompresi otomatis pada `reqwest 0.13.5` dipisahkan ke dalam feature flag eksplisit.

### C. `bytes` (`1.6.0` - `1.12.1`): Mitigasi `RUSTSEC-2026-0007` / `CVE-2026-25541`
- **Apa yang berubah**: Pada `bytes < 1.11.1`, terdapat celah *integer overflow* pada perhitungan kapasitas di dalam `BytesMut::reserve` ketika ukuran reservasi mendekati `usize::MAX`, yang berpotensi menyebabkan alokasi buffer lebih kecil daripada ukuran yang diminta (*heap buffer overflow*).
- **Perbaikan Upstream (`1.11.1` - `1.12.1`)**: Menambahkan pemeriksaan aritmetika terikat (*checked arithmetic*) pada seluruh jalur ekspansi kapasitas `BytesMut` dan pemotongan zero-copy `Bytes::slice`.

---

## 3.3 Adaptasi pada Kode Sumber LuminaMoon

1. **Dukungan `StringViewArray` dan `BinaryViewArray` (`dataloader/src/source_local.rs` & `dataloader/src/multimodal.rs`)**:
   - Fungsi `extract_strings_from_batch` dan `extract_array_values` menangani ketiga keluarga tipe Arrow (`StringArray` / `LargeStringArray` / `StringViewArray` serta `BinaryArray` / `LargeBinaryArray` / `BinaryViewArray`).
2. **Caching `ArrowReaderMetadata` dan `RowSelection` (`dataloader/src/source_local.rs` & `dataloader/src/multimodal.rs`)**:
   - `LocalParquetSource` dan `LocalMultimodalParquetSource` memuat `ArrowReaderMetadata` sekali saat `open()` dan memadukannya dengan `ProjectionMask::roots(...)` serta `RowSelection::from(vec![RowSelector::skip(...), RowSelector::select(...)])`.
3. **Reborrow `&mut *self` pada `HttpParquetReader` (`dataloader/src/source_hf.rs`)**:
   - Implementasi `AsyncFileReader::get_metadata` untuk `HttpParquetReader` memanggil `ParquetMetaDataReader::new().with_arrow_reader_options(options).load_and_finish(&mut *self, file_size).await?` dan menyimpan hasilnya di `cached_metadata: Option<Arc<ParquetMetaData>>`.
4. **Konfigurasi `reqwest 0.13.5` dan `bytes 1.12.1` (`Cargo.toml`)**:
   - Mengonfigurasi `reqwest = { version = "0.13.5", default-features = false, features = ["default-tls", "http2", "system-proxy", "json", "stream", "gzip"] }` dan `bytes = "1.12.1"` agar kompilasi lintas platform bebas dari ketergantungan CMake/NASM `aws-lc-sys` sekaligus terlindung dari `CVE-2026-25541`.
5. **Unduhan Multi-Range Konkuren (`fetch_ranges_concurrent`) & Multi-Thread Tokio Pool (`dataloader/src/source_hf.rs` & `dataloader/src/preram.rs`)**:
   - `HttpRangeClient::fetch_ranges_concurrent` memanfaatkan `futures::stream::iter(...).buffered(concurrency)` untuk mengunduh beberapa rentang byte HTTP 206 secara paralel lintas worker dengan urutan rentang yang terjaga.
   - `HttpParquetReader::get_byte_ranges` meneruskan daftar rentang kolom Arrow/Parquet ke `fetch_ranges_concurrent(..., 8)`, sementara `RemoteLineSource` dan `PreRamDataLoader` menjalankan runtime Tokio `Builder::new_multi_thread().worker_threads(download_workers)` untuk prefetching multi-chunk paralel.

