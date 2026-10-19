# Dokumentasi Audit Versi & Perubahan Pustaka (Oktober 2026)

Folder ini mendokumentasikan hasil penelusuran riwayat perubahan (*changelog*), depresiasi API/ABI pada rentang versi antara, mitigasi advisory keamanan (`RUSTSEC` / `CVE`), hingga adaptasi kode pada rilis stabil terbaru untuk seluruh pustaka eksternal (*external libraries*) yang digunakan oleh crate **LuminaMoon DataLoader (`luminamoon-dataloader v1.0.0`)**.

---

## Matriks Ringkasan Ekosistem

| Komponen / Pustaka | Rentang Versi Diaudit | Versi Stabil Terbaru | Status Adaptasi | Dokumen Rincian |
| :--- | :--- | :--- | :--- | :--- |
| **Rust Toolchain & `std::arch`** | `1.85.0` - `1.99.1` | `1.99.1` (`edition = "2024"`) | Terapkan `std::cfg_select!`, `[T]::as_chunks::<3>()`, kepatuhan `unsafe_op_in_unsafe_fn` pada intrinsik `AVX-512BW` / `AVX2` / `NEON` | [01_rust_toolchain_simd_and_mmap.md](01_rust_toolchain_simd_and_mmap.md) |
| **Crate Sistem & Paralelisme** | `2024` - `2026` | `memmap2 0.9.11`, `rayon 1.12.0`, `thiserror 2.0.21`, `serde 1.0.229`, `serde_json 1.0.154` | Mitigasi `RUSTSEC-2026-0186` (`memmap2`), pemrosesan batch terindeks `rayon`, streaming `serde_json::to_writer` | [01_rust_toolchain_simd_and_mmap.md](01_rust_toolchain_simd_and_mmap.md) |
| **Python, PyO3 & Maturin** | `Py 3.10` - `3.15.0`, `pyo3 0.24` - `0.30.1` | `Python 3.15.0`, `pyo3 0.30.1`, `maturin 1.15.0` | Hapus `extension-module`, aktifkan `abi3t`, migrasi `py.detach`, `Bound::cast::<T>()`, dan `#[pyclass(frozen, subclass)]` | [02_python_pyo3_abi3t_and_maturin.md](02_python_pyo3_abi3t_and_maturin.md) |
| **Apache Arrow & Parquet** | `53.0.0` - `60.0.0` | `parquet 60.0.0`, `arrow-array 60.0.0`, `arrow-schema 60.0.0` | Dukungan `StringViewArray`/`BinaryViewArray`, caching `ArrowReaderMetadata`, reborrow `&mut *self` pada `ParquetMetaDataReader` | [03_parquet_arrow_and_http206_streaming.md](03_parquet_arrow_and_http206_streaming.md) |
| **Jaringan Async & Buffer HTTP** | `reqwest 0.12` - `0.13.5`, `tokio 1.38` - `1.53.3`, `bytes 1.6` - `1.12.1` | `reqwest 0.13.5`, `tokio 1.53.3`, `bytes 1.12.1`, `futures 0.3.34` | Mitigasi `CVE-2026-25541` (`bytes 1.12.1`), seleksi fitur eksplisit `default-tls` + `stream` + `json` pada `reqwest 0.13.5` | [03_parquet_arrow_and_http206_streaming.md](03_parquet_arrow_and_http206_streaming.md) |
| **Hashing, Tabel & Sinkronisasi** | `2024` - `2026` | `xxhash-rust 0.8.19`, `hashbrown 0.17.1`, `crc32fast 1.5.2`, `parking_lot 0.12.5` | Bypass `foldhash` bawaan `hashbrown 0.17` via `IdentityBuildHasher`, akselerasi `vpclmulqdq` `crc32fast`, dan protokol NDJSON HuggingFace Hub | [04_xxh3_dedup_crc32_and_hf_hub_security.md](04_xxh3_dedup_crc32_and_hf_hub_security.md) |

---

## Peta Navigasi Berkas

| Jalur Berkas | Peran Singkat | Status |
| :--- | :--- | :--- |
| `dataloader/docs/library_audit/README.md` | Indeks utama dan matriks ringkasan versi seluruh pustaka eksternal `luminamoon-dataloader`. | Aktif |
| `dataloader/docs/library_audit/01_rust_toolchain_simd_and_mmap.md` | Riwayat perubahan Rust 1.85-1.99.1 (`std::arch`), `memmap2 0.9.11`, `rayon 1.12.0`, `thiserror 2.0.21`, dan `serde`. | Aktif |
| `dataloader/docs/library_audit/02_python_pyo3_abi3t_and_maturin.md` | Riwayat perubahan Python 3.10-3.15, PyO3 0.24-0.30.1 (`abi3t`, `py.detach`, `Bound::cast`), dan Maturin 1.15. | Aktif |
| `dataloader/docs/library_audit/03_parquet_arrow_and_http206_streaming.md` | Riwayat perubahan Apache Arrow/Parquet 53-60.0.0, `reqwest 0.13.5`, `tokio 1.53.3`, dan `bytes 1.12.1`. | Aktif |
| `dataloader/docs/library_audit/04_xxh3_dedup_crc32_and_hf_hub_security.md` | Riwayat perubahan `xxhash-rust 0.8.19`, `hashbrown 0.17.1`, `crc32fast 1.5.2`, `parking_lot 0.12.5`, dan spesifikasi API HuggingFace Hub. | Aktif |
