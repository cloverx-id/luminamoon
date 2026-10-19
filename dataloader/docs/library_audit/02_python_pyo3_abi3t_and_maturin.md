# 2. Python, PyO3 (`abi3t`), dan Maturin

## 2.1 Versi Stabil Terbaru
- **Python**: `3.15.0` (mendukung CPython standar dan *free-threaded* `Py_GIL_DISABLED`)
- **`pyo3`**: `0.30.1` (`features = ["abi3-py310", "abi3t"]`)
- **`maturin`**: `1.15.0` (`requires = ["maturin>=1.15,<2.0"]`)

---

## 2.2 Riwayat Perubahan pada Rentang Versi (`pyo3 0.24.0` - `0.30.1` & `maturin 1.8` - `1.15.0`)

### A. PyO3 0.25.0 - 0.26.0: Depresiasi Fitur `extension-module`
- **Apa yang berubah**: Fitur Cargo `pyo3/extension-module` didepresiasi mulai `pyo3 0.26.0`.
- **Alasan teknis upstream**: Fitur Cargo bersifat aditif di seluruh *workspace*. Mengaktifkan `extension-module` di dalam `Cargo.toml` menyebabkan `cargo test` gagal pada tahap *linking* karena simbol `libpython` sengaja tidak ditautkan ketika fitur tersebut aktif.
- **Idiom baru**: Pembuatan modul ekstensi dikendalikan langsung oleh `maturin >= 1.8` (atau variabel lingkungan `PYO3_BUILD_EXTENSION_MODULE=1`), bukan melalui deklarasi statis di `Cargo.toml`.

### B. PyO3 0.26.0 - 0.28.0: Penghapusan `Python::allow_threads` $\rightarrow$ `Python::detach`
- **Apa yang berubah**: Metode `py.allow_threads(|| ...)` didepresiasi pada `pyo3 0.26.0` dan **dihapus sepenuhnya** pada `pyo3 0.28.0`.
- **Alasan teknis upstream**: Pada runtime Python *free-threaded* (PEP 703 / PEP 803 tanpa GIL), thread Python sudah berjalan paralel; operasi pelepasan konteks sebenarnya adalah melepaskan keterikatan (*detaching*) thread OS dari state runtime CPython.
- **Idiom baru**: Gunakan `py.detach(|| ...)`.

### C. PyO3 0.27.0 - 0.30.1: Penghapusan `Bound::downcast` $\rightarrow$ `Bound::cast::<T>()`
- **Apa yang berubah**: Metode `Bound<'py, PyAny>::downcast::<T>()` dan `downcast_into::<T>()` didepresiasi dan dihapus pada `pyo3 0.30.1`, digantikan oleh `Bound<'py, PyAny>::cast::<T>()` dan `cast_into::<T>()`.
- **Alasan teknis upstream**: Penyeragaman penamaan konversi tipe referensi cerdas `Bound<'py, T>` di seluruh API PyO3.

### D. PyO3 0.29.0 - 0.30.1 & Python 3.15: Stable ABI `abi3t` (PEP 803) dan `#[pyclass(frozen, subclass)]`
- **Apa yang berubah**:
  1. Penambahan fitur `abi3t` di samping `abi3-py310` untuk mendukung *Stable ABI* lintas versi Python 3.10–3.15 baik pada build GIL standar maupun build *free-threaded* (`python3.14t` / `python3.15t`).
  2. Kelas Rust yang diekspos ke Python melalui `#[pyclass]` tidak lagi mengizinkan pewarisan (*subclassing*) di sisi Python secara bawaan kecuali atribut `subclass` ditambahkan, serta dianjurkan menggunakan `frozen` untuk konkurensi tanpa kunci pinjam `PyCell` pada mode *free-threaded*.
  3. Di dalam modul deklaratif `#[pymodule] mod ...`, pendaftaran sub-komponen eksternal dilakukan menggunakan fungsi beratribut `#[pymodule_init]`.

---

## 2.3 Adaptasi pada Kode Sumber LuminaMoon

1. **`Cargo.toml` & `dataloader/Cargo.toml`**:
   - Mengonfigurasi `pyo3 = { version = "0.30.1", features = ["abi3-py310", "abi3t"] }` pada `[workspace.dependencies]` tanpa fitur `"extension-module"`, sehingga `cargo test -p luminamoon-dataloader` dapat berjalan secara mandiri.
2. **Pelepasan Runtime via `py.detach` (`dataloader/src/py_bridge.rs`)**:
   - Seluruh operasi I/O disk, pembacaan `mmap`, streaming jaringan HTTP 206, inferensi skema Parquet, dan unggahan NDJSON HuggingFace Hub dijalankan di dalam `py.detach(|| ...)`:
     ```rust
     let batch_opt = py
         .detach(|| self.inner.next_batch())
         .map_err(to_py_err)?;
     ```
3. **Konversi Tipe `Bound::cast::<T>()` dan Hierarki `PyBool` vs `PyInt` (`dataloader/src/py_bridge.rs`)**:
   - Menggunakan `columns.cast::<PyDict>()`, `v.cast::<PySequence>()`, dan `pair.cast::<PyTuple>()` pada `extract_py_columns_table`.
   - Karena di CPython `bool` merupakan subkelas dari `int`, fungsi `py_any_to_multimodal_value` selalu memeriksa `item.is_instance_of::<PyBool>()` sebelum `item.is_instance_of::<PyInt>()` agar kolom boolean tidak terkonversi menjadi integer `1`/`0`.
4. **Deklarasi `#[pyclass(frozen, subclass)]` dan Registrasi `#[pymodule_init]` (`dataloader/src/py_bridge.rs` & `optimizer/src/lib.rs`)**:
   - Kelas `PyDataStreamLoader`, `PyMultimodalParquetLoader`, dan `PyHfHubUploader` dideklarasikan dengan `#[pyclass(frozen, subclass, ...)]` dan disinkronkan di sisi Rust menggunakan `parking_lot::Mutex`.
   - Fungsi `register_dataloader_classes(m)` dipanggil dari fungsi `#[pymodule_init] fn init(m: &Bound<'_, PyModule>)` di dalam `optimizer/src/lib.rs`, mendaftarkan kelas serta fungsi tingkat modul `serialize_table_to_bytes` dan `postprocess_and_dedup_records`.
5. **Transposisi Baris-ke-Kolom Otomatis & Fungsi Tabel/Postprocess Bebas-GIL (`dataloader/src/py_bridge.rs`)**:
   - `extract_py_columns_table` mendukung tiga bentuk input Python sekaligus: `dict[str, Sequence[Any]]`, `Sequence[tuple[str, Sequence[Any]]]`, dan `Sequence[dict[str, Any]]` (mentransposisi daftar *row dictionary* menjadi kolom terurut berdasarkan kemunculan kunci pertama dan mengisi kunci yang absen dengan `MultimodalValue::Null`).
   - `serialize_table_to_bytes` dan `postprocess_and_dedup_records` (dengan flag `deduplicate: bool = true`) mengeksekusi serialisasi Parquet/JSONL/JSON serta normalisasi Rayon + SIMD + deduplikasi `xxh3_64` sepenuhnya di luar GIL melalui `py.detach(|| ...)`.

