# 2. Python, PyO3, dan Maturin

## 2.1 Versi Stabil Terbaru
- **Python**: `3.15.0` (dirilis 9 Oktober 2026)
- **`pyo3`**: `0.30.1`
- **`maturin`**: `1.15.0`

---

## 2.2 Riwayat Perubahan pada Rentang Versi (`pyo3 0.24.0` - `0.30.1`)

### A. PyO3 0.25.0 - 0.26.0: Depresiasi Fitur `extension-module`
- **Apa yang berubah**: Fitur Cargo `pyo3/extension-module` didepresiasi pada `pyo3 0.26.0`.
- **Alasan teknis upstream**: Fitur Cargo bersifat aditif (*additive*) di seluruh *workspace*. Ketika `extension-module` diaktifkan di `Cargo.toml` untuk membangun modul `.pyd`/`.so`, menjalankan `cargo test` pada *workspace* yang sama sering gagal pada tahap *linking* karena simbol `libpython` tidak ditautkan saat fitur `extension-module` aktif.
- **Idiom baru**: Pengaturan pembuatan modul ekstensi kini dikendalikan oleh sistem *build* (`maturin >= 1.8`) atau melalui variabel lingkungan `PYO3_BUILD_EXTENSION_MODULE=1` saat kompilasi manual, bukan melalui `features = ["extension-module"]` di `Cargo.toml`.

### B. PyO3 0.26.0 - 0.28.0: Penghapusan `Python::allow_threads` $\rightarrow$ `Python::detach`
- **Apa yang berubah**: Metode `py.allow_threads(|| ...)` didepresiasi pada `pyo3 0.26.0` dan **dihapus sepenuhnya** pada `pyo3 0.28.0`.
- **Alasan teknis upstream**: Dengan hadirnya Python *free-threaded* (PEP 703 / PEP 803 tanpa GIL pada Python 3.13–3.15), istilah `allow_threads` tidak lagi akurat karena thread Python sudah berjalan paralel tanpa GIL; operasi tersebut sebenarnya adalah melepaskan keterikatan (*detaching*) thread Rust dari *runtime* Python.
- **Idiom baru**: Gunakan `py.detach(|| ...)`.

### C. PyO3 0.24.0 - 0.30.1: Deklarasi Modul `#[pymodule] mod`
- **Apa yang berubah**: Pola prosedural lama `fn m(m: &Bound<'_, PyModule>) -> PyResult<()>` dengan `m.add_function(wrap_pyfunction!(..., m)?)?` digantikan oleh deklarasi modul berbasis `mod`.
- **Idiom baru**:
  ```rust
  #[pymodule]
  mod _luminamoon_optimizer_rs {
      #[pymodule_export]
      use super::step_pyo3;

      #[pymodule_export]
      use super::detect_hardware_pyo3;
  }
  ```

### D. PyO3 0.29.0 - 0.30.1 & Python 3.15: Stable ABI `abi3t` (PEP 803)
- **Apa yang berubah**: Penambahan fitur `abi3t` di samping `abi3-py310`.
- **Alasan teknis upstream**: PEP 803 memperluas *Stable ABI* (`abi3`) agar mencakup *build* Python *free-threaded* (`python3.14t` / `python3.15t`) selain *build* GIL klasik.

---

## 2.3 Adaptasi pada Kode Sumber LuminaMoon

1. **`Cargo.toml` & `pyproject.toml`**:
   - Menghapus `"extension-module"` dari daftar fitur `pyo3`.
   - Mengonfigurasi `pyo3 = { version = "0.30.1", features = ["abi3-py310", "abi3t"] }`.
   - Memperbarui `requires = ["maturin>=1.15,<2.0"]` pada `pyproject.toml`.
2. **`optimizer/src/python_ffi.rs`**:
   - Mengganti `py.allow_threads(...)` pada fungsi `step_pyo3` menjadi `py.detach(...)`:
     ```rust
     let result = py.detach(|| {
         engine.step_batch(&mut views, &config, &steps, &param_indices)
     });
     ```
3. **`optimizer/src/lib.rs`**:
   - Mengganti fungsi inisialisasi modul imperatif menjadi modul deklaratif `#[pymodule] mod _luminamoon_optimizer_rs` dengan atribut `#[pymodule_export]`.

---

## 2.4 Ketahanan Maju (Forward-Compatibility Architecture: Python 3.10 - 3.15+)

Untuk menjamin repositori dapat berjalan tanpa kegagalan pada versi Python masa depan atau *environment* tanpa *toolchain* Rust/C++:

1. **Dual-Tier Forward Compatibility**:
   - **Tier 1 (Stable ABI `abi3` & `abi3t`)**: Ekstensi CPython dikompilasi dengan PyO3 `features = ["abi3-py310", "abi3t"]`. Satu *wheel* binary yang dibangun untuk CPython 3.10 Stable ABI secara otomatis kompatibel maju (*forward-compatible*) ke Python 3.11, 3.12, 3.13, 3.14, hingga 3.15 baik mode standar maupun *free-threaded* (`Py_GIL_DISABLED`).
   - **Tier 2 (Universal Pure-PyTorch Fallback Engine)**: Apabila *binary module* `_luminamoon_rs` tidak tersedia, korup, atau dijalankan pada interpreter alternatif, `python/luminamoon/optimizer/luminav.py` tidak melempar `RuntimeError`. Sebaliknya, sistem secara transparan mengalihkan eksekusi ke mesin PyTorch murni (`_step_gpu_group`) yang mendukung operasi tensor CPU/GPU penuh dengan konvergensi matematis identik dan *stochastic rounding*.
