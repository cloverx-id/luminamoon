# 1. Rust Toolchain, SIMD Intrinsics, dan Crate Ekosistem

## 1.1 Versi Stabil Terbaru
- **Rust Compiler (`rustc` & `cargo`)**: `1.99.1` (`edition = "2024"`, rilis beta berikutnya `1.100.0`)
- **`rayon`**: `1.12.0`
- **`thiserror`**: `2.0.21`
- **`libloading`**: `0.9.0`
- **`serde`**: `1.0.229`
- **`serde_json`**: `1.0.154`
- **`cfg-if`**: Dihapus dari dependensi (digantikan oleh makro standar Rust `std::cfg_select!`)

---

## 1.2 Riwayat Perubahan pada Rentang Versi (`Rust 1.85.0` - `1.99.1`)

### A. Rust 1.85.0 (`edition = "2024"`)
- **Perubahan**: Lint `unsafe_op_in_unsafe_fn` diaktifkan secara bawaan pada edisi 2024. Fungsi `unsafe fn` tidak lagi secara implisit mengizinkan pemanggilan operasi `unsafe` di dalamnya tanpa blok `unsafe {}` eksplisit.
- **Perubahan Syntax**: Stabilisasi rantai `let` (`if let Some(x) = a && x > 0`) di dalam kondisi `if` dan `while`.

### B. Rust 1.86.0 - 1.87.0 (Target Feature & SIMD Intrinsics)
- **Rust 1.86.0**: Fungsi dengan atribut `#[target_feature(enable = "...")]` dapat dipanggil secara aman dari fungsi lain yang mengaktifkan himpunan fitur target yang sama.
- **Rust 1.87.0**: Intrinsik SIMD berbasis nilai (*value-based intrinsics* seperti `_mm256_add_ps`, `_mm512_fmadd_ps`, `_mm512_cmp_ps_mask`) tidak lagi memerlukan blok `unsafe` apabila dipanggil di dalam fungsi yang memiliki atribut `#[target_feature]` yang sesuai. Hanya intrinsik yang melakukan dereferensi pointer mentah (seperti `_mm256_loadu_ps`, `_mm256_storeu_ps`, `_mm512_loadu_ps`, `_mm512_storeu_ps`) yang tetap berstatus `unsafe`.

### C. Rust 1.89.0 (Stabilisasi Target Feature AVX-512)
- **Perubahan**: Seluruh fitur target `avx512f`, `avx512dq`, `avx512bw`, `avx512vl` beserta intrinsik `std::arch::x86_64::_mm512_*` distabilkan sepenuhnya pada kanal *stable* tanpa memerlukan `#![feature(stdarch_x86_avx512)]`.

### D. Rust 1.95.0 (`std::cfg_select!`)
- **Perubahan**: Penambahan makro pustaka standar `std::cfg_select!` untuk percabangan kompilasi bersyarat (*compile-time configuration branching*).
- **Dampak**: Menggantikan kebutuhan akan crate pihak ketiga `cfg-if`.

---

## 1.3 Adaptasi pada Kode Sumber LuminaMoon

### 1. Penggantian `cfg-if` dengan `std::cfg_select!`
- **Berkas**: `Cargo.toml`, `optimizer/crates/luminav-simd/Cargo.toml`, `optimizer/crates/luminav-simd/src/lib.rs`
- **Perubahan**:
  Dependensi eksternal `cfg-if` dihapus dari `Cargo.toml`. Fungsi `detect_simd_backend()` pada `optimizer/crates/luminav-simd/src/lib.rs` diperbarui menggunakan makro standar:
  ```rust
  pub fn detect_simd_backend() -> SimdBackend {
      std::cfg_select! {
          target_arch = "x86_64" => {
              if std::arch::is_x86_feature_detected!("avx512f")
                  && std::arch::is_x86_feature_detected!("avx512dq")
                  && std::arch::is_x86_feature_detected!("avx512bw")
                  && std::arch::is_x86_feature_detected!("avx512vl")
                  && std::arch::is_x86_feature_detected!("fma")
              {
                  SimdBackend::Avx512
              } else if std::arch::is_x86_feature_detected!("avx2")
                  && std::arch::is_x86_feature_detected!("fma")
              {
                  SimdBackend::Avx2
              } else {
                  SimdBackend::Scalar
              }
          }
          _ => {
              SimdBackend::Scalar
          }
      }
  }
  ```

### 2. Pemisahan Jalur `SimdBackend::Scalar` dan `SimdBackend::Avx2`
- **Berkas**: `optimizer/crates/luminav-simd/src/parallel.rs`
- **Masalah Sebelumnya**: Cabang `match backend` pada fungsi pembantu *chunk dispatch* menggabungkan `SimdBackend::Avx2 | SimdBackend::Scalar` ke pemanggilan fungsi `avx2::*_avx2`. Meskipun pemanggil tingkat atas mencegat `SimdBackend::Scalar` lebih awal, penggabungan tersebut berisiko memicu instruksi ilegal (`SIGILL`) jika fungsi pembantu dipanggil langsung pada CPU tanpa AVX2.
- **Perbaikan**: Cabang `SimdBackend::Scalar` dipisahkan secara eksplisit dari `SimdBackend::Avx2` pada seluruh 6 fungsi *chunk dispatch* di `parallel.rs`.
