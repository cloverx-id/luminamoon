# 3. PyTorch dan Kompiler Triton (JIT & AOT)

## 3.1 Versi Stabil Terbaru
- **PyTorch (`torch`)**: `2.14.1` (`2.14.0` rilis mayor)
- **Triton (`triton-lang`)**: `3.8.0`

---

## 3.2 Riwayat Perubahan PyTorch (`2.4.0` - `2.14.1`)

### A. Stabilisasi API `torch.accelerator` (`2.6.0` - `2.14.1`)
- **Apa yang berubah**: PyTorch memperkenalkan dan menstabilkan modul `torch.accelerator` sebagai lapisan abstraksi perangkat keras terpadu untuk CUDA, Intel XPU, Apple MPS, dan akselerator kustom.
- **API Utama**:
  - `torch.accelerator.is_available()`
  - `torch.accelerator.device(dev)` (konteks perangkat agnostik)
  - `torch.accelerator.current_device_index()` (resolusi indeks perangkat aktif)
- **Adaptasi di `optimizer/reference/luminav_v153.py`**:
  Fungsi `_get_device_context` dan `_get_generator` diperbarui agar memeriksa `torch.cuda`, `torch.xpu`, serta `torch.accelerator.current_device_index()` secara berurutan sehingga instansiasi `torch.Generator` untuk *Stochastic Rounding* tidak jatuh ke indeks `0` yang salah saat dijalankan pada multi-GPU XPU atau akselerator non-CUDA.

---

## 3.3 Riwayat Perubahan Triton (`3.0.0` - `3.8.0`)

### A. Pengetatan AST Parser `@triton.jit` (`3.2.0` - `3.8.0`)
- **Apa yang berubah**: Kompiler AST Triton menolak resolusi simbol fungsi `torch.*` di dalam fungsi yang didekorasi dengan `@triton.jit`. Seluruh operasi matematika di dalam kernel wajib menggunakan primitif `triton.language` (`tl.*` atau `tl.math.*`).
- **Temuan Bug & Perbaikan di `optimizer/reference/luminav_v153.py`**:
  Pada `_lumina_tdclm_pass2_kernel` (baris 454), ditemukan pemanggilan `torch.sqrt(p_init * p_init + 1e-4)` pada cabang `bound_mode == 2`. Baris tersebut diperbaiki menjadi `tl.sqrt(p_init * p_init + 1e-4)` agar kompilasi JIT pada Triton `3.2`–`3.8` berjalan tanpa galat `CompilationError`.

### B. Namespacing `tl.math` vs `tl` (`3.0.0` - `3.8.0`)
- **Apa yang berubah**: Fungsi transendental seperti `tanh` berada di bawah `tl.math.tanh` (menggantikan `tl.libdevice.tanh` yang sudah dihapus dan `tl.tanh` pada versi lama).
- **Adaptasi**: Fungsi `_triton_tanh_fast` pada `luminav_v153.py` memprioritaskan `tl.math.tanh(x)` dengan fallback aman jika dijalankan pada instalasi minimal.

### C. Konvensi Trailing Scratch Pointers pada Triton AOT (`3.0.0` - `3.8.0`)
- **Apa yang berubah**:
  - **Triton 3.0 - 3.2**: Kernel hasil kompilasi AOT (`cubin` / `hsaco`) menambahkan 1 pointer implisit di akhir daftar parameter runtime: `CUdeviceptr global_scratch`.
  - **Triton 3.3 - 3.8**: ABI kernel AOT mendukung hingga 2 pointer scratch implisit di akhir daftar parameter: `global_scratch` dan `profile_scratch` (diisi pointer ke `0u64` apabila kernel tidak mengalokasikan *global scratch* atau *proton profiling*).
- **Adaptasi di `optimizer/crates/luminav-triton-aot/`**:
  - Struktur `KernelMetadata` dan `AotManifest` menyimpan atribut `num_trailing_scratch_ptrs`.
  - Fungsi `pack_triton_abi_args(user_arg_ptrs, num_trailing_scratch_ptrs)` pada `cuda_driver.rs` menyisipkan pointer `&TRITON_NULL_SCRATCH` sebanyak `num_trailing_scratch_ptrs` secara dinamis (mendukung konvensi 1 pointer maupun 2 pointer pada Triton `3.8.0`).
  - Pengujian unit `test_triton3_abi_trailing_scratch_pointer` memverifikasi pengemasan ABI untuk 1 dan 2 pointer scratch.
