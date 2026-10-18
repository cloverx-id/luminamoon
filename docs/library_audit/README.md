# Dokumentasi Audit Versi & Perubahan Pustaka (Oktober 2026)

Folder ini mendokumentasikan hasil penelusuran riwayat perubahan (*changelog*), depresiasi API/ABI pada rentang versi antara, hingga adaptasi kode pada rilis stabil terbaru untuk seluruh dependensi proyek **LuminaMoon (`LuminaV v1.0.0`)**.

---

## Matriks Ringkasan Ekosistem

| Komponen / Pustaka | Rentang Versi Diaudit | Versi Stabil Terbaru | Status Adaptasi | Dokumen Rincian |
| :--- | :--- | :--- | :--- | :--- |
| **Rust Toolchain & SIMD** | `1.85.0` - `1.99.1` | `1.99.1` (`edition = "2024"`) | Terapkan `std::cfg_select!`, isolasi jalur `Scalar` vs `Avx2` | [01_rust_toolchain_and_simd.md](01_rust_toolchain_and_simd.md) |
| **Crate Pendukung Rust** | `2024` - `2026` | `rayon 1.12.0`, `thiserror 2.0.18`, `libloading 0.9.0`, `serde 1.0.228`, `serde_json 1.0.149` | Bebas crate usang (`cfg-if` dihapus) | [01_rust_toolchain_and_simd.md](01_rust_toolchain_and_simd.md) |
| **Python, PyO3 & Maturin** | `Py 3.10` - `3.15.0`, `pyo3 0.24` - `0.30.1` | `Python 3.15.0`, `pyo3 0.30.1`, `maturin 1.15.0` | Hapus `extension-module`, aktifkan `abi3t`, migrasi `py.detach` & `#[pymodule] mod` | [02_python_pyo3_and_maturin.md](02_python_pyo3_and_maturin.md) |
| **PyTorch & Triton JIT** | `torch 2.4` - `2.14.1`, `triton 3.0` - `3.8.0` | `torch 2.14.1`, `triton 3.8.0` | Integrasi `torch.accelerator`, koreksi `tl.sqrt` di AST JIT, dukungan 1-2 scratch pointers | [03_pytorch_and_triton_jit.md](03_pytorch_and_triton_jit.md) |
| **CUDA & AMD ROCm / HIP** | `CUDA 12.0` - `13.x`, `ROCm 5.7` - `7.x / TheRock 10.1.0` | `ROCm 7.x / TheRock 10.1.0` | Probing `amdhip64_7.dll`, ABI `R0600`, resolusi Wave32/Wave64 CDNA4 & RDNA4, SM Rubin/Blackwell | [04_cuda_and_rocm_hip_drivers.md](04_cuda_and_rocm_hip_drivers.md) |
| **Vulkan Compute & `ash`** | `Vulkan 1.2` - `1.4`, `ash 0.37` - `0.38.0` | `Vulkan 1.4`, `ash 0.38.0` | Negosiasi instance `1.4`/`1.3`, validasi batas subgroup `VkPhysicalDeviceVulkan13Properties` | [05_vulkan_and_ash.md](05_vulkan_and_ash.md) |

---

## Peta Navigasi Berkas

| Jalur Berkas | Peran Singkat | Status |
| :--- | :--- | :--- |
| `docs/library_audit/README.md` | Indeks utama dan matriks ringkasan versi seluruh pustaka. | Aktif |
| `docs/library_audit/01_rust_toolchain_and_simd.md` | Riwayat perubahan Rust 1.85-1.99.1, intrinsik SIMD x86_64, dan crate inti. | Aktif |
| `docs/library_audit/02_python_pyo3_and_maturin.md` | Riwayat perubahan Python 3.10-3.15, PyO3 0.24-0.30.1, ABI3t, dan Maturin. | Aktif |
| `docs/library_audit/03_pytorch_and_triton_jit.md` | Riwayat perubahan PyTorch 2.4-2.14.1 dan kompiler Triton 3.0-3.8.0. | Aktif |
| `docs/library_audit/04_cuda_and_rocm_hip_drivers.md` | Riwayat perubahan driver CUDA serta ROCm 5-7 / TheRock 10.1 pada Windows dan Linux. | Aktif |
| `docs/library_audit/05_vulkan_and_ash.md` | Riwayat promosi fitur inti Vulkan 1.3/1.4 dan binding `ash 0.38.0`. | Aktif |
