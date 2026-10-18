# LuminaMoon Documentation

LuminaMoon adalah engine optimasi matematika berbasis **LuminaV v1.0.0** dengan akselerasi multi-tier perangkat keras: SIMD AVX-512/AVX2 pada CPU host, compute shader Vulkan 1.3/1.4 UMA Wave32, kernel terfusi Triton JIT untuk GPU, serta fallback universal zero-copy untuk NumPy dan PyTorch.

Dokumentasi ini disusun secara terstruktur, to-the-point, dan komprehensif untuk memudahkan integrasi, konfigurasi, dan pemahaman arsitektur internal.

## Peta Dokumentasi

| Dokumen | Deskripsi |
| :--- | :--- |
| [01. Panduan Memulai (Getting Started)](01_getting_started.md) | Kebutuhan sistem, instalasi via pip/wheel, kompilasi dari sumber (Maturin & Rust), serta contoh kilat NumPy dan PyTorch. |
| [02. Panduan Penggunaan (User Guide)](02_user_guide.md) | Panduan praktis buffer modes (CLM vs TD-CLM), mixed-precision & stochastic rounding, boundary dynamics, checkpointing, dan distributed training (FSDP). |
| [03. Referensi Konfigurasi & API (Configuration & API)](03_configuration_api.md) | Referensi parameter lengkap `LuminaV`, tabel tipe/default/rentang nilai, metode publik, validasi, dan auto-koreksi typo. |
| [04. Arsitektur & Akselerasi Hardware (Architecture & Hardware)](04_architecture_and_hardware.md) | Bedah arsitektur crate Rust (`luminav-core`, `luminav-simd`, `luminav-vulkan`, `luminav-dispatch`, `luminav-triton-aot`), zero-copy buffer protocol, dan C4 container model. |
| [Library Audit Suite](../library_audit/README.md) | Audit mendalam rantai dependensi, ABI compatibility, SIMD intrinsics, driver GPU (CUDA/ROCm/Vulkan), dan PyO3/Maturin. |

## Ringkasan Fitur Inti

- **Zero Heavy Dependency**: Pustaka inti memiliki jejak `dependencies = []` tanpa memaksa instalasi framework berat.
- **Dual-Buffer CLM & Single-Buffer TD-CLM**: Optimasi berbasis Cautious Learning Momentum dengan dynamic defect compensation.
- **Universal Zero-Copy Memory Access**: Kompatibel langsung dengan buffer NumPy 2.x via `__array_interface__` dan PyTorch Tensor.
- **Cross-Platform Free-Threaded Ready**: Mendukung Python 3.10 hingga 3.15+ melalui CPython Stable ABI (`abi3` & `abi3t` / PEP 703).
