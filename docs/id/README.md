# LuminaMoon Documentation (Bahasa Indonesia)

LuminaMoon (**PROJECT LUMI**) adalah suite optimasi deep learning dan streaming dataset berkinerja tinggi yang dibangun di atas Rust (Edition 2024) dengan jembatan ekstensi C Python bebas-GIL (`PyO3 0.30.1` `abi3-py310` & `abi3t`). Ekosistem ini menyatukan dua subsistem inti:

1. **LuminaV Optimizer v1.0.0**: Mesin optimasi matematika berpresisi rendah dengan akselerasi SIMD CPU host (`AVX-512`, `AVX2`), compute shader Vulkan 1.3/1.4 UMA Wave32, kernel GPU Triton JIT/AOT, serta eksekusi zero-copy universal untuk NumPy dan PyTorch.
2. **LuminaMoon DataLoader v1.0.0**: Mesin streaming dataset `mmap` lokal zero-copy dan HTTP 206 konkuren dengan ring buffer watermark Pre-RAM 100-batch, pemulihan crash Ping-Pong 128-byte (`Slot A` / `Slot B`) berproteksi CRC32, deduplikasi dua tahap `xxh3_64`, pemindaian byte SIMD CPU mandiri (`AVX-512BW` / `AVX2` / `NEON`), serialisasi multimodal Parquet/JSONL/JSON, serta integrasi resmi HuggingFace Hub.

## Peta Dokumentasi

| Dokumen | Deskripsi |
| :--- | :--- |
| [01. Panduan Memulai (Getting Started)](01_getting_started.md) | Kebutuhan sistem, instalasi via pip/wheel, kompilasi dari sumber (Maturin & Rust), audit hardware & SIMD ISA (`LuminaV.detect_hardware()`, `lm.dataloader_cpu_isa()`), serta contoh kilat NumPy, PyTorch, dan DataLoader (`lm.stream`, `lm.from_hf_dataset`, `lm.push_to_hub`). |
| [02. Panduan Penggunaan (User Guide)](02_user_guide.md) | Panduan praktis mode buffer Optimizer (CLM vs TD-CLM), mixed-precision & stochastic rounding, boundary dynamics, distributed training (FSDP), serta panduan DataLoader streaming $O(1)$ lokal/HTTP-206, ring buffer Pre-RAM 100-batch, pemulihan crash Ping-Pong, deduplikasi dua tahap XXH3-64, dan pipeline multimodal HuggingFace. |
| [03. Referensi Konfigurasi & API (Configuration & API)](03_configuration_api.md) | Referensi parameter lengkap, tanda tangan fungsi/kelas, metode publik, dan validasi untuk `LuminaV`, `lm.stream`, `DataStreamLoader`, `MultimodalParquetLoader`, `lm.from_hf_dataset`, `lm.push_to_hub`, `DataBatch`, `MultimodalBatch`, dan `StreamCoordinate`. |
| [04. Arsitektur & Akselerasi Hardware (Architecture & Hardware)](04_architecture_and_hardware.md) | Bedah arsitektur crate Rust Optimizer (`luminav-core`, `luminav-simd`, `luminav-vulkan`, `luminav-dispatch`, `luminav-triton-aot`), modul Rust DataLoader (`coord`, `dedup`, `metadata`, `simd_cpu`, `postprocess`, `source_local`, `source_hf`, `preram`, `multimodal`, `hf_hub`, `py_bridge`), protokol memori zero-copy, dan dukungan Python *free-threaded* (PEP 703). |
| [Dokumentasi Modular Developer Optimizer](../optimize/README.id.md) | Spesifikasi teknis internal untuk 7 modul Optimizer (`core`, `simd`, `vulkan`, `triton_aot`, `dispatch`, `python_ffi`, `python_runtime`). |
| [Dokumentasi Modular Developer DataLoader](../dataloader/README.id.md) | Spesifikasi teknis internal untuk 7 modul DataLoader (`coord_and_metadata`, `dedup`, `simd_and_postprocess`, `sources_and_streaming`, `preram`, `multimodal_and_hf_hub`, `python_ffi_and_runtime`). |
| [Suite Audit Pustaka Optimizer](../library_audit/README.md) | Laporan audit mendalam rantai dependensi, kompatibilitas ABI, intrinsik SIMD, driver GPU (CUDA/ROCm/Vulkan), dan PyO3/Maturin. |
| [Suite Audit Pustaka DataLoader](../../dataloader/docs/library_audit/README.md) | Laporan audit mendalam Arrow/Parquet 60, `reqwest 0.13.5`, `tokio 1.53.3`, `bytes 1.12.1` (`CVE-2026-25541`), `memmap2`, `xxhash-rust`, `hashbrown`, `crc32fast`, dan protokol API HuggingFace Hub. |

## Ringkasan Fitur Inti

- **Zero Heavy Dependency**: Pustaka inti memiliki jejak `dependencies = []` tanpa memaksa instalasi framework berat.
- **Dual-Buffer CLM & Single-Buffer TD-CLM**: Optimasi berbasis Cautious Learning Momentum dengan kompensasi defek dinamis dan penghematan memori status 50%.
- **Streaming Zero-Copy & Pemulihan Crash $O(1)$**: `mmap` lokal, lompatan halaman `RowSelection` Arrow, worker multi-range HTTP 206 konkuren, prefetching Pre-RAM 100-batch, dan checkpoint CRC32 Ping-Pong 128-byte.
- **Cross-Platform Free-Threaded Ready**: Mendukung Python 3.10 hingga 3.15+ melalui CPython Stable ABI (`abi3` & `abi3t` / PEP 703) dengan pembebasan GIL penuh via `py.detach()`.
