# Portal Dokumentasi LuminaMoon

Selamat datang di portal dokumentasi **LuminaMoon** (**PROJECT LUMI**), monorepo untuk optimasi numerik, streaming data multimodal, dan pengujian formal yang dikembangkan oleh **Lumina Moon ([@cloverx-id](https://github.com/cloverx-id))** dan **Silver Moon ([@cloverxion](https://github.com/cloverxion))**, menaungi **LuminaV Optimizer**, **LuminaMoon DataLoader**, dan rangkaian pengujian verifikasi formal.

[English Version (README.md)](README.md)

---

## 1. Panduan Pengguna (User Documentation)

### [Dokumentasi Bahasa Indonesia (`docs/id/`)](id/README.md)
1. **[01. Panduan Memulai](id/01_getting_started.md)**: Kebutuhan sistem, instalasi via pip, kompilasi dari sumber via Maturin/Rust, audit kapabilitas hardware & SIMD ISA (`LuminaV.detect_hardware()`, `lm.dataloader_cpu_isa()`), serta contoh kilat NumPy, PyTorch, dan DataLoader (`lm.stream`, `lm.from_hf_dataset`, `lm.push_to_hub`).
2. **[02. Panduan Penggunaan](id/02_user_guide.md)**: Pemilihan mode buffer Optimizer (Dual-Buffer CLM vs Single-Buffer TD-CLM), mixed-precision & stochastic rounding, boundary dynamics, distributed training (FSDP), serta panduan DataLoader streaming $O(1)$ lokal/HTTP-206, ring buffer Pre-RAM 100-batch, pemulihan crash Ping-Pong, deduplikasi dua tahap XXH3-64, dan pipeline multimodal HuggingFace Hub.
3. **[03. Referensi Konfigurasi & API](id/03_configuration_api.md)**: Tabel referensi parameter lengkap, metode publik, dan aturan validasi untuk `LuminaV`, `lm.stream`, `DataStreamLoader`, `MultimodalParquetLoader`, `lm.from_hf_dataset`, `lm.push_to_hub`, `DataBatch`, `MultimodalBatch`, dan `StreamCoordinate`.
4. **[04. Arsitektur & Akselerasi Hardware](id/04_architecture_and_hardware.md)**: Alur dispatch C4, bedah crate Rust Optimizer (`core`, `simd`, `vulkan`, `dispatch`, `triton-aot`), bedah modul Rust DataLoader (`coord`, `dedup`, `metadata`, `simd_cpu`, `postprocess`, `source_local`, `source_hf`, `preram`, `multimodal`, `hf_hub`, `py_bridge`), protokol memori zero-copy, dan kompatibilitas Python *free-threaded* (`abi3t` / PEP 703).

### [English Documentation (`docs/en/`)](en/README.md)
1. **[01. Getting Started](en/01_getting_started.md)**
2. **[02. User Guide](en/02_user_guide.md)**
3. **[03. Configuration & API Reference](en/03_configuration_api.md)**
4. **[04. Architecture & Hardware Acceleration](en/04_architecture_and_hardware.md)**

---

## 2. Dokumentasi Modular Pengembang (`docs/optimize/` & `docs/dataloader/`)

Spesifikasi teknis mendalam untuk perekayasa sistem, pengembang kernel, dan pengembang pustaka:

### A. LuminaV Optimizer (`docs/optimize/` & `optimizer/`)
- **[Ikhtisar & Panduan Pengguna LuminaV Optimizer (`optimizer/README.id.md`)](../optimizer/README.id.md)**
- **[Indeks Dokumentasi Modular Optimizer (`docs/optimize/README.id.md`)](optimize/README.id.md)** / **[Versi Inggris (`docs/optimize/README.md`)](optimize/README.md)**
- **[01. Core & Math](optimize/core/README.id.md)** / **[EN](optimize/core/README.md)**: CLM, selisih temporal TD-CLM, pembulatan stokastik PRNG, dan koreksi bias.
- **[02. Vektorisasi SIMD](optimize/simd/README.id.md)** / **[EN](optimize/simd/README.md)**: Intrinsik AVX-512 dan AVX2/FMA x86_64 serta pool thread paralel Rayon.
- **[03. Komputasi Vulkan](optimize/vulkan/README.id.md)** / **[EN](optimize/vulkan/README.md)**: Compute shader SPIR-V Ash Vulkan 1.3/1.4 dan buffer host zero-copy UMA.
- **[04. Triton AOT & Driver GPU](optimize/triton_aot/README.id.md)** / **[EN](optimize/triton_aot/README.md)**: Kompilasi AOT, pemuatan driver dinamis CUDA/HIP, dan C-API mandiri.
- **[05. Perutean Dispatch](optimize/dispatch/README.id.md)** / **[EN](optimize/dispatch/README.md)**: Audit kapabilitas hardware dan ambang batas perutean 16.384 elemen.
- **[06. Jembatan FFI Python & PyO3](optimize/python_ffi/README.id.md)** / **[EN](optimize/python_ffi/README.md)**: Pembebasan GIL PyO3 (`py.detach()`), validasi anti-aliasing pointer, dan protokol buffer NumPy.
- **[07. Runtime Python & GPU Mixin](optimize/python_runtime/README.id.md)** / **[EN](optimize/python_runtime/README.md)**: Class `LuminaV`, kernel Triton JIT, sinkronisasi FSDP, dan checkpoint status.

### B. LuminaMoon DataLoader (`docs/dataloader/` & `dataloader/`)
- **[Ikhtisar & Panduan Pengguna LuminaMoon DataLoader (`dataloader/README.id.md`)](../dataloader/README.id.md)**
- **[Portal Mandiri DataLoader (`dataloader/docs/README.id.md`)](../dataloader/docs/README.id.md)** / **[Versi Inggris (`dataloader/docs/README.md`)](../dataloader/docs/README.md)**
- **[Indeks Dokumentasi Modular DataLoader (`docs/dataloader/README.id.md`)](dataloader/README.id.md)** / **[Versi Inggris (`docs/dataloader/README.md`)](dataloader/README.md)**
- **[01. Koordinat & Metadata Ganda](dataloader/coord_and_metadata/README.id.md)** / **[EN](dataloader/coord_and_metadata/README.md)**: `StreamCoordinate` 32-byte, checkpoint CRC32 Ping-Pong 128-byte (`Slot A` / `Slot B`), dan WAL `.dedup`.
- **[02. Deduplikasi Dua Tahap](dataloader/dedup/README.id.md)** / **[EN](dataloader/dedup/README.md)**: `TwoStageDeduplicator`, `xxh3_64`, bypass SwissTable `IdentityBuildHasher`, dan himpunan staged vs GPU-committed.
- **[03. SIMD CPU & Pasca-Pemrosesan Paralel](dataloader/simd_and_postprocess/README.id.md)** / **[EN](dataloader/simd_and_postprocess/README.md)**: Kernel byte `AVX-512BW` / `AVX2` / `NEON`, `CacheAlignedBuffer`, dan pemfilteran UTF-8/CRLF/JSONL Rayon.
- **[04. Sumber Shard Lokal & HTTP 206](dataloader/sources_and_streaming/README.id.md)** / **[EN](dataloader/sources_and_streaming/README.md)**: Zero-copy `memmap2`, lompatan halaman Arrow `RowSelection`, dan worker multi-range HTTP 206 konkuren (`download_workers`).
- **[05. Mesin Watermark Pre-RAM](dataloader/preram/README.id.md)** / **[EN](dataloader/preram/README.md)**: Ring buffer 100-batch, pemicu isi ulang batas bawah, pool produsen Tokio multi-thread, dan penggabungan batch lintas-shard.
- **[06. Klien Multimodal & HF Hub](dataloader/multimodal_and_hf_hub/README.id.md)** / **[EN](dataloader/multimodal_and_hf_hub/README.md)**: `MultimodalValue`, Parquet `struct<bytes, path>`, serializer JSONL/JSON, pengaman token HTTPS, dan pengunggahan komit atomik NDJSON.
- **[07. Jembatan FFI Python & Runtime HF](dataloader/python_ffi_and_runtime/README.id.md)** / **[EN](dataloader/python_ffi_and_runtime/README.md)**: Binding PyO3 0.30.1 `abi3t`, `py.detach()`, `lm.stream`, `lm.from_hf_dataset`, `lm.push_to_hub`, dan `auto_commit` tertunda.

---

## 3. Audit Pustaka & Rekayasa Eksternal (`docs/library_audit/` & `dataloader/docs/library_audit/`)

- **[Rangkaian Audit Pustaka Optimizer (`docs/library_audit/README.md`)](library_audit/README.md)**: Verifikasi teknis dan kepatuhan ABI untuk toolchain Rust 1.99.1, PyO3/Maturin, PyTorch 2.14.1, Triton JIT 3.8.0, CUDA/ROCm, dan pipeline komputasi Ash Vulkan.
- **[Rangkaian Audit Pustaka DataLoader Mandiri (`dataloader/docs/library_audit/README.md`)](../dataloader/docs/library_audit/README.md)**: Verifikasi teknis dan adaptasi basis kode untuk Rust 1.99.1 / Edition 2024, `pyo3 0.30.1` (`abi3t`), Apache Arrow/Parquet `60.0.0`, `reqwest 0.13.5`, `tokio 1.53.3`, `bytes 1.12.1` (`CVE-2026-25541`), `memmap2 0.9.11` (`RUSTSEC-2026-0186`), `xxhash-rust`, `hashbrown`, `crc32fast`, dan protokol API HuggingFace Hub.
