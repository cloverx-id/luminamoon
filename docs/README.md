# LuminaMoon Documentation Portal

Welcome to the documentation portal for **LuminaMoon (LuminaV v1.0.0)**.  
Selamat datang di portal dokumentasi **LuminaMoon (LuminaV v1.0.0)**.

Please select your preferred language / Silakan pilih bahasa dokumentasi:

---

### [English Documentation (`docs/en/`)](en/README.md)
1. **[01. Getting Started](en/01_getting_started.md)**: System requirements, installation, source compilation via Maturin/Rust, hardware audit, and quickstart guides for Standalone NumPy and PyTorch.
2. **[02. User Guide](en/02_user_guide.md)**: Buffer mode selection (Dual-Buffer CLM vs Single-Buffer TD-CLM), mixed-precision & stochastic rounding, boundary dynamics, state checkpointing, and distributed training (FSDP).
3. **[03. Configuration & API Reference](en/03_configuration_api.md)**: Comprehensive constructor hyperparameters table, public methods, validation rules, and Levenshtein typo suggestion system.
4. **[04. Architecture & Hardware Acceleration](en/04_architecture_and_hardware.md)**: C4 container diagram, Rust workspace crates breakdown (`core`, `simd`, `vulkan`, `dispatch`, `triton-aot`), zero-copy buffer protocol, and Python free-threaded runtime (PEP 703).

---

### [Dokumentasi Bahasa Indonesia (`docs/id/`)](id/README.md)
1. **[01. Panduan Memulai](id/01_getting_started.md)**: Kebutuhan sistem, instalasi via pip, kompilasi dari sumber via Maturin/Rust, audit kapabilitas hardware, serta contoh kilat NumPy dan PyTorch.
2. **[02. Panduan Penggunaan](id/02_user_guide.md)**: Pemilihan mode buffer (Dual-Buffer CLM vs Single-Buffer TD-CLM hemat 50% memori), mixed-precision & stochastic rounding, boundary dynamics, checkpointing, dan distributed training (FSDP).
3. **[03. Referensi Konfigurasi & API](id/03_configuration_api.md)**: Tabel lengkap seluruh hyperparameter konstruktor `LuminaV`, metode publik, validasi, dan auto-koreksi typo berbasis jarak Levenshtein.
4. **[04. Arsitektur & Akselerasi Hardware](id/04_architecture_and_hardware.md)**: Diagram C4 alur dispatch, bedah crate Rust (`core`, `simd`, `vulkan`, `dispatch`, `triton-aot`), protokol zero-copy memori, dan kompatibilitas Python free-threaded (PEP 703).

---

### [Internal Engineering & Library Audit Suite](library_audit/README.md)
- Deep technical verification and ABI compliance audits for Rust toolchain, PyO3/Maturin, PyTorch, Triton JIT, CUDA/ROCm, and Vulkan Ash compute pipeline.
