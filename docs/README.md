# LuminaMoon Documentation Portal

Welcome to the documentation portal for **LuminaMoon**.  
Selamat datang di portal dokumentasi resmi **LuminaMoon**.

---

## 1. User Documentation / Panduan Pengguna

### [English Documentation (`docs/en/`)](en/README.md)
1. **[01. Getting Started](en/01_getting_started.md)**: System requirements, installation, source compilation via Maturin/Rust, hardware audit, and quickstarts for NumPy and PyTorch.
2. **[02. User Guide](en/02_user_guide.md)**: Buffer mode selection (Dual-Buffer CLM vs Single-Buffer TD-CLM), mixed-precision & stochastic rounding, boundary dynamics, state checkpointing, and distributed training (FSDP).
3. **[03. Configuration & API Reference](en/03_configuration_api.md)**: Hyperparameter table, public methods, validation rules, and Levenshtein typo suggestion system.
4. **[04. Architecture & Hardware Acceleration](en/04_architecture_and_hardware.md)**: Dispatch hierarchy, Rust workspace crates breakdown (`core`, `simd`, `vulkan`, `dispatch`, `triton-aot`), zero-copy buffer protocol, and Python free-threaded runtime (PEP 703).

### [Dokumentasi Bahasa Indonesia (`docs/id/`)](id/README.md)
1. **[01. Panduan Memulai](id/01_getting_started.md)**: Kebutuhan sistem, instalasi via pip, kompilasi dari sumber via Maturin/Rust, audit kapabilitas hardware, serta contoh kilat NumPy dan PyTorch.
2. **[02. Panduan Penggunaan](id/02_user_guide.md)**: Pemilihan mode buffer (Dual-Buffer CLM vs Single-Buffer TD-CLM), mixed-precision & stochastic rounding, boundary dynamics, checkpointing, dan distributed training (FSDP).
3. **[03. Referensi Konfigurasi & API](id/03_configuration_api.md)**: Tabel hyperparameter `LuminaV`, metode publik, validasi, dan auto-koreksi typo berbasis jarak Levenshtein.
4. **[04. Arsitektur & Akselerasi Hardware](id/04_architecture_and_hardware.md)**: Alur dispatch, bedah crate Rust (`core`, `simd`, `vulkan`, `dispatch`, `triton-aot`), protokol zero-copy memori, dan kompatibilitas Python free-threaded (PEP 703).

---

## 2. Developer Modular Documentation (`docs/optimize/`)

Deep technical documentation for systems engineers, kernel writers, and library developers working on the optimizer subsystem:

- **[Developer Optimizer Index](optimize/README.md)** / **[Versi Indonesia](optimize/README.id.md)**
- **[01. Core & Math](optimize/core/README.md)**: CLM, TD-CLM temporal difference, PRNG stochastic rounding, and bias correction.
- **[02. SIMD Vectorization](optimize/simd/README.md)**: AVX-512 and AVX2/FMA x86_64 intrinsics and Rayon parallel threadpool.
- **[03. Vulkan Compute](optimize/vulkan/README.md)**: Ash Vulkan 1.3/1.4 SPIR-V compute shaders and UMA zero-copy host buffers.
- **[04. Triton AOT & GPU Drivers](optimize/triton_aot/README.md)**: AOT compilation, CUDA/HIP driver dynamic loading, and standalone C-API.
- **[05. Dispatch & Routing](optimize/dispatch/README.md)**: Hardware capability detector and 131,072-element threshold heuristic.
- **[06. Python FFI & PyO3 Bridge](optimize/python_ffi/README.md)**: PyO3 GIL release (`py.detach()`), anti-aliasing safety, and NumPy buffer protocols.
- **[07. Python Runtime & GPU Mixin](optimize/python_runtime/README.md)**: `LuminaV` class, Triton JIT kernels, FSDP synchronization, and state checkpointing.

---

## 3. Engineering Audits (`docs/library_audit/`)

- **[Library Audit Suite](library_audit/README.md)**: Technical verification and ABI compliance audits for Rust toolchain, PyO3/Maturin, PyTorch, Triton JIT, CUDA/ROCm, and Vulkan Ash compute pipeline.
