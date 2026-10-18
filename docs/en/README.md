# LuminaMoon Documentation (English)

LuminaMoon is an ultra-high performance mathematical optimization engine powered by the **LuminaV v1.0.0** architecture. Engineered with multi-tier hardware acceleration, it unifies host CPU SIMD (AVX-512, AVX2), Vulkan 1.3/1.4 UMA Wave32 compute shaders, Triton JIT GPU kernels, and universal zero-copy fallback execution for both NumPy and PyTorch.

This documentation is designed to be structured, direct, and comprehensive for quick integration, configuration, and internal architectural understanding.

## Documentation Map

| Document | Description |
| :--- | :--- |
| [01. Getting Started](01_getting_started.md) | System requirements, pip installation, source compilation via Maturin/Rust, hardware auditing, and standalone NumPy & PyTorch quickstarts. |
| [02. User Guide](02_user_guide.md) | In-depth guide on buffer modes (Dual-Buffer CLM vs Single-Buffer TD-CLM), mixed-precision & stochastic rounding, boundary dynamics, state checkpointing, and distributed training (FSDP). |
| [03. Configuration & API Reference](03_configuration_api.md) | Complete constructor parameters table for `LuminaV`, public methods, parameter validation, and Levenshtein typo auto-suggestion. |
| [04. Architecture & Hardware Acceleration](04_architecture_and_hardware.md) | C4 dispatch flow, Rust crate breakdown (`luminav-core`, `luminav-simd`, `luminav-vulkan`, `luminav-dispatch`, `luminav-triton-aot`), zero-copy buffer protocol, and Python free-threaded support (PEP 703). |
| [Library Audit Suite](../library_audit/README.md) | In-depth audit reports on toolchain ABI, SIMD intrinsics, GPU drivers (CUDA/ROCm/Vulkan), and PyO3/Maturin bindings. |

## Core Architectural Highlights

- **Zero Heavy Mandatory Dependencies**: The core package ships with an empty `dependencies = []` footprint and a sub-1 MB compiled binary.
- **Dual-Buffer CLM & Single-Buffer TD-CLM**: Cautious Learning Momentum with dynamic defect compensation and optional 50% state memory savings.
- **Universal Zero-Copy Memory Access**: Direct hardware execution on NumPy 2.x arrays via `__array_interface__` and PyTorch Tensors.
- **Cross-Platform Free-Threaded Ready**: Compatible with Python 3.10 through 3.15+ via CPython Stable ABI (`abi3` & `abi3t` / PEP 703).
