# LuminaV Optimizer v1.6.0 (luminamoon v1.1.1.post1)

LuminaV is a numerical optimization library implementing the LuminaV algorithm for low-precision training (BF16/FP16) without requiring FP32 master weights. Using bitwise stochastic rounding, it supports hardware execution across CPU SIMD (AVX-512, AVX2), Vulkan compute shaders, and Triton GPU kernels, with buffer interoperability for NumPy and PyTorch.

## Features

- **Contiguous Bucket Fusion & JIT Pre-Warmup**: Optionally coalesces parameter groups into 1D contiguous allocations (`fuse_tensors=True`) partitioned into bounded chunks (`bucket_cap_mb=500.0`) to reduce kernel launch overhead on models with many tensors, and pre-warms Triton JIT kernels during initialization (`prewarm_jit=True`).
- **Dual-Buffer CLM, Single-Buffer TD-CLM & Triangle Family (T-Family, `T1` / `T2`)**: Implements **Convex Lookahead Momentum (CLM)** (`buffer=2`), **Triangle-Defect Convex Lookahead Momentum (TD-CLM)** (`buffer=1`), and the `tanh`-free **Triangle Family** (`buffer="T2"` 2-Buffer Triangle Family Mode and `buffer="T1"` 1-Buffer Triangle Family Mode with the Triangle Inequality envelope, triangle-defect penalty, and rational update bounds). Single-buffer modes (`1` and `"T1"`) reduce optimizer state memory by 50% compared to two-buffer modes (`2` and `"T2"`).
- **Optional Framework Dependencies**: The core package has `dependencies = []` with a compiled binary under 1 MB. PyTorch, NumPy, and Triton are optional extras.
- **Hardware Acceleration**:
  - **x86_64 CPU**: AVX-512 and AVX2 SIMD kernels parallelized across cores via Rayon.
  - **Vulkan 1.3/1.4**: Unified Memory Architecture (UMA) SPIR-V compute shaders for integrated and discrete GPUs.
  - **Triton AOT / JIT**: Compiled kernels for NVIDIA CUDA (SM75–SM120) and AMD ROCm (RDNA3/CDNA).
  - **Scalar Core**: Portable Rust fallback for other architectures (ARM/aarch64).
- **Framework Integration**:
  - **NumPy 2.x**: Direct buffer access via the Python buffer protocol and `__array_interface__`.
  - **PyTorch**: Standard `torch.optim.Optimizer` interface with FSDP collective synchronization and optional FP16/BF16 master weights.
- **Runtime Compatibility**: Built with CPython Stable ABI (`abi3` and `abi3t`), supporting Python 3.10 through 3.15+ including free-threaded execution (PEP 703).

## Architecture & Dispatch

The runtime dispatcher selects the execution backend based on tensor device, memory size, and available hardware:

```mermaid
flowchart TD
    PyTorch["PyTorch Tensors (FP32 / BF16 / FP16)"] --> FFI["LuminaV PyO3 abi3t FFI"]
    NumPy["NumPy 2.x Arrays (Buffer Protocol)"] --> FFI
    FFI --> Dispatcher["Hardware Capability Dispatcher"]
    Dispatcher --> Triton["Triton AOT/JIT GPU (CUDA / ROCm)"]
    Dispatcher --> Vulkan["Vulkan 1.3/1.4 SPIR-V (UMA, N >= 16,384)"]
    Dispatcher --> SIMD["Rayon CPU SIMD (x86_64, N < 16,384)"]
    Dispatcher --> Scalar["Reference Scalar Core (Fallback)"]
    Triton & Vulkan & SIMD & Scalar --> Math["CLM, TD-CLM & Triangle Family (T1/T2) Update"]
```

## Workspace Crate Structure

```
optimizer/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Rust crate entry point
│   ├── optimizer.rs        # Core PyO3 optimizer implementation
│   └── python_ffi/         # Python-Rust FFI bridge (mod.rs, config_parser.rs, extract.rs, worker.rs)
├── crates/
│   ├── luminav-core/       # Mathematical formulas, PRNG, and state structs
│   ├── luminav-simd/       # AVX-512 and AVX2 explicit SIMD intrinsics
│   ├── luminav-vulkan/     # Vulkan 1.3/1.4 UMA compute pipeline (Ash)
│   ├── luminav-triton-aot/ # Triton AOT runner for CUDA and ROCm HIP
│   └── luminav-dispatch/   # Hardware capability detection and routing
├── tests/                  # Standalone NumPy and PyTorch verification tests
└── tools/                  # AOT compilers for SPIR-V and Triton
```

## Installation & Build

### From Wheel (PyPI)

```bash
pip install luminamoon
```

### From Source (Maturin)

```bash
git clone https://github.com/cloverx-id/luminamoon.git
cd luminamoon
pip install maturin
maturin develop --release
```

### Optional Dependencies

```bash
# PyTorch integration
pip install "luminamoon[torch]"

# Triton GPU acceleration (CUDA / ROCm)
pip install "luminamoon[gpu]"

# All optional dependencies
pip install "luminamoon[all]"
```

## Quickstart

### Standalone NumPy (No PyTorch Required)

```python
import numpy as np
from luminamoon import LuminaV

w = np.random.randn(64, 10).astype(np.float32)
b = np.zeros(10, dtype=np.float32)

optimizer = LuminaV(
    [w, b],
    lr=1e-2,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    buffer=2,  # 2 = Dual-Buffer CLM, 1 = Single-Buffer TD-CLM
)

grad_w = np.random.randn(64, 10).astype(np.float32)
grad_b = np.random.randn(10).astype(np.float32)

optimizer.step(grads=[grad_w, grad_b])
```

### PyTorch Integration

```python
import torch
import torch.nn as nn
from luminamoon import LuminaV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = LuminaV(
    model.parameters(),
    lr=8e-4,
    betas=(0.9, 0.999),
    weight_decay=8e-2,
    buffer=2,
    bound=True,
    bound_type="coordinate",
    cautious=True,
    fuse_tensors=False,    # Coalesce parameter groups into contiguous buckets
    bucket_cap_mb=500.0,   # Maximum chunk allocation size in MB
    prewarm_jit=True,      # Pre-compile Triton JIT kernels at initialization
)

inputs = torch.randn(64, 128, device=device)
targets = torch.randint(0, 10, (64,), device=device)

optimizer.zero_grad()
loss = criterion(model(inputs), targets)
loss.backward()
optimizer.step()
```

## Hardware Verification

To inspect detected hardware backends and active dispatch targets:

```python
from luminamoon import LuminaV

print(LuminaV.detect_hardware())
```

## Documentation & Library Audits

- **[Developer Modular Documentation (`docs/optimize/README.md`)](../docs/optimize/README.md)**
- **[User Documentation (`docs/en/README.md`)](../docs/en/README.md)**
- **[Optimizer External Library Audits (`docs/library_audit/`)](../docs/library_audit/README.md)**

## Running Tests

```bash
# Lint workspace
cargo clippy --workspace --all-targets -- -D warnings

# Rust workspace unit and parity tests
cargo test --workspace

# Standalone NumPy test
python optimizer/tests/test_standalone_numpy.py

# PyTorch integration test
python optimizer/tests/test_end_to_end_training.py
```

## References

- **Triangle Inequality Defect & Lookahead**: L. Maligranda, "Some remarks on the triangle inequality for norms," *Banach J. Math. Anal.*, vol. 2, no. 2, pp. 31–41, 2008; M. R. Zhang *et al.*, "Lookahead optimizer: k steps forward, 1 step back," *NeurIPS*, 2019.
- **Belief Variance & Decoupled Weight Decay**: J. Zhuang *et al.*, "AdaBelief optimizer: Adapting stepsizes by the belief in observed gradients," *NeurIPS*, 2020; S. Taniguchi *et al.*, "ADOPT: Modified Adam can converge with any β₂ with the optimal rate," *NeurIPS*, 2024; I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," *ICLR*, 2019.
- **Cautious Alignment & Trust-Region Clipping**: K. Liang *et al.*, "Cautious optimizers: Improving training with one line of code," *ICLR*, 2026; A. Brock *et al.*, "High-performance large-scale image recognition without normalization," *ICML*, 2021.
- **Padé Rational Approximants & Stochastic Rounding**: A. Molina *et al.*, "Padé activation units: End-to-end learning of flexible activation functions in deep networks," *ICLR*, 2020; S. Gupta *et al.*, "Deep learning with limited numerical precision," *ICML*, 2015; P. Zamirai *et al.*, "Revisiting BFloat16 training," *arXiv:2010.06192*, 2021.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../LICENSE) for details.
