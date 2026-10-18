# LuminaMoon: LuminaV Optimizer Suite v1.0.0

LuminaMoon is an ultra-high performance mathematical optimization engine featuring the **LuminaV v1.0.0** architecture. Engineered with multi-tier hardware acceleration, it unifies native Rust SIMD (AVX-512, AVX2), Vulkan 1.3/1.4 compute shaders with Wave32/UMA zero-copy host buffers, Triton JIT acceleration, and native zero-copy memory buffers across NumPy and PyTorch.

## Key Architecture Highlights

- **Zero Mandatory Heavy Dependencies**: The core package has an empty `dependencies = []` footprint. Compiled Rust binary is under 1 MB, ensuring near-instant installation and execution without dragging heavy frameworks into lightweight environments.
- **Dual-Buffer CLM & Single-Buffer TD-CLM**: Cautious Learning Momentum (CLM) with continuous boundary dynamics and defect compensation.
- **Native Hardware Execution**:
  - **x86_64 CPU**: AVX-512 (F, DQ, BW, VL) and AVX2 SIMD intrinsics with multithreaded Rayon chunking.
  - **aarch64 / ARM**: Vectorized via LLVM auto-vectorization across the high-precision scalar core (hand-crafted ARM NEON intrinsic kernels scheduled on the v1.6.0 roadmap).
  - **Vulkan 1.3 / 1.4**: Zero-copy Unified Memory Architecture (UMA) SPIR-V compute shaders with Wave32 subgroup operations.
- **Polymorphic Python Integration (NumPy & PyTorch)**:
  - **NumPy 2.x Standard**: Native zero-copy memory access via `__array_interface__` and Python buffer protocol.
  - **PyTorch (Optional)**: Full `torch.optim.Optimizer` compliance, FSDP/DDP collective synchronization, and mixed-precision master weights (FP16/BF16).
  - **Triton JIT (Optional)**: Fused single-pass GPU execution for CUDA and Intel XPU.

## Architecture Overview (C4 Container Diagram)

```mermaid
graph TD
    UserScript["User Code (NumPy Arrays or PyTorch Tensors)"] --> LuminaVOpt["LuminaV Optimizer (Python)"]

    subgraph LuminaMoon Engine
        LuminaVOpt --> Dispatcher{"Device & Runtime Dispatcher"}
        Dispatcher -->|CPU (NumPy or PyTorch)| RustFFI["PyO3 Native Bridge (_luminamoon_optimizer_rs)"]
        Dispatcher -->|GPU Vulkan UMA| Vulkan["luminav-vulkan Compute Engine"]
        Dispatcher -->|CUDA / XPU (Optional Triton)| TritonJIT["Triton JIT Fused Kernels"]
        Dispatcher -->|PyTorch Fallback| PyTorchEngine["Universal PyTorch Engine"]
        Dispatcher -->|NumPy Fallback| NumpyEngine["Universal NumPy Engine"]

        RustFFI --> RustDispatch["luminav-dispatch Engine"]
        RustDispatch -->|AVX-512 / AVX2| SIMD["luminav-simd Engine"]
        RustDispatch -->|Scalar / aarch64| Scalar["luminav-core Scalar Core"]

        Vulkan --> VulkanGPU["Vulkan Device / UMA Hardware"]
        SIMD --> CPUHardware["Host CPU (x86_64)"]
        Scalar --> ARMHardware["ARM / aarch64 CPU"]
    end
```

## Python Forward-Compatibility & Free-Threaded Runtime (Python 3.10 - 3.15+)

1. **CPython Stable ABI (`abi3` & `abi3t` - PEP 384 & PEP 803)**:
   - Built against CPython 3.10 Stable ABI (`features = ["abi3-py310", "abi3t"]`).
   - A single compiled wheel runs natively on Python 3.10 through Python 3.15+ without recompilation.
   - Fully compatible with free-threaded CPython (`Py_GIL_DISABLED` / PEP 703).

2. **Universal Pure-Python Fallbacks**:
   - Automatic graceful fallback to pure NumPy or PyTorch if native acceleration binaries are absent.

## Installation

### Lightweight Base (Standalone Native & NumPy)

```bash
pip install luminamoon
```

Or install from source with Maturin:

```bash
git clone https://github.com/cloverx-id/luminamoon.git
cd luminamoon
pip install maturin
maturin develop --release
```

### Optional GPU & PyTorch Extensions

```bash
# For PyTorch integration
pip install ".[torch]"

# For Triton GPU acceleration (CUDA / XPU)
pip install ".[gpu]"

# All optional dependencies
pip install ".[all]"
```

## Quickstart: Standalone NumPy (Zero PyTorch Dependency)

```python
import numpy as np
from luminamoon import LuminaV

# 1. Define model weights
w = np.random.randn(64, 10).astype(np.float32)
b = np.zeros(10, dtype=np.float32)

# 2. Instantiate LuminaV
optimizer = LuminaV(
    [w, b],
    lr=0.01,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    buffer=2,  # 2 = Dual-Buffer CLM, 1 = Single-Buffer TD-CLM
)

# 3. Training step with explicit gradients
grad_w = np.random.randn(64, 10).astype(np.float32)
grad_b = np.random.randn(10).astype(np.float32)

optimizer.step(grads=[grad_w, grad_b])
```

## Quickstart: PyTorch Integration

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
    lr=8.0e-4,
    betas=(0.9, 0.999),
    weight_decay=8.0e-2,
    buffer=2,
    bound=True,
    bound_type="coordinate",
    cautious=True,
    master_weights="full",
)

inputs = torch.randn(64, 128, device=device)
targets = torch.randint(0, 10, (64,), device=device)

optimizer.zero_grad()
loss = criterion(model(inputs), targets)
loss.backward()
optimizer.step()
```

## Running Tests

```bash
# 1. Test Rust workspace crates
cargo test --workspace

# 2. Run Standalone NumPy test suite
python optimizer/tests/test_standalone_numpy.py

# 3. Run PyTorch End-to-End training & validation suite
python optimizer/tests/test_end_to_end_training.py
```

## License

Licensed under the Apache License, Version 2.0. See LICENSE for details.
