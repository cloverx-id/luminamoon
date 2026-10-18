# LuminaMoon: LuminaV Optimizer v1.0.0

LuminaMoon is a mathematical optimization engine implementing the **LuminaV v1.0.0** algorithm. It provides native hardware acceleration across CPU SIMD (AVX-512, AVX2), Vulkan compute shaders, and Triton GPU kernels, with zero-copy buffer interoperability for NumPy and PyTorch.

## Features

- **Dual-Buffer CLM & Single-Buffer TD-CLM**: Implements Cautious Learning Momentum with continuous defect compensation. Single-buffer mode reduces optimizer state memory by 50%.
- **Zero Heavy Dependencies**: Core package has an empty `dependencies = []` footprint and a sub-1 MB compiled binary. PyTorch, NumPy, and Triton remain optional.
- **Hardware Acceleration**:
  - **x86_64 CPU**: AVX-512 and AVX2 SIMD kernels parallelized across cores via Rayon.
  - **Vulkan 1.3/1.4**: Unified Memory Architecture (UMA) SPIR-V compute shaders for integrated and discrete non-NVIDIA GPUs.
  - **Triton AOT / JIT**: Ahead-of-time compiled kernels for NVIDIA CUDA (SM75–SM120) and AMD ROCm (RDNA3/CDNA).
  - **Scalar Core**: Portable pure-Rust fallback for other architectures (ARM/aarch64).
- **Framework Integration**:
  - **NumPy 2.x**: Native zero-copy execution via the Python buffer protocol and `__array_interface__`.
  - **PyTorch**: Standard `torch.optim.Optimizer` interface with FSDP collective synchronization and FP16/BF16 master weights.
- **Runtime Compatibility**: Built with CPython Stable ABI (`abi3` and `abi3t`), supporting Python 3.10 through 3.15+ including free-threaded execution (PEP 703).

## Architecture

The following diagram illustrates how runtime dispatch routes execution based on the input tensor type and available hardware:

```mermaid
graph TD
    UserScript["User Code (NumPy Arrays or PyTorch Tensors)"] --> LuminaVOpt["LuminaV Optimizer (Python)"]

    subgraph LuminaMoon Engine
        LuminaVOpt --> Dispatcher{"Runtime Dispatcher"}
        Dispatcher -->|CPU (NumPy / PyTorch)| RustFFI["PyO3 Native Bridge (_luminamoon_optimizer_rs)"]
        Dispatcher -->|GPU Vulkan UMA| Vulkan["luminav-vulkan Engine"]
        Dispatcher -->|CUDA / ROCm (Triton)| TritonJIT["Triton Kernels"]
        Dispatcher -->|PyTorch Fallback| PyTorchEngine["Pure PyTorch Fallback"]
        Dispatcher -->|NumPy Fallback| NumpyEngine["Pure NumPy Fallback"]

        RustFFI --> RustDispatch["luminav-dispatch"]
        RustDispatch -->|AVX-512 / AVX2| SIMD["luminav-simd"]
        RustDispatch -->|Scalar / aarch64| Scalar["luminav-core Scalar Core"]

        Vulkan --> VulkanGPU["Vulkan Device / UMA Hardware"]
        SIMD --> CPUHardware["Host CPU (x86_64)"]
        Scalar --> ARMHardware["ARM / aarch64 CPU"]
    end
```

## Installation

### Minimal Installation (Standalone Native & NumPy)

```bash
pip install luminamoon
```

Or build from source using Maturin:

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

# Define model parameters
w = np.random.randn(64, 10).astype(np.float32)
b = np.zeros(10, dtype=np.float32)

# Initialize LuminaV optimizer
optimizer = LuminaV(
    [w, b],
    lr=1e-2,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    buffer=2,  # 2 = Dual-Buffer CLM, 1 = Single-Buffer TD-CLM
)

# Optimization step with explicit gradients
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

## Running Tests

```bash
# Run Rust workspace unit and parity tests
cargo test --workspace

# Run standalone NumPy test suite
python optimizer/tests/test_standalone_numpy.py

# Run PyTorch integration test suite
python optimizer/tests/test_end_to_end_training.py
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.