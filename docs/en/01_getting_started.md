# Getting Started

Comprehensive setup guide, system requirements, source compilation, and quickstart examples for the **LuminaV v1.0.0** optimizer.

## System Requirements

- **Python**: Version 3.10 or later (fully compatible with Python 3.10 through 3.15+, including free-threaded runtime `abi3t` / PEP 703).
- **Operating Systems**: Linux (x86_64, aarch64), Windows (x86_64), or macOS (Apple Silicon & Intel).
- **Source Compilation (Optional)**:
  - Rust Toolchain: `rustc` and `cargo` version 1.75+.
  - Python Build Tool: `maturin >= 1.15, < 2.0`.
- **Hardware Accelerators (Optional)**:
  - CPU SIMD: x86_64 processor supporting AVX2 or AVX-512 (F, DQ, BW, VL).
  - Vulkan Compute: Vulkan 1.3 / 1.4 capable driver with compute shader support (SPIR-V).
  - Triton JIT: NVIDIA CUDA (SM75–SM120) or AMD ROCm (RDNA3/CDNA) with `triton >= 3.0.0` (or `triton-windows`).

## Installation

### 1. Standard Lightweight Installation (Zero Heavy Dependencies)

The core `luminamoon` package is ultra-lightweight, with a compiled Rust binary under 1 MB and zero mandatory heavy dependencies:

```bash
pip install luminamoon
```

### 2. Installation with Optional Extras

If you plan to use LuminaMoon with specific ML frameworks or hardware backends, install with extras:

```bash
# PyTorch integration
pip install "luminamoon[torch]"

# Triton JIT GPU acceleration (CUDA / ROCm)
pip install "luminamoon[gpu]"

# NumPy 2.x standard
pip install "luminamoon[numpy]"

# Install all optional dependencies
pip install "luminamoon[all]"
```

### 3. Compiling from Source

To compile the native extension locally for maximum CPU instruction optimization:

```bash
git clone https://github.com/cloverx-id/luminamoon.git
cd luminamoon

# Create and activate virtual environment (optional)
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install maturin build tool
pip install "maturin>=1.15,<2.0"

# Compile native extension in release mode
maturin develop --release
```

The compiled native library `_luminamoon_optimizer_rs` will be placed directly into the Python package directory `luminamoon/optimizer/`.

## Hardware Capability Detection

LuminaMoon provides a static method to query the runtime capabilities of the host system:

```python
from luminamoon import LuminaV

hw = LuminaV.detect_hardware()
print("Hardware Capabilities:", hw)
# Example output on an x86_64 system with AVX-512 and Vulkan:
# {
#     'simd_backend': 'x86_64 AVX-512 (avx512f+dq+bw+vl+fma)',
#     'triton_aot_available': False,
#     'triton_arch': None,
#     'vulkan_available': True,
#     'gpu_device_name': 'AMD Radeon (TM) 740M Graphics',
#     'wave32_supported': True,
#     'gpu_dispatch_threshold': 131072
# }
```

## Quickstart: Standalone NumPy (Zero PyTorch Dependency)

LuminaMoon can be used directly for data science, numerical computing, or custom neural networks using NumPy without requiring PyTorch:

```python
import numpy as np
from luminamoon import LuminaV

# 1. Initialize parameters as contiguous FP32 NumPy arrays
rng = np.random.default_rng(42)
weights = rng.standard_normal((64, 10), dtype=np.float32)
bias = np.zeros((10,), dtype=np.float32)

# 2. Instantiate LuminaV optimizer
optimizer = LuminaV(
    [weights, bias],
    lr=0.01,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    buffer=2,  # 2: Dual-Buffer CLM, 1: Single-Buffer TD-CLM
    bound=True,
    cautious=True,
)

# 3. Training loop with explicit gradients
for step in range(1, 11):
    # Compute analytical gradients from your loss function
    grad_w = rng.standard_normal((64, 10), dtype=np.float32) * 0.1
    grad_b = rng.standard_normal((10,), dtype=np.float32) * 0.1

    # Execute optimizer step passing gradient list
    optimizer.step(grads=[grad_w, grad_b])
    print(f"Step {step:02d} executed via Rust SIMD / Vulkan UMA.")
```

## Quickstart: PyTorch Integration

Direct integration with PyTorch `nn.Module` following the standard `torch.optim.Optimizer` interface:

```python
import torch
import torch.nn as nn
from luminamoon import LuminaV

# 1. Define PyTorch model
class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)

model = Regressor()

# 2. Instantiate LuminaV optimizer
optimizer = LuminaV(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    buffer=2,             # 2 = Dual-Buffer CLM
    master_weights="fp32", # Maintain FP32 master weights during FP16/BF16 training
    stochastic_rounding=True,
    cautious=True,
)

criterion = nn.MSELoss()
x_batch = torch.randn(32, 64)
y_target = torch.randn(32, 10)

# 3. Standard training loop
for step in range(1, 11):
    optimizer.zero_grad(set_to_none=True)
    out = model(x_batch)
    loss = criterion(out, y_target)
    loss.backward()
    optimizer.step()

    print(f"Step {step:02d} | Loss: {loss.item():.6f}")
```
