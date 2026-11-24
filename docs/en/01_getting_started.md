# Getting Started

Setup guide, system requirements, source compilation, hardware verification, and quickstart examples for the **LuminaV Optimizer v2.0.0** and **LuminaMoon Data Engine v2.0.0**.

## System Requirements

- **Python**: Version 3.10 or later (compatible with Python 3.10 through 3.15+, including free-threaded runtime `abi3t` / PEP 703).
- **Operating Systems**: Linux (x86_64, aarch64), Windows (x86_64), or macOS (Apple Silicon & Intel).
- **Source Compilation (Optional)**:
  - Rust Toolchain: `rustc` and `cargo` version 1.95+ (Edition 2024, required for `std::cfg_select!`).
  - Python Build Tool: `maturin >= 1.15, < 2.0`.
- **Hardware Accelerators (Optional)**:
  - CPU SIMD: x86_64 processor supporting AVX2 or AVX-512 (F, DQ, BW, VL), or ARM64 processor with NEON (supported by both Optimizer and Data Engine).
  - CubeCL Portable GPU (optional Cargo feature): Cross-vendor GPU compute via `wgpu`, Vulkan, Metal, CUDA, HIP, or CPU backends (`luminav-cubecl`).
  - Triton AOT / JIT: NVIDIA CUDA (SM75–SM120) or AMD ROCm (`gfx908`, `gfx90a`, `gfx942`, `gfx1100`, `gfx1103`) with `triton >= 3.0.0` (or `triton-windows`) for JIT, or pre-compiled AOT ELF64 binaries via Rust.

## Installation

### 1. Standard Installation

The core `luminamoon` package has no mandatory Python dependencies (`dependencies = []`), providing `LuminaV`, `luminamoon.nn`, `luminamoon.data`, and the `luminamoon` CLI:

```bash
pip install luminamoon
```

### 2. Installation with Optional Extras

If you plan to use LuminaMoon with specific ML frameworks, Arrow/Polars exports, or GPU backends, install with extras:

```bash
# PyTorch integration
pip install "luminamoon[torch]"

# Triton JIT GPU acceleration (CUDA / ROCm)
pip install "luminamoon[gpu]"

# Apache Arrow & Polars integration for Data Engine
pip install "luminamoon[data]"

# NumPy 2.x standard
pip install "luminamoon[numpy]"

# Install all optional dependencies
pip install "luminamoon[all]"
```

### 3. Compiling from Source

To compile the native Rust workspace (`optimizer` + `data`) locally:

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

The compiled native library `_luminamoon_optimizer_rs` (exposing both Optimizer and Data Engine symbols) is placed into the `luminamoon/optimizer/` subpackage.

## Hardware Capability Detection & Interactive CLI

LuminaMoon provides static runtime probes for both the Optimizer and Data Engine subsystems, plus a built-in CLI reference:

```python
import luminamoon as lm
from luminamoon import LuminaV

# 1. Inspect Optimizer SIMD, CubeCL, and Triton AOT capabilities
hw = LuminaV.detect_hardware()
print("Optimizer Hardware:", hw)

# 2. Inspect Data Engine CPU SIMD byte-scanning instruction set ('AVX-512', 'AVX2', 'NEON', or 'SCALAR')
print("Data Engine CPU ISA:", lm.data_cpu_isa())

# 3. Print interactive parameter reference in Python / notebooks
lm.help("optimizer")
```

```bash
# Inspect API reference and live hardware from the terminal
luminamoon --help
luminamoon --topic=hardware
luminamoon --version
```

## Quickstart: Standalone NumPy (Without PyTorch)

LuminaV can be used directly for numerical computing or custom neural networks using NumPy without requiring PyTorch:

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
    grad_w = rng.standard_normal((64, 10), dtype=np.float32) * 0.1
    grad_b = rng.standard_normal((10,), dtype=np.float32) * 0.1

    optimizer.step(grads=[grad_w, grad_b])
    print(f"Step {step:02d} executed via Rust SIMD / CubeCL.")
```

## Quickstart: PyTorch Integration

Works directly with PyTorch `nn.Module` using the standard `torch.optim.Optimizer` interface:

```python
import torch
import torch.nn as nn
from luminamoon import LuminaV

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

optimizer = LuminaV(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    buffer=2,               # 2 = Dual-Buffer CLM
    master_weights="full",  # Maintain FP32 master weights and states ("none", "semi", or "full")
    stochastic_rounding=True,
    cautious=True,
)

criterion = nn.MSELoss()
x_batch = torch.randn(32, 64)
y_target = torch.randn(32, 10)

for step in range(1, 11):
    optimizer.zero_grad(set_to_none=True)
    out = model(x_batch)
    loss = criterion(out, y_target)
    loss.backward()
    optimizer.step()

    print(f"Step {step:02d} | Loss: {loss.item():.6f}")
```

## Quickstart: LuminaMoon Data Engine (`DataPipeline` & `StringTensor`)

Use `DataPipeline` as the main v2.0 entrypoint for streaming curation, `XXH3-128` deduplication, document-isolated Best-Fit sequence packing (`TokenBatch`), and resumable checkpointing:

```python
from luminamoon.data import DataPipeline, FusedPipeline

raw_docs = [
    {"text": "  Sample training document for language model pretraining.\r\n"},
    {"text": "Another valid high-entropy training document with diverse vocabulary."},
]

# 1. High-level streaming pipeline with document-isolated packing
pipe = DataPipeline(
    source=raw_docs,
    tokenizer=lambda s: [ord(c) for c in s],  # Or any HuggingFace / tiktoken tokenizer
    batch_size=2,
    seq_len=128,
    packing="best_fit",
    quality="pretraining",
    dedup="exact",
    checkpoint="./cache/train",
)

for batch in pipe:
    attn_mask = batch.causal_doc_mask()
    # Forward / backward / optimizer.step() ...
    pipe.commit()  # Atomically persists exact consumed boundaries and O(delta) .dedup WAL

# 2. Low-level fused SIMD preprocessing into StringTensor (Arrow PyCapsule + GPU buffers)
fused = FusedPipeline(trim_whitespace=True, normalize_newlines=True, min_chars=10, min_entropy=2.0)
tensor = fused([d["text"] for d in raw_docs])
arrow_array = tensor.to_arrow()    # Exports via Arrow C-Data Interface (__arrow_c_array__)
polars_series = tensor.to_polars() # Requires optional extra: pip install "luminamoon[data]"
```
