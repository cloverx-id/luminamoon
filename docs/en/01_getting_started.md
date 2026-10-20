# Getting Started

Comprehensive setup guide, system requirements, source compilation, hardware verification, and quickstart examples for the **LuminaV Optimizer v1.0.0** and **LuminaMoon DataLoader v1.0.0**.

## System Requirements

- **Python**: Version 3.10 or later (fully compatible with Python 3.10 through 3.15+, including free-threaded runtime `abi3t` / PEP 703).
- **Operating Systems**: Linux (x86_64, aarch64), Windows (x86_64), or macOS (Apple Silicon & Intel).
- **Source Compilation (Optional)**:
  - Rust Toolchain: `rustc` and `cargo` version 1.85+ (Edition 2024).
  - Python Build Tool: `maturin >= 1.15, < 2.0`.
- **Hardware Accelerators (Optional)**:
  - CPU SIMD: x86_64 processor supporting AVX2 or AVX-512 (F, DQ, BW, VL), or ARM64 NEON.
  - Vulkan Compute: Vulkan 1.3 / 1.4 capable driver with compute shader support (SPIR-V).
  - Triton JIT: NVIDIA CUDA (SM75–SM120) or AMD ROCm (RDNA3/CDNA) with `triton >= 3.0.0` (or `triton-windows`).

## Installation

### 1. Standard Installation (Zero Heavy Dependencies)

The core `luminamoon` package ships with zero mandatory Python dependencies (`dependencies = []`) and exposes both `LuminaV` and the `luminamoon.dataloader` pipeline:

```bash
pip install luminamoon
```

### 2. Installation with Optional Extras

If you plan to use LuminaMoon with specific ML frameworks or GPU backends, install with extras:

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

To compile the native Rust workspace (`optimizer` + `dataloader`) locally:

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

The compiled native library `_luminamoon_optimizer_rs` (linking both `luminamoon-optimizer` and `luminamoon-dataloader`) is placed directly into `python/luminamoon/optimizer/`.

## Hardware Capability Detection

LuminaMoon provides static runtime probes for both the Optimizer and DataLoader subsystems:

```python
import luminamoon as lm
from luminamoon import LuminaV

# 1. Inspect Optimizer SIMD, Vulkan UMA, and Triton AOT capabilities
hw = LuminaV.detect_hardware()
print("Optimizer Hardware:", hw)

# 2. Inspect DataLoader CPU SIMD byte-scanning instruction set ('AVX-512', 'AVX2', 'NEON', or 'SCALAR')
print("DataLoader CPU ISA:", lm.dataloader_cpu_isa())
```

## Quickstart: Standalone NumPy (Zero PyTorch Dependency)

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
    print(f"Step {step:02d} executed via Rust SIMD / Vulkan UMA.")
```

## Quickstart: PyTorch Integration

Direct integration with PyTorch `nn.Module` following the standard `torch.optim.Optimizer` interface:

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
    buffer=2,             # 2 = Dual-Buffer CLM
    master_weights="fp32", # Maintain FP32 master weights during FP16/BF16 training
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

## Quickstart: LuminaMoon DataLoader & HuggingFace Pipeline

Stream local or remote HuggingFace datasets (`.parquet`, `.jsonl`, `.txt`) with $O(1)$ crash recovery, 100-batch Pre-RAM watermark prefetching, two-stage `xxh3_64` deduplication, and multimodal Hub uploads:

```python
import luminamoon as lm

# 1. Stream local or remote hf://datasets/... with repo auto-discovery and tokenizer hook
for batch in lm.stream(
    "hf://datasets/org/corpus@main",
    split="train",
    batch_size=64,
    preram_batches=100,
    refill_trigger=10,
    download_workers=4,
    deduplicate=True,
    tokenizer=lambda texts, **kw: {"input_ids": [[len(t)] for t in texts]},
):
    records = batch.records
    input_ids = batch.tokens["input_ids"]
    # Deferred auto_commit persists GPU progress to the 128B Ping-Pong .meta slot automatically

# 2. Wrap an existing HuggingFace datasets.Dataset / IterableDataset with Rust SIMD + Rayon + XXH3-64 dedup
for batch in lm.from_hf_dataset(
    [{"text": "  Sample A \r\n"}, {"text": "Sample A"}, {"text": "Sample B"}],
    column="text",
    batch_size=32,
    deduplicate=True,
):
    print(batch.records)  # ['Sample A', 'Sample B']

# 3. Serialize & upload multimodal tables (Text, Image Bytes, JSON, Int64, Float64, Bool) to HuggingFace Hub
uri = lm.push_to_hub(
    repo="org/cleaned-multimodal-ds",
    data=[
        {"prompt": "hello", "image": b"\x89PNG...", "score": 0.98, "verified": True},
    ],
    path_in_repo="train/part-00000.parquet",
)
```
