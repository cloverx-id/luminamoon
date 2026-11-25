# Getting Started

Setup guide, system requirements, source compilation, hardware verification, and ecosystem integration examples for **LuminaMoon `v2.0.0rc1`** (**LuminaV Math `v2.0.0`**, **Ra-Family**, and **Data Engine**).

## System Requirements

- **Python**: Version 3.10 or later (compatible with Python 3.10 through 3.15+, including free-threaded builds).
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

The core `luminamoon` package has no mandatory Python dependencies (`dependencies = []`), providing `LuminaV`, `luminamoon.nn`, `luminamoon.data`, and the `luminamoon` CLI. Because `2.0.0rc1` is currently a Release Candidate, pass `--pre` (once `2.0.0` GA ships, `pip install luminamoon` installs `2.0.0+` directly):

```bash
pip install --pre luminamoon
```

### 2. Installation with Optional Extras

If you plan to use LuminaMoon with specific ML frameworks, Arrow/Polars exports, or GPU backends, install with extras:

```bash
# PyTorch integration
pip install --pre "luminamoon[torch]"

# Triton JIT GPU acceleration (CUDA / ROCm)
pip install --pre "luminamoon[gpu]"

# Apache Arrow & Polars integration for Data Engine
pip install --pre "luminamoon[data]"

# NumPy 2.x standard
pip install --pre "luminamoon[numpy]"

# Core ML stack (NumPy + PyTorch + Triton GPU)
pip install --pre "luminamoon[all]"

# Core ML stack + Apache Arrow & Polars Data Engine extras
pip install --pre "luminamoon[all,data]"
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
from luminamoon.optimizer import AutoTune, LuminaV

# 1. Inspect Optimizer SIMD, CubeCL, and Triton AOT capabilities
hw = LuminaV.detect_hardware()
print("Optimizer Hardware:", hw)

# 2. Optional: AutoTune hyperparameters + steady-state backend speed (with optional loss_fn probe)
# tune = AutoTune(model)                                # Architecture/memory priors + speed benchmark
# tune = AutoTune(model, loss_fn=lambda m: loss_fn(m))  # + empirical full-model LR & spike probe
# optimizer = LuminaV(model.parameters(), tuning=tune)  # Overrides defaults & saves metadata in state_dict()
# Also works with external PyTorch optimizers:
# adamw = tune.build(torch.optim.AdamW, model.parameters())

# 3. Inspect Data Engine CPU SIMD byte-scanning instruction set ('AVX-512', 'AVX2', 'NEON', or 'SCALAR')
print("Data Engine CPU ISA:", lm.data_cpu_isa())

# 4. Print interactive parameter reference in Python / notebooks
lm.help("optimizer")
```

```bash
# Inspect API reference, run live backend probes/benchmarks, or install Triton (OS-aware)
luminamoon --help
luminamoon --topic=hardware
luminamoon doctor --benchmark
luminamoon setup triton --dry-run
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
polars_series = tensor.to_polars() # Requires optional extra: pip install --pre "luminamoon[data]"
```

## Ecosystem Integrations (Works Alongside Your Stack)

LuminaMoon is designed as a **middle-layer companion**, not a framework replacement. You keep PyTorch, HuggingFace, Arrow, Polars, or `accelerate` and drop in only the optimizer, neural layers, or data pipeline components you need.

### 1. HuggingFace `transformers` (`Trainer` or `accelerate`) + `LuminaV`

Pass `LuminaV` directly into HuggingFace `Trainer` via `optimizers=(optimizer, scheduler)` (or wrap it with `accelerate.Accelerator`) without altering your HuggingFace model:

```python
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, get_cosine_schedule_with_warmup
from luminamoon import LuminaV

model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.bfloat16)

optimizer = LuminaV(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.08,
    buffer="T2",                 # Or 2 (CLM), 1 (TD-CLM), "T1"
    cautious_weight_decay=True,  # Optional CWD
    stochastic_rounding=True,
)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=2000)

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./out", bf16=True, per_device_train_batch_size=4),
    train_dataset=my_dataset,
    optimizers=(optimizer, scheduler),
)
trainer.train()
```

### 2. PyTorch `FSDP2` (`DTensor`) + `torch.compile` + Drop-In `Ra-Family`

Mix `luminamoon.nn` layers selectively inside an existing PyTorch `nn.Module` alongside standard `nn.Linear` / `nn.Embedding`, compile with `torch.compile`, and shard with PyTorch FSDP2 (`fully_shard`):

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from luminamoon import LuminaV
from luminamoon.nn import RaNorm, RaSwiGLU, RaAttention, LinearCrossEntropy

class HybridTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.norm1 = RaNorm(dim)
        self.attn = RaAttention(dim, num_heads=num_heads, num_kv_heads=num_kv_heads, qk_norm=True, is_causal=True)
        self.norm2 = RaNorm(dim)
        self.ffn = RaSwiGLU(dim, int(dim * 8 / 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))

block = HybridTransformerBlock(dim=512, num_heads=8, num_kv_heads=2).to("cuda", dtype=torch.bfloat16)
fully_shard(block)                           # PyTorch FSDP2 (DTensor dim-0 sharding)
compiled_block = torch.compile(block)        # Standard torch.compile graph capture

loss_fn = LinearCrossEntropy(in_features=512, vocab_size=50304).to("cuda", dtype=torch.bfloat16)
optimizer = LuminaV(
    list(compiled_block.parameters()) + list(loss_fn.parameters()),
    lr=3e-4,
    buffer="T1",
    fsdp_sync=True,                          # Coalesced shard-group sync, DCP-safe DTensor state
)
```

### 3. HuggingFace `datasets` / `tokenizers` + `PyArrow` / `Polars` -> PyTorch

Place `FusedPipeline` and `DataPipeline` right between raw storage (Parquet, JSONL, HuggingFace `datasets`) and your existing tokenizer and PyTorch training loop:

```python
import pyarrow as pa
import polars as pl
from datasets import load_dataset
from tokenizers import Tokenizer
from luminamoon.data import FusedPipeline, DataPipeline

# A. Clean and filter raw text, then zero-copy export to Arrow / Polars via PyCapsule
cleaner = FusedPipeline(trim_whitespace=True, normalize_newlines=True, min_chars=32, min_entropy=2.5)
st = cleaner(["  Clean sample document for training.\r\n", "   ", "Another curated corpus paragraph."])

pa_arr = pa.array(st)              # Zero-copy Arrow Utf8View (__arrow_c_array__)
pl_series = pl.from_arrow(pa_arr)  # Direct Polars Series (or call st.to_polars())

# B. Stream Parquet / HuggingFace records into Best-Fit packed batches for PyTorch
hf_tok = Tokenizer.from_pretrained("gpt2")
ds = load_dataset("parquet", data_files="data/corpus/*.parquet", split="train", streaming=True)
pipe = DataPipeline(
    source=ds,
    tokenizer=lambda text: hf_tok.encode(text).ids,
    batch_size=8,
    seq_len=2048,
    packing="best_fit",
    pack_window=64,
    quality="pretraining",
    dedup="exact",
    checkpoint="./checkpoints/data_state",
)

for batch in pipe:
    input_ids = batch.input_ids          # Standard torch.Tensor / array ready for your model
    labels = batch.labels
    doc_mask = batch.causal_doc_mask()   # Prevents cross-document attention leakage
    # ... forward / backward / optimizer.step() ...
    pipe.commit()                        # Crash-safe .state.json + O(delta) .dedup WAL commit
```
