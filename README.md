# Loomy

<p align="left">
  <a href="https://doi.org/10.5281/zenodo.23973325"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.23973325.svg" alt="DOI"></a>
  <a href="https://pypi.org/project/luminamoon/"><img src="https://img.shields.io/pypi/v/luminamoon.svg?color=4f46e5" alt="PyPI version"></a>
  <a href="https://github.com/cloverx-id/luminamoon/actions/workflows/ci.yml"><img src="https://github.com/cloverx-id/luminamoon/actions/workflows/ci.yml/badge.svg" alt="CI Status"></a>
  <a href="eval/LSO_Eval.pdf"><img src="https://img.shields.io/badge/Eval_Docs-PDF-b31b1b" alt="Eval Docs"></a>
  <a href="https://cloverx-id.github.io/luminamoon/"><img src="https://img.shields.io/badge/docs-GitHub_Pages-06b6d4" alt="Documentation"></a>
  <a href="https://pypi.org/project/luminamoon/"><img src="https://img.shields.io/badge/python-3.10%20%E2%86%92%203.15t%20(free--threaded)-10b981" alt="Python Versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
</p>

**Loomy (`luminamoon`)** provides low-precision numerical optimization, rational neural network layers, and dataset text preprocessing utilities developed by **Silver Moon ([@cloverxion](https://github.com/cloverxion))** and **Lumina Moon ([@cloverx-id](https://github.com/cloverx-id)) contributors**.

Built with a modular Rust and Python architecture, `luminamoon` provides components for CPU/GPU parameter updates and text dataset preprocessing:

## Projects & Components

| Project / Component | Path | Status | Description | Documentation |
| :--- | :--- | :--- | :--- | :--- |
| **LuminaV Optimizer** | [`optimizer/`](optimizer/README.md) | `Active` (`v2.0.0`) | LuminaV v2.0.0 optimizer with CPU SIMD (AVX-512/AVX2/NEON), Vulkan UMA, Triton AOT (182 ELF64 GPU binaries) / JIT kernels, Triangle Family (`buffer="T1"`, `"T2"`), MurmurHash3 + 3D Weyl-Kronecker stochastic rounding, optional bucket fusion, and Triton JIT pre-warmup. | [Optimizer Guide](optimizer/README.md) |
| **LuminaMoon Data Engine** | [`data/`](data/README.md) | `Active` (`v2.0.0`) | Modular text preprocessing library: AVX-512/AVX2/NEON SIMD text normalization, `StringTensor` (CPU StringView & GPU contiguous buffers), Shannon entropy filtering, MinHash/XXH3-64 deduplication, PII masking, `TokenRingBuffer`, and 128B Ping-Pong checkpoint recovery. | [Data Engine Guide](data/README.md) |
| **LSO-Eval** | [`eval/`](eval/README.md) | `Active` | Algorithm-agnostic optimizer evaluation & verification: Z3 SMT solver checks, 2D trajectory benchmarks, numerical parity tests, and Two-Class 4B LLM & Vision evaluation. | [LSO-Eval Guide](eval/README.md) |
| **Changelogs** | [`changelogs/`](changelogs/README.md) | `Active` | Version histories separating Software Package releases (`v1.0.0`–`v2.0.0`) and LuminaV Algorithm Math versions (`v1.0.0`–`v2.0.0`). | [Changelogs](changelogs/README.md) |
| **Documentation** | [`docs/`](docs/README.md) | `Active` | Documentation index, developer guides (`docs/optimize/`, `docs/dataloader/`), and dependency version notes. | [Docs Index](docs/README.md) |
| **Ra-Family Architecture** | [`python/luminamoon/nn/`](python/luminamoon/nn/) | `Active` (`v2.0.0`) | Rational, `tanh`-free & square-root-free neural components: `RaNorm`, `RaBatchNorm2d`, `RaTanh`, `RaGELU`, `PosRaGLU`, `RaGLU`, `TriGLU`, `ra_softmax`, `RaAttention` (`ra_attention`), `RaEMA` (`ra_ema`), `LinearCrossEntropy` (`linear_cross_entropy`), and `RaMoE`. | [Ra-Family Guide](docs/nn/README.md) |
| **Python Package** | [`python/luminamoon/`](python/luminamoon/__init__.py) | `Active` (`v2.0.0`) | Lazy-loaded free-threaded (`abi3t`) Python namespace exposing `optimizer`, `nn`, and `data` with PyTorch, NumPy, and Hugging Face integration. | [API Reference](docs/en/03_configuration_api.md) |
| **Theoretical References** | [`references/`](references/README.md) | `Active` | Mathematical formulas, component mapping, literature comparisons, and BibTeX citations for LuminaV, Ra-Family, and Data Engine. | [References Guide](references/README.md) |

## Quickstart

### Installation

Install the compiled package wheel from PyPI:

```bash
pip install luminamoon
```

Or build from source using Maturin and the Rust 2024 toolchain:

```bash
git clone https://github.com/cloverx-id/luminamoon.git
cd luminamoon
pip install maturin
maturin develop --release
```

### Basic Usage (LuminaV Optimizer)

```python
import numpy as np
from luminamoon import LuminaV

# Initialize parameters
weights = np.random.randn(64, 10).astype(np.float32)
biases = np.zeros(10, dtype=np.float32)

# Instantiate LuminaV optimizer (uses default hyperparameters)
optimizer = LuminaV([weights, biases], lr=8e-4, weight_decay=0.08, buffer=2)

# Execute optimization step with explicit gradients
grad_w = np.random.randn(64, 10).astype(np.float32)
grad_b = np.random.randn(10, dtype=np.float32)
optimizer.step(grads=[grad_w, grad_b])
```

For PyTorch deep learning workflows, see the [PyTorch Integration Guide](docs/en/01_getting_started.md#quickstart-pytorch-integration).

### LuminaV Hyperparameter Quick Reference

All default values below are exposed in IDE autocompletion and `LuminaV.__init__` docstrings:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `lr` | `float` | `3e-4` (`0.0003`) | Learning rate (`η`). |
| `betas` | `Tuple[float, float]` | `(0.9, 0.999)` | Exponential moving average factors `(β₁, β₂)` for first-moment momentum and second-moment variance. |
| `eps` | `float` | `1e-8` | Numerical stability constant (`ε`) in the adaptive denominator. |
| `weight_decay` | `float` | `0.08` (`8e-2`) | Decoupled weight decay coefficient. |
| `tau` | `float` | `0.8` | Denominator temperature scaling factor (`τ`) for 2-buffer variance damping. |
| `lam` | `float` | `0.3` | Triangle-defect penalty weight (`λ`) on `\|d_t\| = \|g_t - m_{t-1}\|` in 1-Buffer TD-CLM (`buffer=1`) and Triangle Family (`buffer="T1"`, `"T2"`). |
| `cautious` | `bool` | `True` | Enables sign-aligned cautious update masking and lagged EMA scaling. |
| `cautious_clamp_min` | `float` | `0.5` | Lower bound in `(0.0, 1.0]` for cautious alignment scaling. |
| `buffer` | `int \| str` | `2` | State mode: `2` for 2-Buffer CLM, `1` for 1-Buffer TD-CLM, `"T2"` (`"t2"`) for 2-Buffer Triangle Family (`tanh`-free), or `"T1"` (`"t1"`) for 1-Buffer Triangle Family (`tanh`-free). |
| `stochastic_rounding` | `bool` | `True` | Applies MurmurHash3 + 3D Weyl-Kronecker stochastic rounding when writing to `bfloat16` / `float16` weights and states. |
| `bound` | `bool` | `True` | Enables adaptive update bounding. |
| `bound_type` | `str` | `"coordinate"` | Bounding mode: `"coordinate"` (element-wise) or `"radial"` (global norm ratio). |
| `bound_ratio` | `float` | `0.05` | Maximum update magnitude relative to parameter scale. |
| `master_weights` | `str` | `"none"` | Mixed-precision master weight mode for `bfloat16`/`float16`: `"none"`, `"semi"`, or `"full"`. |
| `execution` | `str` | `"auto"` | Hardware backend: `"auto"`, `"triton"`, `"vulkan"`, `"simd"`, `"eager"`, or `"scalar"`. |
| `fused_single_pass` | `bool` | `True` | Fuses parameter update and scalar reduction statistics into a single kernel pass. |
| `fuse_tensors` | `bool` | `False` | Coalesces parameters into contiguous 1D memory buffers to reduce kernel launch latency on fragmented architectures (e.g. Vision models / ViT). Not recommended for large LLMs or distributed FSDP training. |
| `bucket_cap_mb` | `float` | `500.0` | Maximum memory capacity per contiguous bucket chunk in megabytes when `fuse_tensors=True`. Partitions large models into bounded chunks. Emits a `UserWarning` if explicitly specified when `fuse_tensors=False`. |
| `prewarm_jit` | `bool` | `True` | Pre-compiles Triton GPU kernels on 64-element dummy buffers (`lr=0.0`) during initialization to avoid first-step JIT compilation delay. |
| `ema_decay` | `str \| float` | `"auto"` | EMA decay for lagged cautious/radial scalars (`"auto"` resolves to `0.80` for `buffer=2`/`"T2"` and `0.95` for `buffer=1`/`"T1"`). |
| `fsdp_sync` | `bool` | `False` | Synchronizes global reduction scratch buffers across `torch.distributed` FSDP shards. |
| `seed` | `int` | `1337` | Base seed for deterministic MurmurHash3 + 3D Weyl-Kronecker stochastic rounding. |

> **Note on `torch.float16` (`FP16`) Training**: Although `LuminaV` protects its internal 16-bit state buffers from underflow via Bitwise-BF16 state promotion (8-bit exponent), the forward pass and `loss.backward()` across external model layers run outside the optimizer boundary. When training language models (Transformers/LLMs) or image/vision models (ViT, CNN, Diffusion) in `torch.float16` (5-bit exponent), enable `torch.amp.GradScaler` so layer-to-layer gradients do not underflow to `0.0` or overflow during `loss.backward()` before reaching `optimizer.step()` (see the [Mixed-Precision Guide](docs/en/02_user_guide.md#fp16-training--gradscaler-language--vision-models)). `torch.bfloat16` already has an 8-bit exponent across the model graph and does not require `GradScaler`.

### Software vs. Algorithm Versioning (`v2.0.0` / Math `v2.0.0`)

`luminamoon` tracks two read-only version identifiers (documented in [`changelogs/`](changelogs/README.md)):

1. **Software Package Version (`v2.0.0`)**:
   - Tracks the package distribution, Python FFI bindings, Rust crates, `luminamoon.nn`, and `luminamoon.data`.
   - To run an older package release, install it directly via `pip install luminamoon==<version>` or check out its git tag.
   - See [`changelogs/SOFTWARE_CHANGELOG.md`](changelogs/SOFTWARE_CHANGELOG.md).

2. **LuminaV Algorithm Math Version (`v2.0.0`)**:
   - Tracks the optimizer mathematical specification (Dual-Buffer CLM, Single-Buffer TD-CLM, Triangle Family `T1`/`T2`, and MurmurHash3 + 3D Weyl-Kronecker stochastic rounding).
   - Exposed read-only via `LuminaV.MATH_VERSION` and `LuminaV.get_math_version()`.
   - See [`changelogs/LUMINAV_ALGORITHM_CHANGELOG.md`](changelogs/LUMINAV_ALGORITHM_CHANGELOG.md).

```python
import luminamoon
from luminamoon import LuminaV

print(luminamoon.__version__)       # Software package version
print(luminamoon.get_version())     # Software package version (read-only helper)
print(LuminaV.MATH_VERSION)         # "2.0.0" (optimizer math version)
print(LuminaV.get_math_version())   # "2.0.0"
```

### Basic Usage (LuminaMoon Data Engine: Fused Text Preprocessing Pipeline)

```python
from luminamoon.data import FusedPipeline, StringTensor, create_resumable_collate, ResumablePipeline

# 1. Fused single-pass text preprocessing in Rust
pipeline = FusedPipeline(
    trim_whitespace=True,
    normalize_newlines=True,  # AVX-512 / AVX2 / NEON SIMD
    min_chars=10,
    min_entropy=2.0,
)
string_tensor = pipeline(["  sample 1\r\n", "sample 2 with proper prose text"])

# 2. Export to Apache Arrow, Polars, and PyTorch buffers
polars_series = string_tensor.to_polars()
arrow_array = string_tensor.to_arrow()
offsets_tensor, payload_tensor = string_tensor.to_torch_gpu(device="cuda")

# 3. Stream dataset with background prefetching & 128B Ping-Pong checkpointing
pipe = ResumablePipeline(
    source=raw_data,
    fused_pipeline=pipeline,
    batch_size=64,
    prefetch_batches=4,
    checkpoint_path="./cache/checkpoint.meta",
)
for batch in pipe:
    # Training step...
    pipe.commit()  # Saves Dual-Slot A/B checkpoint and deduplication WAL
```

For detailed architecture, Shannon entropy filtering, MinHash deduplication, and PyTorch `collate_fn` adapters, see the [Data Engine Guide](data/README.md).

### Basic Usage (Ra-Family Rational Neural Architecture)

```python
import torch
from luminamoon.nn import RaNorm, PosRaGLU, RaAttention

# 1. Square-root-free token normalization (L1-mean)
norm = RaNorm(512)
x = torch.randn(2, 64, 512)
x_norm = norm(x)

# 2. Rational gated linear unit with strictly positive bounds in (0, 1)
mlp = PosRaGLU(512, hidden_features=2048)
x_ffn = mlp(x_norm)

# 3. Cayley-Padé rational multi-head attention (SFU-free softmax)
attn = RaAttention(dim=512, num_heads=8, qk_norm=True, is_causal=True)
out = attn(x_norm)
```

For complete architecture, parameter folding for Conv2d, and standalone NumPy usage, see the [Ra-Family Guide](docs/nn/README.md).

## Verification & CI/CD

All pull requests and releases are tested across Linux, Windows, and macOS:

```bash
# Run static analysis across the Rust workspace
cargo clippy --workspace --all-targets -- -D warnings

# Run Rust workspace unit and parity tests (Optimizer + Data Engine)
cargo test --workspace

# Run Python static type check
npx pyright python/luminamoon

# Run Python test suites
pytest -q tests/
python optimizer/tests/test_standalone_numpy.py
python optimizer/tests/test_end_to_end_training.py
python -B tests/test_gpu_vs_cpu_strings.py

# Run LSO-Eval verification
python -m eval
```

## Developer Resources

For contributors working on **Loomy (`luminamoon`)**:

- **Changelogs (Software & Algorithm)**: [`changelogs/`](changelogs/README.md)
- **Optimizer Modular Documentation**: [`docs/optimize/`](docs/optimize/README.md)
- **Data Engine Modular Documentation**: [`data/`](data/README.md)
- **Optimizer Library Notes**: [`docs/library_audit/`](docs/library_audit/README.md)
- **Data Standalone Library Notes**: [`data/docs/library_audit/`](data/docs/library_audit/README.md)
- **Data Engine Architecture & User Guide**: [`data/README.md`](data/README.md)
- **User Documentation**: [`docs/en/`](docs/en/README.md)

## Theoretical References & Citation

LuminaMoon's algorithms are built upon published mathematical principles, including Maligranda's triangle inequality envelopes, rational neural networks, and decoupled optimization mechanics.

For the full component mapping, literature comparisons, and BibTeX entries, see [`references/README.md`](references/README.md) and [`references/bibliography.bib`](references/bibliography.bib).

To cite this repository in academic work:

```bibtex
@misc{loomy2026,
  author = {Silver Moon and {Lumina Moon contributors}},
  title = {Loomy: Low-Precision Numerical Optimization, Rational Neural Architecture, and Data Processing Repository},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/cloverx-id/luminamoon}},
  doi = {10.5281/zenodo.23973325},
  note = {Loomy repository}
}
```

## Authors & Maintenance

**Loomy (`luminamoon`)** is developed and maintained by:

- **Silver Moon ([@cloverxion](https://github.com/cloverxion))**
- **Lumina Moon ([@cloverx-id](https://github.com/cloverx-id)) contributors**

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
