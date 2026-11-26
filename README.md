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

Written in Rust and Python, it includes:

## Projects & Components

| Project / Component | Path | Status | Description | Documentation |
| :--- | :--- | :--- | :--- | :--- |
| **LuminaV Optimizer** | [`optimizer/`](optimizer/README.md) | `Active` (`v2.0.0rc1`) | LuminaV v2.0.0 optimizer with CPU SIMD (AVX-512/AVX2/NEON), optional CubeCL GPU backend, Triton AOT (182 ELF64 GPU binaries) / JIT kernels, Triangle Family (`buffer="T1"`, `"T2"`), CWD, RTV, opt-in RPM (`momentum_type="rpm"`, pairwise Cayley rotation), configurable `bound_floor`, FSDP2/DTensor/HSDP sync, and MurmurHash3 + 3D Weyl-Kronecker stochastic rounding. | [Optimizer Guide](optimizer/README.md) |
| **LuminaMoon Data Engine** | [`data/`](data/README.md) | `Active` (`v2.0.0rc1`) | Data pipeline and text preprocessing library: `DataPipeline`, `TokenBatch`, `SequencePacker`, AVX-512/AVX2/NEON SIMD normalization, `StringTensor` (Arrow PyCapsule + GPU buffers), Shannon entropy filtering, XXH3-128 WAL & MinHash LSH deduplication, PII masking, `TokenRingBuffer`, and 256B `LMDL v2` Ping-Pong checkpoint recovery. | [Data Engine Guide](data/README.md) |
| **LSO-Eval** | [`eval/`](eval/README.md) | `Active` | Optimizer evaluation and Z3/parity checks: Z3 SMT solver checks, 2D trajectory benchmarks, numerical parity tests, and Two-Class 4B LLM & Vision evaluation. | [LSO-Eval Guide](eval/README.md) |
| **Changelogs** | [`changelogs/`](changelogs/README.md) | `Active` | Version histories separating Software Package releases (`v1.0.0`–`v2.0.0rc1`) and LuminaV Algorithm Math versions (`v1.0.0`–`v2.0.0`). | [Changelogs](changelogs/README.md) |
| **Documentation** | [`docs/`](docs/README.md) | `Active` | Documentation index, developer guides (`docs/optimize/`, `docs/dataloader/`), and dependency compatibility notes (`docs/dependencies/`). | [Docs Index](docs/README.md) |
| **Ra-Family Architecture** | [`python/luminamoon/nn/`](python/luminamoon/nn/) | `Active` (`v2.0.0rc1`) | Rational, `tanh`-free & square-root-free neural components: `RaNorm`, `RaBatchNorm2d`, `RaTanh`, `RaGELU`, `RaSwiGLU`, `RaGLU`, `ra_softmax`, `RaAttention` (`ra_attention`), `LinearCrossEntropy` (`linear_cross_entropy`), and `RaMoE`. | [Ra-Family Guide](docs/nn/README.md) |
| **Python Package & CLI** | [`python/luminamoon/`](python/luminamoon/__init__.py) | `Active` (`v2.0.0rc1`) | Lazy-loaded free-threaded (`abi3t`) Python namespace exposing `optimizer`, `nn`, `data`, and interactive CLI (`luminamoon --help` / `lm.help()`). | [API Reference](docs/en/03_configuration_api.md) |
| **Theoretical References** | [`references/`](references/README.md) | `Active` | Comparison with prior work, mathematical formulas, component mapping, literature notes, and BibTeX citations for LuminaV, Ra-Family, and Data Engine. | [References Guide](references/README.md) |

## Quickstart

### Installation

Install the current `v2.0.0rc1` pre-release from PyPI (once `2.0.0` GA ships, `pip install luminamoon` installs `2.0.0+` directly):

```bash
pip install --pre luminamoon
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
optimizer = LuminaV([weights, biases], lr=3e-4, weight_decay=0.08, buffer=2)

# Execute optimization step with explicit gradients
grad_w = np.random.randn(64, 10).astype(np.float32)
grad_b = np.random.randn(10, dtype=np.float32)
optimizer.step(grads=[grad_w, grad_b])
```

For PyTorch deep learning workflows, see the [PyTorch Integration Guide](docs/en/01_getting_started.md#quickstart-pytorch-integration).

### LuminaV Hyperparameter Quick Reference

All default values below are defined in `luminamoon.optimizer.constants` and printed by `lm.help("optimizer")`:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `lr` | `float` | `3e-4` (`0.0003`) | Learning rate (η). |
| `betas` | `Tuple[float, float]` | `(0.9, 0.999)` | Exponential moving average factors (β₁, β₂) for first-moment momentum and second-moment variance. |
| `eps` | `float` | `1e-8` | Numerical stability constant (ε) in the adaptive denominator. |
| `weight_decay` | `float` | `0.08` (`8e-2`) | Decoupled weight decay coefficient. |
| `cautious_weight_decay` | `bool` | `False` | Applies decoupled weight decay only where `step_val * param >= 0.0` (CWD). |
| `variance_type` | `str` | `"standard"` | 2-buffer variance update: `"standard"` (L2) or `"rtv"` (Rational-Tempered Variance). |
| `variance_temper` | `float` | `3.0` | Outlier-tempering ratio for `"rtv"` relative to `max(sigma_prev, abs(m_prev))`. |
| `momentum_type` | `str` | `"standard"` | Momentum update mode: `"standard"` (EMA) or `"rpm"` (opt-in pairwise Cayley rotation on adjacent coordinates with no extra state). |
| `tau` | `float` | `0.8` | Denominator temperature scaling factor (τ) for 2-buffer variance damping. |
| `lam` | `float` | `0.3` | Triangle-defect penalty weight (λ) on `abs(d_t) = abs(g_t - m_{t-1})` in 1-Buffer TD-CLM (`buffer=1`) and Triangle Family (`buffer="T1"`, `"T2"`). |
| `cautious` | `bool` | `True` | Enables sign-aligned cautious update masking and lagged EMA scaling. |
| `cautious_clamp_min` | `float` | `0.5` | Lower bound in `(0.0, 1.0]` for cautious alignment scaling. |
| `buffer` | <code>int &#124; str</code> | `2` | State mode: `2` for 2-Buffer CLM, `1` for 1-Buffer TD-CLM, `"T2"` (`"t2"`) for 2-Buffer Triangle Family (`tanh`-free), or `"T1"` (`"t1"`) for 1-Buffer Triangle Family (`tanh`-free). |
| `stochastic_rounding` | `bool` | `True` | Applies MurmurHash3 + 3D Weyl-Kronecker stochastic rounding when writing to `bfloat16` / `float16` weights and states. |
| `bound` | `bool` | `True` | Enables adaptive update bounding. |
| `bound_type` | `str` | `"coordinate"` | Bounding mode: `"coordinate"` (element-wise) or `"radial"` (global norm ratio). |
| `bound_ratio` | `float` | `0.05` | Maximum update magnitude relative to parameter scale. |
| `bound_floor` | `float` | `1e-2` | Coordinate bound floor radius `f`: `rho * (abs(p) + f)` or `rho * sqrt(p^2 + f^2)`. |
| `master_weights` | `str` | `"none"` | Mixed-precision master weight mode for `bfloat16`/`float16`: `"none"`, `"semi"`, or `"full"`. |
| `execution` | `str` | `"auto"` | Hardware backend: `"auto"`, `"triton"`, `"cubecl"`, `"simd"`, `"eager"`, or `"scalar"`. |
| `fused_single_pass` | `bool` | `True` | Fuses parameter update and scalar reduction statistics into a single kernel pass. |
| `fuse_tensors` | `bool` | `False` | Coalesces parameters into contiguous 1D memory buffers to reduce kernel launch latency on fragmented architectures (e.g. Vision models / ViT). Not recommended for large LLMs or distributed FSDP training. |
| `bucket_cap_mb` | `float` | `500.0` | Maximum memory capacity per contiguous bucket chunk in megabytes when `fuse_tensors=True`. Partitions large models into bounded chunks. Emits a `UserWarning` if set to a non-default value (`!= 500.0`) or overridden in a param group while `fuse_tensors=False`. |
| `prewarm_jit` | `bool` | `True` | Pre-compiles Triton GPU kernels on 64-element dummy buffers (`lr=0.0`) during initialization to avoid first-step JIT compilation delay. |
| `ema_decay` | <code>str &#124; float</code> | `"auto"` | EMA decay for lagged cautious/radial scalars (`"auto"` resolves to `0.80` for `buffer=2`/`"T2"` and `0.95` for `buffer=1`/`"T1"`). |
| `fsdp_sync` | `bool` | `False` | Synchronizes global reduction scratch buffers across `torch.distributed` FSDP1, FSDP2 (`DTensor`), HSDP, and DDP shards. |
| `seed` | `int` | `1337` | Base seed for deterministic MurmurHash3 + 3D Weyl-Kronecker stochastic rounding. |
| `tuning` | <code>AutoTune &#124; dict &#124; None</code> | `None` | Optional `AutoTune(model[, loss_fn=...])` or override dict that replaces omitted defaults while preserving explicit arguments and saving resolved settings in `state_dict()["luminav_tuning"]`. |

> **Note on `torch.float16` (`FP16`) Training**: Although `LuminaV` protects its internal 16-bit state buffers from underflow via Bitwise-BF16 state promotion (8-bit exponent), the forward pass and `loss.backward()` across external model layers run outside the optimizer boundary. When training language models (Transformers/LLMs) or image/vision models (ViT, CNN, Diffusion) in `torch.float16` (5-bit exponent), enable `torch.amp.GradScaler` so layer-to-layer gradients do not underflow to `0.0` or overflow during `loss.backward()` before reaching `optimizer.step()` (see the [Mixed-Precision Guide](docs/en/02_user_guide.md#fp16-training-and-gradscaler-language-and-vision-models)). `torch.bfloat16` already has an 8-bit exponent across the model graph and does not require `GradScaler`.

### Software vs. Algorithm Versioning (`v2.0.0rc1` / Math `v2.0.0`) & CLI Help

`luminamoon` tracks two read-only version identifiers (documented in [`changelogs/`](changelogs/README.md)):

1. **Software Package Version (`v2.0.0rc1`)**:
    - Tracks the package distribution, Python FFI bindings, Rust crates, `luminamoon.nn`, and `luminamoon.data`.
    - To run an older package release, install it directly via `pip install luminamoon==<version>` or check out its git tag.
    - See [`changelogs/SOFTWARE_CHANGELOG.md`](changelogs/SOFTWARE_CHANGELOG.md).

2. **LuminaV Algorithm Math Version (`v2.0.0`)**:
    - Tracks the optimizer mathematical specification (Dual-Buffer CLM, Single-Buffer TD-CLM, Triangle Family `T1`/`T2`, CWD, RTV, RPM, and MurmurHash3 + 3D Weyl-Kronecker stochastic rounding).
    - Exposed read-only via `LuminaV.MATH_VERSION` and `LuminaV.get_math_version()`.
    - See [`changelogs/LUMINAV_ALGORITHM_CHANGELOG.md`](changelogs/LUMINAV_ALGORITHM_CHANGELOG.md).

3. **API 2.0 Stability & Deprecation Policy (Effective at `2.0.0` GA)**:
    - Note that `v2.0.0rc1` is a release candidate and public APIs may still adjust before `2.0.0` GA.
    - Starting from `2.0.0` GA, public interfaces (`LuminaV`, `luminamoon.nn`, `luminamoon.data`, and `luminamoon.cli`) aim to keep backward compatibility across `2.x.y` releases; any deprecated public symbol or parameter emits an explicit `DeprecationWarning` and remains supported for **5 to 8 minor releases (`2.<minor>.x`)** before removal.
    - See [`docs/en/03_configuration_api.md`](docs/en/03_configuration_api.md#api-20-stability-and-deprecation-policy).

```python
import luminamoon as lm
from luminamoon import LuminaV

print(lm.__version__)               # Software package version ("2.0.0rc1")
print(lm.get_version())             # Software package version (read-only helper)
print(LuminaV.MATH_VERSION)         # "2.0.0" (optimizer math version)
print(LuminaV.get_math_version())   # "2.0.0"
lm.help("optimizer")                # Print parameter reference & hardware summary
```

```bash
# Terminal / notebook interactive API & hardware reference
luminamoon --help
luminamoon --topic=optimizer --output=luminav_help.txt
luminamoon --version
```

### Basic Usage (LuminaMoon Data Engine: `DataPipeline` & Packed `TokenBatch`)

```python
from luminamoon.data import DataPipeline

# v2.0 streaming pipeline: clean, quality-filter, XXH3-128 dedup, pack, and checkpoint
pipe = DataPipeline(
    source=raw_data,
    tokenizer=my_tokenizer,
    batch_size=4,
    seq_len=2048,
    packing="best_fit",           # Document-isolated Best-Fit packing
    quality="pretraining",        # Heuristic Shannon entropy, repetition, and symbol filters
    dedup="exact",                # 128-bit XXH3-128 with O(delta) append-only LMWL v2 WAL
    checkpoint="./cache/train",   # Resumable .state.json + .dedup checkpointing
)

for batch in pipe:
    # Pass batch.doc_ids directly to RaAttention / ra_attention for Triton SRAM
    # block-sparse document isolation without materializing an [N, N] mask.
    # Or call batch.causal_doc_mask() for standard attention layers.
    # Forward / backward / optimizer.step() ...
    pipe.commit()                         # Saves consumed document positions atomically
```

For lower-level `FusedPipeline`, `StringTensor` (Arrow PyCapsule + GPU buffers), `SequencePacker`, `TokenRingBuffer`, and `ResumablePipeline`, see the [Data Engine Guide](data/README.md).

### Basic Usage (Ra-Family Rational Neural Architecture)

```python
import torch
from luminamoon.nn import RaNorm, RaSwiGLU, RaAttention

# 1. Square-root-free token normalization (L1-mean)
norm = RaNorm(512)
x = torch.randn(2, 64, 512)
x_norm = norm(x)

# 2. Rational SwiGLU FFN block (strictly positive rational gate, 0 per-element exp)
mlp = RaSwiGLU(512, hidden_features=1365)
x_ffn = mlp(x_norm)

# 3. Cayley-Padé rational multi-head attention (SFU-free softmax)
attn = RaAttention(dim=512, num_heads=8, qk_norm=True, is_causal=True)
out = attn(x_norm)
```

For architecture details, `Conv2d` parameter folding, and standalone NumPy usage, see the [Ra-Family Guide](docs/nn/README.md).

## Verification & CI/CD

All pull requests and releases are tested across Linux, Windows, and macOS:

```bash
# Run static analysis across the Rust workspace
cargo clippy --workspace --all-targets -- -D warnings

# Run Rust workspace unit and parity tests (Optimizer + Data Engine)
cargo test --workspace

# Run Python static type check
npx pyright python/luminamoon

# Run Python tests
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
- **Data Engine Modular Documentation**: [`docs/dataloader/`](docs/dataloader/README.md)
- **Workspace Dependency Notes**: [`docs/dependencies/`](docs/dependencies/README.md)
- **Data Engine Architecture & User Guide**: [`data/README.md`](data/README.md)
- **User Documentation**: [`docs/en/`](docs/en/README.md)

## Theoretical References & Citation

LuminaMoon's algorithms are built upon published mathematical principles, including Maligranda's triangle inequality envelopes, rational neural networks, and decoupled weight decay.

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
