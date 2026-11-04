# Ra-Family: Rational Neural Architecture

The **Ra-Family (`luminamoon.nn`)** provides lightweight neural network components designed around exact rational arithmetic and bounded polynomial approximations.

By removing transcendental functions ($\tanh$, $\exp$) and square-root operations from normalization and activation layers, Ra-Family primitives reduce hardware evaluation complexity while maintaining stable gradient propagation and numerical robustness.

---

## Design Principles

Standard transformer and convolutional architectures rely heavily on non-linear operations that map to specialized hardware circuits:

1. **Square-Root-Free Normalization**: Standard `LayerNorm` computes root-mean-square variance ($\sqrt{\sigma^2 + \epsilon}$), which translates to reciprocal square-root instructions (`rsqrt`) on CPUs and GPUs. `RaNorm` instead computes mean absolute deviation ($L_1$-mean), evaluating only a single scalar reciprocal per token vector.
2. **Strictly Positive Bounded Gating**: Standard GLU and SwiGLU activations can produce large outliers under low-precision quantization (FP8 / INT4). `PosRaGLU` enforces a strictly positive rational gate bounded within $(0, 1)$, dampening outlier explosion without evaluating exponential functions.
3. **Rational Softmax & Attention**: Standard softmax computes $\exp(x)$, which executes on GPU Special Function Units (SFU) at 1/4 to 1/8 the throughput of standard fused multiply-add (FMA) pipelines. `ra_softmax` uses a Cayley-Padé rational approximation with exact compact support for large negative differences ($\Delta \ge 16$), running directly on standard arithmetic units.
4. **Zero Mandatory Dependencies**: The core package remains lightweight (~2.4 MB) with `dependencies = []`. When PyTorch is installed, components dynamically inherit from `torch.nn.Module`. When PyTorch is absent, components fall back cleanly to standalone NumPy execution.

---

## Architectural Components

### 1. Normalization Layers

#### `RaNorm`
Token-level normalization using zero square-root, mean-free $L_1$-norm scale:

$$\mu _ {|x|} = \frac{1}{D} \sum _ {i=1}^D |x _ i|, \qquad \mathrm{RaNorm}(x) = \frac{x}{\mu _ {|x|} + \epsilon} \odot \gamma + \beta$$

- Evaluates zero square roots and eliminates mean subtraction overhead (following the scale-invariance principle established by Zhang & Sennrich 2019).
- Amortizes a single scalar reciprocal $(\mu _ {|x|} + \epsilon)^{-1}$ across all $D$ features.
- Scale-homogeneous: scaling $x$ by $\alpha > 0$ preserves the normalized output $\mathrm{RaNorm}(x)$.

```python
import torch
from luminamoon.nn import RaNorm

norm = RaNorm(dim=768, eps=1e-5)
x = torch.randn(2, 128, 768)
out = norm(x)
```

#### `RaBatchNorm2d`
2D batch normalization using cross-batch mean absolute deviation (MAD). Includes inference weight folding to fuse batch normalization directly into preceding `torch.nn.Conv2d` layers:

```python
import torch.nn as nn
from luminamoon.nn import RaBatchNorm2d

conv = nn.Conv2d(32, 64, kernel_size=3, padding=1)
bn = RaBatchNorm2d(num_features=64)

# During training: forward as normal
# ...

# Before inference: fuse BN parameters into Conv2d
fused_conv = bn.fold_into_conv(conv)
```

---

### 2. Rational Activation & Gating

#### `RaTanh` & `ra_tanh`
Rational approximation of hyperbolic tangent with scale parameter $b$, unit derivative at the origin, and algebraic gradient tails:

$$\mathrm{RaTanh}(x; b) = \frac{b (b + |x|) x}{b (b + |x|) + x^2}$$

```python
from luminamoon.nn import RaTanh, ra_tanh

act = RaTanh()
y = act(x)
```

#### `PosRaGLU`
Gated Linear Unit with strictly positive rational gating bounded within $(0, 1)$:

$$\sigma_{\text{Rat}}(z; b) = \frac{1}{2} \left( 1 + \mathrm{RaTanh}(z; b) \right) \in (0, 1)$$

$$\mathrm{PosRaGLU}(x) = (W _ {\text{up}} x) \odot \sigma_{\text{Rat}}(W _ {\text{gate}} x)$$

- Strictly bounded: because $\sigma_{\text{Rat}}(z) \in (0, 1)$, $\|\mathrm{PosRaGLU}(x)\|_\infty \le \|W _ {\text{up}} x\|_\infty$, eliminating catastrophic activation outliers ($>1000$) for stable INT4 / FP8 quantization.
- Includes consensus gating variants such as `TriGLU`: $\mathrm{TriGLU}(x) = (W _ {\text{up}} x) \odot \sigma_{\text{Rat}}(W _ {\text{gate1}} x) \odot \sigma_{\text{Rat}}(W _ {\text{gate2}} x)$.

```python
from luminamoon.nn import PosRaGLU

glu = PosRaGLU(dim=768, expansion=4)  # maps 768 -> 3072 -> 768
out = glu(x)
```

---

### 3. Rational Attention & Softmax

#### `ra_softmax`
Cayley-Padé rational soft-argmax with compact support:

- Computes shift $\Delta _ i = x _ {\max} - x _ i \ge 0$.
- For $\Delta _ i < 16$, evaluates the rational numerator:

  $$u _ i = \frac{\Delta _ i}{16}, \qquad q _ i = \frac{1 - u _ i}{1 + u _ i}, \qquad N _ i = q _ i^8$$

- For $\Delta _ i \ge 16$, the numerator is set exactly to $0.0$.
- Because $x _ {\max}$ produces $\Delta = 0$ ($N _ {\max} = 1.0$), the denominator satisfies $D = \sum N _ i \ge 1.0$, guaranteeing numerical stability without an artificial epsilon floor.

#### `RaAttention` (Alias: `RaAttn`)
Multi-head attention module powered by `ra_softmax`:

```python
from luminamoon.nn import RaAttention

attn = RaAttention(dim=768, num_heads=12, qk_norm=True, dropout=0.1)
x = torch.randn(2, 128, 768)
out = attn(x, is_causal=True)
```

---

## Standalone NumPy Execution (Zero-PyTorch Mode)

When PyTorch is not installed, all Ra-Family components continue to run using pure NumPy arrays:

```python
import numpy as np
from luminamoon.nn import RaNorm, PosRaGLU, ra_softmax

# 1. Normalization
norm = RaNorm(dim=64)
x_np = np.random.randn(4, 64).astype(np.float32)
out_np = norm(x_np)

# 2. Rational Softmax
logits = np.array([[2.0, 1.0, 0.0], [5.0, 1.0, -10.0]], dtype=np.float32)
probs = ra_softmax(logits, dim=-1)
```

---

## Hardware Acceleration

Ra-Family modules include built-in hardware acceleration fast-paths:

1. **GPU (Triton JIT)**: When running on a CUDA-enabled GPU with Triton available, `RaNorm`, `PosRaGLU`, and `ra_softmax` automatically route execution to register-fused Triton kernels (`triton_kernels.py`), keeping intermediate tensors entirely in SRAM.
2. **CPU (Rust SIMD)**: When running on x86_64 CPUs, forward passes can dispatch to vectorized AVX2/FMA kernels in `luminav-simd` via zero-copy PyO3 FFI pointers.


---

## Academic Aliases

For seamless integration with academic papers and standard deep learning terminology, `luminamoon.nn` exports aliases alongside the `Ra*` family names:

| Academic Alias | Original Component | Literature Foundation |
| :--- | :--- | :--- |
| `L1Norm`, `L1LayerNorm` | `RaNorm` | Hoffer et al. (NeurIPS 2018), Wu et al. (IEEE TNNLS 2019) |
| `L1BatchNorm2d` | `RaBatchNorm2d` | Wu et al. (IEEE TNNLS 2019) |
| `RationalTanh`, `rational_tanh` | `RaTanh`, `ra_tanh` | Boullé et al. (NeurIPS 2020), Molina et al. (ICLR 2020) |
| `RationalGELU`, `rational_gelu` | `RaGELU`, `ra_gelu` | Boullé et al. (NeurIPS 2020), Tang & Townsend (ICML 2026) |
| `RationalGLU`, `BoundedRationalGLU` | `PosRaGLU` | Dauphin et al. (ICML 2017), Tang & Townsend (ICML 2026) |
| `RationalAttention`, `PadeAttention` | `RaAttention` | Duan et al. (NeurIPS 2024), Prieto et al. (ICLR 2025) |
| `rational_softmax`, `pade_softmax` | `ra_softmax` | Prieto et al. (ICLR 2025), Saratchandran et al. (arXiv 2024) |

```python
# Direct imports
from luminamoon.nn import L1Norm, RationalAttention, RationalTanh
```

---

## References & Scientific Foundations

For detailed mathematical derivations, algorithm comparisons, and formal BibTeX entries, see:
- [Scientific Foundations and Theoretical References](../references.md)
- Formal BibTeX entries: [`references/bibliography.bib`](https://github.com/cloverx-id/luminamoon/blob/main/references/bibliography.bib)
