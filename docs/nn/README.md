# Ra-Family: Rational Neural Architecture

The **Ra-Family (`luminamoon.nn`)** provides lightweight neural network components built around rational arithmetic and bounded polynomial approximations.

By removing transcendental functions ($\tanh$, $\exp$) and square-root operations from normalization and activation layers, Ra-Family primitives reduce hardware evaluation complexity while maintaining stable gradient propagation and numerical robustness.

---

## Design Principles

Standard transformer and convolutional architectures rely on non-linear operations that map to specialized hardware circuits:

1. **Square-Root-Free Normalization**: Standard `LayerNorm` computes root-mean-square variance ($\sqrt{\sigma^2 + \epsilon}$), which translates to reciprocal square-root instructions (`rsqrt`) on CPUs and GPUs. `RaNorm` instead computes mean absolute deviation ($L_1$-mean), evaluating only a single scalar reciprocal per token vector.
2. **Strictly Positive Bounded Gating**: Standard GLU and SwiGLU activations can produce large outliers under low-precision quantization (FP8 / INT4). `PosRaGLU` enforces a strictly positive rational gate bounded within $(0, 1)$, dampening outlier growth without evaluating exponential functions.
3. **Rational Softmax & Tiled Attention**: Standard softmax computes $\exp(x)$, which executes on GPU Special Function Units (SFU) at 1/4 to 1/8 the throughput of standard fused multiply-add (FMA) pipelines. `ra_softmax` uses a Cayley-Padé rational approximation with exact compact support for large negative differences ($\Delta \ge 16$), running directly on standard arithmetic units, while `ra_attention` and `RaAttention` tile the computation so the full $[B, H, N, N]$ matrix is not materialized in GPU memory.
4. **Zero Mandatory Dependencies**: The core package remains lightweight (~2.4 MB) with `dependencies = []`. When PyTorch is installed, components dynamically inherit from `torch.nn.Module`. When PyTorch is absent, components fall back cleanly to standalone NumPy execution.

---

## Components

### 1. Normalization Layers

#### `RaNorm`
Token-level normalization using square-root-free, mean-free $L_1$-norm scaling:

$$\mu _ {|x|} = \frac{1}{D} \sum _ {i=1}^D |x _ i|, \qquad \mathrm{RaNorm}(x) = \frac{x}{\mu _ {|x|} + \epsilon} \odot \gamma + \beta$$

- Evaluates zero square roots and drops both mean subtraction and the Gaussian $C _ {L_1} = \sqrt{\pi/2}$ factor.
- Amortizes a single scalar reciprocal $(\mu _ {|x|} + \epsilon)^{-1}$ across all $D$ features.
- Scale-homogeneous: scaling $x$ by $\alpha > 0$ preserves the normalized output $\mathrm{RaNorm}(x)$.

```python
import torch
from luminamoon.nn import RaNorm

norm = RaNorm(768, eps=1e-5)
x = torch.randn(2, 128, 768)
out = norm(x)
```

#### `RaBatchNorm2d`
2D batch normalization using cross-batch mean absolute deviation (MAD). Includes inference weight folding (`fold_into_conv`) to compute folded `(w_conv, b_conv)` tensors for preceding `torch.nn.Conv2d` layers:

```python
import torch.nn as nn
from luminamoon.nn import RaBatchNorm2d

conv = nn.Conv2d(32, 64, kernel_size=3, padding=1)
bn = RaBatchNorm2d(num_features=64)

# Before inference: compute folded (w_conv, b_conv) parameters
bn.eval()
w_folded, b_folded = bn.fold_into_conv(conv)
```

---

### 2. Rational Activation & Gating

#### `RaTanh`, `ra_tanh`, `RaGELU`, & `ra_gelu`
Rational approximation of hyperbolic tangent and GELU with scale parameter $b$, unit derivative at the origin, $C^2$ smoothness across $x = 0$, and algebraic $\mathcal{O}(1/|x|^3)$ gradient tails:

$$\mathrm{RaTanh}(x; b) = \frac{b (b + |x|) x}{b (b + |x|) + x^2}, \qquad \mathrm{RaGELU}(x) = \frac{1}{2} x \left(1 + \mathrm{RaTanh}\left(\sqrt{\frac{2}{\pi}}\,x;\, 1\right)\right)$$

```python
from luminamoon.nn import RaTanh, RaGELU, ra_tanh, ra_gelu

act = RaTanh(b=1.0)
gelu = RaGELU()
y = act(x)
```

#### `PosRaGLU`, `RaGLU`, & `TriGLU`
Gated Linear Units with strictly positive rational gating bounded within $(0, 1)$:

$$\sigma _ {\text{Rat}}(z) = \frac{1}{2} \left( 1 + \mathrm{RaTanh}(z; 1) \right) \in (0, 1)$$

$$\mathrm{PosRaGLU}(x) = W _ {\text{down}}\left(W _ {\text{up}}(x) \odot \sigma _ {\text{Rat}}(W _ {\text{gate}}(x))\right)$$

- Bounded output: because $\lim _ {x \to \pm\infty} \mathrm{RaTanh}(x; b) = \pm b$, using the default unit anchor $b = 1$ ensures $\mathrm{RaTanh}(z; 1) \in (-1, 1)$ and $\sigma _ {\text{Rat}}(z) \in (0, 1)$ strictly. When parameterized with $b > 0$, the gate normalizes as $\sigma _ {\text{Rat}}(z; b) = \frac{1}{2}(1 + \frac{1}{b}\mathrm{RaTanh}(z; b))$. This keeps hidden activations bounded by $\|W _ {\text{up}}(x)\| _ \infty$, reducing activation outliers during INT4 and FP8 quantization.
- Also provides symmetric `RaGLU` ($W _ {\text{down}}(W _ {\text{up}}(x) \odot \mathrm{RaTanh}(W _ {\text{gate}}(x); 1))$) and three-way `TriGLU` ($W _ {\text{down}}(W _ {\text{val}}(x) \odot \sigma _ {\text{Rat}}(W _ {g1}(x)) \odot (1 + \mathrm{RaTanh}(W _ {g2}(x); 1)))$).

```python
from luminamoon.nn import PosRaGLU

glu = PosRaGLU(768, hidden_features=3072)  # maps 768 -> 3072 -> 768
out = glu(x)
```

---

### 3. Rational Attention & Softmax

#### `ra_softmax`
Cayley-Padé rational soft-argmax with compact support and closed-form autograd:

- Computes shift $\Delta _ i = x _ {\max} - x _ i \ge 0$.
- For $\Delta _ i < 16$, evaluates the rational numerator:

  $$u _ i = \frac{\Delta _ i}{16}, \qquad q _ i = \frac{1 - u _ i}{1 + u _ i}, \qquad \phi _ i = q _ i^8$$

- For $\Delta _ i \ge 16$ ($u _ i \ge 1$), $\phi _ i$ is set exactly to $0.0$.
- Because $x _ {\max}$ produces $\Delta = 0$ ($\phi _ {\max} = 1.0$), the denominator satisfies $\sum \phi _ i \ge 1.0$, ensuring numerical stability without an artificial epsilon floor.
- **Closed-Form Backward Pass**: Evaluates the analytical derivative directly from the output probabilities $p _ i$ and scaled shift $u _ i$ without saving intermediate power buffers ($q, q^2, q^4, \phi$) in memory:

  $$\frac{\partial \phi _ i}{\partial u _ i} = -\frac{16}{1 - u _ i^2}\,\phi _ i \, \mathbb{I}(u _ i < 1)$$

#### `ra_attention` (Aliases: `flash_ra_attention`, `rational_attention`, `pade_attention`) & `RaAttention` (`RaAttn`)
Fused tiled rational attention and multi-head attention module:

- Computes $(Q, K, V) \to O$ without materializing the $[B, H, N _ Q, N _ K]$ attention matrix in GPU memory.
- On CUDA with Triton, executes a 2-pass SRAM tiled kernel (`BLOCK_M = 64`, `BLOCK_N = 64`) that skips loading $V$ tiles whenever all entries in a block fall outside the compact support ($\min u \ge 1$).
- When Triton is unavailable or an external `attn_mask` is passed, falls back to query-chunked tiling (`q_chunk_size=128`) with closed-form backward recomputation, keeping saved autograd state strictly $\mathcal{O}(B \cdot H \cdot N _ Q \cdot D)$.

```python
from luminamoon.nn import RaAttention, ra_attention

# 1. Multi-Head Attention Module
attn = RaAttention(dim=768, num_heads=12, qk_norm=True, attn_drop=0.0, proj_drop=0.1, causal=True)
x = torch.randn(2, 128, 768)
out = attn(x)

# 2. Functional Tiled Attention (drop-in for scaled_dot_product_attention)
q = torch.randn(2, 12, 128, 64)
k = torch.randn(2, 12, 128, 64)
v = torch.randn(2, 12, 128, 64)
out_fn = ra_attention(q, k, v, is_causal=True)
```

---

## Standalone NumPy Execution (Zero-PyTorch Mode)

When PyTorch is not installed, `RaNorm`, `RaBatchNorm2d`, `RaTanh`, `RaGELU`, `ra_tanh`, `ra_gelu`, `pos_ra_gate`, `pos_ra_glu`, `ra_softmax`, and `ra_attention` run on NumPy arrays:

```python
import numpy as np
from luminamoon.nn import RaNorm, ra_gelu, ra_softmax

# 1. Normalization & Activation
norm = RaNorm(64)
x_np = np.random.randn(4, 64).astype(np.float32)
out_np = ra_gelu(norm(x_np))

# 2. Rational Softmax
logits = np.array([[2.0, 1.0, 0.0], [5.0, 1.0, -10.0]], dtype=np.float32)
probs = ra_softmax(logits, dim=-1)
```

---

## Hardware Acceleration

Ra-Family modules include built-in hardware fast-paths:

1. **GPU (Triton JIT)**: When running on a CUDA-enabled GPU with Triton available, `RaNorm`, `PosRaGLU`, `ra_softmax` (forward and backward), and `ra_attention` / `RaAttention` (tiled forward and backward) dispatch to register- and SRAM-fused Triton kernels (`triton_kernels.py`).
2. **CPU (Rust SIMD)**: When running on x86_64 CPUs, FP32 forward passes dispatch to vectorized AVX2/AVX-512 kernels in `luminav-simd` via zero-copy PyO3 FFI pointers.

---

## Academic Aliases

`luminamoon.nn` exports descriptive aliases alongside the `Ra*` names:

| Academic Alias | Component | Primary Literature |
| :--- | :--- | :--- |
| `L1Norm`, `L1LayerNorm` | `RaNorm` | Hoffer et al. (NeurIPS 2018), Wu et al. (IEEE TNNLS 2019), Zhang & Sennrich (NeurIPS 2019), Zhu et al. (CVPR 2025) |
| `L1BatchNorm2d` | `RaBatchNorm2d` | Wu et al. (IEEE TNNLS 2019), Hoffer et al. (NeurIPS 2018) |
| `RationalTanh`, `rational_tanh` | `RaTanh`, `ra_tanh` | Boullé et al. (NeurIPS 2020), Molina et al. (ICLR 2020), Tang & Townsend (ICML 2026) |
| `RationalGELU`, `rational_gelu` | `RaGELU`, `ra_gelu` | Boullé et al. (NeurIPS 2020), Kim et al. (ICML 2021), Tang & Townsend (ICML 2026) |
| `RationalGLU`, `BoundedRationalGLU` | `PosRaGLU` | Dauphin et al. (ICML 2017), Tang & Townsend (ICML 2026) |
| `RationalAttention`, `PadeAttention` | `RaAttention` | Duan et al. (NeurIPS 2024), Yang et al. (ICML 2024), Su et al. (arXiv 2023), Dao et al. (NeurIPS 2022) |
| `rational_attention`, `pade_attention`, `flash_ra_attention` | `ra_attention` | Dao et al. (NeurIPS 2022), Duan et al. (NeurIPS 2024), Prieto et al. (ICLR 2025) |
| `rational_softmax`, `pade_softmax` | `ra_softmax` | Prieto et al. (ICLR 2025), Saratchandran et al. (arXiv 2024/2026), Kim et al. (ICML 2021) |

```python
# Direct imports
from luminamoon.nn import L1Norm, RationalAttention, RationalTanh, rational_attention
```

---

## Theoretical References

For mathematical derivations, literature comparisons, and BibTeX entries, see:
- [Theoretical References](../references.md)
- BibTeX database: [`references/bibliography.bib`](https://github.com/cloverx-id/luminamoon/blob/main/references/bibliography.bib)
