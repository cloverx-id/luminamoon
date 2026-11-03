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
Token-level normalization using mean absolute deviation ($L_1$-mean):

$$\mu = \frac{1}{D} \sum _ {i=1}^D x _ i, \qquad s = \frac{1}{D} \sum _ {i=1}^D |x _ i - \mu|, \qquad \hat{x} = \frac{x - \mu}{s + \epsilon}$$

- Evaluates zero square roots.
- Amortizes a single scalar reciprocal $(s + \epsilon)^{-1}$ across all $D$ features.
- Scale-homogeneous: scaling $x$ by $\alpha > 0$ preserves the normalized output $\hat{x}$.

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
Rational approximation of hyperbolic tangent with unit derivative at the origin and algebraic gradient tails:

$$\text{ra-tanh}(x) = \frac{x (1 + |x|)}{1 + |x| + x^2}$$

```python
from luminamoon.nn import RaTanh, ra_tanh

act = RaTanh()
y = act(x)
```

#### `PosRaGLU`
Gated Linear Unit with strictly positive rational gating bounded within $(0, 1)$:

$$\mathrm{gate}(x) = \frac{x^2}{1 + |x| + x^2} \in [0, 1)$$

$$\mathrm{PosRaGLU}(x) = W_1 x \odot \mathrm{gate}(W _ {\text{gate}} x)$$

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
