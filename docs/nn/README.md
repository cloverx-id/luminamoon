# Ra-Family: Rational Neural Architecture

The **Ra-Family (`luminamoon.nn`)** provides neural network components built around rational arithmetic and bounded polynomial approximations.

By avoiding $\tanh$, $\exp$, and square-root operations in normalization, activation, and attention layers, these components simplify arithmetic evaluation while keeping gradient propagation stable.

---

## Design Principles

Standard transformer and convolutional architectures rely on non-linear operations that map to specialized hardware circuits:

1. **Square-Root-Free Normalization**: Standard `LayerNorm` computes root-mean-square variance ($\sqrt{\sigma^2 + \epsilon}$), which translates to reciprocal square-root instructions (`rsqrt`) on CPUs and GPUs. `RaNorm` instead computes mean absolute deviation ($L_1$-mean), evaluating one scalar reciprocal per token vector.
2. **Rational Gated FFNs**: `RaSwiGLU` replaces `SiLU(x) = x / (1 + exp(-x))` with `x * pos_ra_gate(x)` (no per-element `exp()`), while `RaGLU` provides hard-bounded gating via `RaTanh(gate; b) * up` for tighter activation bounds under low-precision quantization (FP8 / INT4).
3. **Rational Softmax & Tiled Attention**: Standard softmax computes $\exp(x)$, which runs on GPU Special Function Units (SFU) at lower throughput than standard FMA pipelines. `ra_softmax` uses a Cayley-Padé rational approximation with exact compact support for large negative differences ($\Delta \ge 16$), while `ra_attention` and `RaAttention` tile the computation so the full $[B, H, N, N]$ matrix is not materialized in GPU memory.
4. **Optional PyTorch Dependency**: The core package declares `dependencies = []`. When PyTorch is installed, modules inherit from `torch.nn.Module`. When PyTorch is absent, components run on standalone NumPy arrays.

---

## Components

### 1. Normalization Layers

#### `RaNorm`
Token-level normalization using square-root-free, mean-free $L_1$-norm scaling:

$$\mu _ {|x|} = \frac{1}{D} \sum _ {i=1}^D |x _ i|, \qquad \mathrm{RaNorm}(x) = \frac{x}{\mu _ {|x|} + \epsilon} \odot \gamma + \beta$$

- Evaluates zero square roots and drops both mean subtraction and the Gaussian $C _ {L_1} = \sqrt{\pi/2}$ factor.
- Amortizes a single scalar reciprocal $(\mu _ {|x|} + \epsilon)^{-1}$ across all $D$ features.
- Scale-homogeneous: scaling $x$ by $\alpha > 0$ preserves the normalized output $\mathrm{RaNorm}(x)$.
- **Input pairing note**: Works cleanly on dense Transformer activations (such as `RaGELU`, `RaSwiGLU`, `RaGLU`, and `RaTanh`). Because $L_1$ scaling can amplify a single isolated spike up to $\mathcal{O}(D)$ on highly sparse vectors (versus $\mathcal{O}(\sqrt{D})$ for $L_2$-based `RMSNorm`), avoid placing `RaNorm` directly after highly sparse activations with isolated spikes.

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

#### `RaSwiGLU`, `ra_swiglu`, `RaGLU`, & `ra_glu`
Rational gated linear units built around the strictly positive rational gate `pos_ra_gate` and `RaTanh`:

$$\sigma _ {\text{Rat}}(z; b) = \frac{1}{2} \left( 1 + \frac{1}{b}\mathrm{RaTanh}(z; b) \right) \in (0, 1)$$

$$\mathrm{RaSwiGLU}(x) = W _ {\text{down}}\left(W _ {\text{up}}(x) \odot \left(W _ {\text{gate}}(x) \odot \sigma _ {\text{Rat}}(W _ {\text{gate}}(x); b)\right)\right)$$

$$\mathrm{RaGLU}(x) = W _ {\text{down}}\left(W _ {\text{up}}(x) \odot \mathrm{RaTanh}(W _ {\text{gate}}(x); b)\right)$$

- `RaSwiGLU` matches standard SwiGLU capacity without per-element `exp()` evaluations (default `hidden_features = int(in_features * 8 / 3)`).
- `RaGLU` bounds the gate strictly to $(-b, b)$, keeping hidden activations linear in $\lVert W _ {\text{up}}(x)\rVert _ \infty$ for tighter FP8/INT4 quantization range.

```python
from luminamoon.nn import RaSwiGLU, RaGLU

ffn = RaSwiGLU(768)                        # defaults to hidden_features=2048
bounded_ffn = RaGLU(768, hidden_features=2048)
out = ffn(x)
```

---

### 3. Rational Attention & Softmax

#### `ra_softmax`
Cayley-Padé rational soft-argmax with compact support and closed-form autograd:

- Computes shift $\Delta _ i = x _ {\max} - x _ i \ge 0$.
- For $\Delta _ i < 16$, evaluates the rational numerator:

  $$u _ i = \frac{\Delta _ i}{16}, \qquad q _ i = \frac{1 - u _ i}{1 + u _ i}, \qquad \phi _ i = q _ i^8$$

- For $\Delta _ i \ge 16$ ($u _ i \ge 1$), $\phi _ i$ is set to $0.0$.
- Because $x _ {\max}$ produces $\Delta = 0$ ($\phi _ {\max} = 1.0$), the denominator satisfies $\sum \phi _ i \ge 1.0$, avoiding an artificial epsilon floor.
- **Closed-Form Backward Pass**: Evaluates the analytical derivative directly from output probabilities $p _ i$ and scaled shift $u _ i$ without saving intermediate power buffers ($q, q^2, q^4, \phi$):

  $$\frac{\partial \phi _ i}{\partial u _ i} = -\frac{16}{1 - u _ i^2}\,\phi _ i \, \mathbb{I}(u _ i < 1)$$

#### `ra_attention` & `RaAttention`
Fused tiled rational attention and multi-head / grouped-query attention module:

- Computes $(Q, K, V) \to O$ without materializing the $[B, H, N _ Q, N _ K]$ attention matrix in GPU memory.
- Supports Multi-Head Attention (MHA) and Grouped-Query Attention (GQA) via `num_kv_heads`, plus both additive floating-point masks and boolean masks (`True` = attend, `False` = mask).
- On CUDA with Triton, executes a 2-pass SRAM tiled kernel (`BLOCK_M = 64`, `BLOCK_N = 64`) that skips loading $V$ tiles whenever all entries in a block fall outside compact support ($\min u \ge 1$).
- When Triton is unavailable or an external `attn_mask` is passed, falls back to query-chunked tiling (`q_chunk_size=128`) with closed-form backward recomputation, keeping saved autograd state $\mathcal{O}(B \cdot H \cdot N _ Q \cdot D)$.
- **Initialization recommendation (`qk_norm=True`)**: Because `ra_softmax` sets $\phi _ i = 0$ for $\Delta _ i \ge 16$, large initial query/key logits can push attention scores outside compact support on step 1 and zero out gradients. Enabling `qk_norm=True` on `RaAttention` is recommended when training from scratch.

```python
from luminamoon.nn import RaAttention, ra_attention

# 1. Multi-Head / Grouped-Query Attention Module
attn = RaAttention(
    dim=768,
    num_heads=12,
    num_kv_heads=4,
    qk_norm=True,
    attn_drop=0.0,
    proj_drop=0.1,
    is_causal=True,
)
x = torch.randn(2, 128, 768)
out = attn(x)

# 2. Functional Tiled Attention
q = torch.randn(2, 12, 128, 64)
k = torch.randn(2, 12, 128, 64)
v = torch.randn(2, 12, 128, 64)
out_fn = ra_attention(q, k, v, is_causal=True)
```

---

### 4. Chunked Loss & Mixture-of-Experts

#### `LinearCrossEntropy` & `linear_cross_entropy`
Chunked linear projection and cross-entropy loss that avoids materializing the full `[B * N, V]` vocabulary logit tensor in memory:

- Supports stateful LM-head mode (`LinearCrossEntropy(in_features=..., vocab_size=...)`) and stateless criterion mode (`LinearCrossEntropy()` or `linear_cross_entropy(x, weight, target)`).
- Supports `ignore_index`, `reduction` (`"mean"`, `"sum"`, `"none"`), `label_smoothing`, `logit_scale`, `chunk_size`, and `save_logits=False` (set `save_logits=True` to trade VRAM for faster backward recomputation).

```python
from luminamoon.nn import LinearCrossEntropy

criterion = LinearCrossEntropy(in_features=768, vocab_size=32000)
hidden = torch.randn(2, 128, 768)
targets = torch.randint(0, 32000, (2, 128))
loss = criterion(hidden, targets)
```

#### `RaMoE` & `permute_tokens_by_expert`
Mixture-of-Experts block using `ra_softmax` routing, packed 3D expert weights with grouped GEMM (`expert_type="ra_swiglu"` or `"ra_glu"`), optional shared experts (`num_shared_experts`), optional `router_logit_bound` (bounding router logits via `ra_tanh` to avoid compact-support dead experts), and contiguous token permutation (`permute_tokens_by_expert`):

```python
from luminamoon.nn import RaMoE

moe = RaMoE(
    in_features=768,
    hidden_features=2048,
    num_experts=8,
    top_k=2,
    num_shared_experts=1,
    router_logit_bound=8.0,
)
x = torch.randn(2, 128, 768)
out = moe(x)
```

---

## Standalone NumPy Execution (Zero-PyTorch Mode)

When PyTorch is not installed, `RaNorm`, `RaBatchNorm2d`, `RaTanh`, `RaGELU`, `ra_tanh`, `ra_gelu`, `pos_ra_gate`, `ra_swiglu`, `RaSwiGLU`, `ra_glu`, `RaGLU`, `ra_softmax`, `ra_attention`, `RaAttention`, `LinearCrossEntropy`, `linear_cross_entropy`, and `RaMoE` run on NumPy arrays:

```python
import numpy as np
from luminamoon.nn import RaNorm, ra_gelu, ra_softmax

norm = RaNorm(64)
x_np = np.random.randn(4, 64).astype(np.float32)
out_np = ra_gelu(norm(x_np))

logits = np.array([[2.0, 1.0, 0.0], [5.0, 1.0, -10.0]], dtype=np.float32)
probs = ra_softmax(logits, dim=-1)
```

---

## Hardware Acceleration

Ra-Family modules include built-in hardware paths:

1. **GPU (Triton JIT)**: On CUDA GPUs with Triton installed, `RaNorm`, `RaSwiGLU`, `RaGLU`, `ra_softmax`, `ra_attention` / `RaAttention`, and `linear_cross_entropy` / `LinearCrossEntropy` dispatch to Triton kernels under `python/luminamoon/nn/kernels/`.
2. **Portable GPU (CubeCL, optional)**: When built with a CubeCL backend feature (e.g., `--features cubecl-wgpu`), `luminav-cubecl` accelerates `RaNorm`, `ra_tanh`, `ra_gelu`, `ra_swiglu`, `ra_glu`, and `ra_softmax` across `wgpu`, Vulkan, Metal, CUDA, and HIP (`is_cubecl_available()`).
3. **CPU (Rust SIMD)**: On x86_64 and ARM64 CPUs, FP32 forward passes dispatch to AVX-512, AVX2, or NEON kernels in `luminav-simd` via PyO3 FFI with `py.detach()`.

---

## Theoretical References

For mathematical derivations, literature comparisons, and BibTeX entries, see:
- [Theoretical References](../references.md)
- BibTeX database: [`references/bibliography.bib`](https://github.com/cloverx-id/luminamoon/blob/main/references/bibliography.bib)
