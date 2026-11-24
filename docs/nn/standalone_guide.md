# Ra-Family Standalone Guide

This guide provides practical examples for integrating **Ra-Family (`luminamoon.nn`)** modules in training and inference pipelines.

---

## 1. Replacing Standard Transformer Layers

You can replace conventional normalization, activation, and attention blocks with their rational counterparts:

```python
from typing import Optional
import torch
import torch.nn as nn
from luminamoon.nn import RaNorm, RaSwiGLU, RaAttention

class RationalTransformerBlock(nn.Module):
    """Transformer block using rational normalization, attention, and gating."""

    def __init__(self, dim: int = 512, num_heads: int = 8, is_causal: bool = False):
        super().__init__()
        self.norm1 = RaNorm(dim)
        self.attn = RaAttention(dim=dim, num_heads=num_heads, qk_norm=True, is_causal=is_causal)
        self.norm2 = RaNorm(dim)
        self.mlp = RaSwiGLU(dim)  # defaults to hidden_features = int(dim * 8 / 3)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x

block = RationalTransformerBlock(dim=256, num_heads=4, is_causal=True)
x = torch.randn(2, 64, 256)
out = block(x)
print(f"Output shape: {out.shape}")  # (2, 64, 256)
```

> **Practical Notes**:
> - Set `qk_norm=True` on `RaAttention` when training from scratch so initial Q/K logits stay within the compact support of `ra_softmax` ($\Delta < 16$) and gradients remain active from step 1.
> - Pair `RaNorm` with dense activations (`RaGELU`, `RaSwiGLU`, `RaGLU`, `RaTanh`); avoid placing it immediately after highly sparse activations with isolated spikes, where $L_1$ scaling can reach $\mathcal{O}(D)$ compared to $\mathcal{O}(\sqrt{D})$ for `RMSNorm`.

For custom attention blocks that apply rotary embeddings (RoPE) or grouped-query projections directly to $(Q, K, V)$, use `ra_attention` to compute tiled rational attention without allocating the $[B, H, N, N]$ score matrix:

```python
import torch
from luminamoon.nn import ra_attention

q = torch.randn(2, 8, 2048, 64, requires_grad=True)
k = torch.randn(2, 8, 2048, 64, requires_grad=True)
v = torch.randn(2, 8, 2048, 64, requires_grad=True)

out = ra_attention(q, k, v, is_causal=True)
```

---

## 2. ConvNet Parameter Folding with `RaBatchNorm2d`

In vision models, folding batch normalization into preceding convolutional weights removes the memory and compute cost of normalization during inference:

```python
import torch
import torch.nn as nn
from luminamoon.nn import RaBatchNorm2d

class SimpleVisionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = RaBatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))

    def prepare_for_inference(self) -> nn.Conv2d:
        """Fuses batch normalization into a standalone Conv2d layer."""
        self.eval()
        w_folded, b_folded = self.bn.fold_into_conv(self.conv)
        fused = nn.Conv2d(
            self.conv.in_channels,
            self.conv.out_channels,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
            bias=True,
        )
        fused.weight.data.copy_(w_folded)
        fused.bias.data.copy_(b_folded)
        return fused

# Verify parameter folding
block = SimpleVisionBlock(in_channels=16, out_channels=32)
block.eval()

# Dummy input
x = torch.randn(1, 16, 28, 28)
out_original = block(x)

# Fold into a single Conv2d
fused_conv = block.prepare_for_inference()
out_fused = fused_conv(x)

# Numerical difference is within machine precision
diff = (out_original - out_fused).abs().max().item()
print(f"Max absolute folding discrepancy: {diff:.2e}")
```

---

## 3. Environment Fallback Introspection

To check which compute backends are active in your environment, use the runtime helper functions:

```python
from luminamoon.nn import (
    is_torch_available,
    is_numpy_available,
    is_triton_available,
    is_rust_simd_available,
    is_cubecl_available,
)

print(f"PyTorch available:   {is_torch_available()}")
print(f"NumPy available:     {is_numpy_available()}")
print(f"Triton GPU JIT:      {is_triton_available()}")
print(f"Rust SIMD CPU:       {is_rust_simd_available()}")
print(f"CubeCL GPU:          {is_cubecl_available()}")
```

If PyTorch is absent, `RaNorm`, `RaBatchNorm2d`, `RaTanh`, `RaGELU`, `ra_tanh`, `ra_gelu`, `pos_ra_gate`, `ra_swiglu`, `RaSwiGLU`, `ra_glu`, `RaGLU`, `ra_softmax`, `ra_attention`, `RaAttention`, `LinearCrossEntropy`, `linear_cross_entropy`, and `RaMoE` run using standard NumPy arrays without import errors.
