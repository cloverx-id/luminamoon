# Ra-Family Standalone Guide

This guide provides practical examples for integrating **Ra-Family (`luminamoon.nn`)** modules in training and inference pipelines.

---

## 1. Replacing Standard Transformer Layers

You can replace conventional normalization, activation, and attention blocks with their rational counterparts:

```python
from typing import Optional
import torch
import torch.nn as nn
from luminamoon.nn import RaNorm, PosRaGLU, RaAttention

class RationalTransformerBlock(nn.Module):
    """Transformer block using rational normalization, attention, and gating."""

    def __init__(self, dim: int = 512, num_heads: int = 8, mlp_ratio: int = 4, causal: bool = False):
        super().__init__()
        self.norm1 = RaNorm(dim)
        self.attn = RaAttention(dim=dim, num_heads=num_heads, qk_norm=True, causal=causal)
        self.norm2 = RaNorm(dim)
        self.mlp = PosRaGLU(dim, hidden_features=dim * mlp_ratio)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention residual
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        # Pre-norm feed-forward residual
        x = x + self.mlp(self.norm2(x))
        return x

# Quick verification
block = RationalTransformerBlock(dim=256, num_heads=4, causal=True)
x = torch.randn(2, 64, 256)
out = block(x)
print(f"Output shape: {out.shape}")  # (2, 64, 256)
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
)

print(f"PyTorch available:   {is_torch_available()}")
print(f"NumPy available:     {is_numpy_available()}")
print(f"Triton GPU JIT:      {is_triton_available()}")
print(f"Rust SIMD CPU:       {is_rust_simd_available()}")
```

If PyTorch is absent, `RaNorm`, `RaBatchNorm2d`, `RaTanh`, `RaGELU`, `ra_tanh`, `ra_gelu`, `pos_ra_gate`, `pos_ra_glu`, and `ra_softmax` run using standard NumPy arrays without import errors.
