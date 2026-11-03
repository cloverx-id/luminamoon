# Ra-Family Standalone Developer Guide

This guide provides practical examples for integrating and benchmarking **Ra-Family (`luminamoon.nn`)** modules in production training and inference pipelines.

---

## 1. Replacing Standard Transformer Layers

You can replace conventional normalization, activation, and attention blocks with their rational counterparts:

```python
import torch
import torch.nn as nn
from luminamoon.nn import RaNorm, PosRaGLU, RaAttention

class RationalTransformerBlock(nn.Module):
    """Transformer block using rational normalization, attention, and gating."""

    def __init__(self, dim: int = 512, num_heads: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = RaNorm(dim)
        self.attn = RaAttention(dim=dim, num_heads=num_heads, qk_norm=True)
        self.norm2 = RaNorm(dim)
        self.mlp = PosRaGLU(dim=dim, expansion=mlp_ratio)

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        # Pre-norm attention residual
        x = x + self.attn(self.norm1(x), is_causal=is_causal)
        # Pre-norm feed-forward residual
        x = x + self.mlp(self.norm2(x))
        return x

# Quick verification
block = RationalTransformerBlock(dim=256, num_heads=4)
x = torch.randn(2, 64, 256)
out = block(x)
print(f"Output shape: {out.shape}")  # (2, 64, 256)
```

---

## 2. ConvNet Parameter Folding with `RaBatchNorm2d`

In vision models, fusing batch normalization into preceding convolutional weights eliminates the memory and compute overhead of the normalization step during deployment:

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
        """Fuses batch normalization into the convolutional layer."""
        self.eval()
        return self.bn.fold_into_conv(self.conv)

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

To check which compute backends are active in your environment, use the internal runtime flags:

```python
from luminamoon.nn._compat import (
    _HAS_TORCH,
    _HAS_NUMPY,
    _HAS_TRITON,
    _HAS_RUST_SIMD,
)

print(f"PyTorch available:   {_HAS_TORCH}")
print(f"NumPy available:     {_HAS_NUMPY}")
print(f"Triton GPU JIT:      {_HAS_TRITON}")
print(f"Rust SIMD CPU:       {_HAS_RUST_SIMD}")
```

If PyTorch is absent, `RaNorm`, `PosRaGLU`, `RaTanh`, and `ra_softmax` run seamlessly using standard NumPy arrays without throwing import errors.
