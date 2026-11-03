from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

from ._compat import (
    BaseModule,
    _HAS_NUMPY,
    _HAS_TORCH,
    np,
    torch,
    nn,
)
from .norm import RaNorm


def ra_softmax(scores: Any, dim: int = -1, scale: float = 0.0625) -> Any:
    """
    Cayley-Padé Rational Softmax with Compact Support.

    Replaces exponential-based softmax with a rational kernel:
        M_i = max_m S_{im}
        Delta_{ij} = M_i - S_{ij} >= 0
        u_{ij} = Delta_{ij} * scale        (default scale = 1/16 = 0.0625)
        phi_{Ra}(Delta) = (max(0, 1 - u) / (1 + u))^8
        RaSoftmax(S)_{ij} = phi_{Ra}(Delta_{ij}) / sum_k phi_{Ra}(Delta_{ik})

    Mathematical Properties:
      1. Order-2 Match with Exponential: Matches Maclaurin series of exp(-Delta)
         up to second derivative: phi(Delta) = 1 - Delta + 0.5 * Delta^2 - O(Delta^3).
      2. Exact Compact Support: For Delta >= 16, u >= 1 -> phi(Delta) == 0.0 exactly.
         Completely eliminates attention dilution and background noise accumulation
         in long contexts (N -> inf).
      3. Pole-Free & Zero-Epsilon: Because the max token has Delta = 0 -> phi(0) = 1.0,
         the denominator sum_k phi >= 1.0 is strictly guaranteed without requiring
         any epsilon safety floor.
      4. Hardware Efficiency: Zero exponential MUFU.EX2 instructions. Requires only
         a single reciprocal (MUFU.RCP) and 3 register squarings (q^2 -> q^4 -> q^8).
      5. Framework Agnostic: Operates seamlessly on torch.Tensor or numpy.ndarray.
    """
    scale = float(scale)

    # 1. PyTorch Tensor Path
    if _HAS_TORCH and isinstance(scores, torch.Tensor):
        max_scores = scores.max(dim=dim, keepdim=True)[0]
        delta = max_scores - scores  # Delta >= 0
        u = delta * scale
        num = torch.clamp(1.0 - u, min=0.0)
        den = 1.0 + u
        q = num / den
        q2 = q * q
        q4 = q2 * q2
        phi = q4 * q4
        sum_phi = phi.sum(dim=dim, keepdim=True)
        return phi / sum_phi

    # 2. NumPy ndarray Path
    if _HAS_NUMPY and isinstance(scores, np.ndarray):
        max_scores = np.max(scores, axis=dim, keepdims=True)
        delta = max_scores - scores
        u = delta * scale
        num = np.maximum(1.0 - u, 0.0)
        den = 1.0 + u
        q = num / den
        q2 = q * q
        q4 = q2 * q2
        phi = q4 * q4
        sum_phi = np.sum(phi, axis=dim, keepdims=True)
        return phi / sum_phi

    raise TypeError(f"ra_softmax expects torch.Tensor or numpy.ndarray, got {type(scores)}")


class RaAttention(BaseModule):
    """
    Rational Multi-Head Attention (RaAttention / RaAttn).

    Uses rational Cayley-Padé softmax (ra_softmax) with compact
    support. Eliminates exponential instructions and prevents attention
    dilution on long sequence lengths.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        causal: bool = False,
        scale: Optional[float] = None,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads})")

        self.head_dim = self.dim // self.num_heads
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(self.head_dim)
        self.causal = bool(causal)

        if _HAS_TORCH:
            self.q_proj = nn.Linear(self.dim, self.dim, bias=qkv_bias, device=device, dtype=dtype)
            self.k_proj = nn.Linear(self.dim, self.dim, bias=qkv_bias, device=device, dtype=dtype)
            self.v_proj = nn.Linear(self.dim, self.dim, bias=qkv_bias, device=device, dtype=dtype)
            self.out_proj = nn.Linear(self.dim, self.dim, bias=qkv_bias, device=device, dtype=dtype)

            if qk_norm:
                self.q_norm: Optional[Any] = RaNorm(self.head_dim, device=device, dtype=dtype)
                self.k_norm: Optional[Any] = RaNorm(self.head_dim, device=device, dtype=dtype)
            else:
                self.q_norm = None
                self.k_norm = None

            self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0.0 else nn.Identity()
            self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()
        else:
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.out_proj = None
            self.q_norm = None
            self.k_norm = None
            self.attn_drop = None
            self.proj_drop = None

    def forward(
        self,
        x: Any,
        attn_mask: Optional[Any] = None,
    ) -> Any:
        if not (_HAS_TORCH and isinstance(x, torch.Tensor)):
            raise TypeError(f"RaAttention forward currently requires a torch.Tensor, got {type(x)}")

        B, N, C = x.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale

        if self.causal:
            mask = torch.triu(torch.full((N, N), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1)
            attn_scores = attn_scores + mask

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn_weights = ra_softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"scale={self.scale:.4f}, "
            f"causal={self.causal}"
        )


# Direct alias for RaAttention
RaAttn = RaAttention
