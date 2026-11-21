from __future__ import annotations

import math
from typing import Any, Optional

from ._attention_ops import _FlashRaAttentionFunction, _gqa_matmul_pv, _gqa_matmul_qk
from ._compat import BaseModule, _HAS_NUMPY, _HAS_TORCH, _is_torch_compiling, nn, np, torch
from .ema import RaEMA
from .norm import RaNorm
from .softmax import ra_softmax


def ra_attention(
    q: Any,
    k: Any,
    v: Any,
    attn_mask: Optional[Any] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    ra_scale: float = 0.0625,
    q_chunk_size: Optional[int] = None,
) -> Any:
    """Fused tiled Rational Scaled Dot-Product Attention supporting MHA and GQA."""
    if _HAS_TORCH and isinstance(q, torch.Tensor):
        squeeze_head = False
        if q.ndim == 3:
            q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
            squeeze_head = True
        elif q.ndim != 4:
            raise ValueError(f"ra_attention expects 3D (B, N, D) or 4D (B, H, N, D) tensors, got {q.ndim}D")

        h_q, h_kv = q.shape[1], k.shape[1]
        if h_kv <= 0 or (h_q % h_kv != 0) or v.shape[1] != h_kv:
            raise ValueError(
                f"ra_attention requires H_q ({h_q}) divisible by H_kv ({h_kv}) and k.shape[1] == v.shape[1] ({v.shape[1]})"
            )

        D = q.shape[-1]
        sm_scale = float(scale) if scale is not None else (1.0 / math.sqrt(D))
        ra_scale = float(ra_scale)

        N_Q, N_K = q.shape[-2], k.shape[-2]
        use_compile_graph = (
            _is_torch_compiling()
            and q_chunk_size is None
            and N_Q <= 2048
            and N_K <= 2048
            and (q.shape[0] * h_q * N_Q * N_K <= 33_554_432)
        )
        if dropout_p <= 0.0 and not use_compile_graph:
            if _is_torch_compiling():
                if k is q:
                    k = k.clone()
                if v is q or v is k:
                    v = v.clone()
                if attn_mask is not None and (attn_mask is q or attn_mask is k or attn_mask is v):
                    attn_mask = attn_mask.clone()
            out = _FlashRaAttentionFunction.apply(
                q,
                k,
                v,
                attn_mask,
                bool(is_causal),
                sm_scale,
                ra_scale,
                int(q_chunk_size) if q_chunk_size is not None else None,
            )
        else:
            scores = _gqa_matmul_qk(q * sm_scale, k)
            if is_causal:
                q_idx = torch.arange(N_Q, device=q.device).unsqueeze(1)
                k_idx = torch.arange(N_K, device=q.device).unsqueeze(0)
                scores = scores.masked_fill(q_idx < k_idx, float("-inf"))
            if attn_mask is not None:
                m_add = attn_mask.unsqueeze(1) if attn_mask.ndim == 3 else attn_mask
                scores = scores + m_add
            weights = ra_softmax(scores, dim=-1, scale=ra_scale)
            if dropout_p > 0.0:
                weights = torch.nn.functional.dropout(weights, p=float(dropout_p), training=True)
            out = _gqa_matmul_pv(weights, v)

        return out.squeeze(1) if squeeze_head else out

    if _HAS_NUMPY and isinstance(q, np.ndarray):
        D = q.shape[-1]
        sm_scale = float(scale) if scale is not None else (1.0 / math.sqrt(D))
        k_np, v_np = k, v
        if q.ndim == 4 and k_np.ndim == 4 and q.shape[1] != k_np.shape[1]:
            if k_np.shape[1] <= 0 or (q.shape[1] % k_np.shape[1] != 0) or v_np.shape[1] != k_np.shape[1]:
                raise ValueError(
                    f"ra_attention requires H_q ({q.shape[1]}) divisible by H_kv ({k_np.shape[1]})"
                )
            g = q.shape[1] // k_np.shape[1]
            k_np = np.repeat(k_np, g, axis=1)
            v_np = np.repeat(v_np, g, axis=1)
        scores = np.matmul(q, np.swapaxes(k_np, -2, -1)) * sm_scale
        if is_causal:
            N_Q, N_K = q.shape[-2], k_np.shape[-2]
            scores = scores + np.triu(np.full((N_Q, N_K), -np.inf, dtype=scores.dtype), k=1)
        if attn_mask is not None:
            m_add = np.expand_dims(attn_mask, axis=1) if (scores.ndim == 4 and attn_mask.ndim == 3) else attn_mask
            scores = scores + m_add
        weights = ra_softmax(scores, dim=-1, scale=ra_scale)
        return np.matmul(weights, v_np)

    raise TypeError(f"ra_attention expects torch.Tensor or numpy.ndarray, got {type(q)}")


class RaAttention(BaseModule):
    """Rational Multi-Head & Grouped-Query Attention with optional pre-attention RaEMA.

    Notes:
        Because `ra_softmax` enforces exact compact support (`phi_i = 0` for
        `Delta_i >= 16`), large initial query/key dot-product scores can push
        entries outside the support on step 1 and zero out gradients before
        training adapts. Enabling `qk_norm=True` is recommended when training
        from scratch.
    """

    q_proj: Any
    k_proj: Any
    v_proj: Any
    out_proj: Any
    q_norm: Any
    k_norm: Any
    ema: Any
    attn_drop: Any
    proj_drop: Any

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        ra_scale: float = 0.0625,
        q_chunk_size: Optional[int] = None,
        use_ema: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads})")

        self.num_kv_heads = self.num_heads if num_kv_heads is None else int(num_kv_heads)
        if self.num_kv_heads <= 0 or (self.num_heads % self.num_kv_heads != 0):
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
            )

        self.head_dim = self.dim // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(self.head_dim)
        self.ra_scale = float(ra_scale)
        self.q_chunk_size = int(q_chunk_size) if q_chunk_size is not None else None
        self.is_causal = bool(is_causal)
        self.attn_drop_p = float(attn_drop)
        self.use_ema = bool(use_ema)

        if _HAS_TORCH:
            self.ema = RaEMA(self.dim, device=device, dtype=dtype) if self.use_ema else None
            self.q_proj = nn.Linear(self.dim, self.dim, bias=qkv_bias, device=device, dtype=dtype)
            self.k_proj = nn.Linear(self.dim, self.kv_dim, bias=qkv_bias, device=device, dtype=dtype)
            self.v_proj = nn.Linear(self.dim, self.kv_dim, bias=qkv_bias, device=device, dtype=dtype)
            self.out_proj = nn.Linear(self.dim, self.dim, bias=qkv_bias, device=device, dtype=dtype)

            if qk_norm:
                self.q_norm = RaNorm(self.head_dim, device=device, dtype=dtype)
                self.k_norm = RaNorm(self.head_dim, device=device, dtype=dtype)
            else:
                self.q_norm = None
                self.k_norm = None

            self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0.0 else nn.Identity()
            self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()
        else:
            self.ema = None
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
        if self.ema is not None:
            x = self.ema(x)

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, N, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q).to(x.dtype)
        if self.k_norm is not None:
            k = self.k_norm(k).to(x.dtype)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        drop_p = self.attn_drop_p if self.training else 0.0
        out = ra_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=drop_p,
            is_causal=self.is_causal,
            scale=self.scale,
            ra_scale=self.ra_scale,
            q_chunk_size=self.q_chunk_size,
        )

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"scale={self.scale:.4f}, "
            f"is_causal={self.is_causal}, "
            f"use_ema={self.use_ema}"
        )


__all__ = ["ra_attention", "RaAttention"]
