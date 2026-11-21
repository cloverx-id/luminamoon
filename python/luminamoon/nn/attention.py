from __future__ import annotations

import math
from typing import Any, Optional

from ._attention_ops import _FlashRaAttentionFunction, _gqa_matmul_pv, _gqa_matmul_qk
from ._compat import (
    BaseModule,
    _HAS_NUMPY,
    _HAS_TORCH,
    _as_numpy,
    _init_linear_np,
    _is_torch_compiling,
    _linear_np,
    nn,
    np,
    torch,
)
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
    ra_scale = float(ra_scale)
    if not (math.isfinite(ra_scale) and ra_scale > 0.0):
        raise ValueError(f"ra_scale must be finite and > 0, got {ra_scale}.")
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
        squeeze_head = False
        q_np, k_np, v_np = q, k, v
        if q_np.ndim == 3:
            q_np = np.expand_dims(q_np, axis=1)
            k_np = np.expand_dims(k_np, axis=1)
            v_np = np.expand_dims(v_np, axis=1)
            squeeze_head = True
        elif q_np.ndim != 4:
            raise ValueError(f"ra_attention expects 3D (B, N, D) or 4D (B, H, N, D) arrays, got {q_np.ndim}D")

        h_q, h_kv = q_np.shape[1], k_np.shape[1]
        if h_kv <= 0 or (h_q % h_kv != 0) or v_np.shape[1] != h_kv:
            raise ValueError(
                f"ra_attention requires H_q ({h_q}) divisible by H_kv ({h_kv}) and k.shape[1] == v.shape[1] ({v_np.shape[1]})"
            )

        D = q_np.shape[-1]
        sm_scale = float(scale) if scale is not None else (1.0 / math.sqrt(D))
        if h_q != h_kv:
            g = h_q // h_kv
            k_np = np.repeat(k_np, g, axis=1)
            v_np = np.repeat(v_np, g, axis=1)
        scores = np.matmul(q_np, np.swapaxes(k_np, -2, -1)) * sm_scale
        if is_causal:
            N_Q, N_K = q_np.shape[-2], k_np.shape[-2]
            causal_mask = np.triu(np.ones((N_Q, N_K), dtype=bool), k=1)
            scores = np.where(causal_mask, -np.inf, scores)
        if attn_mask is not None:
            m_add = np.expand_dims(attn_mask, axis=1) if attn_mask.ndim == 3 else attn_mask
            scores = scores + m_add
        weights = ra_softmax(scores, dim=-1, scale=ra_scale)
        out = np.matmul(weights, v_np)
        return np.squeeze(out, axis=1) if squeeze_head else out

    raise TypeError(f"ra_attention expects torch.Tensor or numpy.ndarray, got {type(q)}")


class RaAttention(BaseModule):
    """Rational Multi-Head & Grouped-Query Attention with compact-support ra_softmax.

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
        if not (math.isfinite(self.ra_scale) and self.ra_scale > 0.0):
            raise ValueError(f"ra_scale must be finite and > 0, got {ra_scale}.")
        self.q_chunk_size = int(q_chunk_size) if q_chunk_size is not None else None
        self.is_causal = bool(is_causal)
        self.attn_drop_p = float(attn_drop)

        if _HAS_TORCH:
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
            self.q_proj = _init_linear_np(self.dim, self.dim, bias=qkv_bias)
            self.k_proj = _init_linear_np(self.dim, self.kv_dim, bias=qkv_bias)
            self.v_proj = _init_linear_np(self.dim, self.kv_dim, bias=qkv_bias)
            self.out_proj = _init_linear_np(self.dim, self.dim, bias=qkv_bias)
            self.q_norm = RaNorm(self.head_dim) if qk_norm else None
            self.k_norm = RaNorm(self.head_dim) if qk_norm else None
            self.attn_drop = None
            self.proj_drop = None

    def forward(
        self,
        x: Any,
        attn_mask: Optional[Any] = None,
    ) -> Any:
        if _HAS_TORCH and isinstance(x, torch.Tensor):
            B, N, C = x.shape

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

        if _HAS_NUMPY and isinstance(x, np.ndarray):
            B, N, C = x.shape
            x_in = x

            q = _linear_np(x_in, self.q_proj).reshape(B, N, self.num_heads, self.head_dim)
            k = _linear_np(x_in, self.k_proj).reshape(B, N, self.num_kv_heads, self.head_dim)
            v = _linear_np(x_in, self.v_proj).reshape(B, N, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

            if self.q_norm is not None:
                q = self.q_norm(q).astype(x_in.dtype, copy=False)
            if self.k_norm is not None:
                k = self.k_norm(k).astype(x_in.dtype, copy=False)

            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)

            attn_mask_np = _as_numpy(attn_mask) if attn_mask is not None else None
            out = ra_attention(
                q,
                k,
                v,
                attn_mask=attn_mask_np,
                dropout_p=0.0,
                is_causal=self.is_causal,
                scale=self.scale,
                ra_scale=self.ra_scale,
                q_chunk_size=self.q_chunk_size,
            )
            out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
            return _linear_np(out, self.out_proj)

        raise TypeError(f"RaAttention forward expects torch.Tensor or numpy.ndarray, got {type(x)}")

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"scale={self.scale:.4f}, "
            f"is_causal={self.is_causal}"
        )


__all__ = ["ra_attention", "RaAttention"]
