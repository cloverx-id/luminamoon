"""Latent-KV attention and slot cross-attention modules for luminamoon.nn."""

from __future__ import annotations

import math
from typing import Any, Callable, Literal, Optional, Tuple

from ._attention_dispatch import (
    AttentionBackend,
    dispatch_attention,
    resolve_attention_backend,
    validate_attention_backend,
)
from ._attention_bias_ops import can_use_triton_ra_fused_bias, ra_attention_fused_bias
from ._compat import BaseModule, _HAS_TORCH, _is_torch_compiling, nn, torch
from .norm import RaNorm
from .position import DistanceDecayBias, QRoPE


class LatentKVAttention(BaseModule):
    """Multi-head attention with low-rank KV latent compression, QK-RaNorm, QRoPE, and DistanceDecayBias.

    Defaults to `attention_backend="auto"`, which inherits an active
    `use_attention_backend(...)` context or otherwise calls PyTorch SDPA without
    overriding external `sdpa_kernel(...)` settings. Also supports explicit
    presets (`"sdpa"`, `"ra"`, `"flash"`, `"cudnn"`, `"fa3"`, `"flex"`) and a
    custom `attention_fn(q, k, v, *, attn_mask, is_causal, scale)` callable.
    When `bias_mode="none"`, `hub_size=0`, and `past_kv is None`, self-attention
    passes `attn_mask=None, is_causal=True` so FlashAttention and cuDNN fast
    paths remain eligible.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        kv_latent_dim: Optional[int] = None,
        hub_size: int = 0,
        num_hub_heads: Optional[int] = None,
        bias_mode: Literal["linear", "rational", "none"] = "linear",
        attention_backend: AttentionBackend = "auto",
        attention_fn: Optional[Callable[..., Any]] = None,
        cache_mode: Literal["latent", "kv"] = "latent",
        max_seq_len: int = 4096,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not _HAS_TORCH:
            raise ImportError("LatentKVAttention requires PyTorch.")
        if dim <= 0 or num_heads <= 0:
            raise ValueError(f"dim and num_heads must be > 0, got dim={dim}, num_heads={num_heads}.")
        resolved_head_dim = int(head_dim) if head_dim is not None else (dim // num_heads)
        if resolved_head_dim <= 0 or resolved_head_dim % 4 != 0:
            raise ValueError(f"head_dim must be > 0 and divisible by 4 for QRoPE, got {resolved_head_dim}.")
        resolved_kv_dim = int(kv_latent_dim) if kv_latent_dim is not None else max(16, dim // 4)
        if resolved_kv_dim <= 0:
            raise ValueError(f"kv_latent_dim must be > 0, got {resolved_kv_dim}.")
        validated_backend = validate_attention_backend(attention_backend)
        if bias_mode not in ("linear", "rational", "none"):
            raise ValueError(f"bias_mode must be 'linear', 'rational', or 'none', got {bias_mode!r}.")
        if cache_mode not in ("latent", "kv"):
            raise ValueError(f"cache_mode must be 'latent' or 'kv', got {cache_mode!r}.")

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = resolved_head_dim
        self.inner_dim = self.num_heads * self.head_dim
        self.kv_latent_dim = resolved_kv_dim
        self.hub_size = int(hub_size)
        self.attention_backend: AttentionBackend = validated_backend
        self.attention_fn = attention_fn
        self.cache_mode: Literal["latent", "kv"] = cache_mode
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.kv_down_proj = nn.Linear(self.dim, self.kv_latent_dim, bias=False)
        self.kv_norm = RaNorm(self.kv_latent_dim, eps=eps, elementwise_affine=True, bias=False)
        self.kv_up_proj = nn.Linear(self.kv_latent_dim, 2 * self.inner_dim, bias=False)

        self.q_proj = nn.Linear(self.dim, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, self.dim, bias=False)

        self.q_norm = RaNorm(self.head_dim, eps=eps, elementwise_affine=True, bias=False)
        self.k_norm = RaNorm(self.head_dim, eps=eps, elementwise_affine=True, bias=False)

        self.qrope = QRoPE(head_dim=self.head_dim, max_seq_len=max_seq_len)
        self.distance_bias: Optional[DistanceDecayBias] = (
            DistanceDecayBias(
                num_heads=self.num_heads,
                hub_size=self.hub_size,
                num_hub_heads=num_hub_heads,
                mode=bias_mode,
            )
            if bias_mode != "none"
            else None
        )
        self._mask_cache: dict[tuple[Any, ...], Any] = {}

    def _build_causal_mask(
        self,
        seq_len: int,
        num_all: int,
        device: Any,
        dtype: Any,
        is_dense_with_hub: bool,
        has_past_kv: bool,
    ) -> Any:
        text_k_len = num_all - self.hub_size
        causal = torch.zeros((1, 1, seq_len, num_all), device=device, dtype=dtype)
        if is_dense_with_hub and not has_past_kv:
            text_q_len = seq_len - self.hub_size
            if self.hub_size > 0 and text_k_len > 0:
                causal[:, :, : self.hub_size, self.hub_size :] = float("-inf")
            rows = torch.arange(text_q_len, device=device).unsqueeze(1)
            cols = torch.arange(text_k_len, device=device).unsqueeze(0)
            causal[:, :, self.hub_size :, self.hub_size :].masked_fill_(cols > rows, float("-inf"))
        else:
            pos_q = torch.arange(text_k_len - seq_len, text_k_len, device=device).unsqueeze(1)
            pos_k = torch.arange(0, text_k_len, device=device).unsqueeze(0)
            causal[:, :, :, self.hub_size :].masked_fill_((pos_k > pos_q).view(1, 1, seq_len, text_k_len), float("-inf"))
        return causal

    def forward(
        self,
        x: Any,
        past_kv: Optional[Any] = None,
        is_dense_with_hub: bool = True,
        cache_position: Optional[Any] = None,
        attn_mask: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        batch, seq_len, _ = x.shape
        c_kv_cur = self.kv_norm(self.kv_down_proj(x))
        # Normalize on contiguous (B, S, H, D) before transposing; QRoPE reads views with unit stride along D.
        q = self.q_norm(self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)).transpose(1, 2)

        has_past = past_kv is not None
        if cache_position is not None:
            if self.cache_mode != "kv" or not has_past:
                raise ValueError("cache_position requires cache_mode='kv' with a pre-allocated past_kv=(k_buf, v_buf).")
            k_up, v_up = torch.split(self.kv_up_proj(c_kv_cur), [self.inner_dim, self.inner_dim], dim=-1)
            k_cur = self.k_norm(k_up.reshape(batch, seq_len, self.num_heads, self.head_dim)).transpose(1, 2)
            v_cur = v_up.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            q = self.qrope(q, offset=cache_position)
            k_cur = self.qrope(k_cur, offset=cache_position)
            k_buf, v_buf = past_kv
            pos = cache_position.reshape(-1).to(device=k_buf.device, dtype=torch.long)
            k_buf.index_copy_(2, pos, k_cur)
            v_buf.index_copy_(2, pos, v_cur)
            resolved_backend, resolved_fn = resolve_attention_backend(self.attention_backend, self.attention_fn)
            attn_out = dispatch_attention(
                resolved_backend,
                resolved_fn,
                q,
                k_buf,
                v_buf,
                attn_mask=attn_mask,
                is_causal=False,
                scale=self.scale,
            )
            attn_out = attn_out.permute(0, 2, 1, 3).contiguous().reshape(batch, seq_len, self.inner_dim)
            return self.out_proj(attn_out), (k_buf, v_buf)

        if self.cache_mode == "kv":
            # Project and rotate only new tokens, then append to cached rotated (K, V) for lower decode latency.
            past_len = int(past_kv[0].shape[2]) if has_past else 0
            num_all = past_len + seq_len
            q_offset = past_len
            k_up, v_up = torch.split(self.kv_up_proj(c_kv_cur), [self.inner_dim, self.inner_dim], dim=-1)
            k_cur = self.k_norm(k_up.reshape(batch, seq_len, self.num_heads, self.head_dim)).transpose(1, 2)
            v_cur = v_up.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            q = self.qrope(q, offset=q_offset)
            k_cur = self.qrope(k_cur, offset=q_offset)
            if has_past:
                k = torch.cat((past_kv[0], k_cur), dim=2)
                v = torch.cat((past_kv[1], v_cur), dim=2)
            else:
                k, v = k_cur, v_cur
            new_cache: Any = (k, v)
        else:
            c_kv_all = torch.cat((past_kv, c_kv_cur), dim=1) if has_past else c_kv_cur
            num_all = c_kv_all.shape[1]
            k_up, v_up = torch.split(self.kv_up_proj(c_kv_all), [self.inner_dim, self.inner_dim], dim=-1)
            k = self.k_norm(k_up.reshape(batch, num_all, self.num_heads, self.head_dim)).transpose(1, 2)
            v = v_up.reshape(batch, num_all, self.num_heads, self.head_dim).transpose(1, 2)
            q_offset = num_all - seq_len
            q = self.qrope(q, offset=q_offset)
            k = self.qrope(k, offset=0)
            new_cache = c_kv_all

        resolved_backend, resolved_fn = resolve_attention_backend(self.attention_backend, self.attention_fn)
        apply_causal = not (has_past and seq_len == 1)
        if (
            resolved_backend == "ra"
            and resolved_fn is None
            and (self.distance_bias is not None or self.hub_size > 0)
            and can_use_triton_ra_fused_bias(q, k, v, self.distance_bias, hub_size=self.hub_size)
        ):
            attn_out = ra_attention_fused_bias(
                q,
                k,
                v,
                distance_bias=self.distance_bias,
                hub_size=self.hub_size,
                q_offset=q_offset,
                check_hub_q=bool(self.hub_size > 0 and is_dense_with_hub and not has_past),
                is_causal=apply_causal,
                scale=self.scale,
            )
        elif resolved_backend == "flex" and resolved_fn is None:
            attn_out = dispatch_attention(
                "flex",
                None,
                q,
                k,
                v,
                attn_mask=None,
                is_causal=False,
                scale=self.scale,
                distance_bias=self.distance_bias,
                hub_size=self.hub_size,
                q_offset=q_offset,
                is_dense_with_hub=is_dense_with_hub,
                has_past_kv=has_past,
                apply_causal=apply_causal,
            )
        else:
            use_causal_flag = False
            full_mask: Optional[Any] = None

            if self.distance_bias is None and self.hub_size == 0 and not has_past:
                # Keep attn_mask=None so FlashAttention and cuDNN fast paths remain eligible.
                use_causal_flag = True
            elif self.distance_bias is None and has_past and seq_len == 1:
                # Single-token decode attends to all cached prefix keys without future masking.
                use_causal_flag = False
            elif self.distance_bias is not None and has_past and seq_len == 1:
                full_mask = self.distance_bias(
                    seq_len=seq_len,
                    num_all=num_all,
                    device=x.device,
                    dtype=q.dtype,
                    is_dense_with_hub=is_dense_with_hub,
                    has_past_kv=True,
                )
            else:
                can_cache_mask = not _is_torch_compiling() and (
                    self.distance_bias is None or not self.distance_bias.learnable
                )
                mask_key: Optional[tuple[Any, ...]] = None
                if can_cache_mask:
                    b_ver = (
                        (int(self.distance_bias.raw_slopes._version), int(self.distance_bias.hub_penalty._version))
                        if self.distance_bias is not None
                        else (0, 0)
                    )
                    mask_key = (int(seq_len), int(num_all), str(x.device), q.dtype, bool(is_dense_with_hub), bool(has_past), b_ver)
                    full_mask = self._mask_cache.get(mask_key)
                if full_mask is None:
                    causal = self._build_causal_mask(seq_len, num_all, x.device, q.dtype, is_dense_with_hub, has_past)
                    if self.distance_bias is not None:
                        decay = self.distance_bias(
                            seq_len=seq_len,
                            num_all=num_all,
                            device=x.device,
                            dtype=q.dtype,
                            is_dense_with_hub=is_dense_with_hub,
                            has_past_kv=has_past,
                        )
                        full_mask = decay + causal
                    else:
                        full_mask = causal
                    if can_cache_mask and mask_key is not None:
                        if len(self._mask_cache) >= 16:
                            self._mask_cache.clear()
                        self._mask_cache[mask_key] = full_mask

            attn_out = dispatch_attention(
                resolved_backend,
                resolved_fn,
                q,
                k,
                v,
                attn_mask=full_mask,
                is_causal=use_causal_flag,
                scale=self.scale,
            )

        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().reshape(batch, seq_len, self.inner_dim)
        return self.out_proj(attn_out), new_cache

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"kv_latent_dim={self.kv_latent_dim}, hub_size={self.hub_size}, "
            f"backend={self.attention_backend!r}, cache_mode={self.cache_mode!r}"
        )


class SlotCrossAttention(BaseModule):
    """Multi-head slot cross-attention with orthogonally initialized per-slot queries and FiLM conditioning.

    Modulates those slot query vectors using FiLM scaling and shifting from
    `terminal_base` before attending to `context_states`, to reduce
    intra-terminal slot collapse.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_terminals: int = 16,
        slots_per_terminal: int = 4,
        attention_backend: AttentionBackend = "auto",
        attention_fn: Optional[Callable[..., Any]] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not _HAS_TORCH:
            raise ImportError("SlotCrossAttention requires PyTorch.")
        if dim <= 0 or num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be positive and divisible by num_heads ({num_heads}).")
        if num_terminals <= 0 or slots_per_terminal <= 0:
            raise ValueError("num_terminals and slots_per_terminal must be > 0.")
        validated_backend = validate_attention_backend(attention_backend)

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.num_terminals = int(num_terminals)
        self.slots_per_terminal = int(slots_per_terminal)
        self.total_slots = self.num_terminals * self.slots_per_terminal
        self.attention_backend: AttentionBackend = validated_backend
        self.attention_fn = attention_fn
        self.scale = 1.0 / math.sqrt(self.head_dim)

        raw_queries = torch.empty(self.num_terminals, self.slots_per_terminal, self.dim)
        for t in range(self.num_terminals):
            nn.init.orthogonal_(raw_queries[t])
        self.slot_queries = nn.Parameter(raw_queries.unsqueeze(0) * math.sqrt(self.dim))
        self.slot_film = nn.Linear(self.dim, 2 * self.slots_per_terminal * self.dim, bias=False)
        nn.init.normal_(self.slot_film.weight, mean=0.0, std=0.02)

        self.q_norm = RaNorm(self.dim, eps=eps, elementwise_affine=True, bias=False)
        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.norm = RaNorm(self.dim, eps=eps, elementwise_affine=True, bias=False)

    def forward(self, terminal_base: Any, context_states: Any) -> Any:
        batch, ctx_len, _ = context_states.shape
        if terminal_base.shape != (batch, self.num_terminals, self.dim):
            raise ValueError(
                f"terminal_base must have shape ({batch}, {self.num_terminals}, {self.dim}), "
                f"got {tuple(terminal_base.shape)}."
            )

        film = self.slot_film(terminal_base).view(batch, self.num_terminals, self.slots_per_terminal, 2, self.dim)
        gamma, beta = film[..., 0, :], film[..., 1, :]
        q_slots = self.q_norm(self.slot_queries * (1.0 + gamma) + beta).view(batch, self.total_slots, self.dim)

        q = self.q_proj(q_slots).view(batch, self.total_slots, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context_states).view(batch, ctx_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context_states).view(batch, ctx_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = dispatch_attention(
            self.attention_backend,
            self.attention_fn,
            q,
            k,
            v,
            attn_mask=None,
            is_causal=False,
            scale=self.scale,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, self.total_slots, self.dim)
        return self.norm(self.out_proj(attn_out))

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, num_terminals={self.num_terminals}, "
            f"slots_per_terminal={self.slots_per_terminal}, backend={self.attention_backend!r}"
        )


__all__ = ["LatentKVAttention", "SlotCrossAttention"]
