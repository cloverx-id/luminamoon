"""Reversible split-state and residual rollout blocks for luminamoon.nn."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any, Callable, Literal, Optional, Sequence, Tuple

from ._attention_dispatch import AttentionBackend
from ._compat import BaseModule, _HAS_TORCH, nn, torch
from ._latent_attention import LatentKVAttention
from ._reversible_ops import (
    SelectiveReversibleType,
    _ReversibleDKDFunction,
    _apply_inplace_compile,
    _call_kick_attn,
    _clone_cache_value,
    _compile_callable,
    _is_cudagraph_compile_mode,
    _mark_cudagraph_step_begin,
    resolve_selective_reversible,
)
from .activation import RaSwiGLU, ra_tanh
from .norm import RaNorm


class ReversibleDKDBlock(BaseModule):
    """Volume-preserving reversible Drift-Kick-Drift (Strang-split) block.

    Operates on split `(q, p)` hidden-state halves of width `dim // 2` (total width `dim`),
    alternating two half-step FFN drifts on `q` with a full-step attention kick
    on `p`. Because every sub-update is a shear map with unit Jacobian determinant,
    the block is analytically invertible and supports optional `reversible=True`
    recomputation during dense training to reduce activation memory.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        kv_latent_dim: Optional[int] = None,
        hub_size: int = 0,
        num_hub_heads: Optional[int] = None,
        dt: float = 0.5,
        num_layers: Optional[int] = None,
        reversible: bool = False,
        layer_idx: Optional[int] = None,
        selective_type: SelectiveReversibleType = "none",
        every: int = 2,
        selective_indices: Optional[Iterable[int]] = None,
        compile_block: bool = False,
        compile_kwargs: Optional[dict[str, Any]] = None,
        bias_mode: Literal["linear", "rational", "none"] = "linear",
        attention_backend: AttentionBackend = "auto",
        attention_fn: Optional[Callable[..., Any]] = None,
        cache_mode: Literal["latent", "kv"] = "latent",
        kick_attn: Optional[Any] = None,
        drift_ffn: Optional[Any] = None,
        norm_cls: Optional[Callable[[int], Any]] = None,
        ffn_multiple_of: int = 16,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not _HAS_TORCH:
            raise ImportError("ReversibleDKDBlock requires PyTorch.")
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"ReversibleDKDBlock requires a positive even dim, got {dim}.")
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError(f"dt must be finite and > 0, got {dt}.")
        if kick_attn is None and int(num_heads) <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}.")
        if drift_ffn is None and int(ffn_multiple_of) <= 0:
            raise ValueError(f"ffn_multiple_of must be > 0, got {ffn_multiple_of}.")

        self.dim = int(dim)
        self.half_dim = self.dim // 2
        self.dt = float(dt)
        self.layer_idx = int(layer_idx) if layer_idx is not None else None
        self.selective_type: SelectiveReversibleType = selective_type
        self.every = int(every)
        self.reversible = resolve_selective_reversible(
            layer_idx=layer_idx,
            num_layers=num_layers,
            reversible=reversible,
            selective_type=selective_type,
            every=every,
            selective_indices=selective_indices,
        )
        self.compile_block = bool(compile_block)

        if norm_cls is not None:
            self.norm_q = norm_cls(self.half_dim)
            self.norm_p = norm_cls(self.half_dim)
        else:
            self.norm_q = RaNorm(self.half_dim, eps=eps, elementwise_affine=True, bias=False)
            self.norm_p = RaNorm(self.half_dim, eps=eps, elementwise_affine=True, bias=False)

        if kick_attn is not None:
            self.kick_attn = kick_attn
        else:
            resolved_full_head_dim = int(head_dim) if head_dim is not None else (self.dim // num_heads)
            if resolved_full_head_dim % 8 != 0:
                raise ValueError(
                    f"ReversibleDKDBlock halves head_dim for QRoPE; head_dim must be divisible by 8, "
                    f"got {resolved_full_head_dim}."
                )
            half_head_dim = resolved_full_head_dim // 2
            half_kv_dim = max(16, (int(kv_latent_dim) if kv_latent_dim is not None else max(32, self.dim // 4)) // 2)
            self.kick_attn = LatentKVAttention(
                dim=self.half_dim,
                num_heads=num_heads,
                head_dim=half_head_dim,
                kv_latent_dim=half_kv_dim,
                hub_size=hub_size,
                num_hub_heads=num_hub_heads,
                bias_mode=bias_mode,
                attention_backend=attention_backend,
                attention_fn=attention_fn,
                cache_mode=cache_mode,
                eps=eps,
            )

        if drift_ffn is not None:
            self.drift_ffn = drift_ffn
        else:
            hidden_dim = ffn_multiple_of * ((int(2 * (self.half_dim * 4) / 3) + ffn_multiple_of - 1) // ffn_multiple_of)
            self.drift_ffn = RaSwiGLU(
                in_features=self.half_dim,
                hidden_features=hidden_dim,
                out_features=self.half_dim,
                bias=False,
                b=1.0,
            )

        if num_layers is not None and num_layers > 0:
            # Depth-scaled residual output init limits activation and gradient growth across deep stacks.
            std_res = 0.02 / math.sqrt(2.0 * float(num_layers))
            with torch.no_grad():
                if kick_attn is None and hasattr(self.kick_attn, "out_proj"):
                    nn.init.normal_(self.kick_attn.out_proj.weight, mean=0.0, std=std_res)
                if drift_ffn is None and hasattr(self.drift_ffn, "w_down"):
                    nn.init.normal_(self.drift_ffn.w_down.weight, mean=0.0, std=std_res)

        self._use_cudagraph_guard = bool(self.compile_block and _is_cudagraph_compile_mode(compile_kwargs))
        self._compiled_forward_fn: Optional[Callable[..., Any]] = None
        if self.compile_block:
            # Compile kick_attn/drift_ffn on reversible layers, _forward_eager when CUDA Graph guards
            # are active, or the block in-place otherwise so state_dict keys stay unchanged.
            if self.reversible:
                _apply_inplace_compile(self.kick_attn, compile_kwargs)
                _apply_inplace_compile(self.drift_ffn, compile_kwargs)
            elif self._use_cudagraph_guard:
                self._compiled_forward_fn = _compile_callable(self._forward_eager, compile_kwargs)
            else:
                _apply_inplace_compile(self, compile_kwargs)

    def _run_drift_ffn(self, x: Any) -> Any:
        if self._use_cudagraph_guard and self.reversible:
            _mark_cudagraph_step_begin()
            out = self.drift_ffn(x)
            return out.clone() if isinstance(out, torch.Tensor) else out
        return self.drift_ffn(x)

    def _run_kick_attn(
        self,
        x: Any,
        past_kv: Optional[Any],
        is_dense_with_hub: bool,
        cache_position: Optional[Any] = None,
        attn_mask: Optional[Any] = None,
    ) -> Tuple[Any, Optional[Any]]:
        if self._use_cudagraph_guard and self.reversible:
            _mark_cudagraph_step_begin()
            out, new_kv = _call_kick_attn(self.kick_attn, x, past_kv, is_dense_with_hub, cache_position, attn_mask)
            kv_out = new_kv if cache_position is not None else _clone_cache_value(new_kv)
            return (out.clone() if isinstance(out, torch.Tensor) else out), kv_out
        return _call_kick_attn(self.kick_attn, x, past_kv, is_dense_with_hub, cache_position, attn_mask)

    def _forward_eager(
        self,
        q: Any,
        p: Any,
        past_kv: Optional[Any] = None,
        is_dense_with_hub: bool = True,
        cache_position: Optional[Any] = None,
        attn_mask: Optional[Any] = None,
    ) -> Tuple[Any, Any, Optional[Any]]:
        half_dt = 0.5 * self.dt
        q_half = q + half_dt * self._run_drift_ffn(self.norm_p(p))
        attn_out, new_kv = self._run_kick_attn(self.norm_q(q_half), past_kv, is_dense_with_hub, cache_position, attn_mask)
        p_next = p + self.dt * attn_out
        q_next = q_half + half_dt * self._run_drift_ffn(self.norm_p(p_next))
        return q_next, p_next, new_kv

    def forward(
        self,
        q: Any,
        p: Any,
        past_kv: Optional[Any] = None,
        is_dense_with_hub: bool = True,
        return_past_kv: bool = False,
        cache_position: Optional[Any] = None,
        attn_mask: Optional[Any] = None,
    ) -> Tuple[Any, Any, Optional[Any]]:
        if (
            self.reversible
            and self.training
            and torch.is_grad_enabled()
            and past_kv is None
            and not return_past_kv
            and cache_position is None
            and attn_mask is None
            and _ReversibleDKDFunction is not None
        ):
            q_next, p_next = _ReversibleDKDFunction.apply(q, p, self, bool(is_dense_with_hub), *tuple(self.parameters()))
            return q_next, p_next, None

        if self._compiled_forward_fn is not None:
            _mark_cudagraph_step_begin()
            q_next, p_next, new_kv = self._compiled_forward_fn(q, p, past_kv, is_dense_with_hub, cache_position, attn_mask)
            kv_out = new_kv if cache_position is not None else _clone_cache_value(new_kv)
            return q_next.clone(), p_next.clone(), kv_out

        return self._forward_eager(q, p, past_kv, is_dense_with_hub, cache_position, attn_mask)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, half_dim={self.half_dim}, dt={self.dt}, "
            f"reversible={self.reversible}, selective_type={self.selective_type!r}, compile_block={self.compile_block}"
        )


class ReversibleDKDStack(BaseModule):
    """Multi-layer stack of `ReversibleDKDBlock` modules with selective reversibility and optional compilation."""

    def __init__(
        self,
        num_layers: int,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        kv_latent_dim: Optional[int] = None,
        hub_size: int = 0,
        num_hub_heads: Optional[int] = None,
        dt: float = 0.5,
        reversible: bool = False,
        selective_type: SelectiveReversibleType = "none",
        every: int = 2,
        selective_indices: Optional[Iterable[int]] = None,
        compile_blocks: bool = False,
        compile_kwargs: Optional[dict[str, Any]] = None,
        bias_mode: Literal["linear", "rational", "none"] = "linear",
        attention_backend: AttentionBackend = "auto",
        attention_fn: Optional[Callable[..., Any]] = None,
        cache_mode: Literal["latent", "kv"] = "latent",
        kick_attn_fn: Optional[Callable[[int], Any]] = None,
        drift_ffn_fn: Optional[Callable[[int], Any]] = None,
        norm_cls: Optional[Callable[[int], Any]] = None,
        ffn_multiple_of: int = 16,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not _HAS_TORCH:
            raise ImportError("ReversibleDKDStack requires PyTorch.")
        if int(num_layers) <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}.")

        self.num_layers = int(num_layers)
        self.selective_type: SelectiveReversibleType = selective_type
        self.every = int(every)
        self.compile_blocks = bool(compile_blocks)
        idx_set = tuple(int(i) for i in selective_indices) if selective_indices is not None else None
        any_reversible = any(
            resolve_selective_reversible(
                layer_idx=idx,
                num_layers=self.num_layers,
                reversible=reversible,
                selective_type=selective_type,
                every=every,
                selective_indices=idx_set,
            )
            for idx in range(self.num_layers)
        )
        # Compile the full stack as one graph when no layer uses custom reversible autograd,
        # avoiding per-layer dispatch overhead and Dynamo recompile_limit collisions across mixed FFN types.
        self.whole_stack_compiled = bool(self.compile_blocks and not any_reversible)
        self._use_cudagraph_guard = bool(self.whole_stack_compiled and _is_cudagraph_compile_mode(compile_kwargs))

        layers: list[ReversibleDKDBlock] = []
        rev_indices: list[int] = []
        for idx in range(self.num_layers):
            blk = ReversibleDKDBlock(
                dim=dim,
                num_heads=num_heads,
                head_dim=head_dim,
                kv_latent_dim=kv_latent_dim,
                hub_size=hub_size,
                num_hub_heads=num_hub_heads,
                dt=dt,
                num_layers=self.num_layers,
                reversible=reversible,
                layer_idx=idx,
                selective_type=selective_type,
                every=every,
                selective_indices=idx_set,
                compile_block=bool(self.compile_blocks and not self.whole_stack_compiled),
                compile_kwargs=compile_kwargs,
                bias_mode=bias_mode,
                attention_backend=attention_backend,
                attention_fn=attention_fn,
                cache_mode=cache_mode,
                kick_attn=kick_attn_fn(idx) if kick_attn_fn is not None else None,
                drift_ffn=drift_ffn_fn(idx) if drift_ffn_fn is not None else None,
                norm_cls=norm_cls,
                ffn_multiple_of=ffn_multiple_of,
                eps=eps,
            )
            if blk.reversible:
                rev_indices.append(idx)
            layers.append(blk)

        self.layers = nn.ModuleList(layers)
        self.reversible_layer_indices: Tuple[int, ...] = tuple(rev_indices)
        self._compiled_stack_fn: Optional[Callable[..., Any]] = (
            _compile_callable(self._forward_layers, compile_kwargs) if self.whole_stack_compiled else None
        )

    def _forward_layers(
        self,
        q: Any,
        p: Any,
        past_kvs: Optional[Sequence[Optional[Any]]],
        is_dense_with_hub: bool,
        want_kvs: bool,
        cache_position: Optional[Any] = None,
        attn_mask: Optional[Any] = None,
    ) -> Tuple[Any, Any, Optional[list[Optional[Any]]]]:
        new_kvs: Optional[list[Optional[Any]]] = [] if want_kvs else None
        for idx, layer in enumerate(self.layers):
            layer_past = past_kvs[idx] if past_kvs is not None else None
            q, p, kv = layer(
                q,
                p,
                past_kv=layer_past,
                is_dense_with_hub=is_dense_with_hub,
                return_past_kv=want_kvs,
                cache_position=cache_position,
                attn_mask=attn_mask,
            )
            if new_kvs is not None:
                new_kvs.append(kv)
        return q, p, new_kvs

    def forward(
        self,
        q: Any,
        p: Any,
        past_kvs: Optional[Sequence[Optional[Any]]] = None,
        is_dense_with_hub: bool = True,
        return_past_kvs: bool = False,
        cache_position: Optional[Any] = None,
        attn_mask: Optional[Any] = None,
    ) -> Tuple[Any, Any, Optional[list[Optional[Any]]]]:
        if past_kvs is not None and len(past_kvs) != self.num_layers:
            raise ValueError(f"past_kvs length ({len(past_kvs)}) must equal num_layers ({self.num_layers}).")
        want_kvs = bool(return_past_kvs or past_kvs is not None)
        if self._compiled_stack_fn is not None:
            if self._use_cudagraph_guard:
                _mark_cudagraph_step_begin()
            q, p, new_kvs = self._compiled_stack_fn(q, p, past_kvs, is_dense_with_hub, want_kvs, cache_position, attn_mask)
            if self._use_cudagraph_guard:
                if cache_position is not None:
                    return q.clone(), p.clone(), new_kvs
                cloned_kvs = [_clone_cache_value(kv) for kv in new_kvs] if new_kvs is not None else None
                return q.clone(), p.clone(), cloned_kvs
            return q, p, new_kvs
        return self._forward_layers(q, p, past_kvs, is_dense_with_hub, want_kvs, cache_position, attn_mask)

    def extra_repr(self) -> str:
        return (
            f"num_layers={self.num_layers}, selective_type={self.selective_type!r}, "
            f"every={self.every}, reversible_layers={len(self.reversible_layer_indices)}, "
            f"compile_blocks={self.compile_blocks}"
        )


class ResidualRollout(BaseModule):
    """Weight-tied residual recurrence block using rational `ra_tanh` (or `tanh`) and `RaNorm`."""

    def __init__(
        self,
        dim: int,
        num_steps: int = 2,
        activation: Literal["ra_tanh", "tanh"] = "ra_tanh",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not _HAS_TORCH:
            raise ImportError("ResidualRollout requires PyTorch.")
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}.")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {num_steps}.")
        if activation not in ("ra_tanh", "tanh"):
            raise ValueError(f"activation must be 'ra_tanh' or 'tanh', got {activation!r}.")

        self.dim = int(dim)
        self.num_steps = int(num_steps)
        self.activation = activation
        self.w_transition = nn.Parameter(torch.randn(self.dim, self.dim) * (0.02 / math.sqrt(self.dim)))
        self.norm = RaNorm(self.dim, eps=eps, elementwise_affine=True, bias=False)

    def forward(self, x: Any) -> Any:
        h = x
        for _ in range(self.num_steps):
            proj = torch.matmul(h, self.w_transition)
            delta = ra_tanh(proj) if self.activation == "ra_tanh" else torch.tanh(proj)
            h = self.norm(h + delta)
        return h

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_steps={self.num_steps}, activation={self.activation!r}"


__all__ = [
    "SelectiveReversibleType",
    "resolve_selective_reversible",
    "ReversibleDKDBlock",
    "ReversibleDKDStack",
    "ResidualRollout",
]
