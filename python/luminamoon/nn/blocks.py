"""Reversible split-state and residual rollout blocks for luminamoon.nn."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any, Callable, Literal, Optional, Sequence, Tuple

from ._attention_dispatch import AttentionBackend
from ._compat import BaseModule, _HAS_TORCH, nn, torch
from ._latent_attention import LatentKVAttention
from .activation import RaSwiGLU, ra_tanh
from .norm import RaNorm

SelectiveReversibleType = Literal[
    "none",
    "all",
    "every",
    "even",
    "odd",
    "first_half",
    "second_half",
    "indices",
]
_VALID_SELECTIVE_TYPES: Tuple[str, ...] = (
    "none",
    "all",
    "every",
    "even",
    "odd",
    "first_half",
    "second_half",
    "indices",
)


def resolve_selective_reversible(
    layer_idx: Optional[int] = None,
    num_layers: Optional[int] = None,
    *,
    reversible: bool = False,
    selective_type: SelectiveReversibleType = "none",
    every: int = 2,
    selective_indices: Optional[Iterable[int]] = None,
) -> bool:
    """Resolve whether a layer should enable reversible recomputation under a selective policy.

    Supported `selective_type` modes:
    - `"none"`: use the explicit `reversible` flag as-is.
    - `"all"`: enable reversibility on every layer.
    - `"every"`: enable reversibility when `layer_idx % every == 0`.
    - `"even"` / `"odd"`: enable reversibility on even or odd `layer_idx`.
    - `"first_half"` / `"second_half"`: enable reversibility on the first or second half of `num_layers`.
    - `"indices"`: enable reversibility when `layer_idx` is in `selective_indices`.
    """
    if selective_type not in _VALID_SELECTIVE_TYPES:
        valid = ", ".join(repr(m) for m in _VALID_SELECTIVE_TYPES)
        raise ValueError(f"selective_type must be one of ({valid}), got {selective_type!r}.")
    if selective_type == "none":
        return bool(reversible)
    if selective_type == "all":
        return True
    if layer_idx is None or int(layer_idx) < 0:
        raise ValueError(f"layer_idx must be a non-negative integer when selective_type={selective_type!r}.")
    idx = int(layer_idx)
    if selective_type == "every":
        step = int(every)
        if step <= 0:
            raise ValueError(f"every must be > 0, got {every}.")
        return (idx % step) == 0
    if selective_type == "even":
        return (idx % 2) == 0
    if selective_type == "odd":
        return (idx % 2) == 1
    if selective_type in ("first_half", "second_half"):
        if num_layers is None or int(num_layers) <= 0:
            raise ValueError(f"num_layers must be > 0 when selective_type={selective_type!r}.")
        mid = int(num_layers) // 2
        return idx < mid if selective_type == "first_half" else idx >= mid
    if selective_indices is None:
        return False
    return idx in {int(i) for i in selective_indices}


def _apply_inplace_compile(module: Any, compile_kwargs: Optional[dict[str, Any]]) -> None:
    if hasattr(module, "compile"):
        module.compile(**(compile_kwargs or {}))


def _call_kick_attn(
    attn_mod: Any,
    x: Any,
    past_kv: Optional[Any],
    is_dense_with_hub: bool,
) -> Tuple[Any, Optional[Any]]:
    if isinstance(attn_mod, LatentKVAttention):
        return attn_mod(x, past_kv=past_kv, is_dense_with_hub=is_dense_with_hub)
    res = attn_mod(x) if past_kv is None else attn_mod(x, past_kv=past_kv)
    if isinstance(res, tuple) and len(res) == 2:
        return res[0], res[1]
    return res, None


if _HAS_TORCH:

    class _ReversibleDKDFunction(torch.autograd.Function):
        """Autograd function for reversible DKD that reconstructs intermediate states analytically during backward."""

        @staticmethod
        def forward(
            ctx: Any,
            q: Any,
            p: Any,
            block: Any,
            is_dense_with_hub: bool,
            *params: Any,
        ) -> Tuple[Any, Any]:
            with torch.no_grad():
                half_dt = 0.5 * block.dt
                q_half = q + half_dt * block.drift_ffn(block.norm_p(p))
                attn_out, _ = _call_kick_attn(block.kick_attn, block.norm_q(q_half), None, is_dense_with_hub)
                p_next = p + block.dt * attn_out
                q_next = q_half + half_dt * block.drift_ffn(block.norm_p(p_next))
            ctx.block = block
            ctx.is_dense_with_hub = is_dense_with_hub
            ctx.save_for_backward(q_next.detach(), p_next.detach())
            return q_next, p_next

        @staticmethod
        def backward(ctx: Any, dq_next: Any, dp_next: Any) -> Tuple[Any, ...]:  # type: ignore[override]
            block = ctx.block
            is_dense_with_hub = ctx.is_dense_with_hub
            q_next, p_next = ctx.saved_tensors
            half_dt = 0.5 * block.dt

            f_params = tuple(p for p in (*block.norm_p.parameters(), *block.drift_ffn.parameters()) if p.requires_grad)
            a_params = tuple(p for p in (*block.norm_q.parameters(), *block.kick_attn.parameters()) if p.requires_grad)
            all_params = tuple(block.parameters())
            # Populate grad_map on demand to skip per-layer zeros_like allocations.
            grad_map: dict[int, Any] = {}

            # Reverse step 3: q_next = q_half + 0.5 * dt * F(p_next)
            p_next_req = p_next.detach().requires_grad_(True)
            with torch.enable_grad():
                f2 = half_dt * block.drift_ffn(block.norm_p(p_next_req))
            grads_f2 = torch.autograd.grad(f2, (p_next_req,) + f_params, dq_next, allow_unused=True)
            dp_next_total = dp_next + grads_f2[0]
            for pr, g in zip(f_params, grads_f2[1:]):
                if g is not None:
                    grad_map[id(pr)] = g

            with torch.no_grad():
                q_half = q_next - f2.detach()
            del f2, grads_f2, p_next_req

            # Reverse step 2: p_next = p_in + dt * A(q_half)
            q_half_req = q_half.detach().requires_grad_(True)
            with torch.enable_grad():
                a_out, _ = _call_kick_attn(block.kick_attn, block.norm_q(q_half_req), None, is_dense_with_hub)
                a_scaled = block.dt * a_out
            grads_a = torch.autograd.grad(a_scaled, (q_half_req,) + a_params, dp_next_total, allow_unused=True)
            dq_half_total = dq_next + grads_a[0]
            for pr, g in zip(a_params, grads_a[1:]):
                if g is not None:
                    prev = grad_map.get(id(pr))
                    grad_map[id(pr)] = g if prev is None else (prev + g)

            with torch.no_grad():
                p_in = p_next - a_scaled.detach()
            del a_out, a_scaled, grads_a, q_half_req, q_half

            # Reverse step 1: q_half = q_in + 0.5 * dt * F(p_in)
            p_in_req = p_in.detach().requires_grad_(True)
            with torch.enable_grad():
                f1 = half_dt * block.drift_ffn(block.norm_p(p_in_req))
            grads_f1 = torch.autograd.grad(f1, (p_in_req,) + f_params, dq_half_total, allow_unused=True)
            dp_in = dp_next_total + grads_f1[0]
            for pr, g in zip(f_params, grads_f1[1:]):
                if g is not None:
                    prev = grad_map.get(id(pr))
                    grad_map[id(pr)] = g if prev is None else (prev + g)
            del f1, grads_f1, p_in_req, p_in

            param_grads = tuple(grad_map.get(id(pr)) for pr in all_params)
            return (dq_half_total, dp_in, None, None) + param_grads

else:
    _ReversibleDKDFunction = None  # type: ignore[assignment]


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

        if self.compile_block:
            # Compile kick_attn and drift_ffn on reversible layers and the full block on non-reversible layers
            # using in-place Module.compile so state_dict keys stay unchanged.
            if self.reversible:
                _apply_inplace_compile(self.kick_attn, compile_kwargs)
                _apply_inplace_compile(self.drift_ffn, compile_kwargs)
            else:
                _apply_inplace_compile(self, compile_kwargs)

    def forward(
        self,
        q: Any,
        p: Any,
        past_kv: Optional[Any] = None,
        is_dense_with_hub: bool = True,
    ) -> Tuple[Any, Any, Optional[Any]]:
        if (
            self.reversible
            and self.training
            and torch.is_grad_enabled()
            and past_kv is None
            and _ReversibleDKDFunction is not None
        ):
            q_next, p_next = _ReversibleDKDFunction.apply(
                q,
                p,
                self,
                bool(is_dense_with_hub),
                *tuple(self.parameters()),
            )
            return q_next, p_next, None

        half_dt = 0.5 * self.dt
        q_half = q + half_dt * self.drift_ffn(self.norm_p(p))
        attn_out, new_kv = _call_kick_attn(self.kick_attn, self.norm_q(q_half), past_kv, is_dense_with_hub)
        p_next = p + self.dt * attn_out
        q_next = q_half + half_dt * self.drift_ffn(self.norm_p(p_next))
        return q_next, p_next, new_kv

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
                compile_block=compile_blocks,
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

    def forward(
        self,
        q: Any,
        p: Any,
        past_kvs: Optional[Sequence[Optional[Any]]] = None,
        is_dense_with_hub: bool = True,
        return_past_kvs: bool = False,
    ) -> Tuple[Any, Any, Optional[list[Optional[Any]]]]:
        if past_kvs is not None and len(past_kvs) != self.num_layers:
            raise ValueError(f"past_kvs length ({len(past_kvs)}) must equal num_layers ({self.num_layers}).")
        new_kvs: Optional[list[Optional[Any]]] = [] if (return_past_kvs or past_kvs is not None) else None
        for idx, layer in enumerate(self.layers):
            layer_past = past_kvs[idx] if past_kvs is not None else None
            q, p, kv = layer(q, p, past_kv=layer_past, is_dense_with_hub=is_dense_with_hub)
            if new_kvs is not None:
                new_kvs.append(kv)
        return q, p, new_kvs

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
