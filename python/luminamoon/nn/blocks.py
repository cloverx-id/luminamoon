"""Reversible split-state and residual rollout blocks for luminamoon.nn."""

from __future__ import annotations

import math
from typing import Any, Literal, Optional, Tuple

from ._compat import BaseModule, _HAS_TORCH, nn, torch
from ._latent_attention import LatentKVAttention
from .activation import RaSwiGLU, ra_tanh
from .norm import RaNorm


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
                attn_out, _ = block.kick_attn(
                    block.norm_q(q_half),
                    past_kv=None,
                    is_dense_with_hub=is_dense_with_hub,
                )
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

            f_params = tuple(list(block.norm_p.parameters()) + list(block.drift_ffn.parameters()))
            a_params = tuple(list(block.norm_q.parameters()) + list(block.kick_attn.parameters()))
            all_params = tuple(block.parameters())
            grad_map: dict[int, Any] = {id(pr): torch.zeros_like(pr) for pr in all_params}

            # Reverse step 3: q_next = q_half + 0.5 * dt * F(p_next)
            p_next_req = p_next.detach().requires_grad_(True)
            with torch.enable_grad():
                f2 = half_dt * block.drift_ffn(block.norm_p(p_next_req))
            grads_f2 = torch.autograd.grad(f2, (p_next_req,) + f_params, dq_next, allow_unused=True)
            dp_next_total = dp_next + grads_f2[0]
            for pr, g in zip(f_params, grads_f2[1:]):
                if g is not None:
                    grad_map[id(pr)].add_(g)

            with torch.no_grad():
                q_half = q_next - f2.detach()

            # Reverse step 2: p_next = p_in + dt * A(q_half)
            q_half_req = q_half.detach().requires_grad_(True)
            with torch.enable_grad():
                a_out, _ = block.kick_attn(
                    block.norm_q(q_half_req),
                    past_kv=None,
                    is_dense_with_hub=is_dense_with_hub,
                )
                a_scaled = block.dt * a_out
            grads_a = torch.autograd.grad(a_scaled, (q_half_req,) + a_params, dp_next_total, allow_unused=True)
            dq_half_total = dq_next + grads_a[0]
            for pr, g in zip(a_params, grads_a[1:]):
                if g is not None:
                    grad_map[id(pr)].add_(g)

            with torch.no_grad():
                p_in = p_next - a_scaled.detach()

            # Reverse step 1: q_half = q_in + 0.5 * dt * F(p_in)
            p_in_req = p_in.detach().requires_grad_(True)
            with torch.enable_grad():
                f1 = half_dt * block.drift_ffn(block.norm_p(p_in_req))
            grads_f1 = torch.autograd.grad(f1, (p_in_req,) + f_params, dq_half_total, allow_unused=True)
            dp_in = dp_next_total + grads_f1[0]
            for pr, g in zip(f_params, grads_f1[1:]):
                if g is not None:
                    grad_map[id(pr)].add_(g)

            param_grads = tuple(grad_map[id(pr)] for pr in all_params)
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
        bias_mode: Literal["linear", "rational", "none"] = "linear",
        attention_backend: Literal["ra", "sdpa"] = "ra",
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
        self.reversible = bool(reversible)

        resolved_full_head_dim = int(head_dim) if head_dim is not None else (self.dim // num_heads)
        if resolved_full_head_dim % 8 != 0:
            raise ValueError(
                f"ReversibleDKDBlock halves head_dim for QRoPE; head_dim must be divisible by 8, "
                f"got {resolved_full_head_dim}."
            )
        half_head_dim = resolved_full_head_dim // 2
        half_kv_dim = max(16, (int(kv_latent_dim) if kv_latent_dim is not None else max(32, self.dim // 4)) // 2)

        self.norm_q = RaNorm(self.half_dim, eps=eps, elementwise_affine=True, bias=False)
        self.norm_p = RaNorm(self.half_dim, eps=eps, elementwise_affine=True, bias=False)
        self.kick_attn = LatentKVAttention(
            dim=self.half_dim,
            num_heads=num_heads,
            head_dim=half_head_dim,
            kv_latent_dim=half_kv_dim,
            hub_size=hub_size,
            num_hub_heads=num_hub_heads,
            bias_mode=bias_mode,
            attention_backend=attention_backend,
            eps=eps,
        )

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
                nn.init.normal_(self.kick_attn.out_proj.weight, mean=0.0, std=std_res)
                nn.init.normal_(self.drift_ffn.w_down.weight, mean=0.0, std=std_res)

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
        attn_out, new_kv = self.kick_attn(
            self.norm_q(q_half),
            past_kv=past_kv,
            is_dense_with_hub=is_dense_with_hub,
        )
        p_next = p + self.dt * attn_out
        q_next = q_half + half_dt * self.drift_ffn(self.norm_p(p_next))
        return q_next, p_next, new_kv

    def extra_repr(self) -> str:
        return f"dim={self.dim}, half_dim={self.half_dim}, dt={self.dt}, reversible={self.reversible}"


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


__all__ = ["ReversibleDKDBlock", "ResidualRollout"]
