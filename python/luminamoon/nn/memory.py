"""Orthogonal hub, directive gating, and differentiable two-tier memory modules for luminamoon.nn."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ._compat import BaseModule, _HAS_TORCH, nn, torch
from .norm import RaNorm


class OrthogonalHub(BaseModule):
    """SO(H) orthogonal prefix hub parameterized via a float32 Cayley transform.

    Computes `Q = (I + S)^{-1}(I - S)` from a skew-symmetric matrix `S` in float32
    so BF16/FP16 models work with `torch.linalg.solve` without low-precision solve error,
    with eval-mode caching keyed on parameter version, device, and dtype.
    """

    def __init__(
        self,
        hub_size: int = 128,
        dim: int = 512,
        use_context_gate: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not _HAS_TORCH:
            raise ImportError("OrthogonalHub requires PyTorch.")
        if hub_size <= 0 or dim <= 0:
            raise ValueError(f"hub_size and dim must be > 0, got hub_size={hub_size}, dim={dim}.")

        self.hub_size = int(hub_size)
        self.dim = int(dim)
        self.use_context_gate = bool(use_context_gate)

        self.weight_skew = nn.Parameter(torch.randn(self.hub_size, self.hub_size) * 0.01)
        self.hub_basis = nn.Parameter(torch.randn(self.hub_size, self.dim) * 0.02)
        self.context_gate: Optional[Any] = nn.Linear(self.dim, self.dim, bias=False) if self.use_context_gate else None
        self.norm = RaNorm(self.dim, eps=eps, elementwise_affine=True, bias=False)

        self._cached_q: Optional[Any] = None
        self._cached_q_version: int = -1

    def train(self, mode: bool = True) -> "OrthogonalHub":
        if mode:
            self._cached_q = None
            self._cached_q_version = -1
        return super().train(mode)

    def get_orthogonal_matrix(self) -> Any:
        if (
            not self.training
            and self._cached_q is not None
            and self._cached_q_version == self.weight_skew._version
            and self._cached_q.device == self.weight_skew.device
            and self._cached_q.dtype == self.hub_basis.dtype
        ):
            return self._cached_q

        # torch.linalg.solve requires float32/float64 on CPU and CUDA.
        w_f32 = self.weight_skew.to(torch.float32)
        skew = 0.5 * (w_f32 - w_f32.T)
        eye = torch.eye(self.hub_size, device=skew.device, dtype=torch.float32)
        q_mat = torch.linalg.solve(eye + skew, eye - skew).to(dtype=self.hub_basis.dtype)

        if not self.training:
            self._cached_q = q_mat.detach()
            self._cached_q_version = self.weight_skew._version
        return q_mat

    def forward(self, batch_size: int, context: Optional[Any] = None) -> Any:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}.")
        q_mat = self.get_orthogonal_matrix()
        basis = torch.matmul(q_mat, self.hub_basis).unsqueeze(0).expand(batch_size, -1, -1)
        if context is not None:
            if self.context_gate is None:
                raise ValueError("context was passed to OrthogonalHub with use_context_gate=False.")
            ctx_summary = self.context_gate(context.mean(dim=1, keepdim=True))
            return self.norm(basis + 0.5 * ctx_summary)
        return self.norm(basis)

    def extra_repr(self) -> str:
        return f"hub_size={self.hub_size}, dim={self.dim}, use_context_gate={self.use_context_gate}"


class DirectiveGate(BaseModule):
    """Pre-norm directive-conditioned gating and FiLM modulation over chunk summaries.

    Applies `RaNorm` before scaling and shifting so cosine-alignment weights and
    channel-wise FiLM parameters are preserved rather than canceled by post-norm
    scale homogeneity.
    """

    def __init__(
        self,
        dim: int,
        init_logit_scale: float = 4.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not _HAS_TORCH:
            raise ImportError("DirectiveGate requires PyTorch.")
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}.")

        self.dim = int(dim)
        self.chunk_norm = RaNorm(self.dim, eps=eps, elementwise_affine=True, bias=False)
        self.directive_norm = RaNorm(self.dim, eps=eps, elementwise_affine=True, bias=False)
        self.directive_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.film = nn.Linear(self.dim, 2 * self.dim, bias=False)
        nn.init.normal_(self.film.weight, mean=0.0, std=0.02)
        self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)))

    def forward(self, chunks: Any, directive_tokens: Any) -> Any:
        h = self.chunk_norm(chunks)
        d_vec = self.directive_norm(self.directive_proj(directive_tokens.mean(dim=1, keepdim=True)))
        align = torch.nn.functional.cosine_similarity(h, d_vec, dim=-1, eps=1e-6).unsqueeze(-1)
        gate = torch.sigmoid(self.logit_scale * align)
        gamma, beta = self.film(d_vec).chunk(2, dim=-1)
        return h * (0.25 + 1.75 * gate) * (1.0 + gamma) + gate * beta

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class TwoTierMemory(BaseModule):
    """Differentiable two-tier memory selector with L1 leading slots (first `max_l1` chunks) and L2 query-gated slots.

    Normalizes chunks before applying query-conditioned soft weights so gradients
    flow into `query` in both small (`num_chunks <= max_l2`) and top-k
    (`num_chunks > max_l2`) cases.
    """

    def __init__(
        self,
        dim: int,
        max_l1: int = 16,
        max_l2: int = 32,
        threshold: float = 0.0,
        temperature: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not _HAS_TORCH:
            raise ImportError("TwoTierMemory requires PyTorch.")
        if dim <= 0 or max_l1 <= 0 or max_l2 <= 0:
            raise ValueError(f"dim, max_l1, and max_l2 must be > 0, got ({dim}, {max_l1}, {max_l2}).")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")

        self.dim = int(dim)
        self.max_l1 = int(max_l1)
        self.max_l2 = int(max_l2)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.norm_l1 = RaNorm(self.dim, eps=eps, elementwise_affine=True, bias=False)
        self.norm_l2 = RaNorm(self.dim, eps=eps, elementwise_affine=True, bias=False)

    def forward(self, chunks: Any, query: Any) -> Tuple[Any, Any]:
        _, num_chunks, hidden_dim = chunks.shape
        q_unit = torch.nn.functional.normalize(query.mean(dim=1, keepdim=True), p=2, dim=-1)
        c_unit = torch.nn.functional.normalize(chunks, p=2, dim=-1)
        scores = torch.matmul(c_unit, q_unit.transpose(-1, -2)).squeeze(-1)

        l1 = self.norm_l1(chunks[:, : min(num_chunks, self.max_l1), :])

        if num_chunks > self.max_l2:
            top_scores, top_idx = torch.topk(scores, self.max_l2, dim=-1)
            gathered = torch.gather(chunks, 1, top_idx.unsqueeze(-1).expand(-1, -1, hidden_dim))
            weights = torch.sigmoid((top_scores - self.threshold) / self.temperature).unsqueeze(-1).to(dtype=gathered.dtype)
            l2 = self.norm_l2(gathered) * weights
        else:
            weights = torch.sigmoid((scores - self.threshold) / self.temperature).unsqueeze(-1).to(dtype=chunks.dtype)
            l2 = self.norm_l2(chunks) * weights
        return l1, l2

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, max_l1={self.max_l1}, max_l2={self.max_l2}, "
            f"threshold={self.threshold}, temperature={self.temperature}"
        )


__all__ = ["OrthogonalHub", "DirectiveGate", "TwoTierMemory"]
