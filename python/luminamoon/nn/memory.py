"""Orthogonal hub, directive gating, and differentiable two-tier memory modules for luminamoon.nn."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

from ._compat import BaseModule, _HAS_TORCH, _is_torch_compiling, nn, torch
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

        self._cached_q_entry: Optional[Tuple[Any, int, int]] = None

    def _invalidate_cache(self) -> None:
        self._cached_q_entry = None

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        self._invalidate_cache()
        super()._load_from_state_dict(*args, **kwargs)

    def train(self, mode: bool = True) -> "OrthogonalHub":
        if mode:
            self._invalidate_cache()
        return super().train(mode)

    def get_orthogonal_matrix(self) -> Any:
        compiling = _is_torch_compiling()
        can_cache = (
            not compiling
            and not self.training
            and not (torch.is_grad_enabled() and self.weight_skew.requires_grad)
        )
        cur_ptr = 0
        if can_cache:
            cur_ptr = int(self.weight_skew.data_ptr())
            entry = self._cached_q_entry
            if entry is not None:
                cached_q, cached_ver, cached_ptr = entry
                if (
                    cached_ver == self.weight_skew._version
                    and cached_ptr == cur_ptr
                    and cached_q.device == self.weight_skew.device
                    and cached_q.dtype == self.hub_basis.dtype
                ):
                    return cached_q

        # torch.linalg.solve requires float32/float64 on CPU and CUDA.
        w_f32 = self.weight_skew.to(torch.float32)
        skew = 0.5 * (w_f32 - w_f32.T)
        eye = torch.eye(self.hub_size, device=skew.device, dtype=torch.float32)
        q_mat = torch.linalg.solve(eye + skew, eye - skew).to(dtype=self.hub_basis.dtype)

        if can_cache:
            self._cached_q_entry = (q_mat.detach(), int(self.weight_skew._version), cur_ptr)
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
        self.eps = float(eps)
        self.chunk_norm = RaNorm(self.dim, eps=self.eps, elementwise_affine=True, bias=False)
        self.directive_norm = RaNorm(self.dim, eps=self.eps, elementwise_affine=True, bias=False)
        self.directive_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.film = nn.Linear(self.dim, 2 * self.dim, bias=False)
        nn.init.normal_(self.film.weight, mean=0.0, std=0.02)
        self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)))

    def forward(self, chunks: Any, directive_tokens: Any) -> Any:
        h = self.chunk_norm(chunks)
        d_vec = self.directive_norm(self.directive_proj(directive_tokens.mean(dim=1, keepdim=True)))
        align = torch.nn.functional.cosine_similarity(h, d_vec, dim=-1, eps=self.eps).unsqueeze(-1)
        gate = torch.sigmoid(self.logit_scale * align)
        gamma, beta = self.film(d_vec).chunk(2, dim=-1)
        return h * (0.25 + 1.75 * gate) * (1.0 + gamma) + gate * beta

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class TwoTierMemory(BaseModule):
    """Differentiable multi-head two-tier memory selector with L1 leading slots and L2 query-gated slots.

    Scores each chunk using token-level max cosine similarity per head followed
    by normalized LogSumExp across heads so multi-topic queries do not collapse
    under query averaging. Supports optional learned query/key projections and
    optional soft-only mode (`use_topk=False`) during warmup.
    """

    def __init__(
        self,
        dim: int,
        max_l1: int = 16,
        max_l2: int = 32,
        num_heads: int = 4,
        use_projections: bool = True,
        use_topk: bool = True,
        threshold: float = 0.0,
        temperature: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not _HAS_TORCH:
            raise ImportError("TwoTierMemory requires PyTorch.")
        if dim <= 0 or max_l1 <= 0 or max_l2 <= 0:
            raise ValueError(f"dim, max_l1, and max_l2 must be > 0, got ({dim}, {max_l1}, {max_l2}).")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}) with num_heads > 0.")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")

        self.dim = int(dim)
        self.max_l1 = int(max_l1)
        self.max_l2 = int(max_l2)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.use_projections = bool(use_projections)
        self.use_topk = bool(use_topk)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.q_proj: Optional[Any] = nn.Linear(self.dim, self.dim, bias=False) if self.use_projections else None
        self.k_proj: Optional[Any] = nn.Linear(self.dim, self.dim, bias=False) if self.use_projections else None
        self.norm_l1 = RaNorm(self.dim, eps=self.eps, elementwise_affine=True, bias=False)
        self.norm_l2 = RaNorm(self.dim, eps=self.eps, elementwise_affine=True, bias=False)

    def encode_keys(self, chunks: Any) -> Any:
        """Project and L2-normalize memory keys to `(B, num_heads, N, head_dim)` for reuse across queries."""
        batch, num_chunks, _ = chunks.shape
        compute_dtype = torch.float64 if chunks.dtype == torch.float64 else torch.float32
        k_in = self.k_proj(chunks) if self.k_proj is not None else chunks
        k_heads = k_in.reshape(batch, num_chunks, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_unit = torch.nn.functional.normalize(k_heads.to(compute_dtype), p=2, dim=-1, eps=self.eps)
        return k_unit.to(dtype=chunks.dtype)

    def forward(
        self,
        chunks: Any,
        query: Any,
        *,
        use_topk: Optional[bool] = None,
        cached_keys: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        batch, num_chunks, hidden_dim = chunks.shape
        q_len = query.shape[1]
        compute_dtype = torch.float64 if (chunks.dtype == torch.float64 or query.dtype == torch.float64) else torch.float32

        q_in = self.q_proj(query) if self.q_proj is not None else query
        q_heads = q_in.reshape(batch, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q_unit = torch.nn.functional.normalize(q_heads.to(compute_dtype), p=2, dim=-1, eps=self.eps)

        if cached_keys is not None:
            expected_shape = (batch, self.num_heads, num_chunks, self.head_dim)
            if tuple(cached_keys.shape) != expected_shape:
                raise ValueError(f"cached_keys must have shape {expected_shape}, got {tuple(cached_keys.shape)}.")
            k_unit = cached_keys
        else:
            k_unit = self.encode_keys(chunks)

        # Max over query tokens per head preserves minority query tokens without averaging them out.
        scores_hn = torch.matmul(q_unit.to(dtype=k_unit.dtype), k_unit.transpose(-1, -2)).to(compute_dtype).max(dim=2).values
        if self.num_heads > 1:
            scores = self.temperature * (
                torch.logsumexp(scores_hn / self.temperature, dim=1) - math.log(self.num_heads)
            )
        else:
            scores = scores_hn.squeeze(1)

        l1 = self.norm_l1(chunks[:, : min(num_chunks, self.max_l1), :])
        apply_topk = self.use_topk if use_topk is None else bool(use_topk)

        if apply_topk and num_chunks > self.max_l2:
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
            f"num_heads={self.num_heads}, use_projections={self.use_projections}, "
            f"use_topk={self.use_topk}, threshold={self.threshold}, temperature={self.temperature}"
        )


__all__ = ["OrthogonalHub", "DirectiveGate", "TwoTierMemory"]
