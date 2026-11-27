"""Orthogonal hub, directive gating, and differentiable two-tier memory modules for luminamoon.nn."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple
import warnings

from ._attention_dispatch import AttentionBackend
from ._compat import BaseModule, _HAS_TORCH, _is_torch_compiling, nn, torch
from ._latent_attention import SlotCrossAttention
from .norm import RaNorm


class OrthogonalHub(BaseModule):
    """SO(H) orthogonal prefix hub parameterized via a float32 Cayley transform.

    Computes `Q = (I + S)^{-1}(I - S)` from a skew-symmetric matrix `S` in float32
    so BF16/FP16 models work with `torch.linalg.solve` without low-precision solve error,
    with eval-mode caching keyed on parameter version, device, and dtype. When
    `use_context_gate=True` in a causal LM, pass only past or prefix tokens as `context`.
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
        self._cached_basis_entry: Optional[Tuple[Any, Tuple[Any, ...]]] = None

    def _invalidate_cache(self) -> None:
        self._cached_q_entry = None
        self._cached_basis_entry = None

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
            # Clone as a normal tensor so cached entries created under inference_mode remain autograd-safe.
            q_cached = q_mat.detach()
            if q_cached.is_inference():
                with torch.inference_mode(False):
                    q_cached = q_cached.clone()
            self._cached_q_entry = (q_cached, int(self.weight_skew._version), cur_ptr)
        return q_mat

    def forward(self, batch_size: int, context: Optional[Any] = None) -> Any:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}.")
        if context is None:
            norm_w = self.norm.weight
            can_cache = (
                not _is_torch_compiling()
                and not self.training
                and not (
                    torch.is_grad_enabled()
                    and (
                        self.weight_skew.requires_grad
                        or self.hub_basis.requires_grad
                        or (norm_w is not None and norm_w.requires_grad)
                    )
                )
            )
            cache_key: Optional[Tuple[Any, ...]] = None
            if can_cache:
                cache_key = (
                    int(self.weight_skew._version),
                    int(self.weight_skew.data_ptr()),
                    int(self.hub_basis._version),
                    int(self.hub_basis.data_ptr()),
                    int(norm_w._version) if norm_w is not None else -1,
                    int(norm_w.data_ptr()) if norm_w is not None else 0,
                    str(self.hub_basis.device),
                    self.hub_basis.dtype,
                )
                entry = self._cached_basis_entry
                if entry is not None and entry[1] == cache_key:
                    return entry[0].expand(batch_size, -1, -1)
            q_mat = self.get_orthogonal_matrix()
            # Normalize the shared (hub_size, dim) basis before expanding across the batch.
            normed = self.norm(torch.matmul(q_mat, self.hub_basis)).unsqueeze(0)
            if can_cache and cache_key is not None:
                cached = normed.detach()
                if cached.is_inference():
                    with torch.inference_mode(False):
                        cached = cached.clone()
                self._cached_basis_entry = (cached, cache_key)
            return normed.expand(batch_size, -1, -1)

        if self.context_gate is None:
            raise ValueError("context was passed to OrthogonalHub with use_context_gate=False.")
        q_mat = self.get_orthogonal_matrix()
        basis = torch.matmul(q_mat, self.hub_basis).unsqueeze(0).expand(batch_size, -1, -1)
        if context.shape[1] == 0:
            return self.norm(basis)
        ctx_summary = self.context_gate(context.mean(dim=1, keepdim=True))
        return self.norm(basis + 0.5 * ctx_summary)

    def extra_repr(self) -> str:
        return f"hub_size={self.hub_size}, dim={self.dim}, use_context_gate={self.use_context_gate}"


class DirectiveGate(BaseModule):
    """Pre-norm directive-conditioned gating and FiLM modulation over chunk summaries.

    Applies `RaNorm` before scaling and shifting so cosine-alignment weights and
    channel-wise FiLM parameters are preserved rather than canceled by post-norm
    scale homogeneity.

    Note:
        Because `forward` mean-pools `directive_tokens` across the sequence dimension,
        causal autoregressive language models must pass only past or prefix tokens
        (e.g., past memory chunks or system/prompt prefix) as `directive_tokens`—never
        the full target sequence being predicted—to avoid future-token leakage.
        Use `assert_causal_no_leak(...)` to verify zero leakage before training.
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

    def forward(self, chunks: Any, directive_tokens: Optional[Any] = None) -> Any:
        if chunks.shape[1] == 0:
            return chunks
        d_in = chunks[:, -min(2, int(chunks.shape[1])) :] if directive_tokens is None else directive_tokens
        h = self.chunk_norm(chunks)
        d_vec = self.directive_norm(self.directive_proj(d_in.mean(dim=1, keepdim=True)))
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

    Note:
        Because `forward` aggregates across all positions of `query` (`max` over `q_len`),
        causal autoregressive language models must pass only past or prefix tokens
        (e.g., past chunk summaries or prompt prefix) as `query`—never the full target
        sequence being predicted—to avoid future-token leakage.
        Use `assert_causal_no_leak(...)` to verify zero leakage before training.
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
        memory_dropout: float = 0.0,
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
        if not (0.0 <= float(memory_dropout) < 1.0):
            raise ValueError(f"memory_dropout must be in [0.0, 1.0), got {memory_dropout}.")

        self.dim = int(dim)
        self.max_l1 = int(max_l1)
        self.max_l2 = int(max_l2)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.use_projections = bool(use_projections)
        self.use_topk = bool(use_topk)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.memory_dropout = float(memory_dropout)
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
        chunks: Optional[Any],
        query: Optional[Any] = None,
        *,
        use_topk: Optional[bool] = None,
        cached_keys: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        if chunks is None:
            if query is None:
                raise ValueError("TwoTierMemory.forward requires either chunks or query to infer device/dtype.")
            empty = torch.empty((query.shape[0], 0, self.dim), device=query.device, dtype=query.dtype)
            return empty, empty
        batch, num_chunks, hidden_dim = chunks.shape
        if num_chunks == 0 or (
            self.training and self.memory_dropout > 0.0 and not _is_torch_compiling()
            and float(torch.rand((), device=chunks.device).item()) < self.memory_dropout
        ):
            empty = torch.empty((batch, 0, hidden_dim), device=chunks.device, dtype=chunks.dtype)
            return empty, empty
        q_src = chunks[:, -min(2, num_chunks) :] if query is None else query
        q_len = q_src.shape[1]
        compute_dtype = torch.float64 if (chunks.dtype == torch.float64 or q_src.dtype == torch.float64) else torch.float32

        q_in = self.q_proj(q_src) if self.q_proj is not None else q_src
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


def _extract_first_tensor(out: Any) -> Any:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for item in out:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Expected callable output to be a torch.Tensor or tuple/list containing one, got {type(out).__name__}.")


def assert_causal_no_leak(
    fn: Callable[[Any], Any],
    sample_input: Any,
    *,
    atol: float = 0.0,
    token_dim: int = 1,
    raise_on_leak: bool = True,
) -> float:
    """Detect causal future-token leakage by perturbing the last position of `sample_input`.

    Perturbs only the final index along `token_dim`, runs `fn` twice under `torch.no_grad()`,
    and measures the maximum absolute difference across all earlier positions `0 .. S-2`.
    Raises `RuntimeError` (or emits `RuntimeWarning` when `raise_on_leak=False`) if `max_diff > atol`.
    """
    if not _HAS_TORCH or not isinstance(sample_input, torch.Tensor):
        raise TypeError("assert_causal_no_leak requires a torch.Tensor sample_input.")
    dim = token_dim if token_dim >= 0 else sample_input.ndim + token_dim
    if dim < 0 or dim >= sample_input.ndim:
        raise ValueError(f"token_dim {token_dim} out of range for input with ndim={sample_input.ndim}.")
    seq_len = int(sample_input.shape[dim])
    if seq_len < 2:
        raise ValueError(f"assert_causal_no_leak requires seq_len >= 2 along token_dim={token_dim}, got {seq_len}.")

    perturbed = sample_input.clone()
    last_slice: list[Any] = [slice(None)] * sample_input.ndim
    last_slice[dim] = -1
    last_idx = tuple(last_slice)
    if perturbed.is_floating_point():
        perturbed[last_idx] = perturbed[last_idx] + 1.0
    else:
        perturbed[last_idx] = perturbed[last_idx] ^ 1

    fn_mod: Any = fn if isinstance(fn, nn.Module) else None
    prev_training = bool(fn_mod.training) if fn_mod is not None else None
    try:
        if fn_mod is not None:
            fn_mod.eval()
        with torch.no_grad():
            out1 = _extract_first_tensor(fn(sample_input))
            out2 = _extract_first_tensor(fn(perturbed))
    finally:
        if fn_mod is not None and prev_training is not None:
            fn_mod.train(prev_training)

    prefix_slice: list[Any] = [slice(None)] * out1.ndim
    prefix_slice[dim] = slice(0, seq_len - 1)
    p_idx = tuple(prefix_slice)
    diff = (out1[p_idx].to(torch.float32) - out2[p_idx].to(torch.float32)).abs()
    max_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0

    if max_diff > float(atol) or math.isnan(max_diff):
        msg = (
            f"Causal future-token leakage detected (max_diff={max_diff:.3e} > atol={atol:.3e}). "
            "Perturbing the final token changed earlier hidden states. Ensure non-causal modules "
            "(such as DirectiveGate, TwoTierMemory, SlotCrossAttention, or sequence mean-pooling) "
            "only receive past or prefix tokens rather than the full target sequence."
        )
        if raise_on_leak:
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return max_diff


class CausalMemoryTap(BaseModule):
    """Causal memory tap combining TwoTierMemory, DirectiveGate, and SlotCrossAttention over past chunks."""

    def __init__(
        self, dim: int, num_heads: int = 4, max_l1: int = 2, max_l2: int = 4,
        num_terminals: int = 4, slots_per_terminal: int = 2, memory_dropout: float = 0.15,
        attention_backend: AttentionBackend = "auto", eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not _HAS_TORCH:
            raise ImportError("CausalMemoryTap requires PyTorch.")
        self.dim, self.num_terminals = int(dim), int(num_terminals)
        self.memory = TwoTierMemory(dim=self.dim, max_l1=max_l1, max_l2=max_l2, num_heads=num_heads, memory_dropout=memory_dropout, eps=eps)
        self.directive_gate = DirectiveGate(dim=self.dim, eps=eps)
        self.slot_attn = SlotCrossAttention(dim=self.dim, num_heads=num_heads, num_terminals=self.num_terminals, slots_per_terminal=slots_per_terminal, attention_backend=attention_backend, eps=eps)
        self.term_proj = nn.Linear(self.dim, self.num_terminals * self.dim, bias=False)
        self.fuse_proj = nn.Linear(self.num_terminals * int(slots_per_terminal) * self.dim, self.dim, bias=False)
        nn.init.zeros_(self.fuse_proj.weight)

    def forward(self, x: Any, past_chunks: Optional[Any] = None, *, cached_keys: Optional[Any] = None) -> Any:
        if past_chunks is None or past_chunks.shape[1] == 0:
            return x
        if past_chunks.data_ptr() == x.data_ptr() and x.shape[1] > 1:
            raise ValueError("past_chunks must not alias the target sequence x in CausalMemoryTap.")
        l1, l2 = self.memory(past_chunks, cached_keys=cached_keys)
        if l1.shape[1] == 0 and l2.shape[1] == 0:
            return x
        mem_ctx = self.directive_gate(torch.cat((l1, l2), dim=1))
        term_base = self.term_proj(past_chunks[:, -1]).view(x.shape[0], self.num_terminals, self.dim)
        slots = self.slot_attn(term_base, mem_ctx)
        return x + self.fuse_proj(slots.reshape(x.shape[0], -1)).unsqueeze(1)


__all__ = ["OrthogonalHub", "DirectiveGate", "TwoTierMemory", "CausalMemoryTap", "assert_causal_no_leak"]
