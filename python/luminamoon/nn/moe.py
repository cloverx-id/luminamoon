from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

from ._compat import (
    BaseModule,
    _HAS_NUMPY,
    _HAS_TORCH,
    _is_torch_compiling,
    np,
    torch,
    nn,
)
from .activation import PosRaGLU, RaGLU, TriGLU
from .softmax import ra_softmax


def _build_rational_expert(
    in_features: int,
    hidden_features: int,
    out_features: int,
    expert_type: str = "pos_ra_glu",
    b: float = 1.0,
    bias: bool = False,
    device: Optional[Any] = None,
    dtype: Optional[Any] = None,
) -> Any:
    et = expert_type.strip().lower().replace("-", "_")
    if et in ("pos_ra_glu", "posraglu", "rational_glu"):
        return PosRaGLU(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            b=b,
            bias=bias,
            device=device,
            dtype=dtype,
        )
    if et in ("ra_glu", "raglu"):
        return RaGLU(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            b=b,
            bias=bias,
            device=device,
            dtype=dtype,
        )
    if et in ("tri_glu", "triglu", "triangle_glu"):
        return TriGLU(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            b=b,
            bias=bias,
            device=device,
            dtype=dtype,
        )
    raise ValueError(
        f"Unsupported expert_type '{expert_type}'. Expected 'pos_ra_glu', 'ra_glu', or 'tri_glu'."
    )


def permute_tokens_by_expert(
    x_2d: Any,
    top_indices: Any,
    top_weights: Any,
    num_experts: int,
) -> Tuple[Any, Any, Any, list[int]]:
    """
    Permutes token representations into a single contiguous buffer grouped by expert ID.

    Eliminates per-expert boolean indexing (`if mask.any():`) and CPU-GPU synchronization
    bottlenecks by sorting expert assignments once via `torch.argsort` and transferring
    only the tiny `[num_experts]` bincount histogram to host memory.

    Args:
        x_2d: Flattened token tensor of shape `(M, D)`.
        top_indices: Selected expert indices of shape `(M, K)`.
        top_weights: Normalized routing weights of shape `(M, K)`.
        num_experts: Total number of routed experts `E`.

    Returns:
        `(permuted_x, token_indices, sorted_weights, counts_list)`
    """
    K = top_indices.shape[-1]
    flat_experts = top_indices.reshape(-1)
    flat_weights = top_weights.reshape(-1)

    perm = torch.argsort(flat_experts, stable=True)
    token_indices = torch.div(perm, K, rounding_mode="floor")
    permuted_x = x_2d.index_select(0, token_indices)
    sorted_weights = flat_weights.index_select(0, perm)

    counts = torch.bincount(flat_experts.to(torch.int64), minlength=int(num_experts))
    counts_list: list[int] = counts.tolist()
    return permuted_x, token_indices, sorted_weights, counts_list


class RaMoE(BaseModule):
    """Rational Mixture-of-Experts with contiguous token permutation dispatch and optional shared experts.

    Note:
        When `hidden_features` is omitted, each expert defaults to `in_features * 4`
        regardless of `expert_type` (unlike standalone `PosRaGLU`/`RaGLU` at `8/3` or `TriGLU` at `2`).
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        num_experts: int = 8,
        top_k: int = 2,
        *,
        num_shared_experts: int = 0,
        expert_type: str = "pos_ra_glu",
        b: float = 1.0,
        ra_scale: float = 0.0625,
        bias: bool = False,
        aux_loss_coef: float = 0.01,
        return_aux_loss: bool = False,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if int(in_features) <= 0:
            raise ValueError(f"in_features must be > 0, got {in_features}.")
        in_features = int(in_features)
        if num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {num_experts}.")
        if not (1 <= top_k <= num_experts):
            raise ValueError(
                f"top_k ({top_k}) must be between 1 and num_experts ({num_experts})."
            )
        if num_shared_experts < 0:
            raise ValueError(f"num_shared_experts must be >= 0, got {num_shared_experts}.")

        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features) if hidden_features is not None else int(in_features * 4)
        self.out_features = int(out_features) if out_features is not None else int(in_features)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.num_shared_experts = int(num_shared_experts)
        self.expert_type = expert_type
        self.b = float(b)
        if not (math.isfinite(self.b) and self.b > 0.0):
            raise ValueError(f"b must be finite and > 0, got {b}.")
        self.ra_scale = float(ra_scale)
        if not (math.isfinite(self.ra_scale) and self.ra_scale > 0.0):
            raise ValueError(f"ra_scale must be finite and > 0, got {ra_scale}.")
        self.aux_loss_coef = float(aux_loss_coef)
        self.return_aux_loss = bool(return_aux_loss)
        self.last_aux_loss: Optional[Any] = None

        if _HAS_TORCH and nn is not None:
            factory_kwargs = {"device": device, "dtype": dtype}
            self.router = nn.Linear(self.in_features, self.num_experts, bias=False, **factory_kwargs)
            self.experts = nn.ModuleList(
                [
                    _build_rational_expert(
                        self.in_features,
                        self.hidden_features,
                        self.out_features,
                        expert_type=self.expert_type,
                        b=self.b,
                        bias=bias,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in range(self.num_experts)
                ]
            )
            if self.num_shared_experts > 0:
                self.shared_experts = nn.ModuleList(
                    [
                        _build_rational_expert(
                            self.in_features,
                            self.hidden_features,
                            self.out_features,
                            expert_type=self.expert_type,
                            b=self.b,
                            bias=bias,
                            device=device,
                            dtype=dtype,
                        )
                        for _ in range(self.num_shared_experts)
                    ]
                )
            else:
                self.shared_experts = None
        elif _HAS_NUMPY:
            bound = 1.0 / math.sqrt(self.in_features)
            self.router_weight = np.random.uniform(
                -bound, bound, size=(self.num_experts, self.in_features)
            ).astype(np.float32)
            self.experts = [
                _build_rational_expert(
                    self.in_features,
                    self.hidden_features,
                    self.out_features,
                    expert_type=self.expert_type,
                    b=self.b,
                    bias=bias,
                )
                for _ in range(self.num_experts)
            ]
            self.shared_experts = (
                [
                    _build_rational_expert(
                        self.in_features,
                        self.hidden_features,
                        self.out_features,
                        expert_type=self.expert_type,
                        b=self.b,
                        bias=bias,
                    )
                    for _ in range(self.num_shared_experts)
                ]
                if self.num_shared_experts > 0
                else None
            )

    def _compute_aux_loss_torch(
        self,
        router_logits: Any,
        top_indices: Any,
    ) -> Any:
        M = router_logits.shape[0]
        if M == 0:
            return torch.zeros((), device=router_logits.device, dtype=router_logits.dtype)

        router_probs = ra_softmax(router_logits.to(torch.float32), dim=-1, scale=self.ra_scale)
        mean_prob_per_expert = router_probs.mean(dim=0)

        flat_indices = top_indices.reshape(-1).to(torch.int64)
        counts = torch.zeros((self.num_experts,), device=router_logits.device, dtype=torch.float32)
        counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float32))
        frac_per_expert = counts / float(max(1, flat_indices.numel()))

        aux_loss = (
            float(self.num_experts)
            * float(self.aux_loss_coef)
            * torch.sum(mean_prob_per_expert * frac_per_expert)
        )
        return aux_loss.to(router_logits.dtype)

    def forward(
        self,
        x: Any,
        *,
        return_aux_loss: Optional[bool] = None,
    ) -> Union[Any, Tuple[Any, Any]]:
        ret_aux = self.return_aux_loss if return_aux_loss is None else bool(return_aux_loss)

        if _HAS_TORCH and isinstance(x, torch.Tensor):
            orig_shape = x.shape
            if x.ndim < 2 or orig_shape[-1] != self.in_features:
                raise ValueError(
                    f"Expected input tensor with shape (..., {self.in_features}), got {tuple(orig_shape)}."
                )

            x_2d = x.reshape(-1, self.in_features)
            M = x_2d.shape[0]
            out_shape = orig_shape[:-1] + (self.out_features,)

            if M == 0:
                empty_out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
                self.last_aux_loss = torch.zeros((), device=x.device, dtype=x.dtype)
                return (empty_out, self.last_aux_loss) if ret_aux else empty_out

            router_logits = self.router(x_2d)
            top_vals, top_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
            if self.top_k == 1:
                # Preserve router gradient flow when top_k=1 via full router probabilities
                full_probs = ra_softmax(router_logits, dim=-1, scale=self.ra_scale)
                top_weights = full_probs.gather(dim=-1, index=top_indices)
            else:
                top_weights = ra_softmax(top_vals, dim=-1, scale=self.ra_scale)

            self.last_aux_loss = self._compute_aux_loss_torch(router_logits, top_indices)

            if _is_torch_compiling():
                # Branchless TorchDynamo-friendly execution path (zero graph breaks)
                out_2d = torch.zeros((M, self.out_features), device=x.device, dtype=x.dtype)
                for e_idx, expert in enumerate(self.experts):
                    mask = top_indices == e_idx
                    weights_e = torch.where(mask, top_weights, torch.zeros_like(top_weights)).sum(
                        dim=-1, keepdim=True
                    )
                    out_2d = out_2d + expert(x_2d) * weights_e.to(x.dtype)
            else:
                # High-throughput Token Permutation Dispatcher (torch.argsort + contiguous slicing)
                permuted_x, token_indices, sorted_weights, counts_list = permute_tokens_by_expert(
                    x_2d, top_indices, top_weights, self.num_experts
                )
                permuted_out = torch.empty(
                    (permuted_x.shape[0], self.out_features),
                    device=x.device,
                    dtype=x.dtype,
                )
                offset = 0
                for e_idx, count in enumerate(counts_list):
                    if count > 0:
                        end = offset + count
                        permuted_out[offset:end] = self.experts[e_idx](permuted_x[offset:end])
                        offset = end

                permuted_out = permuted_out * sorted_weights.to(x.dtype).unsqueeze(-1)
                out_2d = torch.zeros((M, self.out_features), device=x.device, dtype=x.dtype)
                out_2d.index_add_(0, token_indices, permuted_out)

            if self.shared_experts is not None:
                for shared_exp in self.shared_experts:
                    out_2d = out_2d + shared_exp(x_2d)

            out = out_2d.reshape(out_shape)
            return (out, self.last_aux_loss) if ret_aux else out

        if _HAS_NUMPY and isinstance(x, np.ndarray):
            x_np = np.asarray(x, dtype=np.float32)
            orig_shape = x_np.shape
            if x_np.ndim < 2 or orig_shape[-1] != self.in_features:
                raise ValueError(
                    f"Expected input array with shape (..., {self.in_features}), got {orig_shape}."
                )
            x_2d = x_np.reshape(-1, self.in_features)
            M = x_2d.shape[0]
            out_shape = orig_shape[:-1] + (self.out_features,)

            if M == 0:
                empty_out = np.empty(out_shape, dtype=np.float32)
                self.last_aux_loss = np.float32(0.0)
                return (empty_out, self.last_aux_loss) if ret_aux else empty_out

            r_w = (
                self.router_weight
                if hasattr(self, "router_weight")
                else self.router.weight.detach().cpu().numpy()
            )
            router_logits = x_2d @ r_w.T
            # Top-k selection in NumPy
            part_idx = np.argpartition(-router_logits, kth=self.top_k - 1, axis=-1)[:, : self.top_k]
            row_idx = np.arange(M)[:, None]
            part_vals = router_logits[row_idx, part_idx]
            order = np.argsort(-part_vals, axis=-1)
            top_indices = part_idx[row_idx, order]
            top_vals = part_vals[row_idx, order]

            if self.top_k == 1:
                full_probs = ra_softmax(router_logits, dim=-1, scale=self.ra_scale)
                top_weights = np.take_along_axis(full_probs, top_indices, axis=-1)
            else:
                top_weights = ra_softmax(top_vals, dim=-1, scale=self.ra_scale)

            flat_experts = top_indices.reshape(-1)
            flat_weights = top_weights.reshape(-1)
            perm = np.argsort(flat_experts, kind="stable")
            token_indices = perm // self.top_k
            permuted_x = x_2d[token_indices]
            sorted_weights = flat_weights[perm]
            counts = np.bincount(flat_experts, minlength=self.num_experts)

            permuted_out = np.zeros((permuted_x.shape[0], self.out_features), dtype=np.float32)
            offset = 0
            for e_idx, count in enumerate(counts.tolist()):
                if count > 0:
                    end = offset + count
                    permuted_out[offset:end] = self.experts[e_idx](permuted_x[offset:end])
                    offset = end

            permuted_out = permuted_out * sorted_weights[:, None]
            out_2d = np.zeros((M, self.out_features), dtype=np.float32)
            np.add.at(out_2d, token_indices, permuted_out)

            if self.shared_experts is not None:
                for shared_exp in self.shared_experts:
                    out_2d = out_2d + shared_exp(x_2d)

            router_probs = ra_softmax(router_logits, dim=-1, scale=self.ra_scale)
            mean_prob = np.mean(router_probs, axis=0)
            frac = counts.astype(np.float32) / float(max(1, flat_experts.size))
            self.last_aux_loss = np.float32(
                self.num_experts * self.aux_loss_coef * np.sum(mean_prob * frac)
            )
            out = out_2d.reshape(out_shape)
            return (out, self.last_aux_loss) if ret_aux else out

        raise TypeError(f"Unsupported input type for RaMoE: {type(x)}")

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, hidden_features={self.hidden_features}, "
            f"out_features={self.out_features}, num_experts={self.num_experts}, "
            f"top_k={self.top_k}, num_shared_experts={self.num_shared_experts}, "
            f"expert_type='{self.expert_type}'"
        )


__all__ = ["RaMoE", "permute_tokens_by_expert"]
