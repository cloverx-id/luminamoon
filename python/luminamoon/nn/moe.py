from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

from ._compat import BaseModule, _HAS_NUMPY, _HAS_TORCH, _as_numpy, _is_torch_compiling, nn, np, torch
from ._custom_ops import _HAS_CUSTOM_OP, compiled_moe_grouped_gemm, compiled_ra_gated_packed
from ._moe_ops import (
    SMALL_BATCH_DECODE_MAX_TOKENS,
    _can_use_grouped_mm,
    _small_batch_routed_forward,
    _sort_routing_indices,
)
from ._simd_ops import _cpu_ra_gated_packed
from .activation import RaGLU, RaSwiGLU, ra_glu, ra_swiglu, ra_tanh
from .kernels import can_use_triton_grouped_gemm, triton_grouped_gemm, triton_ra_gated_packed
from .softmax import ra_softmax


def _normalize_expert_type(expert_type: str) -> str:
    et = expert_type.strip().lower().replace("-", "_")
    if et in ("ra_swiglu", "raswiglu"):
        return "ra_swiglu"
    if et in ("ra_glu", "raglu"):
        return "ra_glu"
    raise ValueError(f"Unsupported expert_type '{expert_type}'. Expected 'ra_swiglu' or 'ra_glu'.")


def _build_shared_expert(
    in_features: int, hidden_features: int, out_features: int, expert_type: str,
    b: float, bias: bool, device: Optional[Any] = None, dtype: Optional[Any] = None,
) -> Any:
    cls = RaSwiGLU if expert_type == "ra_swiglu" else RaGLU
    return cls(in_features, hidden_features, out_features, bias=bias, b=b, device=device, dtype=dtype)


def permute_tokens_by_expert(
    x_2d: Any,
    top_indices: Any,
    top_weights: Any,
    num_experts: int,
) -> Tuple[Any, Any, Any, Any]:
    """Permutes tokens into an expert-grouped buffer with on-device int64 `counts`."""
    token_indices, sorted_weights, counts, _ = _sort_routing_indices(top_indices, top_weights, num_experts)
    permuted_x = x_2d.index_select(0, token_indices)
    return permuted_x, token_indices, sorted_weights, counts


class RaMoE(BaseModule):
    """Rational Mixture-of-Experts with packed 3D expert weights and optional grouped GEMM.

    Note: Setting `router_logit_bound` (e.g. `8.0` or `12.0`) softly bounds router logits via
    `ra_tanh` before `ra_softmax`, preventing any lagging expert from exceeding the compact-support
    dead-gradient horizon (`Delta >= 1 / ra_scale`, default `16.0`) during long training runs.
    """

    router: Any
    w_gate_up: Any
    w_down: Any
    b_gate_up: Any
    b_down: Any
    shared_experts: Any

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        num_experts: int = 8,
        top_k: int = 2,
        *,
        num_shared_experts: int = 0,
        expert_type: str = "ra_swiglu",
        b: float = 1.0,
        ra_scale: float = 0.0625,
        router_logit_bound: Optional[float] = None,
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
            raise ValueError(f"top_k ({top_k}) must be between 1 and num_experts ({num_experts}).")
        if num_shared_experts < 0:
            raise ValueError(f"num_shared_experts must be >= 0, got {num_shared_experts}.")

        self.in_features = in_features
        self.hidden_features = int(hidden_features) if hidden_features is not None else int(in_features * 8 / 3)
        self.out_features = int(out_features) if out_features is not None else in_features
        if self.hidden_features <= 0 or self.out_features <= 0:
            raise ValueError(f"hidden_features ({self.hidden_features}) and out_features ({self.out_features}) must be > 0.")
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.num_shared_experts = int(num_shared_experts)
        self.expert_type = _normalize_expert_type(expert_type)
        self.b = float(b)
        if not (math.isfinite(self.b) and self.b > 0.0):
            raise ValueError(f"b must be finite and > 0, got {b}.")
        self.ra_scale = float(ra_scale)
        if not (math.isfinite(self.ra_scale) and self.ra_scale > 0.0):
            raise ValueError(f"ra_scale must be finite and > 0, got {ra_scale}.")
        if router_logit_bound is not None and not (math.isfinite(float(router_logit_bound)) and float(router_logit_bound) > 0.0):
            raise ValueError(f"router_logit_bound must be finite and > 0 when set, got {router_logit_bound}.")
        self.router_logit_bound = float(router_logit_bound) if router_logit_bound is not None else None
        self.aux_loss_coef = float(aux_loss_coef)
        if not (math.isfinite(self.aux_loss_coef) and self.aux_loss_coef >= 0.0):
            raise ValueError(f"aux_loss_coef must be finite and >= 0, got {aux_loss_coef}.")
        self.return_aux_loss = bool(return_aux_loss)
        self.has_bias = bool(bias)
        self.last_aux_loss: Optional[Any] = None

        if _HAS_TORCH and nn is not None:
            factory_kwargs = {"device": device, "dtype": dtype}
            self.router = nn.Linear(self.in_features, self.num_experts, bias=False, **factory_kwargs)
            self.w_gate_up = nn.Parameter(torch.empty((self.num_experts, self.in_features, 2 * self.hidden_features), **factory_kwargs))
            self.w_down = nn.Parameter(torch.empty((self.num_experts, self.hidden_features, self.out_features), **factory_kwargs))
            if self.has_bias:
                self.b_gate_up = nn.Parameter(torch.zeros((self.num_experts, 2 * self.hidden_features), **factory_kwargs))
                self.b_down = nn.Parameter(torch.zeros((self.num_experts, self.out_features), **factory_kwargs))
            else:
                self.register_parameter("b_gate_up", None)
                self.register_parameter("b_down", None)
            self._reset_expert_parameters()

            if self.num_shared_experts > 0:
                self.shared_experts = nn.ModuleList([
                    _build_shared_expert(
                        self.in_features, self.hidden_features, self.out_features,
                        self.expert_type, self.b, self.has_bias, device=device, dtype=dtype,
                    )
                    for _ in range(self.num_shared_experts)
                ])
            else:
                self.shared_experts = None
        elif _HAS_NUMPY:
            bound_in = 1.0 / math.sqrt(self.in_features)
            bound_hid = 1.0 / math.sqrt(max(1, self.hidden_features))
            self.router_weight = np.random.uniform(
                -bound_in, bound_in, size=(self.num_experts, self.in_features)
            ).astype(np.float32)
            self.w_gate_up = np.random.uniform(
                -bound_in, bound_in, size=(self.num_experts, self.in_features, 2 * self.hidden_features)
            ).astype(np.float32)
            self.w_down = np.random.uniform(
                -bound_hid, bound_hid, size=(self.num_experts, self.hidden_features, self.out_features)
            ).astype(np.float32)
            self.b_gate_up = np.zeros((self.num_experts, 2 * self.hidden_features), dtype=np.float32) if self.has_bias else None
            self.b_down = np.zeros((self.num_experts, self.out_features), dtype=np.float32) if self.has_bias else None
            self.shared_experts = [
                _build_shared_expert(
                    self.in_features, self.hidden_features, self.out_features,
                    self.expert_type, self.b, self.has_bias,
                )
                for _ in range(self.num_shared_experts)
            ] if self.num_shared_experts > 0 else None

    def _reset_expert_parameters(self) -> None:
        bound_in = 1.0 / math.sqrt(self.in_features)
        bound_hid = 1.0 / math.sqrt(max(1, self.hidden_features))
        with torch.no_grad():
            self.w_gate_up.uniform_(-bound_in, bound_in)
            self.w_down.uniform_(-bound_hid, bound_hid)

    def eval_expert(self, expert_idx: int, x_e: Any) -> Any:
        """Evaluates a single routed expert on a 2D token batch `(M_e, in_features)`."""
        act_fn = ra_swiglu if self.expert_type == "ra_swiglu" else ra_glu
        if _HAS_TORCH and isinstance(x_e, torch.Tensor):
            w_gu = self.w_gate_up[expert_idx]
            w_dn = self.w_down[expert_idx]
            gu = torch.mm(x_e, w_gu if w_gu.dtype == x_e.dtype else w_gu.to(x_e.dtype))
            if self.b_gate_up is not None:
                b_gu = self.b_gate_up[expert_idx]
                gu = gu + (b_gu if b_gu.dtype == x_e.dtype else b_gu.to(x_e.dtype))
            h = self._apply_packed_activation(gu)
            out = torch.mm(h, w_dn if w_dn.dtype == h.dtype else w_dn.to(h.dtype))
            if self.b_down is not None:
                b_dn = self.b_down[expert_idx]
                out = out + (b_dn if b_dn.dtype == out.dtype else b_dn.to(out.dtype))
            return out
        w_gu = _as_numpy(self.w_gate_up)[expert_idx]
        w_dn = _as_numpy(self.w_down)[expert_idx]
        gu_np = np.matmul(x_e, w_gu)
        if self.b_gate_up is not None:
            gu_np = gu_np + _as_numpy(self.b_gate_up)[expert_idx]
        gate_np = np.ascontiguousarray(gu_np[:, : self.hidden_features])
        up_np = np.ascontiguousarray(gu_np[:, self.hidden_features :])
        h_np = act_fn(gate_np, up_np, b=self.b)
        out_np = np.matmul(h_np, w_dn)
        if self.b_down is not None:
            out_np = out_np + _as_numpy(self.b_down)[expert_idx]
        return out_np

    def _apply_packed_activation(self, gu: Any, *, use_custom_op: bool = False) -> Any:
        is_swi = self.expert_type == "ra_swiglu"
        if use_custom_op and _HAS_CUSTOM_OP and gu.is_cuda and gu.ndim == 2 and (gu.shape[1] % 2 == 0):
            return compiled_ra_gated_packed(gu, b=self.b, is_swiglu=is_swi)
        packed = triton_ra_gated_packed(gu, b=self.b, is_swiglu=is_swi)
        if packed is not None:
            return packed
        cpu_packed = _cpu_ra_gated_packed(gu, b=self.b, is_swiglu=is_swi)
        if cpu_packed is not None:
            return cpu_packed
        gate, up = gu.split(self.hidden_features, dim=-1)
        return (ra_swiglu if is_swi else ra_glu)(gate.contiguous(), up.contiguous(), b=self.b)

    def _dispatch_routed_torch(
        self, x_2d: Any, w_gu: Any, w_dn: Any, token_indices: Any, counts: Any, sorted_experts: Any
    ) -> Any:
        offs: Any = None
        permuted_x: Any = None
        b_gu = self.b_gate_up.to(x_2d.dtype) if (self.b_gate_up is not None and self.b_gate_up.dtype != x_2d.dtype) else self.b_gate_up
        b_dn = self.b_down.to(x_2d.dtype) if (self.b_down is not None and self.b_down.dtype != x_2d.dtype) else self.b_down
        if _can_use_grouped_mm(x_2d, w_gu, w_dn):
            try:
                _grouped_mm = getattr(torch, "_grouped_mm")
                offs = counts.cumsum(0).to(torch.int32)
                permuted_x = x_2d.index_select(0, token_indices)
                gu = _grouped_mm(permuted_x, w_gu, offs=offs)
                if b_gu is not None:
                    gu = gu + b_gu.index_select(0, sorted_experts)
                out = _grouped_mm(self._apply_packed_activation(gu), w_dn, offs=offs)
                return out + b_dn.index_select(0, sorted_experts) if b_dn is not None else out
            except RuntimeError:
                pass

        if can_use_triton_grouped_gemm(x_2d, w_gu) and w_dn.dtype == x_2d.dtype:
            offs = counts.cumsum(0).to(torch.int32) if offs is None else offs
            gu = triton_grouped_gemm(x_2d, w_gu, offs, gather_idx=token_indices)
            if b_gu is not None:
                gu = gu + b_gu.index_select(0, sorted_experts)
            out = triton_grouped_gemm(self._apply_packed_activation(gu), w_dn, offs)
            return out + b_dn.index_select(0, sorted_experts) if b_dn is not None else out

        if permuted_x is None:
            permuted_x = x_2d.index_select(0, token_indices)
        permuted_out = torch.empty((permuted_x.shape[0], self.out_features), device=x_2d.device, dtype=x_2d.dtype)
        offset = 0
        for e_idx, count in enumerate(counts.tolist()):
            if count > 0:
                end = offset + count
                permuted_out[offset:end] = self.eval_expert(e_idx, permuted_x[offset:end])
                offset = end
        return permuted_out

    def _compute_aux_loss_torch(self, router_logits: Any, top_indices: Any, router_probs: Optional[Any] = None) -> Any:
        if router_logits.shape[0] == 0:
            return torch.zeros((), device=router_logits.device, dtype=router_logits.dtype)
        probs = router_probs.to(torch.float32) if router_probs is not None else ra_softmax(router_logits.to(torch.float32), dim=-1, scale=self.ra_scale)
        flat_indices = top_indices.reshape(-1).to(torch.int64)
        if flat_indices.device.type == "cpu" and not _is_torch_compiling():
            counts = torch.bincount(flat_indices, minlength=self.num_experts).to(torch.float32)
        else:
            counts = torch.zeros((self.num_experts,), device=router_logits.device, dtype=torch.float32)
            counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float32))
        frac_per_expert = counts / float(max(1, flat_indices.numel()))
        aux_loss = float(self.num_experts) * float(self.aux_loss_coef) * torch.sum(probs.mean(dim=0) * frac_per_expert)
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
                z = x_2d.sum() * 0.0 + sum(p.to(x.dtype).sum() * 0.0 for p in self.parameters())
                empty_out = torch.zeros(out_shape, device=x.device, dtype=x.dtype) + z
                if not _is_torch_compiling():
                    self.last_aux_loss = z.detach()
                return (empty_out, z) if ret_aux else empty_out

            router_logits = self.router(x_2d) if x_2d.dtype == self.router.weight.dtype else torch.nn.functional.linear(x_2d, self.router.weight.to(x_2d.dtype))
            if self.router_logit_bound is not None:
                router_logits = ra_tanh(router_logits, b=self.router_logit_bound)
            top_vals, top_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
            full_probs: Optional[Any] = None
            if self.top_k == 1:
                full_probs = ra_softmax(router_logits, dim=-1, scale=self.ra_scale)
                top_weights = full_probs.gather(dim=-1, index=top_indices)
            else:
                top_weights = ra_softmax(top_vals, dim=-1, scale=self.ra_scale)

            aux_loss: Any = None
            if self.training or ret_aux:
                aux_loss = self._compute_aux_loss_torch(router_logits, top_indices, router_probs=full_probs)
                if not _is_torch_compiling():
                    self.last_aux_loss = aux_loss.detach()

            amp_cast = x_2d.is_cuda and x_2d.dtype in (torch.bfloat16, torch.float16) and self.w_gate_up.dtype == torch.float32
            w_gu = self.w_gate_up.to(x_2d.dtype) if amp_cast else self.w_gate_up
            w_dn = self.w_down.to(x_2d.dtype) if amp_cast else self.w_down
            if M <= SMALL_BATCH_DECODE_MAX_TOKENS and x.device.type != "meta":
                out_2d = _small_batch_routed_forward(
                    x_2d,
                    top_indices,
                    top_weights,
                    w_gu,
                    w_dn,
                    self.b_gate_up,
                    self.b_down,
                    hidden_features=self.hidden_features,
                    expert_type=self.expert_type,
                    b=self.b,
                )
            else:
                can_compile_triton_moe = (
                    _is_torch_compiling()
                    and _HAS_CUSTOM_OP
                    and can_use_triton_grouped_gemm(x_2d, w_gu)
                    and w_dn.dtype == x_2d.dtype
                )
                if (_is_torch_compiling() and not can_compile_triton_moe) or x.device.type == "meta":
                    out_2d = torch.zeros((M, self.out_features), device=x.device, dtype=x.dtype)
                    for e_idx in range(self.num_experts):
                        mask = top_indices == e_idx
                        weights_e = torch.where(mask, top_weights, torch.zeros_like(top_weights)).sum(
                            dim=-1, keepdim=True
                        )
                        out_2d = out_2d + self.eval_expert(e_idx, x_2d) * weights_e.to(x.dtype)
                else:
                    token_indices, sorted_weights, counts, sorted_experts = _sort_routing_indices(
                        top_indices, top_weights, self.num_experts
                    )
                    if can_compile_triton_moe:
                        offs = counts.cumsum(0).to(torch.int32)
                        gu = compiled_moe_grouped_gemm(x_2d, w_gu, offs, gather_idx=token_indices)
                        if self.b_gate_up is not None:
                            gu = gu + self.b_gate_up.to(x_2d.dtype).index_select(0, sorted_experts)
                        permuted_out = compiled_moe_grouped_gemm(
                            self._apply_packed_activation(gu, use_custom_op=True), w_dn, offs
                        )
                        if self.b_down is not None:
                            permuted_out = permuted_out + self.b_down.to(x_2d.dtype).index_select(0, sorted_experts)
                    else:
                        permuted_out = self._dispatch_routed_torch(
                            x_2d, w_gu, w_dn, token_indices, counts, sorted_experts if self.has_bias else None
                        )
                    permuted_out = permuted_out * sorted_weights.to(x.dtype).unsqueeze(-1)
                    out_2d = torch.zeros((M, self.out_features), device=x.device, dtype=x.dtype)
                    out_2d.index_add_(0, token_indices, permuted_out)

            if self.shared_experts is not None:
                for shared_exp in self.shared_experts:
                    out_2d = out_2d + shared_exp(x_2d)

            out = out_2d.reshape(out_shape)
            return (out, aux_loss) if ret_aux else out

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

            r_w = self.router_weight if hasattr(self, "router_weight") else self.router.weight.detach().cpu().numpy()
            router_logits = x_2d @ r_w.T
            if self.router_logit_bound is not None:
                router_logits = ra_tanh(router_logits, b=self.router_logit_bound)
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
                    permuted_out[offset:end] = self.eval_expert(e_idx, permuted_x[offset:end])
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
            aux_np = np.float32(
                self.num_experts * self.aux_loss_coef * np.sum(mean_prob * frac)
            )
            self.last_aux_loss = aux_np
            out = out_2d.reshape(out_shape)
            return (out, aux_np) if ret_aux else out

        raise TypeError(f"Unsupported input type for RaMoE: {type(x)}")

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, hidden_features={self.hidden_features}, "
            f"out_features={self.out_features}, num_experts={self.num_experts}, "
            f"top_k={self.top_k}, num_shared_experts={self.num_shared_experts}, "
            f"expert_type='{self.expert_type}'"
        )


__all__ = ["RaMoE", "permute_tokens_by_expert"]
