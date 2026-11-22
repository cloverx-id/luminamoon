from __future__ import annotations

import math
from typing import Any, Optional

from ._compat import BaseModule, _HAS_NUMPY, _HAS_TORCH, _as_numpy, _is_torch_compiling, nn, np, torch
from ._custom_ops import _HAS_CUSTOM_OP, compiled_linear_ce
from ._loss_ops import _FusedLinearCrossEntropyFunction, _resolve_chunk_size, _validate_ce_args


def _materialized_ce_torch(
    x: Any,
    weight: Any,
    target: Any,
    bias: Optional[Any],
    ignore_index: int,
    red: str,
    label_smoothing: float,
    logit_scale: float,
) -> Any:
    D = x.shape[-1]
    V = weight.shape[0]
    out_dtype = (
        x.dtype
        if x.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        else torch.float32
    )
    if x.numel() == 0:
        if red == "none":
            return torch.empty(target.shape, device=x.device, dtype=out_dtype)
        return torch.zeros((), device=x.device, dtype=out_dtype)

    b_t = bias if isinstance(bias, torch.Tensor) else None
    w_in = weight
    b_in = b_t
    if x.dtype in (torch.float16, torch.bfloat16) and weight.dtype == torch.float32:
        w_in = weight.to(x.dtype)
        if b_in is not None and b_in.dtype != x.dtype:
            b_in = b_in.to(x.dtype)
    logits = torch.nn.functional.linear(x.reshape(-1, D), w_in, b_in)
    if logit_scale != 1.0:
        logits = logits * float(logit_scale)
    t_flat = target.reshape(-1).to(torch.long)
    valid_t = (t_flat != int(ignore_index)) & (t_flat >= 0) & (t_flat < V)
    safe_t = torch.where(valid_t, t_flat, int(ignore_index))
    loss = torch.nn.functional.cross_entropy(
        logits,
        safe_t,
        ignore_index=int(ignore_index),
        reduction=red,
        label_smoothing=float(label_smoothing),
    )
    if red == "mean":
        loss = torch.where(valid_t.any(), loss, torch.zeros_like(loss))
    loss = loss.to(out_dtype)
    return loss.reshape(target.shape) if red == "none" else loss


def linear_cross_entropy(
    x: Any,
    weight: Any,
    target: Any,
    bias: Optional[Any] = None,
    *,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    chunk_size: Optional[int] = None,
    save_logits: bool = False,
) -> Any:
    """Fused Linear + Cross-Entropy loss without materializing full `[B * N, V]` logits.

    Set `save_logits=True` to opt into materialized logits (2-GEMM backward, TorchInductor
    fusable) when VRAM headroom is ample and raw throughput is preferred over O(1) memory.

    Note:
        Targets equal to `ignore_index` or outside `[0, vocab_size)` are silently ignored
        (zero loss and zero gradient) to avoid out-of-bounds memory access in fused kernels.
        This differs from `torch.nn.functional.cross_entropy`, which raises on out-of-range targets.
    """
    red = _validate_ce_args(reduction, label_smoothing, logit_scale)

    if _HAS_TORCH and isinstance(x, torch.Tensor):
        if x.ndim < 1:
            raise ValueError(f"Expected x to have at least 1 dimension (..., D), got shape {tuple(x.shape)}.")
        if weight.ndim != 2:
            raise ValueError(f"Expected weight to be a 2D tensor (V, D), got shape {tuple(weight.shape)}.")
        D = x.shape[-1]
        V, w_d = weight.shape
        if w_d != D:
            raise ValueError(
                f"Hidden dimension mismatch: x has last dim {D}, but weight has shape ({V}, {w_d})."
            )
        if tuple(target.shape) != tuple(x.shape[:-1]):
            raise ValueError(
                f"Target shape {tuple(target.shape)} must match x batch dimensions {tuple(x.shape[:-1])}."
            )
        if bias is not None and (bias.ndim != 1 or bias.shape[0] != V):
            raise ValueError(f"Expected bias to have shape ({V},), got {tuple(bias.shape)}.")

        if save_logits:
            return _materialized_ce_torch(
                x, weight, target, bias, int(ignore_index), red, float(label_smoothing), float(logit_scale)
            )

        if _is_torch_compiling():
            if _HAS_CUSTOM_OP and x.is_cuda and x.numel() > 0:
                return compiled_linear_ce(
                    x, weight, target, bias, int(ignore_index), red,
                    float(label_smoothing), float(logit_scale), chunk_size,
                )
            if chunk_size is None and (x.numel() // max(1, D)) * V <= 4_194_304 and x.numel() > 0:
                return _materialized_ce_torch(
                    x, weight, target, bias, int(ignore_index), red, float(label_smoothing), float(logit_scale)
                )
            if weight is x:
                weight = weight.clone()
            if bias is not None and (bias is x or bias is weight):
                bias = bias.clone()

        return _FusedLinearCrossEntropyFunction.apply(
            x,
            weight,
            target,
            bias,
            int(ignore_index),
            red,
            float(label_smoothing),
            float(logit_scale),
            chunk_size,
        )

    if _HAS_NUMPY and isinstance(x, np.ndarray):
        x_np = np.asarray(x, dtype=np.float32)
        w_np = np.asarray(_as_numpy(weight), dtype=np.float32)
        t_np = np.asarray(_as_numpy(target), dtype=np.int64)
        b_np = np.asarray(_as_numpy(bias), dtype=np.float32) if bias is not None else None

        if x_np.ndim < 1:
            raise ValueError(f"Expected x to have at least 1 dimension, got shape {x_np.shape}.")
        if w_np.ndim != 2 or w_np.shape[1] != x_np.shape[-1]:
            raise ValueError(f"Shape mismatch between x {x_np.shape} and weight {w_np.shape}.")
        if t_np.shape != x_np.shape[:-1]:
            raise ValueError(
                f"Target shape {t_np.shape} must match x batch dimensions {x_np.shape[:-1]}."
            )

        D, V = x_np.shape[-1], w_np.shape[0]
        x_2d = x_np.reshape(-1, D)
        t_1d = t_np.reshape(-1)
        M = x_2d.shape[0]

        eff_chunk = _resolve_chunk_size(max(1, M), chunk_size, is_cuda=False)
        losses_1d = np.zeros((M,), dtype=np.float32)

        for start in range(0, M, eff_chunk):
            end = min(start + eff_chunk, M)
            z_c = x_2d[start:end] @ w_np.T
            if b_np is not None:
                z_c = z_c + b_np
            if logit_scale != 1.0:
                z_c = z_c * float(logit_scale)

            tc = t_1d[start:end]
            valid_c = (tc != ignore_index) & (tc >= 0) & (tc < V)
            safe_t = np.where(valid_c, tc, 0)

            max_z = np.max(z_c, axis=-1, keepdims=True)
            lse_c = np.squeeze(max_z, axis=-1) + np.log(np.sum(np.exp(z_c - max_z), axis=-1))
            z_target = z_c[np.arange(end - start), safe_t]

            if label_smoothing > 0.0:
                s_c = np.sum(z_c, axis=-1)
                loss_c = lse_c - (1.0 - label_smoothing) * z_target - (label_smoothing / V) * s_c
            else:
                loss_c = lse_c - z_target

            losses_1d[start:end] = np.where(valid_c, loss_c, 0.0)

        if red == "none":
            return losses_1d.reshape(t_np.shape)
        if red == "sum":
            return np.float32(np.sum(losses_1d))

        valid_all = (t_1d != ignore_index) & (t_1d >= 0) & (t_1d < V)
        n_valid = int(np.sum(valid_all))
        if n_valid == 0:
            return np.float32(0.0)
        return np.float32(np.sum(losses_1d) / n_valid)

    raise TypeError(f"Unsupported input type for linear_cross_entropy: {type(x)}")


class LinearCrossEntropy(BaseModule):
    weight: Any
    bias: Any
    """Fused Linear + Cross-Entropy module supporting stateful LM head and stateless criterion modes."""

    def __init__(
        self,
        in_features: Optional[int] = None,
        vocab_size: Optional[int] = None,
        *,
        bias: bool = False,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        logit_scale: float = 1.0,
        chunk_size: Optional[int] = None,
        save_logits: bool = False,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if (in_features is None) != (vocab_size is None):
            raise ValueError(
                "Both in_features and vocab_size must be specified together for stateful mode, "
                "or both omitted (None) for stateless criterion mode."
            )
        self.in_features = int(in_features) if in_features is not None else None
        self.vocab_size = int(vocab_size) if vocab_size is not None else None
        self.ignore_index = int(ignore_index)
        self.reduction = _validate_ce_args(reduction, label_smoothing, logit_scale)
        self.label_smoothing = float(label_smoothing)
        self.logit_scale = float(logit_scale)
        self.chunk_size = chunk_size
        self.save_logits = bool(save_logits)

        if self.in_features is not None and self.vocab_size is not None:
            if self.in_features <= 0 or self.vocab_size <= 0:
                raise ValueError(
                    f"in_features ({self.in_features}) and vocab_size ({self.vocab_size}) must be > 0."
                )
            if _HAS_TORCH and nn is not None:
                factory_kwargs = {"device": device, "dtype": dtype}
                self.weight = nn.Parameter(
                    torch.empty((self.vocab_size, self.in_features), **factory_kwargs)
                )
                if bias:
                    self.bias = nn.Parameter(torch.empty(self.vocab_size, **factory_kwargs))
                else:
                    self.register_parameter("bias", None)
                self.reset_parameters()
            elif _HAS_NUMPY:
                bound = 1.0 / math.sqrt(self.in_features)
                self.weight = np.random.uniform(
                    -bound, bound, size=(self.vocab_size, self.in_features)
                ).astype(np.float32)
                self.bias = np.zeros((self.vocab_size,), dtype=np.float32) if bias else None
        else:
            if _HAS_TORCH and nn is not None:
                self.register_parameter("weight", None)
                self.register_parameter("bias", None)
            else:
                self.weight = None
                self.bias = None

    def reset_parameters(self) -> None:
        if _HAS_TORCH and nn is not None and isinstance(getattr(self, "weight", None), torch.Tensor):
            w = self.weight
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            b = getattr(self, "bias", None)
            if isinstance(b, torch.Tensor):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
                bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                nn.init.uniform_(b, -bound, bound)

    def forward(
        self,
        x: Any,
        weight_or_target: Optional[Any] = None,
        target: Optional[Any] = None,
        bias: Optional[Any] = None,
    ) -> Any:
        if self.weight is not None:
            if weight_or_target is None and target is None:
                if _HAS_TORCH and isinstance(x, torch.Tensor) and isinstance(self.weight, torch.Tensor):
                    b_t = self.bias if isinstance(self.bias, torch.Tensor) else None
                    out = torch.nn.functional.linear(x, self.weight, b_t)
                    return out * self.logit_scale if self.logit_scale != 1.0 else out
                if _HAS_NUMPY and isinstance(x, np.ndarray):
                    w_np = np.asarray(_as_numpy(self.weight), dtype=np.float32)
                    out = np.asarray(x, dtype=np.float32) @ w_np.T
                    if self.bias is not None:
                        out = out + np.asarray(_as_numpy(self.bias), dtype=np.float32)
                    return out * self.logit_scale if self.logit_scale != 1.0 else out

            if target is None:
                eff_weight = self.weight
                eff_target = weight_or_target
                eff_bias = bias if bias is not None else self.bias
            else:
                eff_weight = weight_or_target
                eff_target = target
                eff_bias = bias if bias is not None else self.bias
        else:
            if weight_or_target is None or target is None:
                raise ValueError(
                    "Stateless LinearCrossEntropy requires (x, weight, target[, bias])."
                )
            eff_weight = weight_or_target
            eff_target = target
            eff_bias = bias

        return linear_cross_entropy(
            x,
            eff_weight,
            eff_target,
            bias=eff_bias,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
            logit_scale=self.logit_scale,
            chunk_size=self.chunk_size,
            save_logits=self.save_logits,
        )

    def extra_repr(self) -> str:
        parts = []
        if self.in_features is not None and self.vocab_size is not None:
            parts.append(f"in_features={self.in_features}, vocab_size={self.vocab_size}")
            parts.append(f"bias={self.bias is not None}")
        parts.append(f"ignore_index={self.ignore_index}")
        parts.append(f"reduction='{self.reduction}'")
        if self.label_smoothing > 0.0:
            parts.append(f"label_smoothing={self.label_smoothing}")
        if self.logit_scale != 1.0:
            parts.append(f"logit_scale={self.logit_scale}")
        if self.save_logits:
            parts.append("save_logits=True")
        return ", ".join(parts)


__all__ = ["LinearCrossEntropy", "linear_cross_entropy"]
