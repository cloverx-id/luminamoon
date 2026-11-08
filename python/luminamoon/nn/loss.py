from __future__ import annotations

import math
from typing import Any, Optional

from ._compat import (
    BaseModule,
    _HAS_NUMPY,
    _HAS_TORCH,
    _HAS_TRITON,
    _is_torch_compiling,
    np,
    torch,
    nn,
)
from .triton_kernels import (
    triton_fused_ce_chunk_bwd_inplace,
    triton_fused_ce_chunk_fwd,
)

# Default token chunk sizes (2048 on CUDA for Tensor Core occupancy, 512 on CPU for L3 cache locality)
_DEFAULT_CUDA_CHUNK_SIZE: int = 2048
_DEFAULT_CPU_CHUNK_SIZE: int = 512


def _resolve_chunk_size(M: int, chunk_size: Optional[int], is_cuda: bool = False) -> int:
    if chunk_size is not None and int(chunk_size) > 0:
        return max(1, min(int(M), int(chunk_size)))
    default_c = _DEFAULT_CUDA_CHUNK_SIZE if is_cuda else _DEFAULT_CPU_CHUNK_SIZE
    return max(1, min(int(M), default_c))


def _validate_ce_args(
    reduction: str,
    label_smoothing: float,
    logit_scale: float,
) -> str:
    red = reduction.strip().lower()
    if red not in ("mean", "sum", "none"):
        raise ValueError(
            f"Invalid reduction '{reduction}'. Expected one of: 'mean', 'sum', 'none'."
        )
    if not (0.0 <= float(label_smoothing) < 1.0) or math.isnan(label_smoothing):
        raise ValueError(
            f"label_smoothing must be in [0.0, 1.0), got {label_smoothing}."
        )
    if not math.isfinite(logit_scale) or float(logit_scale) <= 0.0:
        raise ValueError(
            f"logit_scale must be a positive finite float, got {logit_scale}."
        )
    return red


_FusedLinearCrossEntropyFunction: Any = None

if _HAS_TORCH and torch is not None:

    class _FusedLinearCrossEntropyFunctionImpl(torch.autograd.Function):
        """
        Custom autograd function for Fused Linear + Cross-Entropy.
        - On CUDA with Triton: Computes tiles of logits H_tile @ W_tile^T in SRAM registers,
          maintains online LogSumExp in SRAM, and computes analytical backward gradients
          dZ = P - I(i == target) in SRAM without ever materializing [M, V] logits in HBM.
        - On CPU / fallback: Uses token-chunked evaluation with a single reusable [C, V]
          scratch buffer and in-place analytical gradient overwrite so peak memory is O(C * V)
          instead of O(M * V).
        """

        @staticmethod
        def forward(
            ctx: Any,
            x: Any,
            weight: Any,
            target: Any,
            bias: Optional[Any],
            ignore_index: int = -100,
            reduction: str = "mean",
            label_smoothing: float = 0.0,
            logit_scale: float = 1.0,
            chunk_size: Optional[int] = None,
        ) -> Any:
            orig_target_shape = target.shape
            D = x.shape[-1]
            V = weight.shape[0]
            x_2d = x.reshape(-1, D)
            target_1d = target.reshape(-1)
            M = x_2d.shape[0]

            out_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64) else torch.float32

            if M == 0:
                empty_lse = torch.empty((0,), device=x.device, dtype=torch.float32)
                valid_count = torch.zeros((), device=x.device, dtype=torch.int64)
                ctx.save_for_backward(
                    x_2d,
                    weight,
                    bias if bias is not None else torch.empty(0, device=x.device),
                    target_1d,
                    empty_lse,
                    valid_count,
                )
                ctx.has_bias = bias is not None
                ctx.ignore_index = int(ignore_index)
                ctx.reduction = reduction
                ctx.label_smoothing = float(label_smoothing)
                ctx.logit_scale = float(logit_scale)
                ctx.chunk_size = chunk_size
                ctx.x_shape = x.shape
                if reduction == "none":
                    return torch.empty(orig_target_shape, device=x.device, dtype=out_dtype)
                return torch.zeros((), device=x.device, dtype=out_dtype)

            eff_chunk = _resolve_chunk_size(M, chunk_size, is_cuda=bool(x_2d.is_cuda))
            losses_1d = torch.empty((M,), device=x.device, dtype=torch.float32)
            lse_1d = torch.empty((M,), device=x.device, dtype=torch.float32)

            use_native_lp_gemm = (
                bool(x_2d.is_cuda)
                and x_2d.dtype == weight.dtype
                and x_2d.dtype in (torch.float16, torch.bfloat16)
            )
            w_mm = weight if use_native_lp_gemm else weight.to(torch.float32)
            b_f32 = bias.to(torch.float32) if bias is not None else None
            z_buf = torch.empty((min(M, eff_chunk), V), device=x.device, dtype=torch.float32)

            for start in range(0, M, eff_chunk):
                end = min(start + eff_chunk, M)
                c = end - start
                z_chunk = z_buf[:c]
                if use_native_lp_gemm:
                    z_chunk.copy_(torch.mm(x_2d[start:end], w_mm.t()))
                else:
                    x_c = x_2d[start:end].to(torch.float32)
                    torch.mm(x_c, w_mm.t(), out=z_chunk)
                if b_f32 is not None:
                    z_chunk.add_(b_f32)
                if logit_scale != 1.0:
                    z_chunk.mul_(logit_scale)

                t_c = target_1d[start:end]
                if _HAS_TRITON and z_chunk.is_cuda and triton_fused_ce_chunk_fwd(
                    z_chunk,
                    t_c,
                    losses_1d[start:end],
                    lse_1d[start:end],
                    ignore_index=ignore_index,
                    label_smoothing=label_smoothing,
                ):
                    continue

                valid_c = (t_c != ignore_index) & (t_c >= 0) & (t_c < V)
                safe_t = torch.where(valid_c, t_c, 0).to(torch.long)

                lse_c = torch.logsumexp(z_chunk, dim=-1)
                lse_c = torch.where(valid_c, lse_c, 0.0)
                z_target = z_chunk.gather(dim=-1, index=safe_t.unsqueeze(-1)).squeeze(-1)

                if label_smoothing > 0.0:
                    s_c = z_chunk.sum(dim=-1)
                    loss_c = (
                        lse_c
                        - (1.0 - label_smoothing) * z_target
                        - (label_smoothing / V) * s_c
                    )
                else:
                    loss_c = lse_c - z_target

                losses_1d[start:end] = torch.where(valid_c, loss_c, 0.0)
                lse_1d[start:end] = lse_c

            valid_mask_all = (target_1d != ignore_index) & (target_1d >= 0) & (target_1d < V)
            valid_count = valid_mask_all.sum()

            ctx.save_for_backward(
                x_2d,
                weight,
                bias if bias is not None else torch.empty(0, device=x.device),
                target_1d,
                lse_1d,
                valid_count,
            )
            ctx.has_bias = bias is not None
            ctx.ignore_index = int(ignore_index)
            ctx.reduction = reduction
            ctx.label_smoothing = float(label_smoothing)
            ctx.logit_scale = float(logit_scale)
            ctx.chunk_size = chunk_size
            ctx.x_shape = x.shape

            if reduction == "none":
                return losses_1d.view(orig_target_shape).to(out_dtype)
            if reduction == "sum":
                return losses_1d.sum().to(out_dtype)

            denom = valid_count.clamp(min=1).to(torch.float32)
            mean_loss = torch.where(
                valid_count > 0,
                losses_1d.sum() / denom,
                torch.zeros((), device=x.device, dtype=torch.float32),
            )
            return mean_loss.to(out_dtype)

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> Any:
            x_2d, weight, bias_saved, target_1d, lse_1d, valid_count = ctx.saved_tensors
            has_bias: bool = ctx.has_bias
            bias = bias_saved if has_bias else None
            ignore_index: int = ctx.ignore_index
            reduction: str = ctx.reduction
            label_smoothing: float = ctx.label_smoothing
            logit_scale: float = ctx.logit_scale
            chunk_size: Optional[int] = ctx.chunk_size
            x_shape = ctx.x_shape

            M, D = x_2d.shape
            V = weight.shape[0]

            need_dx = bool(ctx.needs_input_grad[0])
            need_dw = bool(ctx.needs_input_grad[1])
            need_db = bool(has_bias and ctx.needs_input_grad[3])

            if M == 0 or not (need_dx or need_dw or need_db):
                dx_empty = torch.zeros(x_shape, device=x_2d.device, dtype=x_2d.dtype) if need_dx else None
                dw_empty = torch.zeros_like(weight) if need_dw else None
                db_empty = torch.zeros_like(bias) if (need_db and bias is not None) else None
                return dx_empty, dw_empty, None, db_empty, None, None, None, None, None

            valid_mask_all = (target_1d != ignore_index) & (target_1d >= 0) & (target_1d < V)
            if reduction == "none":
                grad_loss_1d = torch.where(
                    valid_mask_all,
                    grad_output.reshape(-1).to(torch.float32),
                    0.0,
                )
            elif reduction == "sum":
                g_scalar = grad_output.reshape(()).to(torch.float32)
                grad_loss_1d = torch.where(valid_mask_all, g_scalar, 0.0)
            else:
                g_scalar = grad_output.reshape(()).to(torch.float32)
                denom = valid_count.clamp(min=1).to(torch.float32)
                scale_val = torch.where(
                    valid_count > 0,
                    g_scalar / denom,
                    torch.zeros((), device=x_2d.device, dtype=torch.float32),
                )
                grad_loss_1d = torch.where(valid_mask_all, scale_val, 0.0)

            eff_chunk = _resolve_chunk_size(M, chunk_size, is_cuda=bool(x_2d.is_cuda))

            use_native_lp_gemm = (
                bool(x_2d.is_cuda)
                and x_2d.dtype == weight.dtype
                and x_2d.dtype in (torch.float16, torch.bfloat16)
            )
            w_mm = weight if use_native_lp_gemm else weight.to(torch.float32)
            b_f32 = bias.to(torch.float32) if bias is not None else None

            gemm_dtype = x_2d.dtype if use_native_lp_gemm else torch.float32
            dx_buf = torch.empty((M, D), device=x_2d.device, dtype=gemm_dtype) if need_dx else None
            dw_buf = torch.zeros((V, D), device=weight.device, dtype=gemm_dtype) if need_dw else None
            db_f32 = torch.zeros((V,), device=weight.device, dtype=torch.float32) if need_db else None
            z_buf = torch.empty((min(M, eff_chunk), V), device=x_2d.device, dtype=torch.float32)

            for start in range(0, M, eff_chunk):
                end = min(start + eff_chunk, M)
                c = end - start
                z_chunk = z_buf[:c]
                x_c = x_2d[start:end] if use_native_lp_gemm else x_2d[start:end].to(torch.float32)

                if use_native_lp_gemm:
                    z_chunk.copy_(torch.mm(x_c, w_mm.t()))
                else:
                    torch.mm(x_c, w_mm.t(), out=z_chunk)
                if b_f32 is not None:
                    z_chunk.add_(b_f32)
                if logit_scale != 1.0:
                    z_chunk.mul_(logit_scale)

                t_c = target_1d[start:end]
                used_triton_bwd = False
                if _HAS_TRITON and z_chunk.is_cuda:
                    used_triton_bwd = triton_fused_ce_chunk_bwd_inplace(
                        z_chunk,
                        t_c,
                        lse_1d[start:end],
                        grad_loss_1d[start:end],
                        ignore_index=ignore_index,
                        label_smoothing=label_smoothing,
                        logit_scale=logit_scale,
                    )

                if not used_triton_bwd:
                    lse_c = lse_1d[start:end].unsqueeze(-1)
                    valid_c = (t_c != ignore_index) & (t_c >= 0) & (t_c < V)
                    safe_t = torch.where(valid_c, t_c, 0).to(torch.long).unsqueeze(-1)

                    # In-place analytical logit gradient: dZ = g_i * (exp(z - LSE) - (1 - eps)*I(v == y) - eps/V)
                    z_chunk.sub_(lse_c)
                    z_chunk.masked_fill_(~valid_c.unsqueeze(-1), 0.0)
                    z_chunk.exp_()
                    if label_smoothing > 0.0:
                        z_chunk.sub_(label_smoothing / V)
                    z_chunk.scatter_add_(
                        dim=-1,
                        index=safe_t,
                        src=torch.full(
                            (c, 1),
                            -(1.0 - label_smoothing),
                            device=z_chunk.device,
                            dtype=torch.float32,
                        ),
                    )
                    z_chunk.mul_((grad_loss_1d[start:end] * logit_scale).unsqueeze(-1))

                if db_f32 is not None:
                    db_f32.add_(z_chunk.sum(dim=0))
                z_mm = z_chunk.to(gemm_dtype) if use_native_lp_gemm else z_chunk
                if dx_buf is not None:
                    torch.mm(z_mm, w_mm, out=dx_buf[start:end])
                if dw_buf is not None:
                    dw_buf.addmm_(z_mm.t(), x_c)

            dx_out = dx_buf.reshape(x_shape).to(x_2d.dtype) if dx_buf is not None else None
            dw_out = dw_buf.to(weight.dtype) if dw_buf is not None else None
            db_out = db_f32.to(bias.dtype) if (db_f32 is not None and bias is not None) else None
            return dx_out, dw_out, None, db_out, None, None, None, None, None

    _FusedLinearCrossEntropyFunction = _FusedLinearCrossEntropyFunctionImpl


def fused_linear_cross_entropy(
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
) -> Any:
    """
    Fused Linear + Cross-Entropy loss without materializing the full [B * N, V] logits tensor.

    Computes:
        z_i = (x_i @ weight^T + bias) * logit_scale
        L_i = LogSumExp(z_i) - (1 - label_smoothing) * z_{i, target_i} - (label_smoothing / V) * sum_v(z_{i, v})

    with closed-form analytical backward gradient:
        dL / dz_i = P_i - (1 - label_smoothing) * I(v == target_i) - label_smoothing / V

    Args:
        x: Hidden state tensor of shape `(..., D)`.
        weight: Vocabulary projection matrix of shape `(V, D)`.
        target: Target token indices of shape `(...)` matching `x.shape[:-1]`.
        bias: Optional projection bias of shape `(V,)` (default: `None`).
        ignore_index: Target index value that is ignored and does not contribute to
            the loss or input gradient (default: `-100`).
        reduction: Reduction mode across valid tokens: `'mean'`, `'sum'`, or `'none'`
            (default: `'mean'`).
        label_smoothing: Label smoothing coefficient in `[0.0, 1.0)` (default: `0.0`).
        logit_scale: Multiplicative scalar factor applied to logits (default: `1.0`).
        chunk_size: Optional token chunk size override for the CPU/fallback path
            (default: `None`, auto-bounded to ~16 MB).
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
            raise ValueError(
                f"Expected bias to have shape ({V},), got {tuple(bias.shape)}."
            )

        # TorchDynamo requires distinct tensor identities for autograd.Function.apply inputs
        if _is_torch_compiling():
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
        w_np = np.asarray(weight, dtype=np.float32)
        t_np = np.asarray(target, dtype=np.int64)
        b_np = np.asarray(bias, dtype=np.float32) if bias is not None else None

        if x_np.ndim < 1:
            raise ValueError(f"Expected x to have at least 1 dimension, got shape {x_np.shape}.")
        if w_np.ndim != 2 or w_np.shape[1] != x_np.shape[-1]:
            raise ValueError(
                f"Shape mismatch between x {x_np.shape} and weight {w_np.shape}."
            )
        if t_np.shape != x_np.shape[:-1]:
            raise ValueError(
                f"Target shape {t_np.shape} must match x batch dimensions {x_np.shape[:-1]}."
            )

        D = x_np.shape[-1]
        V = w_np.shape[0]
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

    raise TypeError(f"Unsupported input type for fused_linear_cross_entropy: {type(x)}")


class FusedLinearCrossEntropy(BaseModule):
    """
    Fused Linear + Cross-Entropy Module (`nn.Module`).

    Supports two ergonomic operating modes:
    1. **Stateful LM Head + Loss Mode** (`in_features` and `vocab_size` provided):
       Owns projection parameter `self.weight` of shape `(vocab_size, in_features)` (and
       optional `self.bias`). Can be called as `loss = head(hidden_states, target)` during
       training (zero HBM logit materialization) or `logits = head(hidden_states)` during
       inference when `target=None`. Also supports weight-tying by passing an external
       `weight` tensor to `forward()`.
    2. **Stateless Criterion Mode** (`in_features=None, vocab_size=None`):
       Owns no parameters and is called as `loss = criterion(hidden_states, weight, target, bias=None)`.
    """

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
        if _HAS_TORCH and nn is not None and getattr(self, "weight", None) is not None:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if getattr(self, "bias", None) is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        x: Any,
        weight_or_target: Optional[Any] = None,
        target: Optional[Any] = None,
        bias: Optional[Any] = None,
    ) -> Any:
        if self.weight is not None:
            # Stateful mode:
            # - head(x) -> inference linear projection logits
            # - head(x, target) -> fused linear cross-entropy using self.weight & self.bias
            # - head(x, weight, target, bias=...) -> fused linear cross-entropy with explicit weight override
            if weight_or_target is None and target is None:
                if _HAS_TORCH and isinstance(x, torch.Tensor):
                    out = torch.nn.functional.linear(x, self.weight, self.bias)
                    return out * self.logit_scale if self.logit_scale != 1.0 else out
                if _HAS_NUMPY and isinstance(x, np.ndarray):
                    out = x @ self.weight.T
                    if self.bias is not None:
                        out = out + self.bias
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
            # Stateless mode: criterion(x, weight, target, bias=None)
            if weight_or_target is None or target is None:
                raise ValueError(
                    "Stateless FusedLinearCrossEntropy requires (x, weight, target[, bias])."
                )
            eff_weight = weight_or_target
            eff_target = target
            eff_bias = bias

        return fused_linear_cross_entropy(
            x,
            eff_weight,
            eff_target,
            bias=eff_bias,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
            logit_scale=self.logit_scale,
            chunk_size=self.chunk_size,
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
        return ", ".join(parts)

ChunkedLinearCrossEntropy = FusedLinearCrossEntropy
RaLinearCrossEntropy = FusedLinearCrossEntropy
LinearCrossEntropy = FusedLinearCrossEntropy
chunked_linear_cross_entropy = fused_linear_cross_entropy
ra_linear_cross_entropy = fused_linear_cross_entropy
linear_cross_entropy = fused_linear_cross_entropy

__all__ = [
    "FusedLinearCrossEntropy",
    "ChunkedLinearCrossEntropy",
    "RaLinearCrossEntropy",
    "LinearCrossEntropy",
    "fused_linear_cross_entropy",
    "chunked_linear_cross_entropy",
    "ra_linear_cross_entropy",
    "linear_cross_entropy",
]
