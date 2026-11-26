from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

from ._compat import (
    BaseModule, _HAS_NUMPY, _HAS_RUST_SIMD, _HAS_TORCH, _HAS_TRITON,
    _as_numpy, _can_use_cubecl_nn, _is_torch_compiling, _rs_engine, nn, np, torch,
)
from .kernels import (
    triton_ra_batch_norm2d, triton_ra_batch_norm2d_backward, triton_ra_norm, triton_ra_norm_backward,
)

_RaNormFunction: Any = None
_RaBatchNorm2dFunction: Any = None


def _update_bn2d_running_stats(
    running_mean: Any, running_mad: Any, num_batches_tracked: Any,
    mean_1d: Any, mad_1d: Any, momentum: Optional[float],
) -> None:
    if running_mean is None or running_mad is None:
        return
    with torch.no_grad():
        if num_batches_tracked is not None:
            num_batches_tracked += 1
            exp_avg = momentum if momentum is not None else 1.0 / float(num_batches_tracked)
        else:
            exp_avg = momentum if momentum is not None else 1.0
        running_mean.lerp_(mean_1d.to(running_mean.dtype), exp_avg)
        running_mad.lerp_(mad_1d.to(running_mad.dtype), exp_avg)

def _safe_cast(t: Any, dt: Any) -> Any:
    return t.clamp(-65504.0, 65504.0).to(dt) if dt == torch.float16 else (t.to(dt) if t.dtype != dt else t)

def _eff_eps(eps: float, dt: Any) -> float:
    return max(float(eps), 2e-5) if dt == torch.float16 else float(eps)

def _ra_norm_eager_torch(x: Any, weight: Optional[Any], bias: Optional[Any], eps: float, dims: Any) -> Any:
    x_f32 = x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
    normed = x_f32 / (x_f32.abs().mean(dim=dims, keepdim=True) + _eff_eps(eps, x.dtype))
    if weight is not None:
        normed = normed * (weight.to(x_f32.dtype) if weight.dtype != x_f32.dtype else weight)
    if bias is not None:
        normed = normed + (bias.to(x_f32.dtype) if bias.dtype != x_f32.dtype else bias)
    return _safe_cast(normed, x.dtype)


def _ra_bn2d_eager_fwd(
    x: Any, weight: Optional[Any], bias: Optional[Any],
    running_mean: Optional[Any], running_mad: Optional[Any], use_batch_stats: bool, eps: float,
) -> Tuple[Any, Any, Any, Any]:
    x_f = x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
    if use_batch_stats:
        mean = x_f.mean(dim=(0, 2, 3), keepdim=True)
        mad = (x_f - mean).abs().mean(dim=(0, 2, 3), keepdim=True)
    else:
        mean = running_mean.view(1, -1, 1, 1).to(x_f.dtype)
        mad = running_mad.view(1, -1, 1, 1).to(x_f.dtype)
    inv_d = 1.0 / (mad + _eff_eps(eps, x.dtype))
    y_f = (x_f - mean) * inv_d
    if weight is not None and bias is not None:
        y_f = y_f * weight.view(1, -1, 1, 1).to(x_f.dtype) + bias.view(1, -1, 1, 1).to(x_f.dtype)
    return _safe_cast(y_f, x.dtype), mean.to(torch.float32).reshape(-1), mad.to(torch.float32).reshape(-1), inv_d.to(torch.float32).reshape(-1)


if _HAS_TORCH and torch is not None:

    class _RaNormFunctionImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any, weight: Optional[Any], bias: Optional[Any], eps: float) -> Any:
            ctx.eps = _eff_eps(eps, x.dtype)
            ctx.has_weight = weight is not None
            ctx.has_bias = bias is not None
            ctx.bias_dtype = bias.dtype if bias is not None else None
            saved_w = weight if weight is not None else torch.empty(0, device=x.device, dtype=x.dtype)
            triton_res = triton_ra_norm(x, weight, bias, eps=ctx.eps, return_inv_denom=True)
            if triton_res is not None:
                out, inv_denom = triton_res
                ctx.save_for_backward(x, saved_w, inv_denom)
                return out
            ctx.save_for_backward(x, saved_w)
            return _ra_norm_eager_torch(x, weight, bias, ctx.eps, -1)

        @staticmethod
        def backward(ctx: Any, grad_y: Any) -> Tuple[Any, Any, Any, None]:  # type: ignore[override]
            x, saved_w = ctx.saved_tensors[0], ctx.saved_tensors[1]
            inv_denom = ctx.saved_tensors[2] if len(ctx.saved_tensors) > 2 else None
            weight = saved_w if ctx.has_weight else None
            triton_res = triton_ra_norm_backward(
                grad_y, x, weight=weight, has_bias=ctx.has_bias, eps=ctx.eps, inv_denom=inv_denom
            )
            if triton_res is not None:
                dx, dw, db = triton_res
                return (
                    dx.reshape(x.shape),
                    dw if (ctx.has_weight and ctx.needs_input_grad[1]) else None,
                    db if (ctx.has_bias and ctx.needs_input_grad[2]) else None,
                    None,
                )

            x_f32 = x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
            gy_f32 = grad_y.to(torch.float32) if grad_y.dtype in (torch.float16, torch.bfloat16) else grad_y
            l1_mean = x_f32.abs().mean(dim=-1, keepdim=True)
            inv_denom = 1.0 / (l1_mean + ctx.eps)
            x_hat = x_f32 * inv_denom
            if weight is not None:
                w_f32 = weight.to(x_f32.dtype)
                dy_w = gy_f32 * w_f32
            else:
                dy_w = gy_f32
            c = (dy_w * x_hat).mean(dim=-1, keepdim=True)
            dx = _safe_cast(inv_denom * (dy_w - torch.sign(x_f32) * c), x.dtype)

            reduce_dims = tuple(range(x.ndim - 1))
            dw = (
                _safe_cast((gy_f32 * x_hat).sum(dim=reduce_dims), weight.dtype)
                if (weight is not None and ctx.needs_input_grad[1])
                else None
            )
            db_dtype = getattr(ctx, "bias_dtype", None) or (weight.dtype if weight is not None else x.dtype)
            db = (
                _safe_cast(gy_f32.sum(dim=reduce_dims), db_dtype)
                if (ctx.has_bias and ctx.needs_input_grad[2])
                else None
            )
            return dx, dw, db, None

    _RaNormFunction = _RaNormFunctionImpl

    class _RaBatchNorm2dFunctionImpl(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any, x: Any, weight: Optional[Any], bias: Optional[Any],
            running_mean: Optional[Any], running_mad: Optional[Any], num_batches_tracked: Optional[Any],
            use_batch_stats: bool, track_running_stats: bool, momentum: Optional[float], eps: float,
        ) -> Any:
            eff_e = _eff_eps(eps, x.dtype)
            triton_res = triton_ra_batch_norm2d(x, weight, bias, running_mean, running_mad, training=use_batch_stats, eps=eff_e)
            if triton_res is not None:
                y, mean_f32, mad_f32, inv_d_f32 = triton_res
            else:
                y, mean_f32, mad_f32, inv_d_f32 = _ra_bn2d_eager_fwd(x, weight, bias, running_mean, running_mad, use_batch_stats, eff_e)
            if use_batch_stats and track_running_stats:
                _update_bn2d_running_stats(running_mean, running_mad, num_batches_tracked, mean_f32, mad_f32, momentum)
            has_affine = weight is not None and bias is not None
            saved_w = weight if has_affine else torch.empty(0, device=x.device, dtype=x.dtype)
            ctx.save_for_backward(x, mean_f32, inv_d_f32, saved_w)
            ctx.has_affine = has_affine
            ctx.use_batch_stats = bool(use_batch_stats)
            ctx.bias_dtype = bias.dtype if bias is not None else None
            return y

        @staticmethod
        def backward(ctx: Any, grad_y: Any) -> Tuple[Any, ...]:  # type: ignore[override]
            x, mean_f32, inv_d_f32, saved_w = ctx.saved_tensors
            weight = saved_w if ctx.has_affine else None
            need_dx = bool(ctx.needs_input_grad[0])
            need_dw = bool(ctx.has_affine and ctx.needs_input_grad[1])
            need_db = bool(ctx.has_affine and ctx.needs_input_grad[2])
            res = triton_ra_batch_norm2d_backward(
                grad_y, x, mean_f32, inv_d_f32, weight, training=ctx.use_batch_stats,
                need_dx=need_dx, need_dw=need_dw, need_db=need_db, bias_dtype=ctx.bias_dtype,
            )
            if res is not None:
                dx, dw, db = res
                return dx, dw, db, None, None, None, None, None, None, None
            gy_f, x_f = grad_y.to(torch.float32), x.to(torch.float32)
            d = x_f - mean_f32.view(1, -1, 1, 1)
            inv_d = inv_d_f32.view(1, -1, 1, 1)
            dw = _safe_cast((gy_f * (d * inv_d)).sum(dim=(0, 2, 3)), weight.dtype) if (need_dw and weight is not None) else None
            db_dt = ctx.bias_dtype or (weight.dtype if weight is not None else x.dtype)
            db = _safe_cast(gy_f.sum(dim=(0, 2, 3)), db_dt) if need_db else None
            dx = None
            if need_dx:
                u = gy_f * weight.to(torch.float32).view(1, -1, 1, 1) if weight is not None else gy_f
                if ctx.use_batch_stats:
                    sgn = torch.sign(d)
                    dx = _safe_cast(inv_d * ((u - u.mean(dim=(0, 2, 3), keepdim=True)) - (u * (d * inv_d)).mean(dim=(0, 2, 3), keepdim=True) * (sgn - sgn.mean(dim=(0, 2, 3), keepdim=True))), x.dtype)
                else:
                    dx = _safe_cast(u * inv_d, x.dtype)
            return dx, dw, db, None, None, None, None, None, None, None

    _RaBatchNorm2dFunction = _RaBatchNorm2dFunctionImpl


class RaNorm(BaseModule):
    """Rational Normalization (RaNorm): square-root-free L1 normalization `(x / (mean(|x|) + eps)) * gamma + beta`.

    Uses single-reciprocal amortization across the trailing dimension with Triton JIT on CUDA
    and Rust SIMD on CPU. Works best on dense activations (RaSwiGLU, RaGLU, RaGELU, RaTanh);
    avoid placing directly after ultra-sparse activations with extreme isolated outliers.
    """

    weight: Any
    bias: Any

    def __init__(
        self,
        normalized_shape: Union[int, Sequence[int]],
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape: Tuple[int, ...] = (normalized_shape,)
        else:
            self.normalized_shape = tuple(int(s) for s in normalized_shape)
        if not self.normalized_shape or any(s <= 0 for s in self.normalized_shape):
            raise ValueError(f"normalized_shape must contain positive integers, got {normalized_shape}.")

        self.eps = float(eps)
        if not (self.eps > 0.0 and self.eps < float("inf")):
            raise ValueError(f"eps must be finite and > 0, got {eps}.")
        self.elementwise_affine = bool(elementwise_affine)

        if _HAS_TORCH:
            if self.elementwise_affine:
                self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
                self.register_parameter(
                    "bias",
                    nn.Parameter(torch.zeros(self.normalized_shape, device=device, dtype=dtype)) if bias else None,
                )
            else:
                self.register_parameter("weight", None)
                self.register_parameter("bias", None)
        elif self.elementwise_affine and _HAS_NUMPY:
            self.weight = np.ones(self.normalized_shape, dtype=np.float32)
            self.bias = np.zeros(self.normalized_shape, dtype=np.float32) if bias else None
        else:
            self.weight = self.bias = None

    def forward(self, x: Any) -> Any:
        n_dims = len(self.normalized_shape)
        if _HAS_TORCH and isinstance(x, torch.Tensor):
            if x.ndim < n_dims or tuple(x.shape[-n_dims:]) != self.normalized_shape:
                raise ValueError(f"Expected input trailing shape {self.normalized_shape}, got {tuple(x.shape)}.")
            needs_grad = torch.is_grad_enabled() and (
                x.requires_grad
                or (self.weight is not None and self.weight.requires_grad)
                or (self.bias is not None and self.bias.requires_grad)
            )
            if _HAS_TRITON and not _is_torch_compiling() and x.is_cuda and n_dims == 1 and 0 < x.shape[-1] <= 8192:
                if needs_grad and _RaNormFunction is not None:
                    return _RaNormFunction.apply(x, self.weight, self.bias, self.eps)
                triton_out = triton_ra_norm(x, self.weight, self.bias, eps=self.eps)
                if triton_out is not None:
                    return triton_out
            if (
                _HAS_RUST_SIMD and not _is_torch_compiling() and x.device.type == "cpu" and not needs_grad
                and x.dtype == torch.float32 and x.is_contiguous() and n_dims == 1
                and x.ndim >= 1 and x.shape[-1] == self.normalized_shape[0]
                and (self.weight is None or (self.weight.dtype == torch.float32 and self.weight.is_contiguous() and self.weight.device.type == "cpu"))
                and (self.bias is None or (self.bias.dtype == torch.float32 and self.bias.is_contiguous() and self.bias.device.type == "cpu"))
            ):
                out = torch.empty_like(x)
                fn = _rs_engine.ra_norm_cubecl_f32_pyo3 if _can_use_cubecl_nn(x.numel(), "ra_norm_cubecl_f32_pyo3") else _rs_engine.ra_norm_f32_pyo3
                fn(x, out, self.weight, self.bias, self.eps, x.shape[-1])
                return out
            return _ra_norm_eager_torch(x, self.weight, self.bias, self.eps, tuple(range(-n_dims, 0)))

        # 2. NumPy Array execution path (zero PyTorch dependency)
        if _HAS_NUMPY and isinstance(x, np.ndarray):
            if x.ndim < n_dims or tuple(x.shape[-n_dims:]) != self.normalized_shape:
                raise ValueError(f"Expected input trailing shape {self.normalized_shape}, got {tuple(x.shape)}.")
            w_np, b_np = _as_numpy(self.weight), _as_numpy(self.bias)
            if (
                _HAS_RUST_SIMD and not _is_torch_compiling() and x.dtype == np.float32 and x.flags.c_contiguous
                and n_dims == 1 and x.ndim >= 1 and x.shape[-1] == self.normalized_shape[0]
                and (w_np is None or (w_np.dtype == np.float32 and w_np.flags.c_contiguous))
                and (b_np is None or (b_np.dtype == np.float32 and b_np.flags.c_contiguous))
            ):
                out = np.empty_like(x)
                fn = _rs_engine.ra_norm_cubecl_f32_pyo3 if _can_use_cubecl_nn(x.size, "ra_norm_cubecl_f32_pyo3") else _rs_engine.ra_norm_f32_pyo3
                fn(x, out, w_np, b_np, self.eps, x.shape[-1])
                return out
            axis = tuple(range(-n_dims, 0))
            normed = x / (np.mean(np.abs(x), axis=axis, keepdims=True) + self.eps)
            if w_np is not None:
                normed = normed * w_np
            if b_np is not None:
                normed = normed + b_np
            return normed

        raise TypeError(f"RaNorm expects torch.Tensor or numpy.ndarray, got {type(x)}")

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}, bias={self.bias is not None}"


class RaBatchNorm2d(BaseModule):
    """Rational 2D Batch Normalization (RaBatchNorm2d) using L1 Mean Absolute Deviation (MAD)."""

    weight: Any
    bias: Any
    running_mean: Any
    running_mad: Any
    num_batches_tracked: Any

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: Optional[float] = 0.1, affine: bool = True, track_running_stats: bool = True, device: Any = None, dtype: Any = None) -> None:
        super().__init__()
        self.num_features, self.eps, self.affine, self.track_running_stats = int(num_features), float(eps), bool(affine), bool(track_running_stats)
        if self.num_features <= 0:
            raise ValueError(f"num_features must be > 0, got {num_features}.")
        if not (self.eps > 0.0 and self.eps < float("inf")):
            raise ValueError(f"eps must be finite and > 0, got {eps}.")
        self.momentum: Optional[float] = float(momentum) if momentum is not None else None
        if self.momentum is not None and not (0.0 <= self.momentum <= 1.0):
            raise ValueError(f"momentum must be in [0, 1] or None, got {momentum}.")

        if _HAS_TORCH:
            if self.affine:
                self.weight = nn.Parameter(torch.ones(self.num_features, device=device, dtype=dtype))
                self.bias = nn.Parameter(torch.zeros(self.num_features, device=device, dtype=dtype))
            else:
                self.register_parameter("weight", None)
                self.register_parameter("bias", None)
            if self.track_running_stats:
                self.register_buffer("running_mean", torch.zeros(self.num_features, device=device, dtype=dtype))
                self.register_buffer("running_mad", torch.ones(self.num_features, device=device, dtype=dtype))
                self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long, device=device))
            else:
                self.register_buffer("running_mean", None)
                self.register_buffer("running_mad", None)
                self.register_buffer("num_batches_tracked", None)
        else:
            self.weight = np.ones(self.num_features, dtype=np.float32) if (self.affine and _HAS_NUMPY) else None
            self.bias = np.zeros(self.num_features, dtype=np.float32) if (self.affine and _HAS_NUMPY) else None
            self.running_mean = np.zeros(self.num_features, dtype=np.float32) if (self.track_running_stats and _HAS_NUMPY) else None
            self.running_mad = np.ones(self.num_features, dtype=np.float32) if (self.track_running_stats and _HAS_NUMPY) else None
            self.num_batches_tracked = 0 if (self.track_running_stats and _HAS_NUMPY) else None

    def forward(self, x: Any) -> Any:
        # PyTorch execution path
        if _HAS_TORCH and isinstance(x, torch.Tensor):
            if x.ndim != 4:
                raise ValueError(f"RaBatchNorm2d expects a 4D tensor (B, C, H, W), got {x.ndim}D.")
            if x.shape[1] != self.num_features:
                raise ValueError(f"Expected {self.num_features} channels, got {x.shape[1]}.")
            use_batch_stats = bool(self.training or not self.track_running_stats)
            if not use_batch_stats and (self.running_mean is None or self.running_mad is None):
                raise RuntimeError("Running statistics not tracked in eval mode.")

            if (
                _HAS_TRITON and not _is_torch_compiling() and x.is_cuda
                and x.dtype in (torch.float16, torch.bfloat16, torch.float32) and x.numel() > 0
            ):
                needs_grad = torch.is_grad_enabled() and (
                    x.requires_grad
                    or (self.weight is not None and self.weight.requires_grad)
                    or (self.bias is not None and self.bias.requires_grad)
                )
                if needs_grad and _RaBatchNorm2dFunction is not None:
                    return _RaBatchNorm2dFunction.apply(
                        x, self.weight, self.bias, self.running_mean, self.running_mad,
                        self.num_batches_tracked, use_batch_stats, self.track_running_stats,
                        self.momentum, self.eps,
                    )
                triton_res = triton_ra_batch_norm2d(
                    x, self.weight, self.bias, self.running_mean, self.running_mad,
                    training=use_batch_stats, eps=self.eps,
                )
                if triton_res is not None:
                    y, mean_f32, mad_f32, _ = triton_res
                    if use_batch_stats and self.track_running_stats:
                        _update_bn2d_running_stats(
                            self.running_mean, self.running_mad, self.num_batches_tracked,
                            mean_f32, mad_f32, self.momentum,
                        )
                    return y

            y, mean_1d, mad_1d, _ = _ra_bn2d_eager_fwd(
                x, self.weight, self.bias, self.running_mean, self.running_mad, use_batch_stats, self.eps
            )
            if use_batch_stats and self.track_running_stats:
                _update_bn2d_running_stats(
                    self.running_mean, self.running_mad, self.num_batches_tracked, mean_1d, mad_1d, self.momentum
                )
            return y

        # NumPy execution path
        if _HAS_NUMPY and isinstance(x, np.ndarray):
            if x.ndim != 4:
                raise ValueError(f"RaBatchNorm2d expects a 4D array (B, C, H, W), got {x.ndim}D.")

            if self.training or not self.track_running_stats:
                mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
                mad = np.mean(np.abs(x - mean), axis=(0, 2, 3), keepdims=True)
                if self.track_running_stats and self.running_mean is not None and self.running_mad is not None:
                    self.num_batches_tracked += 1
                    exp_avg = self.momentum if self.momentum is not None else 1.0 / float(self.num_batches_tracked)
                    if _HAS_TORCH and isinstance(self.running_mean, torch.Tensor):
                        with torch.no_grad():
                            mean_t = torch.as_tensor(mean.squeeze(), dtype=self.running_mean.dtype, device=self.running_mean.device)
                            mad_t = torch.as_tensor(mad.squeeze(), dtype=self.running_mad.dtype, device=self.running_mad.device)
                            self.running_mean.lerp_(mean_t, exp_avg)
                            self.running_mad.lerp_(mad_t, exp_avg)
                    else:
                        self.running_mean = (1.0 - exp_avg) * self.running_mean + exp_avg * mean.squeeze()
                        self.running_mad = (1.0 - exp_avg) * self.running_mad + exp_avg * mad.squeeze()
            else:
                mean = _as_numpy(self.running_mean).reshape(1, -1, 1, 1)
                mad = _as_numpy(self.running_mad).reshape(1, -1, 1, 1)

            normed = (x - mean) / (mad + self.eps)
            w_np = _as_numpy(self.weight)
            b_np = _as_numpy(self.bias)
            if w_np is not None and b_np is not None:
                return normed * w_np.reshape(1, -1, 1, 1) + b_np.reshape(1, -1, 1, 1)
            return normed

        raise TypeError(f"RaBatchNorm2d expects torch.Tensor or numpy.ndarray, got {type(x)}")

    def fold_into_conv(self, conv: Any) -> Tuple[Any, Any]:
        """Fold running normalization stats into a preceding Conv2d layer."""
        if self.running_mean is None or self.running_mad is None:
            raise RuntimeError("Cannot fold running statistics: layer has no running stats.")

        if _HAS_TORCH and isinstance(conv, nn.Conv2d):
            if conv.out_channels != self.num_features:
                raise ValueError(f"Expected Conv2d with out_channels={self.num_features}, got {conv.out_channels}.")
            with torch.no_grad():
                rm, rmad = self.running_mean.float(), self.running_mad.float()
                gamma = self.weight.float() if self.weight is not None else torch.ones_like(rm)
                beta = self.bias.float() if self.bias is not None else torch.zeros_like(rm)
                scale = gamma / (rmad + self.eps)
                w_conv = _safe_cast(conv.weight.float() * scale.view(-1, 1, 1, 1), conv.weight.dtype)
                b_raw = (conv.bias.float() - rm) if conv.bias is not None else -rm
                b_conv = _safe_cast(b_raw * scale + beta, conv.weight.dtype)
                return w_conv, b_conv

        if _HAS_NUMPY and hasattr(conv, "weight"):
            rm_np, rmad_np = _as_numpy(self.running_mean), _as_numpy(self.running_mad)
            w_np, b_np = _as_numpy(self.weight), _as_numpy(self.bias)
            gamma = w_np if w_np is not None else np.ones_like(rm_np)
            beta = b_np if b_np is not None else np.zeros_like(rm_np)
            scale = gamma / (rmad_np + self.eps)
            w_conv = conv.weight * scale.reshape(-1, 1, 1, 1)
            b_conv = ((conv.bias - rm_np) if (hasattr(conv, "bias") and conv.bias is not None) else -rm_np) * scale + beta
            return w_conv, b_conv

        raise TypeError("Expected torch.nn.Conv2d or duck-typed Conv layer with .weight and .bias.")

    def extra_repr(self) -> str:
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats}"
