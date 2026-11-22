from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

from ._compat import (
    BaseModule,
    _HAS_NUMPY,
    _HAS_RUST_SIMD,
    _HAS_TORCH,
    _HAS_TRITON,
    _as_numpy,
    _can_use_cubecl_nn,
    _is_torch_compiling,
    _rs_engine,
    np,
    torch,
    nn,
)
from ._custom_ops import _HAS_CUSTOM_OP, compiled_ra_norm
from .kernels import triton_ra_norm, triton_ra_norm_backward

_RaNormFunction: Any = None

if _HAS_TORCH and torch is not None:

    class _RaNormFunctionImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any, weight: Optional[Any], bias: Optional[Any], eps: float) -> Any:
            ctx.eps = float(eps)
            ctx.has_weight = weight is not None
            ctx.has_bias = bias is not None
            saved_w = weight if weight is not None else torch.empty(0, device=x.device, dtype=x.dtype)
            triton_res = triton_ra_norm(x, weight, bias, eps=ctx.eps, return_inv_denom=True)
            if triton_res is not None:
                out, inv_denom = triton_res
                ctx.save_for_backward(x, saved_w, inv_denom)
                return out
            ctx.save_for_backward(x, saved_w)
            l1_mean = x.abs().mean(dim=-1, keepdim=True)
            normed = x / (l1_mean + ctx.eps)
            if weight is not None:
                normed = normed * weight
            if bias is not None:
                normed = normed + bias
            return normed

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
            dx = (inv_denom * (dy_w - torch.sign(x_f32) * c)).to(x.dtype)

            reduce_dims = tuple(range(x.ndim - 1))
            dw = (
                (gy_f32 * x_hat).sum(dim=reduce_dims).to(weight.dtype)
                if (weight is not None and ctx.needs_input_grad[1])
                else None
            )
            db = (
                gy_f32.sum(dim=reduce_dims).to(weight.dtype if weight is not None else x.dtype)
                if (ctx.has_bias and ctx.needs_input_grad[2])
                else None
            )
            return dx, dw, db, None

    _RaNormFunction = _RaNormFunctionImpl


class RaNorm(BaseModule):
    """
    Rational Normalization (RaNorm) for Neural Networks.

    Replaces LayerNorm and RMSNorm by substituting the L2 root-mean-square (sqrt)
    operation with the L1 mean absolute deviation:
        mu_|x| = (1 / d) * sum_{i=1}^d |x_i|
        RaNorm(x) = (x / (mu_|x| + eps)) * gamma + beta

    Characteristics:
      1. Square-Root-Free: Avoids square-root and inverse-square-root
         computations (e.g., MUFU.RSQ on GPUs/NPUs).
      2. Single-Reciprocal Amortization: The scalar reciprocal 1 / (mu_|x| + eps)
         is computed once per token vector and broadcast across all d dimensions.
      3. Scale-Homogeneous: Satisfies RaNorm(alpha * x) = sgn(alpha) * RaNorm(x).
      4. Auto-Detecting: Inherits from torch.nn.Module when PyTorch is available,
         or StandaloneModule when running in a NumPy-only environment.
      5. Hardware Acceleration: Dispatches to Triton JIT on CUDA GPUs and Rust
         SIMD on CPUs when available.

    Pairing Scope:
      Works well on dense Transformer activations (such as RaSwiGLU, RaGLU, RaGELU,
      and RaTanh). Unlike L2-based RMSNorm (whose output magnitude is bounded
      by sqrt(D)), L1 scaling can amplify an isolated spike up to O(D) on
      highly sparse vectors. Avoid placing RaNorm directly after ultra-sparse
      activations with extreme isolated outliers.
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
            self.normalized_shape = tuple(normalized_shape)

        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)

        if _HAS_TORCH:
            if self.elementwise_affine:
                self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
                if bias:
                    self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=device, dtype=dtype))
                else:
                    self.register_parameter("bias", None)
            else:
                self.register_parameter("weight", None)
                self.register_parameter("bias", None)
        else:
            if self.elementwise_affine and _HAS_NUMPY:
                self.weight = np.ones(self.normalized_shape, dtype=np.float32)
                self.bias = np.zeros(self.normalized_shape, dtype=np.float32) if bias else None
            else:
                self.weight = None
                self.bias = None

    def forward(self, x: Any) -> Any:
        # 1. PyTorch Tensor execution path
        if _HAS_TORCH and isinstance(x, torch.Tensor):
            # 1a. GPU fast-path via Triton JIT (eager default or custom_op under torch.compile)
            if (
                _HAS_TRITON
                and x.is_cuda
                and len(self.normalized_shape) == 1
                and x.ndim >= 1
                and 0 < x.shape[-1] <= 8192
            ):
                if _is_torch_compiling():
                    if _HAS_CUSTOM_OP:
                        return compiled_ra_norm(x, self.weight, self.bias, self.eps)
                else:
                    needs_grad = torch.is_grad_enabled() and (
                        x.requires_grad
                        or (self.weight is not None and self.weight.requires_grad)
                        or (self.bias is not None and self.bias.requires_grad)
                    )
                    if needs_grad and _RaNormFunction is not None:
                        return _RaNormFunction.apply(x, self.weight, self.bias, self.eps)
                    triton_out = triton_ra_norm(x, self.weight, self.bias, eps=self.eps)
                    if triton_out is not None:
                        return triton_out

            # 1b. CPU fast-path via Rust AVX2/AVX-512 SIMD (1D and ND contiguous tensors)
            needs_grad_cpu = torch.is_grad_enabled() and (
                x.requires_grad
                or (self.weight is not None and self.weight.requires_grad)
                or (self.bias is not None and self.bias.requires_grad)
            )
            if (
                _HAS_RUST_SIMD
                and not _is_torch_compiling()
                and x.device.type == "cpu"
                and not needs_grad_cpu
                and x.dtype == torch.float32
                and x.is_contiguous()
                and len(self.normalized_shape) == 1
                and x.ndim >= 1
                and x.shape[-1] == self.normalized_shape[0]
                and (
                    self.weight is None
                    or (self.weight.dtype == torch.float32 and self.weight.is_contiguous() and self.weight.device.type == "cpu")
                )
                and (
                    self.bias is None
                    or (self.bias.dtype == torch.float32 and self.bias.is_contiguous() and self.bias.device.type == "cpu")
                )
            ):
                out = torch.empty_like(x)
                fn = _rs_engine.ra_norm_cubecl_f32_pyo3 if _can_use_cubecl_nn(x.numel(), "ra_norm_cubecl_f32_pyo3") else _rs_engine.ra_norm_f32_pyo3
                fn(x, out, self.weight, self.bias, self.eps, x.shape[-1])
                return out

            # 1c. Standard autograd execution path
            dims = tuple(range(-len(self.normalized_shape), 0))
            l1_mean = x.abs().mean(dim=dims, keepdim=True)
            normed = x / (l1_mean + self.eps)
            if self.weight is not None:
                normed = normed * self.weight
            if self.bias is not None:
                normed = normed + self.bias
            return normed

        # 2. NumPy Array execution path (zero PyTorch dependency)
        if _HAS_NUMPY and isinstance(x, np.ndarray):
            w_np = _as_numpy(self.weight)
            b_np = _as_numpy(self.bias)

            # 2a. CPU fast-path via Rust AVX2/AVX-512 SIMD (1D and ND contiguous arrays)
            if (
                _HAS_RUST_SIMD
                and not _is_torch_compiling()
                and x.dtype == np.float32
                and x.flags.c_contiguous
                and len(self.normalized_shape) == 1
                and x.ndim >= 1
                and x.shape[-1] == self.normalized_shape[0]
                and (w_np is None or (w_np.dtype == np.float32 and w_np.flags.c_contiguous))
                and (b_np is None or (b_np.dtype == np.float32 and b_np.flags.c_contiguous))
            ):
                out = np.empty_like(x)
                fn = _rs_engine.ra_norm_cubecl_f32_pyo3 if _can_use_cubecl_nn(x.size, "ra_norm_cubecl_f32_pyo3") else _rs_engine.ra_norm_f32_pyo3
                fn(x, out, w_np, b_np, self.eps, x.shape[-1])
                return out

            # 2b. Standard NumPy execution path
            axis = tuple(range(-len(self.normalized_shape), 0))
            l1_mean = np.mean(np.abs(x), axis=axis, keepdims=True)
            normed = x / (l1_mean + self.eps)
            if w_np is not None:
                normed = normed * w_np
            if b_np is not None:
                normed = normed + b_np
            return normed

        raise TypeError(f"RaNorm expects torch.Tensor or numpy.ndarray, got {type(x)}")

    def extra_repr(self) -> str:
        return (
            f"{self.normalized_shape}, "
            f"eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}, "
            f"bias={self.bias is not None}"
        )


class RaBatchNorm2d(BaseModule):
    weight: Any
    bias: Any
    running_mean: Any
    running_mad: Any
    num_batches_tracked: Any
    """
    Rational 2D Batch Normalization (RaBatchNorm2d) for Vision & Edge Models.

    Replaces standard BatchNorm2d variance and square-root operations with
    cross-batch L1 Mean Absolute Deviation (MAD):
        mu_B = (1 / m) * sum x_i
        MAD_B = (1 / m) * sum |x_i - mu_B|
        y = ((x - mu_B) / (MAD_B + eps)) * gamma + beta

    Characteristics:
      1. Zero-Square-Root: Eliminates square-root evaluation.
      2. Low-Precision Safety: Avoids variance squaring (x - mu)^2 which can
         overflow FP16 (max 65504).
      3. Inference Weight-Folding: Running statistics fold cleanly into preceding
         Conv2d weights with zero inference latency overhead.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum: Optional[float] = float(momentum) if momentum is not None else None
        self.affine = bool(affine)
        self.track_running_stats = bool(track_running_stats)

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
            if self.affine and _HAS_NUMPY:
                self.weight = np.ones(self.num_features, dtype=np.float32)
                self.bias = np.zeros(self.num_features, dtype=np.float32)
            else:
                self.weight = None
                self.bias = None
            if self.track_running_stats and _HAS_NUMPY:
                self.running_mean = np.zeros(self.num_features, dtype=np.float32)
                self.running_mad = np.ones(self.num_features, dtype=np.float32)
                self.num_batches_tracked = 0
            else:
                self.running_mean = None
                self.running_mad = None
                self.num_batches_tracked = None

    def forward(self, x: Any) -> Any:
        # PyTorch execution path
        if _HAS_TORCH and isinstance(x, torch.Tensor):
            if x.ndim != 4:
                raise ValueError(f"RaBatchNorm2d expects a 4D tensor (B, C, H, W), got {x.ndim}D.")

            if self.training or not self.track_running_stats:
                mean = x.mean(dim=(0, 2, 3), keepdim=True)
                mad = (x - mean).abs().mean(dim=(0, 2, 3), keepdim=True)

                if self.track_running_stats and self.running_mean is not None and self.running_mad is not None:
                    with torch.no_grad():
                        n = self.num_batches_tracked
                        if n is not None:
                            n += 1
                            exp_avg_factor = self.momentum if self.momentum is not None else 1.0 / float(n)
                        else:
                            exp_avg_factor = self.momentum if self.momentum is not None else 1.0

                        self.running_mean.lerp_(mean.squeeze(), exp_avg_factor)
                        self.running_mad.lerp_(mad.squeeze(), exp_avg_factor)
            else:
                if self.running_mean is None or self.running_mad is None:
                    raise RuntimeError("Running statistics not tracked in eval mode.")
                mean = self.running_mean.view(1, -1, 1, 1)
                mad = self.running_mad.view(1, -1, 1, 1)

            normed = (x - mean) / (mad + self.eps)
            if self.weight is not None and self.bias is not None:
                return normed * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
            return normed

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
            gamma = self.weight if self.weight is not None else torch.ones_like(self.running_mean)
            beta = self.bias if self.bias is not None else torch.zeros_like(self.running_mean)
            scale = gamma / (self.running_mad + self.eps)
            w_conv = conv.weight * scale.view(-1, 1, 1, 1)
            if conv.bias is not None:
                b_conv = (conv.bias - self.running_mean) * scale + beta
            else:
                b_conv = -self.running_mean * scale + beta
            return w_conv, b_conv

        if _HAS_NUMPY and hasattr(conv, "weight"):
            rm_np = _as_numpy(self.running_mean)
            rmad_np = _as_numpy(self.running_mad)
            w_np = _as_numpy(self.weight)
            b_np = _as_numpy(self.bias)
            gamma = w_np if w_np is not None else np.ones_like(rm_np)
            beta = b_np if b_np is not None else np.zeros_like(rm_np)
            scale = gamma / (rmad_np + self.eps)
            w_conv = conv.weight * scale.reshape(-1, 1, 1, 1)
            if hasattr(conv, "bias") and conv.bias is not None:
                b_conv = (conv.bias - rm_np) * scale + beta
            else:
                b_conv = -rm_np * scale + beta
            return w_conv, b_conv

        raise TypeError("Expected torch.nn.Conv2d or duck-typed Conv layer with .weight and .bias.")

    def extra_repr(self) -> str:
        return (
            f"{self.num_features}, "
            f"eps={self.eps}, "
            f"momentum={self.momentum}, "
            f"affine={self.affine}, "
            f"track_running_stats={self.track_running_stats}"
        )
