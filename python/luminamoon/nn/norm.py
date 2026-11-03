from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

from ._compat import (
    BaseModule,
    _HAS_NUMPY,
    _HAS_TORCH,
    np,
    torch,
    nn,
)


class RaNorm(BaseModule):
    """
    Rational Normalization (RaNorm) for Neural Networks.

    Replaces LayerNorm and RMSNorm by substituting the L2 root-mean-square (sqrt)
    operation with the L1 mean absolute deviation:
        mu_|x| = (1 / d) * sum_{i=1}^d |x_i|
        RaNorm(x) = (x / (mu_|x| + eps)) * gamma + beta

    Characteristics:
      1. Zero Square-Root Overhead: Eliminates square-root and inverse-square-root
         computations (e.g., MUFU.RSQ on GPUs/NPUs).
      2. Single-Reciprocal Amortization: The scalar reciprocal 1 / (mu_|x| + eps)
         is computed once per token vector and broadcasted across all d dimensions.
      3. Scale-Homogeneous: Satisfies RaNorm(alpha * x) = sgn(alpha) * RaNorm(x).
      4. Auto-Detecting: Inherits from torch.nn.Module when PyTorch is available,
         or StandaloneModule when in a lightweight, zero-dependency environment.
    """

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
            axis = tuple(range(-len(self.normalized_shape), 0))
            l1_mean = np.mean(np.abs(x), axis=axis, keepdims=True)
            normed = x / (l1_mean + self.eps)
            if self.weight is not None:
                normed = normed * self.weight
            if self.bias is not None:
                normed = normed + self.bias
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
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
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
                            exp_avg_factor = self.momentum

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
                    exp_avg = self.momentum
                    self.running_mean = (1.0 - exp_avg) * self.running_mean + exp_avg * mean.squeeze()
                    self.running_mad = (1.0 - exp_avg) * self.running_mad + exp_avg * mad.squeeze()
            else:
                mean = self.running_mean.reshape(1, -1, 1, 1)
                mad = self.running_mad.reshape(1, -1, 1, 1)

            normed = (x - mean) / (mad + self.eps)
            if self.weight is not None and self.bias is not None:
                return normed * self.weight.reshape(1, -1, 1, 1) + self.bias.reshape(1, -1, 1, 1)
            return normed

        raise TypeError(f"RaBatchNorm2d expects torch.Tensor or numpy.ndarray, got {type(x)}")

    def fold_into_conv(self, conv: Any) -> Tuple[Any, Any]:
        """
        Folds running normalization parameters directly into a preceding Conv2d layer
        for zero-overhead edge and hardware inference deployment.
        """
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
            gamma = self.weight if self.weight is not None else np.ones_like(self.running_mean)
            beta = self.bias if self.bias is not None else np.zeros_like(self.running_mean)
            scale = gamma / (self.running_mad + self.eps)
            w_conv = conv.weight * scale.reshape(-1, 1, 1, 1)
            b = getattr(conv, "bias", None)
            if b is not None:
                b_conv = (b - self.running_mean) * scale + beta
            else:
                b_conv = -self.running_mean * scale + beta
            return w_conv, b_conv

        raise NotImplementedError("fold_into_conv requires PyTorch nn.Conv2d or NumPy layer.")

    def extra_repr(self) -> str:
        return (
            f"{self.num_features}, "
            f"eps={self.eps}, "
            f"momentum={self.momentum}, "
            f"affine={self.affine}, "
            f"track_running_stats={self.track_running_stats}"
        )
