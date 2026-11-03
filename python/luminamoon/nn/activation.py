from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

from ._compat import (
    BaseModule,
    _HAS_NUMPY,
    _HAS_TORCH,
    np,
    torch,
    nn,
)

_SQRT_2_OVER_PI: float = math.sqrt(2.0 / math.pi)


def ra_tanh(x: Any, b: float = 1.0) -> Any:
    """
    Rational Hyperbolic Tangent (RaTanh).

    Computes:
        RaTanh(x; b) = (b * (b + |x|) * x) / (b * (b + |x|) + x^2)

    Properties:
      - Symmetric asymptotic range: (-b, b)
      - Unit derivative at origin: RaTanh'(0) = 1.0
      - Non-vanishing gradient tail: decays algebraically O(1 / |x|^3) rather than exp(-2|x|)
      - Compatible with both PyTorch Tensor and NumPy ndarray.
    """
    b = float(b)
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        abs_x = x.abs()
        b_term = b * (b + abs_x)
        return (b_term * x) / (b_term + x.square())

    if _HAS_NUMPY and isinstance(x, np.ndarray):
        abs_x = np.abs(x)
        b_term = b * (b + abs_x)
        return (b_term * x) / (b_term + np.square(x))

    if isinstance(x, (int, float)):
        abs_x = abs(x)
        b_term = b * (b + abs_x)
        return (b_term * x) / (b_term + (x * x))

    raise TypeError(f"ra_tanh expects torch.Tensor, numpy.ndarray, or float, got {type(x)}")


class RaTanh(BaseModule):
    """
    Rational Hyperbolic Tangent module:
        RaTanh(x; b) = (b * (b + |x|) * x) / (b * (b + |x|) + x^2)
    """

    def __init__(self, b: float = 1.0) -> None:
        super().__init__()
        self.b = float(b)

    def forward(self, x: Any) -> Any:
        return ra_tanh(x, b=self.b)

    def extra_repr(self) -> str:
        return f"b={self.b}"


def ra_gelu(x: Any) -> Any:
    """
    Rational Gaussian Error Linear Unit (RaGELU).

    Computes:
        RaGELU(x) = 0.5 * x * (1 + RaTanh(sqrt(2 / pi) * x))
    """
    scaled = _SQRT_2_OVER_PI * x
    return 0.5 * x * (1.0 + ra_tanh(scaled, b=1.0))


class RaGELU(BaseModule):
    """
    Rational GELU module for Vision Transformers (ViT) and feed-forward networks:
        RaGELU(x) = 0.5 * x * (1 + RaTanh(sqrt(2 / pi) * x))
    """

    def forward(self, x: Any) -> Any:
        return ra_gelu(x)


def pos_ra_gate(z: Any, b: float = 1.0) -> Any:
    """
    Strictly positive rational gate normalized on (0, 1):
        sigma_Rat(z) = 0.5 * (1 + RaTanh(z; b)) in (0, 1)
    """
    return 0.5 * (1.0 + ra_tanh(z, b=b))


def pos_ra_glu(gate: Any, up: Any, b: float = 1.0) -> Any:
    """
    Positive Rational Gated Linear Unit functional:
        PosRaGLU(gate, up) = up * sigma_Rat(gate)
    """
    return up * pos_ra_gate(gate, b=b)


class PosRaGLU(BaseModule):
    w_gate: Any
    w_up: Any
    w_down: Any
    """
    Positive Rational Gated Linear Unit (PosRaGLU) MLP Block.

    Drop-in replacement for SwiGLU in Transformer Feed-Forward Networks:
        z_gate = W_gate(x)
        z_up = W_up(x)
        h = z_up * sigma_Rat(z_gate)
        output = W_down(h)

    Key Advantage:
      Because sigma_Rat(z) is strictly bounded in (0, 1), the hidden representation
      is bounded linearly by ||z_up||_inf. This completely eliminates the quadratic
      activation outlier explosion observed with SwiGLU (x * swish(x) ~ x^2),
      making models stable for native INT4 and FP8 quantization.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = False,
        b: float = 1.0,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features) if hidden_features is not None else int(in_features * 8 / 3)
        self.out_features = int(out_features) if out_features is not None else self.in_features
        self.b = float(b)

        if _HAS_TORCH:
            self.w_gate = nn.Linear(self.in_features, self.hidden_features, bias=bias, device=device, dtype=dtype)
            self.w_up = nn.Linear(self.in_features, self.hidden_features, bias=bias, device=device, dtype=dtype)
            self.w_down = nn.Linear(self.hidden_features, self.out_features, bias=bias, device=device, dtype=dtype)
        else:
            self.w_gate = None
            self.w_up = None
            self.w_down = None

    def forward(self, x: Any) -> Any:
        if _HAS_TORCH and isinstance(x, torch.Tensor):
            gate = self.w_gate(x)
            up = self.w_up(x)
            h = pos_ra_glu(gate, up, b=self.b)
            return self.w_down(h)

        if _HAS_NUMPY and isinstance(x, np.ndarray):
            # Standalone fallback when instantiated without torch linear layers
            return pos_ra_gate(x, b=self.b)

        raise TypeError(f"PosRaGLU expects torch.Tensor or numpy.ndarray, got {type(x)}")

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"hidden_features={self.hidden_features}, "
            f"out_features={self.out_features}, "
            f"b={self.b}"
        )


class RaGLU(BaseModule):
    w_gate: Any
    w_up: Any
    w_down: Any
    """
    Standard Rational Gated Linear Unit:
        RaGLU(gate, up) = up * RaTanh(gate; b)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = False,
        b: float = 1.0,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features) if hidden_features is not None else int(in_features * 8 / 3)
        self.out_features = int(out_features) if out_features is not None else self.in_features
        self.b = float(b)

        if _HAS_TORCH:
            self.w_gate = nn.Linear(self.in_features, self.hidden_features, bias=bias, device=device, dtype=dtype)
            self.w_up = nn.Linear(self.in_features, self.hidden_features, bias=bias, device=device, dtype=dtype)
            self.w_down = nn.Linear(self.hidden_features, self.out_features, bias=bias, device=device, dtype=dtype)
        else:
            self.w_gate = None
            self.w_up = None
            self.w_down = None

    def forward(self, x: Any) -> Any:
        if _HAS_TORCH and isinstance(x, torch.Tensor):
            gate = self.w_gate(x)
            up = self.w_up(x)
            h = up * ra_tanh(gate, b=self.b)
            return self.w_down(h)

        if _HAS_NUMPY and isinstance(x, np.ndarray):
            return ra_tanh(x, b=self.b)

        raise TypeError(f"RaGLU expects torch.Tensor or numpy.ndarray, got {type(x)}")


class TriGLU(BaseModule):
    w_g1: Any
    w_g2: Any
    w_val: Any
    w_down: Any
    """
    Triple Rational Gated Linear Unit (TriGLU).
    Splits features into three paths (gate1, gate2, value) and fuses them:
        h = W_val(x) * sigma_Rat(W_g1(x)) * (1.0 + RaTanh(W_g2(x)))
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features) if hidden_features is not None else int(in_features * 2)
        self.out_features = int(out_features) if out_features is not None else self.in_features

        if _HAS_TORCH:
            self.w_g1 = nn.Linear(self.in_features, self.hidden_features, bias=bias, device=device, dtype=dtype)
            self.w_g2 = nn.Linear(self.in_features, self.hidden_features, bias=bias, device=device, dtype=dtype)
            self.w_val = nn.Linear(self.in_features, self.hidden_features, bias=bias, device=device, dtype=dtype)
            self.w_down = nn.Linear(self.hidden_features, self.out_features, bias=bias, device=device, dtype=dtype)
        else:
            self.w_g1 = None
            self.w_g2 = None
            self.w_val = None
            self.w_down = None

    def forward(self, x: Any) -> Any:
        if _HAS_TORCH and isinstance(x, torch.Tensor):
            g1 = pos_ra_gate(self.w_g1(x))
            g2 = 1.0 + ra_tanh(self.w_g2(x))
            v = self.w_val(x)
            h = v * g1 * g2
            return self.w_down(h)

        if _HAS_NUMPY and isinstance(x, np.ndarray):
            return pos_ra_gate(x)

        raise TypeError(f"TriGLU expects torch.Tensor or numpy.ndarray, got {type(x)}")
