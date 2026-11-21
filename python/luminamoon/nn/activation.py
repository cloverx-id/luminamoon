from __future__ import annotations

import math
from typing import Any, Optional, Tuple

from ._compat import (
    BaseModule,
    _HAS_NUMPY,
    _HAS_RUST_SIMD,
    _HAS_TORCH,
    _HAS_TRITON,
    _is_torch_compiling,
    _rs_engine,
    np,
    torch,
    nn,
)
from .kernels import triton_pos_ra_glu, triton_pos_ra_glu_backward

_SQRT_2_OVER_PI: float = math.sqrt(2.0 / math.pi)
_PosRaGLUFunction: Any = None

if _HAS_TORCH and torch is not None:

    class _PosRaGLUFunctionImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, gate: Any, up: Any, b: float) -> Any:
            ctx.b = float(b)
            ctx.save_for_backward(gate, up)
            out = triton_pos_ra_glu(gate, up, b=ctx.b)
            if out is not None:
                return out
            abs_z = gate.abs()
            b_term = ctx.b * (ctx.b + abs_z)
            ra_t = (b_term * gate) / (b_term + gate.square())
            sigma = 0.5 * (1.0 + ra_t) if ctx.b == 1.0 else 0.5 * (1.0 + ra_t / ctx.b)
            return up * sigma

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Tuple[Any, Any, None]:  # type: ignore[override]
            gate, up = ctx.saved_tensors
            b_val: float = ctx.b
            triton_res = triton_pos_ra_glu_backward(grad_out, gate, up, b=b_val)
            if triton_res is not None:
                dgate, dup = triton_res
                return (
                    dgate.reshape(gate.shape) if ctx.needs_input_grad[0] else None,
                    dup.reshape(up.shape) if ctx.needs_input_grad[1] else None,
                    None,
                )

            z = gate.to(torch.float32) if gate.dtype in (torch.float16, torch.bfloat16) else gate
            u = up.to(torch.float32) if up.dtype in (torch.float16, torch.bfloat16) else up
            g = grad_out.to(torch.float32) if grad_out.dtype in (torch.float16, torch.bfloat16) else grad_out
            abs_z = z.abs()
            b_term = b_val * (b_val + abs_z)
            denom = b_term + z.square()
            ra_t = (b_term * z) / denom
            sigma = 0.5 * (1.0 + ra_t) if b_val == 1.0 else 0.5 * (1.0 + ra_t / b_val)
            d_sigma = (0.5 * (b_val * b_val) * (b_val + 2.0 * abs_z)) / denom.square()
            dgate = (g * u * d_sigma).to(gate.dtype) if ctx.needs_input_grad[0] else None
            dup = (g * sigma).to(up.dtype) if ctx.needs_input_grad[1] else None
            return dgate, dup, None

    _PosRaGLUFunction = _PosRaGLUFunctionImpl


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
      - Accelerated via Rust AVX-512/AVX2 SIMD when available on CPU.
    """
    b = float(b)

    # 1. PyTorch Tensor path
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        if (
            _HAS_RUST_SIMD
            and not _is_torch_compiling()
            and not x.is_cuda
            and not x.requires_grad
            and x.dtype == torch.float32
            and x.is_contiguous()
        ):
            out = torch.empty_like(x)
            _rs_engine.ra_tanh_f32_pyo3(x, out, b)
            return out

        abs_x = x.abs()
        b_term = b * (b + abs_x)
        return (b_term * x) / (b_term + x.square())

    # 2. NumPy ndarray path
    if _HAS_NUMPY and isinstance(x, np.ndarray):
        if _HAS_RUST_SIMD and x.dtype == np.float32 and x.flags.c_contiguous:
            out = np.empty_like(x)
            _rs_engine.ra_tanh_f32_pyo3(x, out, b)
            return out

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
        sigma_Rat(z; b) = 0.5 * (1 + RaTanh(z; b) / b) in (0, 1)
        (For default unit anchor b=1: sigma_Rat(z) = 0.5 * (1 + RaTanh(z; 1))).
    """
    b = float(b)
    if b == 1.0:
        return 0.5 * (1.0 + ra_tanh(z, b=1.0))
    return 0.5 * (1.0 + ra_tanh(z, b=b) / b)


def pos_ra_glu(gate: Any, up: Any, b: float = 1.0) -> Any:
    """
    Positive Rational Gated Linear Unit functional:
        PosRaGLU(gate, up) = up * sigma_Rat(gate)
    """
    b_val = float(b)
    if (
        _HAS_TORCH
        and _HAS_TRITON
        and not _is_torch_compiling()
        and isinstance(gate, torch.Tensor)
        and isinstance(up, torch.Tensor)
        and gate.is_cuda
        and up.is_cuda
        and gate.shape == up.shape
    ):
        if torch.is_grad_enabled() and (gate.requires_grad or up.requires_grad) and _PosRaGLUFunction is not None:
            return _PosRaGLUFunction.apply(gate, up, b_val)
        triton_out = triton_pos_ra_glu(gate, up, b=b_val)
        if triton_out is not None:
            return triton_out
    return up * pos_ra_gate(gate, b=b_val)


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

            # Triton GPU fast-path (both training autograd and inference)
            if _HAS_TRITON and not _is_torch_compiling() and gate.is_cuda:
                h = pos_ra_glu(gate, up, b=self.b)
            # Rust CPU SIMD fast-path
            elif (
                _HAS_RUST_SIMD
                and not _is_torch_compiling()
                and not gate.is_cuda
                and not x.requires_grad
                and gate.dtype == torch.float32
                and gate.is_contiguous()
                and up.is_contiguous()
            ):
                h = torch.empty_like(gate)
                _rs_engine.pos_ra_glu_f32_pyo3(gate, up, h, self.b)
            else:
                h = pos_ra_glu(gate, up, b=self.b)

            return self.w_down(h)

        if _HAS_NUMPY and isinstance(x, np.ndarray):
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
