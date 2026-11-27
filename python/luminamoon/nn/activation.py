from __future__ import annotations

import math
from typing import Any, Optional, Tuple

from ._compat import (
    BaseModule, _HAS_NUMPY, _HAS_RUST_SIMD, _HAS_TORCH, _HAS_TRITON,
    _can_use_cubecl_nn, _init_linear_np, _is_torch_compiling, _linear_np, _rs_engine, nn, np, torch,
)
from ._simd_ops import _cpu_ra_gated_bwd, _cpu_ra_gated_fwd, _cpu_ra_tanh_bwd, _cpu_ra_tanh_fwd
from .kernels import (
    triton_ra_gelu, triton_ra_gelu_backward, triton_ra_glu, triton_ra_glu_backward,
    triton_ra_swiglu, triton_ra_swiglu_backward, triton_ra_tanh, triton_ra_tanh_backward,
)

_SQRT_2_OVER_PI: float = math.sqrt(2.0 / math.pi)
_RaTanhFunction: Any = None
_RaGELUFunction: Any = None
_RaSwiGLUFunction: Any = None
_RaGLUFunction: Any = None


def _validate_positive_b(b: float) -> float:
    b_val = float(b)
    if not (math.isfinite(b_val) and b_val > 0.0):
        raise ValueError(f"b must be finite and > 0, got {b}.")
    return b_val


def _ra_norm_terms_torch(z: Any, b: float) -> Tuple[Any, Any, Any, Any]:
    abs_z = z.abs()
    inv_m = 1.0 / torch.clamp(abs_z, min=b)
    ub = b * inv_m
    ua = abs_z * inv_m
    uz = z * inv_m
    inv_dn = 1.0 / (ub * (ub + ua) + ua * ua)
    return ub, ua, uz, inv_dn


def _ra_tanh_torch_stable(x: Any, b: float) -> Any:
    x_w = x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
    ub, ua, uz, inv_dn = _ra_norm_terms_torch(x_w, b)
    return (b * (ub + ua) * uz * inv_dn).to(x.dtype)


def _pos_ra_gate_torch(z: Any, b: float) -> Any:
    ub, ua, _, inv_dn = _ra_norm_terms_torch(z, b)
    q = 0.5 * (ub * ub) * inv_dn
    return torch.where(z >= 0.0, 1.0 - q, q)


if _HAS_TORCH and torch is not None:

    class _RaTanhFunctionImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any, b: float) -> Any:
            ctx.b = float(b)
            ctx.save_for_backward(x)
            out = triton_ra_tanh(x, b=ctx.b)
            if out is None:
                out = _cpu_ra_tanh_fwd(x, ctx.b)
            return out if out is not None else _ra_tanh_torch_stable(x, ctx.b)

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Tuple[Any, None]:  # type: ignore[override]
            (x,) = ctx.saved_tensors
            b_val: float = ctx.b
            if not ctx.needs_input_grad[0]:
                return None, None
            dx = triton_ra_tanh_backward(grad_out, x, b=b_val)
            if dx is None:
                dx = _cpu_ra_tanh_bwd(grad_out, x, b_val)
            if dx is not None:
                return dx.reshape(x.shape), None
            x_f32 = x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
            g_f32 = grad_out.to(torch.float32) if grad_out.dtype in (torch.float16, torch.bfloat16) else grad_out
            ub, ua, _, inv_dn = _ra_norm_terms_torch(x_f32, b_val)
            deriv = (ub * ub * ub) * (ub + 2.0 * ua) * (inv_dn * inv_dn)
            return (g_f32 * deriv).to(x.dtype), None

    _RaTanhFunction = _RaTanhFunctionImpl

    class _RaSwiGLUFunctionImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, gate: Any, up: Any, b: float) -> Any:
            ctx.b = float(b)
            ctx.save_for_backward(gate, up)
            out = triton_ra_swiglu(gate, up, b=ctx.b)
            if out is None:
                out = _cpu_ra_gated_fwd(gate, up, ctx.b, is_swiglu=True)
            if out is not None:
                return out
            z = gate.to(torch.float32) if gate.dtype in (torch.float16, torch.bfloat16) else gate
            u = up.to(torch.float32) if up.dtype in (torch.float16, torch.bfloat16) else up
            ub, ua, uz, inv_dn = _ra_norm_terms_torch(z, ctx.b)
            q = 0.5 * (ub * ub) * inv_dn
            s = torch.where(z >= 0.0, z * (1.0 - q), (0.5 * ctx.b * ub) * uz * inv_dn)
            return (s * u).to(gate.dtype)

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Tuple[Any, Any, None]:  # type: ignore[override]
            gate, up = ctx.saved_tensors
            b_val: float = ctx.b
            res = triton_ra_swiglu_backward(grad_out, gate, up, b=b_val)
            if res is not None:
                return (res[0].reshape(gate.shape) if ctx.needs_input_grad[0] else None), (res[1].reshape(up.shape) if ctx.needs_input_grad[1] else None), None
            cpu_res = _cpu_ra_gated_bwd(grad_out, gate, up, b_val, is_swiglu=True, need_dg=bool(ctx.needs_input_grad[0]), need_du=bool(ctx.needs_input_grad[1]))
            if cpu_res is not None:
                return cpu_res[0], cpu_res[1], None

            z = gate.to(torch.float32) if gate.dtype in (torch.float16, torch.bfloat16) else gate
            u = up.to(torch.float32) if up.dtype in (torch.float16, torch.bfloat16) else up
            g = grad_out.to(torch.float32) if grad_out.dtype in (torch.float16, torch.bfloat16) else grad_out
            ub, ua, uz, inv_dn = _ra_norm_terms_torch(z, b_val)
            q = 0.5 * (ub * ub) * inv_dn
            is_pos = z >= 0.0
            sigma = torch.where(is_pos, 1.0 - q, q)
            s = torch.where(is_pos, z * sigma, (0.5 * b_val * ub) * uz * inv_dn)
            ds_dz = torch.where(is_pos, sigma + q * ua * (ub + 2.0 * ua) * inv_dn, q * (ub * ub - ua * ua) * inv_dn)
            return ((g * u * ds_dz).to(gate.dtype) if ctx.needs_input_grad[0] else None), ((g * s).to(up.dtype) if ctx.needs_input_grad[1] else None), None

    _RaSwiGLUFunction = _RaSwiGLUFunctionImpl

    class _RaGLUFunctionImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, gate: Any, up: Any, b: float) -> Any:
            ctx.b = float(b)
            ctx.save_for_backward(gate, up)
            out = triton_ra_glu(gate, up, b=ctx.b)
            if out is None:
                out = _cpu_ra_gated_fwd(gate, up, ctx.b, is_swiglu=False)
            if out is not None:
                return out
            z = gate.to(torch.float32) if gate.dtype in (torch.float16, torch.bfloat16) else gate
            u = up.to(torch.float32) if up.dtype in (torch.float16, torch.bfloat16) else up
            ub, ua, uz, inv_dn = _ra_norm_terms_torch(z, ctx.b)
            return (ctx.b * (ub + ua) * uz * inv_dn * u).to(gate.dtype)

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Tuple[Any, Any, None]:  # type: ignore[override]
            gate, up = ctx.saved_tensors
            b_val: float = ctx.b
            res = triton_ra_glu_backward(grad_out, gate, up, b=b_val)
            if res is not None:
                return (res[0].reshape(gate.shape) if ctx.needs_input_grad[0] else None), (res[1].reshape(up.shape) if ctx.needs_input_grad[1] else None), None
            cpu_res = _cpu_ra_gated_bwd(grad_out, gate, up, b_val, is_swiglu=False, need_dg=bool(ctx.needs_input_grad[0]), need_du=bool(ctx.needs_input_grad[1]))
            if cpu_res is not None:
                return cpu_res[0], cpu_res[1], None

            z = gate.to(torch.float32) if gate.dtype in (torch.float16, torch.bfloat16) else gate
            u = up.to(torch.float32) if up.dtype in (torch.float16, torch.bfloat16) else up
            g = grad_out.to(torch.float32) if grad_out.dtype in (torch.float16, torch.bfloat16) else grad_out
            ub, ua, uz, inv_dn = _ra_norm_terms_torch(z, b_val)
            ra_t = b_val * (ub + ua) * uz * inv_dn
            deriv = (ub * ub * ub) * (ub + 2.0 * ua) * (inv_dn * inv_dn)
            dgate = (g * u * deriv).to(gate.dtype) if ctx.needs_input_grad[0] else None
            dup = (g * ra_t).to(up.dtype) if ctx.needs_input_grad[1] else None
            return dgate, dup, None

    _RaGLUFunction = _RaGLUFunctionImpl

    class _RaGELUFunctionImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any) -> Any:
            ctx.save_for_backward(x)
            out = triton_ra_gelu(x)
            if out is not None:
                return out
            x_w = x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
            return (x_w * _pos_ra_gate_torch(_SQRT_2_OVER_PI * x_w, 1.0)).to(x.dtype)

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Tuple[Any]:  # type: ignore[override]
            (x,) = ctx.saved_tensors
            if not ctx.needs_input_grad[0]:
                return (None,)
            dx_triton = triton_ra_gelu_backward(grad_out, x)
            if dx_triton is not None:
                return (dx_triton.reshape(x.shape),)
            x_f32 = x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
            g_f32 = grad_out.to(torch.float32) if grad_out.dtype in (torch.float16, torch.bfloat16) else grad_out
            z = _SQRT_2_OVER_PI * x_f32
            ub, ua, _, inv_dn = _ra_norm_terms_torch(z, 1.0)
            q = 0.5 * (ub * ub) * inv_dn
            is_pos = z >= 0.0
            sigma = torch.where(is_pos, 1.0 - q, q)
            deriv = torch.where(is_pos, sigma + q * ua * (ub + 2.0 * ua) * inv_dn, q * (ub * ub - ua * ua) * inv_dn)
            return ((g_f32 * deriv).to(x.dtype),)

    _RaGELUFunction = _RaGELUFunctionImpl


def ra_tanh(x: Any, b: float = 1.0) -> Any:
    """Rational Hyperbolic Tangent: RaTanh(x; b) = (b * (b + |x|) * x) / (b * (b + |x|) + x^2)."""
    b = _validate_positive_b(b)

    if _HAS_TORCH and isinstance(x, torch.Tensor):
        if not _is_torch_compiling():
            if torch.is_grad_enabled() and x.requires_grad and _RaTanhFunction is not None:
                return _RaTanhFunction.apply(x, b)
            if _HAS_TRITON and x.is_cuda:
                triton_out = triton_ra_tanh(x, b=b)
                if triton_out is not None:
                    return triton_out

        if (
            _HAS_RUST_SIMD
            and not _is_torch_compiling()
            and x.device.type == "cpu"
            and not (torch.is_grad_enabled() and x.requires_grad)
            and x.dtype == torch.float32
            and x.is_contiguous()
        ):
            out = torch.empty_like(x)
            fn = _rs_engine.ra_tanh_cubecl_f32_pyo3 if _can_use_cubecl_nn(x.numel(), "ra_tanh_cubecl_f32_pyo3") else _rs_engine.ra_tanh_f32_pyo3
            fn(x, out, b)
            return out

        return _ra_tanh_torch_stable(x, b)

    if _HAS_NUMPY and isinstance(x, np.ndarray):
        if _HAS_RUST_SIMD and x.dtype == np.float32 and x.flags.c_contiguous:
            out = np.empty_like(x)
            fn = _rs_engine.ra_tanh_cubecl_f32_pyo3 if _can_use_cubecl_nn(x.size, "ra_tanh_cubecl_f32_pyo3") else _rs_engine.ra_tanh_f32_pyo3
            fn(x, out, b)
            return out

        x_w = x.astype(np.float32, copy=False) if x.dtype == np.float16 else x
        abs_x = np.abs(x_w)
        inv_m = 1.0 / np.maximum(b, abs_x)
        ub = b * inv_m
        ua = abs_x * inv_m
        uz = x_w * inv_m
        res = (b * (ub + ua) * uz) / (ub * (ub + ua) + ua * ua)
        return res.astype(x.dtype, copy=False) if x.dtype == np.float16 else res

    if isinstance(x, (int, float)):
        xf = float(x)
        abs_x = abs(xf)
        inv_m = 1.0 / max(b, abs_x)
        ub = b * inv_m
        ua = abs_x * inv_m
        uz = xf * inv_m
        return (b * (ub + ua) * uz) / (ub * (ub + ua) + ua * ua)

    raise TypeError(f"ra_tanh expects torch.Tensor, numpy.ndarray, or float, got {type(x)}")


class RaTanh(BaseModule):
    """Rational Hyperbolic Tangent module."""

    def __init__(self, b: float = 1.0) -> None:
        super().__init__()
        self.b = _validate_positive_b(b)

    def forward(self, x: Any) -> Any:
        return ra_tanh(x, b=self.b)

    def extra_repr(self) -> str:
        return f"b={self.b}"


def ra_gelu(x: Any) -> Any:
    """Rational Gaussian Error Linear Unit: x * pos_ra_gate(sqrt(2 / pi) * x, b=1.0)."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        if not _is_torch_compiling():
            if torch.is_grad_enabled() and x.requires_grad and _RaGELUFunction is not None:
                return _RaGELUFunction.apply(x)
            if _HAS_TRITON and x.is_cuda:
                triton_out = triton_ra_gelu(x)
                if triton_out is not None:
                    return triton_out
        x_w = x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
        return (x_w * _pos_ra_gate_torch(_SQRT_2_OVER_PI * x_w, 1.0)).to(x.dtype)
    return x * pos_ra_gate(_SQRT_2_OVER_PI * x, b=1.0)


class RaGELU(BaseModule):
    """Rational GELU module."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Any) -> Any:
        return ra_gelu(x)


def pos_ra_gate(z: Any, b: float = 1.0) -> Any:
    """Strictly positive rational gate on (0, 1), cancellation-free across all z."""
    b_val = _validate_positive_b(b)
    if _HAS_TORCH and isinstance(z, torch.Tensor):
        if z.dtype in (torch.float16, torch.bfloat16):
            return _pos_ra_gate_torch(z.to(torch.float32), b_val).to(z.dtype)
        return _pos_ra_gate_torch(z, b_val)
    if _HAS_NUMPY and isinstance(z, np.ndarray):
        z_w = z.astype(np.float32, copy=False) if z.dtype == np.float16 else z
        abs_z = np.abs(z_w)
        inv_m = 1.0 / np.maximum(b_val, abs_z)
        ub = b_val * inv_m
        ua = abs_z * inv_m
        q = (0.5 * ub * ub) / (ub * (ub + ua) + ua * ua)
        res = np.where(z_w >= 0.0, 1.0 - q, q)
        return res.astype(z.dtype, copy=False) if z.dtype == np.float16 else res
    if isinstance(z, (int, float)):
        zf = float(z)
        abs_z = abs(zf)
        inv_m = 1.0 / max(b_val, abs_z)
        ub = b_val * inv_m
        ua = abs_z * inv_m
        q = (0.5 * ub * ub) / (ub * (ub + ua) + ua * ua)
        return (1.0 - q) if zf >= 0.0 else q
    raise TypeError(f"pos_ra_gate expects torch.Tensor, numpy.ndarray, or float, got {type(z)}")


def _dispatch_ra_gated(gate: Any, up: Any, b_val: float, *, is_swiglu: bool) -> Any:
    triton_fn = triton_ra_swiglu if is_swiglu else triton_ra_glu
    autograd_cls = _RaSwiGLUFunction if is_swiglu else _RaGLUFunction
    cubecl_name = "ra_swiglu_cubecl_f32_pyo3" if is_swiglu else "ra_glu_cubecl_f32_pyo3"
    simd_name = "ra_swiglu_f32_pyo3" if is_swiglu else "ra_glu_f32_pyo3"

    if _HAS_TORCH and isinstance(gate, torch.Tensor) and isinstance(up, torch.Tensor) and gate.shape == up.shape:
        if not _is_torch_compiling():
            if torch.is_grad_enabled() and (gate.requires_grad or up.requires_grad) and autograd_cls is not None:
                return autograd_cls.apply(gate, up, b_val)
            if _HAS_TRITON and gate.is_cuda and up.is_cuda:
                triton_out = triton_fn(gate, up, b=b_val)
                if triton_out is not None:
                    return triton_out
        if (
            _HAS_RUST_SIMD and not _is_torch_compiling()
            and gate.device.type == "cpu" and up.device.type == "cpu"
            and not (torch.is_grad_enabled() and (gate.requires_grad or up.requires_grad))
            and gate.dtype == torch.float32 and up.dtype == torch.float32
            and gate.is_contiguous() and up.is_contiguous()
        ):
            if _can_use_cubecl_nn(gate.numel(), cubecl_name):
                out = torch.empty_like(gate)
                getattr(_rs_engine, cubecl_name)(gate, up, out, b_val)
                return out
            if hasattr(_rs_engine, simd_name):
                out = torch.empty_like(gate)
                getattr(_rs_engine, simd_name)(gate, up, out, b_val)
                return out
        z = gate.to(torch.float32) if gate.dtype in (torch.float16, torch.bfloat16) else gate
        u = up.to(torch.float32) if up.dtype in (torch.float16, torch.bfloat16) else up
        ub, ua, uz, inv_dn = _ra_norm_terms_torch(z, b_val)
        if is_swiglu:
            q = 0.5 * (ub * ub) * inv_dn
            s = torch.where(z >= 0.0, z * (1.0 - q), (0.5 * b_val * ub) * uz * inv_dn)
        else:
            s = b_val * (ub + ua) * uz * inv_dn
        return (s * u).to(gate.dtype)

    if (
        _HAS_NUMPY and _HAS_RUST_SIMD and isinstance(gate, np.ndarray) and isinstance(up, np.ndarray)
        and gate.shape == up.shape and gate.dtype == np.float32 and up.dtype == np.float32
        and gate.flags.c_contiguous and up.flags.c_contiguous
    ):
        if _can_use_cubecl_nn(gate.size, cubecl_name):
            out_np = np.empty_like(gate)
            getattr(_rs_engine, cubecl_name)(gate, up, out_np, b_val)
            return out_np
        if hasattr(_rs_engine, simd_name):
            out_np = np.empty_like(gate)
            getattr(_rs_engine, simd_name)(gate, up, out_np, b_val)
            return out_np

    if is_swiglu:
        return (gate * pos_ra_gate(gate, b=b_val)) * up
    return ra_tanh(gate, b=b_val) * up


def ra_swiglu(gate: Any, up: Any, b: float = 1.0) -> Any:
    """Rational SwiGLU functional: (gate * sigma_Rat(gate; b)) * up."""
    return _dispatch_ra_gated(gate, up, _validate_positive_b(b), is_swiglu=True)


def ra_glu(gate: Any, up: Any, b: float = 1.0) -> Any:
    """Hard-bounded Rational GLU functional: RaTanh(gate; b) * up."""
    return _dispatch_ra_gated(gate, up, _validate_positive_b(b), is_swiglu=False)


class _RaGatedFFN(BaseModule):
    w_gate: Any
    w_up: Any
    w_down: Any
    _is_swiglu: bool = True

    def __init__(
        self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None,
        bias: bool = False, b: float = 1.0, device: Any = None, dtype: Any = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features) if hidden_features is not None else int(in_features * 8 / 3)
        self.out_features = int(out_features) if out_features is not None else self.in_features
        self.b = _validate_positive_b(b)

        if _HAS_TORCH:
            self.w_gate = nn.Linear(self.in_features, self.hidden_features, bias=bias, device=device, dtype=dtype)
            self.w_up = nn.Linear(self.in_features, self.hidden_features, bias=bias, device=device, dtype=dtype)
            self.w_down = nn.Linear(self.hidden_features, self.out_features, bias=bias, device=device, dtype=dtype)
        else:
            self.w_gate = _init_linear_np(self.in_features, self.hidden_features, bias=bias)
            self.w_up = _init_linear_np(self.in_features, self.hidden_features, bias=bias)
            self.w_down = _init_linear_np(self.hidden_features, self.out_features, bias=bias)

    def forward(self, x: Any) -> Any:
        act_fn = ra_swiglu if self._is_swiglu else ra_glu
        if _HAS_TORCH and isinstance(x, torch.Tensor):
            return self.w_down(act_fn(self.w_gate(x), self.w_up(x), b=self.b))
        if _HAS_NUMPY and isinstance(x, np.ndarray):
            gate = np.ascontiguousarray(_linear_np(x, self.w_gate))
            up = np.ascontiguousarray(_linear_np(x, self.w_up))
            return _linear_np(act_fn(gate, up, b=self.b), self.w_down)
        raise TypeError(f"{self.__class__.__name__} expects torch.Tensor or numpy.ndarray, got {type(x)}")

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"hidden_features={self.hidden_features}, "
            f"out_features={self.out_features}, "
            f"b={self.b}"
        )


class RaSwiGLU(_RaGatedFFN):
    """Rational SwiGLU MLP Block: W_down((W_gate(x) * sigma_Rat(W_gate(x); b)) * W_up(x))."""

    _is_swiglu: bool = True


class RaGLU(_RaGatedFFN):
    """Hard-bounded Rational GLU MLP Block: W_down(RaTanh(W_gate(x); b) * W_up(x))."""

    _is_swiglu: bool = False
