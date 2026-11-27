"""CPU Rust SIMD forward/backward dispatch helpers for luminamoon.nn."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ._compat import _HAS_RUST_SIMD, _HAS_TORCH, _is_torch_compiling, _rs_engine, torch


def _is_cpu_f32(t: Any) -> bool:
    return isinstance(t, torch.Tensor) and t.device.type == "cpu" and t.dtype == torch.float32


def _cpu_ra_norm_fwd(x: Any, weight: Optional[Any], bias: Optional[Any], eps: float) -> Optional[Any]:
    if not (
        _HAS_RUST_SIMD and not _is_torch_compiling() and _is_cpu_f32(x)
        and x.ndim >= 1 and x.shape[-1] > 0
        and (weight is None or _is_cpu_f32(weight))
        and (bias is None or _is_cpu_f32(bias))
    ):
        return None
    x_c = x if x.is_contiguous() else x.contiguous()
    w_c = (weight if weight.is_contiguous() else weight.contiguous()) if weight is not None else None
    b_c = (bias if bias.is_contiguous() else bias.contiguous()) if bias is not None else None
    out = torch.empty_like(x_c)
    _rs_engine.ra_norm_f32_pyo3(x_c, out, w_c, b_c, float(eps), x_c.shape[-1])
    return out


def _cpu_ra_norm_bwd(
    grad_y: Any,
    x: Any,
    weight: Optional[Any],
    need_dw: bool,
    need_db: bool,
    eps: float,
) -> Optional[Tuple[Any, Optional[Any], Optional[Any]]]:
    if not (
        _HAS_RUST_SIMD and not _is_torch_compiling()
        and hasattr(_rs_engine, "ra_norm_bwd_f32_pyo3")
        and _is_cpu_f32(x) and _is_cpu_f32(grad_y) and grad_y.shape == x.shape
        and x.ndim >= 1 and x.shape[-1] > 0
        and (weight is None or _is_cpu_f32(weight))
    ):
        return None
    cols = x.shape[-1]
    gy_c = grad_y if grad_y.is_contiguous() else grad_y.contiguous()
    x_c = x if x.is_contiguous() else x.contiguous()
    w_c = (weight if weight.is_contiguous() else weight.contiguous()) if weight is not None else None
    dx = torch.empty_like(x_c)
    dw = torch.empty((cols,), device=x_c.device, dtype=torch.float32) if (need_dw and w_c is not None) else None
    db = torch.empty((cols,), device=x_c.device, dtype=torch.float32) if need_db else None
    _rs_engine.ra_norm_bwd_f32_pyo3(gy_c, x_c, w_c, dx, dw, db, float(eps), cols)
    return dx, dw, db


def _cpu_ra_tanh_fwd(x: Any, b: float) -> Optional[Any]:
    if not (_HAS_RUST_SIMD and not _is_torch_compiling() and _is_cpu_f32(x)):
        return None
    x_c = x if x.is_contiguous() else x.contiguous()
    out = torch.empty_like(x_c)
    _rs_engine.ra_tanh_f32_pyo3(x_c, out, float(b))
    return out


def _cpu_ra_tanh_bwd(grad_out: Any, x: Any, b: float) -> Optional[Any]:
    if not (
        _HAS_RUST_SIMD and not _is_torch_compiling()
        and hasattr(_rs_engine, "ra_tanh_bwd_f32_pyo3")
        and _is_cpu_f32(x) and _is_cpu_f32(grad_out) and grad_out.shape == x.shape
    ):
        return None
    g_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
    x_c = x if x.is_contiguous() else x.contiguous()
    dx = torch.empty_like(x_c)
    _rs_engine.ra_tanh_bwd_f32_pyo3(g_c, x_c, dx, float(b))
    return dx


def _cpu_ra_gated_fwd(gate: Any, up: Any, b: float, *, is_swiglu: bool) -> Optional[Any]:
    fn_name = "ra_swiglu_f32_pyo3" if is_swiglu else "ra_glu_f32_pyo3"
    if not (
        _HAS_RUST_SIMD and not _is_torch_compiling()
        and hasattr(_rs_engine, fn_name)
        and _is_cpu_f32(gate) and _is_cpu_f32(up) and gate.shape == up.shape
    ):
        return None
    g_c = gate if gate.is_contiguous() else gate.contiguous()
    u_c = up if up.is_contiguous() else up.contiguous()
    out = torch.empty_like(g_c)
    getattr(_rs_engine, fn_name)(g_c, u_c, out, float(b))
    return out


def _cpu_ra_gated_bwd(
    grad_out: Any,
    gate: Any,
    up: Any,
    b: float,
    *,
    is_swiglu: bool,
    need_dg: bool,
    need_du: bool,
) -> Optional[Tuple[Optional[Any], Optional[Any]]]:
    fn_name = "ra_swiglu_bwd_f32_pyo3" if is_swiglu else "ra_glu_bwd_f32_pyo3"
    if not (
        _HAS_RUST_SIMD and not _is_torch_compiling()
        and hasattr(_rs_engine, fn_name)
        and _is_cpu_f32(grad_out) and _is_cpu_f32(gate) and _is_cpu_f32(up)
        and grad_out.shape == gate.shape and gate.shape == up.shape
    ):
        return None
    go_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
    g_c = gate if gate.is_contiguous() else gate.contiguous()
    u_c = up if up.is_contiguous() else up.contiguous()
    dg = torch.empty_like(g_c)
    du = torch.empty_like(u_c)
    getattr(_rs_engine, fn_name)(go_c, g_c, u_c, dg, du, float(b))
    return (dg.reshape(gate.shape) if need_dg else None), (du.reshape(up.shape) if need_du else None)


_RaGatedPackedCPUFunction: Any = None
if _HAS_TORCH and torch is not None:

    class _RaGatedPackedCPUFunctionImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, gu: Any, b: float, is_swiglu: bool) -> Any:
            gu_c = gu if gu.is_contiguous() else gu.contiguous()
            rows, two_h = gu_c.shape
            hidden = two_h // 2
            out = torch.empty((rows, hidden), device=gu_c.device, dtype=torch.float32)
            _rs_engine.ra_gated_packed_fwd_f32_pyo3(gu_c, out, rows, hidden, float(b), bool(is_swiglu))
            ctx.save_for_backward(gu_c)
            ctx.b = float(b)
            ctx.is_swiglu = bool(is_swiglu)
            return out

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Tuple[Optional[Any], None, None]:  # type: ignore[override]
            (gu_c,) = ctx.saved_tensors
            if not ctx.needs_input_grad[0]:
                return None, None, None
            g_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
            rows, two_h = gu_c.shape
            hidden = two_h // 2
            dgu = torch.empty_like(gu_c)
            _rs_engine.ra_gated_packed_bwd_f32_pyo3(g_c, gu_c, dgu, rows, hidden, ctx.b, ctx.is_swiglu)
            return dgu, None, None

    _RaGatedPackedCPUFunction = _RaGatedPackedCPUFunctionImpl


def _cpu_ra_gated_packed(gu: Any, b: float, *, is_swiglu: bool) -> Optional[Any]:
    if not (
        _HAS_RUST_SIMD and not _is_torch_compiling()
        and hasattr(_rs_engine, "ra_gated_packed_fwd_f32_pyo3")
        and _is_cpu_f32(gu) and gu.ndim == 2 and (gu.shape[1] % 2 == 0)
    ):
        return None
    if torch.is_grad_enabled() and gu.requires_grad and _RaGatedPackedCPUFunction is not None:
        return _RaGatedPackedCPUFunction.apply(gu, float(b), bool(is_swiglu))
    gu_c = gu if gu.is_contiguous() else gu.contiguous()
    rows, two_h = gu_c.shape
    hidden = two_h // 2
    out = torch.empty((rows, hidden), device=gu_c.device, dtype=torch.float32)
    _rs_engine.ra_gated_packed_fwd_f32_pyo3(gu_c, out, rows, hidden, float(b), bool(is_swiglu))
    return out


def _cpu_qrope(x: Any, cos_l: Any, sin_l: Any, cos_r: Any, sin_r: Any, sign: float) -> Optional[Any]:
    if not (
        _HAS_RUST_SIMD and not _is_torch_compiling()
        and hasattr(_rs_engine, "qrope_f32_pyo3")
        and _is_cpu_f32(x) and x.ndim == 4 and x.shape[-1] > 0 and (x.shape[-1] % 4 == 0)
        and cos_l.device.type == "cpu" and sin_l.device.type == "cpu"
        and cos_r.device.type == "cpu" and sin_r.device.type == "cpu"
    ):
        return None
    batch, heads, seq_len, d_head = x.shape
    num_q = d_head // 4
    x_c = x if x.is_contiguous() else x.contiguous()
    cl_c = cos_l.to(torch.float32).contiguous()
    sl_c = sin_l.to(torch.float32).contiguous()
    cr_c = cos_r.to(torch.float32).contiguous()
    sr_c = sin_r.to(torch.float32).contiguous()
    out = torch.empty_like(x_c)
    _rs_engine.qrope_f32_pyo3(x_c, out, cl_c, sl_c, cr_c, sr_c, batch * heads, seq_len, num_q, float(sign))
    return out


__all__ = [
    "_cpu_ra_norm_fwd",
    "_cpu_ra_norm_bwd",
    "_cpu_ra_tanh_fwd",
    "_cpu_ra_tanh_bwd",
    "_cpu_ra_gated_fwd",
    "_cpu_ra_gated_bwd",
    "_cpu_ra_gated_packed",
    "_cpu_qrope",
]
