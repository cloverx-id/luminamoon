from __future__ import annotations

from typing import Any, Optional, Tuple

from ._compat import (
    BaseModule,
    _HAS_NUMPY,
    _HAS_TORCH,
    _HAS_TRITON,
    _is_torch_compiling,
    nn,
    np,
    torch,
)
from .activation import pos_ra_gate
from .kernels import triton_fused_ra_ema_bwd, triton_fused_ra_ema_fwd


def _reduce_broadcast_grad(grad_expanded: Any, target_shape: Tuple[int, ...]) -> Any:
    """Reduces an expanded gradient tensor back to `target_shape` after broadcast expansion."""
    if grad_expanded.shape == target_shape:
        return grad_expanded
    g = grad_expanded
    ndim_diff = g.ndim - len(target_shape)
    if ndim_diff > 0:
        g = g.sum(dim=tuple(range(ndim_diff)))
    reduce_axes = tuple(i for i, (gs, ts) in enumerate(zip(g.shape, target_shape)) if ts == 1 and gs > 1)
    if reduce_axes:
        g = g.sum(dim=reduce_axes, keepdim=True)
    return g.reshape(target_shape)


_FusedRaEMAFunction: Any = None

if _HAS_TORCH:

    class _FusedRaEMAFunctionImpl(torch.autograd.Function):
        """Fused Rational EMA autograd function with Triton SRAM or zero-graph CPU scan."""

        @staticmethod
        def forward(ctx: Any, x: Any, decay: Any, h0: Optional[Any]) -> Tuple[Any, Any]:
            with torch.no_grad():
                orig_shape = x.shape
                orig_dtype = x.dtype
                T = orig_shape[-2]
                D = orig_shape[-1]
                B_flat = x.numel() // max(1, T * D)
                state_shape = orig_shape[:-2] + (D,)

                decay_3d = decay.expand(orig_shape).reshape(B_flat, T, D)
                x_3d = x.reshape(B_flat, T, D)
                h0_2d = h0.expand(state_shape).reshape(B_flat, D) if h0 is not None else None

                out_3d: Any = None
                h_last_2d: Any = None
                if (
                    _HAS_TRITON
                    and x.is_cuda
                    and orig_dtype in (torch.float16, torch.bfloat16, torch.float32)
                ):
                    triton_res = triton_fused_ra_ema_fwd(
                        x_3d,
                        decay_3d.to(orig_dtype),
                        h0_2d.to(orig_dtype) if h0_2d is not None else None,
                    )
                    if triton_res is not None:
                        out_3d, h_last_2d = triton_res

                if out_3d is None:
                    work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
                    x_w = x_3d.to(work_dtype)
                    a_w = decay_3d.to(work_dtype)
                    out_w = torch.empty_like(x_w)
                    h = (
                        h0_2d.to(work_dtype).clone()
                        if h0_2d is not None
                        else torch.zeros((B_flat, D), device=x.device, dtype=work_dtype)
                    )
                    for t in range(T):
                        a_t = a_w[:, t, :]
                        h = a_t * h + (1.0 - a_t) * x_w[:, t, :]
                        out_w[:, t, :] = h
                    out_3d = out_w.to(orig_dtype)
                    h_last_2d = h.to(orig_dtype)

                if h0 is not None:
                    ctx.save_for_backward(x, decay, h0, out_3d)
                    ctx.has_h0 = True
                else:
                    ctx.save_for_backward(x, decay, out_3d)
                    ctx.has_h0 = False
                ctx.orig_shape = orig_shape
                ctx.state_shape = state_shape
                ctx.decay_shape = decay.shape
                ctx.h0_shape = h0.shape if h0 is not None else None
                return out_3d.reshape(orig_shape), h_last_2d.reshape(state_shape)

        @staticmethod
        def backward(  # type: ignore[override]
            ctx: Any,
            grad_out: Any,
            grad_h_last: Any,
        ) -> Tuple[Any, Any, Optional[Any]]:
            with torch.no_grad():
                if ctx.has_h0:
                    x, decay, h0, out_3d = ctx.saved_tensors
                else:
                    x, decay, out_3d = ctx.saved_tensors
                    h0 = None

                orig_shape = ctx.orig_shape
                state_shape = ctx.state_shape
                orig_dtype = x.dtype
                T = orig_shape[-2]
                D = orig_shape[-1]
                B_flat = x.numel() // max(1, T * D)

                x_3d = x.reshape(B_flat, T, D)
                decay_3d = decay.expand(orig_shape).reshape(B_flat, T, D)
                h0_2d = h0.expand(state_shape).reshape(B_flat, D) if h0 is not None else None

                go_3d = (
                    grad_out.reshape(B_flat, T, D)
                    if grad_out is not None
                    else torch.zeros((B_flat, T, D), device=x.device, dtype=orig_dtype)
                )
                dh_last_2d = grad_h_last.reshape(B_flat, D) if grad_h_last is not None else None
                need_dh0 = bool(ctx.has_h0 and ctx.needs_input_grad[2])

                if (
                    _HAS_TRITON
                    and x.is_cuda
                    and orig_dtype in (torch.float16, torch.bfloat16, torch.float32)
                ):
                    triton_bwd = triton_fused_ra_ema_bwd(
                        go_3d.to(orig_dtype),
                        dh_last_2d.to(orig_dtype) if dh_last_2d is not None else None,
                        x_3d,
                        decay_3d.to(orig_dtype),
                        h0_2d.to(orig_dtype) if h0_2d is not None else None,
                        out_3d,
                        need_dh0=need_dh0,
                    )
                    if triton_bwd is not None:
                        dx_3d, da_3d, dh0_2d = triton_bwd
                        dx = dx_3d.reshape(orig_shape).to(orig_dtype)
                        da = _reduce_broadcast_grad(da_3d.reshape(orig_shape), ctx.decay_shape).to(decay.dtype)
                        dh0 = (
                            _reduce_broadcast_grad(dh0_2d.reshape(state_shape), ctx.h0_shape).to(h0.dtype)
                            if (dh0_2d is not None and h0 is not None and ctx.h0_shape is not None)
                            else None
                        )
                        return dx, da, dh0

                work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
                x_w = x_3d.to(work_dtype)
                a_w = decay_3d.to(work_dtype)
                out_w = out_3d.to(work_dtype)
                go_w = go_3d.to(work_dtype)
                h0_w = h0_2d.to(work_dtype) if h0_2d is not None else None

                dx_w = torch.empty_like(x_w)
                da_w = torch.empty_like(a_w)
                delta = (
                    dh_last_2d.to(work_dtype).clone()
                    if dh_last_2d is not None
                    else torch.zeros((B_flat, D), device=x.device, dtype=work_dtype)
                )

                for t in range(T - 1, -1, -1):
                    delta = delta + go_w[:, t, :]
                    a_t = a_w[:, t, :]
                    x_t = x_w[:, t, :]
                    if t > 0:
                        h_prev = out_w[:, t - 1, :]
                    elif h0_w is not None:
                        h_prev = h0_w
                    else:
                        h_prev = torch.zeros_like(x_t)

                    dx_w[:, t, :] = (1.0 - a_t) * delta
                    da_w[:, t, :] = (h_prev - x_t) * delta
                    delta = a_t * delta

                dx = dx_w.reshape(orig_shape).to(orig_dtype)
                da = _reduce_broadcast_grad(da_w.reshape(orig_shape), ctx.decay_shape).to(decay.dtype)
                dh0 = (
                    _reduce_broadcast_grad(delta.reshape(state_shape), ctx.h0_shape).to(h0.dtype)
                    if (need_dh0 and h0 is not None and ctx.h0_shape is not None)
                    else None
                )
                return dx, da, dh0

    _FusedRaEMAFunction = _FusedRaEMAFunctionImpl


def fused_ra_ema(
    x: Any,
    decay: Any,
    h0: Optional[Any] = None,
    return_state: bool = False,
) -> Any:
    """Causal O(N) Fused Rational Exponential Moving Average along `dim = -2`."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        if x.ndim < 2:
            raise ValueError(f"fused_ra_ema expects at least a 2D (T, D) tensor, got {x.ndim}D")
        if not isinstance(decay, torch.Tensor):
            decay = torch.as_tensor(decay, device=x.device, dtype=x.dtype)
        if _is_torch_compiling():
            if decay is x:
                decay = decay.clone()
            if h0 is not None and (h0 is x or h0 is decay):
                h0 = h0.clone()
        out, h_last = _FusedRaEMAFunction.apply(x, decay, h0)
        return (out, h_last) if return_state else out

    if _HAS_NUMPY and isinstance(x, np.ndarray):
        if x.ndim < 2:
            raise ValueError(f"fused_ra_ema expects at least a 2D (T, D) array, got {x.ndim}D")
        orig_shape = x.shape
        T = orig_shape[-2]
        D = orig_shape[-1]
        B_flat = x.size // max(1, T * D)
        state_shape = orig_shape[:-2] + (D,)

        a_arr = np.broadcast_to(np.asarray(decay, dtype=x.dtype), orig_shape).reshape(B_flat, T, D)
        x_3d = x.reshape(B_flat, T, D)
        out_3d = np.empty_like(x_3d)
        h = (
            np.broadcast_to(np.asarray(h0, dtype=x.dtype), state_shape).reshape(B_flat, D).copy()
            if h0 is not None
            else np.zeros((B_flat, D), dtype=x.dtype)
        )
        for t in range(T):
            a_t = a_arr[:, t, :]
            h = a_t * h + (1.0 - a_t) * x_3d[:, t, :]
            out_3d[:, t, :] = h

        out = out_3d.reshape(orig_shape)
        h_last = h.reshape(state_shape)
        return (out, h_last) if return_state else out

    raise TypeError(f"fused_ra_ema expects torch.Tensor or numpy.ndarray, got {type(x)}")


class RaEMA(BaseModule):
    decay_proj: Any
    decay_logit: Any
    residual_logit: Any
    """O(N) causal sequence mixer with algebraic rational decay gates and optional residual blending."""

    def __init__(
        self,
        dim: int,
        input_dependent_decay: bool = True,
        use_residual_gate: bool = True,
        init_decay_logit: float = 2.0,
        b: float = 1.0,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        self.input_dependent_decay = bool(input_dependent_decay)
        self.use_residual_gate = bool(use_residual_gate)
        self.init_decay_logit = float(init_decay_logit)
        self.b = float(b)

        if _HAS_TORCH:
            if self.input_dependent_decay:
                self.decay_proj = nn.Linear(self.dim, self.dim, bias=True, device=device, dtype=dtype)
                with torch.no_grad():
                    nn.init.zeros_(self.decay_proj.weight)
                    nn.init.constant_(self.decay_proj.bias, self.init_decay_logit)
                self.decay_logit = None
            else:
                self.decay_proj = None
                self.decay_logit = nn.Parameter(
                    torch.full((self.dim,), self.init_decay_logit, device=device, dtype=dtype)
                )

            if self.use_residual_gate:
                self.residual_logit = nn.Parameter(
                    torch.full((self.dim,), -1.0, device=device, dtype=dtype)
                )
            else:
                self.residual_logit = None
        else:
            self.decay_proj = None
            self.decay_logit = (
                np.full((self.dim,), self.init_decay_logit, dtype=np.float32) if _HAS_NUMPY else None
            )
            self.residual_logit = (
                np.full((self.dim,), -1.0, dtype=np.float32)
                if (_HAS_NUMPY and self.use_residual_gate)
                else None
            )

    def forward(
        self,
        x: Any,
        h0: Optional[Any] = None,
        return_state: bool = False,
    ) -> Any:
        if _HAS_TORCH and isinstance(x, torch.Tensor):
            if self.input_dependent_decay and self.decay_proj is not None:
                decay = pos_ra_gate(self.decay_proj(x), b=self.b)
            else:
                decay = pos_ra_gate(self.decay_logit, b=self.b)

            ema_res = fused_ra_ema(x, decay, h0=h0, return_state=return_state)
            h_seq, h_last = ema_res if return_state else (ema_res, None)

            if self.use_residual_gate and self.residual_logit is not None:
                gamma = pos_ra_gate(self.residual_logit, b=self.b).to(x.dtype)
                out = gamma * h_seq + (1.0 - gamma) * x
            else:
                out = h_seq
            return (out, h_last) if return_state else out

        if _HAS_NUMPY and isinstance(x, np.ndarray):
            logit = (
                self.decay_logit
                if isinstance(self.decay_logit, np.ndarray)
                else np.full((self.dim,), self.init_decay_logit, dtype=x.dtype)
            )
            decay = pos_ra_gate(logit, b=self.b)
            ema_res = fused_ra_ema(x, decay, h0=h0, return_state=return_state)
            h_seq, h_last = ema_res if return_state else (ema_res, None)

            if self.use_residual_gate:
                r_logit = (
                    self.residual_logit
                    if isinstance(self.residual_logit, np.ndarray)
                    else np.full((self.dim,), -1.0, dtype=x.dtype)
                )
                gamma = pos_ra_gate(r_logit, b=self.b)
                out = gamma * h_seq + (1.0 - gamma) * x
            else:
                out = h_seq
            return (out, h_last) if return_state else out

        raise TypeError(f"RaEMA expects torch.Tensor or numpy.ndarray, got {type(x)}")

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"input_dependent_decay={self.input_dependent_decay}, "
            f"use_residual_gate={self.use_residual_gate}"
        )


ra_ema = fused_ra_ema
RationalEMA = RaEMA
FusedRaEMA = RaEMA

__all__ = ["fused_ra_ema", "ra_ema", "RaEMA", "RationalEMA", "FusedRaEMA"]
