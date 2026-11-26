"""Selective reversibility helpers and autograd function for ReversibleDKDBlock."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any, Literal, Optional, Tuple

from ._compat import _HAS_TORCH, torch
from ._latent_attention import LatentKVAttention

SelectiveReversibleType = Literal[
    "none",
    "all",
    "every",
    "even",
    "odd",
    "first_half",
    "second_half",
    "indices",
]
_VALID_SELECTIVE_TYPES: Tuple[str, ...] = (
    "none",
    "all",
    "every",
    "even",
    "odd",
    "first_half",
    "second_half",
    "indices",
)


def resolve_selective_reversible(
    layer_idx: Optional[int] = None,
    num_layers: Optional[int] = None,
    *,
    reversible: bool = False,
    selective_type: SelectiveReversibleType = "none",
    every: int = 2,
    selective_indices: Optional[Iterable[int]] = None,
) -> bool:
    """Resolve whether a layer should enable reversible recomputation under a selective policy.

    Supported `selective_type` modes:
    - `"none"`: use the explicit `reversible` flag as-is.
    - `"all"`: enable reversibility on every layer.
    - `"every"`: enable reversibility when `layer_idx % every == 0`.
    - `"even"` / `"odd"`: enable reversibility on even or odd `layer_idx`.
    - `"first_half"` / `"second_half"`: enable reversibility on the first or second half of `num_layers`.
    - `"indices"`: enable reversibility when `layer_idx` is in `selective_indices`.
    """
    if selective_type not in _VALID_SELECTIVE_TYPES:
        valid = ", ".join(repr(m) for m in _VALID_SELECTIVE_TYPES)
        raise ValueError(f"selective_type must be one of ({valid}), got {selective_type!r}.")
    if selective_type == "none":
        return bool(reversible)
    if selective_type == "all":
        return True
    if layer_idx is None or int(layer_idx) < 0:
        raise ValueError(f"layer_idx must be a non-negative integer when selective_type={selective_type!r}.")
    idx = int(layer_idx)
    if selective_type == "every":
        step = int(every)
        if step <= 0:
            raise ValueError(f"every must be > 0, got {every}.")
        return (idx % step) == 0
    if selective_type == "even":
        return (idx % 2) == 0
    if selective_type == "odd":
        return (idx % 2) == 1
    if selective_type in ("first_half", "second_half"):
        if num_layers is None or int(num_layers) <= 0:
            raise ValueError(f"num_layers must be > 0 when selective_type={selective_type!r}.")
        mid = int(num_layers) // 2
        return idx < mid if selective_type == "first_half" else idx >= mid
    if selective_indices is None:
        return False
    return idx in {int(i) for i in selective_indices}


def _apply_inplace_compile(module: Any, compile_kwargs: Optional[dict[str, Any]]) -> None:
    if hasattr(module, "compile"):
        module.compile(**(compile_kwargs or {}))


def _compile_callable(fn: Any, compile_kwargs: Optional[dict[str, Any]]) -> Any:
    if _HAS_TORCH and hasattr(torch, "compile"):
        return torch.compile(fn, **(compile_kwargs or {}))
    return fn


def _is_cudagraph_compile_mode(compile_kwargs: Optional[dict[str, Any]]) -> bool:
    if not isinstance(compile_kwargs, dict):
        return False
    mode = compile_kwargs.get("mode")
    if mode in ("reduce-overhead", "max-autotune"):
        return True
    options = compile_kwargs.get("options")
    return bool(isinstance(options, dict) and options.get("triton.cudagraphs"))


def _mark_cudagraph_step_begin() -> None:
    compiler = getattr(torch, "compiler", None) if _HAS_TORCH else None
    mark_fn = getattr(compiler, "cudagraph_mark_step_begin", None)
    if callable(mark_fn):
        mark_fn()


def _clone_cache_value(value: Any) -> Any:
    if value is None:
        return None
    if _HAS_TORCH and isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_cache_value(v) for v in value)
    if isinstance(value, list):
        return [_clone_cache_value(v) for v in value]
    return value


def _call_kick_attn(
    attn_mod: Any,
    x: Any,
    past_kv: Optional[Any],
    is_dense_with_hub: bool,
) -> Tuple[Any, Optional[Any]]:
    if isinstance(attn_mod, LatentKVAttention):
        return attn_mod(x, past_kv=past_kv, is_dense_with_hub=is_dense_with_hub)
    res = attn_mod(x) if past_kv is None else attn_mod(x, past_kv=past_kv)
    if isinstance(res, tuple) and len(res) == 2:
        return res[0], res[1]
    return res, None


def _capture_autocast_state(device_type: str) -> Tuple[str, bool, Optional[Any]]:
    if hasattr(torch, "is_autocast_enabled"):
        try:
            enabled = bool(torch.is_autocast_enabled(device_type))
        except TypeError:
            enabled = bool(torch.is_autocast_cpu_enabled()) if device_type == "cpu" else bool(torch.is_autocast_enabled())
    else:
        enabled = False
    dtype: Optional[Any] = None
    if enabled and hasattr(torch, "get_autocast_dtype"):
        dtype = torch.get_autocast_dtype(device_type)
    return device_type, enabled, dtype


def _autocast_ctx(state: Tuple[str, bool, Optional[Any]]) -> Any:
    device_type, enabled, dtype = state
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=True)


def _capture_rng_state(device: Any) -> Tuple[Any, Optional[Any]]:
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu_state, cuda_state


def _restore_rng_state(device: Any, state: Tuple[Any, Optional[Any]]) -> None:
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


if _HAS_TORCH:

    class _ReversibleDKDFunction(torch.autograd.Function):
        """Autograd function for reversible DKD that reconstructs intermediate states analytically during backward."""

        @staticmethod
        def forward(
            ctx: Any,
            q: Any,
            p: Any,
            block: Any,
            is_dense_with_hub: bool,
            *params: Any,
        ) -> Tuple[Any, Any]:
            device = q.device
            ctx.autocast_state = _capture_autocast_state(device.type)
            with torch.no_grad():
                half_dt = 0.5 * block.dt
                rng_f1 = _capture_rng_state(device)
                q_half = q + half_dt * block._run_drift_ffn(block.norm_p(p))
                rng_a = _capture_rng_state(device)
                attn_out, _ = block._run_kick_attn(block.norm_q(q_half), None, is_dense_with_hub)
                p_next = p + block.dt * attn_out
                rng_f2 = _capture_rng_state(device)
                q_next = q_half + half_dt * block._run_drift_ffn(block.norm_p(p_next))
            ctx.block = block
            ctx.is_dense_with_hub = is_dense_with_hub
            ctx.rng_states = (rng_f1, rng_a, rng_f2)
            ctx.save_for_backward(q_next.detach(), p_next.detach())
            return q_next, p_next

        @staticmethod
        def backward(ctx: Any, dq_next: Any, dp_next: Any) -> Tuple[Any, ...]:  # type: ignore[override]
            block = ctx.block
            is_dense_with_hub = ctx.is_dense_with_hub
            q_next, p_next = ctx.saved_tensors
            rng_f1, rng_a, rng_f2 = ctx.rng_states
            device = q_next.device
            saved_rng = _capture_rng_state(device)
            half_dt = 0.5 * block.dt

            f_params = tuple(p for p in (*block.norm_p.parameters(), *block.drift_ffn.parameters()) if p.requires_grad)
            a_params = tuple(p for p in (*block.norm_q.parameters(), *block.kick_attn.parameters()) if p.requires_grad)
            all_params = tuple(block.parameters())
            # Populate grad_map on demand to skip per-layer zeros_like allocations.
            grad_map: dict[int, Any] = {}

            try:
                # Reverse step 3: q_next = q_half + 0.5 * dt * F(p_next)
                p_next_req = p_next.detach().requires_grad_(True)
                _restore_rng_state(device, rng_f2)
                with torch.enable_grad(), _autocast_ctx(ctx.autocast_state):
                    f2 = half_dt * block._run_drift_ffn(block.norm_p(p_next_req))
                if f2.requires_grad:
                    grads_f2 = torch.autograd.grad(f2, (p_next_req,) + f_params, dq_next, allow_unused=True)
                    dp_next_total = dp_next + grads_f2[0] if grads_f2[0] is not None else dp_next
                    for pr, g in zip(f_params, grads_f2[1:]):
                        if g is not None:
                            grad_map[id(pr)] = g
                else:
                    dp_next_total = dp_next

                with torch.no_grad():
                    q_half = q_next - f2.detach()
                del f2, p_next_req

                # Reverse step 2: p_next = p_in + dt * A(q_half)
                q_half_req = q_half.detach().requires_grad_(True)
                _restore_rng_state(device, rng_a)
                with torch.enable_grad(), _autocast_ctx(ctx.autocast_state):
                    a_out, _ = block._run_kick_attn(block.norm_q(q_half_req), None, is_dense_with_hub)
                    a_scaled = block.dt * a_out
                if a_scaled.requires_grad:
                    grads_a = torch.autograd.grad(a_scaled, (q_half_req,) + a_params, dp_next_total, allow_unused=True)
                    dq_half_total = dq_next + grads_a[0] if grads_a[0] is not None else dq_next
                    for pr, g in zip(a_params, grads_a[1:]):
                        if g is not None:
                            prev = grad_map.get(id(pr))
                            grad_map[id(pr)] = g if prev is None else (prev + g)
                else:
                    dq_half_total = dq_next

                with torch.no_grad():
                    p_in = p_next - a_scaled.detach()
                del a_out, a_scaled, q_half_req, q_half

                # Reverse step 1: q_half = q_in + 0.5 * dt * F(p_in)
                p_in_req = p_in.detach().requires_grad_(True)
                _restore_rng_state(device, rng_f1)
                with torch.enable_grad(), _autocast_ctx(ctx.autocast_state):
                    f1 = half_dt * block._run_drift_ffn(block.norm_p(p_in_req))
                if f1.requires_grad:
                    grads_f1 = torch.autograd.grad(f1, (p_in_req,) + f_params, dq_half_total, allow_unused=True)
                    dp_in = dp_next_total + grads_f1[0] if grads_f1[0] is not None else dp_next_total
                    for pr, g in zip(f_params, grads_f1[1:]):
                        if g is not None:
                            prev = grad_map.get(id(pr))
                            grad_map[id(pr)] = g if prev is None else (prev + g)
                else:
                    dp_in = dp_next_total
                del f1, p_in_req, p_in
            finally:
                _restore_rng_state(device, saved_rng)

            param_grads = tuple(grad_map.get(id(pr)) for pr in all_params)
            return (dq_half_total, dp_in, None, None) + param_grads

else:
    _ReversibleDKDFunction = None  # type: ignore[assignment]


__all__ = [
    "SelectiveReversibleType",
    "resolve_selective_reversible",
    "_apply_inplace_compile",
    "_compile_callable",
    "_is_cudagraph_compile_mode",
    "_mark_cudagraph_step_begin",
    "_clone_cache_value",
    "_call_kick_attn",
    "_ReversibleDKDFunction",
]
