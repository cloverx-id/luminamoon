"""Backend dispatch and inherited context management for luminamoon.nn attention modules."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import threading
from typing import Any, Callable, Iterator, Literal, Optional, Tuple, Union

from ._compat import _HAS_TORCH, _is_torch_compiling, torch

AttentionBackend = Literal["auto", "sdpa", "ra", "flash", "cudnn", "fa3", "flex"]
ATTENTION_BACKENDS: Tuple[str, ...] = ("auto", "sdpa", "ra", "flash", "cudnn", "fa3", "flex")

_COMPILED_FLEX_ATTENTION: Optional[Callable[..., Any]] = None
# Serialize sdpa_kernel context transitions because PyTorch stores SDPA backend flags process-globally.
_SDPA_KERNEL_LOCK = threading.RLock()


def _get_flex_attention(is_cuda: bool) -> Callable[..., Any]:
    global _COMPILED_FLEX_ATTENTION
    try:
        from torch.nn.attention.flex_attention import flex_attention
    except ImportError as exc:
        raise ImportError("attention_backend='flex' requires torch.nn.attention.flex_attention.") from exc

    # Compile flex_attention once on CUDA so PyTorch generates a fused kernel instead of materializing scores.
    if is_cuda and not _is_torch_compiling() and hasattr(torch, "compile"):
        compiled = _COMPILED_FLEX_ATTENTION
        if compiled is None:
            compiled = torch.compile(flex_attention)
            _COMPILED_FLEX_ATTENTION = compiled
        return compiled
    return flex_attention

_ACTIVE_ATTENTION_BACKEND: ContextVar[Optional[Union[str, Callable[..., Any]]]] = ContextVar(
    "_luminamoon_attention_backend",
    default=None,
)


def validate_attention_backend(backend: str) -> AttentionBackend:
    """Validate and return an attention backend name."""
    if not isinstance(backend, str) or backend not in ATTENTION_BACKENDS:
        valid = ", ".join(repr(b) for b in ATTENTION_BACKENDS)
        raise ValueError(f"attention_backend must be one of ({valid}), got {backend!r}.")
    return backend  # type: ignore[return-value]


def get_attention_backend() -> Optional[Union[str, Callable[..., Any]]]:
    """Return the active inherited attention backend set by `use_attention_backend`, or `None`."""
    return _ACTIVE_ATTENTION_BACKEND.get()


@contextmanager
def use_attention_backend(
    backend: Union[AttentionBackend, Callable[..., Any]],
) -> Iterator[None]:
    """Temporarily set the attention backend inherited by modules with `attention_backend='auto'`.

    Accepts a backend name (`'auto'`, `'sdpa'`, `'ra'`, `'flash'`, `'cudnn'`, `'fa3'`, `'flex'`)
    or a callable with signature `(q, k, v, *, attn_mask, is_causal, scale)`. Passing `'auto'`
    clears the override inside the context so modules inherit PyTorch's outer SDPA configuration.
    """
    value: Optional[Union[str, Callable[..., Any]]]
    if callable(backend):
        value = backend
    else:
        validated = validate_attention_backend(backend)
        value = None if validated == "auto" else validated

    token = _ACTIVE_ATTENTION_BACKEND.set(value)
    try:
        yield
    finally:
        _ACTIVE_ATTENTION_BACKEND.reset(token)


def resolve_attention_backend(
    backend: str,
    attention_fn: Optional[Callable[..., Any]],
) -> Tuple[str, Optional[Callable[..., Any]]]:
    """Resolve explicit module settings and inherited context to `(backend_name, custom_fn)`."""
    if attention_fn is not None:
        return "custom", attention_fn
    if backend == "auto":
        inherited = _ACTIVE_ATTENTION_BACKEND.get()
        if callable(inherited):
            return "custom", inherited
        if isinstance(inherited, str):
            return inherited, None
        return "sdpa", None
    return backend, None


def _build_flex_score_mod(
    *,
    distance_bias: Optional[Any],
    hub_size: int,
    q_offset: int,
    is_dense_with_hub: bool,
    has_past_kv: bool,
    apply_causal: bool,
    device: Any,
) -> Optional[Callable[..., Any]]:
    if distance_bias is None and hub_size == 0 and not apply_causal:
        return None

    check_hub_q = hub_size > 0 and is_dense_with_hub and not has_past_kv
    q_off: Any = torch.tensor(q_offset, device=device, dtype=torch.int32) if q_offset != 0 else 0

    if distance_bias is None:
        if hub_size == 0:
            def _causal_only(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
                return torch.where(kv_idx <= (q_idx + q_off), score, float("-inf"))
            return _causal_only

        def _hub_causal(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
            is_hub_k = kv_idx < hub_size
            if check_hub_q:
                allowed = is_hub_k | ((q_idx >= hub_size) & (kv_idx <= (q_idx + q_off)))
            else:
                allowed = is_hub_k | (kv_idx <= (q_idx + q_off))
            return torch.where(allowed, score, float("-inf"))

        return _hub_causal

    slopes_1d = distance_bias.slopes.reshape(-1).to(device=device, dtype=torch.float32)
    is_rational = distance_bias.mode == "rational"

    if hub_size == 0:
        def _decay_only(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
            raw_dist = (q_idx + q_off - kv_idx).clamp(min=0).to(torch.float32)
            s_h = slopes_1d[h]
            if is_rational:
                bias = -torch.log((1.0 + s_h * raw_dist).clamp(min=1e-8))
            else:
                bias = -s_h * raw_dist
            bias = bias.clamp(min=-10.0, max=0.0).to(score.dtype)
            if apply_causal:
                return torch.where(kv_idx <= (q_idx + q_off), score + bias, float("-inf"))
            return score + bias

        return _decay_only

    hub_pen_1d = distance_bias.hub_penalty.reshape(-1).to(device=device, dtype=torch.float32)

    def _hub_decay(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        is_hub_k = kv_idx < hub_size
        raw_dist = (q_idx + q_off - kv_idx).clamp(min=0).to(torch.float32)
        s_h = slopes_1d[h]
        if is_rational:
            text_bias = -torch.log((1.0 + s_h * raw_dist).clamp(min=1e-8))
        else:
            text_bias = -s_h * raw_dist
        bias = torch.where(is_hub_k, -hub_pen_1d[h], text_bias).clamp(min=-10.0, max=0.0).to(score.dtype)
        if apply_causal:
            if check_hub_q:
                allowed = is_hub_k | ((q_idx >= hub_size) & (kv_idx <= (q_idx + q_off)))
            else:
                allowed = is_hub_k | (kv_idx <= (q_idx + q_off))
            return torch.where(allowed, score + bias, float("-inf"))
        return score + bias

    return _hub_decay


def _run_sdpa_forced(
    backend_name: str,
    q: Any,
    k: Any,
    v: Any,
    *,
    attn_mask: Optional[Any],
    is_causal: bool,
    scale: float,
) -> Any:
    if attn_mask is not None:
        raise ValueError(
            f"attention_backend={backend_name!r} requires attn_mask=None "
            "(use bias_mode='none' and hub_size=0, or switch to attention_backend='flex' or 'sdpa')."
        )
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError as exc:
        raise ImportError(f"attention_backend={backend_name!r} requires torch.nn.attention.sdpa_kernel.") from exc

    sdp_target = SDPBackend.FLASH_ATTENTION if backend_name == "flash" else getattr(SDPBackend, "CUDNN_ATTENTION", None)
    if sdp_target is None:
        raise RuntimeError("SDPBackend.CUDNN_ATTENTION is not available in this PyTorch build.")

    if _is_torch_compiling():
        with sdpa_kernel(sdp_target):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=is_causal, scale=scale
            )
    with _SDPA_KERNEL_LOCK, sdpa_kernel(sdp_target):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=is_causal, scale=scale
        )


def _run_fa3(
    q: Any,
    k: Any,
    v: Any,
    *,
    attn_mask: Optional[Any],
    is_causal: bool,
    scale: float,
) -> Any:
    if attn_mask is not None:
        raise ValueError(
            "attention_backend='fa3' requires attn_mask=None "
            "(use bias_mode='none' and hub_size=0, or switch to attention_backend='flex' or 'sdpa')."
        )
    flash_fn: Optional[Callable[..., Any]] = None
    try:
        import flash_attn_interface  # type: ignore[import-not-found]
        flash_fn = getattr(flash_attn_interface, "flash_attn_func", None)
    except ImportError:
        flash_fn = None
    if flash_fn is None:
        try:
            import flash_attn  # type: ignore[import-not-found]
            flash_fn = getattr(flash_attn, "flash_attn_func", None)
        except ImportError:
            flash_fn = None
    if flash_fn is None:
        raise ImportError(
            "attention_backend='fa3' requires `flash_attn_interface` (FlashAttention-3) or `flash_attn`."
        )

    # FlashAttention packages expect (B, S, H, D); transpose from/to (B, H, S, D).
    res = flash_fn(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        softmax_scale=scale,
        causal=is_causal,
    )
    out = res[0] if isinstance(res, tuple) else res
    return out.transpose(1, 2).contiguous()


def dispatch_attention(
    backend: str,
    attention_fn: Optional[Callable[..., Any]],
    q: Any,
    k: Any,
    v: Any,
    *,
    attn_mask: Optional[Any],
    is_causal: bool,
    scale: float,
    distance_bias: Optional[Any] = None,
    hub_size: int = 0,
    q_offset: int = 0,
    is_dense_with_hub: bool = True,
    has_past_kv: bool = False,
    apply_causal: bool = False,
) -> Any:
    """Execute attention via the resolved backend preset, inherited context, or custom callable."""
    if not _HAS_TORCH:
        raise ImportError("dispatch_attention requires PyTorch.")

    resolved, fn = resolve_attention_backend(backend, attention_fn)
    if resolved == "custom" and fn is not None:
        return fn(q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale)
    if resolved == "ra":
        if attn_mask is None and (distance_bias is not None or hub_size > 0):
            from ._attention_bias_ops import can_use_triton_ra_fused_bias, ra_attention_fused_bias

            if can_use_triton_ra_fused_bias(q, k, v, distance_bias, hub_size=hub_size):
                return ra_attention_fused_bias(
                    q,
                    k,
                    v,
                    distance_bias=distance_bias,
                    hub_size=hub_size,
                    q_offset=q_offset,
                    check_hub_q=bool(hub_size > 0 and is_dense_with_hub and not has_past_kv),
                    is_causal=bool(apply_causal or is_causal),
                    scale=scale,
                )
        from .attention import ra_attention
        return ra_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale)
    if resolved == "sdpa":
        # Do not override sdpa_kernel so external PyTorch SDPA contexts and flags are inherited.
        if _is_torch_compiling():
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale
            )
        with _SDPA_KERNEL_LOCK:
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale
            )
    if resolved in ("flash", "cudnn"):
        return _run_sdpa_forced(resolved, q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale)
    if resolved == "fa3":
        return _run_fa3(q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale)
    if resolved == "flex":
        if attn_mask is not None:
            raise ValueError(
                "attention_backend='flex' does not accept an explicit attn_mask; pass distance_bias/hub_size or use another backend."
            )
        flex_fn = _get_flex_attention(bool(q.is_cuda))
        score_mod = _build_flex_score_mod(
            distance_bias=distance_bias,
            hub_size=hub_size,
            q_offset=q_offset,
            is_dense_with_hub=is_dense_with_hub,
            has_past_kv=has_past_kv,
            apply_causal=bool(apply_causal or is_causal),
            device=q.device,
        )
        return flex_fn(q, k, v, score_mod=score_mod, scale=scale)

    raise ValueError(f"Unsupported attention backend: {resolved!r}.")


__all__ = [
    "AttentionBackend",
    "ATTENTION_BACKENDS",
    "get_attention_backend",
    "use_attention_backend",
    "validate_attention_backend",
    "resolve_attention_backend",
    "dispatch_attention",
]
