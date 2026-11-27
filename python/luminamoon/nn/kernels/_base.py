from __future__ import annotations

from collections import OrderedDict
import inspect
import threading
from typing import TYPE_CHECKING, Any, Callable

from .._compat import _HAS_TORCH, torch

if TYPE_CHECKING:
    HAS_TRITON: bool = False
    HAS_FDIV_APPROX: bool = False
    triton: Any = None
    tl: Any = None
else:
    try:
        import triton
        import triton.language as tl

        HAS_TRITON = True
        HAS_FDIV_APPROX = hasattr(tl, "fdiv") and ("approx" in inspect.signature(tl.fdiv).parameters)
    except ImportError:
        HAS_TRITON = False
        HAS_FDIV_APPROX = False
        triton = None
        tl = None

_CK_LOCK = threading.Lock()
_CK_MAX_ENTRIES = 256


def _dev_idx(dev: Any) -> int:
    return int(dev.index) if getattr(dev, "index", None) is not None else int(torch.cuda.current_device())


def _int_spec(v: int) -> int:
    iv = int(v)
    return 0 if iv == 1 else (1 if (iv & 15) == 0 else 2)


def _ptr_aligned(*tensors: Any) -> bool:
    return all((t.data_ptr() & 15) == 0 for t in tensors)


def _store_ck(cache: OrderedDict[Any, Any], key: Any, val: Any) -> None:
    with _CK_LOCK:
        cache[key] = val
        cache.move_to_end(key)
        while len(cache) > _CK_MAX_ENTRIES:
            cache.popitem(last=False)


def _launch_ck_direct(
    cache: OrderedDict[Any, Any],
    key: Any,
    dev: Any,
    grid: tuple[int, int, int],
    args: tuple[Any, ...],
) -> bool:
    with _CK_LOCK:
        ck = cache.get(key)
    if ck is None:
        return False
    try:
        with torch.cuda.device(dev):
            ck[grid](*args, stream=torch.cuda.current_stream(dev).cuda_stream)
        return True
    except Exception:
        with _CK_LOCK:
            cache.pop(key, None)
        return False


def _launch_ck_autotuned(
    kernel: Any,
    cache: OrderedDict[Any, Any],
    key: Any,
    dev: Any,
    grid_from_blk: Callable[[int], tuple[int, int, int]],
    grid_meta: Callable[[dict[str, Any]], tuple[int, ...]],
    meta_key: str,
    args: tuple[Any, ...],
    kw: dict[str, Any],
) -> None:
    with _CK_LOCK:
        entry = cache.get(key)
    if entry is not None:
        ck, blk = entry
        try:
            with torch.cuda.device(dev):
                ck[grid_from_blk(blk)](*args, stream=torch.cuda.current_stream(dev).cuda_stream)
            return
        except Exception:
            with _CK_LOCK:
                cache.pop(key, None)
    with _CK_LOCK:
        with torch.cuda.device(dev):
            ck = kernel[grid_meta](*args, **kw)
        best = getattr(kernel, "best_config", None)
        if best is not None and hasattr(ck, "__getitem__"):
            cache[key] = (ck, int(best.kwargs[meta_key]))
            cache.move_to_end(key)
            while len(cache) > _CK_MAX_ENTRIES:
                cache.popitem(last=False)


def _launch_attn_bwd_cached(
    kernel: Any, cache: OrderedDict[Any, Any], key: Any, bh: int, n_dim: int, meta_key: str,
    args: tuple[Any, ...], kw: dict[str, Any], dev: Any = None,
) -> None:
    d = dev if dev is not None else args[0].device
    _launch_ck_autotuned(
        kernel, cache, key, d,
        lambda blk: (bh, (n_dim + blk - 1) // blk, 1),
        lambda meta: (bh, triton.cdiv(n_dim, meta[meta_key])),
        meta_key, args, kw,
    )


__all__ = [
    "HAS_TRITON", "HAS_FDIV_APPROX", "_HAS_TORCH", "torch", "triton", "tl",
    "_dev_idx", "_int_spec", "_ptr_aligned", "_store_ck", "_launch_ck_direct",
    "_launch_ck_autotuned", "_launch_attn_bwd_cached",
]
