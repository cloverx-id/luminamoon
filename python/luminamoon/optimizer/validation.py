from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


def _no_grad_decorator(fn: Callable) -> Callable:
    if HAS_TORCH and torch is not None and hasattr(torch, "no_grad"):
        return torch.no_grad()(fn)
    return fn


def _numel(x: Any) -> int:
    if hasattr(x, "numel"):
        return int(x.numel())
    if hasattr(x, "size"):
        sz: Any = getattr(x, "size")
        res: Any = sz() if callable(sz) else sz
        return int(res)
    if hasattr(x, "__len__"):
        return len(x)
    return 1


def _is_contiguous(x: Any) -> bool:
    if hasattr(x, "is_contiguous"):
        return bool(x.is_contiguous())
    if hasattr(x, "flags"):
        return bool(getattr(x.flags, "c_contiguous", True))
    return True


def _make_contiguous(x: Any) -> Any:
    if hasattr(x, "contiguous"):
        return x.contiguous()
    if HAS_NUMPY and np is not None and hasattr(x, "flags"):
        return np.ascontiguousarray(x)
    return x


def _validate_supported_dtype(tensor: Any, name: str = "param") -> None:
    if not hasattr(tensor, "dtype"):
        return
    dtype_str = str(tensor.dtype).lower()
    if (
        "float32" in dtype_str
        or dtype_str in ("f4", "<f4", "=f4")
        or "bfloat16" in dtype_str
        or "float16" in dtype_str
        or dtype_str in ("f2", "<f2", "=f2")
    ):
        return
    raise ValueError(
        f"Unsupported dtype '{dtype_str}' for '{name}'. Supported dtypes are "
        "torch.float32 (float32), torch.bfloat16 (bfloat16), and torch.float16 (float16)."
    )


def _prng_numpy(seed: int, indices: Any) -> Any:
    z = (indices.astype(np.int64) + int(seed) - 1640531527).astype(np.int32)
    z = (((z ^ (z >> 16)).astype(np.int64) * -2048173461) & 0xFFFFFFFF).astype(np.int32)
    z = (((z ^ (z >> 13)).astype(np.int64) * -1028477387) & 0xFFFFFFFF).astype(np.int32)
    return (z ^ (z >> 16)).astype(np.int32)


def _quantize_numpy(target: Any, val_f32: Any, use_sr: bool, seed: int) -> None:
    if not hasattr(target, "dtype") or target.dtype == np.float32:
        target[:] = val_f32
        return
    dtype_str = str(target.dtype).lower()
    if "float16" in dtype_str or dtype_str in ("f2", "<f2", "=f2"):
        if not use_sr:
            target[:] = np.clip(val_f32, -65504.0, 65504.0).astype(target.dtype)
            return
        work = np.ascontiguousarray(val_f32, dtype=np.float32).copy()
        p_int = work.view(np.int32).reshape(-1)
        offsets = np.arange(p_int.size, dtype=np.int32)
        abs_int = p_int & 0x7FFFFFFF
        exp_f32 = (abs_int >> 23) & 0xFF
        sub_shift = np.clip(113 - exp_f32, 0, 11)
        total_shift = 13 + sub_shift
        mask_bits = (np.left_shift(1, total_shift) - 1).astype(np.int32)
        noise = _prng_numpy(seed, offsets) & mask_bits
        rounded_abs = np.where(exp_f32 < 103, np.int32(0), (abs_int + noise) & (~mask_bits))
        rounded_abs = np.minimum(rounded_abs, np.int32(0x477FE000))
        sign_bit = p_int & np.int32(-2147483648)
        out_int = (sign_bit | rounded_abs).astype(np.int32)
        target[:] = out_int.view(np.float32).astype(target.dtype).reshape(target.shape)
    else:
        target[:] = val_f32.astype(target.dtype)


def _levenshtein(a: str, b: str) -> int:
    a_low = a.lower()
    b_low = b.lower()
    m = len(a_low)
    n = len(b_low)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(m):
        curr[0] = i + 1
        for j in range(n):
            cost = 0 if a_low[i] == b_low[j] else 1
            curr[j + 1] = min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost)
        prev, curr = curr, prev
    return prev[n]


def _suggest_closest(
    input_val: str, candidates: Tuple[str, ...], max_dist: int = 3
) -> Optional[str]:
    trimmed = input_val.strip().lower()
    if not trimmed:
        return None
    best_cand: Optional[str] = None
    best_dist = max_dist + 1
    for cand in candidates:
        dist = _levenshtein(trimmed, cand)
        if dist < best_dist:
            best_dist = dist
            best_cand = cand
        elif (
            dist > max_dist
            and len(trimmed) >= 2
            and (cand.startswith(trimmed) or trimmed.startswith(cand))
            and best_cand is None
        ):
            best_cand = cand
    return best_cand


def _is_valid_number(val: Any) -> bool:
    return (
        not isinstance(val, bool)
        and isinstance(val, (int, float))
        and math.isfinite(float(val))
    )
