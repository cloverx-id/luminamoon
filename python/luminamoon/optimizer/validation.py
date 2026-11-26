from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple

try:
    import torch as torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import numpy as _np
    np = _np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


def _no_grad_decorator(fn: Callable) -> Callable:
    if HAS_TORCH and torch is not None and hasattr(torch, "no_grad"):
        return torch.no_grad()(fn)
    return fn


def _is_torch_compiling() -> bool:
    return bool(
        HAS_TORCH
        and torch is not None
        and hasattr(torch, "compiler")
        and hasattr(torch.compiler, "is_compiling")
        and torch.compiler.is_compiling()
    )


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
    z = (indices.astype(np.uint32) + np.uint32(seed) - np.uint32(1640531527)).astype(np.uint32)
    z = ((z ^ (z >> np.uint32(16))) * np.uint32(0x85EB5A6B)).astype(np.uint32)
    z = ((z ^ (z >> np.uint32(13))) * np.uint32(0xC2B2AE35)).astype(np.uint32)
    return (z ^ (z >> np.uint32(16))).view(np.int32)


def _prng_weyl_numpy(seed: int, indices: Any, step_offset: int = 0) -> Any:
    h0 = _prng_numpy(seed, indices).view(np.uint32)
    return (h0 + np.uint32(step_offset & 0xFFFFFFFF)).astype(np.uint32)


def _bf16_bits_from_f32_numpy(val_f32: Any, use_sr: bool, seed: int, step_offset: int = 0) -> Any:
    work = np.ascontiguousarray(val_f32, dtype=np.float32).copy()
    p_int = work.view(np.int32).reshape(-1)
    abs_int = p_int & np.int32(0x7FFFFFFF)
    is_nan = abs_int > np.int32(0x7F800000)
    sign_bit = p_int & np.int32(-2147483648)
    if use_sr:
        offsets = np.arange(p_int.size, dtype=np.int32)
        noise = (_prng_weyl_numpy(seed, offsets, step_offset) >> np.uint32(16)).astype(np.int32)
        added = (abs_int + noise) & np.int32(-65536)
        rounded_abs = np.where((added > np.int32(0x7F7F0000)) | (added < 0), np.int32(0x7F7F0000), added)
    else:
        lsb = (abs_int >> np.int32(16)) & np.int32(1)
        added = (abs_int + np.int32(0x7FFF) + lsb) & np.int32(-65536)
        rounded_abs = np.where((added > np.int32(0x7F7F0000)) | (added < 0), np.int32(0x7F7F0000), added)
    out_int = np.where(is_nan, (p_int | np.int32(0x00400000)) & np.int32(-65536), sign_bit | rounded_abs).astype(np.int32)
    return (out_int.view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def _quantize_numpy(target: Any, val_f32: Any, use_sr: bool, seed: int, step_offset: int = 0) -> None:
    if not hasattr(target, "dtype") or target.dtype == np.float32:
        target[...] = val_f32
        return
    dtype_str = str(target.dtype).lower()
    if "bfloat16" in dtype_str:
        out_u16 = _bf16_bits_from_f32_numpy(val_f32, use_sr, seed, step_offset=step_offset)
        target.view(np.uint16)[...] = out_u16.reshape(target.shape)
    elif "float16" in dtype_str or dtype_str in ("f2", "<f2", "=f2"):
        if not use_sr:
            target[...] = np.clip(val_f32, -65504.0, 65504.0).astype(target.dtype)
            return
        work = np.ascontiguousarray(val_f32, dtype=np.float32).copy()
        p_int = work.view(np.int32).reshape(-1)
        offsets = np.arange(p_int.size, dtype=np.int32)
        abs_int = p_int & 0x7FFFFFFF
        is_nan = abs_int > np.int32(0x7F800000)
        exp_f32 = (abs_int >> 23) & 0xFF
        sub_shift = np.clip(113 - exp_f32, 0, 11)
        total_shift = 13 + sub_shift
        mask_bits = (np.left_shift(1, total_shift) - 1).astype(np.int32)
        rshift = (np.uint32(32) - total_shift.astype(np.uint32)).astype(np.uint32)
        noise = (_prng_weyl_numpy(seed, offsets, step_offset) >> rshift).astype(np.int32)
        rounded_abs = np.where(exp_f32 < 103, np.int32(0), (abs_int + noise) & (~mask_bits))
        rounded_abs = np.minimum(rounded_abs, np.int32(0x477FE000))
        sign_bit = p_int & np.int32(-2147483648)
        out_int = np.where(is_nan, p_int | np.int32(0x00400000), sign_bit | rounded_abs).astype(np.int32)
        target[...] = out_int.view(np.float32).astype(target.dtype).reshape(target.shape)
    else:
        target[...] = val_f32.astype(target.dtype)


def _decode_state_numpy(state_arr: Any) -> Any:
    if not hasattr(state_arr, "dtype"):
        return np.array(state_arr, dtype=np.float32, copy=True)
    dtype_str = str(state_arr.dtype).lower()
    if "bfloat16" in dtype_str or "float16" in dtype_str or dtype_str in ("f2", "<f2", "=f2"):
        contig = np.ascontiguousarray(state_arr)
        bits_u32 = contig.view(np.uint16).astype(np.uint32) << np.uint32(16)
        return bits_u32.view(np.float32).reshape(state_arr.shape).copy()
    return np.array(state_arr, dtype=np.float32, copy=True)


def _quantize_state_numpy(
    target: Any, val_f32: Any, use_sr: bool, seed: int, step_offset: int = 0
) -> None:
    if not hasattr(target, "dtype") or target.dtype == np.float32:
        target[...] = val_f32
        return
    dtype_str = str(target.dtype).lower()
    if "bfloat16" in dtype_str or "float16" in dtype_str or dtype_str in ("f2", "<f2", "=f2"):
        out_u16 = _bf16_bits_from_f32_numpy(val_f32, use_sr, seed, step_offset=step_offset)
        target.view(np.uint16)[...] = out_u16.reshape(target.shape)
    else:
        target[...] = val_f32.astype(target.dtype)


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
