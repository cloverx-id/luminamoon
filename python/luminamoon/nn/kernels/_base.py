from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._compat import _HAS_TORCH, torch

if TYPE_CHECKING:
    HAS_TRITON: bool = False
    triton: Any = None
    tl: Any = None
else:
    try:
        import triton
        import triton.language as tl

        HAS_TRITON = True
    except ImportError:
        HAS_TRITON = False
        triton = None
        tl = None

__all__ = ["HAS_TRITON", "_HAS_TORCH", "torch", "triton", "tl"]
