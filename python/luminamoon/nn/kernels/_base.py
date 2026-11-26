from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

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

__all__ = ["HAS_TRITON", "HAS_FDIV_APPROX", "_HAS_TORCH", "torch", "triton", "tl"]
