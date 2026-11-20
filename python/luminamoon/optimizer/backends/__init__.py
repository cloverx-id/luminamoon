from __future__ import annotations

from .numpy_backend import step_numpy_group

try:
    from .gpu import step_gpu_group
    from .triton import HAS_TRITON, prewarm_all_configs
except ImportError:
    HAS_TRITON = False
    step_gpu_group = None  # type: ignore[assignment]
    prewarm_all_configs = None  # type: ignore[assignment]

__all__ = [
    "HAS_TRITON",
    "prewarm_all_configs",
    "step_gpu_group",
    "step_numpy_group",
]
