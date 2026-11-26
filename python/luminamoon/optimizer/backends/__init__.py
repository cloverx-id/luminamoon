from __future__ import annotations

from .autotune import TuneStrategy, benchmark_backends, reset_autotune_cache, resolve_auto_execution
from .numpy_backend import step_numpy_group
from .probe import (
    backend_unavailable_message,
    dispatch_cpu_batch,
    probe_all_backends,
    probe_triton_runtime,
    reset_probe_caches,
    triton_install_spec,
)

try:
    from .gpu import step_gpu_group
    from .triton import HAS_TRITON, prewarm_all_configs
except ImportError:
    HAS_TRITON = False
    step_gpu_group = None  # type: ignore[assignment]
    prewarm_all_configs = None  # type: ignore[assignment]

__all__ = [
    "HAS_TRITON",
    "TuneStrategy",
    "backend_unavailable_message",
    "benchmark_backends",
    "dispatch_cpu_batch",
    "prewarm_all_configs",
    "probe_all_backends",
    "probe_triton_runtime",
    "reset_autotune_cache",
    "reset_probe_caches",
    "resolve_auto_execution",
    "step_gpu_group",
    "step_numpy_group",
    "triton_install_spec",
]
