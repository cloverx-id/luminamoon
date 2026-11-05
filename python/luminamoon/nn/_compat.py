from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# 1. Automatic PyTorch Detection
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _HAS_TORCH = False

# 2. Automatic NumPy Detection
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

# 3. Automatic Triton Detection
try:
    import triton  # type: ignore[import-untyped]
    _HAS_TRITON = True
except ImportError:
    triton = None  # type: ignore[assignment]
    _HAS_TRITON = False

# 4. Automatic Rust SIMD Engine Detection
try:
    try:
        from luminamoon.optimizer import _luminamoon_optimizer_rs as _rs_engine  # type: ignore[import-not-found]
    except ImportError:
        try:
            from luminamoon import _luminamoon_optimizer_rs as _rs_engine  # type: ignore[import-not-found]
        except ImportError:
            import _luminamoon_optimizer_rs as _rs_engine  # type: ignore[import-not-found]
    _HAS_RUST_SIMD = hasattr(_rs_engine, "ra_norm_f32_pyo3")
except ImportError:
    _rs_engine = None  # type: ignore[assignment]
    _HAS_RUST_SIMD = False


def is_torch_available() -> bool:
    """Returns True if PyTorch is installed and available in the environment."""
    return _HAS_TORCH


def is_numpy_available() -> bool:
    """Returns True if NumPy is installed and available in the environment."""
    return _HAS_NUMPY


def is_triton_available() -> bool:
    """Returns True if Triton GPU compiler is available."""
    return _HAS_TRITON


def is_rust_simd_available() -> bool:
    """Returns True if compiled Rust SIMD extension is available."""
    return _HAS_RUST_SIMD


class StandaloneModule:
    """
    Lightweight fallback base class when PyTorch is not installed.
    Allows Ra-Family components to be imported, instantiated, and inspected
    in zero-dependency environments (e.g. edge microcontrollers, web servers)
    without raising ModuleNotFoundError.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._training: bool = True
        self._parameters: Dict[str, Any] = {}
        self._buffers: Dict[str, Any] = {}
        self._modules: Dict[str, Any] = {}

    @property
    def training(self) -> bool:
        return self._training

    def train(self, mode: bool = True) -> StandaloneModule:
        self._training = mode
        for module in self._modules.values():
            if isinstance(module, StandaloneModule):
                module.train(mode)
        return self

    def eval(self) -> StandaloneModule:
        return self.train(False)

    def parameters(self) -> List[Any]:
        params = list(self._parameters.values())
        for module in self._modules.values():
            if hasattr(module, "parameters"):
                params.extend(module.parameters())
        return params

    def register_parameter(self, name: str, param: Any) -> None:
        self._parameters[name] = param

    def register_buffer(self, name: str, tensor: Any) -> None:
        self._buffers[name] = tensor

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Forward pass must be implemented by subclass.")

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        extra = self.extra_repr()
        if extra:
            return f"{cls_name}({extra})"
        return f"{cls_name}()"

    def extra_repr(self) -> str:
        return ""


if TYPE_CHECKING:
    import torch.nn as _nn_type  # type: ignore[import-not-found]
    BaseModule = _nn_type.Module
else:
    # BaseModule dynamically aliases torch.nn.Module when available,
    # or StandaloneModule when in a pure zero-dependency environment.
    BaseModule = nn.Module if _HAS_TORCH else StandaloneModule

__all__ = [
    "BaseModule",
    "StandaloneModule",
    "is_torch_available",
    "is_numpy_available",
    "is_triton_available",
    "is_rust_simd_available",
    "_HAS_TORCH",
    "_HAS_NUMPY",
    "_HAS_TRITON",
    "_HAS_RUST_SIMD",
    "_rs_engine",
    "torch",
    "nn",
    "np",
]
