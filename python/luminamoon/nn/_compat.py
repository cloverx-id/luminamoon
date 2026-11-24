from __future__ import annotations

import math
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
    import triton as triton  # type: ignore[import-untyped]
    _HAS_TRITON = True
except ImportError:
    triton = None  # type: ignore[assignment]
    _HAS_TRITON = False

# 4. Automatic Rust SIMD Engine Detection
_rs_engine: Any = None
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


_HAS_CUBECL: Optional[bool] = None
_CUBECL_NN_MIN_ELEMENTS: int = 65_536


def is_rust_simd_available() -> bool:
    """Returns True if compiled Rust SIMD extension is available."""
    return _HAS_RUST_SIMD


def is_cubecl_available() -> bool:
    """Returns True if the compiled CubeCL GPU backend is available on this host."""
    global _HAS_CUBECL
    if _HAS_CUBECL is None:
        if _rs_engine is not None and hasattr(_rs_engine, "is_cubecl_available_pyo3"):
            try:
                _HAS_CUBECL = bool(_rs_engine.is_cubecl_available_pyo3())
            except Exception:
                _HAS_CUBECL = False
        else:
            _HAS_CUBECL = False
    return _HAS_CUBECL


def _can_use_cubecl_nn(numel: int, fn_name: str) -> bool:
    # Only offload large non-CUDA tensors so PCIe/UMA transfer cost is amortized.
    return (
        numel >= _CUBECL_NN_MIN_ELEMENTS
        and _rs_engine is not None
        and hasattr(_rs_engine, fn_name)
        and is_cubecl_available()
    )


def _is_torch_compiling() -> bool:
    """Returns True when TorchDynamo / torch.compile is actively tracing."""
    if not _HAS_TORCH:
        return False
    compiler_mod = getattr(torch, "compiler", None)
    if compiler_mod is not None and hasattr(compiler_mod, "is_compiling"):
        return bool(compiler_mod.is_compiling())
    dynamo_mod = getattr(torch, "_dynamo", None)
    if dynamo_mod is not None and hasattr(dynamo_mod, "is_compiling"):
        return bool(dynamo_mod.is_compiling())
    return False


def _as_numpy(val: Any) -> Any:
    if val is None:
        return None
    if _HAS_TORCH and isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    return val


def _init_linear_np(in_features: int, out_features: int, bias: bool = False) -> Any:
    if not _HAS_NUMPY:
        return None
    bound = 1.0 / math.sqrt(max(1, int(in_features)))
    w = np.random.uniform(-bound, bound, size=(int(out_features), int(in_features))).astype(np.float32)
    b = np.zeros((int(out_features),), dtype=np.float32) if bias else None
    return (w, b)


def _linear_np(x: Any, layer: Any) -> Any:
    if layer is None:
        return x
    if isinstance(layer, tuple):
        w_np, b_np = layer
    else:
        w_np = _as_numpy(getattr(layer, "weight", None))
        b_np = _as_numpy(getattr(layer, "bias", None))
    out = np.matmul(x, w_np.T)
    if b_np is not None:
        out = out + b_np
    return out


class StandaloneModule:
    """
    Lightweight fallback base class when PyTorch is not installed.
    Allows Ra-Family components to be imported, instantiated, and inspected
    in zero-dependency environments (e.g. edge microcontrollers, web servers)
    without raising ModuleNotFoundError.
    """

    _training: bool
    _parameters: Dict[str, Any]
    _buffers: Dict[str, Any]
    _modules: Dict[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        object.__setattr__(self, "_training", True)
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "_modules", {})

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name.startswith("_"):
            return
        params = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        buffers = self.__dict__.get("_buffers")
        if params is None or modules is None or buffers is None:
            return
        if isinstance(value, StandaloneModule):
            modules[name] = value
            params.pop(name, None)
        elif isinstance(value, list) and value and all(isinstance(m, StandaloneModule) for m in value):
            for idx, m in enumerate(value):
                modules[f"{name}.{idx}"] = m
        elif isinstance(value, dict) and "weight" in value and _HAS_NUMPY and isinstance(value.get("weight"), np.ndarray):
            params[f"{name}.weight"] = value["weight"]
            if isinstance(value.get("bias"), np.ndarray):
                params[f"{name}.bias"] = value["bias"]
        elif _HAS_NUMPY and isinstance(value, np.ndarray) and name not in buffers:
            params[name] = value
        elif value is None:
            params.pop(name, None)
            modules.pop(name, None)

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
    "is_cubecl_available",
    "_HAS_TORCH",
    "_HAS_NUMPY",
    "_HAS_TRITON",
    "_HAS_RUST_SIMD",
    "_CUBECL_NN_MIN_ELEMENTS",
    "_can_use_cubecl_nn",
    "_is_torch_compiling",
    "_as_numpy",
    "_init_linear_np",
    "_linear_np",
    "_rs_engine",
    "torch",
    "nn",
    "np",
]
