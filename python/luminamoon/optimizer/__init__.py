from __future__ import annotations

from typing import Any

from ..cli import help
from .config import BoundType, BufferMode, ExecutionMode, MasterWeights, VarianceType
from .luminav import LuminaV
from .migration import removed_optimizer_attr_message
from .version import MATH_VERSION, get_math_version

__all__ = [
    "BoundType",
    "BufferMode",
    "ExecutionMode",
    "LuminaV",
    "MATH_VERSION",
    "MasterWeights",
    "VarianceType",
    "get_math_version",
    "help",
]


def __getattr__(name: str) -> Any:
    msg = removed_optimizer_attr_message(name)
    if msg is not None:
        raise AttributeError(msg)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
