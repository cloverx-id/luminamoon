from __future__ import annotations

from typing import Any

from .._version import __version__ as PACKAGE_VERSION
from .migration import removed_version_attr_message

MATH_VERSION: str = "2.0.0"


def get_math_version() -> str:
    """Return the latest active LuminaV optimizer math version."""
    return MATH_VERSION


def get_package_version() -> str:
    """Return the installed LuminaMoon package version."""
    return PACKAGE_VERSION


__all__ = [
    "MATH_VERSION",
    "PACKAGE_VERSION",
    "get_math_version",
    "get_package_version",
]


def __getattr__(name: str) -> Any:
    msg = removed_version_attr_message(name)
    if msg is not None:
        raise AttributeError(msg)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
