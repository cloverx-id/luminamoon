from __future__ import annotations

from .._version import __version__ as PACKAGE_VERSION

MATH_VERSION: str = "1.6.0"


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
