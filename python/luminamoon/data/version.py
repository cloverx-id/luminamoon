from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Tuple

from .._version import __version__ as CARGO_VERSION

DEFAULT_DATA_VERSION: str = CARGO_VERSION
REFERENCE_DATA_VERSION: str = "1.5.3"

_global_default_version: str = DEFAULT_DATA_VERSION


def validate_version(version: str) -> str:
    """Validates and normalizes data pipeline version."""
    v_clean = str(version).strip().lower()
    if v_clean in ("latest", "default"):
        return CARGO_VERSION
    if v_clean.startswith("v"):
        v_clean = v_clean[1:]
    parts = v_clean.split(".")
    try:
        nums = [int(p) for p in parts]
    except ValueError as err:
        raise ValueError(
            f"Invalid data version format: '{version}'. Expected semantic version e.g. '1.5.3'."
        ) from err
    if len(nums) < 2:
        raise ValueError(f"Invalid data version format: '{version}'. Expected at least major.minor e.g. '1.5.3'.")
    while len(nums) < 3:
        nums.append(0)
    parsed = (nums[0], nums[1], nums[2])
    if parsed < (1, 0, 0):
        raise ValueError(
            f"Unsupported data version '{version}'. Versions below 1.0.0 are not tracked."
        )
    return f"{nums[0]}.{nums[1]}.{nums[2]}"


def set_version(version: str) -> str:
    """Sets the global default version for luminamoon.data."""
    global _global_default_version
    _global_default_version = validate_version(version)
    return _global_default_version


def get_version() -> str:
    """Gets the global default version for luminamoon.data."""
    return _global_default_version


__all__ = [
    "DEFAULT_DATA_VERSION",
    "REFERENCE_DATA_VERSION",
    "get_version",
    "set_version",
    "validate_version",
]
