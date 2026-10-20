from __future__ import annotations

import re

import json
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from .._version import __version__ as CARGO_VERSION

REFERENCE_DATA_VERSION: str = "1.5.3"
DEFAULT_DATA_VERSION: str = REFERENCE_DATA_VERSION
SUPPORTED_BUNDLED_DATA_VERSIONS = frozenset({CARGO_VERSION, REFERENCE_DATA_VERSION, "latest", "default"})

_global_default_version: str = DEFAULT_DATA_VERSION


def validate_version(version: str) -> str:
    """Validates and normalizes data pipeline version.

    Supports 'latest', canonical reference '1.5.3', dynamic package version,
    or ANY historical / legacy semantic version (e.g. '0.1.0', '0.9.0', '1.0.0').
    """
    v_clean = str(version).strip().lower()
    if v_clean in ("latest", "default", "reference", "canonical", "baseline"):
        return REFERENCE_DATA_VERSION
    if v_clean in ("cargo", "package"):
        return CARGO_VERSION
    if v_clean.startswith("v"):
        v_clean = v_clean[1:]
    m = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?(?:[.-]?(post\d+|dev\d+|rc\d+))?$", v_clean)
    if not m:
        raise ValueError(
            f"Invalid data version format: '{version}'. Expected semantic or PEP 440 version e.g. '1.5.3' or '1.0.2.post1'."
        )

    major, minor, patch, suffix = m.group(1), m.group(2), m.group(3) or "0", m.group(4) or ""
    if suffix:
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        return f"{major}.{minor}.{patch}{suffix}"
    return f"{major}.{minor}.{patch}"


def set_version(version: str) -> str:
    """Sets the global default version for luminamoon.data (supports older, reference, and latest versions)."""
    global _global_default_version
    _global_default_version = validate_version(version)
    return _global_default_version


def get_version() -> str:
    """Gets the global default version for luminamoon.data."""
    return _global_default_version


__all__ = [
    "DEFAULT_DATA_VERSION",
    "REFERENCE_DATA_VERSION",
    "SUPPORTED_BUNDLED_DATA_VERSIONS",
    "get_version",
    "set_version",
    "validate_version",
]
