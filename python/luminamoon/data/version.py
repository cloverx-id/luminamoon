from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List

try:
    from .._version import __version__ as DEFAULT_VERSION
except (ImportError, ValueError):
    try:
        from luminamoon._version import __version__ as DEFAULT_VERSION
    except Exception:
        DEFAULT_VERSION = "1.1.1"

SUPPORTED_VERSIONS: List[str] = [
    "1.1.1",
    "1.1.0.post3",
    "1.1.0.post2",
    "1.1.0.post1",
    "1.1.0",
    "1.0.9",
    "1.0.8.post1",
    "1.0.8",
    "1.0.7",
    "1.0.6.post1",
    "1.0.6",
    "1.0.5.post1",
    "1.0.5",
    "1.0.4",
    "1.0.3",
    "1.0.2.post2",
    "1.0.2",
    "1.0.1",
    "1.0.0.post1",
    "1.0.0",
]

if DEFAULT_VERSION not in SUPPORTED_VERSIONS:
    SUPPORTED_VERSIONS.insert(0, DEFAULT_VERSION)

GITHUB_REPO: str = "cloverx-id/luminamoon"
CACHE_ROOT: Path = Path.home() / ".cache" / "luminamoon" / "versions"

VERSION_TAG_MAP: Dict[str, str] = {
    "1.1.1": "v1.1.1",
    "1.1.0.post3": "v1.1.0.post3",
    "1.1.0.post2": "v1.1.0.post2",
    "1.1.0.post1": "v1.1.0.post1",
    "1.1.0": "v1.1.0",
    "1.0.9": "v1.0.9",
    "1.0.8.post1": "v1.0.8.post1",
    "1.0.8": "v1.0.8",
    "1.0.7": "v1.0.7",
    "1.0.6.post1": "v1.0.6.post1",
    "1.0.6": "v1.0.6",
    "1.0.5.post1": "v1.0.5.post1",
    "1.0.5": "v1.0.5",
    "1.0.4": "v1.0.4",
    "1.0.3": "v1.0.3",
    "1.0.2.post2": "v1.0.2.post2",
    "1.0.2": "v1.0.2",
    "1.0.1": "v1.0.1",
    "1.0.0.post1": "v1.0.0.post1",
    "1.0.0": "v1.0.0",
}


def get_version() -> str:
    """Return the currently active LuminaMoon Data version string."""
    return DEFAULT_VERSION


def list_versions() -> List[str]:
    """Return all supported LuminaMoon Data version strings.

    Note: Historical specification versioning for the data engine is legacy.
    LuminaMoon Data operates on a single unified native SIMD engine.
    """
    warnings.warn(
        "luminamoon.data.list_versions() is legacy. The data engine operates on "
        "a single unified native SIMD engine. For package release information, "
        "use `luminamoon.__version__`.",
        DeprecationWarning,
        stacklevel=2,
    )
    return list(SUPPORTED_VERSIONS)


def set_version(version: str) -> str:
    """Validate data version compatibility.

    Deprecated: LuminaMoon Data operates on a single unified native SIMD engine.
    This function is maintained for backward compatibility.
    """
    warnings.warn(
        "luminamoon.data.set_version() is deprecated and has no effect. "
        "The data engine always operates on the unified native SIMD implementation. "
        "For algorithm specification locking, use `luminamoon.optimizer.LuminaV`.",
        DeprecationWarning,
        stacklevel=2,
    )
    ver = version.strip().lstrip("v")
    if ver not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"Unsupported LuminaMoon Data version '{version}'. "
            f"Supported versions: {SUPPORTED_VERSIONS}"
        )

    return DEFAULT_VERSION


def get_active_data_module() -> Any:
    """Retrieve the active data module."""
    import luminamoon.data as current_mod

    return current_mod


__all__ = [
    "CACHE_ROOT",
    "DEFAULT_VERSION",
    "GITHUB_REPO",
    "SUPPORTED_VERSIONS",
    "VERSION_TAG_MAP",
    "get_active_data_module",
    "get_version",
    "list_versions",
    "set_version",
]
