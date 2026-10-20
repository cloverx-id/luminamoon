from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .._version import __version__ as CARGO_VERSION

DEFAULT_LUMINAV_VERSION: str = CARGO_VERSION
REFERENCE_LUMINAV_VERSION: str = "1.5.3"
SUPPORTED_BUNDLED_VERSIONS = frozenset({CARGO_VERSION, REFERENCE_LUMINAV_VERSION, "latest", "default"})


def validate_version(version: str) -> str:
    """Validates and normalizes LuminaV version.

    Accepts 'latest', the dynamic Cargo version, or canonical reference version '1.5.3'.
    """
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
            f"Invalid LuminaV version format: '{version}'. Expected semantic version e.g. '1.5.3'."
        ) from err

    if len(nums) < 2:
        raise ValueError(
            f"Invalid LuminaV version format: '{version}'. Expected at least major.minor e.g. '1.5.3'."
        )

    while len(nums) < 3:
        nums.append(0)

    parsed_tuple = (nums[0], nums[1], nums[2])
    # Baseline tracking: versions below 1.0.0 are not supported
    if parsed_tuple < (1, 0, 0):
        raise ValueError(
            f"Unsupported LuminaV version '{version}'. Versions below 1.0.0 are not tracked or supported."
        )

    return f"{nums[0]}.{nums[1]}.{nums[2]}"


class HybridVersionMethod:
    """Descriptor supporting both instance-level (opt.setVersion) and class-level (LuminaV.setVersion) invocation."""

    def __init__(self, fn_inst: Callable[..., Any], fn_cls: Callable[..., Any]):
        self.fn_inst = fn_inst
        self.fn_cls = fn_cls

    def __get__(self, obj: Any, objtype: Any = None) -> Callable[..., Any]:
        if obj is None:
            if hasattr(self.fn_cls, "__get__"):
                return self.fn_cls.__get__(None, objtype)
            return lambda *args, **kwargs: self.fn_cls(objtype, *args, **kwargs)
        return lambda *args, **kwargs: self.fn_inst(obj, *args, **kwargs)


class ModularVersionResolver:
    """Manages version resolution, local cache directories, and on-demand modular loading."""

    _active_default: str = DEFAULT_LUMINAV_VERSION

    @classmethod
    def get_cache_root(cls) -> Path:
        if os.name == "nt":
            base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
            return Path(base) / "luminamoon" / "versions"
        cache_dir = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        return Path(cache_dir) / "luminamoon" / "versions"

    @classmethod
    def get_version_dir(cls, version: str) -> Path:
        v = validate_version(version)
        return cls.get_cache_root() / v

    @classmethod
    def is_available(cls, version: str) -> bool:
        v = validate_version(version)
        if v in SUPPORTED_BUNDLED_VERSIONS or v == CARGO_VERSION or v == REFERENCE_LUMINAV_VERSION:
            return True
        return cls.get_version_dir(v).is_dir()

    @classmethod
    def set_default_version(cls, version: str) -> str:
        v = validate_version(version)
        cls._active_default = v
        return v

    @classmethod
    def get_default_version(cls) -> str:
        return cls._active_default


__all__ = [
    "DEFAULT_LUMINAV_VERSION",
    "HybridVersionMethod",
    "ModularVersionResolver",
    "REFERENCE_LUMINAV_VERSION",
    "SUPPORTED_BUNDLED_VERSIONS",
    "validate_version",
]
