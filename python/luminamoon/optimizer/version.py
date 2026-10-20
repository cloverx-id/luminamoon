from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import urllib.error
import urllib.request

from .._version import __version__ as CARGO_VERSION

DEFAULT_LUMINAV_VERSION: str = CARGO_VERSION
REFERENCE_LUMINAV_VERSION: str = "1.5.3"
SUPPORTED_BUNDLED_VERSIONS = frozenset({CARGO_VERSION, REFERENCE_LUMINAV_VERSION, "latest", "default"})


def validate_version(version: str) -> str:
    """Validates and normalizes LuminaV version.

    Supports 'latest', dynamic Cargo SSOT version, canonical reference '1.5.3',
    or ANY historical / legacy semantic version (e.g. '0.1.0', '0.9.0', '1.0.0').
    """
    v_clean = str(version).strip().lower()
    if v_clean in ("latest", "default"):
        return CARGO_VERSION
    if v_clean in ("reference", "canonical", "baseline"):
        return REFERENCE_LUMINAV_VERSION

    if v_clean.startswith("v"):
        v_clean = v_clean[1:]

    parts = v_clean.split(".")
    try:
        nums = [int(p) for p in parts]
    except ValueError as err:
        raise ValueError(
            f"Invalid LuminaV version format: '{version}'. Expected semantic version e.g. '1.5.3' or '0.9.0'."
        ) from err

    if len(nums) < 2:
        raise ValueError(
            f"Invalid LuminaV version format: '{version}'. Expected at least major.minor e.g. '1.5.3'."
        )

    while len(nums) < 3:
        nums.append(0)

    if any(n < 0 for n in nums):
        raise ValueError(f"Version components cannot be negative: '{version}'")

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
    """Manages version resolution, local cache directories, modular downloading, and legacy compatibility."""

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
    def is_bundled(cls, version: str) -> bool:
        v = validate_version(version)
        return v in SUPPORTED_BUNDLED_VERSIONS or v == CARGO_VERSION or v == REFERENCE_LUMINAV_VERSION

    @classmethod
    def is_available(cls, version: str) -> bool:
        v = validate_version(version)
        if cls.is_bundled(v):
            return True
        return cls.get_version_dir(v).is_dir()

    @classmethod
    def download_version(cls, version: str, timeout: int = 5) -> Path:
        """Downloads or initializes local profile for the specified modular version."""
        v = validate_version(version)
        v_dir = cls.get_version_dir(v)
        v_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = v_dir / "manifest.json"

        # Attempt to download release asset from GitHub releases if available
        url = f"https://github.com/cloverx-id/luminamoon/releases/download/v{v}/luminamoon-v{v}-modular.tar.gz"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LuminaMoon-VersionResolver"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                (v_dir / "bundle.tar.gz").write_bytes(data)
                manifest = {"version": v, "source": "remote_release", "downloaded": True}
        except Exception:
            # Offline or unreleased legacy version: write local compatibility descriptor
            manifest = {
                "version": v,
                "source": "local_compatibility_profile",
                "downloaded": False,
                "status": "ready",
            }

        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return v_dir

    @classmethod
    def ensure_version_ready(cls, version: str) -> Path:
        """Ensures that the requested version (whether older, reference, or new) is prepared and accessible."""
        v = validate_version(version)
        target_dir = cls.get_version_dir(v)
        if cls.is_bundled(v):
            return target_dir
        if target_dir.is_dir():
            return target_dir
        return cls.download_version(v)

    @classmethod
    def set_default_version(cls, version: str) -> str:
        v = validate_version(version)
        cls._active_default = v
        cls.ensure_version_ready(v)
        return v

    @classmethod
    def get_default_version(cls) -> str:
        return cls._active_default

    @classmethod
    def list_available_versions(cls) -> List[str]:
        versions = list(SUPPORTED_BUNDLED_VERSIONS - {"latest", "default"})
        cache_root = cls.get_cache_root()
        if cache_root.is_dir():
            for child in cache_root.iterdir():
                if child.is_dir() and child.name not in versions:
                    versions.append(child.name)
        return sorted(versions)


__all__ = [
    "DEFAULT_LUMINAV_VERSION",
    "HybridVersionMethod",
    "ModularVersionResolver",
    "REFERENCE_LUMINAV_VERSION",
    "SUPPORTED_BUNDLED_VERSIONS",
    "validate_version",
]
