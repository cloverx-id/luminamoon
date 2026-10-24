from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Callable, Dict, List, Type

from .._version import __version__ as CARGO_VERSION

ALGORITHM_SPEC_VERSION: str = "1.5.3"
REFERENCE_LUMINAV_VERSION: str = ALGORITHM_SPEC_VERSION
DEFAULT_LUMINAV_VERSION: str = REFERENCE_LUMINAV_VERSION

SUPPORTED_VERSIONS = [
    "1.5.3",
    "1.3.3",
    "1.3.2",
    "1.3.1",
    "1.3.0",
    "1.2.1",
    "1.2.0",
    "1.1.0",
    "1.0.0",
]

DEFAULT_VERSION = "1.5.3"
GITHUB_REPO = "cloverx-id/luminamoon"


def _resolve_default_cache_root() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "luminamoon" / "versions"
    cache_dir = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache_dir) / "luminamoon" / "versions"


CACHE_ROOT = _resolve_default_cache_root()

VERSION_TAG_MAP: Dict[str, str] = {
    "1.5.3": "main",
}

SUPPORTED_BUNDLED_VERSIONS = frozenset(
    {
        "1.5.3",
        "1.3.3",
        "1.3.2",
        "1.3.1",
        "1.3.0",
        "1.2.1",
        "1.2.0",
        "1.1.0",
        "1.0.0",
        CARGO_VERSION,
        "latest",
        "default",
    }
)

_ACTIVE_VERSION: str = DEFAULT_VERSION
_LOADED_MODULES: Dict[str, Any] = {}


def validate_version(version: str) -> str:
    """Validates and normalizes LuminaV algorithm specification version.

    Supports 'latest', canonical reference '1.5.3', dynamic package version,
    or ANY historical / legacy semantic version (e.g. '0.1.0', '0.9.0', '1.0.0', '1.3.3').
    """
    v_clean = str(version).strip().lower()
    if v_clean in ("latest", "default", "reference", "canonical", "baseline"):
        return REFERENCE_LUMINAV_VERSION
    if v_clean in ("cargo", "package"):
        return CARGO_VERSION

    if v_clean.startswith("v"):
        v_clean = v_clean[1:]

    m = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?(?:[.-]?(post\d+|dev\d+|rc\d+))?$", v_clean)
    if not m:
        raise ValueError(
            f"Invalid LuminaV version format: '{version}'. Expected semantic or PEP 440 version e.g. '1.5.3' or '1.0.2.post1'."
        )

    major, minor, patch, suffix = m.group(1), m.group(2), m.group(3) or "0", m.group(4) or ""
    if suffix:
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        return f"{major}.{minor}.{patch}{suffix}"
    return f"{major}.{minor}.{patch}"


def get_version() -> str:
    """Return the currently active LuminaV algorithm specification version string."""
    return _ACTIVE_VERSION


def list_versions() -> List[str]:
    """Return all supported LuminaV algorithm specification version strings."""
    return list(SUPPORTED_VERSIONS)


def _ensure_version_cached(version: str) -> Path:
    """Ensure the specified version's Python module files exist in local cache without network calls."""
    target_dir = CACHE_ROOT / version / "optimizer"
    init_file = target_dir / "__init__.py"
    if init_file.exists():
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    local_dir = Path(__file__).resolve().parent

    files_to_copy = [
        "__init__.py",
        "constants.py",
        "fsdp_sync.py",
        "gpu_backend.py",
        "luminav.py",
        "numpy_backend.py",
        "standalone.py",
        "torch_ops.py",
        "triton_kernels.py",
        "validation.py",
        "version.py",
    ]
    for fname in files_to_copy:
        local_src = local_dir / fname
        dest = target_dir / fname
        if local_src.exists():
            shutil.copyfile(local_src, dest)

    return target_dir


def set_version(version: str) -> str:
    """Set the active LuminaV algorithm specification version dynamically."""
    global _ACTIVE_VERSION
    ver = version.strip().lstrip("v")
    if ver not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"Unsupported LuminaMoon Optimizer version '{version}'. "
            f"Supported versions: {SUPPORTED_VERSIONS}"
        )

    if ver != DEFAULT_VERSION:
        _ensure_version_cached(ver)

    _ACTIVE_VERSION = ver
    return _ACTIVE_VERSION


def get_active_optimizer_class() -> Type[Any]:
    """Retrieve the LuminaV class for the currently active version."""
    if _ACTIVE_VERSION == DEFAULT_VERSION:
        from .luminav import LuminaV
        return LuminaV

    if _ACTIVE_VERSION in _LOADED_MODULES:
        return _LOADED_MODULES[_ACTIVE_VERSION].LuminaV

    cached_dir = _ensure_version_cached(_ACTIVE_VERSION)
    target_file = cached_dir / "luminav.py"
    if not target_file.exists():
        target_file = cached_dir / "core.py"
    if not target_file.exists():
        from .luminav import LuminaV
        return LuminaV

    pkg_name = f"_luminamoon_opt_v{_ACTIVE_VERSION.replace('.', '_')}"
    init_file = cached_dir / "__init__.py"
    if pkg_name not in sys.modules and init_file.exists():
        pkg_spec = importlib.util.spec_from_file_location(
            pkg_name,
            str(init_file),
            submodule_search_locations=[str(cached_dir)],
        )
        if pkg_spec is not None:
            pkg_mod = importlib.util.module_from_spec(pkg_spec)
            sys.modules[pkg_name] = pkg_mod

    module_name = f"{pkg_name}.luminav"
    spec = importlib.util.spec_from_file_location(module_name, str(target_file))
    if spec is None or spec.loader is None:
        from .luminav import LuminaV
        return LuminaV

    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg_name
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    _LOADED_MODULES[_ACTIVE_VERSION] = mod
    return mod.LuminaV


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
    """Manages version resolution, local cache directories, and offline legacy compatibility."""

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
        """Initializes an offline local compatibility profile for the specified modular version."""
        _ = timeout
        v = validate_version(version)
        v_dir = cls.get_version_dir(v)
        v_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = v_dir / "manifest.json"
        manifest = {
            "version": v,
            "algorithm_spec": ALGORITHM_SPEC_VERSION,
            "package_version": CARGO_VERSION,
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
        return list(SUPPORTED_VERSIONS)


__all__ = [
    "ALGORITHM_SPEC_VERSION",
    "CACHE_ROOT",
    "DEFAULT_LUMINAV_VERSION",
    "DEFAULT_VERSION",
    "GITHUB_REPO",
    "HybridVersionMethod",
    "ModularVersionResolver",
    "REFERENCE_LUMINAV_VERSION",
    "SUPPORTED_BUNDLED_VERSIONS",
    "SUPPORTED_VERSIONS",
    "VERSION_TAG_MAP",
    "get_active_optimizer_class",
    "get_version",
    "list_versions",
    "set_version",
    "validate_version",
]
