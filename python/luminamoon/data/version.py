from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List
import urllib.error
import urllib.request

SUPPORTED_VERSIONS = [
    "1.0.3",
    "1.0.2.post2",
    "1.0.2",
    "1.0.1",
    "1.0.0.post1",
    "1.0.0",
]

DEFAULT_VERSION = "1.0.3"
GITHUB_REPO = "cloverx-id/luminamoon"
CACHE_ROOT = Path.home() / ".cache" / "luminamoon" / "versions"

VERSION_TAG_MAP: Dict[str, str] = {
    "1.0.3": "main",
    "1.0.2.post2": "v1.0.2.post2",
    "1.0.2": "v1.0.2",
    "1.0.1": "v1.0.1",
    "1.0.0.post1": "v1.0.0.post1",
    "1.0.0": "v1.0.0",
}

_ACTIVE_VERSION: str = DEFAULT_VERSION
_LOADED_MODULES: Dict[str, Any] = {}


def get_version() -> str:
    """Return the currently active LuminaMoon Data version string."""
    return _ACTIVE_VERSION


def list_versions() -> List[str]:
    """Return all supported LuminaMoon Data version strings."""
    return list(SUPPORTED_VERSIONS)


def _ensure_version_cached(version: str) -> Path:
    """Ensure the specified version's Python module files exist in local cache."""
    target_dir = CACHE_ROOT / version / "data"
    init_file = target_dir / "__init__.py"
    if init_file.exists():
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    tag = VERSION_TAG_MAP.get(version, f"v{version}")
    base_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{tag}/python/luminamoon/data"

    files_to_fetch = [
        "__init__.py",
        "_ffi.py",
        "version.py",
        "cleaner.py",
        "quality.py",
        "dedup.py",
        "minhash.py",
        "masker.py",
        "checkpoint.py",
        "string_tensor.py",
        "fused.py",
        "adapters.py",
        "serialization.py",
    ]
    for fname in files_to_fetch:
        url = f"{base_url}/{fname}"
        dest = target_dir / fname
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LuminaMoon-VersionFetcher/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read()
                dest.write_bytes(content)
        except (urllib.error.URLError, OSError) as exc:
            local_fallback = Path(__file__).resolve().parent / fname
            if local_fallback.exists():
                shutil.copyfile(local_fallback, dest)
            else:
                raise RuntimeError(
                    f"Failed to fetch LuminaMoon Data v{version} file '{fname}' from {url}: {exc}"
                ) from exc

    return target_dir


def set_version(version: str) -> str:
    """Set the active LuminaMoon Data version dynamically."""
    global _ACTIVE_VERSION
    ver = version.strip().lstrip("v")
    if ver not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"Unsupported LuminaMoon Data version '{version}'. "
            f"Supported versions: {SUPPORTED_VERSIONS}"
        )

    if ver != DEFAULT_VERSION:
        _ensure_version_cached(ver)

    _ACTIVE_VERSION = ver
    return _ACTIVE_VERSION


def get_active_data_module() -> Any:
    """Retrieve the active data module for the currently selected version."""
    if _ACTIVE_VERSION == DEFAULT_VERSION:
        import luminamoon.data as current_mod
        return current_mod

    if _ACTIVE_VERSION in _LOADED_MODULES:
        return _LOADED_MODULES[_ACTIVE_VERSION]

    cached_dir = _ensure_version_cached(_ACTIVE_VERSION)
    target_file = cached_dir / "__init__.py"
    if not target_file.exists():
        import luminamoon.data as current_mod
        return current_mod

    module_name = f"_luminamoon_data_v{_ACTIVE_VERSION.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(target_file),
        submodule_search_locations=[str(cached_dir)],
    )
    if spec is None or spec.loader is None:
        import luminamoon.data as current_mod
        return current_mod

    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = module_name
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    _LOADED_MODULES[_ACTIVE_VERSION] = mod
    return mod


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
