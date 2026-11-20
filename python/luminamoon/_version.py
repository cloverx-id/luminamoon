from __future__ import annotations

import importlib.metadata
from pathlib import Path
import re


def _detect_version() -> str:
    """Resolve the package version from an editable luminamoon checkout or installed metadata."""
    try:
        # Only read pyproject.toml at the exact repository root (python/luminamoon/_version.py -> parents[2])
        # and verify it belongs to this package so an outer user project cannot shadow our version.
        repo_root = Path(__file__).resolve().parents[2]
        pyproj_path = repo_root / "pyproject.toml"
        if pyproj_path.is_file():
            text = pyproj_path.read_text(encoding="utf-8")
            if re.search(r'(?m)^\s*name\s*=\s*["\']luminamoon["\']', text):
                m = re.search(r'(?m)^\s*version\s*=\s*["\']([^"\']+)["\']', text)
                if m:
                    return m.group(1).strip()
    except Exception:
        pass

    try:
        return importlib.metadata.version("luminamoon")
    except Exception:
        pass

    return "1.1.1.post2"


__version__: str = _detect_version()


def get_version() -> str:
    """Return the active LuminaMoon package version."""
    return __version__


__all__ = ["__version__", "get_version"]
