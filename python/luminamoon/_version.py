from __future__ import annotations

import importlib.metadata
from pathlib import Path
import re


def _detect_version() -> str:
    """Dynamically resolves package version from pyproject.toml, Cargo.toml, or installed wheel metadata."""
    try:
        cur = Path(__file__).resolve().parent
        for parent in [cur] + list(cur.parents):
            # Check pyproject.toml for Python / PyPI distribution version (e.g. 1.0.2.post1)
            pyproj_path = parent / "pyproject.toml"
            if pyproj_path.is_file():
                text = pyproj_path.read_text(encoding="utf-8")
                m = re.search(r'version\s*=\s*["\']([^"\']+)["\']', text)
                if m:
                    return m.group(1).strip()

            # Check Cargo.toml for root workspace package version
            cargo_path = parent / "Cargo.toml"
            if cargo_path.is_file():
                text = cargo_path.read_text(encoding="utf-8")
                m = re.search(r'\[workspace\.package\][\s\S]*?version\s*=\s*["\']([^"\']+)["\']', text)
                if m:
                    return m.group(1).strip()
    except Exception:
        pass

    try:
        return importlib.metadata.version("luminamoon")
    except Exception:
        pass

    return "1.0.6"


__version__ = _detect_version()

__all__ = ["__version__", "_detect_version"]
