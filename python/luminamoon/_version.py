from __future__ import annotations

from pathlib import Path
import re


def _detect_cargo_version() -> str:
    """Dynamically resolves the version from Cargo.toml or installed wheel metadata (Single Source of Truth)."""
    # 1. Search Cargo.toml in parent directories (source / development mode)
    try:
        cur = Path(__file__).resolve().parent
        for parent in [cur] + list(cur.parents):
            cargo_path = parent / "Cargo.toml"
            if cargo_path.is_file():
                text = cargo_path.read_text(encoding="utf-8")
                # Look for [workspace.package] version first
                m = re.search(r'\[workspace\.package\][\s\S]*?version\s*=\s*["\']([^"\']+)["\']', text)
                if m:
                    return m.group(1).strip()
                # General package version fallback
                m = re.search(r'version\s*=\s*["\']([^"\']+)["\']', text)
                if m:
                    return m.group(1).strip()
    except Exception:
        pass

    # 2. Inspect installed distribution package metadata (wheel / site-packages mode)
    try:
        import importlib.metadata

        return importlib.metadata.version("luminamoon")
    except Exception:
        pass

    # 3. Reference fallback
    return "1.0.2"


__version__ = _detect_cargo_version()

__all__ = ["__version__", "_detect_cargo_version"]
