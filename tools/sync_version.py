from __future__ import annotations

import re
import sys
from pathlib import Path


def sync_version(tag_or_version: str) -> str:
    """Synchronizes version from Git Tag (e.g.

    'v1.0.2.post2' or '1.0.2.post2') into pyproject.toml and internal metadata.
    """
    ver = tag_or_version.strip()
    if ver.startswith("v"):
        ver = ver[1:]

    # Match standard semver with optional PEP 440 post/dev suffix
    match = re.match(r"^\d+\.\d+\.\d+(?:\.(?:post|dev|a|b|rc)\d+)?$", ver)
    if not match:
        print(f"[sync_version] Warning: '{ver}' is not a standard PEP 440 version string. Proceeding anyway.")

    root = Path(__file__).resolve().parent.parent

    # 1. Update pyproject.toml
    pyproject_path = root / "pyproject.toml"
    if pyproject_path.is_file():
        text = pyproject_path.read_text(encoding="utf-8")
        updated_text = re.sub(
            r'version\s*=\s*["\'][^"\']+["\']',
            f'version = "{ver}"',
            text,
            count=1,
        )
        pyproject_path.write_text(updated_text, encoding="utf-8")
        print(f"[sync_version] Updated {pyproject_path.name} -> version = \"{ver}\"")

    # 2. Update python/luminamoon/_version.py fallback string
    version_py_path = root / "python" / "luminamoon" / "_version.py"
    if version_py_path.is_file():
        text = version_py_path.read_text(encoding="utf-8")
        updated_text = re.sub(
            r'return\s+["\'][^"\']+["\']',
            f'return "{ver}"',
            text,
            count=1,
        )
        version_py_path.write_text(updated_text, encoding="utf-8")
        print(f"[sync_version] Updated {version_py_path.name} fallback -> \"{ver}\"")

    return ver


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_version = sys.argv[1]
        resolved = sync_version(target_version)
        print(f"[sync_version] Successfully synchronized project version to: {resolved}")
    else:
        print("Usage: python tools/sync_version.py <tag_or_version>")
        sys.exit(1)
