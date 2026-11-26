from __future__ import annotations

import re
import sys
from pathlib import Path


def _pep440_to_cargo_semver(ver: str) -> str:
    # PEP 440: 2.0.0rc1, 2.0.0a1, 2.0.0b1, 2.0.0.post1, 2.0.0.dev1 -> Cargo SemVer: 2.0.0-rc.1, etc.
    return re.sub(r"\.?(post|dev|a|b|rc)(\d+)$", r"-\1.\2", ver)


def _cargo_semver_to_pep440(ver: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        tag, num = match.group(1), match.group(2)
        if tag in ("post", "dev"):
            return f".{tag}{num}"
        return f"{tag}{num}"

    return re.sub(r"-(post|dev|a|b|rc)\.(\d+)$", _repl, ver)


def check_version() -> bool:
    root = Path(__file__).resolve().parent.parent
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    cargo = (root / "Cargo.toml").read_text(encoding="utf-8")
    v_py = (root / "python" / "luminamoon" / "_version.py").read_text(encoding="utf-8")

    m_py = re.search(r'version\s*=\s*["\']([^"\']+)["\']', pyproject)
    m_cg = re.search(r'\[workspace\.package\][^\[]*?version\s*=\s*["\']([^"\']+)["\']', cargo, re.DOTALL)
    m_vp = re.search(r'return\s+["\']([^"\']+)["\']', v_py)

    v1 = m_py.group(1) if m_py else None
    v2 = m_cg.group(1) if m_cg else None
    v3 = m_vp.group(1) if m_vp else None

    normalized_v2 = _cargo_semver_to_pep440(v2) if v2 else None
    legacy_dot_v2 = re.sub(r"-(post|dev|a|b|rc)\.(\d+)$", r".\1\2", v2) if v2 else None

    if v1 and (v1 == normalized_v2 or v1 == legacy_dot_v2 or v1 == v2) and v1 == v3:
        print(f"[sync_version] All versions in sync: {v1} (Cargo: {v2})")
        return True
    else:
        print(f"[sync_version] Mismatch detected: pyproject={v1}, Cargo={v2}, _version.py={v3}", file=sys.stderr)
        return False


def sync_version(tag_or_version: str) -> str:
    """Synchronizes version from Git Tag (e.g.

    'v2.0.0rc1', 'v2.0.0', or '2.0.0.post1') into pyproject.toml, Cargo.toml, and internal metadata.
    """
    ver = tag_or_version.strip()
    if ver.startswith("v"):
        ver = ver[1:]

    match = re.match(r"^(\d+\.\d+\.\d+)(?:\.?(a|b|rc)(\d+)|\.(post|dev)(\d+))?$", ver)
    if not match:
        raise ValueError(f"[sync_version] Error: '{ver}' is not a valid PEP 440 version string.")

    # Normalize pre-releases to standard PEP 440 form (e.g., 2.0.0.rc1 -> 2.0.0rc1)
    base, pre_tag, pre_num, post_tag, post_num = match.groups()
    if pre_tag is not None:
        ver = f"{base}{pre_tag}{pre_num}"
    elif post_tag is not None:
        ver = f"{base}.{post_tag}{post_num}"
    else:
        ver = base

    root = Path(__file__).resolve().parent.parent

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

    cargo_path = root / "Cargo.toml"
    if cargo_path.is_file():
        text = cargo_path.read_text(encoding="utf-8")
        cargo_ver = _pep440_to_cargo_semver(ver)
        updated_text = re.sub(
            r'(\[workspace\.package\][^\[]*?version\s*=\s*)["\'][^"\']+["\']',
            rf'\1"{cargo_ver}"',
            text,
            count=1,
            flags=re.DOTALL,
        )
        cargo_path.write_text(updated_text, encoding="utf-8")
        print(f"[sync_version] Updated {cargo_path.name} -> version = \"{cargo_ver}\"")

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
        arg = sys.argv[1].strip()
        if arg in ("--check", "-c"):
            if not check_version():
                sys.exit(1)
        else:
            resolved = sync_version(arg)
            print(f"[sync_version] Successfully synchronized project version to: {resolved}")
    else:
        print("Usage: python tools/sync_version.py [--check | <tag_or_version>]")
        sys.exit(1)
