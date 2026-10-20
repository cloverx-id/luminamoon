#!/usr/bin/env python3
"""LuminaMoon Universal Pure-Python Wheel Builder.

Builds a PEP 427 compliant universal wheel (py3-none-any) containing the pure-Python
implementation of LuminaMoon. This wheel acts as Tier-2 zero-failure distribution
fallback for platforms without pre-compiled native Rust binaries or C compilers.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
from pathlib import Path
import re
import sys
import zipfile

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore


def _sha256_b64(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _parse_pyproject(root_dir: Path) -> dict[str, str | list[str] | dict[str, list[str]]]:
    pyproject_path = root_dir / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")

    content = pyproject_path.read_text(encoding="utf-8")
    if tomllib is not None:
        data = tomllib.loads(content)
        proj = data.get("project", {})
        return {
            "name": str(proj.get("name", "luminamoon")),
            "version": str(proj.get("version", "1.0.0")),
            "description": str(proj.get("description", "")),
            "requires_python": str(proj.get("requires-python", ">=3.10")),
            "license": "Apache-2.0",
            "optional_dependencies": proj.get("optional-dependencies", {}),
        }

    # Fallback basic TOML parser if tomllib/tomli is unavailable
    name_m = re.search(r'name\s*=\s*"([^"]+)"', content)
    ver_m = re.search(r'version\s*=\s*"([^"]+)"', content)
    desc_m = re.search(r'description\s*=\s*"([^"]+)"', content)
    req_py_m = re.search(r'requires-python\s*=\s*"([^"]+)"', content)

    # Simple regex extraction for optional dependencies
    opt_deps: dict[str, list[str]] = {}
    current_section: str | None = None
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("[project.optional-dependencies]"):
            current_section = "opt"
            continue
        if line.startswith("[") and current_section:
            current_section = None
            continue
        if current_section == "opt":
            match_key = re.match(r"^([a-zA-Z0-9_-]+)\s*=\s*\[", line)
            if match_key:
                current_group = match_key.group(1)
                opt_deps[current_group] = []
                deps = re.findall(r'"([^"]+)"', line)
                opt_deps[current_group].extend(deps)

    return {
        "name": name_m.group(1) if name_m else "luminamoon",
        "version": ver_m.group(1) if ver_m else "1.0.0",
        "description": desc_m.group(1) if desc_m else "",
        "requires_python": req_py_m.group(1) if req_py_m else ">=3.10",
        "license": "Apache-2.0",
        "optional_dependencies": opt_deps,
    }


def _generate_metadata(
    meta: dict[str, str | list[str] | dict[str, list[str]]],
    readme_text: str,
) -> bytes:
    name = str(meta["name"])
    version = str(meta["version"])
    description = str(meta["description"])
    requires_python = str(meta["requires_python"])
    license_text = str(meta["license"])

    lines: list[str] = [
        "Metadata-Version: 2.4",
        f"Name: {name}",
        f"Version: {version}",
    ]

    opt_deps = meta.get("optional_dependencies", {})
    if isinstance(opt_deps, dict):
        for extra, deps in opt_deps.items():
            for dep in deps:
                if ";" in dep:
                    req_part, env_part = dep.split(";", 1)
                    lines.append(f"Requires-Dist: {req_part.strip()} ; {env_part.strip()} and extra == '{extra}'")
                else:
                    lines.append(f"Requires-Dist: {dep} ; extra == '{extra}'")

        for extra in sorted(opt_deps.keys()):
            lines.append(f"Provides-Extra: {extra}")

    lines.extend([
        "License-File: LICENSE",
        f"Summary: {description}",
        "Author: Lumina Moon (cloverx-id)",
        'Author-email: "Silver Moon (cloverxion)" <cloverxion@luminamoon.org>',
        f"License: {license_text}",
        f"Requires-Python: {requires_python}",
        "Description-Content-Type: text/markdown; charset=UTF-8; variant=GFM",
        "",
        readme_text,
    ])

    return "\n".join(lines).encode("utf-8")


def _generate_wheel_file() -> bytes:
    content = (
        "Wheel-Version: 1.0\n"
        "Generator: luminamoon-wheel-builder (1.0.0)\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
    )
    return content.encode("utf-8")


def build_universal_wheel(root_dir: Path, out_dir: Path) -> Path:
    meta = _parse_pyproject(root_dir)
    name = str(meta["name"])
    version = str(meta["version"])

    readme_path = root_dir / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    license_path = root_dir / "LICENSE"
    license_bytes = license_path.read_bytes() if license_path.exists() else b""

    out_dir.mkdir(parents=True, exist_ok=True)
    wheel_filename = f"{name}-{version}-py3-none-any.whl"
    wheel_path = out_dir / wheel_filename

    python_source_dir = root_dir / "python" / name
    if not python_source_dir.exists():
        raise FileNotFoundError(f"Python source dir not found at {python_source_dir}")

    dist_info = f"{name}-{version}.dist-info"
    record_entries: list[str] = []

    # Temporary file storage in memory / zip writer
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Package Python files (pure-python only, skip binaries and caches)
        for dirpath, _, filenames in os.walk(python_source_dir):
            if "__pycache__" in dirpath:
                continue
            for fname in sorted(filenames):
                # Skip compiled native binaries, debug symbols, and cache files
                if fname.endswith((".pyc", ".pyo", ".pyd", ".so", ".dylib", ".pdb", ".bak")):
                    continue
                file_full = Path(dirpath) / fname
                rel_to_py = file_full.relative_to(python_source_dir.parent)
                arcname = rel_to_py.as_posix()

                data = file_full.read_bytes()
                zf.writestr(arcname, data)
                record_entries.append(f"{arcname},sha256={_sha256_b64(data)},{len(data)}")

        # 2. Dist-info METADATA
        metadata_bytes = _generate_metadata(meta, readme_text)
        meta_arc = f"{dist_info}/METADATA"
        zf.writestr(meta_arc, metadata_bytes)
        record_entries.append(f"{meta_arc},sha256={_sha256_b64(metadata_bytes)},{len(metadata_bytes)}")

        # 3. Dist-info WHEEL
        wheel_bytes = _generate_wheel_file()
        wheel_arc = f"{dist_info}/WHEEL"
        zf.writestr(wheel_arc, wheel_bytes)
        record_entries.append(f"{wheel_arc},sha256={_sha256_b64(wheel_bytes)},{len(wheel_bytes)}")

        # 4. Dist-info LICENSE
        if license_bytes:
            lic_arc = f"{dist_info}/licenses/LICENSE"
            zf.writestr(lic_arc, license_bytes)
            record_entries.append(f"{lic_arc},sha256={_sha256_b64(license_bytes)},{len(license_bytes)}")

        # 5. Dist-info RECORD
        record_arc = f"{dist_info}/RECORD"
        record_entries.append(f"{record_arc},,")
        record_content = "\n".join(record_entries) + "\n"
        zf.writestr(record_arc, record_content.encode("utf-8"))

    return wheel_path


def verify_wheel(wheel_path: Path) -> None:
    if not zipfile.is_zipfile(wheel_path):
        raise ValueError(f"Generated wheel {wheel_path} is not a valid zip file")

    with zipfile.ZipFile(wheel_path, "r") as zf:
        namelist = zf.namelist()
        dist_info_dirs = {name.split("/")[0] for name in namelist if name.split("/")[0].endswith(".dist-info")}
        if not dist_info_dirs:
            raise ValueError("Missing .dist-info directory in wheel")
        dist_info = next(iter(dist_info_dirs))

        required_prefixes = ["luminamoon/", f"{dist_info}/"]
        for pref in required_prefixes:
            if not any(name.startswith(pref) for name in namelist):
                raise ValueError(f"Missing required path prefix '{pref}' in wheel")

        # Ensure no binaries were leaked
        for name in namelist:
            if name.endswith((".pyd", ".so", ".dylib", ".pdb", ".pyc")):
                raise ValueError(f"Leaked compiled binary in universal wheel: {name}")

        record_content = zf.read(f"{dist_info}/RECORD").decode("utf-8")
        if "luminamoon/optimizer/luminav.py" not in record_content:
            raise ValueError("luminav.py missing from wheel RECORD")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LuminaMoon Universal Pure-Python Wheel")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent, help="Repository root path")
    parser.add_argument("--out", "-o", type=Path, default=Path("dist"), help="Output directory for wheel")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify wheel structure after build")

    args = parser.parse_args()
    wheel_path = build_universal_wheel(args.root, args.out)

    if args.verify:
        verify_wheel(wheel_path)

    size_kb = wheel_path.stat().st_size / 1024
    print(f"Universal wheel built successfully: {wheel_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
