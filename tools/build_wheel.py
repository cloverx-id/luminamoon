#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


def _b64_sha256(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _sanitize_wheel_privacy(wheel_path: Path, root_dir: Path) -> None:
    """Scrubs local workstation user paths from wheel metadata and SBOMs dynamically."""
    import getpass
    import urllib.parse

    current_user = getpass.getuser()
    user_home = Path.home()
    replacement_bytes = b"runner"

    patterns_to_scrub: list[bytes] = []
    if current_user and len(current_user) > 2:
        patterns_to_scrub.append(current_user.encode("utf-8"))

    patterns_to_scrub.extend([
        root_dir.as_posix().encode("utf-8"),
        str(root_dir).encode("utf-8"),
        user_home.as_posix().encode("utf-8"),
        str(user_home).encode("utf-8"),
    ])

    escaped_root = str(root_dir).replace("\\", "\\\\").encode("utf-8")
    escaped_home = str(user_home).replace("\\", "\\\\").encode("utf-8")
    patterns_to_scrub.extend([escaped_root, escaped_home])

    patterns_to_scrub.append(urllib.parse.quote(root_dir.as_posix()).encode("utf-8"))
    patterns_to_scrub.append(urllib.parse.quote(user_home.as_posix()).encode("utf-8"))

    modified_any = False
    temp_fd, temp_zip = tempfile.mkstemp(suffix=".whl")
    os.close(temp_fd)
    temp_zip_path = Path(temp_zip)

    try:
        record_filename: str | None = None
        entries_data: dict[str, bytes] = {}

        with zipfile.ZipFile(wheel_path, "r") as src_zf:
            for item in src_zf.infolist():
                data = src_zf.read(item.filename)
                if item.filename.endswith("/RECORD") or item.filename == "RECORD":
                    record_filename = item.filename
                    continue

                is_text_target = (
                    "dist-info" in item.filename
                    or item.filename.endswith((".json", ".txt", ".md"))
                )
                if is_text_target:
                    original_data = data
                    for pat in patterns_to_scrub:
                        if pat and pat in data:
                            data = data.replace(pat, replacement_bytes)
                            modified_any = True

                entries_data[item.filename] = data

        if modified_any and record_filename:
            record_lines: list[str] = []
            for fname in sorted(entries_data.keys()):
                fdata = entries_data[fname]
                hash_str = f"sha256={_b64_sha256(fdata)}"
                record_lines.append(f"{fname},{hash_str},{len(fdata)}")
            record_lines.append(f"{record_filename},,")
            entries_data[record_filename] = "\n".join(record_lines).encode("utf-8")

            with zipfile.ZipFile(temp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as dst_zf:
                for fname, fdata in entries_data.items():
                    dst_zf.writestr(fname, fdata)

            shutil.move(str(temp_zip_path), str(wheel_path))
            print(f"[PRIVACY] Sanitized local user paths from {wheel_path.name}")
        else:
            if temp_zip_path.exists():
                temp_zip_path.unlink()
    except Exception as exc:
        if temp_zip_path.exists():
            temp_zip_path.unlink()
        print(f"[PRIVACY WARNING] Could not sanitize wheel metadata: {exc}")


def _compute_source_fingerprint(root_dir: Path) -> str:
    """Computes a deterministic SHA-256 hash across all workspace source files."""
    hasher = hashlib.sha256()

    critical_root_files = [
        root_dir / "Cargo.toml",
        root_dir / "Cargo.lock",
        root_dir / "pyproject.toml",
    ]
    for rf in critical_root_files:
        if rf.is_file():
            hasher.update(rf.name.encode("utf-8"))
            hasher.update(rf.read_bytes())

    scan_dirs = [
        root_dir / "optimizer",
        root_dir / "dataloader",
        root_dir / "python",
    ]

    valid_extensions = (
        ".rs",
        ".toml",
        ".py",
        ".typed",
        ".c",
        ".h",
        ".cu",
        ".comp",
        ".spv",
    )
    ignored_dir_names = {
        "target",
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        "venv",
        ".venv",
    }

    tracked_files: list[Path] = []
    for s_dir in scan_dirs:
        if not s_dir.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(s_dir):
            dirnames[:] = [d for d in dirnames if d not in ignored_dir_names]
            for fname in filenames:
                if fname.endswith(valid_extensions):
                    if fname.endswith((".pyd", ".so", ".dylib", ".pyc")):
                        continue
                    full_p = Path(dirpath) / fname
                    tracked_files.append(full_p)

    tracked_files.sort(key=lambda p: str(p.relative_to(root_dir)).replace("\\", "/"))

    for fpath in tracked_files:
        rel_path_bytes = str(fpath.relative_to(root_dir)).replace("\\", "/").encode("utf-8")
        hasher.update(rel_path_bytes)
        try:
            hasher.update(fpath.read_bytes())
        except OSError:
            pass

    return hasher.hexdigest()


def _find_matching_wheel(out_dir: Path) -> Path | None:
    """Finds the most recently modified native wheel in the output directory."""
    if not out_dir.is_dir():
        return None
    wheels = [
        p
        for p in out_dir.glob("luminamoon-*.whl")
        if "py3-none-any" not in p.name  
    ]
    if not wheels:
        return None
    wheels.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return wheels[0]


def build_wheel(
    root_dir: Path,
    out_dir: Path,
    force: bool = False,
    install: bool = False,
    extra_maturin_args: list[str] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / ".native_build_cache.json"

    fingerprint = _compute_source_fingerprint(root_dir)
    fp_short = fingerprint[:12]

    target_wheel: Path | None = None

    if not force and cache_file.is_file():
        try:
            cache_data = json.loads(cache_file.read_text(encoding="utf-8"))
            cached_fp = cache_data.get("fingerprint")
            cached_wheel_name = cache_data.get("wheel_file")
            if cached_fp == fingerprint and cached_wheel_name:
                cached_wheel_path = out_dir / cached_wheel_name
                if cached_wheel_path.is_file():
                    size_mb = cached_wheel_path.stat().st_size / (1024 * 1024)
                    print(
                        f"[CACHE HIT] Source files unchanged (fingerprint: {fp_short}). "
                        f"Reusing existing wheel: {cached_wheel_path.name} ({size_mb:.2f} MB)"
                    )
                    target_wheel = cached_wheel_path
        except Exception:
            pass

    if target_wheel is None:
        print(
            f"[BUILD] Building LuminaMoon native wheel via Maturin (fingerprint: {fp_short}, codegen-units: all)..."
        )

        cmd = [
            sys.executable,
            "-m",
            "maturin",
            "build",
            "--release",
            "--out",
            str(out_dir),
        ]
        if extra_maturin_args:
            cmd.extend(extra_maturin_args)

        try:
            subprocess.run(cmd, cwd=root_dir, check=True)
        except FileNotFoundError:
            cmd[0:3] = ["maturin"]
            subprocess.run(cmd, cwd=root_dir, check=True)

        built_wheel = _find_matching_wheel(out_dir)
        if built_wheel is None or not built_wheel.is_file():
            raise RuntimeError(f"Build succeeded but no native wheel found in {out_dir}")

        _sanitize_wheel_privacy(built_wheel, root_dir)

        cache_payload = {
            "fingerprint": fingerprint,
            "wheel_file": built_wheel.name,
            "wheel_size_bytes": built_wheel.stat().st_size,
        }
        cache_file.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")

        size_mb = built_wheel.stat().st_size / (1024 * 1024)
        print(f"[SUCCESS] Native wheel built successfully: {built_wheel.name} ({size_mb:.2f} MB)")
        target_wheel = built_wheel

    if install:
        print(f"[INSTALL] Installing {target_wheel.name} into current Python environment...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(target_wheel)],
            check=True,
        )
        print("[SUCCESS] Package installed successfully.")

    return target_wheel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build LuminaMoon Native Wheel with Source Fingerprint Caching"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root directory",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("dist"),
        help="Output directory for generated wheels (default: dist)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force rebuild even if source fingerprint matches cache",
    )
    parser.add_argument(
        "--install",
        "-i",
        action="store_true",
        help="Install the built or cached wheel into the current Python environment",
    )

    args, unknown = parser.parse_known_args()
    build_wheel(
        root_dir=args.root,
        out_dir=args.out,
        force=args.force,
        install=args.install,
        extra_maturin_args=unknown,
    )


if __name__ == "__main__":
    main()
