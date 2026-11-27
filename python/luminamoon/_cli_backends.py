from __future__ import annotations

import importlib.metadata
import platform as _platform
import shlex
import shutil
import subprocess
import sys
import textwrap
from typing import Callable, Optional

from .optimizer.backends.autotune import benchmark_backends
from .optimizer.backends.probe import (
    detect_torch_version,
    is_distributed_worker_env,
    probe_all_backends,
    triton_install_spec,
)

_SEP = "-" * 60


def _quote_cmd(cmd: list[str], *, platform: str) -> str:
    if platform.startswith("win"):
        out: list[str] = []
        for arg in cmd:
            if any(ch in arg for ch in (' ', '\t', '<', '>', '&', '|', '^', '"')):
                out.append('"' + arg.replace('"', '\\"') + '"')
            else:
                out.append(arg)
        return " ".join(out)
    return shlex.join(cmd)


def _format_benchmark_lines(steps: int, measure_last: int) -> list[str]:
    lines = [
        f"Steady-State Backend Benchmark ({steps} steps, timing last {measure_last}):",
    ]
    try:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        sample = [
            torch.nn.Parameter(torch.randn(512, 512, device=dev)),
            torch.nn.Parameter(torch.randn(512, device=dev)),
        ]
    except ImportError:
        try:
            import numpy as np

            sample = [np.random.randn(512, 512).astype(np.float32), np.random.randn(512).astype(np.float32)]
        except ImportError:
            lines.append("  skipped (neither PyTorch nor NumPy installed)")
            return lines

    try:
        res = benchmark_backends(sample, steps=steps, measure_last=measure_last, use_cache=False)
        for name, ms in sorted(res["timings_ms"].items(), key=lambda kv: kv[1]):
            tag = " [fastest]" if name == res["winner"] else ""
            lines.append(f"  {name:<15} {ms:8.3f} ms/step{tag}")
        for name, err in res["errors"].items():
            lines.append(f"  {name:<15}   FAILED ({err})")
    except Exception as exc:
        lines.append(f"  benchmark failed: {exc}")
    return lines


def format_doctor_report(
    *,
    force: bool = True,
    benchmark: bool = False,
    steps: int = 7,
    measure_last: int = 2,
) -> str:
    """Run live micro-probes across all backends and format a concise diagnostic report."""
    report = probe_all_backends(force=force)
    py_ver = ".".join(str(x) for x in sys.version_info[:3])
    torch_ver = detect_torch_version() or "not installed"
    pkg_triton, _ = triton_install_spec()

    lines = [
        "LUMINAMOON BACKEND DOCTOR",
        _SEP,
        f"  Platform        : {sys.platform} ({_platform.machine()})",
        f"  Python          : {py_ver}",
        f"  PyTorch         : {torch_ver}",
        f"  Triton Target   : {pkg_triton}",
        "",
        "  Backend         Status       Details",
    ]
    hints = []
    for name, info in report.items():
        status = str(info.get("status", "unknown")).upper()
        detail = str(info.get("detail", ""))
        lines.append(f"  {name:<15} {status:<12} {detail}")
        hint = info.get("install_hint")
        if hint and status != "OK":
            hints.append((name, str(hint)))

    if benchmark:
        lines.append("")
        lines.extend(_format_benchmark_lines(steps=steps, measure_last=measure_last))

    if hints:
        lines.append("")
        lines.append("Recommended Actions:")
        for name, hint in hints:
            lines.append(f"  - {name}: {hint}")
    lines.append("")
    return "\n".join(lines)


def _has_dist(pkg_name: str) -> bool:
    try:
        importlib.metadata.distribution(pkg_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def run_setup(
    target: str,
    *,
    yes: bool = False,
    dry_run: bool = False,
    input_fn: Callable[[str], str] = input,
    platform_override: Optional[str] = None,
    torch_version_override: Optional[str] = None,
) -> int:
    """Explicit opt-in CLI helper to install Triton (OS-aware) or explain CubeCL build steps."""
    norm = target.strip().lower()
    if norm not in ("triton", "gpu", "cubecl"):
        print(
            f"Unknown setup target: '{target}'. Expected one of: 'triton', 'gpu', 'cubecl'.",
            file=sys.stderr,
        )
        return 2

    if norm == "cubecl":
        print("LUMINAMOON SETUP: cubecl")
        print(_SEP)
        print("`luminav-cubecl` is an opt-in compile-time Rust Cargo feature, not a standalone PyPI package.")
        print("To build LuminaMoon with portable CubeCL GPU support from source, run:")
        print("  maturin develop --release --features cubecl")
        print("  # Or target a specific backend feature: cubecl-wgpu, cubecl-cuda, cubecl-hip, cubecl-cpu")
        return 0

    if is_distributed_worker_env():
        print(
            "Refusing to run package installation inside a multi-rank distributed worker "
            "(WORLD_SIZE > 1). Run `luminamoon setup triton` once on the host environment before launching training.",
            file=sys.stderr,
        )
        return 1

    plat = (platform_override or sys.platform).lower()
    pkg_spec, note = triton_install_spec(platform=plat, torch_version=torch_version_override)
    cmd = [sys.executable, "-m", "pip", "install", pkg_spec]

    print(f"LUMINAMOON SETUP: {norm}")
    print(_SEP)
    for wrapped in textwrap.wrap(note, width=94):
        print(f"  {wrapped}")
    if plat.startswith("win"):
        if _has_dist("triton") and not _has_dist("triton-windows"):
            print("  Warning: upstream 'triton' metadata detected on Windows; consider `pip uninstall triton` first.")
        if shutil.which("cl") is None and shutil.which("gcc") is None:
            print("  Note: No C compiler (`cl.exe` or `gcc`) found on PATH; Triton JIT requires MSVC Build Tools on Windows.")
    print(f"  Command: {_quote_cmd(cmd, platform=plat)}")

    if dry_run:
        print("Dry run enabled; no packages were modified.")
        return 0

    if not yes:
        try:
            ans = input_fn("Proceed with installation? [y/N]: ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("Installation cancelled.")
            return 0

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        print(f"Installation exited with code {proc.returncode}.", file=sys.stderr)
        return int(proc.returncode)

    print("Installation completed. Verifying backends via `luminamoon doctor`:")
    print(format_doctor_report(force=True))
    return 0
