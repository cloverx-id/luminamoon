from __future__ import annotations

import os
import re
import shutil
import sys
import threading
from typing import Any, Dict, Optional, Tuple

_PROBE_LOCK = threading.RLock()
_TRITON_PROBE_CACHE: Dict[Tuple[str, int], Tuple[bool, str]] = {}
_PROBE_KERNEL: Any = None


def reset_probe_caches() -> None:
    """Clear cached runtime probe results (primarily for testing)."""
    with _PROBE_LOCK:
        _TRITON_PROBE_CACHE.clear()


def _parse_torch_minor(torch_version: Optional[str]) -> Optional[Tuple[int, int]]:
    if not torch_version:
        return None
    m = re.match(r"^\s*(\d+)\.(\d+)", str(torch_version))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def detect_torch_version() -> Optional[str]:
    try:
        import torch

        return str(getattr(torch, "__version__", "")) or None
    except Exception:
        return None


def triton_install_spec(
    platform: Optional[str] = None,
    torch_version: Optional[str] = None,
) -> Tuple[str, str]:
    """Return `(pip_spec, guidance_note)` matched to host OS and active PyTorch version."""
    plat = (platform or sys.platform).lower()
    tv = torch_version if torch_version is not None else detect_torch_version()
    minor = _parse_torch_minor(tv)

    if plat.startswith("win"):
        if minor == (2, 14):
            pkg = "triton-windows<3.9"
        elif minor == (2, 15):
            pkg = "triton-windows<3.10"
        else:
            pkg = "triton-windows>=3.0.0"
        note = (
            f"Windows detected (PyTorch {tv or 'not installed'}): upstream 'triton' does not ship "
            f"official Windows wheels; use the community 'triton-windows' package ({pkg}) or "
            'pip install --pre "luminamoon[gpu]". Ensure a C compiler (MSVC Build Tools) is '
            "available for JIT compilation."
        )
        return pkg, note

    pkg = "triton>=3.0.0"
    note = (
        f"Platform '{plat}' (PyTorch {tv or 'not installed'}): install Triton via "
        f'`pip install "{pkg}"` or `pip install --pre "luminamoon[gpu]"`.'
    )
    return pkg, note


def backend_unavailable_message(
    backend: str,
    *,
    reason: Optional[str] = None,
    platform: Optional[str] = None,
    torch_version: Optional[str] = None,
) -> str:
    """Build an actionable fail-fast RuntimeError message for explicit backend requests."""
    b = backend.strip().lower()
    why = f" ({reason})" if reason else ""
    if b == "triton":
        pkg, note = triton_install_spec(platform=platform, torch_version=torch_version)
        return (
            f"LuminaV: explicit execution='triton' is unavailable{why}. {note} "
            f'Run `pip install "{pkg}"` or `luminamoon setup triton`, inspect active backends with '
            "`luminamoon doctor`, or use `execution='auto'` for automatic fallback."
        )
    if b == "cubecl":
        return (
            f"LuminaV: explicit execution='cubecl' is unavailable{why}. "
            "`luminav-cubecl` is an opt-in compile-time Rust Cargo feature (not a separate PyPI package). "
            "Build from source with `maturin develop --release --features cubecl` (or `cubecl-wgpu` / "
            "`cubecl-cuda` / `cubecl-hip`), inspect active backends with `luminamoon doctor`, or use "
            "`execution='auto'` for automatic fallback."
        )
    return (
        f"LuminaV: explicit execution='{b}' is unavailable{why}. "
        "Run `luminamoon doctor` to inspect available backends or use `execution='auto'`."
    )


def _get_probe_kernel() -> Any:
    global _PROBE_KERNEL
    if _PROBE_KERNEL is not None:
        return _PROBE_KERNEL
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]

    @triton.jit
    def _luminamoon_triton_probe_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
        tl.store(y_ptr + offs, vals + 1.0, mask=mask)

    _PROBE_KERNEL = _luminamoon_triton_probe_kernel
    return _PROBE_KERNEL


def probe_triton_runtime(
    device: Any = None,
    *,
    force: bool = False,
) -> Tuple[bool, str]:
    """Execute a cached 16-element Triton JIT micro-kernel to verify end-to-end usability."""
    try:
        import torch
    except ImportError:
        return False, "PyTorch is not installed"

    try:
        import triton  # type: ignore[import-untyped]
    except ImportError:
        return False, "Triton package is not installed"

    target_dev: Any = device
    if target_dev is None:
        if torch.cuda.is_available():
            target_dev = torch.device("cuda", torch.cuda.current_device())
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            target_dev = torch.device("xpu", torch.xpu.current_device())
        else:
            return False, "No CUDA or XPU accelerator device detected"
    elif isinstance(target_dev, str):
        target_dev = torch.device(target_dev)

    dev_type = getattr(target_dev, "type", "")
    if dev_type not in ("cuda", "xpu"):
        return False, f"Device '{target_dev}' is not a CUDA/XPU accelerator"

    dev_idx = int(getattr(target_dev, "index", 0) or 0)
    cache_key = (dev_type, dev_idx)
    with _PROBE_LOCK:
        if not force and cache_key in _TRITON_PROBE_CACHE:
            return _TRITON_PROBE_CACHE[cache_key]

        try:
            kernel = _get_probe_kernel()
            x = torch.ones(16, device=target_dev, dtype=torch.float32)
            y = torch.empty_like(x)
            kernel[(1,)](x, y, 16, BLOCK=16, enable_fp_fusion=True)
            if not bool(torch.allclose(y, torch.full_like(y, 2.0))):
                res = (False, "Triton probe kernel returned unexpected numerical values")
            else:
                triton_ver = str(getattr(triton, "__version__", "unknown"))
                res = (True, f"triton {triton_ver} verified on {target_dev}")
        except Exception as exc:
            first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            res = (False, f"Triton JIT probe failed: {first_line}")

        _TRITON_PROBE_CACHE[cache_key] = res
        return res


def resolve_cpu_backend_label(
    exec_mode: str,
    max_numel: int,
    hw: Dict[str, Any],
    *,
    has_rust: bool,
    is_torch: bool,
) -> str:
    """Resolve the active CPU/host backend label executed for telemetry (`opt.last_backend`)."""
    if exec_mode == "eager":
        return "eager" if is_torch else "numpy"
    if not has_rust:
        return "eager" if is_torch else "numpy"
    if exec_mode == "simd":
        return "simd"
    if exec_mode == "scalar":
        return "scalar"
    if exec_mode == "cubecl":
        return "cubecl"
    if exec_mode == "triton":
        return "triton_aot"
    threshold = int(hw.get("gpu_dispatch_threshold", 16_384))
    if max_numel >= threshold:
        if hw.get("triton_aot_available", False):
            return "triton_aot"
        if hw.get("cubecl_available", False):
            return "cubecl"
    return "simd"


def probe_all_backends(*, force: bool = False) -> Dict[str, Dict[str, Any]]:
    """Run live micro-probes across all Optimizer and Data Engine backends."""
    report: Dict[str, Dict[str, Any]] = {}
    pkg_triton, _ = triton_install_spec()

    # 1. PyTorch & Eager
    torch_ver = detect_torch_version()
    if torch_ver is not None:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
        dev_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU only"
        report["torch_eager"] = {
            "status": "ok",
            "detail": f"torch {torch_ver} ({dev_name})",
            "install_hint": None,
        }
    else:
        report["torch_eager"] = {
            "status": "unavailable",
            "detail": "PyTorch not installed",
            "install_hint": 'pip install "luminamoon[torch]"',
        }

    # 2. Triton JIT (live micro-kernel probe)
    triton_ok, triton_detail = probe_triton_runtime(force=force)
    if triton_ok:
        report["triton_jit"] = {"status": "ok", "detail": triton_detail, "install_hint": None}
    else:
        is_failed = "probe failed" in triton_detail.lower()
        no_gpu = "no cuda or xpu" in triton_detail.lower()
        hint: Optional[str] = None if no_gpu else f'pip install "{pkg_triton}" (or run `luminamoon setup triton`)'
        if hint is not None and sys.platform.startswith("win") and is_failed and shutil.which("cl") is None:
            hint += "; install Visual Studio C++ Build Tools (cl.exe) for JIT compilation"
        report["triton_jit"] = {
            "status": "failed" if is_failed else "unavailable",
            "detail": triton_detail,
            "install_hint": hint,
        }

    # 3. Native Rust extension (Triton AOT, CubeCL, SIMD, Scalar)
    try:
        from ..luminav import LuminaV, _luminamoon_rs

        hw = LuminaV.detect_hardware()
        has_rs = _luminamoon_rs is not None
    except Exception as exc:
        hw = {}
        has_rs = False
        report["rust_simd"] = {"status": "failed", "detail": str(exc), "install_hint": None}

    if "rust_simd" not in report:
        if has_rs:
            report["rust_simd"] = {
                "status": "ok",
                "detail": str(hw.get("simd_backend", "native")),
                "install_hint": None,
            }
            report["rust_scalar"] = {"status": "ok", "detail": "luminav-core scalar", "install_hint": None}
            aot_ok = bool(hw.get("triton_aot_available", False))
            report["triton_aot"] = {
                "status": "ok" if aot_ok else "unavailable",
                "detail": f"arch={hw.get('triton_arch')} ({hw.get('gpu_device_name')})" if aot_ok else "No supported CUDA/HIP driver or AOT target",
                "install_hint": None if aot_ok else "Requires NVIDIA CUDA (sm_75..sm_120) or AMD ROCm HIP (gfx908..gfx1103) driver",
            }
            ccl_ok = bool(hw.get("cubecl_available", False))
            report["cubecl"] = {
                "status": "ok" if ccl_ok else "unavailable",
                "detail": f"backend={hw.get('cubecl_backend')}" if ccl_ok else "Not compiled into native extension (or no GPU adapter)",
                "install_hint": None if ccl_ok else "maturin develop --release --features cubecl",
            }
        else:
            report["rust_simd"] = {
                "status": "unavailable",
                "detail": "Native extension _luminamoon_optimizer_rs not loaded",
                "install_hint": "maturin develop --release",
            }
            report["rust_scalar"] = {"status": "unavailable", "detail": "Native extension not loaded", "install_hint": None}
            report["triton_aot"] = {"status": "unavailable", "detail": "Native extension not loaded", "install_hint": None}
            report["cubecl"] = {
                "status": "unavailable",
                "detail": "Native extension not loaded",
                "install_hint": "maturin develop --release --features cubecl",
            }

    # 4. NumPy & Data Engine SIMD
    try:
        import numpy as np

        report["numpy"] = {"status": "ok", "detail": f"numpy {np.__version__}", "install_hint": None}
    except ImportError:
        report["numpy"] = {"status": "unavailable", "detail": "NumPy not installed", "install_hint": 'pip install "luminamoon[numpy]"'}

    try:
        from ...data import data_cpu_isa

        isa = data_cpu_isa()
        report["data_simd"] = {"status": "ok", "detail": f"ISA={isa}", "install_hint": None}
    except Exception as exc:
        report["data_simd"] = {"status": "failed", "detail": str(exc), "install_hint": None}

    return report


def is_distributed_worker_env() -> bool:
    """Return True when running inside a multi-rank distributed launcher (e.g. torchrun)."""
    try:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            return True
    except ValueError:
        pass
    return "LOCAL_RANK" in os.environ and os.environ.get("WORLD_SIZE", "1") != "1"


def dispatch_cpu_batch(
    opt: Any,
    group: Dict[str, Any],
    params: list[Any],
    grads: list[Any],
    exp_avgs: list[Any],
    exp_avg_sqs: list[Any],
    master_params: list[Any],
    scratches: list[Any],
    steps: list[int],
    param_indices: list[int],
    *,
    rs_ext: Any,
    build_rust_config_fn: Any,
    step_gpu_group_fn: Any,
    step_numpy_group_fn: Any,
    logger: Any,
) -> None:
    """Dispatch a CPU/host parameter batch with fail-fast explicit backend checks and telemetry."""
    try:
        import torch as _torch

        has_torch = True
        is_cpu_torch = isinstance(params[0], _torch.Tensor)
        deterministic = bool(
            is_cpu_torch
            and hasattr(_torch, "are_deterministic_algorithms_enabled")
            and _torch.are_deterministic_algorithms_enabled()
        )
    except ImportError:
        has_torch = False
        is_cpu_torch = False
        deterministic = False

    req_mode = str(group["execution"])
    active_auto = getattr(opt, "_active_auto_exec", None) if req_mode == "auto" else None
    exec_mode = active_auto if active_auto in ("simd", "scalar", "eager", "triton", "cubecl") else req_mode
    hw = opt.detect_hardware() if hasattr(opt, "detect_hardware") else {}

    if req_mode == "cubecl":
        if rs_ext is None:
            raise RuntimeError(backend_unavailable_message("cubecl", reason="Native Rust extension is not loaded"))
        if not hw.get("cubecl_available", False):
            raise RuntimeError(
                backend_unavailable_message(
                    "cubecl",
                    reason="CubeCL backend is not compiled in or no GPU adapter was detected",
                )
            )
    elif req_mode == "triton":
        if deterministic and is_cpu_torch and step_gpu_group_fn is not None:
            if not getattr(opt, "_warned_triton_deterministic", False):
                logger.warning(
                    "LuminaV: torch.are_deterministic_algorithms_enabled() is True; "
                    "routing execution='triton' to deterministic PyTorch 'eager' backend."
                )
                opt._warned_triton_deterministic = True
            step_gpu_group_fn(
                opt, group, params, grads, exp_avgs, exp_avg_sqs, master_params,
                scratches, steps, orig_indices=param_indices,
            )
            return
        if rs_ext is None or not hw.get("triton_aot_available", False):
            reason = (
                "Triton AOT GPU driver not detected on CPU/host tensors; move tensors to CUDA/XPU "
                "for Triton JIT or ensure a supported CUDA/ROCm driver is available"
            )
            raise RuntimeError(backend_unavailable_message("triton", reason=reason))

    if rs_ext is not None and exec_mode != "eager":
        max_numel = max(int(p.numel() if hasattr(p, "numel") else p.size) for p in params)
        cfg_group = group if exec_mode == req_mode else {**group, "execution": exec_mode}
        rs_ext.step_pyo3(
            params, grads, exp_avgs, exp_avg_sqs, master_params, scratches,
            steps, param_indices, build_rust_config_fn(cfg_group),
        )
        opt._last_backend = resolve_cpu_backend_label(
            exec_mode, max_numel, hw, has_rust=True, is_torch=is_cpu_torch
        )
        return

    if has_torch and is_cpu_torch and step_gpu_group_fn is not None:
        if rs_ext is None and not getattr(opt, "_warned_rs_missing", False):
            logger.info("LuminaV: Native Rust extension not loaded; executing via PyTorch eager backend.")
            opt._warned_rs_missing = True
        step_gpu_group_fn(
            opt, group, params, grads, exp_avgs, exp_avg_sqs, master_params,
            scratches, steps, orig_indices=param_indices,
        )
        return

    try:
        import numpy as _np  # noqa: F401

        has_numpy = True
    except ImportError:
        has_numpy = False

    if has_numpy:
        if rs_ext is None and not getattr(opt, "_warned_rs_missing", False):
            logger.info("LuminaV: Native Rust extension not loaded; executing via NumPy fallback backend.")
            opt._warned_rs_missing = True
        step_numpy_group_fn(
            opt, group, params, grads, exp_avgs, exp_avg_sqs, master_params,
            scratches, steps, param_indices=param_indices,
        )
        opt._last_backend = "numpy"
        return

    raise RuntimeError(
        "LuminaV: Neither native Rust extension (_luminamoon_optimizer_rs), PyTorch, nor NumPy was found."
    )
