from __future__ import annotations

import inspect
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / "optimizer" / "crates" / "luminav-triton-aot" / "artifacts"
PYTHON_SRC = REPO_ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

CUDA_ARCHS: List[Dict[str, Any]] = [
    {"arch": "sm_75", "cc": 75, "warp_size": 32, "family": "Turing", "bf16_mode": "fp32_upcast_emulation"},
    {"arch": "sm_80", "cc": 80, "warp_size": 32, "family": "Ampere Data Center", "bf16_mode": "native_hardware"},
    {"arch": "sm_86", "cc": 86, "warp_size": 32, "family": "Ampere Workstation/Client", "bf16_mode": "native_hardware"},
    {"arch": "sm_89", "cc": 89, "warp_size": 32, "family": "Ada Lovelace", "bf16_mode": "native_hardware_fp8"},
    {"arch": "sm_90", "cc": 90, "warp_size": 32, "family": "Hopper (sm_90a)", "bf16_mode": "native_tma_wgmma"},
    {"arch": "sm_100", "cc": 100, "warp_size": 32, "family": "Blackwell Data Center", "bf16_mode": "native_tmem_clc_tma2"},
    {"arch": "sm_103", "cc": 103, "warp_size": 32, "family": "Blackwell Ultra Data Center", "bf16_mode": "native_tmem_ultra_nvlink5"},
    {"arch": "sm_120", "cc": 120, "warp_size": 32, "family": "Blackwell Client/Workstation", "bf16_mode": "native_blackwell_client_no_tmem"},
]

ROCM_ARCHS: List[Dict[str, Any]] = [
    {"arch": "gfx908", "warp_size": 64, "family": "CDNA 1", "bf16_mode": "native_cdna1_wave64"},
    {"arch": "gfx90a", "warp_size": 64, "family": "CDNA 2", "bf16_mode": "native_cdna2_packed_fp32_wave64"},
    {"arch": "gfx942", "warp_size": 64, "family": "CDNA 3", "bf16_mode": "native_cdna3_xgmi_uma_wave64"},
    {"arch": "gfx1100", "warp_size": 32, "family": "RDNA 3 Workstation/Client", "bf16_mode": "native_rdna3_wave32"},
    {"arch": "gfx1103", "warp_size": 32, "family": "RDNA 3 Integrated APU", "bf16_mode": "native_rdna3_apu_wave32"},
]

STUB_MAGICS = (b"// LUMINAV_TRITON_AOT_CUBIN", b"; LUMINAV_TRITON_AOT_HSACO")
DRIVE_PATH_RE = re.compile(rb"[A-Za-z]:[\\/][^\x00\n\x22]{2,80}")


def _dual_ptr_args() -> List[Tuple[str, str]]:
    return [
        ("p_ptr", "*fp32"), ("master_p_ptr", "*fp32"), ("grad_ptr", "*fp32"),
        ("exp_avg_ptr", "*fp32"), ("exp_avg_sq_ptr", "*fp32"), ("scratch_ptr", "*fp32"),
        ("n_elements", "i32"),
    ]


def _tdclm_ptr_args() -> List[Tuple[str, str]]:
    return [
        ("p_ptr", "*fp32"), ("master_p_ptr", "*fp32"), ("grad_ptr", "*fp32"),
        ("exp_avg_ptr", "*fp32"), ("scratch_ptr", "*fp32"), ("n_elements", "i32"),
    ]


def _kernel_specs(modules: Dict[str, Any]) -> List[Dict[str, Any]]:
    common, dual, tdclm, triangle = modules["common"], modules["dual"], modules["tdclm"], modules["triangle"]
    c_std = {"bound_mode": 2, "dtype_mode": 0, "state_dtype_mode": 0, "use_sr": False, "cautious": True, "has_master": False, "BLOCK_SIZE": 1024}
    c_fast = {"bound_mode": 2, "dtype_mode": 0, "state_dtype_mode": 0, "use_sr": False, "has_master": False, "BLOCK_SIZE": 1024}
    c_p1 = {"bound_mode": 2, "dtype_mode": 0, "state_dtype_mode": 0, "cautious": True, "has_master": False, "BLOCK_SIZE": 1024}
    dual_tail = [("beta1", "fp32"), ("beta2", "fp32"), ("c1", "fp32"), ("c2", "fp32"), ("lr", "fp32"), ("weight_decay", "fp32"), ("bound_ratio", "fp32"), ("seed", "i32"), ("step_i32", "i32")]
    t2_tail = [("beta1", "fp32"), ("beta2", "fp32"), ("c1", "fp32"), ("c2", "fp32"), ("lr", "fp32"), ("weight_decay", "fp32"), ("eps", "fp32"), ("bound_ratio", "fp32"), ("seed", "i32"), ("step_i32", "i32")]
    tdclm_tail = [("beta1", "fp32"), ("lr", "fp32"), ("weight_decay", "fp32"), ("eps", "fp32"), ("lam", "fp32"), ("bound_ratio", "fp32"), ("seed", "i32"), ("step_i32", "i32")]
    return [
        {
            "name": "_lumina_dual_fused_single_pass_kernel",
            "fn": dual._lumina_dual_fused_single_pass_kernel,
            "block_size": 1024, "num_warps": 4, "num_stages": 2,
            "runtime_args": _dual_ptr_args() + dual_tail,
            "constexprs": dict(c_std),
        },
        {
            "name": "_lumina_dual_single_pass_kernel",
            "fn": dual._lumina_dual_single_pass_kernel,
            "block_size": 1024, "num_warps": 4, "num_stages": 2,
            "runtime_args": _dual_ptr_args() + dual_tail,
            "constexprs": dict(c_fast),
        },
        {
            "name": "_lumina_dual_pass1_kernel",
            "fn": dual._lumina_dual_pass1_kernel,
            "block_size": 1024, "num_warps": 4, "num_stages": 2,
            "runtime_args": _dual_ptr_args() + [("beta1", "fp32"), ("beta2", "fp32"), ("c1", "fp32"), ("c2", "fp32")],
            "constexprs": dict(c_p1),
        },
        {
            "name": "_lumina_dual_pass2_kernel",
            "fn": dual._lumina_dual_pass2_kernel,
            "block_size": 1024, "num_warps": 4, "num_stages": 2,
            "runtime_args": _dual_ptr_args() + [
                ("total_elements", "fp32"), ("beta1", "fp32"), ("beta2", "fp32"), ("c1", "fp32"),
                ("c2", "fp32"), ("lr", "fp32"), ("weight_decay", "fp32"), ("clamp_min", "fp32"),
                ("bound_ratio", "fp32"), ("seed", "i32"), ("step_i32", "i32"),
            ],
            "constexprs": dict(c_std),
        },
        {
            "name": "_lumina_tdclm_single_pass_kernel",
            "fn": tdclm._lumina_tdclm_single_pass_kernel,
            "block_size": 1024, "num_warps": 4, "num_stages": 2,
            "runtime_args": _tdclm_ptr_args() + tdclm_tail,
            "constexprs": dict(c_std),
        },
        {
            "name": "_lumina_tdclm_pass1_kernel",
            "fn": tdclm._lumina_tdclm_pass1_kernel,
            "block_size": 1024, "num_warps": 4, "num_stages": 2,
            "runtime_args": _tdclm_ptr_args() + [("beta1", "fp32"), ("eps", "fp32"), ("lam", "fp32")],
            "constexprs": dict(c_p1),
        },
        {
            "name": "_lumina_tdclm_pass2_kernel",
            "fn": tdclm._lumina_tdclm_pass2_kernel,
            "block_size": 1024, "num_warps": 4, "num_stages": 2,
            "runtime_args": _tdclm_ptr_args() + [
                ("total_elements", "fp32"), ("beta1", "fp32"), ("lr", "fp32"), ("weight_decay", "fp32"),
                ("eps", "fp32"), ("lam", "fp32"), ("clamp_min", "fp32"), ("bound_ratio", "fp32"),
                ("seed", "i32"), ("step_i32", "i32"),
            ],
            "constexprs": {**c_std, "is_triangle": False},
        },
        {
            "name": "_lumina_t1_pass2_kernel",
            "fn": tdclm._lumina_tdclm_pass2_kernel,
            "block_size": 1024, "num_warps": 4, "num_stages": 2,
            "runtime_args": _tdclm_ptr_args() + [
                ("total_elements", "fp32"), ("beta1", "fp32"), ("lr", "fp32"), ("weight_decay", "fp32"),
                ("eps", "fp32"), ("lam", "fp32"), ("clamp_min", "fp32"), ("bound_ratio", "fp32"),
                ("seed", "i32"), ("step_i32", "i32"),
            ],
            "constexprs": {**c_std, "is_triangle": True},
        },
        {
            "name": "_lumina_t1_single_pass_kernel",
            "fn": triangle._lumina_t1_single_pass_kernel,
            "block_size": 1024, "num_warps": 4, "num_stages": 2,
            "runtime_args": _tdclm_ptr_args() + tdclm_tail,
            "constexprs": dict(c_std),
        },
        {
            "name": "_lumina_t2_fused_single_pass_kernel",
            "fn": triangle._lumina_t2_fused_single_pass_kernel,
            "block_size": 1024, "num_warps": 4, "num_stages": 2,
            "runtime_args": _dual_ptr_args() + t2_tail,
            "constexprs": dict(c_std),
        },
        {
            "name": "_lumina_t2_single_pass_kernel",
            "fn": triangle._lumina_t2_single_pass_kernel,
            "block_size": 1024, "num_warps": 4, "num_stages": 2,
            "runtime_args": _dual_ptr_args() + t2_tail,
            "constexprs": dict(c_fast),
        },
        {
            "name": "_lumina_t2_pass1_kernel",
            "fn": triangle._lumina_t2_pass1_kernel,
            "block_size": 1024, "num_warps": 4, "num_stages": 2,
            "runtime_args": _dual_ptr_args() + [("beta1", "fp32"), ("beta2", "fp32"), ("c1", "fp32"), ("c2", "fp32"), ("eps", "fp32")],
            "constexprs": dict(c_p1),
        },
        {
            "name": "_lumina_t2_pass2_kernel",
            "fn": triangle._lumina_t2_pass2_kernel,
            "block_size": 1024, "num_warps": 4, "num_stages": 2,
            "runtime_args": _dual_ptr_args() + [
                ("total_elements", "fp32"), ("beta1", "fp32"), ("beta2", "fp32"), ("c1", "fp32"),
                ("c2", "fp32"), ("lr", "fp32"), ("weight_decay", "fp32"), ("eps", "fp32"),
                ("clamp_min", "fp32"), ("bound_ratio", "fp32"), ("seed", "i32"), ("step_i32", "i32"),
            ],
            "constexprs": dict(c_std),
        },
        {
            "name": "_lumina_post_step_ema_kernel",
            "fn": common._lumina_post_step_ema_kernel,
            "block_size": 1, "num_warps": 1, "num_stages": 2,
            "runtime_args": [
                ("scratch_ptr", "*fp32"), ("total_elements", "fp32"), ("lr", "fp32"),
                ("clamp_min", "fp32"), ("ema_decay", "fp32"), ("bound_ratio", "fp32"),
            ],
            "constexprs": {"bound_mode": 2, "cautious": True, "is_triangle": False},
        },
    ]


def _normalize_jit_paths(modules: Dict[str, Any]) -> None:
    """Replace absolute host source paths on JIT functions with repo-relative paths for deterministic, clean ELFs."""
    for mod_name, mod in modules.items():
        rel = f"luminamoon/optimizer/backends/triton/{mod_name}.py"
        for _, obj in inspect.getmembers(mod):
            if hasattr(obj, "fn") and hasattr(obj.fn, "__code__") and hasattr(obj, "file_name"):
                obj.file_name = rel
                obj.fn.__code__ = obj.fn.__code__.replace(co_filename=rel)


def _count_ptx_params(ptx: str, symbol: str) -> int:
    match = re.search(rf"\.visible\s+\.entry\s+{re.escape(symbol)}\s*\(([\s\S]*?)\)", ptx)
    if not match:
        raise RuntimeError(f"PTX entry for {symbol} not found")
    return len(re.findall(r"\.param\s+", match.group(1)))


def _count_amdgcn_params(amdgcn: str) -> int:
    match = re.search(
        r"^\s*(?:-\s+)?\.args:\s*\r?\n([\s\S]*?)^\s*\.(?:group_segment|kernarg_segment|max_flat_workgroup|name)",
        amdgcn,
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError("AMDGCN kernel argument metadata not found")
    block = match.group(1)
    kinds = re.findall(r"\.value_kind:\s*(\S+)", block)
    visible = [k for k in kinds if not k.startswith("hidden_")]
    if not visible:
        raise RuntimeError("AMDGCN kernel visible argument count is zero")
    return len(visible)


def _validate_elf_artifact(raw: bytes, label: str) -> None:
    if len(raw) < 128 or not raw.startswith(b"\x7fELF"):
        raise RuntimeError(f"{label}: invalid ELF64 artifact ({len(raw)} bytes)")
    payload = raw[64:]
    for magic in STUB_MAGICS:
        if payload.startswith(magic):
            raise RuntimeError(f"{label}: synthetic stub binary detected")
    leaks = [m.group(0) for m in DRIVE_PATH_RE.finditer(raw) if not m.group(0).startswith(b"s://")]
    if leaks:
        raise RuntimeError(f"{label}: embedded absolute host path detected: {leaks[0][:48]!r}")


def _compile_target(
    tc: Any,
    gpu_target_cls: Any,
    spec: Dict[str, Any],
    platform: str,
    arch_cfg: Dict[str, Any],
) -> Tuple[bytes, Dict[str, Any], str, int]:
    signature = {name: ty for name, ty in spec["runtime_args"]}
    src = tc.ASTSource(fn=spec["fn"], signature=signature, constexprs=spec["constexprs"])
    if platform == "cuda":
        target = gpu_target_cls("cuda", arch_cfg["cc"], arch_cfg["warp_size"])
        ext, asm_key = "cubin", "cubin"
    else:
        target = gpu_target_cls("hip", arch_cfg["arch"], arch_cfg["warp_size"])
        ext, asm_key = "hsaco", "hsaco"

    compiled = tc.compile(
        src,
        target=target,
        options={"num_warps": spec["num_warps"], "num_stages": spec["num_stages"], "debug": False},
    )
    raw: bytes = compiled.asm[asm_key]
    arch = arch_cfg["arch"]
    fname = f"{spec['name']}_{arch}.{ext}"
    _validate_elf_artifact(raw, fname)

    symbol = str(compiled.metadata.name)
    total_params = (
        _count_ptx_params(compiled.asm["ptx"], symbol)
        if platform == "cuda"
        else _count_amdgcn_params(compiled.asm["amdgcn"])
    )
    trailing = total_params - len(spec["runtime_args"])
    if trailing < 1:
        raise RuntimeError(f"{fname}: expected >= 1 trailing scratch ptr, got {trailing}")

    rel_path = f"artifacts/{fname}"
    target_meta = {
        "platform": platform,
        "arch": arch,
        "family": arch_cfg["family"],
        "binary_path": rel_path,
        "shared_memory_bytes": int(compiled.metadata.shared),
        "num_registers": 0,
        "warp_size": arch_cfg["warp_size"],
        "num_warps": spec["num_warps"],
        "block_threads": spec["num_warps"] * arch_cfg["warp_size"],
        "num_trailing_scratch_ptrs": trailing,
        "bf16_mode": arch_cfg["bf16_mode"],
    }
    return raw, target_meta, symbol, trailing


def _write_bytes_safe(path: Path, raw: bytes) -> None:
    if path.exists() and path.read_bytes() == raw:
        return
    for attempt in range(5):
        try:
            path.write_bytes(raw)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.1 * (attempt + 1))


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="luminav_triton_aot_") as tmp_cache:
        os.environ["TRITON_CACHE_DIR"] = tmp_cache
        os.environ["TRITON_DISABLE_LINE_INFO"] = "1"
        import triton
        import triton.compiler as tc
        from triton.backends.compiler import GPUTarget
        from luminamoon.optimizer.backends.triton import common, dual, tdclm, triangle

        if hasattr(triton, "knobs") and hasattr(triton.knobs, "compilation"):
            triton.knobs.compilation.disable_line_info = True

        modules = {"common": common, "dual": dual, "tdclm": tdclm, "triangle": triangle}
        _normalize_jit_paths(modules)
        specs = _kernel_specs(modules)

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        generated_files: set[str] = set()
        manifest_kernels: List[Dict[str, Any]] = []
        observed_trailing: set[int] = set()

        for spec in specs:
            kname = spec["name"]
            binaries: Dict[str, str] = {}
            targets: Dict[str, Dict[str, Any]] = {}
            symbol_name = kname
            max_shared = 0
            kernel_trailing = 2

            for platform, cfgs in (("cuda", CUDA_ARCHS), ("rocm", ROCM_ARCHS)):
                for cfg in cfgs:
                    raw, tmeta, sym, trailing = _compile_target(tc, GPUTarget, spec, platform, cfg)
                    fname = Path(tmeta["binary_path"]).name
                    _write_bytes_safe(ARTIFACT_DIR / fname, raw)
                    generated_files.add(fname)
                    binaries[cfg["arch"]] = tmeta["binary_path"]
                    targets[cfg["arch"]] = tmeta
                    symbol_name = sym
                    max_shared = max(max_shared, tmeta["shared_memory_bytes"])
                    kernel_trailing = trailing
                    observed_trailing.add(trailing)

            manifest_kernels.append(
                {
                    "kernel_name": kname,
                    "symbol_name": symbol_name,
                    "shared_mem_bytes": max_shared,
                    "num_regs": 0,
                    "num_warps": spec["num_warps"],
                    "num_stages": spec["num_stages"],
                    "block_size": spec["block_size"],
                    "num_trailing_scratch_ptrs": kernel_trailing,
                    "abi_spec": {
                        "triton_version": str(triton.__version__),
                        "trailing_scratch_convention": "pass &0u64 (*const u64) for each trailing scratch parameter",
                        "ast_parameters": list(spec["fn"].arg_names),
                        "runtime_signature": [{"name": n, "ty": t} for n, t in spec["runtime_args"]],
                        "constexpr_constants": {k: int(v) for k, v in spec["constexprs"].items()},
                    },
                    "binaries": binaries,
                    "targets": targets,
                }
            )
            print(f"[AOT] Compiled {kname}: {len(targets)} ELF64 binaries (trailing_scratch={kernel_trailing}, max_shared={max_shared}B)")

        for stale in ARTIFACT_DIR.iterdir():
            if stale.suffix in {".cubin", ".hsaco"} and stale.name not in generated_files:
                stale.unlink()

        manifest = {
            "schema_version": "2.0.0",
            "math_version": "2.0.0",
            "generator": "optimizer/tools/aot_compiler/compile_triton_aot.py",
            "num_trailing_scratch_ptrs": max(observed_trailing),
            "supported_cuda_archs": [c["arch"] for c in CUDA_ARCHS],
            "supported_rocm_archs": [r["arch"] for r in ROCM_ARCHS],
            "kernels": manifest_kernels,
        }
        (ARTIFACT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Compiled {len(generated_files)} real GPU ELF64 binaries across {len(specs)} kernels.")


if __name__ == "__main__":
    main()
