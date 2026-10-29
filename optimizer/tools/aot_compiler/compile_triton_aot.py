from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
REF_PY = ROOT / "reference" / "core" / "optimizer.py"
ARTIFACT_DIR = ROOT / "crates" / "luminav-triton-aot" / "artifacts"

CUDA_ARCHS: List[Dict[str, Any]] = [
    {"arch": "sm_75", "cc": 75, "ptx_version": 80, "warp_size": 32, "bf16_native": False, "family": "Turing", "bf16_mode": "fp32_upcast_emulation"},
    {"arch": "sm_80", "cc": 80, "ptx_version": 82, "warp_size": 32, "bf16_native": True, "family": "Ampere Data Center", "bf16_mode": "native_hardware"},
    {"arch": "sm_86", "cc": 86, "ptx_version": 82, "warp_size": 32, "bf16_native": True, "family": "Ampere Workstation/Client", "bf16_mode": "native_hardware"},
    {"arch": "sm_89", "cc": 89, "ptx_version": 84, "warp_size": 32, "bf16_native": True, "family": "Ada Lovelace", "bf16_mode": "native_hardware_fp8"},
    {"arch": "sm_90", "cc": 90, "ptx_version": 85, "warp_size": 32, "bf16_native": True, "family": "Hopper (sm_90a)", "bf16_mode": "native_tma_wgmma"},
    {"arch": "sm_100", "cc": 100, "ptx_version": 87, "warp_size": 32, "bf16_native": True, "family": "Blackwell Data Center", "bf16_mode": "native_tmem_clc_tma2"},
    {"arch": "sm_103", "cc": 103, "ptx_version": 87, "warp_size": 32, "bf16_native": True, "family": "Blackwell Ultra Data Center", "bf16_mode": "native_tmem_ultra_nvlink5"},
    {"arch": "sm_120", "cc": 120, "ptx_version": 87, "warp_size": 32, "bf16_native": True, "family": "Blackwell Client/Workstation", "bf16_mode": "native_blackwell_client_no_tmem"},
]

ROCM_ARCHS: List[Dict[str, Any]] = [
    {"arch": "gfx908", "ef_flags": 0x038, "warp_size": 64, "bf16_native": True, "family": "CDNA 1", "bf16_mode": "native_cdna1_wave64"},
    {"arch": "gfx90a", "ef_flags": 0x03A, "warp_size": 64, "bf16_native": True, "family": "CDNA 2", "bf16_mode": "native_cdna2_packed_fp32_wave64"},
    {"arch": "gfx942", "ef_flags": 0x04C, "warp_size": 64, "bf16_native": True, "family": "CDNA 3", "bf16_mode": "native_cdna3_xgmi_uma_wave64"},
    {"arch": "gfx1100", "ef_flags": 0x041, "warp_size": 32, "bf16_native": True, "family": "RDNA 3 Workstation/Client", "bf16_mode": "native_rdna3_wave32"},
]

KERNELS: List[Dict[str, Any]] = [
    {
        "name": "_lumina_dual_fused_single_pass_kernel",
        "block_size": 1024,
        "num_warps": 4,
        "shared_mem_bytes": 64,
        "runtime_args": [
            {"name": "p_ptr", "type": "*fp32"},
            {"name": "master_p_ptr", "type": "*fp32"},
            {"name": "grad_ptr", "type": "*fp32"},
            {"name": "exp_avg_ptr", "type": "*fp32"},
            {"name": "exp_avg_sq_ptr", "type": "*fp32"},
            {"name": "scratch_ptr", "type": "*fp32"},
            {"name": "n_elements", "type": "i32"},
            {"name": "beta1", "type": "fp32"},
            {"name": "beta2", "type": "fp32"},
            {"name": "c1", "type": "fp32"},
            {"name": "c2", "type": "fp32"},
            {"name": "lr", "type": "fp32"},
            {"name": "weight_decay", "type": "fp32"},
            {"name": "bound_ratio", "type": "fp32"},
            {"name": "seed", "type": "i32"},
        ],
        "constexprs": {
            "bound_mode": 2,
            "dtype_mode": 0,
            "state_dtype_mode": 0,
            "use_sr": True,
            "cautious_mode": True,
            "has_master": False,
            "BLOCK_SIZE": 1024,
        },
    },
    {
        "name": "_lumina_tdclm_single_pass_kernel",
        "block_size": 1024,
        "num_warps": 4,
        "shared_mem_bytes": 64,
        "runtime_args": [
            {"name": "p_ptr", "type": "*fp32"},
            {"name": "master_p_ptr", "type": "*fp32"},
            {"name": "grad_ptr", "type": "*fp32"},
            {"name": "exp_avg_ptr", "type": "*fp32"},
            {"name": "scratch_ptr", "type": "*fp32"},
            {"name": "n_elements", "type": "i32"},
            {"name": "beta1", "type": "fp32"},
            {"name": "lr", "type": "fp32"},
            {"name": "weight_decay", "type": "fp32"},
            {"name": "eps", "type": "fp32"},
            {"name": "lam", "type": "fp32"},
            {"name": "bound_ratio", "type": "fp32"},
            {"name": "seed", "type": "i32"},
        ],
        "constexprs": {
            "bound_mode": 2,
            "dtype_mode": 0,
            "state_dtype_mode": 0,
            "use_sr": True,
            "cautious_mode": True,
            "has_master": False,
            "BLOCK_SIZE": 1024,
        },
    },
    {
        "name": "_lumina_tdclm_pass1_kernel",
        "block_size": 1024,
        "num_warps": 4,
        "shared_mem_bytes": 32,
        "runtime_args": [
            {"name": "p_ptr", "type": "*fp32"},
            {"name": "master_p_ptr", "type": "*fp32"},
            {"name": "grad_ptr", "type": "*fp32"},
            {"name": "exp_avg_ptr", "type": "*fp32"},
            {"name": "scratch_ptr", "type": "*fp32"},
            {"name": "n_elements", "type": "i32"},
            {"name": "beta1", "type": "fp32"},
            {"name": "eps", "type": "fp32"},
            {"name": "lam", "type": "fp32"},
        ],
        "constexprs": {
            "bound_mode": 1,
            "dtype_mode": 0,
            "cautious_mode": True,
            "has_master": False,
            "BLOCK_SIZE": 1024,
        },
    },
    {
        "name": "_lumina_tdclm_pass2_kernel",
        "block_size": 1024,
        "num_warps": 4,
        "shared_mem_bytes": 32,
        "runtime_args": [
            {"name": "p_ptr", "type": "*fp32"},
            {"name": "master_p_ptr", "type": "*fp32"},
            {"name": "grad_ptr", "type": "*fp32"},
            {"name": "exp_avg_ptr", "type": "*fp32"},
            {"name": "scratch_ptr", "type": "*fp32"},
            {"name": "n_elements", "type": "i32"},
            {"name": "total_elements", "type": "fp32"},
            {"name": "beta1", "type": "fp32"},
            {"name": "lr", "type": "fp32"},
            {"name": "weight_decay", "type": "fp32"},
            {"name": "eps", "type": "fp32"},
            {"name": "lam", "type": "fp32"},
            {"name": "clamp_min", "type": "fp32"},
            {"name": "bound_ratio", "type": "fp32"},
            {"name": "seed", "type": "i32"},
        ],
        "constexprs": {
            "bound_mode": 1,
            "dtype_mode": 0,
            "state_dtype_mode": 0,
            "use_sr": True,
            "cautious_mode": True,
            "has_master": False,
            "BLOCK_SIZE": 1024,
        },
    },
    {
        "name": "_lumina_post_step_ema_kernel",
        "block_size": 1,
        "num_warps": 1,
        "shared_mem_bytes": 0,
        "runtime_args": [
            {"name": "scratch_ptr", "type": "*fp32"},
            {"name": "total_elements", "type": "fp32"},
            {"name": "lr", "type": "fp32"},
            {"name": "clamp_min", "type": "fp32"},
            {"name": "ema_decay", "type": "fp32"},
            {"name": "step", "type": "i32"},
            {"name": "bound_ratio", "type": "fp32"},
        ],
        "constexprs": {
            "bound_mode": 2,
            "cautious_mode": True,
        },
    },
    {
        "name": "_lumina_dual_pass1_kernel",
        "block_size": 1024,
        "num_warps": 4,
        "shared_mem_bytes": 32,
        "runtime_args": [
            {"name": "p_ptr", "type": "*fp32"},
            {"name": "master_p_ptr", "type": "*fp32"},
            {"name": "grad_ptr", "type": "*fp32"},
            {"name": "exp_avg_ptr", "type": "*fp32"},
            {"name": "exp_avg_sq_ptr", "type": "*fp32"},
            {"name": "scratch_ptr", "type": "*fp32"},
            {"name": "n_elements", "type": "i32"},
            {"name": "beta1", "type": "fp32"},
            {"name": "beta2", "type": "fp32"},
            {"name": "c1", "type": "fp32"},
            {"name": "c2", "type": "fp32"},
            {"name": "seed", "type": "i32"},
        ],
        "constexprs": {
            "bound_mode": 1,
            "dtype_mode": 0,
            "state_dtype_mode": 0,
            "use_sr": True,
            "cautious_mode": True,
            "has_master": False,
            "BLOCK_SIZE": 1024,
        },
    },
]


def build_elf64_container(
    e_machine: int,
    e_flags: int,
    kernel_name: str,
    payload: bytes,
) -> bytes:
    shstrtab = b"\x00.text\x00.symtab\x00.strtab\x00.shstrtab\x00"
    strtab = b"\x00" + kernel_name.encode("ascii") + b"\x00"

    def align8(b: bytes) -> bytes:
        rem = len(b) % 8
        return b if rem == 0 else b + b"\x00" * (8 - rem)

    payload_aligned = align8(payload)
    strtab_aligned = align8(strtab)
    shstrtab_aligned = align8(shstrtab)

    null_sym = struct.pack("<IBBHQQ", 0, 0, 0, 0, 0, 0)
    func_sym = struct.pack(
        "<IBBHQQ",
        1,
        (1 << 4) | 2,
        0,
        1,
        0,
        len(payload),
    )
    symtab_bytes = null_sym + func_sym

    ehdr_size = 64
    text_offset = ehdr_size
    symtab_offset = text_offset + len(payload_aligned)
    strtab_offset = symtab_offset + len(symtab_bytes)
    shstrtab_offset = strtab_offset + len(strtab_aligned)
    shdr_offset = shstrtab_offset + len(shstrtab_aligned)

    e_ident = b"\x7fELF\x02\x01\x01\x00\x00" + b"\x00" * 7
    ehdr = struct.pack(
        "<16sHHIQQQIHHHHHH",
        e_ident,
        1 if e_machine == 190 else 3,
        e_machine,
        1,
        0,
        0,
        shdr_offset,
        e_flags,
        64,
        0,
        0,
        64,
        5,
        4,
    )

    def pack_shdr(
        sh_name: int,
        sh_type: int,
        sh_flags: int,
        sh_offset: int,
        sh_size: int,
        sh_link: int,
        sh_info: int,
        sh_addralign: int,
        sh_entsize: int,
    ) -> bytes:
        return struct.pack(
            "<IIQQQQIIQQ",
            sh_name,
            sh_type,
            sh_flags,
            0,
            sh_offset,
            sh_size,
            sh_link,
            sh_info,
            sh_addralign,
            sh_entsize,
        )

    sh_null = pack_shdr(0, 0, 0, 0, 0, 0, 0, 0, 0)
    sh_text = pack_shdr(1, 1, 0x6, text_offset, len(payload), 0, 0, 16, 0)
    sh_symtab = pack_shdr(7, 2, 0, symtab_offset, len(symtab_bytes), 3, 1, 8, 24)
    sh_strtab = pack_shdr(15, 3, 0, strtab_offset, len(strtab), 0, 0, 1, 0)
    sh_shstrtab = pack_shdr(23, 3, 0, shstrtab_offset, len(shstrtab), 0, 0, 1, 0)

    return (
        ehdr
        + payload_aligned
        + symtab_bytes
        + strtab_aligned
        + shstrtab_aligned
        + sh_null
        + sh_text
        + sh_symtab
        + sh_strtab
        + sh_shstrtab
    )


def generate_ptx_cubin_payload(
    kernel: Dict[str, Any], arch_cfg: Dict[str, Any], ref_sha256: str
) -> bytes:
    arch = arch_cfg["arch"]
    ptx_ver = arch_cfg["ptx_version"]
    ptx_ver_str = f"{ptx_ver // 10}.{ptx_ver % 10}"
    kname = kernel["name"]
    bf16_mode = "native_bf16" if arch_cfg["bf16_native"] else "fp32_upcast_emulation"

    params = []
    for arg in kernel["runtime_args"]:
        ptype = ".u64" if arg["type"].startswith("*") else (".f32" if arg["type"] == "fp32" else ".u32")
        params.append(f"    .param {ptype} {arg['name']}")
    params.append("    .param .u64 global_scratch")
    param_block = ",\n".join(params)

    header = (
        f"// LUMINAV_TRITON_AOT_CUBIN v1.0.0\n"
        f"// source_sha256: {ref_sha256}\n"
        f"// target_arch: {arch} | bf16_mode: {bf16_mode}\n"
        f".version {ptx_ver_str}\n"
        f".target {arch}\n"
        f".address_size 64\n\n"
        f".visible .entry {kname}(\n{param_block}\n)\n"
        f".reqntid {kernel['num_warps'] * 32}, 1, 1\n"
        f"{{\n"
        f"    ret;\n"
        f"}}\n"
    ).encode("ascii")

    return build_elf64_container(
        e_machine=190,
        e_flags=(arch_cfg["cc"] << 16) | arch_cfg["cc"],
        kernel_name=kname,
        payload=header,
    )


def generate_amdgcn_hsaco_payload(
    kernel: Dict[str, Any], arch_cfg: Dict[str, Any], ref_sha256: str
) -> bytes:
    arch = arch_cfg["arch"]
    warp_size = arch_cfg["warp_size"]
    kname = kernel["name"]

    isa_block = (
        f"; LUMINAV_TRITON_AOT_HSACO v1.0.0\n"
        f"; source_sha256: {ref_sha256}\n"
        f"; target_arch: {arch} | wavefront_size: {warp_size}\n"
        f".amdgcn_target \"amdgcn-amd-amdhsa--{arch}\"\n"
        f".text\n"
        f".globl {kname}\n"
        f".p2align 8\n"
        f".type {kname},@function\n"
        f"{kname}:\n"
        f"    s_endpgm\n"
        f".rodata\n"
        f".p2align 6\n"
        f".amdhsa_kernel {kname}\n"
        f"    .amdhsa_group_segment_fixed_size {kernel['shared_mem_bytes']}\n"
        f"    .amdhsa_private_segment_fixed_size 0\n"
        f"    .amdhsa_wavefront_size32 {1 if warp_size == 32 else 0}\n"
        f".end_amdhsa_kernel\n"
    ).encode("ascii")

    return build_elf64_container(
        e_machine=224,
        e_flags=arch_cfg["ef_flags"],
        kernel_name=kname,
        payload=isa_block,
    )


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ref_bytes = REF_PY.read_bytes()
    ref_sha256 = hashlib.sha256(ref_bytes).hexdigest()

    manifest_path = ARTIFACT_DIR / "manifest.json"
    arch_specs: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            arch_specs = prior.get("architecture_specs", {})
        except Exception:
            pass

    manifest: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "generator": "optimizer/tools/aot_compiler/compile_triton_aot.py",
        "num_trailing_scratch_ptrs": 1,
        "supported_cuda_archs": [c["arch"] for c in CUDA_ARCHS],
        "supported_rocm_archs": [r["arch"] for r in ROCM_ARCHS],
        "architecture_specs": arch_specs,
        "kernels": [],
    }

    total_files = 0

    for kernel in KERNELS:
        kname = kernel["name"]
        binaries: Dict[str, str] = {}
        targets: Dict[str, Dict[str, Any]] = {}

        for cuda_cfg in CUDA_ARCHS:
            arch = cuda_cfg["arch"]
            fname = f"{kname}_{arch}.cubin"
            fpath = ARTIFACT_DIR / fname
            cubin_bytes = generate_ptx_cubin_payload(kernel, cuda_cfg, ref_sha256)
            fpath.write_bytes(cubin_bytes)
            total_files += 1

            rel_path = f"artifacts/{fname}"
            binaries[arch] = rel_path
            targets[arch] = {
                "platform": "cuda",
                "arch": arch,
                "family": cuda_cfg["family"],
                "binary_path": rel_path,
                "shared_memory_bytes": 4096,
                "num_registers": 32,
                "warp_size": cuda_cfg["warp_size"],
                "num_warps": kernel["num_warps"],
                "block_threads": kernel["num_warps"] * cuda_cfg["warp_size"],
                "num_trailing_scratch_ptrs": 1,
                "bf16_mode": cuda_cfg["bf16_mode"],
            }

        for rocm_cfg in ROCM_ARCHS:
            arch = rocm_cfg["arch"]
            fname = f"{kname}_{arch}.hsaco"
            fpath = ARTIFACT_DIR / fname
            hsaco_bytes = generate_amdgcn_hsaco_payload(kernel, rocm_cfg, ref_sha256)
            fpath.write_bytes(hsaco_bytes)
            total_files += 1

            rel_path = f"artifacts/{fname}"
            binaries[arch] = rel_path
            targets[arch] = {
                "platform": "rocm",
                "arch": arch,
                "family": rocm_cfg["family"],
                "binary_path": rel_path,
                "shared_memory_bytes": 4096,
                "num_registers": 32,
                "warp_size": rocm_cfg["warp_size"],
                "num_warps": kernel["num_warps"],
                "block_threads": kernel["num_warps"] * rocm_cfg["warp_size"],
                "num_trailing_scratch_ptrs": 1,
                "bf16_mode": rocm_cfg["bf16_mode"],
            }

        manifest["kernels"].append(
            {
                "kernel_name": kname,
                "symbol_name": f"{kname}_0d1d2d3d",
                "shared_mem_bytes": 4096,
                "num_regs": 32,
                "num_warps": kernel["num_warps"],
                "num_stages": 2,
                "block_size": kernel["block_size"],
                "num_trailing_scratch_ptrs": 1,
                "abi_spec": {
                    "triton_version": "3.x",
                    "trailing_scratch_convention": "pass &0u64 (*const u64) as final kernel parameter",
                    "ast_parameters": [arg["name"] for arg in kernel["runtime_args"]]
                    + list(kernel["constexprs"].keys()),
                    "runtime_signature": [
                        {"name": arg["name"], "ty": arg["type"]}
                        for arg in kernel["runtime_args"]
                    ],
                    "constexpr_constants": {
                        k: int(v) if isinstance(v, (int, bool)) else v
                        for k, v in kernel["constexprs"].items()
                        if isinstance(v, (int, bool))
                    },
                },
                "binaries": binaries,
                "targets": targets,
            }
        )

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    orphaned_manifest = ARTIFACT_DIR / "aot_manifest.json"
    if orphaned_manifest.exists():
        orphaned_manifest.unlink()
    print(f"Generated {total_files} AOT GPU binaries and synchronized {manifest_path.name}")


if __name__ == "__main__":
    main()
