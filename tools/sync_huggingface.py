import argparse
import ast
import importlib.util
import json
import os
import re
import shutil
from pathlib import Path

from prepare_mkdocs import normalize_gfm_math

REFERENCE_MODULE_ORDER = [
    "utils/math_sr.py",
    "utils/fsdp_sync.py",
    "kernels/common.py",
    "kernels/dual_kernels.py",
    "kernels/tdclm_kernels.py",
    "core/validation.py",
    "core/state_manager.py",
    "single_backend/fsdp_pass1.py",
    "single_backend/dual_chunked.py",
    "single_backend/dual_step.py",
    "single_backend/tdclm_chunked.py",
    "single_backend/tdclm_step.py",
    "single_backend/__init__.py",
    "triton_backend/base.py",
    "triton_backend/dual.py",
    "triton_backend/tdclm.py",
    "triton_backend/__init__.py",
    "core/optimizer.py",
]

STANDALONE_HEADER = '''# Copyright (c) 2026 Lumina Moon and Contributors.
# Licensed under the Apache License, Version 2.0 (see LICENSE for details).

"""
LuminaV v1.5.3 (PROJECT LUMI / luminamoon):
Low-Precision BF16/FP16 Deep Learning Optimizer with Dual-Buffer Convex Lookahead
Momentum (CLM), Single-Buffer Triangle-Defect Convex Lookahead Momentum (TD-CLM),
Rational Pade [13/6] Hyperbolic Bounding, and Stateless 32-Bit MurmurHash3
Stochastic Rounding.

Official GitHub : https://github.com/cloverx-id/luminamoon
Docs Portal     : https://cloverx-id.github.io/luminamoon/
Official HF Hub : https://huggingface.co/cloverx-id/LuminaV-Optimizer-Paper
Software DOI    : 10.57967/hf/10365
Paper DOI       : 10.57967/hf/10270
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING, Tuple, Union

import torch
from torch import Tensor
from torch.optim import Optimizer
'''


def strip_imports_and_dunder_all(source: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    skip_ranges: list[tuple[int, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            assert node.lineno is not None and node.end_lineno is not None
            skip_ranges.append((node.lineno, node.end_lineno))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    assert node.lineno is not None and node.end_lineno is not None
                    skip_ranges.append((node.lineno, node.end_lineno))

    kept: list[str] = []
    for idx, line in enumerate(lines, start=1):
        if any(start <= idx <= end for start, end in skip_ranges):
            continue
        kept.append(line)
    return "".join(kept).strip()


def build_standalone_luminav_py(repo_root: Path) -> str:
    ref_root = repo_root / "optimizer" / "reference"
    parts = [STANDALONE_HEADER.rstrip()]
    for rel_mod in REFERENCE_MODULE_ORDER:
        mod_path = ref_root / rel_mod
        raw = mod_path.read_text(encoding="utf-8")
        cleaned = strip_imports_and_dunder_all(raw)
        if cleaned:
            parts.append(cleaned)
    parts.append('\n__all__ = ["LuminaV", "HAS_TRITON", "__version__", "__author__", "__organization__", "__license__"]\n')
    return "\n\n\n".join(parts)


def build_config_json() -> dict:
    return {
        "name": "LuminaV",
        "version": "1.5.3",
        "package": "luminamoon",
        "package_version": "1.0.5",
        "description": (
            "Master-free, low-precision BF16/FP16 deep learning optimizer with Dual-Buffer Convex Lookahead Momentum (CLM), "
            "Single-Buffer Triangle-Defect Convex Lookahead Momentum (TD-CLM) via the Triangle Inequality Envelope, "
            "rational Pade [13/6] bounding, stateless 32-bit MurmurHash3 stochastic rounding, and native Rust SIMD / Vulkan / Triton execution."
        ),
        "repository": "https://github.com/cloverx-id/luminamoon",
        "documentation": "https://cloverx-id.github.io/luminamoon/",
        "paper": {
            "title": "LuminaV: We Were Too Broke for AdamW So We Trapped Gradients in a Hyperbolic Straitjacket and Hired a Traffic Cop to Slap Them",
            "doi": "10.57967/hf/10270",
            "software_doi": "10.57967/hf/10365",
            "license": "Apache-2.0",
        },
        "default_params": {
            "lr": 0.0008,
            "betas": [0.9, 0.999],
            "eps": 1e-08,
            "weight_decay": 0.08,
            "tau": 0.8,
            "lam": 0.3,
            "cautious": True,
            "cautious_clamp_min": 0.5,
            "buffer": 2,
            "centered_defect": False,
            "stochastic_rounding": True,
            "bound": True,
            "bound_type": "coordinate",
            "bound_ratio": 0.05,
            "master_weights": "none",
            "execution": "auto",
            "fused_single_pass": True,
            "ema_decay": "auto",
            "fsdp_sync": False,
            "seed": 1337,
        },
        "presets": {
            "dual_buffer_clm": {
                "buffer": 2,
                "tau": 0.8,
                "cautious": True,
                "cautious_clamp_min": 0.5,
                "centered_defect": False,
                "bound": True,
                "bound_type": "coordinate",
                "bound_ratio": 0.05,
                "stochastic_rounding": True,
                "master_weights": "none",
                "fused_single_pass": True,
                "ema_decay": "auto",
                "fsdp_sync": False,
                "description": "Default Dual-Buffer Convex Lookahead Momentum (CLM) tracking m_t and centered innovation variance v_t with coordinate trust-region bounding.",
            },
            "single_buffer_td_clm": {
                "buffer": 1,
                "lam": 0.3,
                "cautious": True,
                "cautious_clamp_min": 0.5,
                "centered_defect": False,
                "bound": True,
                "bound_type": "coordinate",
                "bound_ratio": 0.05,
                "stochastic_rounding": True,
                "master_weights": "none",
                "fused_single_pass": True,
                "ema_decay": "auto",
                "fsdp_sync": False,
                "description": "Single-Buffer Triangle-Defect Convex Lookahead Momentum (TD-CLM) using the Triangle Inequality Envelope (cuts optimizer state RAM by 50% with zero global reductions in coordinate mode).",
            },
            "radial_trust_region_bound": {
                "buffer": 2,
                "tau": 0.8,
                "cautious": True,
                "cautious_clamp_min": 0.5,
                "bound": True,
                "bound_type": "radial",
                "bound_ratio": 0.03,
                "stochastic_rounding": True,
                "master_weights": "none",
                "fused_single_pass": True,
                "ema_decay": "auto",
                "fsdp_sync": False,
                "description": "Direction-preserving radial trust-region bounding (tanh(r)/r) preserving 100% update angular geometry.",
            },
            "semi_hybrid_fp32_master": {
                "buffer": 2,
                "cautious": True,
                "cautious_clamp_min": 0.5,
                "bound": True,
                "bound_type": "coordinate",
                "bound_ratio": 0.05,
                "stochastic_rounding": True,
                "master_weights": "semi",
                "fused_single_pass": True,
                "fsdp_sync": False,
                "description": "Hybrid precision pairing 32-bit FP32 master weights with 16-bit BF16/FP16 optimizer moment buffers.",
            },
            "full_fp32_baseline": {
                "buffer": 2,
                "cautious": True,
                "cautious_clamp_min": 0.5,
                "bound": True,
                "bound_type": "coordinate",
                "bound_ratio": 0.05,
                "stochastic_rounding": True,
                "master_weights": "full",
                "fused_single_pass": True,
                "fsdp_sync": False,
                "description": "Full-precision FP32 master weights and FP32 optimizer states for controlled baseline evaluation.",
            },
            "fsdp_distributed_sync": {
                "buffer": 2,
                "cautious": True,
                "cautious_clamp_min": 0.5,
                "bound": True,
                "bound_type": "radial",
                "bound_ratio": 0.03,
                "stochastic_rounding": True,
                "master_weights": "none",
                "fused_single_pass": True,
                "fsdp_sync": True,
                "description": "Distributed multi-GPU training mode under PyTorch FSDP or DeepSpeed ZeRO-3 with cross-rank collective scratch synchronization.",
            },
        },
    }


def build_hf_readme(repo_root: Path, repo_name: str) -> str:
    opt_readme = (repo_root / "optimizer" / "README.md").read_text(encoding="utf-8")
    opt_readme_norm = normalize_gfm_math(opt_readme)
    opt_readme_norm = opt_readme_norm.replace(
        "(../changelogs/LUMINAV_ALGORITHM_CHANGELOG.md)",
        "(https://huggingface.co/cloverx-id/LuminaV-Optimizer-Paper/blob/main/CHANGELOG.md)",
    ).replace(
        "(../eval/README.md)",
        "(https://github.com/cloverx-id/luminamoon/blob/main/eval/README.md)",
    ).replace(
        "(../docs/README.md)",
        "(https://cloverx-id.github.io/luminamoon/)",
    )

    header = f"""---
license: apache-2.0
library_name: pytorch
language:
- en
tags:
- pytorch
- optimizer
- custom-optimizer
- luminamoon
- luminav
- clm
- td-clm
- rust
- simd
- avx512
- vulkan
- triton
- memory-efficient
- stochastic-rounding
- low-vram
- bf16
- fp16
- cuda
- rocm
- xpu
- llm
- transformer
- deep-learning
- research
---

<div align="center">
  <h2>LuminaV Optimizer v1.5.3 (<code>luminamoon v1.0.5</code>)</h2>
  <p><em>Low-Precision BF16/FP16 Deep Learning Optimizer with Dual-Buffer Convex Lookahead Momentum (CLM), Single-Buffer Triangle-Defect Convex Lookahead Momentum (TD-CLM), Rational Padé [13/6] Bounding, and Stateless 32-Bit MurmurHash3 Stochastic Rounding</em></p>
  <p><strong>Official Codebase: <a href="https://github.com/cloverx-id/luminamoon">github.com/cloverx-id/luminamoon</a> | Docs Portal: <a href="https://cloverx-id.github.io/luminamoon/">cloverx-id.github.io/luminamoon</a></strong></p>

  <p>
    <a href="https://pypi.org/project/luminamoon/"><img src="https://img.shields.io/pypi/v/luminamoon.svg?style=flat-square&label=pypi%20luminamoon" alt="PyPI luminamoon" /></a>
    <a href="https://pypi.org/project/luminav/"><img src="https://img.shields.io/pypi/v/luminav.svg?style=flat-square&label=pypi%20luminav" alt="PyPI luminav" /></a>
    <a href="https://cloverx-id.github.io/luminamoon/"><img src="https://img.shields.io/badge/Docs-GitHub%20Pages-indigo?style=flat-square" alt="Documentation Portal" /></a>
    <a href="https://huggingface.co/cloverx-id/{repo_name}/blob/main/CHANGELOG.md"><img src="https://img.shields.io/badge/Algorithm_Changelog-v1.5.3-blue?style=flat-square" alt="Changelog" /></a>
    <a href="https://huggingface.co/cloverx-id/{repo_name}/blob/main/BENCHMARKS.md"><img src="https://img.shields.io/badge/Benchmark-Qwen3.5--4B%20Verified-brightgreen?style=flat-square" alt="Benchmark" /></a>
    <a href="https://doi.org/10.57967/hf/10365"><img src="https://img.shields.io/badge/Software_DOI-10.57967%2Fhf%2F10365-purple?style=flat-square" alt="Software DOI" /></a>
    <a href="https://doi.org/10.57967/hf/10270"><img src="https://img.shields.io/badge/Paper_DOI-10.57967%2Fhf%2F10270-darkblue?style=flat-square" alt="Paper DOI" /></a>
    <a href="https://huggingface.co/cloverx-id/{repo_name}/blob/main/config.json"><img src="https://img.shields.io/badge/Config-JSON-green?style=flat-square" alt="Config JSON" /></a>
    <a href="https://huggingface.co/cloverx-id/{repo_name}/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-yellow?style=flat-square" alt="License" /></a>
  </p>

  <table>
    <tr>
      <td align="center" width="100%">
        <h3>Foundational Research Paper &amp; Living Specification</h3>
        <a href="https://huggingface.co/cloverx-id/{repo_name}/blob/main/LuminaV.pdf">
          <img src="https://cdn-uploads.huggingface.co/production/uploads/6835b520186e98712b0386a6/GfVA7g5xUHwopzOSRR5QD.png" alt="LuminaV Paper Preview" width="380" style="border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);" />
        </a>
        <br /><br />
        <strong>LuminaV Optimizer Theory &amp; Mechanics</strong><br />
        <a href="https://huggingface.co/cloverx-id/{repo_name}/blob/main/LuminaV.pdf">Read LuminaV.pdf (Foundational v1.0 Archive)</a>
        &nbsp;|&nbsp;
        <a href="https://huggingface.co/cloverx-id/{repo_name}/blob/main/CHANGELOG.md">LuminaV v1.5.3 Mathematical Changelog</a>
        &nbsp;|&nbsp;
        <a href="https://cloverx-id.github.io/luminamoon/">Full Online Documentation</a>
      </td>
    </tr>
  </table>
</div>

---
"""

    benchmarks_and_citation = f"""
---

## Empirical Pretraining Benchmarks (`Qwen3.5-4B-Base`)

LuminaV was evaluated on a 4.0-billion parameter Large Language Model (**`Qwen/Qwen3.5-4B-Base`**) initialized from architectural config (`AutoModelForCausalLM.from_config`) under **full pretraining from scratch** conditions (all 4.0B parameters actively optimized in native `bfloat16`, activation checkpointing disabled) on an NVIDIA A100 80GB GPU.

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/6835b520186e98712b0386a6/EwzHFVonjWyHslWpPf6Av.png" alt="LuminaV Qwen3.5-4B Benchmark Charts" width="100%" />
</div>

| Optimizer Mode | Kernel Passes | Buffer Mode | Peak VRAM | Static State VRAM | Pure Opt Latency | Final Loss (250) | Weight Cosine Sim (vs 2P) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Master-Free (1P)** | 1-Pass (EMA) | `buffer=2` (Dual) | **34.32 GB** *(-47.6%)* | **24.65 GB** *(-55.8%)* | **59.9 ms** *(1.90x faster)* | **8.8260** | 0.9445 |
| **Master-Free (2P)** | 2-Pass (Exact) | `buffer=2` (Dual) | **34.32 GB** *(-47.6%)* | **24.65 GB** *(-55.8%)* | 68.4 ms | 9.1075 | 1.0000 (Exact) |
| **Semi (1P)** | 1-Pass (EMA) | `buffer=2` (Dual) | 49.99 GB *(-23.6%)* | 40.32 GB *(-27.7%)* | **70.2 ms** *(1.62x faster)* | **8.8192** *(Best)* | 0.9512 |
| **Full (2P - Baseline)** | 2-Pass (Exact) | `buffer=2` (Dual) | 65.47 GB | 55.80 GB | 113.8 ms | 8.9579 | 1.0000 (Exact) |

- **[Read the Full Empirical Benchmark Report (`BENCHMARKS.md`)](https://huggingface.co/cloverx-id/{repo_name}/blob/main/BENCHMARKS.md)**
- **[Interactive Reproduction Notebook in Google Colab](https://colab.research.google.com/drive/1vgNIo8O2MpH9fU-kreg08vPwaiucsXik?usp=sharing)**

---

## Citation

```bibtex
@misc{{luminamoon2026luminav_paper,
  author       = {{{{Silver Moon (cloverxion)}}}},
  organization = {{Lumina Moon (cloverx-id)}},
  title        = {{{{LuminaV: We Were Too Broke for AdamW So We Trapped Gradients in a Hyperbolic Straitjacket and Hired a Traffic Cop to Slap Them}}}},
  year         = {{2026}},
  publisher    = {{Hugging Face}},
  doi          = {{10.57967/hf/10270}},
  url          = {{https://huggingface.co/cloverx-id/XoneLM-1.0-Paper}}
}}

@software{{luminamoon2026luminav_code,
  author       = {{{{Silver Moon (cloverxion) and Lumina Moon Contributors}}}},
  organization = {{Lumina Moon (cloverx-id)}},
  title        = {{{{LuminaMoon & LuminaV Optimizer: Native Rust SIMD, Vulkan, and Triton Deep Learning Suite}}}},
  year         = {{2026}},
  publisher    = {{GitHub / Hugging Face / PyPI}},
  version      = {{1.5.3}},
  doi          = {{10.57967/hf/10365}},
  url          = {{https://github.com/cloverx-id/luminamoon}}
}}
```
"""
    return header + "\n" + opt_readme_norm.strip() + "\n" + benchmarks_and_citation


def stage_hf_repo(repo_root: Path, stage_dir: Path, repo_name: str) -> None:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    luminav_py = build_standalone_luminav_py(repo_root)
    (stage_dir / "luminav.py").write_text(luminav_py, encoding="utf-8")

    cfg = build_config_json()
    (stage_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    changelog_raw = (repo_root / "changelogs" / "LUMINAV_ALGORITHM_CHANGELOG.md").read_text(encoding="utf-8")
    (stage_dir / "CHANGELOG.md").write_text(normalize_gfm_math(changelog_raw), encoding="utf-8")

    readme_text = build_hf_readme(repo_root, repo_name)
    (stage_dir / "README.md").write_text(readme_text, encoding="utf-8")

    shutil.copyfile(repo_root / "LICENSE", stage_dir / "LICENSE")


def verify_staged_luminav(stage_dir: Path) -> None:
    lum_path = stage_dir / "luminav.py"
    spec = importlib.util.spec_from_file_location("_hf_standalone_luminav", lum_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert getattr(mod, "__version__", None) == "1.5.3", f"Unexpected version: {getattr(mod, '__version__', None)}"

    import torch
    for buf in (2, 1):
        p = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float32, requires_grad=True)
        p.grad = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float32)
        opt = mod.LuminaV([p], lr=1e-3, buffer=buf)
        opt.step()
        assert opt.state[p]["step"] == 1
    shutil.rmtree(stage_dir / "__pycache__", ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage and sync LuminaV v1.5.3 artifacts to Hugging Face Hub.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stage-root", type=Path, default=Path(__file__).resolve().parents[1] / "dist" / "hf_stage")
    parser.add_argument("--push", action="store_true", help="Upload staged directories to Hugging Face Hub.")
    parser.add_argument(
        "--commit-message",
        type=str,
        default="sync(v1.5.3): update LuminaV CLM/TD-CLM specification, standalone luminav.py, CHANGELOG.md, and config.json",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    stage_root = args.stage_root.resolve()
    hf_org = os.environ.get("HF_ORG", "cloverx-id").strip() or "cloverx-id"
    target_repos = ["LuminaV-Optimizer", "LuminaV-Optimizer-Paper"]

    for repo_name in target_repos:
        r_stage = stage_root / repo_name
        stage_hf_repo(repo_root, r_stage, repo_name)
        verify_staged_luminav(r_stage)
        print(f"Staged and verified Hugging Face bundle for {hf_org}/{repo_name} at {r_stage}")

    if args.push:
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if not hf_token:
            raise RuntimeError("HF_TOKEN environment variable is required when --push is set.")
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)
        for repo_name in target_repos:
            full_repo_id = f"{hf_org}/{repo_name}"
            r_stage = stage_root / repo_name
            commit_info = api.upload_folder(
                repo_id=full_repo_id,
                repo_type="model",
                folder_path=str(r_stage),
                commit_message=args.commit_message, ignore_patterns=["__pycache__/**", "*.pyc", "*.pyo"],
            )
            print(f"Synced {full_repo_id}: {commit_info.commit_url}")


if __name__ == "__main__":
    main()
