import argparse
import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from prepare_mkdocs import MARKDOWN_LINK_RE, normalize_gfm_math

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
LuminaV v1.5.5 (PROJECT LOOMY / luminamoon):
Low-Precision BF16/FP16 Deep Learning Optimizer with Dual-Buffer Convex Lookahead
Momentum (CLM), Single-Buffer Triangle-Defect Convex Lookahead Momentum (TD-CLM),
Rational Pade [13/6] Hyperbolic Bounding, and Stateless 32-Bit MurmurHash3
Stochastic Rounding.

Official GitHub : https://github.com/cloverx-id/luminamoon
Docs Portal     : https://cloverx-id.github.io/luminamoon/
Official HF Hub : https://huggingface.co/cloverx-id/LuminaV-Optimizer-Paper
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
        "version": "1.5.5",
        "package": "luminamoon",
        "package_version": "1.0.8.post1",
        "lso_eval_version": "1.1.0",
        "description": (
            "Master-free, low-precision BF16/FP16 deep learning optimizer with Dual-Buffer Convex Lookahead Momentum (CLM), "
            "Single-Buffer Triangle-Defect Convex Lookahead Momentum (TD-CLM) via the Triangle Inequality Envelope, "
            "rational Pade [13/6] bounding, stateless 32-bit MurmurHash3 stochastic rounding, and native Rust SIMD / Vulkan / Triton execution."
        ),
        "repository": "https://github.com/cloverx-id/luminamoon",
        "documentation": "https://cloverx-id.github.io/luminamoon/",
        "paper": {
            "title": "LuminaV: We Were Too Broke for AdamW So We Trapped Gradients in a Hyperbolic Straitjacket and Hired a Traffic Cop to Slap Them",
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
            "stochastic_rounding": True,
            "bound": True,
            "bound_type": "coordinate",
            "bound_ratio": 0.05,
            "master_weights": "none",
            "execution": "auto",
            "fused_single_pass": True,
            "fuse_tensors": False,
            "bucket_cap_mb": 500.0,
            "prewarm_jit": True,
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
                "bound": True,
                "bound_type": "coordinate",
                "bound_ratio": 0.05,
                "stochastic_rounding": True,
                "master_weights": "none",
                "fused_single_pass": True,
                "ema_decay": "auto",
                "fsdp_sync": False,
                "description": "Default Dual-Buffer Convex Lookahead Momentum (CLM) tracking m_t and innovation variance v_t with coordinate trust-region bounding.",
            },
            "single_buffer_td_clm": {
                "buffer": 1,
                "lam": 0.3,
                "cautious": True,
                "cautious_clamp_min": 0.5,
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


def rewrite_hf_relative_links(
    content: str,
    origin_file: Path,
    repo_root: Path,
    repo_name: str,
) -> str:
    hf_blob_base = f"https://huggingface.co/cloverx-id/{repo_name}/blob/main/"
    hf_tree_base = f"https://huggingface.co/cloverx-id/{repo_name}/tree/main/"
    hf_map = {
        "changelogs/LUMINAV_ALGORITHM_CHANGELOG.md": f"{hf_blob_base}CHANGELOG.md",
        "LICENSE": f"{hf_blob_base}LICENSE",
        "docs/README.md": "https://cloverx-id.github.io/luminamoon/",
    }

    def _replace_link(match: re.Match[str]) -> str:
        label = match.group(1)
        raw_target = match.group(2)
        if raw_target.startswith(("http://", "https://", "mailto:", "#")):
            return match.group(0)

        parts = raw_target.split("#", 1)
        target_path_str = parts[0]
        anchor = f"#{parts[1]}" if len(parts) > 1 else ""
        if not target_path_str:
            return match.group(0)

        resolved = (origin_file.parent / target_path_str).resolve()
        try:
            rel_to_repo = resolved.relative_to(repo_root).as_posix()
        except ValueError:
            return match.group(0)

        if rel_to_repo in hf_map:
            return f"[{label}]({hf_map[rel_to_repo]}{anchor})"

        if resolved.is_dir():
            return f"[{label}]({hf_tree_base}{rel_to_repo}{anchor})"
        return f"[{label}]({hf_blob_base}{rel_to_repo}{anchor})"

    return MARKDOWN_LINK_RE.sub(_replace_link, content)


def strip_unrendered_mermaid(content: str) -> str:
    return re.sub(r"```mermaid\s*\n[\s\S]*?```\r?\n?", "", content)


def build_hf_readme(repo_root: Path, repo_name: str) -> str:
    root_readme_path = (repo_root / "README.md").resolve()
    root_readme = root_readme_path.read_text(encoding="utf-8")
    root_readme_norm = strip_unrendered_mermaid(rewrite_hf_relative_links(normalize_gfm_math(root_readme), root_readme_path, repo_root, repo_name))

    opt_readme_path = (repo_root / "optimizer" / "README.md").resolve()
    opt_readme = opt_readme_path.read_text(encoding="utf-8")
    opt_readme_norm = normalize_gfm_math(opt_readme)
    opt_readme_norm = strip_unrendered_mermaid(rewrite_hf_relative_links(opt_readme_norm, opt_readme_path, repo_root, repo_name))

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
- lso-eval
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
  <h2>LuminaV Optimizer v1.5.5 (<code>luminamoon v1.0.8.post1</code>)</h2>
  <p><em>Low-Precision BF16/FP16 Deep Learning Optimizer with Dual-Buffer Convex Lookahead Momentum (CLM), Single-Buffer Triangle-Defect Convex Lookahead Momentum (TD-CLM), Rational Padé [13/6] Bounding, and Stateless 32-Bit MurmurHash3 Stochastic Rounding</em></p>
  <p><strong>Official Codebase: <a href="https://github.com/cloverx-id/luminamoon">github.com/cloverx-id/luminamoon</a> | Docs Portal: <a href="https://cloverx-id.github.io/luminamoon/">cloverx-id.github.io/luminamoon</a></strong></p>

  <p>
    <a href="https://pypi.org/project/luminamoon/"><img src="https://img.shields.io/pypi/v/luminamoon.svg?style=flat-square&label=pypi%20luminamoon" alt="PyPI luminamoon" /></a>
    <a href="https://cloverx-id.github.io/luminamoon/"><img src="https://img.shields.io/badge/Docs-GitHub%20Pages-indigo?style=flat-square" alt="Documentation Portal" /></a>
    <a href="https://huggingface.co/cloverx-id/{repo_name}/blob/main/CHANGELOG.md"><img src="https://img.shields.io/badge/Algorithm_Changelog-v1.5.5-blue?style=flat-square" alt="Changelog" /></a>
    <a href="https://huggingface.co/cloverx-id/{repo_name}/blob/main/eval/README.md"><img src="https://img.shields.io/badge/LSO--Eval_v1.1.0-Qwen3.5--4B%20Verified-brightgreen?style=flat-square" alt="LSO-Eval v1.1.0" /></a>
    <a href="https://huggingface.co/cloverx-id/{repo_name}/blob/main/config.json"><img src="https://img.shields.io/badge/Config-JSON-green?style=flat-square" alt="Config JSON" /></a>
    <a href="https://huggingface.co/cloverx-id/{repo_name}/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-yellow?style=flat-square" alt="License" /></a>
  </p>
</div>

---
"""

    lso_eval_and_citation = f"""
---

## LSO-Eval v1.1.0: Formal Verification & 4B LLM Qualification (`Qwen3.5-4B-Base`)

LuminaV (`v1.5.3`) is formally verified and empirically qualified under the **LSO-Eval (`v1.1.0`)** standard across Stage I (12/12 Z3 SMT solver `UNSAT` safety proofs & 9/9 SymPy/TOST precision checks), Stage II (`B1–B5` non-convex surfaces, `λ* = 0.3008` equilibrium, and 1-Pass EMA parity), Stage III (`MProxyLLM` `F₁ ∧ F₂ ∧ F₃` screening), and Stage IV (`Qwen/Qwen3.5-4B-Base` 100% from-scratch pretraining in native `torch.bfloat16` with cryptographic XXH3-64 zero-leakage validation).

> **Empirical Version Attribution (`LuminaV v1.5.3`)**: The Stage II–IV empirical qualification tables below were evaluated on **`LuminaV v1.5.3`** under native `torch.bfloat16` (`dtype_mode=1`, `centered_defect=False`). In the `v1.5.3` evaluation data loader script, parallel dataset records were sampled by filtering uniqueness based on joint `(source, domain)` pairs rather than allocating unique records evenly per domain up to each domain's total target allocation; because all candidate and baseline optimizers shared this identical deterministic stream schedule, relative cross-optimizer comparisons in the `v1.5.3` tables remain strictly controlled.

### Stage IV-B: 2048 Context Full Schedule (`20,000 Steps`, `Grad Accum 1x`, `450,560` Held-Out Tokens, `SHA256=108e517e1c181cb6`)

<div align="center">
  <img src="https://huggingface.co/cloverx-id/{repo_name}/resolve/main/eval/artifacts/v1.5.3/stage4b_lso_eval_results.png" alt="LSO-Eval Stage IV-B Qwen3.5-4B Qualification (ctx=2048)" width="100%" />
</div>

| Optimizer Configuration | Class | Final EMA | Val PPL | ΔPPL | VRAM<sub>peak</sub> | **VRAM<sub>opt</sub>** | Max R<sub>t,l</sub> | Step / Opt Time | LSO-Eval Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`LuminaV_Buffer2_Dual` (`8e-4`)** | Class I (Dual) | **`4.5549`** | **`89.14`** | **`-43.43`** | `43.89 GB` | `15.86 GB` | `0.0189` | `297.1 / 48.31 ms` | **`PASSED (Class I)`** |
| **`LuminaV_Buffer1_TDCLM_3e4` (`3e-4`)** | Class II (Single) | `4.9173` | `165.13` | `+32.56` | **`35.97 GB`** | **`7.94 GB (-50.0%)`** | `0.0087` | **`283.5 / 36.99 ms`** | **`PASSED (Class II)`** |
| `AdamW` (Native BF16, `3e-4`) | Baseline | `4.7621` | `132.57` | `+0.00` | `43.89 GB` | `15.86 GB` | `0.0037` | `288.0 / 41.48 ms` | `PASSED` |
| `AdamW_FP32_Master` (`3e-4`) | Master Ref | `4.8116` | `124.43` | `-8.14` | `80.11 GB` | `52.08 GB` | `0.0045` | `370.1 / 123.33 ms` | `PASSED` |

### Stage IV-A: 512 Context Qualification (`4,800 Steps`, `Grad Accum 8x`, `450,560` Held-Out Tokens, `SHA256=4d278cb87e4abbe1`)

<div align="center">
  <img src="https://huggingface.co/cloverx-id/{repo_name}/resolve/main/eval/artifacts/v1.5.3/stage4a_lso_eval_results.png" alt="LSO-Eval Stage IV-A Qwen3.5-4B Qualification (ctx=512)" width="100%" />
</div>

| Optimizer Configuration | Class | Final EMA | Val PPL | ΔPPL | VRAM<sub>peak</sub> | **VRAM<sub>opt</sub>** | Max R<sub>t,l</sub> | Step / Opt Time | LSO-Eval Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`LuminaV_Buffer2_Dual` (`8e-4`)** | Class I (Dual) | **`4.7621`** | **`141.96`** | **`-39.21`** | `36.75 GB` | `15.88 GB` | `0.0047` | `841.6 / 48.37 ms` | **`PASSED (Class I)`** |
| **`LuminaV_Buffer1_TDCLM_3e4` (`3e-4`)** | Class II (Single) | `5.2569` | `195.58` | `+14.41` | **`28.81 GB`** | **`7.94 GB (-50.0%)`** | `0.0120` | **`829.4 / 36.77 ms`** | **`PASSED (Class II)`** |
| **`LuminaV_Buffer1_TDCLM` (`8e-4`)** | Class II (Single) | `5.2953` | `221.07` | `+39.90` | **`28.81 GB`** | **`7.94 GB (-50.0%)`** | `0.0191` | **`830.7 / 36.46 ms`** | **`PASSED (Class II)`** |
| `AdamW` (Native BF16, `3e-4`) | Baseline | `5.4497` | `181.17` | `+0.00` | `36.75 GB` | `15.88 GB` | `0.0034` | `835.4 / 41.58 ms` | `PASSED` |
| `AdamW_FP32_Master` (`3e-4`) | Master Ref | `5.0234` | `153.78` | `-27.39` | `75.36 GB` | `54.49 GB` | `0.0040` | `919.5 / 123.35 ms` | `PASSED` |

> **VRAM Decomposition**: `VRAM_peak = VRAM_base + VRAM_opt`, where the fixed non-optimizer baseline (`VRAM_base = W_BF16 + G_BF16 + A_ctx`) is **`20.87 GB`** at `ctx=512` and **`28.03 GB`** at `ctx=2048`.

- **[LSO-Eval v1.1.0 Formal Specification Manuscript (`eval/LSO-Eval.pdf`)](https://huggingface.co/cloverx-id/{repo_name}/blob/main/eval/LSO-Eval.pdf)**
- **[Full LSO-Eval v1.1.0 Verification Summary & Suite Guide (`eval/README.md`)](https://huggingface.co/cloverx-id/{repo_name}/blob/main/eval/README.md)**
- **[Archived Stage IV Telemetry, Plots & Held-Out Validation Tensors (`eval/artifacts/v1.5.3/`)](https://huggingface.co/cloverx-id/{repo_name}/tree/main/eval/artifacts/v1.5.3)**
- **[Interactive LSO-Eval v1.1.0 Reproduction Notebook in Google Colab](https://colab.research.google.com/drive/1UepDza5Wm3KRNooCA2Bn69GIeKB21a4a?usp=sharing)**

---

## Citation

```bibtex
@software{{luminamoon2026luminav_code,
  author       = {{{{Silver Moon (cloverxion) and Lumina Moon Contributors}}}},
  organization = {{Lumina Moon (cloverx-id)}},
  title        = {{{{LuminaMoon & LuminaV Optimizer: Native Rust SIMD, Vulkan, and Triton Deep Learning Suite}}}},
  year         = {{2026}},
  publisher    = {{GitHub / Hugging Face / PyPI}},
  version      = {{1.0.8}},
  url          = {{https://github.com/cloverx-id/luminamoon}}
}}
```
"""
    return (
        header
        + "\n"
        + root_readme_norm.strip()
        + "\n\n---\n\n"
        + opt_readme_norm.strip()
        + "\n"
        + lso_eval_and_citation
    )


def stage_hf_repo(repo_root: Path, stage_dir: Path, repo_name: str) -> int:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    copied_count = 0
    for rel_path in tracked_files:
        src_file = repo_root / rel_path
        if not src_file.is_file():
            continue
        dst_file = stage_dir / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        if src_file.suffix.lower() == ".md":
            raw_md = src_file.read_text(encoding="utf-8")
            dst_file.write_text(normalize_gfm_math(raw_md), encoding="utf-8")
        else:
            shutil.copyfile(src_file, dst_file)
        copied_count += 1

    luminav_py = build_standalone_luminav_py(repo_root)
    (stage_dir / "luminav.py").write_text(luminav_py, encoding="utf-8")

    cfg = build_config_json()
    (stage_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    changelog_path = (repo_root / "changelogs" / "LUMINAV_ALGORITHM_CHANGELOG.md").resolve()
    changelog_raw = changelog_path.read_text(encoding="utf-8")
    changelog_norm = rewrite_hf_relative_links(normalize_gfm_math(changelog_raw), changelog_path, repo_root, repo_name)
    (stage_dir / "CHANGELOG.md").write_text(changelog_norm, encoding="utf-8")

    readme_text = build_hf_readme(repo_root, repo_name)
    (stage_dir / "README.md").write_text(readme_text, encoding="utf-8")

    shutil.copyfile(repo_root / "LICENSE", stage_dir / "LICENSE")
    return copied_count


def verify_staged_luminav(stage_dir: Path) -> None:
    required_paths = (
        "Cargo.toml",
        "Cargo.lock",
        "pyproject.toml",
        "SECURITY.md",
        "optimizer/Cargo.toml",
        "data/Cargo.toml",
        "eval/README.md",
        "eval/LSO-Eval.pdf",
        "eval/artifacts/v1.5.3/stage4a_lso_eval_results.png",
        "eval/artifacts/v1.5.3/stage4b_lso_eval_results.png",
        "python/luminamoon/__init__.py",
        "docs/README.md",
        "changelogs/README.md",
        "luminav.py",
        "config.json",
        "CHANGELOG.md",
        "README.md",
    )
    for req_rel in required_paths:
        assert (stage_dir / req_rel).exists(), f"Missing staged repository path: {req_rel}"

    readme_content = (stage_dir / "README.md").read_text(encoding="utf-8")
    for obsolete_token in ("BENCHMARKS.md", "1vgNIo8O2MpH9fU", "EwzHFVonjWyHslWpPf6Av", "34.32 GB", "\\text{", "\\Delta", "\\mathcal{", "$", "v1.5.3–v1.5.5"):
        assert obsolete_token not in readme_content, (
            f"Obsolete benchmark reference found in staged README.md: {obsolete_token}"
        )

    for md_name in ("README.md", "CHANGELOG.md"):
        md_text = (stage_dir / md_name).read_text(encoding="utf-8")
        for m in MARKDOWN_LINK_RE.finditer(md_text):
            target = m.group(2)
            assert target.startswith(("http://", "https://", "mailto:", "#")), (
                f"Unresolved relative link in {md_name}: [{m.group(1)}]({target})"
            )

    lum_path = stage_dir / "luminav.py"
    spec = importlib.util.spec_from_file_location("_hf_standalone_luminav", lum_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert getattr(mod, "__version__", None) == "1.5.5", f"Unexpected version: {getattr(mod, '__version__', None)}"

    import torch
    for buf in (2, 1):
        p = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float32, requires_grad=True)
        p.grad = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float32)
        opt = mod.LuminaV([p], lr=1e-3, buffer=buf)
        opt.step()
        assert opt.state[p]["step"] == 1
    shutil.rmtree(stage_dir / "__pycache__", ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage and sync full LuminaMoon repository and LuminaV v1.5.5 artifacts to Hugging Face Hub.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stage-root", type=Path, default=Path(__file__).resolve().parents[1] / "dist" / "hf_stage")
    parser.add_argument("--push", action="store_true", help="Upload staged directories to Hugging Face Hub.")
    parser.add_argument(
        "--commit-message",
        type=str,
        default="sync(v1.5.5 / v1.0.8.post1): synchronize full luminamoon monorepo, LSO-Eval v1.1.0 results, and LuminaV v1.5.5 bundle",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    stage_root = args.stage_root.resolve()
    hf_org = os.environ.get("HF_ORG", "cloverx-id").strip() or "cloverx-id"
    target_repos = ["LuminaV-Optimizer", "LuminaV-Optimizer-Paper"]

    for repo_name in target_repos:
        r_stage = stage_root / repo_name
        copied_count = stage_hf_repo(repo_root, r_stage, repo_name)
        verify_staged_luminav(r_stage)
        print(f"Staged and verified full repository ({copied_count} tracked files + standalone bundle) for {hf_org}/{repo_name} at {r_stage}")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN", "").strip() or None)
        for repo_name in target_repos:
            full_repo_id = f"{hf_org}/{repo_name}"
            r_stage = stage_root / repo_name
            remote_files = set(api.list_repo_files(repo_id=full_repo_id, repo_type="model"))
            staged_files = {
                p.relative_to(r_stage).as_posix()
                for p in r_stage.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}
            }
            stale_remote = sorted(remote_files - staged_files - {".gitattributes"})
            commit_info = api.upload_folder(
                repo_id=full_repo_id,
                repo_type="model",
                folder_path=str(r_stage),
                commit_message=args.commit_message,
                ignore_patterns=["__pycache__/**", "*.pyc", "*.pyo"],
                delete_patterns=stale_remote if stale_remote else None,
            )
            print(f"Synced {full_repo_id} (pruned {len(stale_remote)} stale files): {commit_info.commit_url}")


if __name__ == "__main__":
    main()
