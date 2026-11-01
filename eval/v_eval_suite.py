from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

_base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_root = _base.parent if _base.name == "eval" else _base
for _p in [str(_root), str(_base), str(_root / "python"), str(_root / "optimizer" / "reference")]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.append(_p)

try:
    from eval.modules.v_eval import (
        SimpleLion,
        TRACK_VA_TASKS,
        TRACK_VB_TASKS,
        VEVAL_CONTENDER_NAMES,
        VEvalRunMetrics,
        VEvalTaskSpec,
        build_veval_optimizer,
        compute_cosine_lr_with_warmup,
        compute_matrix_update_ratios,
        evaluate_veval_pareto_verdict,
        load_official_veval_model,
        split_vision_parameter_groups,
    )
except ImportError:
    from modules.v_eval import (
        SimpleLion,
        TRACK_VA_TASKS,
        TRACK_VB_TASKS,
        VEVAL_CONTENDER_NAMES,
        VEvalRunMetrics,
        VEvalTaskSpec,
        build_veval_optimizer,
        compute_cosine_lr_with_warmup,
        compute_matrix_update_ratios,
        evaluate_veval_pareto_verdict,
        load_official_veval_model,
        split_vision_parameter_groups,
    )


def _build_ci_vision_probe() -> nn.Module:
    """
    Minimal standard PyTorch vision probe containing standard nn.Conv2d, nn.BatchNorm2d,
    nn.GroupNorm, nn.LayerNorm, and nn.Linear layers to verify V-Eval parameter grouping
    and all 6 symmetric optimizer contenders in CI without external model downloads.
    """
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16, bias=True),
        nn.GroupNorm(4, 16),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.LayerNorm(16),
        nn.Linear(16, 100, bias=True),
    )


def verify_veval_architecture(seed: int = 42) -> bool:
    """
    Executes a deterministic CI verification of the LSO-Eval Vision (V-Eval) protocol:
    1. Track V-A & Track V-B specification & epoch-range validation.
    2. Decoupled Weight Decay Vision Parameter Grouping (split_vision_parameter_groups).
    3. 6-Contender Symmetric Matrix execution in native torch.bfloat16 (including SimpleLion
       and FP32MasterOptimizerWrapper).
    4. Two-Tiered Pareto Acceptance criteria (Class I & Class II).
    """
    torch.manual_seed(seed)
    print("LSO-Eval Vision (V-Eval) Protocol & 6-Contender CI Verification")
    print("-" * 76)

    all_tasks: Dict[str, VEvalTaskSpec] = {**TRACK_VA_TASKS, **TRACK_VB_TASKS}
    for code, spec in all_tasks.items():
        spec.validate_epochs(spec.canonical_epochs)
        lo, hi = spec.epoch_range
        spec.validate_epochs(lo)
        spec.validate_epochs(hi)
        warmup_lo = spec.scaled_warmup_epochs(lo)
        assert warmup_lo >= 1, f"Warmup epochs must be >= 1 for {code}"
        print(
            f"[{spec.track}] {code:<5} | {spec.backbone:<40} | "
            f"Epochs: {lo}-{hi} (Canonical: {spec.canonical_epochs}, Warmup: {spec.warmup_epochs})"
        )

    probe = _build_ci_vision_probe().to(dtype=torch.bfloat16)
    groups = split_vision_parameter_groups(probe, weight_decay=0.08)
    assert len(groups) == 2, "Expected 2 parameter groups (decayed and no-decay)."
    decayed_group, no_decay_group = groups[0], groups[1]
    assert decayed_group["weight_decay"] == 0.08
    assert no_decay_group["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decayed_group["params"])
    assert all(p.ndim <= 1 for p in no_decay_group["params"])

    x_batch = torch.randn(4, 3, 16, 16, dtype=torch.bfloat16)
    y_batch = torch.tensor([1, 5, 12, 99], dtype=torch.long)

    print("-" * 76)
    for contender in VEVAL_CONTENDER_NAMES:
        torch.manual_seed(seed)
        model = _build_ci_vision_probe().to(dtype=torch.bfloat16)
        lr = TRACK_VA_TASKS["V-A1"].learning_rates[contender][0]
        opt = build_veval_optimizer(contender, model, lr=lr, track="V-A")

        prev_weights = {
            name: p.detach().clone() for name, p in model.named_parameters() if p.ndim >= 2
        }
        loss_val = 0.0
        for step_idx in range(1, 4):
            cur_lr = compute_cosine_lr_with_warmup(
                step=step_idx, total_steps=10, warmup_steps=2, base_lr=lr
            )
            for grp in opt.param_groups:
                grp["lr"] = cur_lr
            logits = model(x_batch)
            loss = F.cross_entropy(logits.float(), y_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_val = float(loss.item())

        r_min, r_max = compute_matrix_update_ratios(prev_weights, model)
        assert math.isfinite(loss_val), f"Non-finite loss for {contender}"
        assert r_max > 0.0, f"Zero update ratio for {contender}"
        print(
            f"Contender: {contender:<24} | Final Loss: {loss_val:.4f} | "
            f"R_{{t,l}} range: [{r_min:.3e}, {r_max:.3e}] | PASS"
        )

    baseline_bf16 = VEvalRunMetrics(
        contender="AdamW_Native_BF16",
        task_code="V-A1",
        passed_f1_f2=True,
        vram_opt_gb=0.10,
        opt_step_ms=4.20,
        val_loss=0.95,
        top1_acc=76.50,
    )
    ref_fp32 = VEvalRunMetrics(
        contender="AdamW_FP32_Master",
        task_code="V-A1",
        passed_f1_f2=True,
        vram_opt_gb=0.30,
        opt_step_ms=9.80,
        val_loss=0.91,
        top1_acc=77.40,
    )
    cand_c1 = VEvalRunMetrics(
        contender="LuminaV_Buffer2_Dual",
        task_code="V-A1",
        passed_f1_f2=True,
        vram_opt_gb=0.10,
        opt_step_ms=4.10,
        val_loss=0.89,
        top1_acc=77.85,
    )
    cand_c2 = VEvalRunMetrics(
        contender="LuminaV_Buffer1_TDCLM",
        task_code="V-A1",
        passed_f1_f2=True,
        vram_opt_gb=0.05,
        opt_step_ms=3.40,
        val_loss=0.98,
        top1_acc=76.10,
    )
    verdict_c1 = evaluate_veval_pareto_verdict(cand_c1, baseline_bf16, ref_fp32)
    verdict_c2 = evaluate_veval_pareto_verdict(cand_c2, baseline_bf16, ref_fp32)
    assert verdict_c1.passed and verdict_c2.passed
    print("-" * 76)
    print(f"Pareto Gate Check -> {verdict_c1.contender}: {verdict_c1.summary}")
    print(
        f"Pareto Gate Check -> {verdict_c2.contender}: {verdict_c2.summary} "
        f"(VRAM vs BF16: {verdict_c2.delta_vram_opt_pct_vs_bf16:.1f}%, "
        f"vs FP32 Master: {verdict_c2.delta_vram_opt_pct_vs_fp32:.1f}%)"
    )
    print("LSO-Eval Vision (V-Eval) CI Verification : PASSED")
    return True


def test_veval_protocol_and_contenders() -> None:
    assert verify_veval_architecture() is True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LSO-Eval v1.1.0 Vision & Spatial Modality Qualification Suite (V-Eval)"
    )
    parser.add_argument(
        "--track",
        choices=("va", "vb", "all"),
        default="all",
        help="Select V-Eval qualification track (va: From-Scratch, vb: Downstream Fine-Tuning, all: Both).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Optional specific task code (e.g., V-A1, V-A2, V-A3, V-A4, V-A5, V-B1, V-B2).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch count override within the task's permitted epoch range.",
    )
    parser.add_argument(
        "--verify-architecture",
        action="store_true",
        help="Run deterministic CI verification of all 6 contenders, parameter grouping, and Pareto rules.",
    )
    args = parser.parse_args()

    if args.task is not None and args.epochs is not None:
        all_tasks = {**TRACK_VA_TASKS, **TRACK_VB_TASKS}
        task_key = args.task.upper()
        if task_key not in all_tasks:
            raise SystemExit(f"Unknown V-Eval task code: {args.task}")
        all_tasks[task_key].validate_epochs(args.epochs)

    if not verify_veval_architecture():
        sys.exit(1)


if __name__ == "__main__":
    main()
