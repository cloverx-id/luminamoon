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
        build_veval_dataloader,
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
        build_veval_dataloader,
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
    4. Two-Class Acceptance criteria (Class I & Class II, including Lion 1-buffer comparison).
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
        res_str = f"{spec.resolution[0]}x{spec.resolution[1]}"
        print(
            f"[{spec.track}] {code:<5} | {spec.backbone:<38} | {spec.dataset:<38} ({res_str}) | "
            f"Epochs: {lo}-{hi} (Default: {spec.canonical_epochs})"
        )

    probe = _build_ci_vision_probe().to(dtype=torch.bfloat16)
    groups = split_vision_parameter_groups(probe, weight_decay=0.08)
    assert len(groups) == 2, "Expected 2 parameter groups (decayed and no-decay)."
    decayed_group, no_decay_group = groups[0], groups[1]
    assert decayed_group["weight_decay"] == 0.08
    assert no_decay_group["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decayed_group["params"])
    assert all(p.ndim <= 1 for p in no_decay_group["params"])

    class _ModernHybridProbe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = nn.Linear(16, 48, bias=False)
            self.q_norm = nn.RMSNorm(16) if hasattr(nn, "RMSNorm") else nn.LayerNorm(16)
            self.grn_gamma = nn.Parameter(torch.zeros(1, 1, 1, 16))
            self.grn_beta = nn.Parameter(torch.zeros(1, 1, 1, 16))
            self.dyt_alpha = nn.Parameter(torch.tensor(0.5))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 16))
            self.register_tokens = nn.Parameter(torch.zeros(1, 4, 16))
            self.pos_embed = nn.Parameter(torch.zeros(1, 16, 16))
            self.A_log = nn.Parameter(torch.zeros(4, 8))
            self.D = nn.Parameter(torch.ones(4))
            self.dt_bias = nn.Parameter(torch.zeros(4))

    modern_probe = _ModernHybridProbe().to(dtype=torch.bfloat16)
    modern_groups = split_vision_parameter_groups(modern_probe, weight_decay=0.08)
    assert len(modern_groups) == 2
    assert len(modern_groups[0]["params"]) == 1 and modern_groups[0]["params"][0] is modern_probe.qkv.weight
    assert len(modern_groups[1]["params"]) == 10

    sorted_labels = torch.arange(32, dtype=torch.long)
    dummy_ds = torch.utils.data.TensorDataset(sorted_labels)
    loader_ep0_a = build_veval_dataloader(dummy_ds, batch_size=8, epoch=0, seed=seed, is_train=True)
    loader_ep0_b = build_veval_dataloader(dummy_ds, batch_size=8, epoch=0, seed=seed, is_train=True)
    loader_ep1 = build_veval_dataloader(dummy_ds, batch_size=8, epoch=1, seed=seed, is_train=True)
    seq_ep0_a = torch.cat([b[0] for b in loader_ep0_a])
    seq_ep0_b = torch.cat([b[0] for b in loader_ep0_b])
    seq_ep1 = torch.cat([b[0] for b in loader_ep1])
    assert torch.equal(seq_ep0_a, seq_ep0_b), "Identical epoch+seed must yield identical batch order"
    assert not torch.equal(seq_ep0_a, sorted_labels), "Training loader must shuffle sequential blocks"
    assert not torch.equal(seq_ep0_a, seq_ep1), "Subsequent epochs must permute batch order"

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
    baseline_lion = VEvalRunMetrics(
        contender="Lion_Native_BF16",
        task_code="V-A1",
        passed_f1_f2=True,
        vram_opt_gb=0.05,
        opt_step_ms=3.60,
        val_loss=1.01,
        top1_acc=75.60,
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
    verdict_c2 = evaluate_veval_pareto_verdict(cand_c2, baseline_bf16, ref_fp32, baseline_lion)
    assert verdict_c1.passed and verdict_c2.passed
    print("-" * 76)
    print(f"Acceptance Check -> {verdict_c1.contender}: {verdict_c1.summary}")
    print(
        f"Acceptance Check -> {verdict_c2.contender}: {verdict_c2.summary} "
        f"(VRAM vs BF16: {verdict_c2.delta_vram_opt_pct_vs_bf16:.1f}%, "
        f"vs FP32 Master: {verdict_c2.delta_vram_opt_pct_vs_fp32:.1f}%, "
        f"Top-1 vs Lion BF16: +{verdict_c2.delta_top1_acc_vs_lion_bf16:.2f}%)"
    )
    print("LSO-Eval Vision (V-Eval) CI Verification : PASSED")
    return True


def test_veval_protocol_and_contenders() -> None:
    assert verify_veval_architecture() is True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LSO-Eval v1.1.0 Vision Evaluation Suite (V-Eval)"
    )
    parser.add_argument(
        "--track",
        choices=("va", "vb", "all"),
        default="all",
        help="Select V-Eval evaluation track (va: From-Scratch, vb: Downstream Fine-Tuning, all: Both).",
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
        help="Run deterministic CI verification of all 6 contenders, parameter grouping, and acceptance rules.",
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
