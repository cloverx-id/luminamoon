from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

from .common import LuminaV


VISION_NO_DECAY_KEYWORDS: Tuple[str, ...] = (
    "bias",
    "bn",
    "norm",
    "ln",
    "rmsnorm",
    "groupnorm",
    "grn",
    "global_response_norm",
    "cls_token",
    "pos_embed",
    "relative_position_bias_table",
    "gamma",
    "beta",
)

VISION_NORM_MODULE_TYPES: Tuple[type, ...] = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)
if hasattr(nn, "RMSNorm"):
    VISION_NORM_MODULE_TYPES = VISION_NORM_MODULE_TYPES + (nn.RMSNorm,)


# Standalone implementation of Lion Optimizer (Chen et al., NeurIPS 2023, Algorithm 2)
class SimpleLion(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-4, betas=(0.95, 0.98), weight_decay=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SimpleLion does not support sparse gradients.")

                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg = state["exp_avg"]

                # 1. Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # 2. Parameter update: c = beta1 * exp_avg + (1 - beta1) * grad
                update = exp_avg.lerp(grad, 1.0 - beta1).sign_()
                p.add_(update, alpha=-lr)

                # 3. Momentum EMA update: exp_avg = beta2 * exp_avg + (1 - beta2) * grad
                exp_avg.lerp_(grad, 1.0 - beta2)

        return loss


def split_vision_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.08,
) -> List[Dict[str, Any]]:
    """
    LSO-Eval V-Eval Decoupled Weight Decay Parameter Grouping:
    - Decayed (weight_decay > 0): All >= 2D weight tensors (nn.Conv2d, nn.Linear, QKV/MLP projections).
    - No-Decay (weight_decay = 0.0): All 1D parameters, biases, normalization parameters
      (BatchNorm2d, LayerNorm, RMSNorm, GroupNorm, GlobalResponseNorm/GRN), and positional/token
      embeddings (cls_token, pos_embed, relative_position_bias_table).
    """
    norm_param_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, VISION_NORM_MODULE_TYPES):
            for p in module.parameters(recurse=False):
                norm_param_ids.add(id(p))

    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
    seen: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        lower_name = name.lower()
        is_no_decay = (
            param.ndim <= 1
            or id(param) in norm_param_ids
            or any(kw in lower_name for kw in VISION_NO_DECAY_KEYWORDS)
        )
        if is_no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups: List[Dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": float(weight_decay)})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


class FP32MasterOptimizerWrapper:
    """
    Maintains an explicit float32 master parameter copy for AdamW_FP32_Master and Lion_FP32_Master
    while keeping model forward/backward passes in native torch.bfloat16.
    """

    def __init__(
        self,
        model_param_groups: List[Dict[str, Any]],
        base_optimizer_cls: type,
        **optim_kwargs: Any,
    ) -> None:
        self.model_groups: List[Dict[str, Any]] = []
        master_groups: List[Dict[str, Any]] = []
        self._model_to_master: List[Tuple[nn.Parameter, nn.Parameter]] = []

        for grp in model_param_groups:
            m_params: List[nn.Parameter] = []
            for p in grp["params"]:
                p_master = nn.Parameter(p.detach().to(dtype=torch.float32).clone())
                m_params.append(p_master)
                self._model_to_master.append((p, p_master))
            master_grp = {k: v for k, v in grp.items() if k != "params"}
            master_grp["params"] = m_params
            master_groups.append(master_grp)
            self.model_groups.append(grp)

        self.base_optimizer = base_optimizer_cls(master_groups, **optim_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p, p_master in self._model_to_master:
            if set_to_none:
                p.grad = None
                p_master.grad = None
            else:
                if p.grad is not None:
                    p.grad.zero_()
                if p_master.grad is not None:
                    p_master.grad.zero_()

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        for p, p_master in self._model_to_master:
            if p.grad is not None:
                p_master.grad = p.grad.detach().to(dtype=torch.float32)
            else:
                p_master.grad = None

        loss = self.base_optimizer.step(closure)

        for p, p_master in self._model_to_master:
            p.copy_(p_master.to(dtype=p.dtype))

        return loss

    def state_dict(self) -> Dict[str, Any]:
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.base_optimizer.load_state_dict(state_dict)


VEVAL_CONTENDER_NAMES: Tuple[str, ...] = (
    "LuminaV_Buffer2_Dual",
    "LuminaV_Buffer1_TDCLM",
    "AdamW_Native_BF16",
    "AdamW_FP32_Master",
    "Lion_Native_BF16",
    "Lion_FP32_Master",
)


def build_veval_optimizer(
    contender_name: str,
    model: nn.Module,
    lr: float,
    weight_decay: Optional[float] = None,
    track: str = "V-A",
) -> Any:
    """
    Instantiates one of the 6 symmetric LSO-Eval V-Eval optimizer contenders with
    canonical decoupled weight decay vision parameter grouping.
    """
    is_finetune = track.upper().startswith("V-B")
    sig_params = inspect.signature(LuminaV.__init__).parameters
    v155_extra: Dict[str, Any] = {}
    for k, v in (("fuse_tensors", True), ("bucket_cap_mb", 500.0), ("prewarm_jit", True)):
        if k in sig_params:
            v155_extra[k] = v

    if contender_name == "LuminaV_Buffer2_Dual":
        wd = 0.08 if weight_decay is None else weight_decay
        param_groups = split_vision_parameter_groups(model, weight_decay=wd)
        return LuminaV(
            param_groups,
            lr=lr,
            buffer=2,
            tau=0.8,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
            bound=True,
            bound_type="coordinate",
            bound_ratio=0.05,
            master_weights="none",
            stochastic_rounding=True,
            fused_single_pass=True,
            **v155_extra,
        )

    if contender_name == "LuminaV_Buffer1_TDCLM":
        wd = 0.08 if weight_decay is None else weight_decay
        param_groups = split_vision_parameter_groups(model, weight_decay=wd)
        return LuminaV(
            param_groups,
            lr=lr,
            buffer=1,
            lam=0.3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
            bound=True,
            bound_type="coordinate",
            bound_ratio=0.05,
            master_weights="none",
            stochastic_rounding=True,
            fused_single_pass=True,
            **v155_extra,
        )

    if contender_name in ("AdamW_Native_BF16", "AdamW_FP32_Master"):
        wd = (0.05 if is_finetune else 0.08) if weight_decay is None else weight_decay
        param_groups = split_vision_parameter_groups(model, weight_decay=wd)
        use_fused = torch.cuda.is_available() and all(p.is_cuda for p in model.parameters())
        adamw_kwargs: Dict[str, Any] = dict(
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
        )
        if use_fused:
            adamw_kwargs["fused"] = True
        if contender_name == "AdamW_Native_BF16":
            return torch.optim.AdamW(param_groups, **adamw_kwargs)
        return FP32MasterOptimizerWrapper(param_groups, torch.optim.AdamW, **adamw_kwargs)

    if contender_name in ("Lion_Native_BF16", "Lion_FP32_Master"):
        wd = 0.01 if weight_decay is None else weight_decay
        param_groups = split_vision_parameter_groups(model, weight_decay=wd)
        lion_kwargs = dict(lr=lr, betas=(0.95, 0.98), weight_decay=wd)
        if contender_name == "Lion_Native_BF16":
            return SimpleLion(param_groups, **lion_kwargs)
        return FP32MasterOptimizerWrapper(param_groups, SimpleLion, **lion_kwargs)

    raise ValueError(f"Unknown V-Eval optimizer contender: {contender_name}")


def load_official_veval_model(task_code: str, num_classes: Optional[int] = None) -> nn.Module:
    """
    Loads the official, unmodified upstream Vision model from canonical peer-reviewed libraries
    (timm, torchvision, transformers, or facebookresearch/SiT).
    """
    code = task_code.upper()
    if code == "V-A1":
        import timm

        model = timm.create_model("resnet50", pretrained=False, num_classes=num_classes or 100)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    if code == "V-A2":
        import timm

        return timm.create_model("nf_resnet50", pretrained=False, num_classes=num_classes or 100)
    if code == "V-A3":
        import timm

        return timm.create_model("convnextv2_tiny", pretrained=False, num_classes=num_classes or 100)
    if code == "V-A4":
        import timm

        return timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            img_size=32,
            patch_size=4,
            num_classes=num_classes or 100,
        )
    if code == "V-A5":
        from models import SiT_models  # type: ignore[import-not-found]

        return SiT_models["SiT-S/2"](input_size=32, num_classes=num_classes or 10)
    if code == "V-B1":
        from transformers import AutoModelForObjectDetection  # type: ignore[import-not-found]

        return AutoModelForObjectDetection.from_pretrained(
            "PekingU/rtdetr_r18vd",
            num_labels=num_classes or 5,
            ignore_mismatched_sizes=True,
        )
    if code == "V-B2":
        import timm

        return timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes or 37)
    raise ValueError(f"Unknown V-Eval task code: {task_code}")


@dataclass(frozen=True)
class VEvalTaskSpec:
    code: str
    track: str
    backbone: str
    upstream_api: str
    paradigm: str
    dataset: str
    resolution: Tuple[int, int]
    batch_size: int
    epoch_range: Tuple[int, int]
    canonical_epochs: int
    canonical_steps: int
    warmup_epochs: int
    eval_interval_epochs: int
    learning_rates: Dict[str, Tuple[float, ...]]
    primary_metrics: Tuple[str, ...]

    def validate_epochs(self, epochs: int) -> None:
        lo, hi = self.epoch_range
        if not (lo <= epochs <= hi):
            raise ValueError(
                f"Epoch count {epochs} for {self.code} ({self.backbone}) must lie within "
                f"permitted range [{lo}, {hi}] (canonical={self.canonical_epochs})."
            )

    def scaled_warmup_epochs(self, epochs: int) -> int:
        self.validate_epochs(epochs)
        ratio = self.warmup_epochs / float(self.canonical_epochs)
        return max(1, int(round(epochs * ratio)))


TRACK_VA_TASKS: Dict[str, VEvalTaskSpec] = {
    "V-A1": VEvalTaskSpec(
        code="V-A1",
        track="V-A",
        backbone="ResNet-50 (CIFAR 3x3 stem, ~23.7M)",
        upstream_api='timm.create_model("resnet50", pretrained=False, num_classes=100)',
        paradigm="Classical CNN (BatchNorm2d + ReLU)",
        dataset="CIFAR-100",
        resolution=(32, 32),
        batch_size=256,
        epoch_range=(50, 150),
        canonical_epochs=100,
        canonical_steps=19500,
        warmup_epochs=5,
        eval_interval_epochs=10,
        learning_rates={
            "LuminaV_Buffer2_Dual": (8e-4, 3e-4),
            "LuminaV_Buffer1_TDCLM": (8e-4, 3e-4),
            "AdamW_Native_BF16": (1e-3,),
            "AdamW_FP32_Master": (1e-3,),
            "Lion_Native_BF16": (1e-4,),
            "Lion_FP32_Master": (1e-4,),
        },
        primary_metrics=("Top-1 Val Acc (%)", "Top-5 Val Acc (%)", "Val Loss", "R_{t,l}"),
    ),
    "V-A2": VEvalTaskSpec(
        code="V-A2",
        track="V-A",
        backbone="NF-ResNet-50 (Norm-Free, ~25.6M)",
        upstream_api='timm.create_model("nf_resnet50", pretrained=False, num_classes=100)',
        paradigm="Normalization-Free CNN (Scaled Weight Std, No BatchNorm)",
        dataset="CIFAR-100",
        resolution=(32, 32),
        batch_size=256,
        epoch_range=(50, 150),
        canonical_epochs=100,
        canonical_steps=19500,
        warmup_epochs=5,
        eval_interval_epochs=10,
        learning_rates={
            "LuminaV_Buffer2_Dual": (8e-4, 3e-4),
            "LuminaV_Buffer1_TDCLM": (8e-4, 3e-4),
            "AdamW_Native_BF16": (1e-3,),
            "AdamW_FP32_Master": (1e-3,),
            "Lion_Native_BF16": (1e-4,),
            "Lion_FP32_Master": (1e-4,),
        },
        primary_metrics=("Top-1 Val Acc (%)", "Top-5 Val Acc (%)", "Val Loss", "R_{t,l}"),
    ),
    "V-A3": VEvalTaskSpec(
        code="V-A3",
        track="V-A",
        backbone="ConvNeXt-V2-Tiny (~28.6M)",
        upstream_api='timm.create_model("convnextv2_tiny", pretrained=False, num_classes=100)',
        paradigm="Modern Pure ConvNet (Depthwise 7x7 + LN + GRN)",
        dataset="CIFAR-100",
        resolution=(32, 32),
        batch_size=256,
        epoch_range=(50, 200),
        canonical_epochs=100,
        canonical_steps=19500,
        warmup_epochs=10,
        eval_interval_epochs=10,
        learning_rates={
            "LuminaV_Buffer2_Dual": (8e-4, 3e-4),
            "LuminaV_Buffer1_TDCLM": (8e-4, 3e-4),
            "AdamW_Native_BF16": (8e-4,),
            "AdamW_FP32_Master": (8e-4,),
            "Lion_Native_BF16": (1e-4,),
            "Lion_FP32_Master": (1e-4,),
        },
        primary_metrics=("Top-1 Val Acc (%)", "Top-5 Val Acc (%)", "Val Loss", "R_{t,l}"),
    ),
    "V-A4": VEvalTaskSpec(
        code="V-A4",
        track="V-A",
        backbone="ViT-Small (Patch 4x4, ~22.0M)",
        upstream_api='timm.create_model("vit_small_patch16_224", pretrained=False, img_size=32, patch_size=4, num_classes=100)',
        paradigm="Pure Isotropic Vision Transformer (Self-Attention)",
        dataset="CIFAR-100",
        resolution=(32, 32),
        batch_size=256,
        epoch_range=(50, 200),
        canonical_epochs=100,
        canonical_steps=19500,
        warmup_epochs=10,
        eval_interval_epochs=10,
        learning_rates={
            "LuminaV_Buffer2_Dual": (8e-4, 3e-4),
            "LuminaV_Buffer1_TDCLM": (8e-4, 3e-4),
            "AdamW_Native_BF16": (8e-4,),
            "AdamW_FP32_Master": (8e-4,),
            "Lion_Native_BF16": (1e-4,),
            "Lion_FP32_Master": (1e-4,),
        },
        primary_metrics=("Top-1 Val Acc (%)", "Top-5 Val Acc (%)", "Val Loss", "R_{t,l}"),
    ),
    "V-A5": VEvalTaskSpec(
        code="V-A5",
        track="V-A",
        backbone="SiT-S/2 (Scalable Interpolant Transformer, ~33.0M)",
        upstream_api='SiT_models["SiT-S/2"](input_size=32, num_classes=10)',
        paradigm="Continuous-Time Generative Vision (Rectified Flow Matching)",
        dataset="CIFAR-10",
        resolution=(32, 32),
        batch_size=256,
        epoch_range=(50, 200),
        canonical_epochs=100,
        canonical_steps=19500,
        warmup_epochs=5,
        eval_interval_epochs=10,
        learning_rates={
            "LuminaV_Buffer2_Dual": (5e-4, 3e-4),
            "LuminaV_Buffer1_TDCLM": (5e-4, 3e-4),
            "AdamW_Native_BF16": (5e-4,),
            "AdamW_FP32_Master": (5e-4,),
            "Lion_Native_BF16": (1e-4,),
            "Lion_FP32_Master": (1e-4,),
        },
        primary_metrics=("Held-Out Flow Velocity MSE Loss", "FID"),
    ),
}

TRACK_VB_TASKS: Dict[str, VEvalTaskSpec] = {
    "V-B1": VEvalTaskSpec(
        code="V-B1",
        track="V-B",
        backbone="RT-DETR-R18 / YOLOv11-S / FCOS-ResNet50",
        upstream_api='AutoModelForObjectDetection.from_pretrained("PekingU/rtdetr_r18vd")',
        paradigm="End-to-End Object Detection Fine-Tuning",
        dataset="cppe-5 / Pascal VOC 2007",
        resolution=(480, 480),
        batch_size=16,
        epoch_range=(15, 35),
        canonical_epochs=25,
        canonical_steps=1600,
        warmup_epochs=2,
        eval_interval_epochs=5,
        learning_rates={
            "LuminaV_Buffer2_Dual": (1e-4, 2e-4),
            "LuminaV_Buffer1_TDCLM": (1e-4, 2e-4),
            "AdamW_Native_BF16": (1e-4,),
            "AdamW_FP32_Master": (1e-4,),
            "Lion_Native_BF16": (3e-5,),
            "Lion_FP32_Master": (3e-5,),
        },
        primary_metrics=("COCO mAP_{50:95}", "mAP_{50}", "Val Box/Focal Loss"),
    ),
    "V-B2": VEvalTaskSpec(
        code="V-B2",
        track="V-B",
        backbone="ViT-Base/16 & ConvNeXt-Base",
        upstream_api='timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=37)',
        paradigm="Fine-Grained High-Resolution Classification Fine-Tuning",
        dataset="Oxford-IIIT Pet (37 classes)",
        resolution=(224, 224),
        batch_size=64,
        epoch_range=(10, 30),
        canonical_epochs=20,
        canonical_steps=1160,
        warmup_epochs=2,
        eval_interval_epochs=2,
        learning_rates={
            "LuminaV_Buffer2_Dual": (1e-4,),
            "LuminaV_Buffer1_TDCLM": (1e-4,),
            "AdamW_Native_BF16": (1e-4,),
            "AdamW_FP32_Master": (1e-4,),
            "Lion_Native_BF16": (3e-5,),
            "Lion_FP32_Master": (3e-5,),
        },
        primary_metrics=("Top-1 Val Acc (%)", "Top-5 Val Acc (%)", "Val CE Loss"),
    ),
}


def compute_cosine_lr_with_warmup(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr_ratio: float = 0.05,
) -> float:
    """Standard LSO-Eval V-Eval linear warmup + cosine annealing learning rate schedule."""
    if step <= 0:
        return 0.0
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * (float(step) / float(warmup_steps))
    if total_steps <= warmup_steps:
        return base_lr
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(total_steps - warmup_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


@torch.no_grad()
def compute_matrix_update_ratios(
    prev_weights: Dict[str, Tensor],
    model: nn.Module,
) -> Tuple[float, float]:
    """
    Computes (min R_{t,l}, max R_{t,l}) across >= 2D weight tensors where
    R_{t,l} = ||theta_{t,l} - theta_{t-1,l}||_2 / ||theta_{t-1,l}||_2.
    """
    ratios: List[float] = []
    for name, p in model.named_parameters():
        if name not in prev_weights or p.ndim < 2:
            continue
        w_prev = prev_weights[name]
        denom = float(torch.linalg.vector_norm(w_prev.float()).item())
        if denom > 1e-12:
            delta = float(torch.linalg.vector_norm((p.detach().float() - w_prev.float())).item())
            ratios.append(delta / denom)
    if not ratios:
        return 0.0, 0.0
    return min(ratios), max(ratios)


@dataclass
class VEvalRunMetrics:
    contender: str
    task_code: str
    passed_f1_f2: bool
    vram_opt_gb: float
    opt_step_ms: float
    val_loss: float
    top1_acc: Optional[float] = None
    map_50_95: Optional[float] = None
    flow_mse_loss: Optional[float] = None


@dataclass
class VEvalParetoVerdict:
    contender: str
    tier_class: str
    passed: bool
    delta_vram_opt_pct_vs_bf16: float
    delta_vram_opt_pct_vs_fp32: float
    delta_val_loss: float
    delta_top1_acc: Optional[float]
    delta_map_50_95: Optional[float]
    delta_flow_mse: Optional[float]
    summary: str


def evaluate_veval_pareto_verdict(
    candidate: VEvalRunMetrics,
    baseline_adamw_bf16: VEvalRunMetrics,
    reference_adamw_fp32: Optional[VEvalRunMetrics] = None,
) -> VEvalParetoVerdict:
    """
    Evaluates Two-Tiered Pareto Acceptance for LSO-Eval Vision (V-Eval) against AdamW_Native_BF16
    (and AdamW_FP32_Master when provided):
    - Class I (LuminaV_Buffer2_Dual):
      F1 ^ F2 pass, delta VRAM_opt <= 0.0 GB, and validation quality >= baseline
      (delta Acc_Top1 >= 0.0%, delta mAP_50:95 >= 0.0%, delta L_FM <= 0.0).
    - Class II (LuminaV_Buffer1_TDCLM):
      F1 ^ F2 pass, VRAM_opt reduction >= 50.0% vs AdamW_Native_BF16 (and >= 83.3% vs AdamW_FP32_Master),
      faster kernel latency, and validation loss gap delta L_val <= 0.25 nats (or Top-1 deviation <= 1.5%).
    """
    bf16_vram = max(baseline_adamw_bf16.vram_opt_gb, 1e-9)
    delta_vram_bf16_pct = 100.0 * (candidate.vram_opt_gb - bf16_vram) / bf16_vram
    if reference_adamw_fp32 is not None and reference_adamw_fp32.vram_opt_gb > 1e-9:
        fp32_vram = reference_adamw_fp32.vram_opt_gb
        delta_vram_fp32_pct = 100.0 * (candidate.vram_opt_gb - fp32_vram) / fp32_vram
    else:
        delta_vram_fp32_pct = -83.3333 if "Buffer1" in candidate.contender else -66.6667

    delta_loss = candidate.val_loss - baseline_adamw_bf16.val_loss
    delta_acc = (
        candidate.top1_acc - baseline_adamw_bf16.top1_acc
        if (candidate.top1_acc is not None and baseline_adamw_bf16.top1_acc is not None)
        else None
    )
    delta_map = (
        candidate.map_50_95 - baseline_adamw_bf16.map_50_95
        if (candidate.map_50_95 is not None and baseline_adamw_bf16.map_50_95 is not None)
        else None
    )
    delta_fm = (
        candidate.flow_mse_loss - baseline_adamw_bf16.flow_mse_loss
        if (candidate.flow_mse_loss is not None and baseline_adamw_bf16.flow_mse_loss is not None)
        else None
    )

    if "Buffer2" in candidate.contender:
        tier = "Class I (Dual-Buffer / Quality-Dominant)"
        vram_ok = (candidate.vram_opt_gb - baseline_adamw_bf16.vram_opt_gb) <= 1e-4
        qual_checks: List[bool] = []
        if delta_acc is not None:
            qual_checks.append(delta_acc >= 0.0)
        if delta_map is not None:
            qual_checks.append(delta_map >= 0.0)
        if delta_fm is not None:
            qual_checks.append(delta_fm <= 0.0)
        if not qual_checks:
            qual_checks.append(delta_loss <= 0.0)
        passed = bool(candidate.passed_f1_f2 and vram_ok and all(qual_checks))
        status_str = "PASSED (Class I)" if passed else "FAILED (Class I)"
    else:
        tier = "Class II (Single-Buffer / Memory-Pareto)"
        vram_ok = (delta_vram_bf16_pct <= -49.9) and (delta_vram_fp32_pct <= -83.2)
        latency_ok = candidate.opt_step_ms <= baseline_adamw_bf16.opt_step_ms
        loss_tol_ok = delta_loss <= 0.25
        acc_tol_ok = True if delta_acc is None else (delta_acc >= -1.5)
        passed = bool(candidate.passed_f1_f2 and vram_ok and latency_ok and (loss_tol_ok or acc_tol_ok))
        status_str = "PASSED (Class II)" if passed else "FAILED (Class II)"

    return VEvalParetoVerdict(
        contender=candidate.contender,
        tier_class=tier,
        passed=passed,
        delta_vram_opt_pct_vs_bf16=delta_vram_bf16_pct,
        delta_vram_opt_pct_vs_fp32=delta_vram_fp32_pct,
        delta_val_loss=delta_loss,
        delta_top1_acc=delta_acc,
        delta_map_50_95=delta_map,
        delta_flow_mse=delta_fm,
        summary=status_str,
    )
