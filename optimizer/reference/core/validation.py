from __future__ import annotations

import logging
from typing import Dict, Tuple, Union

import torch
from torch.optim import Optimizer

logger = logging.getLogger("LuminaV")


class LuminaVValidationMixin(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 8e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 8e-2,
        tau: float = 0.8,
        lam: float = 0.3,
        cautious: bool = True,
        cautious_clamp_min: float = 0.5,
        buffer: int = 2,
        stochastic_rounding: bool = True,
        bound: bool = True,
        bound_type: str = "coordinate",
        bound_ratio: float = 0.05,
        master_weights: Union[bool, str] = False,
        execution: str = "auto",
        fused_single_pass: bool = True,
        ema_decay: Union[float, str] = "auto",
        fsdp_sync: bool = False,
        seed: int = 1337,
    ):
        self.seed = seed
        self._warned_non_contiguous = False
        self._warned_triton_failure = False
        self._generators: Dict[Tuple[str, int, int], torch.Generator] = {}

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            tau=tau, lam=lam, cautious=cautious,
            cautious_clamp_min=cautious_clamp_min, buffer=buffer,
            stochastic_rounding=stochastic_rounding,
            bound=bound, bound_type=bound_type, bound_ratio=bound_ratio,
            master_weights=master_weights, execution=execution,
            fused_single_pass=fused_single_pass, ema_decay=ema_decay,
            fsdp_sync=fsdp_sync,
            seed=seed,
        )
        is_auto_ema = isinstance(ema_decay, str) and ema_decay.lower().strip() == "auto"
        self._validate_and_normalize_group(defaults)
        if is_auto_ema:
            defaults["ema_decay"] = "auto"
        super().__init__(params, defaults)

    @staticmethod
    def _validate_and_normalize_group(group: dict) -> None:
        lr = group["lr"]
        betas = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        tau = group["tau"]
        lam = group.get("lam", 0.3)
        cautious_clamp_min = group["cautious_clamp_min"]
        buffer = group["buffer"]
        bound_type = group["bound_type"]
        bound_ratio = group["bound_ratio"]
        seed = group["seed"]
        execution = group["execution"]
        master_weights = group["master_weights"]
        fsdp_sync = group.get("fsdp_sync", False)
        ema_decay = group.get("ema_decay", "auto")

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError(f"Invalid betas parameter: {betas}. Must be a tuple or list of two floats")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if tau <= 0.0:
            raise ValueError(f"Invalid tau parameter: {tau}")
        if not 0.0 < cautious_clamp_min <= 1.0:
            raise ValueError(f"Invalid cautious_clamp_min: {cautious_clamp_min}")
        if isinstance(buffer, bool) or buffer not in (1, 2):
            raise ValueError(f"Buffer count must be 1 (Single-Buffer TD-CLM) or 2 (Dual-Buffer CLM), got: {buffer}")
        if bound_type not in ("radial", "coordinate"):
            raise ValueError(f"Invalid bound_type: {bound_type}. Must be 'radial' or 'coordinate'")
        if bound_ratio <= 0.0:
            raise ValueError(f"Invalid bound_ratio: {bound_ratio}. Must be > 0.0")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"Invalid seed parameter: {seed}. Must be a non-negative integer")
        if execution not in ("auto", "triton", "single", "foreach"):
            raise ValueError(f"Invalid execution mode: {execution}. Must be 'auto', 'triton', or 'single'")
        if lam < 0.0:
            raise ValueError(f"Invalid lam (triangle defect ratio) parameter: {lam}. Must be >= 0.0")

        if execution == "foreach":
            logger.info(
                "LuminaV: execution='foreach' has been deprecated in favor of high-efficiency "
                "'triton' and memory-lean 'single'. Automatically setting execution to 'auto'."
            )
            group["execution"] = "auto"

        if isinstance(master_weights, bool):
            group["master_weights"] = "full" if master_weights else "none"
        elif isinstance(master_weights, str):
            m_str = master_weights.lower().strip()
            if m_str in ("false", "none", "off"):
                group["master_weights"] = "none"
            elif m_str in ("semi", "hybrid", "half"):
                group["master_weights"] = "semi"
            elif m_str in ("full", "true", "fp32", "masterweight"):
                group["master_weights"] = "full"
            else:
                raise ValueError(f"Invalid master_weights option: {master_weights}. Must be False, 'semi', or 'full'")
        else:
            raise TypeError(f"master_weights must be bool or str, got: {type(master_weights).__name__}")

        if fsdp_sync and not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            logger.warning(
                "LuminaV: fsdp_sync=True was specified, but torch.distributed is not initialized. "
                "Disabling fsdp_sync to prevent collective communication errors."
            )
            group["fsdp_sync"] = False

        if isinstance(ema_decay, str):
            if ema_decay.lower().strip() == "auto":
                group["ema_decay"] = 0.95 if buffer == 1 else 0.8
            else:
                raise ValueError(f"Invalid ema_decay string: {ema_decay}. Must be 'auto' or a float in (0.0, 1.0)")
        elif isinstance(ema_decay, (int, float)) and not isinstance(ema_decay, bool):
            if not 0.0 < float(ema_decay) < 1.0:
                raise ValueError(f"Invalid ema_decay: {ema_decay}. Must be in (0.0, 1.0)")
            group["ema_decay"] = float(ema_decay)
        else:
            raise TypeError(f"ema_decay must be float or 'auto', got {type(ema_decay).__name__}")

    def add_param_group(self, param_group: dict) -> None:
        for k, v in self.defaults.items():
            param_group.setdefault(k, v)
        self._validate_and_normalize_group(param_group)
        super().add_param_group(param_group)
