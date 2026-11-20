from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any, Dict, Literal, Tuple

from .constants import (
    DEFAULT_BUCKET_CAP_MB,
    DEFAULT_FUSED_SINGLE_PASS,
    DEFAULT_FUSE_TENSORS,
    DEFAULT_PREWARM_JIT,
    VALID_BOUND_TYPES,
    VALID_BUFFER_CANDIDATES,
    VALID_EXECUTION_MODES,
    VALID_MASTER_WEIGHTS,
)
from .validation import HAS_TORCH, _is_valid_number, _suggest_closest, torch

logger = logging.getLogger("LuminaV")

BufferMode = Literal[2, 1, "t1", "t2"]
BoundType = Literal["coordinate", "radial"]
MasterWeights = Literal["none", "semi", "full"]
ExecutionMode = Literal["auto", "triton", "vulkan", "simd", "eager", "scalar"]

_BOUND_CODE_MAP: Dict[str, int] = {"radial": 1, "coordinate": 2}
_MASTER_CODE_MAP: Dict[str, int] = {"none": 0, "semi": 1, "full": 2}
_RUST_EXEC_CODE_MAP: Dict[str, int] = {
    "auto": 0,
    "triton": 1,
    "eager": 2,
    "vulkan": 3,
    "simd": 4,
    "scalar": 5,
}


def is_dual_buffer(buf: BufferMode) -> bool:
    return buf in (2, "t2")


def is_triangle_buffer(buf: BufferMode) -> bool:
    return buf in ("t1", "t2")


def bound_mode_code(bound: bool, bound_type: str, bound_ratio: float) -> int:
    if not bound or bound_ratio <= 0.0:
        return 0
    return _BOUND_CODE_MAP[bound_type]


def master_weight_code(master_weights: str) -> int:
    return _MASTER_CODE_MAP[master_weights]


@dataclass(frozen=True)
class StepOptions:
    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    tau: float
    lam: float
    cautious: bool
    clamp_min: float
    buffer: BufferMode
    use_sr: bool
    bound: bool
    bound_type: BoundType
    bound_ratio: float
    master_weights: MasterWeights
    execution: ExecutionMode
    fused_single_pass: bool
    ema_decay: float
    fsdp_sync: bool
    seed: int

    @property
    def is_dual(self) -> bool:
        return is_dual_buffer(self.buffer)

    @property
    def is_triangle(self) -> bool:
        return is_triangle_buffer(self.buffer)

    @property
    def is_t2(self) -> bool:
        return self.buffer == "t2"

    @property
    def bound_mode(self) -> int:
        return bound_mode_code(self.bound, self.bound_type, self.bound_ratio)

    @property
    def master_mode(self) -> int:
        return master_weight_code(self.master_weights)

    @property
    def needs_ema_sync(self) -> bool:
        return self.cautious or self.bound_mode == 1

    def dual_coefficients(self, step: int) -> Tuple[float, float]:
        safe_step = max(step, 1)
        bc1_nes = 1.0 - (self.beta1 ** (safe_step + 1))
        bc2 = max(1.0 - (self.beta2 ** safe_step), 1e-15)
        if self.is_t2:
            return (0.5 * self.lam) / math.sqrt(bc2), 0.5 * self.lam
        return (bc1_nes * self.tau) / math.sqrt(bc2), self.eps * bc1_nes * self.tau

    @classmethod
    def from_group(cls, group: Dict[str, Any], fallback_seed: int = 1337) -> StepOptions:
        buf: BufferMode = group["buffer"]
        beta1, beta2 = group["betas"]
        return cls(
            lr=float(group["lr"]),
            beta1=float(beta1),
            beta2=float(beta2),
            eps=float(group["eps"]),
            weight_decay=float(group["weight_decay"]),
            tau=float(group["tau"]),
            lam=float(group.get("lam", 0.3)),
            cautious=bool(group.get("cautious", True)),
            clamp_min=float(group["cautious_clamp_min"]),
            buffer=buf,
            use_sr=bool(group.get("stochastic_rounding", True)),
            bound=bool(group.get("bound", True)),
            bound_type=group["bound_type"],
            bound_ratio=float(group.get("bound_ratio", 0.05)),
            master_weights=group["master_weights"],
            execution=group["execution"],
            fused_single_pass=bool(group.get("fused_single_pass", DEFAULT_FUSED_SINGLE_PASS)),
            ema_decay=float(group.get("ema_decay", 0.95 if not is_dual_buffer(buf) else 0.80)),
            fsdp_sync=bool(group.get("fsdp_sync", False)),
            seed=int(group.get("seed", fallback_seed)),
        )


def _normalize_buffer(value: Any) -> BufferMode:
    if isinstance(value, bool):
        raise ValueError(
            f"Invalid buffer: {value!r}. Expected 2, 1, 't1', or 't2'."
        )
    if isinstance(value, int):
        if value == 1:
            return 1
        if value == 2:
            return 2
        raise ValueError(
            f"Invalid buffer: {value}. Expected 2, 1, 't1', or 't2'."
        )
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower == "t1":
            return "t1"
        if lower == "t2":
            return "t2"
        sug = _suggest_closest(value, VALID_BUFFER_CANDIDATES, 2)
        hint = f" Did you mean '{sug}'?" if sug is not None else ""
        raise ValueError(
            f"Invalid buffer: '{value}'. Expected 2, 1, 't1', or 't2'.{hint}"
        )
    raise ValueError(f"Invalid buffer: {value!r}. Expected 2, 1, 't1', or 't2'.")


def _normalize_choice(name: str, value: Any, valid: Tuple[str, ...]) -> str:
    expected = ", ".join(f"'{v}'" for v in valid)
    if not isinstance(value, str):
        raise ValueError(f"Invalid {name}: {value!r}. Expected one of {expected}.")
    lower = value.strip().lower()
    if lower in valid:
        return lower
    sug = _suggest_closest(value, valid, 3)
    hint = f" Did you mean '{sug}'?" if sug is not None else ""
    raise ValueError(f"Invalid {name}: '{value}'. Expected one of {expected}.{hint}")


def compute_group_sig(group: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        group.get("lr"),
        group.get("betas"),
        group.get("eps"),
        group.get("weight_decay"),
        group.get("tau"),
        group.get("lam"),
        group.get("cautious"),
        group.get("cautious_clamp_min"),
        group.get("buffer"),
        group.get("stochastic_rounding"),
        group.get("bound"),
        group.get("bound_type"),
        group.get("bound_ratio"),
        group.get("master_weights"),
        group.get("execution"),
        group.get("fused_single_pass"),
        group.get("fuse_tensors"),
        group.get("bucket_cap_mb"),
        group.get("prewarm_jit"),
        group.get("ema_decay"),
        group.get("fsdp_sync"),
        group.get("seed"),
    )


def validate_and_normalize_group(group: Dict[str, Any]) -> None:
    lr = group["lr"]
    betas = group["betas"]
    eps = group["eps"]
    weight_decay = group["weight_decay"]
    tau = group["tau"]
    lam = group.get("lam", 0.3)
    cautious = group.get("cautious", True)
    cautious_clamp_min = group["cautious_clamp_min"]
    buffer = group["buffer"]
    stochastic_rounding = group.get("stochastic_rounding", True)
    bound = group.get("bound", True)
    bound_type = group["bound_type"]
    bound_ratio = group["bound_ratio"]
    master_weights = group["master_weights"]
    execution = group["execution"]
    fused_single_pass = group.get("fused_single_pass", DEFAULT_FUSED_SINGLE_PASS)
    fuse_tensors = group.get("fuse_tensors", DEFAULT_FUSE_TENSORS)
    prewarm_jit = group.get("prewarm_jit", DEFAULT_PREWARM_JIT)
    ema_decay = group.get("ema_decay", None)
    fsdp_sync = group.get("fsdp_sync", False)
    seed = group["seed"]

    if not isinstance(fuse_tensors, bool):
        raise ValueError(f"Invalid fuse_tensors: {fuse_tensors!r}. Expected a boolean (True or False).")
    if not isinstance(prewarm_jit, bool):
        raise ValueError(f"Invalid prewarm_jit: {prewarm_jit!r}. Expected a boolean (True or False).")
    if not isinstance(bound, bool):
        raise ValueError(f"Invalid bound: {bound!r}. Expected a boolean (True or False).")
    if not _is_valid_number(lr) or float(lr) < 0.0:
        raise ValueError(f"Invalid lr: {lr}. Expected a finite value in range [0.0, inf).")
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError(f"Invalid betas: {betas}. Expected a 2-tuple (beta1, beta2) in [0.0, 1.0).")
    if not _is_valid_number(betas[0]) or not (0.0 <= float(betas[0]) < 1.0):
        raise ValueError(f"Invalid beta1: {betas[0]}. Expected value in range [0.0, 1.0).")
    if not _is_valid_number(betas[1]) or not (0.0 <= float(betas[1]) < 1.0):
        raise ValueError(f"Invalid beta2: {betas[1]}. Expected value in range [0.0, 1.0).")
    if not _is_valid_number(eps) or float(eps) <= 0.0:
        raise ValueError(f"Invalid eps: {eps}. Expected value in range (0.0, inf).")
    if not _is_valid_number(weight_decay) or float(weight_decay) < 0.0:
        raise ValueError(f"Invalid weight_decay: {weight_decay}. Expected value in range [0.0, inf).")
    if not _is_valid_number(tau) or float(tau) <= 0.0:
        raise ValueError(f"Invalid tau: {tau}. Expected value in range (0.0, inf).")
    if not _is_valid_number(lam) or float(lam) < 0.0:
        raise ValueError(f"Invalid lam: {lam}. Expected value in range [0.0, inf).")
    if not _is_valid_number(cautious_clamp_min) or not (0.0 < float(cautious_clamp_min) <= 1.0):
        raise ValueError(f"Invalid cautious_clamp_min: {cautious_clamp_min}. Expected value in range (0.0, 1.0].")
    if not _is_valid_number(bound_ratio) or float(bound_ratio) <= 0.0:
        raise ValueError(f"Invalid bound_ratio: {bound_ratio}. Expected value in range (0.0, inf).")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Invalid seed: {seed}. Expected a non-negative integer in range [0, 4294967295].")

    buf_val = _normalize_buffer(buffer)
    bt_val = _normalize_choice("bound_type", bound_type, VALID_BOUND_TYPES)
    mw_val = _normalize_choice("master_weights", master_weights, VALID_MASTER_WEIGHTS)
    ex_val = _normalize_choice("execution", execution, VALID_EXECUTION_MODES)

    if ema_decay is None or (isinstance(ema_decay, str) and ema_decay.strip().lower() == "auto"):
        ema_decay_val = 0.95 if not is_dual_buffer(buf_val) else 0.80
    elif _is_valid_number(ema_decay):
        ema_decay_val = float(ema_decay)
        if not (0.0 < ema_decay_val < 1.0):
            raise ValueError(f"Invalid ema_decay: {ema_decay}. Expected value in range (0.0, 1.0).")
    else:
        raise ValueError(f"Invalid ema_decay: {ema_decay!r}. Expected 'auto', None, or a float in (0.0, 1.0).")

    if fsdp_sync and not (
        HAS_TORCH
        and torch is not None
        and hasattr(torch, "distributed")
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        logger.warning(
            "LuminaV: fsdp_sync=True was specified, but torch.distributed is not initialized. "
            "Disabling fsdp_sync to prevent collective communication errors."
        )
        fsdp_sync = False

    raw_cap = group.get("bucket_cap_mb", DEFAULT_BUCKET_CAP_MB)
    if raw_cap is None or not _is_valid_number(raw_cap) or float(raw_cap) <= 0.0:
        raise ValueError(f"bucket_cap_mb must be a positive number, got: {raw_cap!r}")

    group["lr"] = float(lr)
    group["betas"] = (float(betas[0]), float(betas[1]))
    group["eps"] = float(eps)
    group["weight_decay"] = float(weight_decay)
    group["tau"] = float(tau)
    group["lam"] = float(lam)
    group["cautious"] = bool(cautious)
    group["cautious_clamp_min"] = float(cautious_clamp_min)
    group["buffer"] = buf_val
    group["stochastic_rounding"] = bool(stochastic_rounding)
    group["bound"] = bool(bound)
    group["bound_type"] = bt_val
    group["bound_ratio"] = float(bound_ratio)
    group["master_weights"] = mw_val
    group["execution"] = ex_val
    group["fused_single_pass"] = bool(fused_single_pass)
    group["fuse_tensors"] = bool(fuse_tensors)
    group["bucket_cap_mb"] = float(raw_cap)
    group["prewarm_jit"] = bool(prewarm_jit)
    group["ema_decay"] = ema_decay_val
    group["fsdp_sync"] = bool(fsdp_sync)
    group["seed"] = int(seed)
    group["_validated_sig"] = compute_group_sig(group)


def build_rust_config(group: Dict[str, Any]) -> Dict[str, Any]:
    bt = str(group["bound_type"])
    return {
        "lr": float(group["lr"]),
        "beta1": float(group["betas"][0]),
        "beta2": float(group["betas"][1]),
        "eps": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
        "tau": float(group["tau"]),
        "lam": float(group["lam"]),
        "cautious": bool(group["cautious"]),
        "cautious_clamp_min": float(group["cautious_clamp_min"]),
        "buffer": group["buffer"],
        "stochastic_rounding": bool(group["stochastic_rounding"]),
        "bound": bool(group["bound"]),
        "bound_type": _BOUND_CODE_MAP[bt],
        "bound_ratio": float(group["bound_ratio"]),
        "master_weights": _MASTER_CODE_MAP[str(group["master_weights"])],
        "execution": _RUST_EXEC_CODE_MAP[str(group["execution"])],
        "fused_single_pass": bool(group["fused_single_pass"]),
        "ema_decay": float(group["ema_decay"]),
        "fsdp_sync": bool(group["fsdp_sync"]),
        "seed": int(group["seed"]),
    }
