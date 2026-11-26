from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
from typing import Any, Dict, List, Literal, Optional, Tuple

from . import constants as _c
from .backends._sampling import extract_params, is_dtensor, local_tensor, param_nbytes
from .validation import HAS_NUMPY, HAS_TORCH, np, torch

MemoryPriority = Literal["balanced", "low_memory", "speed", "high_precision"]
_VALID_PRIORITIES = ("balanced", "low_memory", "speed", "high_precision")


@dataclass(frozen=True)
class ModelProfile:
    """Structural, precision, and memory summary inferred from a model or parameter list."""

    architecture: str
    total_params: int
    param_tensors: int
    param_bytes: int
    median_width: int
    max_width: int
    depth: int
    dominant_dtype: str
    has_low_precision: bool
    has_fp16: bool
    has_bf16: bool
    has_batchnorm: bool
    has_moe: bool
    has_dtensor: bool
    device_type: str
    free_vram_bytes: Optional[int]
    total_vram_bytes: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON- and weights_only-safe dictionary of profile fields."""
        return asdict(self)


def round_sig(value: float, sig: int = 2) -> float:
    """Round a positive finite float to `sig` significant digits for clean readable priors."""
    if value <= 0.0 or not math.isfinite(value):
        return value
    exp = math.floor(math.log10(abs(value)))
    factor = 10.0 ** (sig - 1 - exp)
    return round(value * factor) / factor


def _dtype_name(p: Any) -> str:
    lp = local_tensor(p)
    dt = getattr(lp, "dtype", None)
    if dt is None:
        return "float32"
    raw = str(dt)
    return raw.split(".")[-1]


def _param_numel(p: Any) -> int:
    lp = local_tensor(p)
    if HAS_TORCH and torch is not None and isinstance(lp, torch.Tensor):
        return int(lp.numel())
    if HAS_NUMPY and np is not None and isinstance(lp, np.ndarray):
        return int(lp.size)
    return 0


def _param_shape(p: Any) -> Tuple[int, ...]:
    lp = local_tensor(p)
    shp = getattr(lp, "shape", ())
    return tuple(int(d) for d in shp)


def _infer_architecture(model_or_params: Any) -> Tuple[str, bool, bool]:
    if not (HAS_TORCH and torch is not None and isinstance(model_or_params, torch.nn.Module)):
        return "generic", False, False

    conv_count = 0
    bn_count = 0
    ln_count = 0
    emb_count = 0
    attn_count = 0
    linear_count = 0
    moe_count = 0

    for mod in model_or_params.modules():
        cls_name = type(mod).__name__.lower()
        if isinstance(mod, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            conv_count += 1
        if isinstance(
            mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)
        ) or "batchnorm" in cls_name:
            bn_count += 1
        if isinstance(mod, torch.nn.LayerNorm) or any(tok in cls_name for tok in ("rmsnorm", "ranorm", "layernorm")):
            ln_count += 1
        if isinstance(mod, torch.nn.Embedding):
            emb_count += 1
        if isinstance(mod, (torch.nn.MultiheadAttention, torch.nn.TransformerEncoderLayer, torch.nn.TransformerDecoderLayer)) or (
            "attention" in cls_name
        ):
            attn_count += 1
        if isinstance(mod, torch.nn.Linear):
            linear_count += 1
        if "moe" in cls_name or "expert" in cls_name:
            moe_count += 1

    has_bn = bn_count > 0
    has_moe = moe_count > 0
    if conv_count > 0 and attn_count > 0:
        return "vit", has_bn, has_moe
    if attn_count > 0 or emb_count > 0:
        return "transformer", has_bn, has_moe
    if conv_count > 0:
        return "cnn", has_bn, has_moe
    if linear_count > 0:
        return "mlp", has_bn, has_moe
    return "generic", has_bn, has_moe


def profile_model(model_or_params: Any) -> ModelProfile:
    """Inspect a PyTorch module or parameter list to summarize architecture, scale, dtype, and VRAM."""
    params = extract_params(model_or_params)
    if not params:
        raise ValueError("AutoTune got an empty parameter list.")

    total_params = 0
    total_bytes = 0
    dtype_bytes: Dict[str, int] = {}
    widths: List[int] = []
    max_numel = 1
    has_fp16 = False
    has_bf16 = False
    has_dt = False

    for p in params:
        n = _param_numel(p)
        nb = param_nbytes(p)
        dt = _dtype_name(p)
        total_params += n
        total_bytes += nb
        max_numel = max(max_numel, n)
        dtype_bytes[dt] = dtype_bytes.get(dt, 0) + max(nb, 1)
        if dt == "float16":
            has_fp16 = True
        elif dt == "bfloat16":
            has_bf16 = True
        if is_dtensor(p):
            has_dt = True

        shp = _param_shape(p)
        if len(shp) >= 2:
            if len(shp) == 2:
                widths.append(max(1, min(shp[0], shp[1])))
            else:
                widths.append(max(1, shp[0]))

    dominant_dtype = max(dtype_bytes, key=lambda k: dtype_bytes[k]) if dtype_bytes else "float32"
    if widths:
        median_width = int(statistics.median(widths))
        max_width = max(widths)
        depth = len(widths)
    else:
        est_w = max(16, math.isqrt(max_numel))
        median_width = est_w
        max_width = est_w
        depth = len(params)

    first = params[0]
    is_torch = HAS_TORCH and torch is not None and isinstance(first, torch.Tensor)
    dev_type = first.device.type if is_torch else "numpy"
    free_vram: Optional[int] = None
    total_vram: Optional[int] = None
    if is_torch and dev_type == "cuda" and torch is not None and torch.cuda.is_available():
        try:
            free_v, tot_v = torch.cuda.mem_get_info(first.device)
            free_vram = int(free_v)
            total_vram = int(tot_v)
        except Exception:
            free_vram, total_vram = None, None

    arch, has_bn, has_moe = _infer_architecture(model_or_params)
    return ModelProfile(
        architecture=arch,
        total_params=total_params,
        param_tensors=len(params),
        param_bytes=total_bytes,
        median_width=median_width,
        max_width=max_width,
        depth=depth,
        dominant_dtype=dominant_dtype,
        has_low_precision=bool(has_fp16 or has_bf16),
        has_fp16=has_fp16,
        has_bf16=has_bf16,
        has_batchnorm=has_bn,
        has_moe=has_moe,
        has_dtensor=has_dt,
        device_type=dev_type,
        free_vram_bytes=free_vram,
        total_vram_bytes=total_vram,
    )


def recommend_hyperparameters(
    profile: ModelProfile,
    *,
    memory_priority: MemoryPriority = "balanced",
    user_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    """Derive architecture- and memory-aware hyperparameter priors for `LuminaV`.

    Returns `(config, sources, notes)` where every non-overridden field is marked `"prior"`.
    """
    if memory_priority not in _VALID_PRIORITIES:
        expected = ", ".join(repr(x) for x in _VALID_PRIORITIES)
        raise ValueError(f"Invalid memory_priority: {memory_priority!r}. Expected one of {expected}.")

    overrides = dict(user_overrides or {})
    notes: List[str] = []
    vram_tight = bool(
        profile.free_vram_bytes is not None
        and profile.param_bytes > 0
        and (profile.param_bytes * 4.5) > (profile.free_vram_bytes * 0.70)
    )

    # 1. Buffer mode selection (memory budget vs speed vs balanced accuracy)
    if memory_priority == "low_memory" or vram_tight or profile.total_params >= 1_000_000_000:
        buffer_mode: Any = "t1"
        notes.append("Selected single-buffer Triangle mode ('t1') to cut optimizer state memory by ~50%.")
    elif memory_priority == "speed" or (
        profile.architecture in ("transformer", "vit") and profile.total_params >= 50_000_000
    ):
        buffer_mode = "t2"
        notes.append("Selected dual-buffer tanh-free Triangle mode ('t2') for high-throughput stability.")
    else:
        buffer_mode = _c.DEFAULT_BUFFER

    if "buffer" in overrides:
        buffer_mode = overrides["buffer"]
    is_dual = buffer_mode in (2, "t2", "T2")

    # 2. Precision & master weights
    if not profile.has_low_precision:
        master_weights = "none"
        stochastic_rounding = False
    elif profile.has_fp16:
        can_semi = (
            memory_priority != "low_memory"
            and (
                profile.free_vram_bytes is None
                or (profile.param_bytes * 6.0) <= (profile.free_vram_bytes * 0.70)
            )
        )
        master_weights = "semi" if can_semi else "none"
        stochastic_rounding = True
        notes.append(
            f"FP16 parameters detected; selected master_weights={master_weights!r} with stochastic_rounding=True."
        )
    else:
        master_weights = "semi" if memory_priority == "high_precision" else "none"
        stochastic_rounding = True

    # 3. Architecture- and width-scaled learning rate prior (muP-inspired 1/sqrt(width) scaling)
    arch_base_lr = {
        "transformer": 4.0e-4,
        "vit": 5.0e-4,
        "cnn": 1.0e-3,
        "mlp": 1.0e-3,
        "generic": _c.DEFAULT_LR,
    }.get(profile.architecture, _c.DEFAULT_LR)
    width_scale = math.sqrt(512.0 / float(max(profile.median_width, 64)))
    depth_scale = (12.0 / float(max(profile.depth, 12))) ** 0.25
    buf_scale = 1.0 if is_dual else 0.75
    raw_lr = arch_base_lr * width_scale * depth_scale * buf_scale
    lr_prior = round_sig(min(max(raw_lr, 3.0e-5), 5.0e-3), sig=2)

    # 4. Weight decay, cautious WD, betas, variance tempering, and trust-region bounds
    if profile.architecture in ("transformer", "vit"):
        weight_decay = 1.0e-1
        cautious_wd = True
        betas = (0.9, 0.95) if (profile.total_params >= 10_000_000 or profile.has_moe) else _c.DEFAULT_BETAS
    elif profile.architecture == "cnn":
        weight_decay = 5.0e-2 if profile.total_params >= 5_000_000 else 1.0e-2
        cautious_wd = profile.total_params >= 1_000_000
        betas = _c.DEFAULT_BETAS
    else:
        weight_decay = _c.DEFAULT_WEIGHT_DECAY if profile.total_params >= 1_000_000 else 1.0e-2
        cautious_wd = False
        betas = _c.DEFAULT_BETAS

    use_rtv_prior = bool(
        is_dual and (profile.has_moe or (profile.architecture == "transformer" and profile.depth >= 24))
    )
    variance_type = "rtv" if use_rtv_prior else "standard"
    if use_rtv_prior:
        notes.append("Enabled variance_type='rtv' to damp heavy-tailed gradient spikes in deep/MoE layers.")

    bound_ratio = 0.03 if (profile.has_moe or profile.has_fp16 or profile.depth >= 32) else _c.DEFAULT_BOUND_RATIO

    cfg: Dict[str, Any] = {
        "lr": lr_prior,
        "betas": betas,
        "weight_decay": weight_decay,
        "cautious_weight_decay": cautious_wd,
        "variance_type": variance_type,
        "buffer": buffer_mode,
        "stochastic_rounding": stochastic_rounding,
        "bound_type": _c.DEFAULT_BOUND_TYPE,
        "bound_ratio": bound_ratio,
        "bound_floor": _c.DEFAULT_BOUND_FLOOR,
        "master_weights": master_weights,
    }
    sources: Dict[str, str] = {k: "prior" for k in cfg}
    for k, v in overrides.items():
        if k in cfg:
            cfg[k] = v
            sources[k] = "user"
    return cfg, sources, notes


__all__ = [
    "MemoryPriority",
    "ModelProfile",
    "profile_model",
    "recommend_hyperparameters",
    "round_sig",
]
