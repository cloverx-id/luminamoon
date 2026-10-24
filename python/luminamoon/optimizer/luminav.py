from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from .version import (
    DEFAULT_LUMINAV_VERSION,
    HybridVersionMethod,
    ModularVersionResolver,
    SUPPORTED_VERSIONS,
    validate_version as _validate_luminav_version,
)
from .standalone import ParamStateDict, StandaloneOptimizer
from .validation import (
    _is_contiguous,
    _is_valid_number,
    _levenshtein,
    _make_contiguous,
    _no_grad_decorator,
    _numel,
    _prng_numpy,
    _quantize_numpy,
    _suggest_closest,
    _validate_supported_dtype,
)
from .numpy_backend import step_numpy_group
from .constants import (
    DEFAULT_BETAS,
    DEFAULT_BOUND_RATIO,
    DEFAULT_CAUTIOUS_CLAMP_MIN,
    DEFAULT_EPS,
    DEFAULT_GPU_DISPATCH_THRESHOLD,
    DEFAULT_LAM,
    DEFAULT_LR,
    DEFAULT_OPTIMIZER_SEED,
    DEFAULT_TAU,
    DEFAULT_WEIGHT_DECAY,
    TEMP_STATE_DICT_KEYS,
    VALID_BOUND_TYPE_CANDIDATES,
    VALID_BUFFER_CANDIDATES,
    VALID_EXECUTION_MODE_CANDIDATES,
    VALID_MASTER_WEIGHTS_CANDIDATES,
)

if TYPE_CHECKING:
    import numpy as np
    import torch
    from torch.optim import Optimizer as TorchOptimizer
    HAS_NUMPY: bool = True
    HAS_TORCH: bool = True
else:
    HAS_NUMPY = False
    try:
        import numpy as np
        HAS_NUMPY = True
    except ImportError:
        np = None
        HAS_NUMPY = False

    HAS_TORCH = False
    try:
        import torch
        from torch.optim import Optimizer as TorchOptimizer
        HAS_TORCH = True
    except ImportError:
        torch = None
        TorchOptimizer = None
        HAS_TORCH = False


if TYPE_CHECKING:
    OptimizerBase = TorchOptimizer
else:
    OptimizerBase = TorchOptimizer if (HAS_TORCH and TorchOptimizer is not None) else StandaloneOptimizer


logger = logging.getLogger("LuminaV")

if TYPE_CHECKING:
    _luminamoon_rs: Any = None
    from .fsdp_sync import (
        _sync_fsdp_diff_only,
        _sync_fsdp_scratch,
    )
    from .gpu_backend import LuminaVGpuMixin
    from .triton_kernels import HAS_TRITON
else:
    try:
        from luminamoon.optimizer import _luminamoon_optimizer_rs as _luminamoon_rs
    except ImportError:
        try:
            import _luminamoon_optimizer_rs as _luminamoon_rs
        except ImportError:
            _luminamoon_rs = None

    try:
        from luminamoon.optimizer.gpu_backend import (
            HAS_TRITON,
            LuminaVGpuMixin,
            _sync_fsdp_diff_only,
            _sync_fsdp_scratch,
        )
    except ImportError:
        try:
            from .gpu_backend import (
                HAS_TRITON,
                LuminaVGpuMixin,
                _sync_fsdp_diff_only,
                _sync_fsdp_scratch,
            )
        except ImportError:
            HAS_TRITON = False
            _sync_fsdp_diff_only = None
            _sync_fsdp_scratch = None

            class LuminaVGpuMixin:
                pass


class LuminaV(OptimizerBase, LuminaVGpuMixin):
    """LuminaV Optimizer: Low-Precision Cautious-Trust-Bounded Adaptive Optimizer.

    Implements the canonical LuminaV v1.5.3 optimization algorithm supporting both
    Dual-Buffer Convex Lookahead Momentum (`buffer=2` / CLM) and Single-Buffer
    Triangle-Defect Convex Lookahead Momentum (`buffer=1` / TD-CLM) modes across
    PyTorch (`torch.Tensor`) and standalone NumPy (`np.ndarray`) parameters.

    Hardware execution automatically dispatches across Triton AOT/JIT GPU kernels,
    Vulkan 1.3/1.4 compute shaders, Rayon CPU SIMD (AVX-512 / AVX2), or the reference
    scalar core.

    Attributes:
        ALGORITHM_SPEC (str): Canonical LuminaV mathematical specification version (`"1.5.3"`).
        state (ParamStateDict): Per-parameter optimizer state dictionary.
    """

    ALGORITHM_SPEC: str = DEFAULT_LUMINAV_VERSION
    state: Any
    _default_version: str = DEFAULT_LUMINAV_VERSION

    def __init__(
        self,
        params,
        lr: float = DEFAULT_LR,
        betas: Tuple[float, float] = DEFAULT_BETAS,
        eps: float = DEFAULT_EPS,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        tau: float = DEFAULT_TAU,
        lam: float = DEFAULT_LAM,
        cautious: bool = True,
        cautious_clamp_min: float = DEFAULT_CAUTIOUS_CLAMP_MIN,
        buffer: Union[int, str] = 2,
        centered_defect: bool = False,
        stochastic_rounding: bool = True,
        bound: bool = True,
        bound_type: Union[int, str] = "coordinate",
        bound_ratio: float = DEFAULT_BOUND_RATIO,
        master_weights: Union[bool, int, str] = "none",
        execution: Union[int, str] = "auto",
        fused_single_pass: bool = True,
        ema_decay: Optional[Union[float, str]] = "auto",
        fsdp_sync: bool = False,
        seed: int = DEFAULT_OPTIMIZER_SEED,
        version: Optional[str] = None,
    ):
        """Initializes the LuminaV optimizer.

        Args:
            params: Iterable of parameters (`torch.Tensor` or `np.ndarray`) or dicts
                defining parameter groups.
            lr: Learning rate (default: `8e-4`).
            betas: Coefficients `(beta1, beta2)` used for computing running averages of
                first-order momentum and centered innovation variance (default: `(0.9, 0.999)`).
            eps: Term added to the denominator to improve numerical stability (default: `1e-8`).
            weight_decay: Decoupled weight decay coefficient (default: `0.08`).
            tau: Soft-projection temperature threshold for adaptive denominator damping
                in Dual-Buffer CLM (`buffer=2`) (default: `0.8`).
            lam: Triangle defect ratio weight controlling the triangle inequality envelope in
                Single-Buffer TD-CLM (`buffer=1`) mode (default: `0.3`).
            cautious: Whether to enable sign-aligned cautious update masking and EMA
                scaling (default: `True`).
            cautious_clamp_min: Minimum lower bound for the cautious alignment scaling
                factor in `(0.0, 1.0]` (default: `0.5`).
            buffer: Optimizer state buffer topology. Use `2` (or `"dual"`, `"clm"`) for
                Dual-Buffer Convex Lookahead Momentum (CLM), or `1` (or `"single"`,
                `"td-clm"`, `"tdclm"`) for Single-Buffer Triangle-Defect Convex Lookahead
                Momentum (TD-CLM) (default: `2`).
            centered_defect: Whether to subtract the mean innovation defect across both
                Dual-Buffer CLM (`buffer=2`) and Single-Buffer TD-CLM (`buffer=1`) modes
                (default: `False`).
            stochastic_rounding: Whether to apply stateless 32-bit MurmurHash3 bitwise
                stochastic rounding when writing to low-precision (`bfloat16` / `float16`)
                parameters and momentum states (default: `True`).
            bound: Whether to enforce adaptive trust-region update bounding (default: `True`).
            bound_type: Trust-region bounding geometry: `"coordinate"` (element-wise clip,
                zero collective sync) or `"radial"` (global norm ratio)
                (default: `"coordinate"`).
            bound_ratio: Maximum allowed update magnitude relative to parameter scale
                (default: `0.05`).
            master_weights: Mixed-precision master weight mode for `bfloat16`/`float16`
                parameters: `"none"` (`False`), `"semi"`, or `"full"` (`True`)
                (default: `"none"`).
            execution: Hardware execution backend: `"auto"`, `"vulkan"`, `"simd"`,
                `"scalar"`, or `"triton"` (default: `"auto"`).
            fused_single_pass: Whether to fuse update and statistics collection into a
                single kernel pass (default: `True`).
            ema_decay: EMA decay factor for lagged cautious/radial scalars (`"auto"`
                selects `0.80` for `buffer=2` and `0.95` for `buffer=1`)
                (default: `"auto"`).
            fsdp_sync: Whether to synchronize global reduction scratch buffers across
                `torch.distributed` FSDP shards (default: `False`).
            seed: Deterministic PRNG seed for stateless stochastic rounding
                (default: `1337`).
            version: LuminaV algorithm specification version lock (default: `"1.5.3"`).
        """
        target_version = version if version is not None else self._default_version
        self.version: str = _validate_luminav_version(target_version)
        ModularVersionResolver.ensure_version_ready(self.version)
        defaults: Dict[str, Any] = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            tau=tau,
            lam=lam,
            cautious=cautious,
            cautious_clamp_min=cautious_clamp_min,
            buffer=buffer,
            centered_defect=centered_defect,
            stochastic_rounding=stochastic_rounding,
            bound=bound,
            bound_type=bound_type,
            bound_ratio=bound_ratio,
            master_weights=master_weights,
            execution=execution,
            fused_single_pass=fused_single_pass,
            ema_decay=ema_decay,
            fsdp_sync=fsdp_sync,
            seed=seed,
            version=self.version,
        )

        is_auto_ema = ema_decay is None or (
            isinstance(ema_decay, str) and ema_decay.strip().lower() == "auto"
        )
        self._validate_and_normalize_group(defaults)
        if is_auto_ema:
            defaults["ema_decay"] = "auto"

        param_list = list(params)
        if len(param_list) == 0:
            raise ValueError("optimizer got an empty parameter list")

        first_elem = param_list[0]
        if isinstance(first_elem, dict):
            first_p = first_elem.get("params", [None])[0] if first_elem.get("params") else None
        else:
            first_p = first_elem

        is_torch = (
            HAS_TORCH
            and torch is not None
            and TorchOptimizer is not None
            and isinstance(first_p, torch.Tensor)
        )
        self._is_torch_mode = is_torch

        if is_torch:
            super().__init__(param_list, defaults)
        else:
            self.defaults = defaults
            self.state = ParamStateDict()
            self.param_groups = []
            self._optimizer_step_pre_hooks = {}
            self._optimizer_step_post_hooks = {}
            if not isinstance(param_list[0], dict):
                p_groups = [{"params": param_list}]
            else:
                p_groups = param_list
            for pg in p_groups:
                self.add_param_group(pg)

        if not isinstance(self.state, ParamStateDict):
            self.state = ParamStateDict(self.state)
        self.seed = int(defaults["seed"])
        self._generators = {}
        self._warned_triton_failure = False
        self._warned_triton_contiguous = False
        self._warned_non_contiguous = False

    def _set_version_inst(self, version: str) -> LuminaV:
        self.version = _validate_luminav_version(version)
        if hasattr(self, "defaults") and isinstance(self.defaults, dict):
            self.defaults["version"] = self.version
        if hasattr(self, "param_groups") and isinstance(self.param_groups, list):
            for pg in self.param_groups:
                if isinstance(pg, dict):
                    pg["version"] = self.version
        ModularVersionResolver.ensure_version_ready(self.version)
        return self

    @classmethod
    def _set_version_cls(cls, version: str) -> str:
        v = _validate_luminav_version(version)
        cls._default_version = v
        ModularVersionResolver.set_default_version(v)
        return v

    def _get_version_inst(self) -> str:
        return self.version

    @classmethod
    def _get_version_cls(cls) -> str:
        return cls._default_version

    set_version = HybridVersionMethod(_set_version_inst, _set_version_cls)
    setVersion = set_version

    get_version = HybridVersionMethod(_get_version_inst, _get_version_cls)
    getVersion = get_version

    @classmethod
    def _list_versions_cls(cls) -> list[str]:
        return list(SUPPORTED_VERSIONS)

    def _list_versions_inst(self) -> list[str]:
        return list(SUPPORTED_VERSIONS)

    list_versions = HybridVersionMethod(_list_versions_inst, _list_versions_cls)
    listVersions = list_versions

    @staticmethod
    def _validate_and_normalize_group(group: Dict[str, Any]) -> None:
        lr = group["lr"]
        betas = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        tau = group["tau"]
        lam = group.get("lam", 0.3)
        cautious = group.get("cautious", True)
        cautious_clamp_min = group["cautious_clamp_min"]
        buffer = group["buffer"]
        centered_defect = group.get("centered_defect", False)
        stochastic_rounding = group.get("stochastic_rounding", True)
        bound = group.get("bound", True)
        bound_type = group["bound_type"]
        bound_ratio = group["bound_ratio"]
        master_weights = group["master_weights"]
        execution = group["execution"]
        fused_single_pass = group.get("fused_single_pass", True)
        ema_decay = group.get("ema_decay", None)
        fsdp_sync = group.get("fsdp_sync", False)
        seed = group["seed"]

        if not _is_valid_number(lr) or float(lr) < 0.0:
            raise ValueError(
                f"Invalid lr: {lr}. Expected a finite value in range [0.0, inf)."
            )
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError(
                f"Invalid betas: {betas}. Expected a 2-tuple (beta1, beta2) with values in range [0.0, 1.0)."
            )
        if not _is_valid_number(betas[0]) or not (0.0 <= float(betas[0]) < 1.0):
            raise ValueError(
                f"Invalid beta1: {betas[0]}. Expected value in range [0.0, 1.0)."
            )
        if not _is_valid_number(betas[1]) or not (0.0 <= float(betas[1]) < 1.0):
            raise ValueError(
                f"Invalid beta2: {betas[1]}. Expected value in range [0.0, 1.0)."
            )
        if not _is_valid_number(eps) or float(eps) <= 0.0:
            raise ValueError(
                f"Invalid eps: {eps}. Expected value in range (0.0, inf)."
            )
        if not _is_valid_number(weight_decay) or float(weight_decay) < 0.0:
            raise ValueError(
                f"Invalid weight_decay: {weight_decay}. Expected value in range [0.0, inf)."
            )
        if not _is_valid_number(tau) or float(tau) <= 0.0:
            raise ValueError(
                f"Invalid tau: {tau}. Expected value in range (0.0, inf)."
            )
        if not _is_valid_number(lam) or float(lam) < 0.0:
            raise ValueError(
                f"Invalid lam: {lam}. Expected value in range [0.0, inf)."
            )
        if not _is_valid_number(cautious_clamp_min) or not (
            0.0 < float(cautious_clamp_min) <= 1.0
        ):
            raise ValueError(
                f"Invalid cautious_clamp_min: {cautious_clamp_min}. Expected value in range (0.0, 1.0]."
            )
        if not _is_valid_number(bound_ratio) or float(bound_ratio) <= 0.0:
            raise ValueError(
                f"Invalid bound_ratio: {bound_ratio}. Expected value in range (0.0, inf)."
            )
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(
                f"Invalid seed: {seed}. Expected a non-negative integer in range [0, 4294967295]."
            )

        if isinstance(buffer, bool):
            raise ValueError(
                f"Invalid buffer count: {buffer}. LuminaV only supports buffer=1 (Single-Buffer TD-CLM) or buffer=2 (Dual-Buffer CLM)."
            )
        elif isinstance(buffer, int):
            if buffer in (1, 2):
                buf_val = int(buffer)
            else:
                raise ValueError(
                    f"Invalid buffer count: {buffer}. LuminaV only supports buffer=1 (Single-Buffer TD-CLM) or buffer=2 (Dual-Buffer CLM)."
                )
        elif isinstance(buffer, str):
            buf_lower = buffer.strip().lower()
            if buf_lower in ("1", "tdclm", "td-clm", "single"):
                buf_val = 1
            elif buf_lower in ("2", "dual", "clm"):
                buf_val = 2
            else:
                sug = _suggest_closest(
                    buffer, VALID_BUFFER_CANDIDATES, 2
                )
                if sug is not None:
                    raise ValueError(
                        f"Invalid buffer count: '{buffer}'. LuminaV only supports buffer=1 (Single-Buffer TD-CLM) or buffer=2 (Dual-Buffer CLM). Did you mean '{sug}'?"
                    )
                raise ValueError(
                    f"Invalid buffer count: '{buffer}'. LuminaV only supports buffer=1 (Single-Buffer TD-CLM) or buffer=2 (Dual-Buffer CLM)."
                )
        else:
            raise ValueError(
                f"Invalid buffer count: {buffer}. LuminaV only supports buffer=1 (Single-Buffer TD-CLM) or buffer=2 (Dual-Buffer CLM)."
            )

        if isinstance(bound_type, str):
            bt_lower = bound_type.strip().lower()
            if bt_lower in ("radial", "1"):
                bt_val = 1
            elif bt_lower in ("coordinate", "2"):
                bt_val = 2
            elif bt_lower in ("none", "0") and not bound:
                bt_val = 0
            else:
                sug = _suggest_closest(bound_type, VALID_BOUND_TYPE_CANDIDATES, 3)
                if sug is not None:
                    raise ValueError(
                        f"Invalid bound_type: '{bound_type}'. Expected 'coordinate' or 'radial'. Did you mean '{sug}'?"
                    )
                raise ValueError(
                    f"Invalid bound_type: '{bound_type}'. Expected 'coordinate' or 'radial'."
                )
        elif not isinstance(bound_type, bool) and isinstance(bound_type, int):
            if bound_type in (0, 1, 2):
                bt_val = int(bound_type)
            else:
                raise ValueError(
                    f"Invalid bound_type: {bound_type}. Expected 'coordinate' or 'radial'."
                )
        else:
            raise ValueError(
                f"Invalid bound_type: {bound_type!r}. Expected 'coordinate' or 'radial'."
            )

        if isinstance(master_weights, bool):
            mw_val = 2 if master_weights else 0
        elif isinstance(master_weights, int):
            if master_weights in (0, 1, 2):
                mw_val = int(master_weights)
            else:
                raise ValueError(
                    f"Invalid master_weights: {master_weights}. Expected False, 'none', 'semi', or 'full'."
                )
        elif isinstance(master_weights, str):
            mw_lower = master_weights.strip().lower()
            if mw_lower in ("none", "false", "off", "0"):
                mw_val = 0
            elif mw_lower in ("semi", "hybrid", "half", "1"):
                mw_val = 1
            elif mw_lower in ("full", "true", "fp32", "masterweight", "2"):
                mw_val = 2
            else:
                sug = _suggest_closest(
                    master_weights, VALID_MASTER_WEIGHTS_CANDIDATES, 3
                )
                if sug is not None:
                    raise ValueError(
                        f"Invalid master_weights: '{master_weights}'. Expected False, 'none', 'semi', or 'full'. Did you mean '{sug}'?"
                    )
                raise ValueError(
                    f"Invalid master_weights: '{master_weights}'. Expected False, 'none', 'semi', or 'full'."
                )
        else:
            raise ValueError(
                f"Invalid master_weights: {master_weights!r}. Expected False, 'none', 'semi', or 'full'."
            )

        if isinstance(execution, str):
            ex_lower = execution.strip().lower()
            if ex_lower in ("auto", "foreach", "0"):
                ex_val = 0
            elif ex_lower in ("triton", "1"):
                ex_val = 1
            elif ex_lower in ("single", "2"):
                ex_val = 2
            elif ex_lower in ("vulkan", "3"):
                ex_val = 3
            elif ex_lower in ("simd", "4"):
                ex_val = 4
            elif ex_lower in ("scalar", "scalar_core", "5"):
                ex_val = 5
            else:
                sug = _suggest_closest(
                    execution,
                    VALID_EXECUTION_MODE_CANDIDATES,
                    3,
                )
                if sug is not None:
                    raise ValueError(
                        f"Invalid execution mode: '{execution}'. Expected 'auto', 'vulkan', 'simd', or 'scalar'. Did you mean '{sug}'?"
                    )
                raise ValueError(
                    f"Invalid execution mode: '{execution}'. Expected 'auto', 'vulkan', 'simd', or 'scalar'."
                )
        elif not isinstance(execution, bool) and isinstance(execution, int):
            if 0 <= execution <= 5:
                ex_val = int(execution)
            else:
                raise ValueError(
                    f"Invalid execution mode: {execution}. Expected 'auto', 'vulkan', 'simd', or 'scalar'."
                )
        else:
            raise ValueError(
                f"Invalid execution mode: {execution!r}. Expected 'auto', 'vulkan', 'simd', or 'scalar'."
            )

        if ema_decay is None or (
            isinstance(ema_decay, str) and ema_decay.strip().lower() == "auto"
        ):
            ema_decay_val = 0.95 if buf_val == 1 else 0.80
        elif _is_valid_number(ema_decay):
            ema_decay_val = float(ema_decay)
            if not (0.0 < ema_decay_val < 1.0):
                raise ValueError(
                    f"Invalid ema_decay: {ema_decay}. Expected value in range (0.0, 1.0)."
                )
        else:
            raise ValueError(
                f"Invalid ema_decay: {ema_decay!r}. Expected 'auto', None, or a float in range (0.0, 1.0)."
            )

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

        group["lr"] = float(lr)
        group["betas"] = (float(betas[0]), float(betas[1]))
        group["eps"] = float(eps)
        group["weight_decay"] = float(weight_decay)
        group["tau"] = float(tau)
        group["lam"] = float(lam)
        group["cautious"] = bool(cautious)
        group["cautious_clamp_min"] = float(cautious_clamp_min)
        group["buffer"] = buf_val
        group["centered_defect"] = bool(centered_defect)
        group["stochastic_rounding"] = bool(stochastic_rounding)
        group["bound"] = bool(bound)
        group["bound_type"] = bt_val
        group["bound_ratio"] = float(bound_ratio)
        group["master_weights"] = mw_val
        group["execution"] = ex_val
        group["fused_single_pass"] = bool(fused_single_pass)
        group["ema_decay"] = ema_decay_val
        group["fsdp_sync"] = bool(fsdp_sync)
        group["seed"] = int(seed)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Adds a parameter group to the optimizer's `param_groups` with validated defaults."""
        for name, default in self.defaults.items():
            param_group.setdefault(name, default)
        self._validate_and_normalize_group(param_group)
        if not getattr(self, "_is_torch_mode", False):
            params = param_group["params"]
            if not hasattr(params, "__iter__"):
                param_group["params"] = [params]
            else:
                param_group["params"] = list(params)
            self.param_groups.append(param_group)
            return
        super().add_param_group(param_group)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clears the gradients of all optimized `torch.Tensor` or standalone parameters."""
        if not getattr(self, "_is_torch_mode", False):
            for group in self.param_groups:
                for p in group["params"]:
                    if hasattr(p, "grad") and p.grad is not None:
                        if set_to_none:
                            p.grad = None
                        elif hasattr(p.grad, "zero_"):
                            p.grad.zero_()
                        elif hasattr(p.grad, "fill"):
                            p.grad.fill(0)
            return
        if HAS_TORCH and torch is not None and hasattr(super(), "zero_grad"):
            super().zero_grad(set_to_none=set_to_none)

    @staticmethod
    def detect_hardware() -> Dict[str, Any]:
        """Queries active CPU SIMD ISA, Vulkan 1.3/1.4 UMA device, and Triton AOT/JIT capabilities."""
        if _luminamoon_rs is not None:
            return _luminamoon_rs.detect_hardware_pyo3()
        return {
            "simd_backend": "PyTorch Universal Fallback",
            "triton_aot_available": False,
            "triton_arch": None,
            "vulkan_available": False,
            "gpu_device_name": None,
            "wave32_supported": False,
            "gpu_dispatch_threshold": DEFAULT_GPU_DISPATCH_THRESHOLD,
        }

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the optimizer as a `dict` stripped of transient scratch pointers."""
        temp_keys = TEMP_STATE_DICT_KEYS
        param_ids = {
            id(p) for group in self.param_groups for p in group.get("params", [])
        }
        if not getattr(self, "_is_torch_mode", False):
            serialized_state = {}
            for k, v in self.state.items():
                if isinstance(v, dict):
                    s_dict = {}
                    for sk, sv in v.items():
                        if sk in temp_keys or (sk == "master_param" and id(sv) in param_ids):
                            continue
                        if sk == "scratch" and HAS_NUMPY and np is not None and isinstance(sv, np.ndarray) and sv.size >= 6:
                            sv_clean = sv.copy()
                            sv_clean[:3] = 0.0
                            sv_clean[5] = 0.0
                            s_dict[sk] = sv_clean
                        else:
                            s_dict[sk] = sv
                    serialized_state[k] = s_dict
                else:
                    serialized_state[k] = v
            return {
                "state": serialized_state,
                "param_groups": self.param_groups,
            }

        raw_dict = super().state_dict()
        clean_state = {}
        for k, p_state in raw_dict.get("state", {}).items():
            if isinstance(p_state, dict):
                entry = {}
                for sk, sv in p_state.items():
                    if sk in temp_keys or (sk == "master_param" and id(sv) in param_ids):
                        continue
                    if (
                        sk == "scratch"
                        and HAS_TORCH
                        and torch is not None
                        and isinstance(sv, torch.Tensor)
                        and sv.numel() >= 6
                    ):
                        sv_clean = sv.detach().clone()
                        sv_clean[:3].zero_()
                        sv_clean[5].zero_()
                        entry[sk] = sv_clean
                    else:
                        entry[sk] = sv
                clean_state[k] = entry
            else:
                clean_state[k] = p_state
        return {
            "state": clean_state,
            "param_groups": raw_dict.get("param_groups", []),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the optimizer state and normalizes buffer dtypes and scratch layouts."""
        if not getattr(self, "_is_torch_mode", False):
            self.state = ParamStateDict(state_dict["state"])
            self.param_groups = state_dict["param_groups"]
            for group in self.param_groups:
                for k, v in self.defaults.items():
                    group.setdefault(k, v)
                self._validate_and_normalize_group(group)
            return

        raw_state = state_dict.get("state", {})
        saved_groups = state_dict.get("param_groups", [])
        raw_param_state: Dict[Any, dict] = {}
        if len(self.param_groups) == len(saved_groups):
            for group, saved_group in zip(self.param_groups, saved_groups):
                for p, p_id in zip(group.get("params", []), saved_group.get("params", [])):
                    st_entry = raw_state.get(p_id) if p_id in raw_state else raw_state.get(p)
                    if isinstance(st_entry, dict):
                        raw_param_state[p] = {
                            k: (
                                v.detach().clone()
                                if (HAS_TORCH and torch is not None and isinstance(v, torch.Tensor))
                                else v
                            )
                            for k, v in st_entry.items()
                        }

        super().load_state_dict(state_dict)
        if not isinstance(self.state, ParamStateDict):
            self.state = ParamStateDict(self.state)

        for group in self.param_groups:
            for k, v in self.defaults.items():
                group.setdefault(k, v)
            self._validate_and_normalize_group(group)
            master_mode = int(group.get("master_weights", 0))
            buf_count = int(group.get("buffer", 2))
            cautious_g = bool(group.get("cautious", True))
            radial_g = bool(
                group.get("bound", True)
                and group.get("bound_type", 2) in (1, "radial")
                and float(group.get("bound_ratio", 0.05)) > 0.0
            )
            for p in group["params"]:
                if p not in self.state:
                    continue
                st = self.state[p]
                raw_st = raw_param_state.get(p, {})
                for temp_key in (
                    "p_contig",
                    "exp_avg_contig",
                    "exp_avg_sq_contig",
                    "mp_contig",
                    "_plan",
                    "_contig_step",
                ):
                    st.pop(temp_key, None)
                if "step" in st and HAS_TORCH and torch is not None and isinstance(st["step"], torch.Tensor):
                    st["step"] = int(st["step"].item())
                if "scratch" in st and HAS_TORCH and torch is not None and isinstance(st["scratch"], torch.Tensor):
                    src_sc = (
                        raw_st["scratch"]
                        if ("scratch" in raw_st and isinstance(raw_st["scratch"], torch.Tensor))
                        else st["scratch"]
                    )
                    old_sc = src_sc.to(device=p.device, dtype=torch.float32).contiguous()
                    if old_sc.numel() < 6:
                        new_sc = torch.zeros((6,), device=p.device, dtype=torch.float32)
                        new_sc[3] = 1.0
                        new_sc[4] = 1.0
                        new_sc[: old_sc.numel()].copy_(old_sc)
                        new_sc[:3].zero_()
                        new_sc[5].zero_()
                        st["scratch"] = new_sc
                    else:
                        old_sc[:3].zero_()
                        old_sc[5].zero_()
                        st["scratch"] = old_sc
                else:
                    new_sc = torch.zeros((6,), device=p.device, dtype=torch.float32)
                    new_sc[3] = 1.0
                    new_sc[4] = 1.0
                    st["scratch"] = new_sc
                if not cautious_g:
                    st["scratch"][3] = 1.0
                if not radial_g:
                    st["scratch"][4] = 1.0
                if master_mode == 0 or p.dtype == torch.float32:
                    st.pop("master_param", None)
                elif "master_param" in st and isinstance(st["master_param"], torch.Tensor):
                    src_mp = (
                        raw_st["master_param"]
                        if ("master_param" in raw_st and isinstance(raw_st["master_param"], torch.Tensor))
                        else st["master_param"]
                    )
                    st["master_param"] = src_mp.to(device=p.device, dtype=torch.float32).contiguous()
                else:
                    st["master_param"] = p.detach().clone().float().contiguous()
                target_state_dtype = (
                    torch.float32
                    if (master_mode == 2 or p.dtype == torch.float32)
                    else p.dtype
                )
                for buf_key in ("exp_avg", "exp_avg_sq"):
                    if buf_key in st and isinstance(st[buf_key], torch.Tensor):
                        src_buf = (
                            raw_st[buf_key]
                            if (buf_key in raw_st and isinstance(raw_st[buf_key], torch.Tensor))
                            else st[buf_key]
                        )
                        st[buf_key] = src_buf.to(device=p.device, dtype=target_state_dtype).contiguous()
                if "exp_avg" not in st or not isinstance(st["exp_avg"], torch.Tensor):
                    st["exp_avg"] = torch.zeros_like(
                        p, dtype=target_state_dtype, memory_format=torch.contiguous_format
                    )
                if buf_count == 1:
                    st.pop("exp_avg_sq", None)
                elif "exp_avg_sq" not in st or not isinstance(st["exp_avg_sq"], torch.Tensor):
                    st["exp_avg_sq"] = torch.zeros_like(
                        p, dtype=target_state_dtype, memory_format=torch.contiguous_format
                    )
                st["buf_count"] = buf_count

    @_no_grad_decorator
    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
        grads: Optional[Any] = None,
    ) -> Optional[float]:
        """Performs a single optimization step across all parameter groups.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional).
            grads: Explicit list, tuple, or dict of gradients for standalone NumPy arrays
                or custom gradient injection (optional).

        Returns:
            The loss value returned by `closure` if provided, otherwise `None`.
        """
        loss = None
        if closure is not None:
            if HAS_TORCH and torch is not None and hasattr(torch, "enable_grad"):
                with torch.enable_grad():
                    loss = closure()
            else:
                loss = closure()

        for group in self.param_groups:
            self._validate_and_normalize_group(group)
            cpu_pairs = []
            cpu_grads = []
            cpu_exp_avgs = []
            cpu_exp_avg_sqs = []
            cpu_master_params = []
            cpu_scratches = []
            cpu_steps = []
            cpu_param_indices = []

            gpu_pairs = []
            gpu_grads = []
            gpu_exp_avgs = []
            gpu_exp_avg_sqs = []
            gpu_master_params = []
            gpu_scratches = []
            gpu_steps = []
            gpu_param_indices = []

            buf_count = int(group["buffer"])
            master_mode = int(group["master_weights"])
            fsdp_sync = bool(group.get("fsdp_sync", False))
            cautious = bool(group.get("cautious", True))
            bound = bool(group.get("bound", True))
            bound_type = group.get("bound_type", 2)
            bound_ratio = float(group.get("bound_ratio", 0.05))
            centered_defect = bool(group.get("centered_defect", False))
            radial_active = bool(bound and bound_type in ("radial", 1) and bound_ratio > 0.0)

            for idx, p in enumerate(group["params"]):
                g = None
                if grads is not None:
                    if isinstance(grads, (list, tuple)) and idx < len(grads):
                        g = grads[idx]
                    elif isinstance(grads, dict) and p in grads:
                        g = grads[p]
                if g is None:
                    g = getattr(p, "grad", None)
                if g is None:
                    continue

                if getattr(g, "is_sparse", False):
                    raise RuntimeError("LuminaV does not support sparse gradients.")

                _validate_supported_dtype(p, "param")
                _validate_supported_dtype(g, "grad")

                p_numel = _numel(p)
                if p_numel == 0 and not fsdp_sync:
                    continue

                if not _is_contiguous(g):
                    g = _make_contiguous(g)

                is_torch_tensor = HAS_TORCH and torch is not None and isinstance(p, torch.Tensor)
                dev_type = p.device.type if (is_torch_tensor and hasattr(p, "device")) else "cpu"

                p_for_step = p
                if dev_type == "cpu" and not (is_torch_tensor and fsdp_sync):
                    if hasattr(p, "is_contiguous") and not p.is_contiguous():
                        raise ValueError(
                            f"LuminaV requires parameters to be contiguous in memory for native zero-copy execution. Call .contiguous() on parameter {idx} before passing to optimizer."
                        )
                    elif hasattr(p, "flags") and not getattr(getattr(p, "flags", None), "c_contiguous", True):
                        raise ValueError(
                            f"LuminaV requires parameters to be contiguous in memory for native zero-copy execution. Call np.ascontiguousarray() on parameter {idx} before passing to optimizer."
                        )

                state = self.state[p]
                state_dtype: Any
                if is_torch_tensor and torch is not None:
                    state_dtype = (
                        torch.float32
                        if (master_mode == 2 or p.dtype == torch.float32)
                        else p.dtype
                    )
                elif HAS_NUMPY and np is not None and hasattr(p, "dtype"):
                    state_dtype = (
                        np.float32
                        if (master_mode == 2 or p.dtype == np.float32)
                        else p.dtype
                    )
                else:
                    state_dtype = "float32"

                if len(state) == 0:
                    state["step"] = 0
                    if is_torch_tensor and torch is not None:
                        state["scratch"] = torch.tensor(
                            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                            dtype=torch.float32,
                            device=p.device,
                        )
                        state["exp_avg"] = torch.zeros_like(
                            p_for_step, dtype=state_dtype, memory_format=torch.contiguous_format
                        )
                        if buf_count == 2:
                            state["exp_avg_sq"] = torch.zeros_like(
                                p_for_step, dtype=state_dtype, memory_format=torch.contiguous_format
                            )
                        if master_mode > 0 and p.dtype in (torch.float16, torch.bfloat16):
                            state["master_param"] = p_for_step.detach().clone().float().contiguous()
                    elif HAS_NUMPY and np is not None:
                        state["scratch"] = np.array(
                            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32
                        )
                        state["exp_avg"] = np.zeros_like(p_for_step, dtype=state_dtype)
                        if buf_count == 2:
                            state["exp_avg_sq"] = np.zeros_like(p_for_step, dtype=state_dtype)
                        if master_mode > 0 and getattr(p, "dtype", None) != np.float32:
                            state["master_param"] = np.array(p_for_step, dtype=np.float32, copy=True)
                    else:
                        state["scratch"] = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
                        state["exp_avg"] = [0.0] * p_numel
                        if buf_count == 2:
                            state["exp_avg_sq"] = [0.0] * p_numel
                    state["buf_count"] = buf_count
                else:
                    if is_torch_tensor and torch is not None:
                        if master_mode > 0:
                            if p.dtype == torch.float32:
                                state.pop("master_param", None)
                            elif "master_param" not in state or state["master_param"] is None:
                                state["master_param"] = p_for_step.detach().clone().float().contiguous()
                            elif state["master_param"].device != p.device or state["master_param"].dtype != torch.float32:
                                state["master_param"] = state["master_param"].to(
                                    device=p.device, dtype=torch.float32, non_blocking=True
                                )
                        else:
                            state.pop("master_param", None)

                        if "scratch" not in state or state["scratch"].numel() < 6:
                            scratch = torch.zeros((6,), device=p.device, dtype=torch.float32)
                            scratch[3] = 1.0
                            scratch[4] = 1.0
                            if "scratch" in state and isinstance(state["scratch"], torch.Tensor):
                                old_sc = state["scratch"].to(device=p.device, dtype=torch.float32)
                                scratch[: old_sc.numel()].copy_(old_sc)
                            state["scratch"] = scratch
                            state["buf_count"] = buf_count
                        elif state["scratch"].device != p.device or state["scratch"].dtype != torch.float32:
                            state["scratch"] = state["scratch"].to(
                                device=p.device, dtype=torch.float32, non_blocking=True
                            )
                            state["buf_count"] = buf_count
                        elif state.get("buf_count", buf_count) != buf_count:
                            state["scratch"].zero_()
                            state["scratch"][3] = 1.0
                            state["scratch"][4] = 1.0
                            state["buf_count"] = buf_count

                        if not cautious:
                            state["scratch"][3] = 1.0
                        if not radial_active:
                            state["scratch"][4] = 1.0

                        if "exp_avg" not in state:
                            state["exp_avg"] = torch.zeros_like(
                                p_for_step, dtype=state_dtype, memory_format=torch.contiguous_format
                            )
                        elif state["exp_avg"].device != p.device or state["exp_avg"].dtype != state_dtype:
                            state["exp_avg"] = state["exp_avg"].to(
                                device=p.device, dtype=state_dtype, non_blocking=True
                            )

                        if buf_count == 2:
                            if "exp_avg_sq" not in state:
                                state["exp_avg_sq"] = torch.zeros_like(
                                    p_for_step, dtype=state_dtype, memory_format=torch.contiguous_format
                                )
                            elif state["exp_avg_sq"].device != p.device or state["exp_avg_sq"].dtype != state_dtype:
                                state["exp_avg_sq"] = state["exp_avg_sq"].to(
                                    device=p.device, dtype=state_dtype, non_blocking=True
                                )
                        else:
                            state.pop("exp_avg_sq", None)
                    else:
                        if state.get("buf_count", buf_count) != buf_count:
                            if HAS_NUMPY and np is not None and hasattr(state["scratch"], "__setitem__"):
                                state["scratch"][:] = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
                            state["buf_count"] = buf_count
                        if not cautious and hasattr(state.get("scratch"), "__setitem__"):
                            state["scratch"][3] = 1.0
                        if not radial_active and hasattr(state.get("scratch"), "__setitem__"):
                            state["scratch"][4] = 1.0
                        if buf_count == 2 and "exp_avg_sq" not in state:
                            if HAS_NUMPY and np is not None:
                                state["exp_avg_sq"] = np.zeros_like(p_for_step, dtype=state_dtype)
                        elif buf_count == 1:
                            state.pop("exp_avg_sq", None)
                        if master_mode == 0 or (HAS_NUMPY and np is not None and getattr(p, "dtype", None) == np.float32):
                            state.pop("master_param", None)
                        elif master_mode > 0 and "master_param" not in state and HAS_NUMPY and np is not None:
                            state["master_param"] = np.array(p_for_step, dtype=np.float32, copy=True)

                curr_step = state["step"]
                step_val = (
                    int(curr_step.item())
                    if hasattr(curr_step, "item")
                    else int(curr_step)
                ) + 1
                state["step"] = step_val

                sq = state.get("exp_avg_sq") if buf_count == 2 else None
                mp = state.get("master_param")

                if dev_type == "cpu" and not (is_torch_tensor and fsdp_sync):
                    cpu_pairs.append((p, p_for_step))
                    cpu_grads.append(g)
                    cpu_exp_avgs.append(state["exp_avg"])
                    cpu_exp_avg_sqs.append(sq)
                    cpu_master_params.append(mp)
                    cpu_scratches.append(state["scratch"])
                    cpu_steps.append(step_val)
                    cpu_param_indices.append(idx)
                else:
                    gpu_pairs.append((p, p_for_step))
                    gpu_grads.append(g)
                    gpu_exp_avgs.append(state["exp_avg"])
                    gpu_exp_avg_sqs.append(sq)
                    gpu_master_params.append(mp)
                    gpu_scratches.append(state["scratch"])
                    gpu_steps.append(step_val)
                    gpu_param_indices.append(idx)

            if cpu_pairs:
                c_params_exec = [pair[1] for pair in cpu_pairs]
                if _luminamoon_rs is not None:
                    config_dict = {
                        "lr": float(group["lr"]),
                        "beta1": float(group["betas"][0]),
                        "beta2": float(group["betas"][1]),
                        "eps": float(group["eps"]),
                        "weight_decay": float(group["weight_decay"]),
                        "tau": float(group["tau"]),
                        "lam": float(group["lam"]),
                        "cautious": cautious,
                        "cautious_clamp_min": float(group["cautious_clamp_min"]),
                        "buffer": buf_count,
                        "centered_defect": centered_defect,
                        "stochastic_rounding": bool(group["stochastic_rounding"]),
                        "bound": bound,
                        "bound_type": int(group["bound_type"]),
                        "bound_ratio": bound_ratio,
                        "master_weights": master_mode,
                        "execution": int(group["execution"]),
                        "fused_single_pass": bool(group["fused_single_pass"]),
                        "ema_decay": float(group["ema_decay"]),
                        "fsdp_sync": fsdp_sync,
                        "seed": int(group["seed"]),
                    }

                    _luminamoon_rs.step_pyo3(
                        c_params_exec,
                        cpu_grads,
                        cpu_exp_avgs,
                        cpu_exp_avg_sqs,
                        cpu_master_params,
                        cpu_scratches,
                        cpu_steps,
                        cpu_param_indices,
                        config_dict,
                    )
                else:
                    is_cpu_torch = (
                        HAS_TORCH
                        and torch is not None
                        and isinstance(c_params_exec[0], torch.Tensor)
                    )
                    if is_cpu_torch:
                        if not getattr(self, "_warned_rs_missing", False):
                            logger.info(
                                "LuminaV: Native acceleration engine (_luminamoon_optimizer_rs) is not loaded. "
                                "Executing step seamlessly using pure PyTorch universal engine."
                            )
                            self._warned_rs_missing = True

                        supports_orig = (
                            hasattr(self._step_gpu_group, "__code__")
                            and "orig_indices" in self._step_gpu_group.__code__.co_varnames
                        )
                        if supports_orig:
                            self._step_gpu_group(
                                group,
                                c_params_exec,
                                cpu_grads,
                                cpu_exp_avgs,
                                cpu_exp_avg_sqs,
                                cpu_master_params,
                                cpu_scratches,
                                cpu_steps,
                                orig_indices=cpu_param_indices,
                            )
                        else:
                            self._step_gpu_group(
                                group,
                                c_params_exec,
                                cpu_grads,
                                cpu_exp_avgs,
                                cpu_exp_avg_sqs,
                                cpu_master_params,
                                cpu_scratches,
                                cpu_steps,
                            )
                    elif HAS_NUMPY and np is not None:
                        if not getattr(self, "_warned_rs_missing", False):
                            logger.info(
                                "LuminaV: Native acceleration engine (_luminamoon_optimizer_rs) is not loaded. "
                                "Executing step seamlessly using NumPy universal engine."
                            )
                            self._warned_rs_missing = True

                        self._step_numpy_group(
                            group,
                            c_params_exec,
                            cpu_grads,
                            cpu_exp_avgs,
                            cpu_exp_avg_sqs,
                            cpu_master_params,
                            cpu_scratches,
                            cpu_steps,
                            param_indices=cpu_param_indices,
                        )
                    else:
                        raise RuntimeError(
                            "LuminaV: Neither native Rust acceleration (_luminamoon_optimizer_rs), "
                            "nor PyTorch, nor NumPy was found."
                        )

                for p_orig, p_step in cpu_pairs:
                    if p_orig is not p_step:
                        if hasattr(p_orig, "copy_"):
                            p_orig.copy_(p_step)
                        elif HAS_NUMPY and np is not None:
                            np.copyto(p_orig, p_step)

            if gpu_pairs:
                g_params_exec = [pair[1] for pair in gpu_pairs]
                supports_orig = (
                    hasattr(self._step_gpu_group, "__code__")
                    and "orig_indices" in self._step_gpu_group.__code__.co_varnames
                )
                if supports_orig:
                    self._step_gpu_group(
                        group,
                        g_params_exec,
                        gpu_grads,
                        gpu_exp_avgs,
                        gpu_exp_avg_sqs,
                        gpu_master_params,
                        gpu_scratches,
                        gpu_steps,
                        orig_indices=gpu_param_indices,
                    )
                else:
                    self._step_gpu_group(
                        group,
                        g_params_exec,
                        gpu_grads,
                        gpu_exp_avgs,
                        gpu_exp_avg_sqs,
                        gpu_master_params,
                        gpu_scratches,
                        gpu_steps,
                    )
                for p_orig, p_step in gpu_pairs:
                    if p_orig is not p_step:
                        if hasattr(p_orig, "copy_"):
                            p_orig.copy_(p_step)

        return loss

    def _step_numpy_group(
        self,
        group: Dict[str, Any],
        params: list[Any],
        grads: list[Any],
        exp_avgs: list[Any],
        exp_avg_sqs: list[Any],
        master_params: list[Any],
        scratches: list[Any],
        steps: list[int],
        param_indices: Optional[List[int]] = None,
    ) -> None:
        step_numpy_group(
            self, group, params, grads, exp_avgs, exp_avg_sqs, master_params, scratches, steps, param_indices
        )
