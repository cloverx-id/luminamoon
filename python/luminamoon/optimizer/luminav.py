from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from . import constants as _c
from .backends import prewarm_all_configs, step_gpu_group, step_numpy_group
from .buckets import build_contiguous_buckets, step_fused_buckets, sync_or_rebuild_fused_buckets
from .config import (
    BoundType,
    BufferMode,
    ExecutionMode,
    MasterWeights,
    VarianceType,
    build_rust_config,
    compute_group_sig,
    is_dual_buffer,
    validate_and_normalize_group,
)
from .migration import reject_removed_kwargs, removed_attr_message
from .standalone import ParamStateDict, StandaloneOptimizer
from .state import ensure_param_state, export_state_dict, import_state_dict
from .validation import (
    HAS_NUMPY,
    HAS_TORCH,
    _is_contiguous,
    _make_contiguous,
    _no_grad_decorator,
    _numel,
    _validate_supported_dtype,
    np,
    torch,
)
from .version import MATH_VERSION

if TYPE_CHECKING:
    from torch.optim import Optimizer as TorchOptimizer

    OptimizerBase = TorchOptimizer
    _luminamoon_rs: Any = None

    class _LuminaVMeta(type):
        def __getattr__(cls, name: str) -> Any:
            ...
else:
    if HAS_TORCH and torch is not None:
        from torch.optim import Optimizer as TorchOptimizer

        OptimizerBase = TorchOptimizer
    else:
        TorchOptimizer = None
        OptimizerBase = StandaloneOptimizer

    class _LuminaVMeta(type(OptimizerBase)):  # type: ignore[misc]
        def __getattr__(cls, name: str) -> Any:
            msg = removed_attr_message(name)
            if msg is not None:
                raise AttributeError(msg)
            raise AttributeError(f"type object '{cls.__name__}' has no attribute '{name}'")

    try:
        from luminamoon.optimizer import _luminamoon_optimizer_rs as _luminamoon_rs
    except ImportError:
        try:
            import _luminamoon_optimizer_rs as _luminamoon_rs
        except ImportError:
            _luminamoon_rs = None

logger = logging.getLogger("LuminaV")


class LuminaV(OptimizerBase, metaclass=_LuminaVMeta):
    """LuminaV Optimizer: Low-Precision Cautious-Trust-Bounded Adaptive Optimizer.

    Supports Dual-Buffer CLM (`buffer=2`), Single-Buffer TD-CLM (`buffer=1`),
    and Triangle Family tanh-free modes (`buffer="t1"`, `buffer="t2"`) across
    PyTorch (`torch.Tensor`) and standalone NumPy (`np.ndarray`) parameters.
    """

    MATH_VERSION: str = MATH_VERSION
    state: Any

    def __init__(
        self,
        params: Any,
        lr: float = _c.DEFAULT_LR,
        betas: Tuple[float, float] = _c.DEFAULT_BETAS,
        eps: float = _c.DEFAULT_EPS,
        weight_decay: float = _c.DEFAULT_WEIGHT_DECAY,
        cautious_weight_decay: bool = _c.DEFAULT_CAUTIOUS_WEIGHT_DECAY,
        variance_type: VarianceType = _c.DEFAULT_VARIANCE_TYPE,
        variance_temper: float = _c.DEFAULT_VARIANCE_TEMPER,
        tau: float = _c.DEFAULT_TAU,
        lam: float = _c.DEFAULT_LAM,
        cautious: bool = True,
        cautious_clamp_min: float = _c.DEFAULT_CAUTIOUS_CLAMP_MIN,
        buffer: BufferMode = _c.DEFAULT_BUFFER,
        stochastic_rounding: bool = True,
        bound: bool = _c.DEFAULT_BOUND,
        bound_type: BoundType = _c.DEFAULT_BOUND_TYPE,
        bound_ratio: float = _c.DEFAULT_BOUND_RATIO,
        bound_floor: float = _c.DEFAULT_BOUND_FLOOR,
        master_weights: MasterWeights = _c.DEFAULT_MASTER_WEIGHTS,
        execution: ExecutionMode = _c.DEFAULT_EXECUTION,
        fused_single_pass: bool = _c.DEFAULT_FUSED_SINGLE_PASS,
        fuse_tensors: bool = _c.DEFAULT_FUSE_TENSORS,
        bucket_cap_mb: float = _c.DEFAULT_BUCKET_CAP_MB,
        prewarm_jit: bool = _c.DEFAULT_PREWARM_JIT,
        ema_decay: Optional[Union[float, str]] = _c.DEFAULT_EMA_DECAY,
        fsdp_sync: bool = _c.DEFAULT_FSDP_SYNC,
        seed: int = _c.DEFAULT_OPTIMIZER_SEED,
        **kwargs: Any,
    ):
        if kwargs:
            reject_removed_kwargs(kwargs)
        defaults: Dict[str, Any] = {
            "lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay,
            "cautious_weight_decay": cautious_weight_decay,
            "variance_type": variance_type, "variance_temper": variance_temper,
            "tau": tau, "lam": lam, "cautious": cautious, "cautious_clamp_min": cautious_clamp_min,
            "buffer": buffer, "stochastic_rounding": stochastic_rounding,
            "bound": bound, "bound_type": bound_type, "bound_ratio": bound_ratio, "bound_floor": bound_floor,
            "master_weights": master_weights, "execution": execution,
            "fused_single_pass": fused_single_pass, "fuse_tensors": fuse_tensors,
            "bucket_cap_mb": bucket_cap_mb, "prewarm_jit": prewarm_jit,
            "ema_decay": ema_decay, "fsdp_sync": fsdp_sync, "seed": seed,
        }

        is_auto_ema = ema_decay is None or (isinstance(ema_decay, str) and ema_decay.strip().lower() == "auto")
        validate_and_normalize_group(defaults)
        if is_auto_ema:
            defaults["ema_decay"] = "auto"

        param_list = list(params)
        if not param_list:
            raise ValueError("optimizer got an empty parameter list")

        first_p = None
        for elem in param_list:
            if isinstance(elem, dict):
                grp_params = elem.get("params")
                if grp_params is not None and not isinstance(grp_params, (list, tuple)):
                    grp_params = list(grp_params)
                    elem["params"] = grp_params
                if grp_params:
                    first_p = grp_params[0]
                    break
            else:
                first_p = elem
                break
        is_torch = HAS_TORCH and torch is not None and TorchOptimizer is not None and isinstance(first_p, torch.Tensor)
        self._is_torch_mode = is_torch

        explicit_cap = bucket_cap_mb != _c.DEFAULT_BUCKET_CAP_MB or any(
            isinstance(g, dict) and "bucket_cap_mb" in g for g in param_list
        )

        if is_torch:
            super().__init__(param_list, defaults)
        else:
            self.defaults = defaults
            self.state = ParamStateDict()
            self.param_groups = []
            self._optimizer_step_pre_hooks = {}
            self._optimizer_step_post_hooks = {}
            p_groups = [{"params": param_list}] if not isinstance(param_list[0], dict) else param_list
            for pg in p_groups:
                self.add_param_group(pg)

        if not isinstance(self.state, ParamStateDict):
            self.state = ParamStateDict(self.state)
        self.seed = int(defaults["seed"])
        self.fuse_tensors: bool = bool(defaults["fuse_tensors"])
        self.bucket_cap_mb: float = float(defaults["bucket_cap_mb"])
        self.prewarm_jit: bool = bool(defaults["prewarm_jit"])
        self._fused_buckets: List[Dict[str, Any]] = []
        self._prewarmed_triton_configs: set = set()
        self._warned_triton_failure = False
        self._warned_non_contiguous = False
        self._warned_rs_missing = False

        if self._is_torch_mode and any(bool(g.get("fuse_tensors", False)) for g in self.param_groups):
            self.fuse_tensors = True
            build_contiguous_buckets(self)
        elif explicit_cap:
            warnings.warn(
                f"LuminaV: 'bucket_cap_mb' was explicitly specified ({self.bucket_cap_mb} MB) "
                "but 'fuse_tensors' is False. Bounded bucket chunking only applies when 'fuse_tensors=True'.",
                UserWarning,
                stacklevel=2,
            )

        if self._is_torch_mode and prewarm_all_configs is not None and any(
            bool(g.get("prewarm_jit", True)) for g in self.param_groups
        ):
            prewarm_all_configs(self)

    @classmethod
    def get_math_version(cls) -> str:
        """Return the latest active LuminaV optimizer math version."""
        return cls.MATH_VERSION

    @classmethod
    def help(
        cls,
        topic: str = "optimizer",
        *,
        output: Optional[Any] = None,
        print_out: bool = True,
    ) -> str:
        """Display and optionally save LuminaV optimizer parameter and API documentation."""
        from ..cli import help as _lm_help

        return _lm_help(topic=topic, output=output, print_out=print_out)

    def __getattr__(self, name: str) -> Any:
        msg = removed_attr_message(name)
        if msg is not None:
            raise AttributeError(msg)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        for name, default in self.defaults.items():
            param_group.setdefault(name, default)
        validate_and_normalize_group(param_group)
        if not getattr(self, "_is_torch_mode", False):
            params = param_group["params"]
            param_group["params"] = list(params) if hasattr(params, "__iter__") else [params]
            self.param_groups.append(param_group)
            return
        super().add_param_group(param_group)
        if hasattr(self, "_fused_buckets"):
            if bool(param_group.get("fuse_tensors", False)):
                self.fuse_tensors = True
                build_contiguous_buckets(self)
            if bool(param_group.get("prewarm_jit", True)) and prewarm_all_configs is not None:
                prewarm_all_configs(self)

    def zero_grad(self, set_to_none: bool = True) -> None:
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
        """Query active CPU SIMD ISA, CubeCL/Vulkan GPU device, and Triton AOT/JIT capabilities."""
        if _luminamoon_rs is not None:
            return _luminamoon_rs.detect_hardware_pyo3()
        return {
            "simd_backend": "PyTorch Universal Fallback",
            "triton_aot_available": False,
            "triton_arch": None,
            "cubecl_available": False,
            "cubecl_backend": None,
            "vulkan_available": False,
            "gpu_device_name": None,
            "wave32_supported": False,
            "gpu_dispatch_threshold": _c.DEFAULT_GPU_DISPATCH_THRESHOLD,
        }

    def state_dict(self) -> Dict[str, Any]:
        return export_state_dict(self, super().state_dict)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        import_state_dict(self, state_dict, super().load_state_dict)
        if not getattr(self, "_is_torch_mode", False):
            return
        self.fuse_tensors = any(bool(g.get("fuse_tensors", False)) for g in self.param_groups)
        self.prewarm_jit = any(bool(g.get("prewarm_jit", True)) for g in self.param_groups)
        if self.fuse_tensors:
            sync_or_rebuild_fused_buckets(self)
        else:
            self._fused_buckets.clear()
        if self.prewarm_jit and prewarm_all_configs is not None:
            prewarm_all_configs(self)

    def _dispatch_cpu_batch(
        self,
        group: Dict[str, Any],
        params: list[Any],
        grads: list[Any],
        exp_avgs: list[Any],
        exp_avg_sqs: list[Any],
        master_params: list[Any],
        scratches: list[Any],
        steps: list[int],
        param_indices: List[int],
    ) -> None:
        if _luminamoon_rs is not None and group["execution"] != "eager":
            _luminamoon_rs.step_pyo3(
                params, grads, exp_avgs, exp_avg_sqs, master_params, scratches,
                steps, param_indices, build_rust_config(group),
            )
            return

        is_cpu_torch = HAS_TORCH and torch is not None and isinstance(params[0], torch.Tensor)
        if is_cpu_torch and step_gpu_group is not None:
            if _luminamoon_rs is None and not self._warned_rs_missing:
                logger.info("LuminaV: Native Rust extension not loaded; executing via PyTorch eager backend.")
                self._warned_rs_missing = True
            step_gpu_group(
                self, group, params, grads, exp_avgs, exp_avg_sqs, master_params,
                scratches, steps, orig_indices=param_indices,
            )
        elif HAS_NUMPY and np is not None:
            if _luminamoon_rs is None and not self._warned_rs_missing:
                logger.info("LuminaV: Native Rust extension not loaded; executing via NumPy fallback backend.")
                self._warned_rs_missing = True
            step_numpy_group(
                self, group, params, grads, exp_avgs, exp_avg_sqs, master_params,
                scratches, steps, param_indices=param_indices,
            )
        else:
            raise RuntimeError(
                "LuminaV: Neither native Rust extension (_luminamoon_optimizer_rs), PyTorch, nor NumPy was found."
            )

    @_no_grad_decorator
    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
        grads: Optional[Any] = None,
    ) -> Optional[float]:
        """Perform a single optimization step across all parameter groups."""
        loss = None
        if closure is not None:
            if HAS_TORCH and torch is not None and hasattr(torch, "enable_grad"):
                with torch.enable_grad():
                    loss = closure()
            else:
                loss = closure()

        group_param_offset = 0
        for group in self.param_groups:
            if group.get("_validated_sig") != compute_group_sig(group):
                validate_and_normalize_group(group)

            buf_count = group["buffer"]
            fsdp_sync = bool(group["fsdp_sync"])
            if (
                self._is_torch_mode
                and bool(group.get("fuse_tensors", False))
                and not fsdp_sync
                and step_gpu_group is not None
                and step_fused_buckets(
                    self, group, grads, _luminamoon_rs, step_gpu_group, group_param_offset=group_param_offset
                )
            ):
                group_param_offset += len(group["params"])
                continue

            cpu_batch: List[Tuple[Any, Any, Any, Any, Any, Any, int, int]] = []
            gpu_batch: List[Tuple[Any, Any, Any, Any, Any, Any, int, int]] = []

            for idx, p in enumerate(group["params"]):
                g = None
                global_grad_idx = group_param_offset + idx
                if grads is not None:
                    if isinstance(grads, (list, tuple)) and global_grad_idx < len(grads):
                        g = grads[global_grad_idx]
                    elif isinstance(grads, dict):
                        try:
                            g = grads.get(p)
                        except TypeError:
                            g = grads.get(id(p))
                if g is None:
                    g = getattr(p, "grad", None)
                if g is None:
                    continue
                if getattr(g, "is_sparse", False):
                    raise RuntimeError("LuminaV does not support sparse gradients.")

                _validate_supported_dtype(p, "param")
                _validate_supported_dtype(g, "grad")
                if _numel(p) == 0 and not fsdp_sync:
                    continue
                if not _is_contiguous(g):
                    g = _make_contiguous(g)

                is_torch_tensor = HAS_TORCH and torch is not None and isinstance(p, torch.Tensor)
                dev_type = p.device.type if (is_torch_tensor and hasattr(p, "device")) else "cpu"

                if dev_type == "cpu" and not (is_torch_tensor and fsdp_sync):
                    if hasattr(p, "is_contiguous") and not p.is_contiguous():
                        raise ValueError(
                            f"LuminaV requires contiguous parameters for native zero-copy execution. "
                            f"Call .contiguous() on parameter {idx}."
                        )
                    if hasattr(p, "flags") and not getattr(getattr(p, "flags", None), "c_contiguous", True):
                        raise ValueError(
                            f"LuminaV requires contiguous parameters for native zero-copy execution. "
                            f"Call np.ascontiguousarray() on parameter {idx}."
                        )

                state = ensure_param_state(self, p, p, group, is_torch_tensor)
                curr_step = state["step"]
                step_val = (int(curr_step.item()) if hasattr(curr_step, "item") else int(curr_step)) + 1
                state["step"] = step_val

                sq = state.get("exp_avg_sq") if is_dual_buffer(buf_count) else None
                entry = (p, g, state["exp_avg"], sq, state.get("master_param"), state["scratch"], step_val, idx)
                if dev_type == "cpu" and not (is_torch_tensor and fsdp_sync):
                    cpu_batch.append(entry)
                else:
                    gpu_batch.append(entry)

            if cpu_batch:
                p_l, g_l, m_l, v_l, mp_l, sc_l, st_l, idx_l = map(list, zip(*cpu_batch))
                self._dispatch_cpu_batch(group, p_l, g_l, m_l, v_l, mp_l, sc_l, st_l, idx_l)

            if gpu_batch and step_gpu_group is not None:
                p_l, g_l, m_l, v_l, mp_l, sc_l, st_l, idx_l = map(list, zip(*gpu_batch))
                step_gpu_group(self, group, p_l, g_l, m_l, v_l, mp_l, sc_l, st_l, orig_indices=idx_l)

            group_param_offset += len(group["params"])

        return loss
