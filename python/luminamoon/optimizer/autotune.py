from __future__ import annotations

from collections.abc import Iterable, Mapping
import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import constants as _c
from ._lr_probe import filter_kwargs_for_optimizer, probe_hyperparameters_empirical
from ._model_profile import MemoryPriority, ModelProfile, profile_model, recommend_hyperparameters
from .backends._sampling import materialize_model_or_params
from .backends.autotune import TuneStrategy, benchmark_backends, reset_autotune_cache
from .config import ExecutionMode
from .migration import reject_removed_kwargs

logger = logging.getLogger("LuminaV.AutoTune")


class _UnsetType:
    """Sentinel distinguishing omitted constructor arguments from explicit default-equal values."""

    def __repr__(self) -> str:
        return "<default>"


_UNSET = _UnsetType()

_TUNABLE_DEFAULT_MAP: Dict[str, Any] = {
    "lr": _c.DEFAULT_LR, "betas": _c.DEFAULT_BETAS, "eps": _c.DEFAULT_EPS,
    "weight_decay": _c.DEFAULT_WEIGHT_DECAY, "cautious_weight_decay": _c.DEFAULT_CAUTIOUS_WEIGHT_DECAY,
    "variance_type": _c.DEFAULT_VARIANCE_TYPE, "variance_temper": _c.DEFAULT_VARIANCE_TEMPER,
    "momentum_type": _c.DEFAULT_MOMENTUM_TYPE,
    "tau": _c.DEFAULT_TAU, "lam": _c.DEFAULT_LAM, "cautious": _c.DEFAULT_CAUTIOUS,
    "cautious_clamp_min": _c.DEFAULT_CAUTIOUS_CLAMP_MIN, "buffer": _c.DEFAULT_BUFFER,
    "stochastic_rounding": _c.DEFAULT_STOCHASTIC_ROUNDING, "bound": _c.DEFAULT_BOUND,
    "bound_type": _c.DEFAULT_BOUND_TYPE, "bound_ratio": _c.DEFAULT_BOUND_RATIO,
    "bound_floor": _c.DEFAULT_BOUND_FLOOR, "master_weights": _c.DEFAULT_MASTER_WEIGHTS,
    "execution": _c.DEFAULT_EXECUTION, "fused_single_pass": _c.DEFAULT_FUSED_SINGLE_PASS,
    "fuse_tensors": _c.DEFAULT_FUSE_TENSORS, "bucket_cap_mb": _c.DEFAULT_BUCKET_CAP_MB,
    "prewarm_jit": _c.DEFAULT_PREWARM_JIT, "ema_decay": _c.DEFAULT_EMA_DECAY,
}


def _to_json_safe(obj: Any) -> Any:
    """Convert nested metadata into weights_only- and JSON-safe primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, Mapping):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    return str(obj)


class AutoTune:
    """Two-mode hyperparameter and runtime tuner for `LuminaV` and PyTorch optimizers.

    Mode 1 (`AutoTune(model)`): architecture/memory hyperparameter priors + steady-state backend benchmark.
    Mode 2 (`AutoTune(model, loss_fn=...)`): adds full-model empirical LR & gradient-spike probing,
    restoring parameters, module buffers, gradients, training mode, and RNG states unchanged.
    Use `LuminaV(model.parameters(), tuning=tune)` or `tune.build(torch.optim.AdamW, model.parameters())`.
    """

    def __init__(
        self, model_or_params: Any, *, loss_fn: Optional[Callable[..., Any]] = None,
        optimizer_cls: Optional[Any] = None, tune_params: Optional[Iterable[str]] = None,
        memory_priority: MemoryPriority = "balanced", tune_hardware: Optional[bool] = None,
        steps: int = 7, measure_last: int = 2, lr_steps: int = 5,
        lr_multipliers: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0), safety_factor: float = 0.65,
        strategy: TuneStrategy = "auto", max_memory_mb: float = 128.0, tune_fuse_tensors: bool = True,
        use_cache: bool = True, candidates: Optional[List[str]] = None, verbose: bool = True,
        **optimizer_kwargs: Any,
    ) -> None:
        from .luminav import LuminaV as _LuminaV

        target_cls = _LuminaV if optimizer_cls is None else optimizer_cls
        is_luminav_target = target_cls is _LuminaV
        if is_luminav_target and optimizer_kwargs:
            known_lum = set(_TUNABLE_DEFAULT_MAP.keys()) | {"fsdp_sync", "seed"}
            extra_unknown = {k: v for k, v in optimizer_kwargs.items() if k not in known_lum}
            if extra_unknown:
                reject_removed_kwargs(extra_unknown)
        target_input = materialize_model_or_params(model_or_params)
        self.optimizer_name: str = getattr(target_cls, "__name__", str(target_cls))
        self.profile: ModelProfile = profile_model(target_input)

        # Stage 1: Architecture, precision, and VRAM hyperparameter priors
        hp_cfg, sources, notes = recommend_hyperparameters(
            self.profile,
            memory_priority=memory_priority,
            user_overrides=optimizer_kwargs,
        )
        self.notes: List[str] = list(notes)
        self.sources: Dict[str, str] = dict(sources)

        # Stage 2: Optional empirical full-model LR & gradient-stability probe
        self.lr_probe_results: Optional[Dict[str, Any]] = None
        if loss_fn is not None:
            probe_res = probe_hyperparameters_empirical(
                target_input,
                loss_fn,
                profile=self.profile,
                base_config=hp_cfg,
                optimizer_cls=target_cls,
                lr_steps=lr_steps,
                lr_multipliers=lr_multipliers,
                safety_factor=safety_factor,
            )
            self.lr_probe_results = probe_res
            if probe_res.get("status") == "skipped_low_vram":
                self.notes.append(str(probe_res.get("reason", "Skipped empirical probe due to low VRAM.")))
            else:
                if "lr" not in optimizer_kwargs and probe_res.get("status") in ("ok", "conservative_fallback"):
                    hp_cfg["lr"] = float(probe_res["selected_lr"])
                    self.sources["lr"] = "empirical"
                if (
                    is_luminav_target
                    and "variance_type" not in optimizer_kwargs
                    and probe_res.get("status") in ("ok", "conservative_fallback")
                ):
                    hp_cfg["variance_type"] = str(probe_res["selected_variance_type"])
                    self.sources["variance_type"] = "empirical"
                    if probe_res.get("spike_detected"):
                        if hp_cfg["variance_type"] == "rtv":
                            self.notes.append(
                                f"Empirical probe detected gradient spikes "
                                f"(max_spike={probe_res['max_grad_spike_ratio']:.1f}x); enabled variance_type='rtv'."
                            )
                        else:
                            self.notes.append(
                                f"Empirical probe detected gradient spikes "
                                f"(max_spike={probe_res['max_grad_spike_ratio']:.1f}x); single-buffer mode retains "
                                "variance_type='standard' (consider buffer='t2' with variance_type='rtv' if spikes persist)."
                            )

        # Stage 3: Steady-state tail-step backend benchmark (LuminaV default; off by default for external optimizers)
        run_hw_bench = is_luminav_target if tune_hardware is None else bool(tune_hardware)
        self.results: Optional[Dict[str, Any]] = None
        self.execution: Optional[ExecutionMode] = None
        self.fuse_tensors: bool = bool(optimizer_kwargs.get("fuse_tensors", _c.DEFAULT_FUSE_TENSORS))
        self.strategy: str = "skipped"
        self.winner_label: str = "none"
        self.timings_ms: Dict[str, float] = {}
        self.per_step_ms: Dict[str, List[float]] = {}
        self.errors: Dict[str, str] = {}
        self.steps: int = steps
        self.measure_last: int = measure_last
        self.sampled_params: int = self.profile.param_tensors
        self.sampled_bytes: int = self.profile.param_bytes

        if run_hw_bench:
            bench_kwargs = {
                k: v for k, v in hp_cfg.items()
                if k in ("lr", "betas", "weight_decay", "cautious_weight_decay", "variance_type",
                         "momentum_type", "buffer", "stochastic_rounding", "bound_type", "bound_ratio",
                         "bound_floor", "master_weights")
            }
            for extra_k, extra_v in optimizer_kwargs.items():
                if extra_k not in ("execution", "fuse_tensors", "tuning", "fsdp_sync"):
                    bench_kwargs[extra_k] = extra_v
            res = benchmark_backends(
                target_input,
                steps=steps,
                measure_last=measure_last,
                strategy=strategy,
                max_memory_mb=max_memory_mb,
                tune_fuse_tensors=tune_fuse_tensors,
                use_cache=use_cache,
                candidates=candidates,
                **bench_kwargs,
            )
            self.results = res
            self.execution = res["execution"]
            self.fuse_tensors = bool(optimizer_kwargs.get("fuse_tensors", res["fuse_tensors"]))
            self.strategy = str(res["strategy"])
            self.winner_label = str(res["winner_label"])
            self.timings_ms = dict(res["timings_ms"])
            self.per_step_ms = {k: list(v) for k, v in res["per_step_ms"].items()}
            self.errors = dict(res["errors"])
            self.steps = int(res["steps"])
            self.measure_last = int(res["measure_last"])
            self.sampled_params = int(res["sampled_params"])
            self.sampled_bytes = int(res["sampled_bytes"])

        # Assemble full tuned configuration (keeping execution='auto' portable while pre-seeding measured winner)
        full_cfg: Dict[str, Any] = dict(hp_cfg)
        if is_luminav_target:
            full_cfg["execution"] = optimizer_kwargs.get("execution", _c.DEFAULT_EXECUTION)
            full_cfg["fuse_tensors"] = self.fuse_tensors
            full_cfg["fused_single_pass"] = bool(
                optimizer_kwargs.get("fused_single_pass", _c.DEFAULT_FUSED_SINGLE_PASS)
            )
            full_cfg["prewarm_jit"] = bool(
                optimizer_kwargs.get("prewarm_jit", False if run_hw_bench else _c.DEFAULT_PREWARM_JIT)
            )
            self.sources["execution"] = "user" if "execution" in optimizer_kwargs else (
                f"measured->{self.execution}" if self.execution is not None else "default"
            )
            self.sources["fuse_tensors"] = "user" if "fuse_tensors" in optimizer_kwargs else (
                "measured" if run_hw_bench else "default"
            )
            self.sources["fused_single_pass"] = "user" if "fused_single_pass" in optimizer_kwargs else "default"
            self.sources["prewarm_jit"] = "user" if "prewarm_jit" in optimizer_kwargs else "runtime"
            for extra_k, extra_v in optimizer_kwargs.items():
                if extra_k in _TUNABLE_DEFAULT_MAP and extra_k not in full_cfg:
                    full_cfg[extra_k] = extra_v
                    self.sources[extra_k] = "user"
        else:
            full_cfg = filter_kwargs_for_optimizer(target_cls, full_cfg, overrides=optimizer_kwargs)
            for k in full_cfg:
                self.sources.setdefault(k, "user" if k in optimizer_kwargs else "prior")

        self.full_config: Dict[str, Any] = dict(full_cfg)
        if tune_params is not None:
            allowed = set(tune_params)
            unknown = allowed - set(full_cfg.keys()) - set(_TUNABLE_DEFAULT_MAP.keys())
            if unknown:
                raise ValueError(f"Unsupported key(s) in tune_params: {sorted(unknown)}.")
            self.config: Dict[str, Any] = {k: v for k, v in full_cfg.items() if k in allowed}
        else:
            self.config = dict(full_cfg)

        text = self.summary()
        logger.info("\n%s", text)
        if verbose:
            print(text)

    @property
    def benchmark_results(self) -> Optional[Dict[str, Any]]:
        """Alias for backend steady-state benchmark results (None when `tune_hardware=False`)."""
        return self.results

    def for_optimizer(self, optimizer_cls: Any, **overrides: Any) -> Dict[str, Any]:
        """Return kwargs from `self.config` compatible with any PyTorch optimizer class (e.g. `AdamW`, `SGD`)."""
        return filter_kwargs_for_optimizer(optimizer_cls, self.config, overrides=overrides)

    def build(self, optimizer_cls: Any, params: Any, **overrides: Any) -> Any:
        """Instantiate `optimizer_cls(params, ...)` with compatible tuned parameters and optional overrides."""
        from .luminav import LuminaV as _LuminaV

        if optimizer_cls is _LuminaV:
            return _LuminaV(params, tuning=self, **overrides)
        kwargs = self.for_optimizer(optimizer_cls, **overrides)
        return optimizer_cls(params, **kwargs)

    def with_overrides(self, **overrides: Any) -> Dict[str, Any]:
        """Return a copy of `self.config` with explicit user overrides applied."""
        merged = dict(self.config)
        merged.update(overrides)
        return merged

    def to_metadata(self) -> Dict[str, Any]:
        """Serialize tuning profile, parameter sources, and benchmark measurements as JSON/weights_only-safe primitives."""
        return _to_json_safe(
            {
                "optimizer_target": self.optimizer_name,
                "tuning_applied": True,
                "profile": self.profile.to_dict(),
                "tuned_config": dict(self.config),
                "hyperparameter_sources": {k: self.sources.get(k, "prior") for k in self.config},
                "lr_probe": self.lr_probe_results,
                "backend_benchmark": self.results,
                "resolved_execution": self.execution,
                "notes": list(self.notes),
            }
        )

    def summary(self) -> str:
        """Format a summary separating empirical measurements and architectural priors."""
        mb = self.profile.param_bytes / (1024.0 * 1024.0)
        p_str = (
            f"{self.profile.total_params / 1e6:.2f}M"
            if self.profile.total_params >= 1_000_000
            else f"{self.profile.total_params:,}"
        )
        lines = [
            f"LuminaV AutoTune Summary (target={self.optimizer_name})",
            "___",
            (
                f"  Model Profile   : {self.profile.architecture} | {p_str} params ({mb:.2f} MiB) | "
                f"dtype={self.profile.dominant_dtype} | width={self.profile.median_width}, "
                f"depth={self.profile.depth} | device={self.profile.device_type}"
            ),
        ]
        if self.lr_probe_results is not None:
            st = self.lr_probe_results.get("status", "unknown")
            n_trials = len(self.lr_probe_results.get("trials", []))
            lr_val = self.config.get("lr", self.full_config.get("lr", self.lr_probe_results.get("selected_lr")))
            lr_disp = f"{lr_val:.3g}" if isinstance(lr_val, (int, float)) else repr(lr_val)
            lines.append(
                f"  Empirical Probe : status={st} ({self.lr_probe_results.get('lr_steps', 0)} steps x {n_trials} candidates "
                f"-> lr={lr_disp})"
            )
        else:
            lines.append("  Empirical Probe : skipped (no loss_fn supplied; using architecture & memory priors)")

        if self.results is not None and self.execution is not None:
            win_ms = self.timings_ms.get(self.winner_label, 0.0)
            lines.append(
                f"  Backend Bench   : strategy={self.strategy} | {self.steps} steps (last {self.measure_last}) -> "
                f"execution='{self.config.get('execution', 'auto')}' (resolved: '{self.execution}', {win_ms:.3f} ms/step), "
                f"fuse_tensors={self.fuse_tensors}"
            )
        else:
            lines.append("  Backend Bench   : skipped")

        lines.append("")
        lines.append("  Active Setup (Parameter -> Value [Source]):")
        for key, val in self.config.items():
            src = self.sources.get(key, "prior")
            val_fmt = f"{val:.3g}" if isinstance(val, float) else repr(val)
            lines.append(f"    {key:<22} = {val_fmt:<16} [{src}]")

        if self.timings_ms:
            lines.append("")
            lines.append(f"  Backend Timings (Steady-State ms/step, tail={self.measure_last}):")
            for label, ms in sorted(self.timings_ms.items(), key=lambda kv: kv[1]):
                tag = " [fastest]" if label == self.winner_label else ""
                tail_str = ", ".join(f"{t:.3f}" for t in self.per_step_ms.get(label, []))
                lines.append(f"    {label:<15} {ms:10.3f}{tag:<12}   [{tail_str}]")
            for label, err in self.errors.items():
                lines.append(f"    {label:<15}     FAILED ({err})")

        if self.notes:
            lines.append("")
            lines.append("  Notes:")
            for note in self.notes:
                lines.append(f"    - {note}")
        return "\n".join(lines)

    @staticmethod
    def clear_cache() -> None:
        reset_autotune_cache()

    def __repr__(self) -> str:
        lr_val = self.config.get("lr", self.full_config.get("lr"))
        win_ms = self.timings_ms.get(self.winner_label, 0.0) if self.timings_ms else 0.0
        return (
            f"AutoTune(target={self.optimizer_name!r}, arch={self.profile.architecture!r}, "
            f"lr={lr_val}, execution={self.execution!r}, fuse_tensors={self.fuse_tensors}, ms={win_ms:.3f})"
        )


def apply_tuning_to_defaults(
    defaults: Dict[str, Any],
    tuning: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[ExecutionMode]]:
    """Apply `AutoTune` (or dict) overrides onto omitted `LuminaV` defaults while preserving all explicit args."""
    if tuning is None:
        for key, def_val in _TUNABLE_DEFAULT_MAP.items():
            if defaults.get(key) is _UNSET:
                defaults[key] = def_val
        return None, None, None
    if isinstance(tuning, AutoTune):
        cfg: Mapping[str, Any] = tuning.config
        results: Optional[Dict[str, Any]] = tuning.results
        meta: Dict[str, Any] = tuning.to_metadata()
        preseeded_exec: Optional[ExecutionMode] = tuning.execution
        base_sources: Dict[str, str] = dict(meta.get("hyperparameter_sources", {}))
    elif isinstance(tuning, Mapping):
        cfg = tuning
        results = None
        meta = _to_json_safe({
            "optimizer_target": "LuminaV", "tuning_applied": True, "tuned_config": dict(tuning),
            "hyperparameter_sources": {str(k): "user_dict" for k in tuning}, "notes": [],
        })
        preseeded_exec = None
        base_sources = {str(k): "user_dict" for k in tuning}
    else:
        raise TypeError(
            f"tuning must be an AutoTune instance, dict, or None, got {type(tuning).__name__}."
        )

    effective_sources = dict(base_sources)
    for key, val in cfg.items():
        if key not in defaults:
            raise ValueError(f"Unsupported key in tuning config: {key!r}.")
        if defaults[key] is _UNSET:
            defaults[key] = val
        else:
            effective_sources[key] = "user_override"

    for key, def_val in _TUNABLE_DEFAULT_MAP.items():
        if defaults.get(key) is _UNSET:
            defaults[key] = def_val

    meta["effective_config"] = _to_json_safe(
        {k: defaults[k] for k in _TUNABLE_DEFAULT_MAP if k in defaults}
    )
    meta["hyperparameter_sources"] = effective_sources
    return results, meta, preseeded_exec


def build_tuning_state_metadata(opt: Any) -> Dict[str, Any]:
    """Build a JSON- and weights_only-safe tuning & execution metadata block for `state_dict()`."""
    raw_info = getattr(opt, "tuning_info", None)
    meta: Dict[str, Any] = dict(copy.deepcopy(raw_info)) if isinstance(raw_info, dict) else {}
    groups = getattr(opt, "param_groups", [])
    primary_cfg = (
        {k: groups[0][k] for k in _TUNABLE_DEFAULT_MAP if k in groups[0]}
        if groups
        else {k: opt.defaults[k] for k in _TUNABLE_DEFAULT_MAP if k in getattr(opt, "defaults", {})}
    )
    meta["math_version"] = str(getattr(opt, "MATH_VERSION", "2.0.0"))
    meta.setdefault("tuning_applied", getattr(opt, "tuning", None) is not None)
    meta["effective_config"] = primary_cfg
    meta["resolved_execution"] = getattr(opt, "resolved_execution", None)
    meta["last_backend"] = getattr(opt, "last_backend", None)
    if getattr(opt, "benchmark_results", None) is not None:
        meta["backend_benchmark"] = opt.benchmark_results
    return _to_json_safe(meta)


def restore_tuning_state_metadata(opt: Any, state_dict: Mapping[str, Any]) -> None:
    """Restore saved tuning metadata into `opt` while clearing host-local auto-exec cache for portability."""
    saved = state_dict.get("luminav_tuning")
    if isinstance(saved, Mapping):
        clean = _to_json_safe(dict(saved))
        opt.tuning_info = clean
        if isinstance(clean.get("backend_benchmark"), dict):
            opt.benchmark_results = dict(clean["backend_benchmark"])
    # Reset per-host auto execution resolution so cross-machine checkpoints re-benchmark on new hardware.
    if hasattr(opt, "_resolved_auto_exec") and isinstance(opt._resolved_auto_exec, dict):
        opt._resolved_auto_exec.clear()
    opt._tuned_auto_exec = None


__all__ = [
    "AutoTune", "MemoryPriority", "ModelProfile", "TuneStrategy",
    "apply_tuning_to_defaults", "build_tuning_state_metadata",
    "filter_kwargs_for_optimizer", "restore_tuning_state_metadata",
]
