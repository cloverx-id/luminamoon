from __future__ import annotations

import logging
import statistics
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, cast

from ..config import ExecutionMode
from ..validation import HAS_TORCH, _is_contiguous, _is_torch_compiling, _validate_supported_dtype, torch
from ._sampling import (
    TuneStrategy,
    build_shared_read_only_grads,
    check_finite,
    empty_device_cache,
    extract_params,
    is_dtensor,
    param_nbytes,
    resolve_strategy,
    sample_representative_clones,
    shape_signature,
    sync_device,
)
from .probe import probe_triton_runtime

logger = logging.getLogger("LuminaV.AutoExec")
_AUTOTUNE_LOCK = threading.RLock()
_AUTOTUNE_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


def reset_autotune_cache() -> None:
    """Clear cached multi-step backend benchmark winners."""
    with _AUTOTUNE_LOCK:
        _AUTOTUNE_CACHE.clear()


def _candidate_specs(
    params: List[Any],
    *,
    has_rust: bool,
    hw: Dict[str, Any],
    deterministic: bool,
    tune_fuse_tensors: bool,
    explicit_candidates: Optional[List[str]],
) -> List[Tuple[str, ExecutionMode, bool]]:
    first = params[0]
    is_torch = HAS_TORCH and torch is not None and isinstance(first, torch.Tensor)
    has_dtensor = any(is_dtensor(p) for p in params)
    allow_fuse = bool(tune_fuse_tensors and is_torch and not has_dtensor and len(params) >= 4)

    execs: List[ExecutionMode]
    if explicit_candidates is not None:
        execs = [cast(ExecutionMode, c) for c in explicit_candidates]
    elif is_torch and first.device.type in ("cuda", "xpu"):
        execs = ["eager"]
        if not deterministic:
            ok, _ = probe_triton_runtime(first.device)
            if ok:
                execs.insert(0, "triton")
    else:
        execs = []
        if has_rust:
            if hw.get("triton_aot_available", False) and not deterministic:
                execs.append("triton")
            if hw.get("cubecl_available", False):
                execs.append("cubecl")
            execs.extend(["simd", "scalar"])
        execs.append("eager")

    specs: List[Tuple[str, ExecutionMode, bool]] = []
    for ex in execs:
        specs.append((ex, ex, False))
        if allow_fuse:
            specs.append((f"{ex}+fuse", ex, True))
    return specs


def _broadcast_if_distributed(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not (HAS_TORCH and torch is not None and hasattr(torch, "distributed")):
        assert payload is not None
        return payload
    dist = torch.distributed
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        assert payload is not None
        return payload
    box: List[Any] = [payload]
    dist.broadcast_object_list(box, src=0)
    return dict(box[0])


def _run_candidate_steps(
    lum_mod: Any,
    run_params: List[Any],
    run_grads: List[Any],
    *,
    is_torch: bool,
    device: Any,
    execution: ExecutionMode,
    fuse_tensors: bool,
    steps: int,
    measure_last: int,
    clean_kwargs: Dict[str, Any],
) -> List[float]:
    if is_torch:
        for p, g in zip(run_params, run_grads):
            p.grad = g
    opt = lum_mod.LuminaV(run_params, execution=execution, fuse_tensors=fuse_tensors, **clean_kwargs)
    try:
        for _ in range(steps - measure_last):
            if is_torch:
                opt.step()
            else:
                opt.step(grads=run_grads)

        tail_times: List[float] = []
        for _ in range(measure_last):
            sync_device(device)
            t0 = time.perf_counter()
            if is_torch:
                opt.step()
            else:
                opt.step(grads=run_grads)
            sync_device(device)
            tail_times.append((time.perf_counter() - t0) * 1000.0)
        if not check_finite(run_params):
            raise RuntimeError("Non-finite parameter values produced during benchmark")
        return tail_times
    finally:
        del opt
        empty_device_cache(device)


def benchmark_backends(
    model_or_params: Any,
    *,
    steps: int = 7,
    measure_last: int = 2,
    strategy: TuneStrategy = "auto",
    max_memory_mb: float = 128.0,
    tune_fuse_tensors: bool = False,
    use_cache: bool = False,
    distributed_sync: bool = True,
    candidates: Optional[List[str]] = None,
    **optimizer_kwargs: Any,
) -> Dict[str, Any]:
    """Benchmark candidate backends over `steps` steps, timing only the last `measure_last` steps."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 3:
        raise ValueError(f"steps must be an int >= 3 (recommended 5-10), got {steps!r}.")
    if not isinstance(measure_last, int) or isinstance(measure_last, bool) or not (1 <= measure_last < steps):
        raise ValueError(f"measure_last must satisfy 1 <= measure_last < steps ({steps}), got {measure_last!r}.")
    if strategy not in ("auto", "full", "budgeted"):
        raise ValueError(f"Invalid strategy: {strategy!r}. Expected 'auto', 'full', or 'budgeted'.")
    if max_memory_mb <= 0:
        raise ValueError(f"max_memory_mb must be > 0, got {max_memory_mb!r}.")

    from .. import luminav as lum_mod

    flat = extract_params(model_or_params)
    if not flat:
        raise ValueError("benchmark_backends got an empty parameter list.")

    first = flat[0]
    is_torch = HAS_TORCH and torch is not None and isinstance(first, torch.Tensor)
    device = first.device if is_torch else None
    dev_str = str(device) if device is not None else "numpy-cpu"
    deterministic = bool(
        is_torch
        and hasattr(torch, "are_deterministic_algorithms_enabled")
        and torch.are_deterministic_algorithms_enabled()
    )

    is_dist_non_zero = bool(
        distributed_sync
        and HAS_TORCH and torch is not None and hasattr(torch, "distributed")
        and torch.distributed.is_available() and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1 and torch.distributed.get_rank() != 0
    )
    if is_dist_non_zero:
        return _broadcast_if_distributed(None)

    clean_kwargs = {
        k: v for k, v in optimizer_kwargs.items()
        if k not in ("execution", "fuse_tensors", "tuning", "fsdp_sync")
    }
    clean_kwargs["fsdp_sync"] = False
    clean_kwargs.setdefault("prewarm_jit", False)

    resolved_strategy = resolve_strategy(
        flat, strategy,
        buffer_mode=clean_kwargs.get("buffer", 2),
        master_weights=str(clean_kwargs.get("master_weights", "none")),
    )
    hw = lum_mod.LuminaV.detect_hardware()
    has_rust = lum_mod._luminamoon_rs is not None
    specs = _candidate_specs(
        flat, has_rust=has_rust, hw=hw, deterministic=deterministic,
        tune_fuse_tensors=tune_fuse_tensors, explicit_candidates=candidates,
    )
    cache_key = (
        dev_str, shape_signature(flat), resolved_strategy, tuple(specs),
        clean_kwargs.get("buffer", 2), clean_kwargs.get("bound", True),
        clean_kwargs.get("bound_type", "coordinate"), clean_kwargs.get("master_weights", "none"),
        clean_kwargs.get("fused_single_pass", True), steps, measure_last, deterministic, has_rust,
    )
    with _AUTOTUNE_LOCK:
        if use_cache and cache_key in _AUTOTUNE_CACHE:
            cached = dict(_AUTOTUNE_CACHE[cache_key])
            return _broadcast_if_distributed(cached) if distributed_sync else cached

        timings_ms: Dict[str, float] = {}
        per_step_ms: Dict[str, List[float]] = {}
        errors: Dict[str, str] = {}
        label_to_cfg: Dict[str, Tuple[ExecutionMode, bool]] = {}

        if resolved_strategy == "full":
            sampled_params = len(flat)
            sampled_bytes = sum(param_nbytes(p) for p in flat)
            cpu_snapshots = [
                p.detach().cpu().clone() if (HAS_TORCH and torch is not None and isinstance(p, torch.Tensor)) else p.copy()
                for p in flat
            ]
            orig_grads = [getattr(p, "grad", None) for p in flat] if is_torch else []
            orig_devices = [p.device for p in flat] if is_torch else []
            orig_dtypes = [p.dtype for p in flat] if is_torch else []
            shared_grads = build_shared_read_only_grads(flat)
            try:
                for label, ex_mode, fuse_flag in specs:
                    label_to_cfg[label] = (ex_mode, fuse_flag)
                    try:
                        tail = _run_candidate_steps(
                            lum_mod, flat, shared_grads, is_torch=is_torch, device=device,
                            execution=ex_mode, fuse_tensors=fuse_flag, steps=steps,
                            measure_last=measure_last, clean_kwargs=clean_kwargs,
                        )
                        per_step_ms[label] = tail
                        timings_ms[label] = float(statistics.median(tail))
                    except Exception as exc:
                        err_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
                        errors[label] = err_line
                    finally:
                        for i, p in enumerate(flat):
                            if HAS_TORCH and torch is not None and isinstance(p, torch.Tensor):
                                # Always copy=True so CPU snapshots are never aliased to mutated p.data storage.
                                if fuse_flag:
                                    p.data = cpu_snapshots[i].to(device=orig_devices[i], dtype=orig_dtypes[i], copy=True)
                                else:
                                    p.data.copy_(cpu_snapshots[i])
                                p.grad = orig_grads[i]
                            else:
                                p[...] = cpu_snapshots[i]
                        empty_device_cache(device)
            finally:
                del shared_grads, cpu_snapshots
                empty_device_cache(device)
        else:
            base_clones, base_grads, sampled_bytes = sample_representative_clones(flat, max_memory_mb)
            sampled_params = len(base_clones)
            for label, ex_mode, fuse_flag in specs:
                label_to_cfg[label] = (ex_mode, fuse_flag)
                run_params = [
                    torch.nn.Parameter(p_ref.detach().clone()) if (HAS_TORCH and torch is not None and isinstance(p_ref, torch.Tensor)) else p_ref.copy()
                    for p_ref in base_clones
                ]
                run_grads = [
                    g_ref.clone() if (HAS_TORCH and torch is not None and isinstance(g_ref, torch.Tensor)) else g_ref.copy()
                    for g_ref in base_grads
                ]
                try:
                    tail = _run_candidate_steps(
                        lum_mod, run_params, run_grads, is_torch=is_torch, device=device,
                        execution=ex_mode, fuse_tensors=fuse_flag, steps=steps,
                        measure_last=measure_last, clean_kwargs=clean_kwargs,
                    )
                    per_step_ms[label] = tail
                    timings_ms[label] = float(statistics.median(tail))
                except Exception as exc:
                    err_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
                    errors[label] = err_line
                finally:
                    del run_params, run_grads
                    empty_device_cache(device)

        if not timings_ms:
            raise RuntimeError(f"LuminaV: all candidate backends failed during benchmark: {errors}")

        winner_label = min(timings_ms, key=lambda k: timings_ms[k])
        win_exec, win_fuse = label_to_cfg[winner_label]
        result: Dict[str, Any] = {
            "winner": win_exec,
            "winner_label": winner_label,
            "execution": win_exec,
            "fuse_tensors": win_fuse,
            "strategy": resolved_strategy,
            "timings_ms": timings_ms,
            "per_step_ms": per_step_ms,
            "steps": steps,
            "measure_last": measure_last,
            "errors": errors,
            "sampled_params": sampled_params,
            "sampled_bytes": sampled_bytes,
        }
        _AUTOTUNE_CACHE[cache_key] = dict(result)
    return _broadcast_if_distributed(result) if distributed_sync else result


def _eligible_auto_params(params: List[Any]) -> List[Any]:
    out: List[Any] = []
    for p in params:
        if param_nbytes(p) <= 0:
            continue
        try:
            _validate_supported_dtype(p, "param")
        except Exception:
            continue
        if not is_dtensor(p):
            dev_type = p.device.type if (HAS_TORCH and torch is not None and isinstance(p, torch.Tensor)) else "cpu"
            if dev_type == "cpu" and not _is_contiguous(p):
                continue
        out.append(p)
    return out


def resolve_auto_execution(opt: Any, group: Dict[str, Any]) -> Optional[ExecutionMode]:
    """Resolve `execution='auto'` via cached steady-state tail-step benchmarking on first active step."""
    if str(group.get("execution", "auto")) != "auto" or _is_torch_compiling():
        return None

    from .. import luminav as lum_mod

    raw_params = list(group.get("params", []))
    is_dist = bool(
        HAS_TORCH and torch is not None and hasattr(torch, "distributed")
        and torch.distributed.is_available() and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    )
    rank = int(torch.distributed.get_rank()) if (is_dist and torch is not None) else 0
    eligible = _eligible_auto_params(raw_params)

    first = eligible[0] if eligible else None
    is_torch = HAS_TORCH and torch is not None and isinstance(first, torch.Tensor)
    dev_str = str(first.device) if is_torch and first is not None else "numpy-cpu"
    dev_type = first.device.type if is_torch and first is not None else "cpu"
    deterministic = bool(
        is_torch and torch is not None
        and hasattr(torch, "are_deterministic_algorithms_enabled")
        and torch.are_deterministic_algorithms_enabled()
    )
    has_rust = lum_mod._luminamoon_rs is not None
    cache_key = (
        id(group), dev_str, deterministic, has_rust,
        group.get("buffer"), group.get("master_weights"), group.get("bound"), group.get("bound_type"),
    )
    resolved_map: Dict[Tuple[Any, ...], ExecutionMode] = opt._resolved_auto_exec
    if cache_key in resolved_map:
        return resolved_map[cache_key]

    if is_dist and rank != 0:
        bcast = _broadcast_if_distributed(None)
        win = bcast.get("execution")
        if win in ("triton", "cubecl", "simd", "eager", "scalar"):
            resolved_map[cache_key] = cast(ExecutionMode, win)
            if isinstance(bcast.get("timings_ms"), dict):
                opt.benchmark_results = bcast
            return resolved_map[cache_key]
        return None

    if not eligible:
        if is_dist:
            _broadcast_if_distributed({"execution": None})
        return None

    hw = lum_mod.LuminaV.detect_hardware()
    specs = _candidate_specs(
        eligible, has_rust=has_rust, hw=hw, deterministic=deterministic,
        tune_fuse_tensors=False, explicit_candidates=None,
    )
    available_execs = {ex for _, ex, _ in specs}
    # Pre-seeded winner from AutoTune(model) avoids redundant re-benchmarking only if live on this host.
    preseeded = getattr(opt, "_tuned_auto_exec", None)
    if preseeded in available_execs:
        resolved_map[cache_key] = cast(ExecutionMode, preseeded)
        if is_dist:
            _broadcast_if_distributed({"execution": preseeded})
        return resolved_map[cache_key]
    if len(specs) == 1:
        sole = specs[0][1]
        resolved_map[cache_key] = sole
        if is_dist:
            _broadcast_if_distributed({"execution": sole})
        return sole

    bench_kwargs = {
        k: group[k] for k in (
            "lr", "betas", "eps", "weight_decay", "cautious_weight_decay",
            "variance_type", "variance_temper", "tau", "lam", "cautious",
            "cautious_clamp_min", "buffer", "stochastic_rounding", "bound",
            "bound_type", "bound_ratio", "bound_floor", "master_weights",
            "fused_single_pass", "ema_decay", "seed",
        )
        if k in group
    }
    try:
        res = benchmark_backends(
            eligible, steps=6, measure_last=2, strategy="budgeted",
            max_memory_mb=32.0, tune_fuse_tensors=False, use_cache=True,
            distributed_sync=False, **bench_kwargs,
        )
    except Exception as exc:
        logger.debug("LuminaV [execution='auto']: benchmark skipped (%s); using heuristic fallback.", exc)
        if is_dist:
            _broadcast_if_distributed({"execution": None})
        return None

    winner = cast(ExecutionMode, res["execution"])
    resolved_map[cache_key] = winner
    opt.benchmark_results = res
    if isinstance(getattr(opt, "tuning_info", None), dict):
        opt.tuning_info["resolved_execution"] = winner
        opt.tuning_info["backend_benchmark"] = dict(res)
    else:
        opt.tuning_info = {
            "tuning_applied": False,
            "resolved_execution": winner,
            "backend_benchmark": dict(res),
        }
    win_ms = float(res["timings_ms"].get(res["winner_label"], 0.0))
    logger.info(
        "LuminaV [execution='auto']: selected '%s' (%.3f ms/step over %d steps, tail=%d).",
        winner, win_ms, int(res["steps"]), int(res["measure_last"]),
    )
    if is_dist:
        _broadcast_if_distributed(res)
    return winner


__all__ = ["TuneStrategy", "benchmark_backends", "reset_autotune_cache", "resolve_auto_execution"]
