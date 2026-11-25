from __future__ import annotations

import inspect
import math
import random
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple

from ._model_profile import ModelProfile, round_sig
from .backends._sampling import empty_device_cache, extract_params, param_nbytes, sync_device
from .validation import HAS_NUMPY, HAS_TORCH, np, torch

_DEFAULT_LR_MULTIPLIERS: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)


def filter_kwargs_for_optimizer(
    optimizer_cls: Any,
    config: Dict[str, Any],
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Filter a tuned config dict to keyword arguments accepted by `optimizer_cls.__init__`."""
    try:
        sig = inspect.signature(optimizer_cls.__init__)
        params = sig.parameters
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        params = {}
        accepts_var_kw = False

    var_kw_names = {n for n, p in params.items() if p.kind == inspect.Parameter.VAR_KEYWORD}
    if accepts_var_kw and getattr(optimizer_cls, "__name__", "") == "LuminaV":
        accepted = set(params.keys()) - {"self", "params"} - var_kw_names
    elif accepts_var_kw:
        accepted = (set(params.keys()) - {"self", "params"} - var_kw_names) | {
            "lr", "betas", "weight_decay", "eps",
        }
    else:
        accepted = {
            name for name, p in params.items()
            if name not in ("self", "params")
            and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }

    filtered: Dict[str, Any] = {k: v for k, v in config.items() if k in accepted}
    # Map beta1 -> momentum cleanly for SGD-family optimizers when betas is not accepted.
    if "momentum" in accepted and "momentum" not in filtered and "betas" not in accepted and "betas" in config:
        betas_val = config["betas"]
        if isinstance(betas_val, (tuple, list)) and len(betas_val) >= 1:
            filtered["momentum"] = float(betas_val[0])
    if overrides:
        filtered.update(overrides)
    return filtered


def _accepts_model_arg(loss_fn: Callable[..., Any]) -> bool:
    try:
        sig = inspect.signature(loss_fn)
    except (TypeError, ValueError):
        return True
    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            return True
    return False


class _LossContractError(ValueError):
    """Raised when user-supplied loss_fn violates the scalar differentiable tensor contract."""


def _invoke_loss(loss_fn: Callable[..., Any], model_or_params: Any, pass_model: bool, *, require_grad: bool = False) -> Any:
    loss = loss_fn(model_or_params) if pass_model else loss_fn()
    if not (HAS_TORCH and torch is not None and isinstance(loss, torch.Tensor)):
        raise _LossContractError(f"AutoTune loss_fn must return a scalar torch.Tensor, got {type(loss).__name__}.")
    if loss.numel() != 1:
        raise _LossContractError(
            f"AutoTune loss_fn must return a scalar tensor (numel=1), got shape {tuple(loss.shape)}."
        )
    if require_grad and not loss.requires_grad:
        raise _LossContractError(
            "AutoTune loss_fn returned a tensor with requires_grad=False; ensure loss_fn runs outside "
            "torch.no_grad() and depends on trainable model parameters."
        )
    return loss


class _FullModelSnapshot:
    """Bit-exact CPU snapshot covering parameters, module buffers, gradients, training mode, and RNGs."""

    def __init__(self, model_or_params: Any, trainable_params: List[Any]) -> None:
        assert torch is not None
        self.is_module = isinstance(model_or_params, torch.nn.Module)
        self.model = model_or_params if self.is_module else None
        self.was_training = bool(model_or_params.training) if self.is_module else True

        self.all_params: List[Any] = (
            list(model_or_params.parameters()) if self.is_module else list(trainable_params)
        )
        self.param_snaps = [p.detach().cpu().clone() for p in self.all_params]
        self.orig_grads = [
            p.grad.detach().cpu().clone() if getattr(p, "grad", None) is not None else None
            for p in self.all_params
        ]
        self.buffers: List[Any] = list(model_or_params.buffers()) if self.is_module else []
        self.buf_snaps = [b.detach().cpu().clone() for b in self.buffers]

        self.py_rng = random.getstate()
        self.torch_rng = torch.random.get_rng_state()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self.np_rng = np.random.get_state() if (HAS_NUMPY and np is not None) else None

    def restore_rng(self) -> None:
        assert torch is not None
        random.setstate(self.py_rng)
        torch.random.set_rng_state(self.torch_rng)
        if self.cuda_rng is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda_rng)
        if self.np_rng is not None and HAS_NUMPY and np is not None:
            np.random.set_state(self.np_rng)

    def restore(self, *, restore_grads: bool) -> None:
        assert torch is not None
        with torch.no_grad():
            for p, snap in zip(self.all_params, self.param_snaps):
                p.copy_(snap.to(device=p.device, dtype=p.dtype, copy=True))
            for b, bsnap in zip(self.buffers, self.buf_snaps):
                b.copy_(bsnap.to(device=b.device, dtype=b.dtype, copy=True))
            for p, gsnap in zip(self.all_params, self.orig_grads):
                if restore_grads:
                    p.grad = None if gsnap is None else gsnap.to(device=p.device, dtype=p.dtype, copy=True)
                else:
                    p.grad = None
        if self.is_module and self.model is not None:
            self.model.train(self.was_training)
        self.restore_rng()

    def weight_displacement_ratio(self, trainable_params: List[Any]) -> float:
        assert torch is not None
        snap_by_id = {id(p): snap for p, snap in zip(self.all_params, self.param_snaps)}
        diff_sq = 0.0
        base_sq = 0.0
        with torch.no_grad():
            for p in trainable_params:
                w0 = snap_by_id.get(id(p))
                if w0 is None:
                    continue
                wf = p.detach().float().cpu()
                w0f = w0.float()
                diff_sq += float((wf - w0f).pow(2).sum().item())
                base_sq += float(w0f.pow(2).sum().item())
        return math.sqrt(diff_sq) / (math.sqrt(base_sq) + 1e-8)


def _measure_grad_stats(trainable_params: List[Any]) -> Tuple[float, float]:
    assert torch is not None
    sq_sum = 0.0
    abs_sum = 0.0
    max_abs = 0.0
    nz_count = 0
    for p in trainable_params:
        g = getattr(p, "grad", None)
        if g is None or g.numel() == 0:
            continue
        gf = g.detach().float()
        g_abs = gf.abs()
        sq_sum += float(gf.pow(2).sum().item())
        abs_sum += float(g_abs.sum().item())
        g_max = float(g_abs.max().item())
        if g_max > max_abs:
            max_abs = g_max
        nz_count += int((g_abs > 0.0).sum().item())
    if not math.isfinite(sq_sum) or not math.isfinite(abs_sum):
        return float("nan"), 0.0
    if nz_count == 0:
        return 0.0, 0.0
    grad_norm = math.sqrt(sq_sum)
    # Use active non-zero mean(|g|) instead of RMS so single-coordinate spikes are not capped at sqrt(N).
    mean_abs_nz = abs_sum / float(nz_count)
    spike_ratio = max_abs / (mean_abs_nz + 1e-12) if mean_abs_nz > 0.0 else 0.0
    return grad_norm, spike_ratio


def _has_vram_for_probe(profile: ModelProfile, buffer_mode: Any, master_weights: str) -> bool:
    if profile.device_type != "cuda" or profile.free_vram_bytes is None:
        return True
    state_mult = 2.0 if buffer_mode in (2, "t2", "T2") else 1.0
    if master_weights in ("semi", "full"):
        state_mult += 2.0
    # Account for optimizer states + temporary forward/backward activation headroom
    est_needed = int(profile.param_bytes * (state_mult + 1.5))
    return est_needed <= int(profile.free_vram_bytes * 0.65)


def _any_rank_diverged(local_bad: bool, device: Any) -> bool:
    """Synchronize trial-abort decision across distributed ranks to prevent DDP/FSDP deadlocks."""
    if not (
        HAS_TORCH and torch is not None and hasattr(torch, "distributed")
        and torch.distributed.is_available() and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        return local_bad
    backend = str(torch.distributed.get_backend()).lower()
    flag_dev = device if (backend == "nccl" and getattr(device, "type", "cpu") == "cuda") else torch.device("cpu")
    flag = torch.tensor([1 if local_bad else 0], dtype=torch.int32, device=flag_dev)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(int(flag.item()) != 0)


def probe_hyperparameters_empirical(
    model_or_params: Any,
    loss_fn: Callable[..., Any],
    *,
    profile: ModelProfile,
    base_config: Dict[str, Any],
    optimizer_cls: Optional[Any] = None,
    lr_steps: int = 5,
    lr_multipliers: Tuple[float, ...] = _DEFAULT_LR_MULTIPLIERS,
    safety_factor: float = 0.65,
    execution_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a full-model multi-step empirical LR & gradient-stability probe with bit-exact reset."""
    if not callable(loss_fn):
        raise TypeError("loss_fn must be callable when provided to AutoTune.")
    if not (HAS_TORCH and torch is not None):
        raise RuntimeError("Empirical loss_fn probing requires PyTorch.")
    if not isinstance(lr_steps, int) or isinstance(lr_steps, bool) or not (2 <= lr_steps <= 20):
        raise ValueError(f"lr_steps must be an int in [2, 20], got {lr_steps!r}.")
    if not (0.1 < float(safety_factor) <= 1.0):
        raise ValueError(f"safety_factor must be in (0.1, 1.0], got {safety_factor!r}.")
    valid_mults: List[float] = []
    for raw_m in lr_multipliers:
        fm = float(raw_m)
        if not math.isfinite(fm) or fm <= 0.0:
            raise ValueError(f"lr_multipliers must contain finite positive values, got {raw_m!r}.")
        valid_mults.append(fm)
    if not valid_mults:
        raise ValueError("lr_multipliers must contain at least one finite positive value.")

    trainable = extract_params(model_or_params)
    if not trainable or not isinstance(trainable[0], torch.Tensor):
        raise TypeError("Empirical loss_fn probing requires a PyTorch nn.Module or torch.Tensor parameters.")

    if not _has_vram_for_probe(profile, base_config["buffer"], str(base_config["master_weights"])):
        return {
            "status": "skipped_low_vram",
            "reason": "Insufficient free CUDA VRAM for safe full-model empirical probe; keeping architecture prior.",
            "selected_lr": float(base_config["lr"]),
            "selected_variance_type": str(base_config["variance_type"]),
            "lr_steps": lr_steps,
            "trials": [],
        }

    from . import luminav as lum_mod

    target_opt_cls = lum_mod.LuminaV if optimizer_cls is None else optimizer_cls
    is_luminav = target_opt_cls is lum_mod.LuminaV

    device = trainable[0].device
    pass_model = _accepts_model_arg(loss_fn)
    base_lr = float(base_config["lr"])
    probe_exec = execution_hint if execution_hint in ("simd", "eager", "triton", "scalar") else (
        "eager" if device.type in ("cuda", "xpu") else ("simd" if lum_mod._luminamoon_rs is not None else "eager")
    )

    trial_base = {
        k: v for k, v in base_config.items()
        if k not in ("lr", "execution", "fuse_tensors", "prewarm_jit", "fsdp_sync")
    }
    if is_luminav:
        trial_base["execution"] = probe_exec
        trial_base["fuse_tensors"] = False
        trial_base["prewarm_jit"] = False
        trial_base["fsdp_sync"] = False
    else:
        trial_base = filter_kwargs_for_optimizer(target_opt_cls, trial_base)

    snapshot = _FullModelSnapshot(model_or_params, trainable)
    trials: List[Dict[str, Any]] = []
    observed_spikes: List[float] = []
    observed_gnorms: List[float] = []

    try:
        for mult in sorted(set(valid_mults)):
            cand_lr = round_sig(base_lr * mult, sig=3)
            snapshot.restore(restore_grads=False)
            tmp_opt = target_opt_cls(trainable, lr=cand_lr, **trial_base)
            losses: List[float] = []
            trial_spikes: List[float] = []
            trial_gnorms: List[float] = []
            status = "ok"
            try:
                for step_idx in range(lr_steps):
                    tmp_opt.zero_grad(set_to_none=True)
                    loss = _invoke_loss(loss_fn, model_or_params, pass_model, require_grad=True)
                    l_val = float(loss.detach().item())
                    losses.append(l_val)
                    div_limit = losses[0] + max(1.5 * abs(losses[0]), 1.0)
                    local_bad = (not math.isfinite(l_val)) or (step_idx > 0 and l_val > div_limit)
                    if _any_rank_diverged(local_bad, device):
                        status = "diverged"
                        break
                    loss.backward()
                    gn, spk = _measure_grad_stats(trainable)
                    if _any_rank_diverged(not math.isfinite(gn), device):
                        status = "diverged"
                        break
                    trial_gnorms.append(gn)
                    trial_spikes.append(spk)
                    tmp_opt.step()

                if status == "ok":
                    sync_device(device)
                    # Evaluate post-update loss under step-0 RNG so Dropout/stochastic noise cancels out.
                    snapshot.restore_rng()
                    with torch.no_grad():
                        l_final = float(_invoke_loss(loss_fn, model_or_params, pass_model, require_grad=False).detach().item())
                    div_limit_end = losses[0] + max(1.5 * abs(losses[0]), 1.0)
                    if _any_rank_diverged((not math.isfinite(l_final)) or (l_final > div_limit_end), device):
                        status = "diverged"
                        losses.append(l_final)
                    else:
                        losses.append(l_final)
                rel_disp = snapshot.weight_displacement_ratio(trainable) if status == "ok" else float("inf")
            except _LossContractError:
                raise
            except Exception as exc:
                status = f"error: {str(exc).splitlines()[0] if str(exc) else type(exc).__name__}"
                rel_disp = float("inf")
            finally:
                del tmp_opt
                empty_device_cache(device)

            loss_0 = losses[0] if losses else float("nan")
            loss_end = losses[-1] if losses else float("nan")
            osc = sum(
                1 for i in range(1, len(losses))
                if math.isfinite(losses[i]) and math.isfinite(losses[i - 1])
                and losses[i] > losses[i - 1] + 0.05 * max(abs(losses[i - 1]), 1e-8)
            )
            rel_drop = (
                (loss_0 - loss_end) / max(abs(loss_0), 1e-8)
                if (status == "ok" and math.isfinite(loss_0) and math.isfinite(loss_end))
                else -1.0
            )
            score = (rel_drop / (1.0 + 0.35 * osc)) if (status == "ok" and rel_drop > 0.0 and rel_disp < 0.75) else -1.0
            if trial_spikes:
                observed_spikes.extend(trial_spikes)
                observed_gnorms.extend(trial_gnorms)
            trials.append(
                {
                    "lr": cand_lr,
                    "multiplier": mult,
                    "status": status,
                    "loss_start": round(loss_0, 6) if math.isfinite(loss_0) else None,
                    "loss_end": round(loss_end, 6) if math.isfinite(loss_end) else None,
                    "rel_drop": round(rel_drop, 6) if math.isfinite(rel_drop) else None,
                    "oscillations": int(osc),
                    "rel_displacement": round(rel_disp, 6) if math.isfinite(rel_disp) else None,
                    "score": round(score, 6) if math.isfinite(score) else None,
                }
            )
    finally:
        snapshot.restore(restore_grads=True)
        empty_device_cache(device)

    # Select winning LR with cliff-edge safety margin
    valid = [t for t in trials if t["status"] == "ok" and (t["score"] or -1.0) > 0.0]
    probe_status = "ok"
    if valid:
        best = max(valid, key=lambda t: float(t["score"]))
        best_lr = float(best["lr"])
        higher = [t for t in trials if float(t["lr"]) > best_lr]
        near_cliff = (not higher) or any(t["status"] != "ok" or int(t["oscillations"]) > 0 for t in higher[:1])
        chosen_lr = round_sig(best_lr * safety_factor, sig=2) if near_cliff else round_sig(best_lr, sig=2)
    else:
        finite = [t for t in trials if t["status"] == "ok"]
        if finite:
            chosen_lr = round_sig(min(float(t["lr"]) for t in finite) * 0.5, sig=2)
            probe_status = "conservative_fallback"
        else:
            chosen_lr = base_lr
            probe_status = "fallback_prior"

    # Gradient heavy-tail / spike detector -> auto-enable RTV on dual-buffer modes
    max_spike = max(observed_spikes) if observed_spikes else 0.0
    norm_jump = (
        (max(observed_gnorms) / max(statistics.median(observed_gnorms), 1e-12))
        if len(observed_gnorms) >= 2
        else 1.0
    )
    is_dual = base_config["buffer"] in (2, "t2", "T2")
    spike_detected = bool(max_spike >= 25.0 or norm_jump >= 5.0)
    selected_vt = "rtv" if (is_dual and spike_detected) else str(base_config["variance_type"])

    result: Dict[str, Any] = {
        "status": probe_status,
        "selected_lr": float(chosen_lr),
        "selected_variance_type": selected_vt,
        "spike_detected": spike_detected,
        "max_grad_spike_ratio": round(float(max_spike), 3),
        "grad_norm_jump_ratio": round(float(norm_jump), 3),
        "lr_steps": int(lr_steps),
        "safety_factor": float(safety_factor),
        "trials": trials,
    }
    # Synchronize empirical probe decision across distributed ranks so DDP/FSDP ranks never diverge on lr.
    if (
        HAS_TORCH and torch is not None and hasattr(torch, "distributed")
        and torch.distributed.is_available() and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        box: List[Any] = [result if torch.distributed.get_rank() == 0 else None]
        torch.distributed.broadcast_object_list(box, src=0)
        return dict(box[0])
    return result


__all__ = ["filter_kwargs_for_optimizer", "probe_hyperparameters_empirical"]
