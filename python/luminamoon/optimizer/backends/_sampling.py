from __future__ import annotations

import gc
from typing import Any, Dict, Iterable, List, Literal, Tuple

from ..validation import HAS_NUMPY, HAS_TORCH, np, torch

TuneStrategy = Literal["auto", "full", "budgeted"]
_FULL_CPU_LIMIT_BYTES: int = 256 * 1024 * 1024


def sync_device(device: Any) -> None:
    if not (HAS_TORCH and torch is not None) or device is None:
        return
    dev_type = getattr(device, "type", "cpu")
    if dev_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif dev_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize(device)
    elif dev_type not in ("cpu", "") and hasattr(torch, "accelerator") and hasattr(torch.accelerator, "synchronize"):
        torch.accelerator.synchronize()


def empty_device_cache(device: Any) -> None:
    gc.collect()
    if not (HAS_TORCH and torch is not None) or device is None:
        return
    dev_type = getattr(device, "type", "cpu")
    if dev_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif dev_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def materialize_model_or_params(model_or_params: Any) -> Any:
    """Materialize one-shot parameter iterators/generators once while preserving `nn.Module`."""
    if HAS_TORCH and torch is not None and isinstance(model_or_params, torch.nn.Module):
        return model_or_params
    raw = list(model_or_params) if not isinstance(model_or_params, (list, tuple)) else list(model_or_params)
    out: List[Any] = []
    for elem in raw:
        if isinstance(elem, dict):
            gp = elem.get("params")
            if gp is not None and not isinstance(gp, (list, tuple)):
                is_single = (HAS_TORCH and torch is not None and isinstance(gp, torch.Tensor)) or (
                    HAS_NUMPY and np is not None and isinstance(gp, np.ndarray)
                )
                copied = dict(elem)
                copied["params"] = [gp] if is_single else list(gp)
                out.append(copied)
            else:
                out.append(elem)
        else:
            out.append(elem)
    return out


def extract_params(model_or_params: Any) -> List[Any]:
    if HAS_TORCH and torch is not None and isinstance(model_or_params, torch.nn.Module):
        all_p = list(model_or_params.parameters())
        trainable = [p for p in all_p if p.requires_grad]
        return trainable if trainable else all_p
    raw = list(model_or_params) if not isinstance(model_or_params, (list, tuple)) else list(model_or_params)
    out: List[Any] = []
    for elem in raw:
        if isinstance(elem, dict):
            gp = elem.get("params")
            if gp is not None:
                if (HAS_TORCH and torch is not None and isinstance(gp, torch.Tensor)) or (
                    HAS_NUMPY and np is not None and isinstance(gp, np.ndarray)
                ):
                    out.append(gp)
                else:
                    out.extend(list(gp))
        else:
            out.append(elem)
    return out


def is_dtensor(p: Any) -> bool:
    return bool(
        HAS_TORCH and torch is not None and isinstance(p, torch.Tensor)
        and hasattr(p, "to_local") and hasattr(p, "device_mesh")
    )


def local_tensor(p: Any) -> Any:
    return getattr(p, "to_local")() if is_dtensor(p) else p


def param_nbytes(p: Any) -> int:
    lp = local_tensor(p)
    if HAS_TORCH and torch is not None and isinstance(lp, torch.Tensor):
        return int(lp.numel() * lp.element_size())
    if HAS_NUMPY and np is not None and isinstance(lp, np.ndarray):
        return int(lp.nbytes)
    return 0


def shape_signature(params: List[Any]) -> Tuple[Tuple[Tuple[int, ...], str], ...]:
    items: List[Tuple[Tuple[int, ...], str]] = []
    for p in params:
        lp = local_tensor(p)
        if HAS_TORCH and torch is not None and isinstance(lp, torch.Tensor):
            items.append((tuple(int(d) for d in lp.shape), str(lp.dtype)))
        elif HAS_NUMPY and np is not None and isinstance(lp, np.ndarray):
            items.append((tuple(int(d) for d in lp.shape), str(lp.dtype)))
    return tuple(items)


def resolve_strategy(
    flat_params: List[Any],
    requested: TuneStrategy,
    *,
    buffer_mode: Any,
    master_weights: str,
) -> Literal["full", "budgeted"]:
    if requested in ("full", "budgeted"):
        return requested
    total_bytes = sum(param_nbytes(p) for p in flat_params)
    max_param_bytes = max((param_nbytes(p) for p in flat_params), default=0)
    first = flat_params[0]
    is_torch = HAS_TORCH and torch is not None and isinstance(first, torch.Tensor)
    if is_torch and first.device.type == "cuda" and torch.cuda.is_available():
        try:
            free_vram, _ = torch.cuda.mem_get_info(first.device)
            state_mult = 2.0 if buffer_mode in (2, "t2", "T2") else 1.0
            if master_weights in ("semi", "full"):
                state_mult += 2.0
            est_needed = int(total_bytes * state_mult) + max_param_bytes
            return "full" if est_needed <= int(free_vram * 0.60) else "budgeted"
        except Exception:
            return "full" if total_bytes <= 128 * 1024 * 1024 else "budgeted"
    return "full" if total_bytes <= _FULL_CPU_LIMIT_BYTES else "budgeted"


def sample_representative_clones(
    flat_params: List[Any],
    max_memory_mb: float,
) -> Tuple[List[Any], List[Any], int]:
    budget_bytes = max(1, int(float(max_memory_mb) * 1024 * 1024))
    indexed = [(i, p, param_nbytes(p)) for i, p in enumerate(flat_params) if param_nbytes(p) > 0]
    if not indexed:
        raise ValueError("benchmark_backends requires at least one non-empty parameter.")

    total_bytes = sum(nb for _, _, nb in indexed)
    if total_bytes <= budget_bytes:
        selected = [p for _, p, _ in indexed]
    else:
        ordered = sorted(indexed, key=lambda item: item[2], reverse=True)
        selected = []
        used = 0
        small_tail = [item for item in reversed(ordered) if item[2] <= 65_536][:8]
        small_ids = {item[0] for item in small_tail}
        for _, p, nb in small_tail:
            if used + nb <= budget_bytes:
                selected.append(p)
                used += nb
        for idx, p, nb in ordered:
            if idx not in small_ids and used + nb <= budget_bytes:
                selected.append(p)
                used += nb
        if not selected:
            p0 = local_tensor(ordered[0][1])
            elem_sz = max(1, param_nbytes(p0) // max(1, int(p0.numel() if hasattr(p0, "numel") else p0.size)))
            cap_elems = max(16, budget_bytes // elem_sz)
            if HAS_TORCH and torch is not None and isinstance(p0, torch.Tensor):
                selected = [torch.empty((cap_elems,), device=p0.device, dtype=p0.dtype)]
            else:
                selected = [np.empty((cap_elems,), dtype=p0.dtype)]

    clones: List[Any] = []
    grads: List[Any] = []
    sampled_bytes = 0
    for p in selected:
        lp = local_tensor(p)
        if HAS_TORCH and torch is not None and isinstance(lp, torch.Tensor):
            shape = tuple(int(d) for d in lp.shape)
            c = torch.full(shape, 0.5, device=lp.device, dtype=lp.dtype).contiguous()
            g = torch.full(shape, 0.01, device=lp.device, dtype=lp.dtype).contiguous()
            clones.append(torch.nn.Parameter(c))
            grads.append(g)
            sampled_bytes += int(c.numel() * c.element_size())
        elif HAS_NUMPY and np is not None and isinstance(lp, np.ndarray):
            shape = tuple(int(d) for d in lp.shape)
            c = np.ascontiguousarray(np.full(shape, 0.5, dtype=lp.dtype))
            g = np.ascontiguousarray(np.full(shape, 0.01, dtype=lp.dtype))
            clones.append(c)
            grads.append(g)
            sampled_bytes += int(c.nbytes)
    return clones, grads, sampled_bytes


def build_shared_read_only_grads(params: List[Any]) -> List[Any]:
    """Allocate one shared read-only gradient pool per `(device, dtype)` sized to max_numel."""
    pools: Dict[Tuple[str, Any], int] = {}
    for p in params:
        lp = local_tensor(p)
        if HAS_TORCH and torch is not None and isinstance(lp, torch.Tensor):
            key = (str(lp.device), lp.dtype)
            pools[key] = max(pools.get(key, 0), int(lp.numel()))
        elif HAS_NUMPY and np is not None and isinstance(lp, np.ndarray):
            key = ("numpy", lp.dtype)
            pools[key] = max(pools.get(key, 0), int(lp.size))

    buffers: Dict[Tuple[str, Any], Any] = {}
    for (dev_s, dt), max_n in pools.items():
        if dev_s == "numpy" and HAS_NUMPY and np is not None:
            buffers[(dev_s, dt)] = np.ascontiguousarray(np.full((max(1, max_n),), 0.01, dtype=dt))
        elif HAS_TORCH and torch is not None:
            buffers[(dev_s, dt)] = torch.full((max(1, max_n),), 0.01, device=dev_s, dtype=dt)

    grads: List[Any] = []
    for p in params:
        lp = local_tensor(p)
        if HAS_TORCH and torch is not None and isinstance(lp, torch.Tensor):
            grads.append(buffers[(str(lp.device), lp.dtype)][: lp.numel()].view(lp.shape))
        elif HAS_NUMPY and np is not None and isinstance(lp, np.ndarray):
            grads.append(buffers[("numpy", lp.dtype)][: lp.size].reshape(lp.shape))
    return grads


def check_finite(params: Iterable[Any]) -> bool:
    for p in params:
        lp = local_tensor(p)
        if HAS_TORCH and torch is not None and isinstance(lp, torch.Tensor):
            if not bool(torch.isfinite(lp).all().item()):
                return False
        elif HAS_NUMPY and np is not None and isinstance(lp, np.ndarray):
            if not bool(np.isfinite(lp).all()):
                return False
    return True
