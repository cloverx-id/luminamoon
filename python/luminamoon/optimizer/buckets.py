from __future__ import annotations

import gc
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .config import build_rust_config, is_dual_buffer, master_weight_code
from .constants import DEFAULT_BUCKET_CAP_MB
from .state import convert_torch_state_buffer
from .validation import (
    HAS_TORCH,
    _is_contiguous,
    _is_torch_compiling,
    _make_contiguous,
    _validate_supported_dtype,
    torch,
)


def partition_into_bounded_chunks(
    indexed_params: List[Tuple[int, Any]], cap_bytes: int
) -> List[List[Tuple[int, Any]]]:
    """Partition indexed parameters into sequential chunks bounded by cap_bytes."""
    chunks: List[List[Tuple[int, Any]]] = []
    curr_chunk: List[Tuple[int, Any]] = []
    curr_bytes = 0
    for param_idx, p in indexed_params:
        p_bytes = int(p.numel() * p.element_size())
        if curr_chunk and (curr_bytes + p_bytes > cap_bytes):
            chunks.append(curr_chunk)
            curr_chunk = [(param_idx, p)]
            curr_bytes = p_bytes
        else:
            curr_chunk.append((param_idx, p))
            curr_bytes += p_bytes
    if curr_chunk:
        chunks.append(curr_chunk)
    return chunks


def build_contiguous_buckets(opt: Any, force_rebuild: bool = False) -> None:
    """Coalesce parameters and optimizer states into 1D contiguous chunks with per-tensor scratch slots.

    Gradients stay managed by PyTorch autograd (no permanent flat_g allocation) so
    `zero_grad(set_to_none=True)` reclaims full gradient VRAM, while per-tensor scratch
    slices preserve exact per-layer cautious/radial statistics and PRNG seeds without distortion.
    """
    if not getattr(opt, "_is_torch_mode", False) or not (HAS_TORCH and torch is not None):
        return
    if force_rebuild:
        opt._fused_buckets.clear()

    existing_groups = {id(b["group"]) for b in opt._fused_buckets}
    seen_param_keys: Set[Any] = {
        (p.device, p.data_ptr(), int(p.numel()))
        for b in opt._fused_buckets
        for p, *_ in b.get("slices", [])
    }
    is_compiling = _is_torch_compiling()

    for group in opt.param_groups:
        if id(group) in existing_groups:
            continue
        if not bool(group.get("fuse_tensors", False)) or bool(group.get("fsdp_sync", False)):
            continue

        buf_count = group["buffer"]
        master_mode = master_weight_code(group["master_weights"])
        cap_bytes = int(float(group.get("bucket_cap_mb", DEFAULT_BUCKET_CAP_MB)) * 1024 * 1024)
        if cap_bytes <= 0:
            cap_bytes = int(DEFAULT_BUCKET_CAP_MB * 1024 * 1024)

        by_dev_dtype: Dict[Tuple[Any, Any], List[Tuple[int, Any]]] = {}
        for idx, p in enumerate(group.get("params", [])):
            if not isinstance(p, torch.Tensor) or (not p.requires_grad and p.grad is None):
                continue
            _validate_supported_dtype(p, "param")
            if p.numel() == 0:
                continue
            p_identity = (p.device, p.data_ptr(), int(p.numel()))
            if id(p) in seen_param_keys or p_identity in seen_param_keys:
                continue
            seen_param_keys.add(id(p))
            seen_param_keys.add(p_identity)
            by_dev_dtype.setdefault((p.device, p.dtype), []).append((idx, p))

        for (dev, dt), indexed_params in by_dev_dtype.items():
            for chunk_params in partition_into_bounded_chunks(indexed_params, cap_bytes):
                total_numel = sum(int(p.numel()) for _, p in chunk_params)
                if total_numel == 0:
                    continue

                state_dtype = torch.float32 if (master_mode == 2 or dt == torch.float32) else dt
                has_master = (master_mode > 0) and (dt in (torch.float16, torch.bfloat16))
                n_tensors = len(chunk_params)

                flat_p = torch.empty(total_numel, dtype=dt, device=dev)
                flat_m = torch.zeros(total_numel, dtype=state_dtype, device=dev)
                flat_v = torch.zeros(total_numel, dtype=state_dtype, device=dev) if is_dual_buffer(buf_count) else None
                flat_mp = torch.empty(total_numel, dtype=torch.float32, device=dev) if has_master else None
                flat_scratches = torch.zeros((n_tensors, 6), dtype=torch.float32, device=dev)
                flat_scratches[:, 3] = 1.0
                flat_scratches[:, 4] = 1.0

                offset = 0
                slices_meta: List[Tuple[Any, Any, Any, Any, Any, Any, int, Dict[str, Any], int]] = []
                for t_idx, (param_idx, p) in enumerate(chunk_params):
                    n = int(p.numel())
                    p_slice = flat_p[offset : offset + n].view(p.shape)
                    m_slice = flat_m[offset : offset + n].view(p.shape)
                    v_slice = flat_v[offset : offset + n].view(p.shape) if flat_v is not None else None
                    mp_slice = flat_mp[offset : offset + n].view(p.shape) if flat_mp is not None else None
                    sc_slice = flat_scratches[t_idx]

                    old_data = p.data
                    p_slice.copy_(old_data.detach())
                    if not is_compiling:
                        p.data = p_slice
                    del old_data

                    if mp_slice is not None:
                        mp_slice.copy_(p_slice.float())

                    st = opt.state[p]
                    if isinstance(st.get("exp_avg"), torch.Tensor) and st["exp_avg"].numel() == n:
                        m_slice.copy_(convert_torch_state_buffer(st["exp_avg"], dev, state_dtype))
                    if v_slice is not None and isinstance(st.get("exp_avg_sq"), torch.Tensor) and st["exp_avg_sq"].numel() == n:
                        v_slice.copy_(convert_torch_state_buffer(st["exp_avg_sq"], dev, state_dtype))
                    if mp_slice is not None and isinstance(st.get("master_param"), torch.Tensor) and st["master_param"].numel() == n:
                        mp_slice.copy_(st["master_param"].to(device=dev, dtype=torch.float32))
                    if "step" in st:
                        s_raw = st["step"]
                        st["step"] = int(s_raw.item() if hasattr(s_raw, "item") else s_raw)
                    else:
                        st["step"] = 0
                    if isinstance(st.get("scratch"), torch.Tensor) and st["scratch"].numel() >= 6:
                        sc_slice.copy_(st["scratch"].to(device=dev, dtype=torch.float32)[:6])

                    st["scratch"] = sc_slice
                    st["exp_avg"] = m_slice
                    if v_slice is not None:
                        st["exp_avg_sq"] = v_slice
                    else:
                        st.pop("exp_avg_sq", None)
                    if mp_slice is not None:
                        st["master_param"] = mp_slice
                    else:
                        st.pop("master_param", None)
                    st["buf_count"] = buf_count

                    slices_meta.append(
                        (p, p_slice, m_slice, v_slice, mp_slice, sc_slice, p_slice.data_ptr(), st, param_idx)
                    )
                    offset += n

                opt._fused_buckets.append(
                    {
                        "group": group,
                        "device": dev,
                        "dtype": dt,
                        "state_dtype": state_dtype,
                        "buf_count": buf_count,
                        "master_mode": master_mode,
                        "bucket_cap_mb": cap_bytes / (1024 * 1024),
                        "flat_p": flat_p,
                        "flat_m": flat_m,
                        "flat_v": flat_v,
                        "flat_mp": flat_mp,
                        "flat_scratches": flat_scratches,
                        "slices": slices_meta,
                    }
                )
    gc.collect()


def sync_or_rebuild_fused_buckets(opt: Any) -> None:
    """Synchronize existing contiguous buckets with opt.state or rebuild if topology changed."""
    if not getattr(opt, "_is_torch_mode", False) or not (HAS_TORCH and torch is not None):
        return
    need_rebuild = False
    bucket_groups = {id(b["group"]) for b in opt._fused_buckets}
    bucket_param_ids = {id(p) for b in opt._fused_buckets for p, *_ in b.get("slices", [])}
    for group in opt.param_groups:
        if bool(group.get("fuse_tensors", False)) and not bool(group.get("fsdp_sync", False)):
            if id(group) not in bucket_groups:
                need_rebuild = True
                break
            if any(
                isinstance(p, torch.Tensor)
                and p.numel() > 0
                and (p.requires_grad or p.grad is not None)
                and id(p) not in bucket_param_ids
                for p in group.get("params", [])
            ):
                need_rebuild = True
                break
    if not need_rebuild:
        for bucket in opt._fused_buckets:
            grp = bucket["group"]
            if (
                not bool(grp.get("fuse_tensors", False))
                or bool(grp.get("fsdp_sync", False))
                or grp.get("buffer", 2) != bucket.get("buf_count", 2)
                or master_weight_code(grp["master_weights"]) != int(bucket.get("master_mode", 0))
                or float(grp.get("bucket_cap_mb", DEFAULT_BUCKET_CAP_MB)) != float(bucket.get("bucket_cap_mb", DEFAULT_BUCKET_CAP_MB))
            ):
                need_rebuild = True
                break
    if need_rebuild:
        build_contiguous_buckets(opt, force_rebuild=True)
        return

    is_compiling = _is_torch_compiling()
    for bucket in opt._fused_buckets:
        flat_m = bucket["flat_m"]
        flat_v = bucket["flat_v"]
        flat_mp = bucket["flat_mp"]
        flat_scratches = bucket["flat_scratches"]
        new_slices = []
        offset = 0
        for t_idx, (p, p_slice, _, _, _, _, p_ptr, _, param_idx) in enumerate(bucket["slices"]):
            n = int(p.numel())
            if p.data_ptr() != p_ptr:
                p_slice.copy_(p.detach())
                if not is_compiling:
                    p.data = p_slice

            target_m = flat_m[offset : offset + n].view(p.shape)
            target_v = flat_v[offset : offset + n].view(p.shape) if flat_v is not None else None
            target_mp = flat_mp[offset : offset + n].view(p.shape) if flat_mp is not None else None
            target_sc = flat_scratches[t_idx]

            st = opt.state[p]
            if isinstance(st, dict):
                if isinstance(st.get("exp_avg"), torch.Tensor) and st["exp_avg"].data_ptr() != target_m.data_ptr():
                    target_m.copy_(st["exp_avg"].detach())
                st["exp_avg"] = target_m

                if target_v is not None:
                    if isinstance(st.get("exp_avg_sq"), torch.Tensor) and st["exp_avg_sq"].data_ptr() != target_v.data_ptr():
                        target_v.copy_(st["exp_avg_sq"].detach())
                    st["exp_avg_sq"] = target_v

                if target_mp is not None:
                    if isinstance(st.get("master_param"), torch.Tensor) and st["master_param"].data_ptr() != target_mp.data_ptr():
                        target_mp.copy_(st["master_param"].detach())
                    st["master_param"] = target_mp

                if "step" in st:
                    s_raw = st["step"]
                    st["step"] = int(s_raw.item() if hasattr(s_raw, "item") else s_raw)
                else:
                    st["step"] = 0

                if isinstance(st.get("scratch"), torch.Tensor) and st["scratch"].numel() >= 6:
                    if st["scratch"].data_ptr() != target_sc.data_ptr():
                        target_sc.copy_(st["scratch"][:6])
                st["scratch"] = target_sc

            new_slices.append((p, p_slice, target_m, target_v, target_mp, target_sc, p_ptr, st, param_idx))
            offset += n
        bucket["slices"] = new_slices


def step_fused_buckets(
    opt: Any,
    group: Dict[str, Any],
    grads: Optional[Any],
    rs_mod: Any,
    step_gpu_fn: Callable[..., None],
    group_param_offset: int = 0,
) -> bool:
    """Steps active slices in 1D contiguous buckets with per-tensor scratch isolation (zero distortion)."""
    buf_count = group["buffer"]
    master_mode = master_weight_code(group["master_weights"])
    group_buckets = [b for b in getattr(opt, "_fused_buckets", []) if b.get("group") is group]
    bucket_param_ids = {id(p) for b in group_buckets for p, *_ in b.get("slices", [])}
    has_unbucketed = any(
        isinstance(p, torch.Tensor)
        and p.numel() > 0
        and id(p) not in bucket_param_ids
        and (p.requires_grad or p.grad is not None)
        for p in group.get("params", [])
    )
    if (
        not group_buckets
        or has_unbucketed
        or any(
            b.get("buf_count", buf_count) != buf_count or int(b.get("master_mode", master_mode)) != master_mode
            for b in group_buckets
        )
    ):
        sync_or_rebuild_fused_buckets(opt)
        group_buckets = [b for b in getattr(opt, "_fused_buckets", []) if b.get("group") is group]
    if not group_buckets:
        return False

    is_compiling = _is_torch_compiling()
    dual = is_dual_buffer(buf_count)

    for bucket in group_buckets:
        active_p: List[Any] = []
        active_g: List[Any] = []
        active_m: List[Any] = []
        active_v: List[Any] = []
        active_mp: List[Any] = []
        active_sc: List[Any] = []
        active_steps: List[int] = []
        active_idx: List[int] = []
        compiling_sync_pairs: List[Tuple[Any, Any]] = []

        for p, p_slice, m_slice, v_slice, mp_slice, sc_slice, p_ptr, st, param_idx in bucket["slices"]:
            if p.data_ptr() != p_ptr:
                p_slice.copy_(p.detach())
                if is_compiling:
                    compiling_sync_pairs.append((p, p_slice))
                else:
                    p.data = p_slice

            g = None
            global_grad_idx = group_param_offset + param_idx
            if grads is not None:
                if isinstance(grads, (list, tuple)) and global_grad_idx < len(grads):
                    g = grads[global_grad_idx]
                elif isinstance(grads, dict):
                    try:
                        g = grads.get(p)
                    except TypeError:
                        g = grads.get(id(p))
            if g is None:
                g = p.grad
            if g is None:
                continue
            if getattr(g, "is_sparse", False):
                raise RuntimeError("LuminaV does not support sparse gradients.")
            _validate_supported_dtype(g, "grad")
            if not _is_contiguous(g):
                g = _make_contiguous(g)

            curr_step = st.get("step", 0)
            step_val = int(curr_step.item() if hasattr(curr_step, "item") else curr_step) + 1
            st["step"] = step_val

            active_p.append(p_slice)
            active_g.append(g)
            active_m.append(m_slice)
            active_v.append(v_slice if dual else None)
            active_mp.append(mp_slice)
            active_sc.append(sc_slice)
            active_steps.append(step_val)
            active_idx.append(param_idx)

        if not active_p:
            continue

        dev = bucket["device"]
        if dev.type == "cpu" and rs_mod is not None and group["execution"] != "eager":
            rs_mod.step_pyo3(
                active_p, active_g, active_m, active_v, active_mp, active_sc,
                active_steps, active_idx, build_rust_config(group),
            )
        else:
            step_gpu_fn(
                opt, group, active_p, active_g, active_m, active_v, active_mp,
                active_sc, active_steps, orig_indices=active_idx,
            )

        for p_comp, p_sl_comp in compiling_sync_pairs:
            p_comp.copy_(p_sl_comp)

    return True
