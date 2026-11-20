from __future__ import annotations

import gc
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .config import (
    bound_mode_code,
    build_rust_config,
    is_dual_buffer,
    master_weight_code,
)
from .constants import DEFAULT_BUCKET_CAP_MB
from .state import convert_torch_state_buffer
from .validation import HAS_TORCH, _is_torch_compiling, _validate_supported_dtype, torch


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
    """Coalesce parameters per (group, device, dtype) into bounded contiguous 1D chunks."""
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
            if not isinstance(p, torch.Tensor) or not p.requires_grad:
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

                flat_p = torch.empty(total_numel, dtype=dt, device=dev)
                flat_g = torch.zeros(total_numel, dtype=dt, device=dev)
                flat_m = torch.zeros(total_numel, dtype=state_dtype, device=dev)
                flat_v = torch.zeros(total_numel, dtype=state_dtype, device=dev) if is_dual_buffer(buf_count) else None
                flat_mp = torch.empty(total_numel, dtype=torch.float32, device=dev) if has_master else None
                flat_scratch = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float32, device=dev)

                offset = 0
                slices_meta: List[Tuple[Any, Any, Any, int, int, Dict[str, Any], int]] = []
                bucket_step = 0
                for param_idx, p in chunk_params:
                    n = int(p.numel())
                    p_slice = flat_p[offset : offset + n].view(p.shape)
                    g_slice = flat_g[offset : offset + n].view(p.shape)
                    m_slice = flat_m[offset : offset + n].view(p.shape)
                    v_slice = flat_v[offset : offset + n].view(p.shape) if flat_v is not None else None
                    mp_slice = flat_mp[offset : offset + n].view(p.shape) if flat_mp is not None else None

                    old_data = p.data
                    p_slice.copy_(old_data.detach())
                    if not is_compiling:
                        p.data = p_slice
                    del old_data

                    if p.grad is not None:
                        old_grad = p.grad
                        g_slice.copy_(old_grad.detach())
                        del old_grad
                    p.grad = g_slice

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
                        bucket_step = max(bucket_step, int(s_raw.item() if hasattr(s_raw, "item") else s_raw))
                    if isinstance(st.get("scratch"), torch.Tensor) and st["scratch"].numel() >= 6:
                        flat_scratch.copy_(st["scratch"].to(device=dev, dtype=torch.float32)[:6])

                    st["step"] = bucket_step
                    st["scratch"] = flat_scratch
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

                    slices_meta.append((p, p_slice, g_slice, p_slice.data_ptr(), g_slice.data_ptr(), st, param_idx))
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
                        "flat_g": flat_g,
                        "flat_m": flat_m,
                        "flat_v": flat_v,
                        "flat_mp": flat_mp,
                        "scratch": flat_scratch,
                        "slices": slices_meta,
                        "step": bucket_step,
                    }
                )
    gc.collect()


def sync_or_rebuild_fused_buckets(opt: Any) -> None:
    """Synchronize existing contiguous buckets with opt.state or rebuild if topology changed."""
    if not getattr(opt, "_is_torch_mode", False) or not (HAS_TORCH and torch is not None):
        return
    need_rebuild = False
    bucket_groups = {id(b["group"]) for b in opt._fused_buckets}
    for group in opt.param_groups:
        if bool(group.get("fuse_tensors", False)) and not bool(group.get("fsdp_sync", False)):
            if id(group) not in bucket_groups:
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
        scratch = bucket["scratch"]
        bucket_step = int(bucket.get("step", 0))
        new_slices = []
        offset = 0
        for p, p_slice, g_slice, p_ptr, g_ptr, _, param_idx in bucket["slices"]:
            n = int(p.numel())
            if p.data_ptr() != p_ptr:
                p_slice.copy_(p.detach())
                if not is_compiling:
                    p.data = p_slice
            if p.grad is not None and p.grad.data_ptr() != g_ptr:
                g_slice.copy_(p.grad.detach())
            p.grad = g_slice

            st = opt.state[p]
            if isinstance(st, dict):
                if isinstance(st.get("exp_avg"), torch.Tensor):
                    target_m = flat_m[offset : offset + n].view(p.shape)
                    if st["exp_avg"].data_ptr() != target_m.data_ptr():
                        target_m.copy_(st["exp_avg"].detach())
                    st["exp_avg"] = target_m
                if flat_v is not None and isinstance(st.get("exp_avg_sq"), torch.Tensor):
                    target_v = flat_v[offset : offset + n].view(p.shape)
                    if st["exp_avg_sq"].data_ptr() != target_v.data_ptr():
                        target_v.copy_(st["exp_avg_sq"].detach())
                    st["exp_avg_sq"] = target_v
                if flat_mp is not None and isinstance(st.get("master_param"), torch.Tensor):
                    target_mp = flat_mp[offset : offset + n].view(p.shape)
                    if st["master_param"].data_ptr() != target_mp.data_ptr():
                        target_mp.copy_(st["master_param"].detach())
                    st["master_param"] = target_mp
                if "step" in st:
                    s_raw = st["step"]
                    bucket_step = int(s_raw.item() if hasattr(s_raw, "item") else s_raw)
                    st["step"] = bucket_step
                if isinstance(st.get("scratch"), torch.Tensor) and st["scratch"].numel() >= 6:
                    if st["scratch"].data_ptr() != scratch.data_ptr():
                        scratch.copy_(st["scratch"][:6])
                    st["scratch"] = scratch
            new_slices.append((p, p_slice, g_slice, p_ptr, g_ptr, st, param_idx))
            offset += n
        bucket["slices"] = new_slices
        bucket["step"] = bucket_step


def _backup_inactive_slice(
    p_slice: Any, m_slice: Any, v_slice: Any, mp_slice: Any, st: Dict[str, Any], prev_step: int
) -> Tuple[Any, ...]:
    return (
        p_slice,
        p_slice.detach().clone(),
        m_slice,
        m_slice.detach().clone(),
        v_slice,
        v_slice.detach().clone() if v_slice is not None else None,
        mp_slice,
        mp_slice.detach().clone() if mp_slice is not None else None,
        st,
        prev_step,
    )


def step_fused_buckets(
    opt: Any,
    group: Dict[str, Any],
    grads: Optional[Any],
    rs_mod: Any,
    step_gpu_fn: Callable[..., None],
) -> bool:
    buf_count = group["buffer"]
    master_mode = master_weight_code(group["master_weights"])
    group_buckets = [b for b in getattr(opt, "_fused_buckets", []) if b.get("group") is group]
    if not group_buckets or any(
        b.get("buf_count", buf_count) != buf_count or int(b.get("master_mode", master_mode)) != master_mode
        for b in group_buckets
    ):
        sync_or_rebuild_fused_buckets(opt)
        group_buckets = [b for b in getattr(opt, "_fused_buckets", []) if b.get("group") is group]
    if not group_buckets:
        return False

    cautious = bool(group["cautious"])
    radial_active = bound_mode_code(bool(group["bound"]), group["bound_type"], float(group["bound_ratio"])) == 1
    is_compiling = _is_torch_compiling()

    for bucket in group_buckets:
        flat_p = bucket["flat_p"]
        flat_g = bucket["flat_g"]
        flat_m = bucket["flat_m"]
        flat_v = bucket["flat_v"]
        flat_mp = bucket["flat_mp"]
        scratch = bucket["scratch"]

        any_grad_present = False
        bucket["step"] = int(bucket["step"]) + 1
        step_t = int(bucket["step"])
        inactive_backups: List[Tuple[Any, ...]] = []
        compiling_sync_pairs: List[Tuple[Any, Any]] = []
        offset_scan = 0

        for p, p_slice, g_slice, p_ptr, g_ptr, st, param_idx in bucket["slices"]:
            n_s = int(p.numel())
            m_slice = flat_m[offset_scan : offset_scan + n_s].view(p.shape)
            v_slice = flat_v[offset_scan : offset_scan + n_s].view(p.shape) if flat_v is not None else None
            mp_slice = flat_mp[offset_scan : offset_scan + n_s].view(p.shape) if flat_mp is not None else None
            offset_scan += n_s

            if p.data_ptr() != p_ptr:
                p_slice.copy_(p.detach())
                if is_compiling:
                    compiling_sync_pairs.append((p, p_slice))
                else:
                    p.data = p_slice

            if grads is None:
                g = p.grad
            elif isinstance(grads, (list, tuple)) and param_idx < len(grads):
                g = grads[param_idx]
            elif isinstance(grads, dict) and p in grads:
                g = grads[p]
            else:
                g = None

            if g is None:
                g_slice.zero_()
                prev_step = int(st.get("step", step_t - 1))
                inactive_backups.append(_backup_inactive_slice(p_slice, m_slice, v_slice, mp_slice, st, prev_step))
                continue

            any_grad_present = True
            if g is not g_slice:
                if getattr(g, "is_sparse", False):
                    bucket["step"] = step_t - 1
                    raise RuntimeError("LuminaV does not support sparse gradients.")
                if grads is not None:
                    _validate_supported_dtype(g, "grad")
                if g.data_ptr() != g_ptr:
                    g_slice.copy_(g.detach())
                    if grads is None and not is_compiling:
                        p.grad = g_slice
            st["step"] = step_t

        if not any_grad_present:
            bucket["step"] = step_t - 1
            for _, _, _, _, _, st, _ in bucket["slices"]:
                st["step"] = int(st.get("step", step_t - 1))
            continue

        if not cautious:
            scratch[3] = 1.0
        if not radial_active:
            scratch[4] = 1.0

        dev = bucket["device"]
        if dev.type == "cpu" and rs_mod is not None and group["execution"] != "eager":
            rs_mod.step_pyo3(
                [flat_p], [flat_g], [flat_m], [flat_v], [flat_mp], [scratch], [step_t], [0], build_rust_config(group)
            )
        else:
            step_gpu_fn(
                opt, group, [flat_p], [flat_g], [flat_m], [flat_v], [flat_mp], [scratch], [step_t], orig_indices=[0]
            )

        for p_sl, p_bak, m_sl, m_bak, v_sl, v_bak, mp_sl, mp_bak, st_inact, prev_step_inact in inactive_backups:
            p_sl.copy_(p_bak)
            m_sl.copy_(m_bak)
            if v_sl is not None and v_bak is not None:
                v_sl.copy_(v_bak)
            if mp_sl is not None and mp_bak is not None:
                mp_sl.copy_(mp_bak)
            st_inact["step"] = prev_step_inact
        for p_comp, p_sl_comp in compiling_sync_pairs:
            p_comp.copy_(p_sl_comp)

    return True
