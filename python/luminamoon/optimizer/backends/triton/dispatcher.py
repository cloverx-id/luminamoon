from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor

from ...config import StepOptions
from ...torch_ops import _get_device_context
from ..eager import compute_pass1_eager_one, step_eager_group
from ..fsdp import (
    fallback_post_step_ema,
    sync_empty_fsdp_shard,
    sync_fsdp_scratch,
    sync_fsdp_scratch_batch,
)
from .common import HAS_TRITON, triton
from .launchers import (
    block_config,
    copy_back_non_contig,
    dtype_mode_code,
    launch_pass1,
    launch_pass2,
    launch_post_step_ema,
    launch_single_pass,
    prepare_triton_buffers,
)
from .multi_tensor import launch_multi_tensor_step

logger = logging.getLogger("LuminaV.GPU")


def _precompute_fsdp_pass1(
    opt: Any,
    pass1_indices: List[int],
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Optional[Tensor]],
    master_params: Optional[List[Tensor]],
    steps: List[Any],
    scratches: List[Tensor],
    opts: StepOptions,
) -> Dict[int, float]:
    for idx in pass1_indices:
        p = params[idx]
        sc = scratches[idx]
        sc[:3].zero_()
        if p.numel() == 0:
            continue
        st_raw = steps[idx]
        step = int(st_raw.item() if isinstance(st_raw, torch.Tensor) else st_raw)
        c1, c2 = opts.dual_coefficients(step) if opts.is_dual else (0.0, 0.0)
        v_ref = exp_avg_sqs[idx] if opts.is_dual else None
        param_has_master = (master_params is not None) and (master_params[idx] is not p)
        mp_ref = master_params[idx] if (param_has_master and master_params is not None) else None
        try:
            (
                n_elements, b_size, n_warps, c_grid, dt_mode, s_dt_mode, _,
                p_c, g_c, m_c, v_c, mp_c,
            ) = prepare_triton_buffers(opt, p, grads[idx], exp_avgs[idx], v_ref, mp_ref, param_has_master)
            launch_pass1(
                c_grid, p_c, mp_c, g_c, m_c, v_c, sc, n_elements,
                opts, c1, c2, dt_mode, s_dt_mode, param_has_master, b_size, n_warps,
                step=step,
            )
        except Exception:
            ref_p = mp_ref if (param_has_master and mp_ref is not None) else p
            compute_pass1_eager_one(p, grads[idx], exp_avgs[idx], v_ref, ref_p, sc, opts, step)

    totals = sync_fsdp_scratch_batch(
        [scratches[idx] for idx in pass1_indices],
        [params[idx].numel() for idx in pass1_indices],
    )
    return dict(zip(pass1_indices, totals))


def _finalize_coalesced_fsdp_ema(
    pending_ema: List[Tuple[Tensor, int, bool]],
    opts: StepOptions,
) -> None:
    if not pending_ema:
        return
    sc_list = [item[0] for item in pending_ema]
    numels = [item[1] for item in pending_ema]
    try:
        totals = sync_fsdp_scratch_batch(sc_list, numels)
        batch_synced = True
    except Exception:
        totals = [max(float(n), 1.0) for n in numels]
        batch_synced = False
    for (sc_p, n_el_p, allow_triton), tot_el in zip(pending_ema, totals):
        if batch_synced and allow_triton and n_el_p > 0:
            try:
                launch_post_step_ema(sc_p, tot_el, opts)
                continue
            except Exception:
                pass
        fallback_post_step_ema(
            sc_p, n_el_p, opts, scratch_synced=batch_synced, synced_total_elements=tot_el
        )


def _step_triton_single_device(
    opt: Any,
    opts: StepOptions,
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Optional[Tensor]],
    master_params: Optional[List[Tensor]],
    steps: List[Any],
    scratches: List[Tensor],
    orig_indices: Optional[List[int]],
    completed_indices: Optional[Set[int]],
) -> None:
    pending_ema: List[Tuple[Tensor, int, bool]] = []
    batched_totals: Dict[int, float] = {}
    handled_indices: Set[int] = set()

    if not opts.fsdp_sync and len(params) >= 2:
        mt_groups: Dict[Tuple[int, int, bool, int, bool], List[int]] = {}
        for i, p in enumerate(params):
            if p.numel() == 0:
                continue
            step_raw = steps[i]
            step_v = int(step_raw.item() if isinstance(step_raw, torch.Tensor) else step_raw)
            if opts.needs_ema_sync and not (opts.fused_single_pass and step_v > 1):
                continue
            g_i = grads[i]
            m_i = exp_avgs[i]
            v_i = exp_avg_sqs[i] if opts.is_dual else None
            has_mp = (master_params is not None) and (master_params[i] is not p)
            mp_i = master_params[i] if (has_mp and master_params is not None) else p
            if not (
                g_i.dtype == p.dtype
                and p.is_contiguous()
                and g_i.is_contiguous()
                and m_i.is_contiguous()
                and (v_i is None or v_i.is_contiguous())
                and (not has_mp or mp_i.is_contiguous())
            ):
                continue
            key = (
                dtype_mode_code(p.dtype),
                dtype_mode_code(m_i.dtype),
                has_mp,
                step_v,
                bool(opts.needs_ema_sync),
            )
            mt_groups.setdefault(key, []).append(i)

        for (dt_m, s_dt_m, has_mp, step_v, fused_ema_flag), idxs in mt_groups.items():
            if len(idxs) < 2:
                continue
            step_committed = False

            def _mark_committed() -> None:
                nonlocal step_committed
                step_committed = True
                for j in idxs:
                    handled_indices.add(j)
                    if completed_indices is not None:
                        completed_indices.add(orig_indices[j] if orig_indices is not None else j)

            try:
                launch_multi_tensor_step(
                    params[idxs[0]].device,
                    opts,
                    [params[j] for j in idxs],
                    [master_params[j] if (has_mp and master_params is not None) else params[j] for j in idxs],
                    [grads[j] for j in idxs],
                    [exp_avgs[j] for j in idxs],
                    [exp_avg_sqs[j] if opts.is_dual else None for j in idxs],
                    [scratches[j] for j in idxs],
                    [orig_indices[j] if orig_indices is not None else j for j in idxs],
                    step_v,
                    dt_m,
                    s_dt_m,
                    has_mp,
                    fused_ema=fused_ema_flag,
                    on_step_committed=_mark_committed,
                )
            except Exception:
                if step_committed:
                    raise

    if opts.fsdp_sync and opts.needs_ema_sync and len(params) > 1:
        pass1_indices = [
            idx
            for idx, st in enumerate(steps)
            if not (opts.fused_single_pass and int(st.item() if isinstance(st, torch.Tensor) else st) > 1)
        ]
        if len(pass1_indices) > 1:
            batched_totals = _precompute_fsdp_pass1(
                opt, pass1_indices, params, grads, exp_avgs, exp_avg_sqs, master_params, steps, scratches, opts
            )

    for i, p in enumerate(params):
        if i in handled_indices:
            continue
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i] if opts.is_dual else None
        step_raw = steps[i]
        sc = scratches[i]
        step = int(step_raw.item() if isinstance(step_raw, torch.Tensor) else step_raw)
        global_idx = orig_indices[i] if orig_indices is not None else i
        cur_synced = i in batched_totals
        cur_total = batched_totals[i] if cur_synced else float(p.numel())

        if p.numel() == 0:
            if opts.fsdp_sync:
                if not cur_synced:
                    sc[:3].zero_()
                sc[5].zero_()
                if opts.fused_single_pass and step > 1 and opts.needs_ema_sync:
                    pending_ema.append((sc, 0, False))
                else:
                    sync_empty_fsdp_shard(sc, step, opts, scratch_synced=cur_synced, synced_total_elements=cur_total)
            if completed_indices is not None:
                completed_indices.add(global_idx)
            continue

        param_has_master = (master_params is not None) and (master_params[i] is not p)
        mp_ref = master_params[i] if (param_has_master and master_params is not None) else None
        p_c, m_c, v_c = p, exp_avg, exp_avg_sq
        mp_c = mp_ref if (param_has_master and mp_ref is not None) else p
        is_all_contig = True
        scratch_synced = cur_synced
        param_updated = False
        ema_updated = False
        copied_back = False
        n_elements = p.numel()
        total_elements = cur_total
        defer_ema = bool(opts.fsdp_sync and opts.fused_single_pass and step > 1 and opts.needs_ema_sync)

        try:
            (
                n_elements, b_size, n_warps, c_grid, dt_mode, s_dt_mode, is_all_contig,
                p_c, g_c, m_c, v_c, mp_c,
            ) = prepare_triton_buffers(opt, p, grad, exp_avg, exp_avg_sq, mp_ref, param_has_master, reuse_contig=cur_synced)

            c1, c2 = opts.dual_coefficients(step) if opts.is_dual else (0.0, 0.0)
            kernel_seed = (opts.seed + global_idx * 10007) & 0x7FFFFFFF
            step_u32 = step & 0xFFFFFFFF
            step_i32 = step_u32 - 0x100000000 if step_u32 >= 0x80000000 else step_u32
            eff_use_sr = False if (s_dt_mode == 0 and (param_has_master or dt_mode == 0)) else opts.use_sr

            if not opts.needs_ema_sync:
                launch_single_pass(
                    c_grid, p_c, mp_c, g_c, m_c, v_c, sc, n_elements, opts, c1, c2,
                    opts.lr, opts.weight_decay, kernel_seed, step_i32, dt_mode, s_dt_mode,
                    eff_use_sr, param_has_master, b_size, n_warps, fused_ema=False, step=step,
                )
                param_updated = True
                ema_updated = True
            elif opts.fused_single_pass and step > 1:
                launch_single_pass(
                    c_grid, p_c, mp_c, g_c, m_c, v_c, sc, n_elements, opts, c1, c2,
                    opts.lr, opts.weight_decay, kernel_seed, step_i32, dt_mode, s_dt_mode,
                    eff_use_sr, param_has_master, b_size, n_warps, fused_ema=True, step=step,
                )
                param_updated = True
                if defer_ema:
                    pending_ema.append((sc, n_elements, True))
                else:
                    total_elements = float(n_elements)
                    if opts.fsdp_sync:
                        total_elements = sync_fsdp_scratch(sc, n_elements)
                        scratch_synced = True
                    launch_post_step_ema(sc, total_elements, opts)
                ema_updated = True
            else:
                if not scratch_synced:
                    sc[:3].zero_()
                    launch_pass1(
                        c_grid, p_c, mp_c, g_c, m_c, v_c, sc, n_elements,
                        opts, c1, c2, dt_mode, s_dt_mode, param_has_master, b_size, n_warps,
                        step=step,
                    )
                    total_elements = float(n_elements)
                    if opts.fsdp_sync:
                        total_elements = sync_fsdp_scratch(sc, n_elements)
                        scratch_synced = True
                launch_pass2(
                    c_grid, p_c, mp_c, g_c, m_c, v_c, sc, n_elements, total_elements, opts,
                    c1, c2, opts.lr, opts.weight_decay, kernel_seed, step_i32, dt_mode, s_dt_mode,
                    eff_use_sr, param_has_master, b_size, n_warps, step=step,
                )
                param_updated = True
                sc[:3].zero_()
                ema_updated = True

            if not is_all_contig:
                copy_back_non_contig(p, p_c, exp_avg, m_c, exp_avg_sq, v_c, mp_ref, mp_c, param_has_master)
                copied_back = True

            if completed_indices is not None:
                completed_indices.add(global_idx)
        except Exception as exc:
            if not getattr(opt, "_warned_triton_failure", False):
                err_line = str(exc).strip().splitlines()[0] if str(exc).strip() else "Unknown error"
                logger.warning("LuminaV: Triton kernel execution failed (%s). Falling back to eager.", err_line)
                opt._warned_triton_failure = True

            if p.is_cuda:
                with contextlib.suppress(Exception):
                    torch.cuda.synchronize(p.device)

            if param_updated:
                if not is_all_contig and not copied_back:
                    copy_back_non_contig(p, p_c, exp_avg, m_c, exp_avg_sq, v_c, mp_ref, mp_c, param_has_master)
                if (opts.fused_single_pass and step > 1) and opts.needs_ema_sync and not defer_ema and not ema_updated:
                    fallback_post_step_ema(
                        sc, n_elements, opts, scratch_synced=scratch_synced, synced_total_elements=total_elements
                    )
            else:
                if not scratch_synced:
                    sc[:3].zero_()
                sc[5].zero_()
                if step <= 1:
                    sc[3] = 1.0
                    sc[4] = 1.0
                step_eager_group(
                    opt, opts, [p], [grad], [exp_avg], [exp_avg_sq] if opts.is_dual else [None],
                    [mp_ref] if (param_has_master and mp_ref is not None) else None,
                    [step], [sc],
                    scratch_synced=scratch_synced,
                    scratch_total_elements=total_elements,
                    orig_indices=[global_idx],
                    defer_fsdp_post_ema=defer_ema,
                )
                if defer_ema:
                    pending_ema.append((sc, n_elements, False))
            if completed_indices is not None:
                completed_indices.add(global_idx)

    if pending_ema:
        _finalize_coalesced_fsdp_ema(pending_ema, opts)
        if completed_indices is not None:
            for idx_done in range(len(params)):
                completed_indices.add(orig_indices[idx_done] if orig_indices is not None else idx_done)


def dispatch_triton_group(
    opt: Any,
    opts: StepOptions,
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Optional[Tensor]],
    master_params: Optional[List[Tensor]],
    steps: List[Any],
    scratches: List[Tensor],
    *,
    orig_indices: Optional[List[int]] = None,
    completed_indices: Optional[Set[int]] = None,
) -> None:
    primary_dev = params[0].device
    if all(p.device == primary_dev for p in params):
        with _get_device_context(primary_dev):
            _step_triton_single_device(
                opt, opts, params, grads, exp_avgs, exp_avg_sqs, master_params,
                steps, scratches, orig_indices, completed_indices,
            )
        return

    device_groups: Dict[torch.device, List[int]] = {}
    for idx, p in enumerate(params):
        device_groups.setdefault(p.device, []).append(idx)

    for dev, indices in device_groups.items():
        with _get_device_context(dev):
            _step_triton_single_device(
                opt, opts,
                [params[i] for i in indices],
                [grads[i] for i in indices],
                [exp_avgs[i] for i in indices],
                [exp_avg_sqs[i] for i in indices] if opts.is_dual else [],
                [master_params[i] for i in indices] if master_params is not None else None,
                [steps[i] for i in indices],
                [scratches[i] for i in indices],
                [orig_indices[i] if orig_indices is not None else i for i in indices],
                completed_indices,
            )


def prewarm_all_configs(opt: Any) -> None:
    """Pre-compile Triton GPU kernels on 64-element dummy buffers (lr=0.0) to avoid first-step JIT latency."""
    if not HAS_TRITON or triton is None or not getattr(opt, "param_groups", None):
        return

    deterministic_mode = (
        torch.are_deterministic_algorithms_enabled()
        if hasattr(torch, "are_deterministic_algorithms_enabled")
        else False
    )
    seen_configs: Set[Any] = getattr(opt, "_prewarmed_triton_configs", set())
    opt._prewarmed_triton_configs = seen_configs
    fused_buckets: List[Dict[str, Any]] = getattr(opt, "_fused_buckets", [])

    for group in opt.param_groups:
        if not bool(group.get("prewarm_jit", True)):
            continue
        exec_mode = group.get("execution", "auto")
        if deterministic_mode or exec_mode not in ("auto", "triton"):
            continue

        opts = StepOptions.from_group(group, fallback_seed=getattr(opt, "seed", 1337))
        targets: Set[Tuple[torch.device, torch.dtype, int, int]] = set()
        group_buckets = [b for b in fused_buckets if b.get("group") is group]
        if bool(group.get("fuse_tensors", False)) and group_buckets:
            for b in group_buckets:
                if b["device"].type in ("cuda", "xpu"):
                    b_size, n_warps = block_config(int(b["flat_p"].numel()))
                    targets.add((b["device"], b["dtype"], b_size, n_warps))
        else:
            for p in group.get("params", []):
                if isinstance(p, torch.Tensor) and p.device.type in ("cuda", "xpu") and p.numel() > 0:
                    b_size, n_warps = block_config(int(p.numel()))
                    targets.add((p.device, p.dtype, b_size, n_warps))

        c1, c2 = opts.dual_coefficients(1) if opts.is_dual else (0.0, 0.0)
        for dev, dt, b_size, n_warps in targets:
            dt_mode = dtype_mode_code(dt)
            state_dtype = torch.float32 if (opts.master_mode == 2 or dt == torch.float32) else dt
            s_dt_mode = dtype_mode_code(state_dtype)
            has_master = (opts.master_mode > 0) and (dt in (torch.float16, torch.bfloat16))
            eff_use_sr = False if (s_dt_mode == 0 and (has_master or dt_mode == 0)) else opts.use_sr
            cfg_key = (
                dev.type, dev.index or 0, dt_mode, s_dt_mode, has_master,
                opts.buffer, opts.cautious, opts.bound_mode, opts.fused_single_pass, eff_use_sr,
                opts.variance_type, opts.cautious_weight_decay, b_size, n_warps,
            )
            if cfg_key in seen_configs:
                continue
            seen_configs.add(cfg_key)

            with contextlib.suppress(Exception), _get_device_context(dev):
                dummy_p = torch.zeros(64, dtype=dt, device=dev)
                dummy_g = torch.zeros(64, dtype=dt, device=dev)
                dummy_m = torch.zeros(64, dtype=state_dtype, device=dev)
                dummy_v = torch.zeros(64, dtype=state_dtype, device=dev) if opts.is_dual else None
                dummy_mp = torch.zeros(64, dtype=torch.float32, device=dev) if has_master else dummy_p
                dummy_sc = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float32, device=dev)

                if not opts.needs_ema_sync:
                    launch_single_pass(
                        (1,), dummy_p, dummy_mp, dummy_g, dummy_m, dummy_v, dummy_sc, 64,
                        opts, c1, c2, 0.0, 0.0, 1337, 1, dt_mode, s_dt_mode, eff_use_sr,
                        has_master, b_size, n_warps, fused_ema=False, step=1,
                    )
                else:
                    launch_pass1(
                        (1,), dummy_p, dummy_mp, dummy_g, dummy_m, dummy_v, dummy_sc, 64,
                        opts, c1, c2, dt_mode, s_dt_mode, has_master, b_size, n_warps, step=1,
                    )
                    launch_pass2(
                        (1,), dummy_p, dummy_mp, dummy_g, dummy_m, dummy_v, dummy_sc, 64, 64.0,
                        opts, c1, c2, 0.0, 0.0, 1337, 1, dt_mode, s_dt_mode, eff_use_sr,
                        has_master, b_size, n_warps, step=1,
                    )
                    if opts.fused_single_pass:
                        dummy_sc[:3].zero_()
                        dummy_sc[3] = 1.0
                        dummy_sc[4] = 1.0
                        launch_single_pass(
                            (1,), dummy_p, dummy_mp, dummy_g, dummy_m, dummy_v, dummy_sc, 64,
                            opts, c1, c2, 0.0, 0.0, 1337, 2, dt_mode, s_dt_mode, eff_use_sr,
                            has_master, b_size, n_warps, fused_ema=True, step=2,
                        )
                        launch_post_step_ema(dummy_sc, 64.0, opts, lr_override=0.0)

                if dev.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize(dev)
