from __future__ import annotations

from typing import Any, List, Optional

import torch
from torch import Tensor

from ...config import StepOptions
from ..fsdp import fallback_post_step_ema
from .common import (
    HAS_TRITON,
    _derive_seeds_and_weyl_dual,
    _derive_seeds_and_weyl_single,
    _rne_bf16_in_fp16_container,
    _sr_bf16_bits,
    _sr_bf16_in_fp16_container,
    _sr_fp16_bits,
    _triton_apply_rpm,
    _triton_apply_step_bounds,
    _triton_apply_wd,
    _triton_tanh_fast,
    _triton_temper_defect,
    tl,
    triton,
)

_MULTI_TENSOR_BLOCK_SIZE: int = 2048
_MULTI_TENSOR_NUM_WARPS: int = 8

if HAS_TRITON:

    @triton.jit
    def _load_typed(raw_ptr, offs_i64, mask, dtype_mode: tl.constexpr):
        if dtype_mode == 1:
            return tl.load(raw_ptr.to(tl.pointer_type(tl.bfloat16)) + offs_i64, mask=mask, other=0.0).to(tl.float32)
        if dtype_mode == 2:
            return tl.load(raw_ptr.to(tl.pointer_type(tl.float16)) + offs_i64, mask=mask, other=0.0).to(tl.float32)
        return tl.load(raw_ptr.to(tl.pointer_type(tl.float32)) + offs_i64, mask=mask, other=0.0)

    @triton.jit
    def _load_state_typed(raw_ptr, offs_i64, mask, state_dtype_mode: tl.constexpr):
        if state_dtype_mode == 2:
            raw_f16 = tl.load(raw_ptr.to(tl.pointer_type(tl.float16)) + offs_i64, mask=mask, other=0.0)
            bits_i32 = raw_f16.to(tl.int16, bitcast=True).to(tl.int32) << 16
            return bits_i32.to(tl.float32, bitcast=True)
        if state_dtype_mode == 1:
            return tl.load(raw_ptr.to(tl.pointer_type(tl.bfloat16)) + offs_i64, mask=mask, other=0.0).to(tl.float32)
        return tl.load(raw_ptr.to(tl.pointer_type(tl.float32)) + offs_i64, mask=mask, other=0.0)

    @triton.jit
    def _store_param_typed(
        raw_ptr, offs_i64, offs_i32, val, mask, seed, weyl, dtype_mode: tl.constexpr, use_sr: tl.constexpr
    ):
        if dtype_mode == 1:
            ptr = raw_ptr.to(tl.pointer_type(tl.bfloat16)) + offs_i64
            if use_sr:
                tl.store(ptr, _sr_bf16_bits(val, seed, offs_i32, weyl), mask=mask)
            else:
                v_c = tl.maximum(tl.minimum(val, 3.3895313892515355e38), -3.3895313892515355e38)
                tl.store(ptr, v_c.to(tl.bfloat16), mask=mask)
        elif dtype_mode == 2:
            ptr = raw_ptr.to(tl.pointer_type(tl.float16)) + offs_i64
            if use_sr:
                tl.store(ptr, _sr_fp16_bits(val, seed, offs_i32, weyl), mask=mask)
            else:
                v_c = tl.maximum(tl.minimum(val, 65504.0), -65504.0)
                tl.store(ptr, v_c.to(tl.float16), mask=mask)
        else:
            tl.store(raw_ptr.to(tl.pointer_type(tl.float32)) + offs_i64, val, mask=mask)

    @triton.jit
    def _store_state_typed(
        raw_ptr, offs_i64, offs_i32, val, mask, seed, weyl, state_dtype_mode: tl.constexpr, use_sr: tl.constexpr
    ):
        if state_dtype_mode == 1:
            ptr = raw_ptr.to(tl.pointer_type(tl.bfloat16)) + offs_i64
            if use_sr:
                tl.store(ptr, _sr_bf16_bits(val, seed, offs_i32, weyl), mask=mask)
            else:
                v_c = tl.maximum(tl.minimum(val, 3.3895313892515355e38), -3.3895313892515355e38)
                tl.store(ptr, v_c.to(tl.bfloat16), mask=mask)
        elif state_dtype_mode == 2:
            ptr = raw_ptr.to(tl.pointer_type(tl.float16)) + offs_i64
            if use_sr:
                tl.store(ptr, _sr_bf16_in_fp16_container(val, seed, offs_i32, weyl), mask=mask)
            else:
                tl.store(ptr, _rne_bf16_in_fp16_container(val), mask=mask)
        else:
            tl.store(raw_ptr.to(tl.pointer_type(tl.float32)) + offs_i64, val, mask=mask)

    @triton.jit
    def _flush_scratch(
        sc_raw, acc_mask, acc_u_sq, acc_p_sq, cautious: tl.constexpr, bound_mode: tl.constexpr
    ):
        sc_ptr = sc_raw.to(tl.pointer_type(tl.float32))
        if cautious:
            tl.atomic_add(sc_ptr + 0, acc_mask, sem="relaxed")
        if bound_mode == 1:
            tl.atomic_add(sc_ptr + 1, acc_u_sq, sem="relaxed")
            tl.atomic_add(sc_ptr + 2, acc_p_sq, sem="relaxed")

    @triton.jit(do_not_specialize=["step_i32", "total_blocks", "num_tensors"])
    def _lumina_multi_tensor_step_kernel(
        p_ptrs_arr,
        mp_ptrs_arr,
        g_ptrs_arr,
        m_ptrs_arr,
        v_ptrs_arr,
        sc_ptrs_arr,
        numels_arr,
        block_offs_arr,
        seeds_arr,
        num_tensors,
        total_blocks,
        beta1,
        beta2,
        c1,
        c2,
        lr,
        weight_decay,
        eps,
        lam,
        bound_ratio,
        bound_floor,
        variance_temper,
        inv_sqrt_bc2_prev,
        step_i32,
        MODE: tl.constexpr,  # 0=T1, 1=TD-CLM, 2=T2, 3=Dual
        bound_mode: tl.constexpr,
        dtype_mode: tl.constexpr,
        state_dtype_mode: tl.constexpr,
        use_sr: tl.constexpr,
        cautious: tl.constexpr,
        has_master: tl.constexpr,
        use_rtv: tl.constexpr,
        use_rpm: tl.constexpr,
        cautious_wd: tl.constexpr,
        NEED_STATS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0).to(tl.int64)
        num_progs = tl.num_programs(axis=0).to(tl.int64)
        blocks_per_cta = (total_blocks + num_progs - 1) // num_progs
        blk_start = pid * blocks_per_cta
        blk_end = tl.minimum(blk_start + blocks_per_cta, total_blocks)
        if blk_start >= blk_end:
            return

        # Binary search starting tensor index t such that block_offs[t] <= blk_start < block_offs[t + 1].
        lo = tl.zeros([], dtype=tl.int64)
        hi = lo + num_tensors
        while lo + 1 < hi:
            mid = (lo + hi) >> 1
            if tl.load(block_offs_arr + mid) <= blk_start:
                lo = mid
            else:
                hi = mid
        t = lo

        t_blk_start = tl.load(block_offs_arr + t)
        t_blk_end = tl.load(block_offs_arr + t + 1)
        n_t = tl.load(numels_arr + t)
        p_raw = tl.load(p_ptrs_arr + t)
        g_raw = tl.load(g_ptrs_arr + t)
        m_raw = tl.load(m_ptrs_arr + t)
        v_raw = tl.load(v_ptrs_arr + t) if (MODE == 2 or MODE == 3) else p_raw
        mp_raw = tl.load(mp_ptrs_arr + t) if has_master else p_raw
        sc_raw = tl.load(sc_ptrs_arr + t)
        seed_t = tl.load(seeds_arr + t)

        sc_ptr = sc_raw.to(tl.pointer_type(tl.float32))
        cautious_scale = tl.load(sc_ptr + 3) if cautious else 1.0
        bound_scale = tl.load(sc_ptr + 4) if bound_mode == 1 else 1.0

        acc_mask = 0.0
        acc_u_sq = 0.0
        acc_p_sq = 0.0
        lane = tl.arange(0, BLOCK_SIZE).to(tl.int64)
        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq

        for blk in range(blk_start, blk_end):
            if blk >= t_blk_end:
                if NEED_STATS:
                    _flush_scratch(sc_raw, acc_mask, acc_u_sq, acc_p_sq, cautious, bound_mode)
                t += 1
                while blk >= tl.load(block_offs_arr + t + 1):
                    t += 1
                t_blk_start = tl.load(block_offs_arr + t)
                t_blk_end = tl.load(block_offs_arr + t + 1)
                n_t = tl.load(numels_arr + t)
                p_raw = tl.load(p_ptrs_arr + t)
                g_raw = tl.load(g_ptrs_arr + t)
                m_raw = tl.load(m_ptrs_arr + t)
                if MODE == 2 or MODE == 3:
                    v_raw = tl.load(v_ptrs_arr + t)
                if has_master:
                    mp_raw = tl.load(mp_ptrs_arr + t)
                sc_raw = tl.load(sc_ptrs_arr + t)
                seed_t = tl.load(seeds_arr + t)
                sc_ptr = sc_raw.to(tl.pointer_type(tl.float32))
                if cautious:
                    cautious_scale = tl.load(sc_ptr + 3)
                if bound_mode == 1:
                    bound_scale = tl.load(sc_ptr + 4)
                acc_mask = 0.0
                acc_u_sq = 0.0
                acc_p_sq = 0.0

            offs_i64 = (blk - t_blk_start) * BLOCK_SIZE + lane
            offs_i32 = offs_i64.to(tl.int32)
            mask = offs_i64 < n_t

            if has_master:
                p = tl.load(mp_raw.to(tl.pointer_type(tl.float32)) + offs_i64, mask=mask, other=0.0)
            else:
                p = _load_typed(p_raw, offs_i64, mask, dtype_mode)
            p_init = p

            g = _load_typed(g_raw, offs_i64, mask, dtype_mode)
            is_finite_g = (g == g) & (g < 1e38) & (g > -1e38)
            g = tl.where(is_finite_g, tl.maximum(tl.minimum(g, 1e18), -1e18), 0.0)
            m = _load_state_typed(m_raw, offs_i64, mask, state_dtype_mode)
            if use_rpm:
                m = _triton_apply_rpm(m, g, beta1, eps, BLOCK_SIZE)

            if MODE == 0 or MODE == 1:
                seed_m, seed_p, weyl_m, weyl_p = _derive_seeds_and_weyl_single(seed_t, step_i32)
                m_new = beta1 * m + (1.0 - beta1) * g
                clm_m = beta1_sq * m + one_minus_beta1_sq * g
                diff = tl.maximum(tl.minimum(g - m, 1e18), -1e18)
                clm_env = beta1_sq * tl.abs(m) + one_minus_beta1_sq * tl.abs(g) + lam * tl.abs(diff) + eps
                u = clm_m / clm_env
                _store_state_typed(m_raw, offs_i64, offs_i32, m_new, mask, seed_m, weyl_m, state_dtype_mode, use_sr)
            elif MODE == 2:
                seed_m, seed_p, seed_v, weyl_m, weyl_v, weyl_p = _derive_seeds_and_weyl_dual(seed_t, step_i32)
                v = _load_state_typed(v_raw, offs_i64, mask, state_dtype_mode)
                m_new = beta1 * m + (1.0 - beta1) * g
                diff = tl.maximum(tl.minimum(g - m, 1e18), -1e18)
                abs_d = tl.abs(diff)
                if use_rtv:
                    d_mem = _triton_temper_defect(abs_d, v, m, inv_sqrt_bc2_prev, variance_temper, eps)
                else:
                    d_mem = abs_d
                v_new = beta2 * v + (1.0 - beta2) * (d_mem * d_mem)
                num = beta1_sq * m + one_minus_beta1_sq * g
                sigma_t = tl.sqrt(tl.maximum(v_new, 0.0)) * c1
                den = beta1_sq * tl.abs(m) + one_minus_beta1_sq * tl.abs(g) + c2 * abs_d + sigma_t + eps
                u = num / den
                _store_state_typed(m_raw, offs_i64, offs_i32, m_new, mask, seed_m, weyl_m, state_dtype_mode, use_sr)
                _store_state_typed(v_raw, offs_i64, offs_i32, v_new, mask, seed_v, weyl_v, state_dtype_mode, use_sr)
            else:
                seed_m, seed_p, seed_v, weyl_m, weyl_v, weyl_p = _derive_seeds_and_weyl_dual(seed_t, step_i32)
                v = _load_state_typed(v_raw, offs_i64, mask, state_dtype_mode)
                m_new = beta1 * m + (1.0 - beta1) * g
                clm_m = beta1 * m_new + (1.0 - beta1) * g
                diff = tl.maximum(tl.minimum(g - m_new, 1e18), -1e18)
                abs_d = tl.abs(diff)
                if use_rtv:
                    d_mem = _triton_temper_defect(abs_d, v, m, inv_sqrt_bc2_prev, variance_temper, eps)
                else:
                    d_mem = abs_d
                v_new = beta2 * v + (1.0 - beta2) * (d_mem * d_mem)
                sigma = tl.sqrt(tl.maximum(v_new, 0.0)) * c1 + c2
                u = _triton_tanh_fast(clm_m / tl.maximum(sigma, 1e-12))
                _store_state_typed(m_raw, offs_i64, offs_i32, m_new, mask, seed_m, weyl_m, state_dtype_mode, use_sr)
                _store_state_typed(v_raw, offs_i64, offs_i32, v_new, mask, seed_v, weyl_v, state_dtype_mode, use_sr)

            m_mask = tl.where((u * g) > 0.0, 1.0, 0.0)
            delta_theta = (u * m_mask) * cautious_scale if cautious else u
            step_val = lr * delta_theta

            step_val = _triton_apply_step_bounds(
                step_val, p_init, bound_ratio, bound_floor, bound_scale, bound_mode, dtype_mode, (MODE == 0 or MODE == 2)
            )

            p = _triton_apply_wd(p_init, step_val, lr, weight_decay, cautious_wd)
            p_updated = p - step_val

            if has_master:
                tl.store(mp_raw.to(tl.pointer_type(tl.float32)) + offs_i64, p_updated, mask=mask)
                _store_param_typed(p_raw, offs_i64, offs_i32, p_updated, mask, seed_p, weyl_p, dtype_mode, False)
            else:
                _store_param_typed(p_raw, offs_i64, offs_i32, p_updated, mask, seed_p, weyl_p, dtype_mode, use_sr)

            if NEED_STATS:
                if cautious:
                    acc_mask += tl.sum(tl.where(mask, m_mask, 0.0), axis=0)  # type: ignore[operator]
                if bound_mode == 1:
                    u_eff = (u * m_mask) if cautious else u
                    acc_u_sq += tl.sum(tl.where(mask, u_eff * u_eff, 0.0), axis=0)  # type: ignore[operator]
                    acc_p_sq += tl.sum(tl.where(mask, p_init * p_init, 0.0), axis=0)  # type: ignore[operator]

        if NEED_STATS:
            _flush_scratch(sc_raw, acc_mask, acc_u_sq, acc_p_sq, cautious, bound_mode)

    @triton.jit
    def _lumina_post_step_ema_multi_kernel(
        sc_ptrs_arr,
        numels_arr,
        lr,
        clamp_min,
        ema_decay,
        bound_mode: tl.constexpr,
        bound_ratio,
        cautious: tl.constexpr,
        is_triangle: tl.constexpr,
    ):
        t = tl.program_id(axis=0).to(tl.int64)
        sc_ptr = tl.load(sc_ptrs_arr + t).to(tl.pointer_type(tl.float32))
        total_elements = tl.load(numels_arr + t).to(tl.float32)

        if cautious:
            mask_sum = tl.load(sc_ptr + 0)
            m_bar = tl.minimum(tl.maximum(mask_sum / total_elements, clamp_min), 1.0)
            target_cautious = 1.0 / m_bar
            old_cautious = tl.load(sc_ptr + 3)
            tl.store(sc_ptr + 3, ema_decay * old_cautious + (1.0 - ema_decay) * target_cautious)

        if bound_mode == 1:
            u_sq_sum = tl.load(sc_ptr + 1)
            p_sq_sum = tl.load(sc_ptr + 2)
            c_scale = tl.load(sc_ptr + 3) if cautious else 1.0
            norm_p = tl.sqrt(tl.maximum(p_sq_sum, 0.0))
            norm_delta = tl.sqrt(tl.maximum(u_sq_sum, 0.0)) * c_scale
            radius = bound_ratio * tl.maximum(norm_p, 1.0)
            r = (lr * norm_delta) / radius
            if is_triangle:
                target_scale = (1.0 + r) / (1.0 + r + r * r)
            else:
                target_scale = tl.where(r < 1e-4, 1.0, _triton_tanh_fast(r) / tl.maximum(r, 1e-8))
            old_bound = tl.load(sc_ptr + 4)
            gamma_eff = tl.where(target_scale < old_bound, tl.minimum(ema_decay, 0.10), ema_decay)
            tl.store(sc_ptr + 4, gamma_eff * old_bound + (1.0 - gamma_eff) * target_scale)

        tl.store(sc_ptr + 0, 0.0)
        tl.store(sc_ptr + 1, 0.0)
        tl.store(sc_ptr + 2, 0.0)
else:
    _lumina_multi_tensor_step_kernel: Any = None
    _lumina_post_step_ema_multi_kernel: Any = None


def _mode_code(opts: StepOptions) -> int:
    if opts.buffer == "t1":
        return 0
    if opts.buffer == 1:
        return 1
    if opts.buffer == "t2":
        return 2
    return 3


def launch_multi_tensor_post_step_ema(
    device: torch.device,
    opts: StepOptions,
    sc_list: List[Tensor],
    total_elements_list: List[int],
) -> None:
    """Batched post-step EMA update for multiple scratch buffers in a single Triton launch."""
    num_tensors = len(sc_list)
    if num_tensors == 0:
        return
    sc_ptrs = [int(sc.data_ptr()) for sc in sc_list]
    meta_i64 = torch.tensor(sc_ptrs + total_elements_list, dtype=torch.int64, device=device)
    sc_arr = meta_i64[:num_tensors]
    numels_arr = meta_i64[num_tensors : 2 * num_tensors]
    _lumina_post_step_ema_multi_kernel[(num_tensors,)](
        sc_arr, numels_arr, opts.lr, opts.clamp_min, opts.ema_decay,
        opts.bound_mode, opts.bound_ratio, opts.cautious,
        is_triangle=opts.is_triangle, num_warps=1, enable_fp_fusion=True,
    )


def launch_multi_tensor_step(
    device: torch.device,
    opts: StepOptions,
    p_list: List[Tensor],
    mp_list: List[Tensor],
    g_list: List[Tensor],
    m_list: List[Tensor],
    v_list: List[Optional[Tensor]],
    sc_list: List[Tensor],
    global_indices: List[int],
    step: int,
    dt_mode: int,
    s_dt_mode: int,
    has_master: bool,
    *,
    fused_ema: bool,
    defer_ema: bool = False,
    on_step_committed: Optional[Any] = None,
) -> None:
    """Executes all homogeneous contiguous tensors in 1 persistent step + 1 batched EMA launch."""
    num_tensors = len(p_list)
    if num_tensors == 0:
        return

    b_size = _MULTI_TENSOR_BLOCK_SIZE
    numels: List[int] = []
    block_offs: List[int] = [0]
    p_ptrs: List[int] = []
    mp_ptrs: List[int] = []
    g_ptrs: List[int] = []
    m_ptrs: List[int] = []
    v_ptrs: List[int] = []
    sc_ptrs: List[int] = []
    seeds: List[int] = []

    running_blocks = 0
    for idx in range(num_tensors):
        p = p_list[idx]
        n = int(p.numel())
        numels.append(n)
        running_blocks += (n + b_size - 1) // b_size
        block_offs.append(running_blocks)
        p_ptrs.append(int(p.data_ptr()))
        mp_ptrs.append(int(mp_list[idx].data_ptr()) if has_master else int(p.data_ptr()))
        g_ptrs.append(int(g_list[idx].data_ptr()))
        m_ptrs.append(int(m_list[idx].data_ptr()))
        v_t = v_list[idx] if opts.is_dual else None
        v_ptrs.append(int(v_t.data_ptr()) if v_t is not None else int(p.data_ptr()))
        sc_ptrs.append(int(sc_list[idx].data_ptr()))
        seeds.append((opts.seed + global_indices[idx] * 10007) & 0x7FFFFFFF)

    if running_blocks == 0:
        return

    meta_i64 = torch.tensor(
        p_ptrs + mp_ptrs + g_ptrs + m_ptrs + v_ptrs + sc_ptrs + numels + block_offs,
        dtype=torch.int64,
        device=device,
    )
    seeds_t = torch.tensor(seeds, dtype=torch.int32, device=device)

    p_arr = meta_i64[0:num_tensors]
    mp_arr = meta_i64[num_tensors : 2 * num_tensors]
    g_arr = meta_i64[2 * num_tensors : 3 * num_tensors]
    m_arr = meta_i64[3 * num_tensors : 4 * num_tensors]
    v_arr = meta_i64[4 * num_tensors : 5 * num_tensors]
    sc_arr = meta_i64[5 * num_tensors : 6 * num_tensors]
    numels_arr = meta_i64[6 * num_tensors : 7 * num_tensors]
    block_offs_arr = meta_i64[7 * num_tensors : 8 * num_tensors + 1]

    num_sms = 64
    if device.type == "cuda" and torch.cuda.is_available():
        num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    elif device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        props = torch.xpu.get_device_properties(device)
        num_sms = int(getattr(props, "max_compute_units", getattr(props, "gpu_subslice_count", 64)))
    num_progs = min(running_blocks, max(1, num_sms * 4))

    c1, c2 = opts.dual_coefficients(step) if opts.is_dual else (0.0, 0.0)
    use_rtv = opts.use_rtv(step)
    inv_sqrt_bc2_prev = opts.inv_sqrt_bc2_prev(step)
    step_u32 = step & 0xFFFFFFFF
    step_i32 = step_u32 - 0x100000000 if step_u32 >= 0x80000000 else step_u32
    eff_use_sr = False if (s_dt_mode == 0 and (has_master or dt_mode == 0)) else opts.use_sr
    mode = _mode_code(opts)

    _lumina_multi_tensor_step_kernel[(num_progs,)](
        p_arr, mp_arr, g_arr, m_arr, v_arr, sc_arr, numels_arr, block_offs_arr, seeds_t,
        num_tensors, running_blocks,
        opts.beta1, opts.beta2, c1, c2, opts.lr, opts.weight_decay, opts.eps, opts.lam,
        opts.bound_ratio, opts.bound_floor, opts.variance_temper, inv_sqrt_bc2_prev, step_i32,
        MODE=mode,
        bound_mode=opts.bound_mode,
        dtype_mode=dt_mode,
        state_dtype_mode=s_dt_mode,
        use_sr=eff_use_sr,
        cautious=opts.cautious,
        has_master=has_master,
        use_rtv=use_rtv,
        use_rpm=opts.use_rpm,
        cautious_wd=opts.cautious_weight_decay,
        NEED_STATS=fused_ema,
        BLOCK_SIZE=b_size,
        num_warps=_MULTI_TENSOR_NUM_WARPS,
        enable_fp_fusion=True,
    )
    if on_step_committed is not None:
        on_step_committed()

    if fused_ema and not defer_ema:
        try:
            _lumina_post_step_ema_multi_kernel[(num_tensors,)](
                sc_arr, numels_arr, opts.lr, opts.clamp_min, opts.ema_decay,
                opts.bound_mode, opts.bound_ratio, opts.cautious,
                is_triangle=opts.is_triangle, num_warps=1, enable_fp_fusion=True,
            )
        except Exception:
            for sc_t, n_t in zip(sc_list, numels):
                fallback_post_step_ema(sc_t, n_t, opts, scratch_synced=False)
