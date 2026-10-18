from __future__ import annotations

import contextlib
import logging
import math
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from torch import Tensor
from torch.optim import Optimizer

__all__ = ["LuminaV"]
__version__ = "1.5.3"
__author__ = "Silver Moon (cloverxion)"
__organization__ = "Lumina Moon (cloverx-id)"
__license__ = "Apache-2.0"

logger = logging.getLogger("LuminaV")

HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def _get_device_context(dev: torch.device):
    if dev.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.device(dev)
    elif hasattr(torch, "xpu") and dev.type == "xpu" and torch.xpu.is_available():
        return torch.xpu.device(dev)
    elif hasattr(torch, "accelerator") and hasattr(torch.accelerator, "device"):
        return torch.accelerator.device(dev)
    return contextlib.nullcontext()


if HAS_TRITON:
    @triton.jit
    def _prng(seed, idx):
        z = (idx + seed - 1640531527).to(tl.int32)
        z = ((z ^ (z >> 16)) * -2048173461).to(tl.int32)
        z = ((z ^ (z >> 13)) * -1028477387).to(tl.int32)
        z = (z ^ (z >> 16)).to(tl.int32)
        return z

    @triton.jit
    def _sr_bf16_bits(val_f32, seed, offsets):
        p_int = val_f32.to(tl.int32, bitcast=True)
        abs_int = p_int & 0x7FFFFFFF
        noise = (_prng(seed, offsets) & 0xFFFF).to(tl.int32)
        rounded_abs = tl.minimum((abs_int + noise) & -65536, 0x7F7F0000)
        sign_bit = p_int & -2147483648
        out_int = sign_bit | rounded_abs
        return out_int.to(tl.float32, bitcast=True).to(tl.bfloat16)

    @triton.jit
    def _sr_fp16_bits(val_f32, seed, offsets):
        p_int = val_f32.to(tl.int32, bitcast=True)
        abs_int = p_int & 0x7FFFFFFF
        exp_f32 = (abs_int >> 23) & 0xFF
        sub_shift = tl.minimum(tl.maximum(113 - exp_f32, 0), 11)
        total_shift = 13 + sub_shift
        mask_bits = (1 << total_shift) - 1
        noise = _prng(seed, offsets) & mask_bits
        rounded_abs = tl.where(exp_f32 < 103, 0, (abs_int + noise) & (~mask_bits))
        rounded_abs = tl.minimum(rounded_abs, 0x477FE000)
        sign_bit = p_int & -2147483648
        out_int = sign_bit | rounded_abs
        return out_int.to(tl.float32, bitcast=True).to(tl.float16)

    @triton.jit
    def _store_param(p_ptr, offsets, p_val, mask, seed, dtype_mode: tl.constexpr, use_sr: tl.constexpr):
        if use_sr:
            if dtype_mode == 1:
                p_out = _sr_bf16_bits(p_val, seed, offsets)
                tl.store(p_ptr + offsets, p_out, mask=mask)
            elif dtype_mode == 2:
                p_out = _sr_fp16_bits(p_val, seed, offsets)
                tl.store(p_ptr + offsets, p_out, mask=mask)
            else:
                tl.store(p_ptr + offsets, p_val, mask=mask)
        else:
            if dtype_mode == 1:
                p_clamped = tl.maximum(tl.minimum(p_val, 3.3895313892515355e38), -3.3895313892515355e38)
                tl.store(p_ptr + offsets, p_clamped.to(tl.bfloat16), mask=mask)
            elif dtype_mode == 2:
                p_clamped = tl.maximum(tl.minimum(p_val, 65504.0), -65504.0)
                tl.store(p_ptr + offsets, p_clamped.to(tl.float16), mask=mask)
            else:
                tl.store(p_ptr + offsets, p_val, mask=mask)

    @triton.jit
    def _store_state_buffer(
        ptr, offsets, val, mask, seed,
        state_dtype_mode: tl.constexpr, use_sr: tl.constexpr
    ):
        if use_sr:
            if state_dtype_mode == 1:
                p_out = _sr_bf16_bits(val, seed, offsets)
                tl.store(ptr + offsets, p_out, mask=mask)
            elif state_dtype_mode == 2:
                p_out = _sr_fp16_bits(val, seed, offsets)
                tl.store(ptr + offsets, p_out, mask=mask)
            else:
                tl.store(ptr + offsets, val, mask=mask)
        else:
            if state_dtype_mode == 1:
                val_clamped = tl.maximum(tl.minimum(val, 3.3895313892515355e38), -3.3895313892515355e38)
                tl.store(ptr + offsets, val_clamped.to(tl.bfloat16), mask=mask)
            elif state_dtype_mode == 2:
                val_clamped = tl.maximum(tl.minimum(val, 65504.0), -65504.0)
                tl.store(ptr + offsets, val_clamped.to(tl.float16), mask=mask)
            else:
                tl.store(ptr + offsets, val, mask=mask)

    _HAS_TL_MATH_TANH = hasattr(tl, "math") and hasattr(tl.math, "tanh")
    _HAS_TL_ROOT_TANH = hasattr(tl, "tanh")

    if _HAS_TL_MATH_TANH:
        @triton.jit
        def _triton_tanh_fast(x):
            return tl.math.tanh(x)
    elif _HAS_TL_ROOT_TANH:
        @triton.jit
        def _triton_tanh_fast(x):
            return tl.tanh(x)
    else:
        @triton.jit
        def _triton_tanh_fast(x):
            x_clamped = tl.maximum(tl.minimum(x, 20.0), -20.0)
            return (2.0 / (1.0 + tl.exp(-2.0 * x_clamped))) - 1.0

    @triton.jit
    def _lumina_diff_sum_kernel(
        grad_ptr, exp_avg_ptr, scratch_ptr,
        n_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        is_finite_g = (g == g) & (g < 1e38) & (g > -1e38)
        g = tl.where(is_finite_g, tl.maximum(tl.minimum(g, 1e18), -1e18), 0.0)
        m = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        diff_gm = tl.maximum(tl.minimum(g - m, 1e18), -1e18)
        block_diff_sum = tl.sum(tl.where(mask, diff_gm, 0.0), axis=0)
        tl.atomic_add(scratch_ptr + 5, block_diff_sum)

    @triton.jit
    def _lumina_dual_fused_single_pass_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
        scratch_ptr,
        n_elements, total_elements, beta1, beta2, c1, c2, lr, weight_decay,
        bound_mode: tl.constexpr, bound_ratio,
        seed, dtype_mode: tl.constexpr, state_dtype_mode: tl.constexpr, use_sr: tl.constexpr,
        cautious: tl.constexpr, has_master: tl.constexpr, centered_defect: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        seed_m = seed
        seed_p = (seed + 0x55555555) & 0x7FFFFFFF

        mask_sum_ptr = scratch_ptr + 0
        u_sq_sum_ptr = scratch_ptr + 1
        p_sq_sum_ptr = scratch_ptr + 2
        cautious_scale_ptr = scratch_ptr + 3
        bound_scale_ptr = scratch_ptr + 4

        cautious_scale = tl.load(cautious_scale_ptr)
        bound_scale = tl.load(bound_scale_ptr)

        if has_master:
            p = tl.load(master_p_ptr + offsets, mask=mask, other=0.0)
        else:
            p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        p_init = p

        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        is_finite_g = (g == g) & (g < 1e38) & (g > -1e38)
        g = tl.where(is_finite_g, tl.maximum(tl.minimum(g, 1e18), -1e18), 0.0)
        m = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        m_new = beta1 * m + (1.0 - beta1) * g
        clm_m = beta1 * m_new + (1.0 - beta1) * g
        diff = tl.maximum(tl.minimum(g - m_new, 1e18), -1e18)
        if centered_defect:
            diff_mean = (tl.load(scratch_ptr + 5) / total_elements) * beta1
            diff = tl.maximum(tl.minimum(diff - diff_mean, 1e18), -1e18)
        v_new = beta2 * v + (1.0 - beta2) * (diff * diff)
        if dtype_mode == 2 or state_dtype_mode == 2:
            v_new = tl.minimum(v_new, 65000.0)

        sigma = tl.sqrt(tl.maximum(v_new, 0.0)) * c1 + c2
        u = _triton_tanh_fast(clm_m / tl.maximum(sigma, 1e-12))
        m_mask = tl.where((u * g) > 0.0, 1.0, 0.0)

        _store_state_buffer(exp_avg_ptr, offsets, m_new, mask, seed_m, state_dtype_mode, use_sr)
        _store_state_buffer(exp_avg_sq_ptr, offsets, v_new, mask, seed_m, state_dtype_mode, False)

        if cautious:
            delta_theta = (u * m_mask) * cautious_scale
        else:
            delta_theta = u
        step_val = lr * delta_theta

        if bound_mode == 1:
            step_val = step_val * bound_scale
        elif bound_mode == 2:
            B_i = bound_ratio * tl.sqrt(p_init * p_init + 1e-4)
            step_val = B_i * _triton_tanh_fast(step_val / B_i)

        if dtype_mode == 2:
            b_safe = tl.maximum(tl.abs(p_init) * 0.01, 1.5e-4)
            step_val = b_safe * _triton_tanh_fast(step_val / b_safe)

        if weight_decay != 0.0:
            p = p * (1.0 - lr * weight_decay)
        p_updated = p - step_val

        if has_master:
            tl.store(master_p_ptr + offsets, p_updated, mask=mask)
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, dtype_mode, False)
        else:
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, dtype_mode, use_sr)

        if cautious:
            block_sum = tl.sum(tl.where(mask, m_mask, 0.0), axis=0)
            tl.atomic_add(mask_sum_ptr, block_sum)

        if bound_mode == 1:
            u_eff = (u * m_mask) if cautious else u
            block_u_sq = tl.sum(tl.where(mask, u_eff * u_eff, 0.0), axis=0)
            tl.atomic_add(u_sq_sum_ptr, block_u_sq)

            block_p_sq = tl.sum(tl.where(mask, p_init * p_init, 0.0), axis=0)
            tl.atomic_add(p_sq_sum_ptr, block_p_sq)

    @triton.jit
    def _lumina_tdclm_single_pass_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr,
        scratch_ptr,
        n_elements, total_elements, beta1, lr, weight_decay, eps, lam,
        bound_mode: tl.constexpr, bound_ratio,
        seed, dtype_mode: tl.constexpr, state_dtype_mode: tl.constexpr, use_sr: tl.constexpr,
        cautious: tl.constexpr, has_master: tl.constexpr, centered_defect: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        seed_m = seed
        seed_p = (seed + 0x55555555) & 0x7FFFFFFF

        mask_sum_ptr = scratch_ptr + 0
        u_sq_sum_ptr = scratch_ptr + 1
        p_sq_sum_ptr = scratch_ptr + 2
        cautious_scale_ptr = scratch_ptr + 3
        bound_scale_ptr = scratch_ptr + 4

        cautious_scale = tl.load(cautious_scale_ptr)
        bound_scale = tl.load(bound_scale_ptr)

        if has_master:
            p = tl.load(master_p_ptr + offsets, mask=mask, other=0.0)
        else:
            p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        p_init = p

        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        is_finite_g = (g == g) & (g < 1e38) & (g > -1e38)
        g = tl.where(is_finite_g, tl.maximum(tl.minimum(g, 1e18), -1e18), 0.0)
        m = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq
        effective_eps = tl.maximum(eps, 1e-4) if dtype_mode == 2 else eps

        m_new = beta1 * m + (1.0 - beta1) * g
        clm_m = beta1_sq * m + one_minus_beta1_sq * g

        diff = tl.maximum(tl.minimum(g - m, 1e18), -1e18)
        if centered_defect:
            diff_mean = tl.load(scratch_ptr + 5) / total_elements
            diff = tl.maximum(tl.minimum(diff - diff_mean, 1e18), -1e18)

        clm_env = (
            beta1_sq * tl.abs(m)
            + one_minus_beta1_sq * tl.abs(g)
            + lam * tl.abs(diff)
            + effective_eps
        )

        u = clm_m / clm_env
        m_mask = tl.where((u * g) > 0.0, 1.0, 0.0)

        _store_state_buffer(exp_avg_ptr, offsets, m_new, mask, seed_m, state_dtype_mode, use_sr)

        if cautious:
            delta_theta = (u * m_mask) * cautious_scale
        else:
            delta_theta = u
        step_val = lr * delta_theta

        if bound_mode == 1:
            step_val = step_val * bound_scale
        elif bound_mode == 2:
            B_i = bound_ratio * tl.sqrt(p_init * p_init + 1e-4)
            step_val = B_i * _triton_tanh_fast(step_val / B_i)

        if dtype_mode == 2:
            b_safe = tl.maximum(tl.abs(p_init) * 0.01, 1.5e-4)
            step_val = b_safe * _triton_tanh_fast(step_val / b_safe)

        if weight_decay != 0.0:
            p = p * (1.0 - lr * weight_decay)
        p_updated = p - step_val

        if has_master:
            tl.store(master_p_ptr + offsets, p_updated, mask=mask)
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, dtype_mode, False)
        else:
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, dtype_mode, use_sr)

        if cautious:
            block_sum = tl.sum(tl.where(mask, m_mask, 0.0), axis=0)
            tl.atomic_add(mask_sum_ptr, block_sum)

        if bound_mode == 1:
            u_eff = (u * m_mask) if cautious else u
            block_u_sq = tl.sum(tl.where(mask, u_eff * u_eff, 0.0), axis=0)
            tl.atomic_add(u_sq_sum_ptr, block_u_sq)

            block_p_sq = tl.sum(tl.where(mask, p_init * p_init, 0.0), axis=0)
            tl.atomic_add(p_sq_sum_ptr, block_p_sq)

    @triton.jit
    def _lumina_tdclm_pass1_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr,
        scratch_ptr,
        n_elements, total_elements, beta1, eps, lam,
        bound_mode: tl.constexpr, dtype_mode: tl.constexpr,
        cautious: tl.constexpr, has_master: tl.constexpr, centered_defect: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        mask_sum_ptr = scratch_ptr + 0
        u_sq_sum_ptr = scratch_ptr + 1
        p_sq_sum_ptr = scratch_ptr + 2

        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        is_finite_g = (g == g) & (g < 1e38) & (g > -1e38)
        g = tl.where(is_finite_g, tl.maximum(tl.minimum(g, 1e18), -1e18), 0.0)
        m = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq
        effective_eps = tl.maximum(eps, 1e-4) if dtype_mode == 2 else eps

        diff = tl.maximum(tl.minimum(g - m, 1e18), -1e18)
        if centered_defect:
            diff_mean = tl.load(scratch_ptr + 5) / total_elements
            diff = tl.maximum(tl.minimum(diff - diff_mean, 1e18), -1e18)

        clm_m = beta1_sq * m + one_minus_beta1_sq * g
        clm_env = beta1_sq * tl.abs(m) + one_minus_beta1_sq * tl.abs(g) + lam * tl.abs(diff) + effective_eps
        u = clm_m / clm_env
        m_mask = tl.where((u * g) > 0.0, 1.0, 0.0)

        if cautious:
            block_sum = tl.sum(tl.where(mask, m_mask, 0.0), axis=0)
            tl.atomic_add(mask_sum_ptr, block_sum)

        if bound_mode == 1:
            u_eff = (u * m_mask) if cautious else u
            block_u_sq = tl.sum(tl.where(mask, u_eff * u_eff, 0.0), axis=0)
            tl.atomic_add(u_sq_sum_ptr, block_u_sq)

            if has_master:
                p = tl.load(master_p_ptr + offsets, mask=mask, other=0.0)
            else:
                p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            block_p_sq = tl.sum(tl.where(mask, p * p, 0.0), axis=0)
            tl.atomic_add(p_sq_sum_ptr, block_p_sq)

    @triton.jit
    def _lumina_tdclm_pass2_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr,
        scratch_ptr,
        n_elements, total_elements, beta1, lr, weight_decay, eps, lam, clamp_min,
        bound_mode: tl.constexpr, bound_ratio,
        seed, dtype_mode: tl.constexpr, state_dtype_mode: tl.constexpr, use_sr: tl.constexpr,
        cautious: tl.constexpr, has_master: tl.constexpr, centered_defect: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        seed_m = seed
        seed_p = (seed + 0x55555555) & 0x7FFFFFFF

        cautious_scale = 1.0
        if cautious:
            mask_sum = tl.load(scratch_ptr + 0)
            m_bar = tl.minimum(tl.maximum(mask_sum / total_elements, clamp_min), 1.0)
            cautious_scale = 1.0 / m_bar

        bound_scale = 1.0
        if bound_mode == 1:
            u_sq_sum = tl.load(scratch_ptr + 1)
            p_sq_sum = tl.load(scratch_ptr + 2)
            norm_p = tl.sqrt(tl.maximum(p_sq_sum, 0.0))
            norm_delta = tl.sqrt(tl.maximum(u_sq_sum, 0.0)) * cautious_scale
            R = bound_ratio * tl.maximum(norm_p, 1.0)
            r = (lr * norm_delta) / R
            bound_scale = tl.where(r < 1e-4, 1.0, _triton_tanh_fast(r) / tl.maximum(r, 1e-8))

        if has_master:
            p = tl.load(master_p_ptr + offsets, mask=mask, other=0.0)
        else:
            p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        p_init = p

        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        is_finite_g = (g == g) & (g < 1e38) & (g > -1e38)
        g = tl.where(is_finite_g, tl.maximum(tl.minimum(g, 1e18), -1e18), 0.0)
        m = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq
        effective_eps = tl.maximum(eps, 1e-4) if dtype_mode == 2 else eps

        diff = tl.maximum(tl.minimum(g - m, 1e18), -1e18)
        if centered_defect:
            diff_mean = tl.load(scratch_ptr + 5) / total_elements
            diff = tl.maximum(tl.minimum(diff - diff_mean, 1e18), -1e18)

        m_new = beta1 * m + (1.0 - beta1) * g
        clm_m = beta1_sq * m + one_minus_beta1_sq * g
        clm_env = beta1_sq * tl.abs(m) + one_minus_beta1_sq * tl.abs(g) + lam * tl.abs(diff) + effective_eps
        u = clm_m / clm_env
        m_mask = tl.where((u * g) > 0.0, 1.0, 0.0)

        _store_state_buffer(exp_avg_ptr, offsets, m_new, mask, seed_m, state_dtype_mode, use_sr)

        delta_theta = (u * m_mask) * cautious_scale if cautious else u
        step_val = lr * delta_theta

        if bound_mode == 1:
            step_val = step_val * bound_scale
        elif bound_mode == 2:
            B_i = bound_ratio * tl.sqrt(p_init * p_init + 1e-4)
            step_val = B_i * _triton_tanh_fast(step_val / B_i)

        if dtype_mode == 2:
            b_safe = tl.maximum(tl.abs(p_init) * 0.01, 1.5e-4)
            step_val = b_safe * _triton_tanh_fast(step_val / b_safe)

        if weight_decay != 0.0:
            p = p * (1.0 - lr * weight_decay)
        p_updated = p - step_val

        if has_master:
            tl.store(master_p_ptr + offsets, p_updated, mask=mask)
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, dtype_mode, False)
        else:
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, dtype_mode, use_sr)

        if tl.program_id(axis=0) == 0:
            if cautious:
                tl.store(scratch_ptr + 3, cautious_scale)
            if bound_mode == 1:
                tl.store(scratch_ptr + 4, bound_scale)

    @triton.jit
    def _lumina_post_step_ema_kernel(
        scratch_ptr, total_elements, lr, clamp_min, ema_decay,
        bound_mode: tl.constexpr, bound_ratio, cautious: tl.constexpr
    ):
        if tl.program_id(axis=0) == 0:
            mask_sum_ptr = scratch_ptr + 0
            u_sq_sum_ptr = scratch_ptr + 1
            p_sq_sum_ptr = scratch_ptr + 2
            cautious_scale_ptr = scratch_ptr + 3
            bound_scale_ptr = scratch_ptr + 4

            if cautious:
                mask_sum = tl.load(mask_sum_ptr)
                m_bar = tl.minimum(tl.maximum(mask_sum / total_elements, clamp_min), 1.0)
                target_cautious_scale = 1.0 / m_bar
                old_cautious = tl.load(cautious_scale_ptr)
                new_cautious = ema_decay * old_cautious + (1.0 - ema_decay) * target_cautious_scale
                tl.store(cautious_scale_ptr, new_cautious)

            if bound_mode == 1:
                u_sq_sum = tl.load(u_sq_sum_ptr)
                p_sq_sum = tl.load(p_sq_sum_ptr)
                c_scale = tl.load(cautious_scale_ptr) if cautious else 1.0

                norm_p = tl.sqrt(tl.maximum(p_sq_sum, 0.0))
                norm_delta = tl.sqrt(tl.maximum(u_sq_sum, 0.0)) * c_scale
                R = bound_ratio * tl.maximum(norm_p, 1.0)
                r = (lr * norm_delta) / R
                target_scale = tl.where(r < 1e-4, 1.0, _triton_tanh_fast(r) / tl.maximum(r, 1e-8))
                old_bound = tl.load(bound_scale_ptr)
                new_bound = ema_decay * old_bound + (1.0 - ema_decay) * target_scale
                tl.store(bound_scale_ptr, new_bound)

            tl.store(mask_sum_ptr, 0.0)
            tl.store(u_sq_sum_ptr, 0.0)
            tl.store(p_sq_sum_ptr, 0.0)
            tl.store(scratch_ptr + 5, 0.0)

    @triton.jit
    def _lumina_dual_pass1_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
        scratch_ptr,
        n_elements, total_elements, beta1, beta2, c1, c2,
        bound_mode: tl.constexpr, dtype_mode: tl.constexpr, state_dtype_mode: tl.constexpr,
        cautious: tl.constexpr, has_master: tl.constexpr,
        centered_defect: tl.constexpr, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        mask_sum_ptr = scratch_ptr + 0
        u_sq_sum_ptr = scratch_ptr + 1
        p_sq_sum_ptr = scratch_ptr + 2

        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        is_finite_g = (g == g) & (g < 1e38) & (g > -1e38)
        g = tl.where(is_finite_g, tl.maximum(tl.minimum(g, 1e18), -1e18), 0.0)
        m = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        m_new = beta1 * m + (1.0 - beta1) * g
        clm_m = beta1 * m_new + (1.0 - beta1) * g
        diff = tl.maximum(tl.minimum(g - m_new, 1e18), -1e18)
        if centered_defect:
            diff_mean = (tl.load(scratch_ptr + 5) / total_elements) * beta1
            diff = tl.maximum(tl.minimum(diff - diff_mean, 1e18), -1e18)
        v_new = beta2 * v + (1.0 - beta2) * (diff * diff)
        if dtype_mode == 2 or state_dtype_mode == 2:
            v_new = tl.minimum(v_new, 65000.0)

        sigma = tl.sqrt(tl.maximum(v_new, 0.0)) * c1 + c2
        u = _triton_tanh_fast(clm_m / tl.maximum(sigma, 1e-12))
        m_mask = tl.where((u * g) > 0.0, 1.0, 0.0)

        if cautious:
            block_sum = tl.sum(tl.where(mask, m_mask, 0.0), axis=0)
            tl.atomic_add(mask_sum_ptr, block_sum)

        if bound_mode == 1:
            u_eff = (u * m_mask) if cautious else u
            block_u_sq = tl.sum(tl.where(mask, u_eff * u_eff, 0.0), axis=0)
            tl.atomic_add(u_sq_sum_ptr, block_u_sq)

            if has_master:
                p = tl.load(master_p_ptr + offsets, mask=mask, other=0.0)
            else:
                p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            block_p_sq = tl.sum(tl.where(mask, p * p, 0.0), axis=0)
            tl.atomic_add(p_sq_sum_ptr, block_p_sq)

    @triton.jit
    def _lumina_dual_pass2_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
        scratch_ptr,
        n_elements, total_elements, beta1, beta2, c1, c2, lr, weight_decay, clamp_min,
        bound_mode: tl.constexpr, bound_ratio,
        seed, dtype_mode: tl.constexpr, state_dtype_mode: tl.constexpr, use_sr: tl.constexpr,
        cautious: tl.constexpr, has_master: tl.constexpr, centered_defect: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        seed_m = seed
        seed_p = (seed + 0x55555555) & 0x7FFFFFFF

        cautious_scale = 1.0
        if cautious:
            mask_sum = tl.load(scratch_ptr + 0)
            m_bar = tl.minimum(tl.maximum(mask_sum / total_elements, clamp_min), 1.0)
            cautious_scale = 1.0 / m_bar

        bound_scale = 1.0
        if bound_mode == 1:
            u_sq_sum = tl.load(scratch_ptr + 1)
            p_sq_sum = tl.load(scratch_ptr + 2)
            norm_p = tl.sqrt(tl.maximum(p_sq_sum, 0.0))
            norm_delta = tl.sqrt(tl.maximum(u_sq_sum, 0.0)) * cautious_scale
            R = bound_ratio * tl.maximum(norm_p, 1.0)
            r = (lr * norm_delta) / R
            bound_scale = tl.where(r < 1e-4, 1.0, _triton_tanh_fast(r) / tl.maximum(r, 1e-8))

        if has_master:
            p = tl.load(master_p_ptr + offsets, mask=mask, other=0.0)
        else:
            p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        p_init = p

        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        is_finite_g = (g == g) & (g < 1e38) & (g > -1e38)
        g = tl.where(is_finite_g, tl.maximum(tl.minimum(g, 1e18), -1e18), 0.0)
        m = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        m_new = beta1 * m + (1.0 - beta1) * g
        clm_m = beta1 * m_new + (1.0 - beta1) * g
        diff = tl.maximum(tl.minimum(g - m_new, 1e18), -1e18)
        if centered_defect:
            diff_mean = (tl.load(scratch_ptr + 5) / total_elements) * beta1
            diff = tl.maximum(tl.minimum(diff - diff_mean, 1e18), -1e18)
        v_new = beta2 * v + (1.0 - beta2) * (diff * diff)
        if dtype_mode == 2 or state_dtype_mode == 2:
            v_new = tl.minimum(v_new, 65000.0)

        sigma = tl.sqrt(tl.maximum(v_new, 0.0)) * c1 + c2
        u = _triton_tanh_fast(clm_m / tl.maximum(sigma, 1e-12))
        m_mask = tl.where((u * g) > 0.0, 1.0, 0.0)

        _store_state_buffer(exp_avg_ptr, offsets, m_new, mask, seed_m, state_dtype_mode, use_sr)
        _store_state_buffer(exp_avg_sq_ptr, offsets, v_new, mask, seed_m, state_dtype_mode, False)

        delta_theta = (u * m_mask) * cautious_scale if cautious else u
        step_val = lr * delta_theta

        if bound_mode == 1:
            step_val = step_val * bound_scale
        elif bound_mode == 2:
            B_i = bound_ratio * tl.sqrt(p_init * p_init + 1e-4)
            step_val = B_i * _triton_tanh_fast(step_val / B_i)

        if dtype_mode == 2:
            b_safe = tl.maximum(tl.abs(p_init) * 0.01, 1.5e-4)
            step_val = b_safe * _triton_tanh_fast(step_val / b_safe)

        if weight_decay != 0.0:
            p = p * (1.0 - lr * weight_decay)
        p_updated = p - step_val

        if has_master:
            tl.store(master_p_ptr + offsets, p_updated, mask=mask)
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, dtype_mode, False)
        else:
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, dtype_mode, use_sr)

        if tl.program_id(axis=0) == 0:
            if cautious:
                tl.store(scratch_ptr + 3, cautious_scale)
            if bound_mode == 1:
                tl.store(scratch_ptr + 4, bound_scale)

    @triton.jit
    def _lumina_dual_single_pass_kernel(
        p_ptr, master_p_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
        scratch_ptr,
        n_elements, total_elements, beta1, beta2, c1, c2, lr, weight_decay,
        bound_mode: tl.constexpr, bound_ratio,
        seed, dtype_mode: tl.constexpr, state_dtype_mode: tl.constexpr, use_sr: tl.constexpr,
        has_master: tl.constexpr, centered_defect: tl.constexpr, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        seed_m = seed
        seed_p = (seed + 0x55555555) & 0x7FFFFFFF

        if has_master:
            p = tl.load(master_p_ptr + offsets, mask=mask, other=0.0)
        else:
            p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        p_init = p

        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        is_finite_g = (g == g) & (g < 1e38) & (g > -1e38)
        g = tl.where(is_finite_g, tl.maximum(tl.minimum(g, 1e18), -1e18), 0.0)
        m = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        m_new = beta1 * m + (1.0 - beta1) * g
        clm_m = beta1 * m_new + (1.0 - beta1) * g
        diff = tl.maximum(tl.minimum(g - m_new, 1e18), -1e18)
        if centered_defect:
            diff_mean = (tl.load(scratch_ptr + 5) / total_elements) * beta1
            diff = tl.maximum(tl.minimum(diff - diff_mean, 1e18), -1e18)
        v_new = beta2 * v + (1.0 - beta2) * (diff * diff)
        if dtype_mode == 2 or state_dtype_mode == 2:
            v_new = tl.minimum(v_new, 65000.0)

        sigma = tl.sqrt(tl.maximum(v_new, 0.0)) * c1 + c2
        u = _triton_tanh_fast(clm_m / tl.maximum(sigma, 1e-12))

        _store_state_buffer(exp_avg_ptr, offsets, m_new, mask, seed_m, state_dtype_mode, use_sr)
        _store_state_buffer(exp_avg_sq_ptr, offsets, v_new, mask, seed_m, state_dtype_mode, False)

        step_val = lr * u
        if bound_mode == 2:
            B_i = bound_ratio * tl.sqrt(p_init * p_init + 1e-4)
            step_val = B_i * _triton_tanh_fast(step_val / B_i)

        if dtype_mode == 2:
            b_safe = tl.maximum(tl.abs(p_init) * 0.01, 1.5e-4)
            step_val = b_safe * _triton_tanh_fast(step_val / b_safe)

        if weight_decay != 0.0:
            p = p * (1.0 - lr * weight_decay)
        p_updated = p - step_val

        if has_master:
            tl.store(master_p_ptr + offsets, p_updated, mask=mask)
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, dtype_mode, False)
        else:
            _store_param(p_ptr, offsets, p_updated, mask, seed_p, dtype_mode, use_sr)


def _compute_step_val(
    update: Tensor,
    p: Tensor,
    lr: float,
    bound: bool,
    bound_type: str,
    bound_ratio: float,
    target_dtype: Optional[torch.dtype] = None,
    radial_scale: Optional[Tensor] = None,
) -> Tensor:
    u_float = torch.nan_to_num(update.float(), nan=0.0, posinf=1.0, neginf=-1.0)
    if lr == 0.0:
        return torch.zeros_like(u_float)

    effective_dtype = target_dtype if target_dtype is not None else p.dtype
    step_val = lr * u_float
    p_ref: Optional[Tensor] = None

    if bound and bound_ratio > 0.0:
        if bound_type == "radial":
            if radial_scale is not None:
                step_val = step_val * radial_scale
            else:
                p_ref = p.float()
                norm_u = torch.linalg.vector_norm(u_float)
                norm_p = torch.linalg.vector_norm(p_ref)
                R = bound_ratio * torch.clamp_min(norm_p, 1.0)
                r = (lr * norm_u) / R
                scale = torch.where(r < 1e-4, torch.ones_like(r), torch.tanh(r) / torch.clamp_min(r, 1e-8))
                step_val = step_val * scale
        elif bound_type == "coordinate":
            p_ref = p.float()
            B = bound_ratio * torch.sqrt(p_ref.square() + 1e-4)
            step_val = B * torch.tanh(step_val / B)

    if effective_dtype == torch.float16:
        if p_ref is None:
            p_ref = p.float()
        b_safe = torch.clamp_min(torch.abs(p_ref) * 0.01, 1.5e-4)
        step_val = b_safe * torch.tanh(step_val / b_safe)

    return step_val


def _sr_quantize(
    target: Tensor,
    val_f32: Tensor,
    use_sr: bool,
    generator: Optional[torch.Generator] = None,
):
    if target.dtype not in (torch.bfloat16, torch.float16) or not use_sr:
        if target.dtype == torch.float16:
            target.copy_(val_f32.clamp(-65504.0, 65504.0).to(dtype=target.dtype))
        elif target.dtype == torch.bfloat16:
            target.copy_(val_f32.clamp(-3.3895313892515355e38, 3.3895313892515355e38).to(dtype=target.dtype))
        else:
            target.copy_(val_f32.to(dtype=target.dtype))
        return

    work_f32 = val_f32.clone() if val_f32.is_contiguous() else val_f32.contiguous()
    int_view = work_f32.view(torch.int32)
    n_elem = int_view.numel()
    chunk_size = 1_048_576
    flat_view = int_view.view(-1)

    cpu_gen = generator if (generator is not None and generator.device.type == "cpu") else None
    use_cpu_rng = (target.device.type == "cpu") or (generator is not None and generator.device.type == "cpu")

    if target.dtype == torch.bfloat16:
        mask_bits = 0xFFFF
        shift_range = 1 << 16
        if n_elem <= chunk_size:
            if use_cpu_rng:
                noise = torch.empty(n_elem, dtype=torch.int32, device="cpu").random_(
                    0, shift_range, generator=cpu_gen
                )
                if target.device.type != "cpu":
                    noise = noise.to(device=target.device, non_blocking=True)
            else:
                try:
                    noise = torch.empty(n_elem, dtype=torch.int32, device=target.device).random_(
                        0, shift_range, generator=generator
                    )
                except RuntimeError:
                    noise = torch.empty(n_elem, dtype=torch.int32, device="cpu").random_(
                        0, shift_range, generator=cpu_gen
                    ).to(device=target.device, non_blocking=True)
            abs_int = flat_view & 0x7FFFFFFF
            sign_bit = flat_view & (-2147483648)
            rounded_abs = torch.clamp_max((abs_int + noise) & (~mask_bits), 0x7F7F0000)
            flat_view.copy_(sign_bit | rounded_abs)
        else:
            if use_cpu_rng:
                buf_device = torch.device("cpu")
                chunk_noise = torch.empty(chunk_size, dtype=torch.int32, device=buf_device)
            else:
                buf_device = target.device
                try:
                    chunk_noise = torch.empty(chunk_size, dtype=torch.int32, device=buf_device)
                except RuntimeError:
                    buf_device = torch.device("cpu")
                    chunk_noise = torch.empty(chunk_size, dtype=torch.int32, device=buf_device)

            for offset in range(0, n_elem, chunk_size):
                curr_len = min(chunk_size, n_elem - offset)
                sub_slice = flat_view[offset : offset + curr_len]
                sub_noise = chunk_noise[:curr_len]
                if buf_device.type == "cpu":
                    sub_noise.random_(0, shift_range, generator=cpu_gen)
                    noise_dev = sub_noise.to(device=target.device, non_blocking=True) if target.device.type != "cpu" else sub_noise
                else:
                    try:
                        sub_noise.random_(0, shift_range, generator=generator)
                        noise_dev = sub_noise
                    except RuntimeError:
                        cpu_sub = torch.empty(curr_len, dtype=torch.int32, device="cpu").random_(
                            0, shift_range, generator=cpu_gen
                        )
                        noise_dev = cpu_sub.to(device=target.device, non_blocking=True)
                abs_int = sub_slice & 0x7FFFFFFF
                sign_bit = sub_slice & (-2147483648)
                rounded_abs = torch.clamp_max((abs_int + noise_dev) & (~mask_bits), 0x7F7F0000)
                sub_slice.copy_(sign_bit | rounded_abs)
    else:
        for offset in range(0, n_elem, chunk_size):
            curr_len = min(chunk_size, n_elem - offset)
            sub_slice = flat_view[offset : offset + curr_len]
            if use_cpu_rng:
                raw_rand = torch.empty(curr_len, dtype=torch.int32, device="cpu").random_(
                    0, 1 << 24, generator=cpu_gen
                )
                if target.device.type != "cpu":
                    raw_rand = raw_rand.to(device=target.device, non_blocking=True)
            else:
                try:
                    raw_rand = torch.empty(curr_len, dtype=torch.int32, device=target.device).random_(
                        0, 1 << 24, generator=generator
                    )
                except RuntimeError:
                    raw_rand = torch.empty(curr_len, dtype=torch.int32, device="cpu").random_(
                        0, 1 << 24, generator=cpu_gen
                    ).to(device=target.device, non_blocking=True)

            abs_int = sub_slice & 0x7FFFFFFF
            sign_bit = sub_slice & (-2147483648)
            exp_f32 = (abs_int >> 23) & 0xFF
            sub_shift = torch.clamp(113 - exp_f32, min=0, max=11)
            total_shift = 13 + sub_shift
            dyn_mask = (1 << total_shift) - 1
            noise = raw_rand & dyn_mask
            rounded_abs = torch.where(exp_f32 < 103, torch.zeros_like(abs_int), (abs_int + noise) & (~dyn_mask))
            rounded_abs = torch.clamp_max(rounded_abs, 0x477FE000)
            sub_slice.copy_(sign_bit | rounded_abs)

    target.copy_(int_view.view(torch.float32).to(dtype=target.dtype))


def _sr_apply_step_val(
    p: Tensor,
    step_val: Tensor,
    lr: float,
    weight_decay: float,
    use_sr: bool,
    generator: Optional[torch.Generator] = None,
):
    step_f32 = torch.nan_to_num(step_val.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if p.dtype not in (torch.bfloat16, torch.float16):
        if weight_decay != 0.0:
            p.mul_(1.0 - lr * weight_decay)
        p.add_(step_f32, alpha=-1.0)
        return

    p_fp32 = p.float()
    if weight_decay != 0.0:
        p_fp32.mul_(1.0 - lr * weight_decay)
    p_fp32.add_(step_f32, alpha=-1.0)
    _sr_quantize(p, p_fp32, use_sr, generator)


def _sync_fsdp_scratch(sc: Tensor, local_numel: int) -> float:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sync_buf = torch.empty((4,), device=sc.device, dtype=torch.float32)
        sync_buf[:3].copy_(sc[:3])
        sync_buf[3] = float(local_numel)
        torch.distributed.all_reduce(sync_buf, op=torch.distributed.ReduceOp.SUM)
        sc[:3].copy_(sync_buf[:3])
        return max(float(sync_buf[3].item()), 1.0)
    return max(float(local_numel), 1.0)


def _sync_fsdp_diff_only(sc: Tensor, local_numel: int) -> float:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sync_buf = torch.empty((2,), device=sc.device, dtype=torch.float32)
        sync_buf[0] = sc[5]
        sync_buf[1] = float(local_numel)
        torch.distributed.all_reduce(sync_buf, op=torch.distributed.ReduceOp.SUM)
        sc[5].copy_(sync_buf[0])
        return max(float(sync_buf[1].item()), 1.0)
    return max(float(local_numel), 1.0)


def _sync_fsdp_diff_batch(scratches: List[Tensor], local_numels: List[int]) -> List[float]:
    n_params = len(scratches)
    if n_params == 0:
        return []
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return [max(float(n), 1.0) for n in local_numels]
    if n_params == 1:
        return [_sync_fsdp_diff_only(scratches[0], local_numels[0])]
    device = scratches[0].device
    sync_buf = torch.empty((n_params, 2), device=device, dtype=torch.float32)
    for idx, (sc, numel) in enumerate(zip(scratches, local_numels)):
        sync_buf[idx, 0] = sc[5]
        sync_buf[idx, 1] = float(numel)
    torch.distributed.all_reduce(sync_buf, op=torch.distributed.ReduceOp.SUM)
    totals: List[float] = []
    for idx, sc in enumerate(scratches):
        sc[5].copy_(sync_buf[idx, 0])
        totals.append(max(float(sync_buf[idx, 1].item()), 1.0))
    return totals


def _sync_fsdp_scratch_batch(scratches: List[Tensor], local_numels: List[int]) -> List[float]:
    n_params = len(scratches)
    if n_params == 0:
        return []
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return [max(float(n), 1.0) for n in local_numels]
    if n_params == 1:
        return [_sync_fsdp_scratch(scratches[0], local_numels[0])]
    device = scratches[0].device
    sync_buf = torch.empty((n_params, 4), device=device, dtype=torch.float32)
    for idx, (sc, numel) in enumerate(zip(scratches, local_numels)):
        sync_buf[idx, :3].copy_(sc[:3])
        sync_buf[idx, 3] = float(numel)
    torch.distributed.all_reduce(sync_buf, op=torch.distributed.ReduceOp.SUM)
    totals: List[float] = []
    for idx, sc in enumerate(scratches):
        sc[:3].copy_(sync_buf[idx, :3])
        totals.append(max(float(sync_buf[idx, 3].item()), 1.0))
    return totals


def _fallback_post_step_ema(
    sc: Tensor,
    n_elements: int,
    lr: float,
    clamp_min: float,
    ema_decay: float,
    bound_mode: int,
    bound_ratio: float,
    cautious: bool,
    fsdp_sync: bool,
    scratch_synced: bool,
    synced_total_elements: float = 0.0,
) -> None:
    total_elements = synced_total_elements if (fsdp_sync and scratch_synced and synced_total_elements > 0.0) else float(n_elements)
    if fsdp_sync and not scratch_synced and (cautious or bound_mode == 1):
        total_elements = _sync_fsdp_scratch(sc, n_elements)
    if cautious:
        mask_sum = sc[0].item()
        m_bar = max(min(mask_sum / max(total_elements, 1.0), 1.0), clamp_min)
        target_cautious = 1.0 / m_bar
        old_cautious = sc[3].item()
        sc[3] = ema_decay * old_cautious + (1.0 - ema_decay) * target_cautious
    if bound_mode == 1:
        u_sq_sum = max(sc[1].item(), 0.0)
        p_sq_sum = max(sc[2].item(), 0.0)
        c_scale = sc[3].item() if cautious else 1.0
        norm_p = math.sqrt(p_sq_sum)
        norm_delta = math.sqrt(u_sq_sum) * c_scale
        R = bound_ratio * max(norm_p, 1.0)
        r = (lr * norm_delta) / R
        target_bound = 1.0 if r < 1e-4 else (math.tanh(r) / max(r, 1e-8))
        old_bound = sc[4].item()
        sc[4] = ema_decay * old_bound + (1.0 - ema_decay) * target_bound
    sc[:3].zero_()
    sc[5].zero_()


def _sync_empty_fsdp_shard(
    sc: Tensor,
    step: int,
    lr: float,
    clamp_min: float,
    ema_decay: float,
    bound_mode: int,
    bound_ratio: float,
    cautious: bool,
    centered_defect: bool,
    fused_single_pass: bool,
    diff_synced: bool = False,
    scratch_synced: bool = False,
    synced_total_elements: float = 0.0,
) -> None:
    if not scratch_synced:
        sc[:3].zero_()
    sc[5].zero_()
    if centered_defect and not diff_synced:
        _sync_fsdp_diff_only(sc, 0)
    if cautious or bound_mode == 1:
        tot_elems = (
            _sync_fsdp_scratch(sc, 0)
            if not scratch_synced
            else (synced_total_elements if synced_total_elements > 0.0 else 1.0)
        )
        eff_decay = ema_decay if (fused_single_pass and step > 1) else 0.0
        _fallback_post_step_ema(
            sc, 0, lr, clamp_min, eff_decay, bound_mode, bound_ratio,
            cautious, True, True, synced_total_elements=tot_elems,
        )
    else:
        sc[:3].zero_()
        sc[5].zero_()


class LuminaV(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 8e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 8e-2,
        tau: float = 0.8,
        lam: float = 0.3,
        cautious: bool = True,
        cautious_clamp_min: float = 0.5,
        buffer: int = 2,
        centered_defect: bool = False,
        stochastic_rounding: bool = True,
        bound: bool = True,
        bound_type: str = "coordinate",
        bound_ratio: float = 0.05,
        master_weights: Union[bool, str] = False,
        execution: str = "auto",
        fused_single_pass: bool = True,
        ema_decay: Union[float, str] = "auto",
        fsdp_sync: bool = False,
        seed: int = 1337,
    ):
        self.seed = seed
        self._warned_non_contiguous = False
        self._warned_triton_failure = False
        self._generators: Dict[Tuple[str, int, int], torch.Generator] = {}

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            tau=tau, lam=lam, cautious=cautious,
            cautious_clamp_min=cautious_clamp_min, buffer=buffer,
            centered_defect=centered_defect,
            stochastic_rounding=stochastic_rounding,
            bound=bound, bound_type=bound_type, bound_ratio=bound_ratio,
            master_weights=master_weights, execution=execution,
            fused_single_pass=fused_single_pass, ema_decay=ema_decay,
            fsdp_sync=fsdp_sync,
            seed=seed,
        )
        is_auto_ema = isinstance(ema_decay, str) and ema_decay.lower().strip() == "auto"
        self._validate_and_normalize_group(defaults)
        if is_auto_ema:
            defaults["ema_decay"] = "auto"
        super().__init__(params, defaults)

    @staticmethod
    def _validate_and_normalize_group(group: dict) -> None:
        lr = group["lr"]
        betas = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        tau = group["tau"]
        lam = group.get("lam", 0.3)
        cautious_clamp_min = group["cautious_clamp_min"]
        buffer = group["buffer"]
        bound_type = group["bound_type"]
        bound_ratio = group["bound_ratio"]
        seed = group["seed"]
        execution = group["execution"]
        master_weights = group["master_weights"]
        fsdp_sync = group.get("fsdp_sync", False)
        ema_decay = group.get("ema_decay", "auto")

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError(f"Invalid betas parameter: {betas}. Must be a tuple or list of two floats")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if tau <= 0.0:
            raise ValueError(f"Invalid tau parameter: {tau}")
        if not 0.0 < cautious_clamp_min <= 1.0:
            raise ValueError(f"Invalid cautious_clamp_min: {cautious_clamp_min}")
        if isinstance(buffer, bool) or buffer not in (1, 2):
            raise ValueError(f"Buffer count must be 1 (Single-Buffer TD-CLM) or 2 (Dual-Buffer CLM), got: {buffer}")
        if bound_type not in ("radial", "coordinate"):
            raise ValueError(f"Invalid bound_type: {bound_type}. Must be 'radial' or 'coordinate'")
        if bound_ratio <= 0.0:
            raise ValueError(f"Invalid bound_ratio: {bound_ratio}. Must be > 0.0")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"Invalid seed parameter: {seed}. Must be a non-negative integer")
        if execution not in ("auto", "triton", "single", "foreach"):
            raise ValueError(f"Invalid execution mode: {execution}. Must be 'auto', 'triton', or 'single'")
        if lam < 0.0:
            raise ValueError(f"Invalid lam (triangle defect ratio) parameter: {lam}. Must be >= 0.0")

        if execution == "foreach":
            logger.info(
                "LuminaV: execution='foreach' has been deprecated in favor of high-efficiency "
                "'triton' and memory-lean 'single'. Automatically setting execution to 'auto'."
            )
            group["execution"] = "auto"

        if isinstance(master_weights, bool):
            group["master_weights"] = "full" if master_weights else "none"
        elif isinstance(master_weights, str):
            m_str = master_weights.lower().strip()
            if m_str in ("false", "none", "off"):
                group["master_weights"] = "none"
            elif m_str in ("semi", "hybrid", "half"):
                group["master_weights"] = "semi"
            elif m_str in ("full", "true", "fp32", "masterweight"):
                group["master_weights"] = "full"
            else:
                raise ValueError(f"Invalid master_weights option: {master_weights}. Must be False, 'semi', or 'full'")
        else:
            raise TypeError(f"master_weights must be bool or str, got: {type(master_weights).__name__}")

        if fsdp_sync and not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            logger.warning(
                "LuminaV: fsdp_sync=True was specified, but torch.distributed is not initialized. "
                "Disabling fsdp_sync to prevent collective communication errors."
            )
            group["fsdp_sync"] = False

        if isinstance(ema_decay, str):
            if ema_decay.lower().strip() == "auto":
                group["ema_decay"] = 0.95 if buffer == 1 else 0.8
            else:
                raise ValueError(f"Invalid ema_decay string: {ema_decay}. Must be 'auto' or a float in (0.0, 1.0)")
        elif isinstance(ema_decay, (int, float)) and not isinstance(ema_decay, bool):
            if not 0.0 < float(ema_decay) < 1.0:
                raise ValueError(f"Invalid ema_decay: {ema_decay}. Must be in (0.0, 1.0)")
            group["ema_decay"] = float(ema_decay)
        else:
            raise TypeError(f"ema_decay must be float or 'auto', got {type(ema_decay).__name__}")

    def add_param_group(self, param_group: dict) -> None:
        for k, v in self.defaults.items():
            param_group.setdefault(k, v)
        self._validate_and_normalize_group(param_group)
        super().add_param_group(param_group)

    def _get_generator(
        self,
        device: torch.device,
        seed: Optional[int] = None,
        slot: int = 0,
        reseed: bool = False,
    ) -> torch.Generator:
        effective_seed = int(self.seed if seed is None else seed) & 0x7FFFFFFF
        dev_type = device.type
        if device.index is not None:
            dev_idx = device.index
        elif dev_type == "cuda" and torch.cuda.is_available():
            dev_idx = torch.cuda.current_device()
        elif dev_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            dev_idx = torch.xpu.current_device()
        elif hasattr(torch, "accelerator") and hasattr(torch.accelerator, "current_device_index") and torch.accelerator.is_available():
            dev_idx = torch.accelerator.current_device_index()
        else:
            dev_idx = 0
        key = (dev_type, dev_idx, slot if reseed else effective_seed)
        if key not in self._generators:
            try:
                gen = torch.Generator(device=device)
            except (RuntimeError, ValueError, TypeError):
                gen = torch.Generator(device="cpu")
            gen.manual_seed(effective_seed)
            self._generators[key] = gen
        elif reseed:
            self._generators[key].manual_seed(effective_seed)
        return self._generators[key]

    def _get_step_generators(
        self,
        device: torch.device,
        seed: int,
        step: int,
        param_index: int,
        need_m: bool,
        need_p: bool,
    ) -> Tuple[Optional[torch.Generator], Optional[torch.Generator]]:
        base_seed = (seed + step * 0x9E3779B9 + param_index * 10007) & 0x7FFFFFFF
        gen_m = self._get_generator(device, seed=base_seed, slot=0, reseed=True) if need_m else None
        seed_p = (base_seed + 0x55555555) & 0x7FFFFFFF
        gen_p = self._get_generator(device, seed=seed_p, slot=1, reseed=True) if need_p else None
        return gen_m, gen_p

    def state_dict(self) -> dict:
        raw_dict = super().state_dict()
        temp_keys = {"p_contig", "exp_avg_contig", "exp_avg_sq_contig", "mp_contig", "_plan", "_contig_step"}
        param_ids = {id(p) for group in self.param_groups for p in group.get("params", [])}
        clean_state = {}
        for k, p_state in raw_dict.get("state", {}).items():
            if isinstance(p_state, dict):
                entry = {}
                for sk, sv in p_state.items():
                    if sk in temp_keys or (sk == "master_param" and id(sv) in param_ids):
                        continue
                    if sk == "scratch" and isinstance(sv, Tensor) and sv.numel() >= 6:
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

    def load_state_dict(self, state_dict: dict) -> None:
        raw_state = state_dict.get("state", {})
        saved_groups = state_dict.get("param_groups", [])
        raw_param_state: Dict[Tensor, dict] = {}
        if len(self.param_groups) == len(saved_groups):
            for group, saved_group in zip(self.param_groups, saved_groups):
                for p, p_id in zip(group.get("params", []), saved_group.get("params", [])):
                    st_entry = raw_state.get(p_id) if p_id in raw_state else raw_state.get(p)
                    if isinstance(st_entry, dict):
                        raw_param_state[p] = {
                            k: v.detach().clone() if isinstance(v, Tensor) else v
                            for k, v in st_entry.items()
                        }

        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for k, v in self.defaults.items():
                group.setdefault(k, v)
            self._validate_and_normalize_group(group)
            master_mode = group.get("master_weights", "none")
            buf_count = group.get("buffer", 2)
            cautious_g = group.get("cautious", True)
            radial_g = bool(
                group.get("bound", True)
                and group.get("bound_type", "coordinate") == "radial"
                and group.get("bound_ratio", 0.05) > 0.0
            )
            for p in group["params"]:
                if p not in self.state:
                    continue
                st = self.state[p]
                raw_st = raw_param_state.get(p, {})
                for temp_key in ("p_contig", "exp_avg_contig", "exp_avg_sq_contig", "mp_contig", "_plan", "_contig_step"):
                    st.pop(temp_key, None)
                if "step" in st and isinstance(st["step"], Tensor):
                    st["step"] = int(st["step"].item())
                if "scratch" in st and isinstance(st["scratch"], Tensor):
                    src_sc = raw_st["scratch"] if ("scratch" in raw_st and isinstance(raw_st["scratch"], Tensor)) else st["scratch"]
                    old_sc = src_sc.to(device=p.device, dtype=torch.float32).contiguous()
                    if old_sc.numel() < 6:
                        new_sc = torch.zeros((6,), device=p.device, dtype=torch.float32)
                        new_sc[3] = 1.0
                        new_sc[4] = 1.0
                        new_sc[:old_sc.numel()].copy_(old_sc)
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
                if master_mode == "none" or p.dtype == torch.float32:
                    st.pop("master_param", None)
                elif "master_param" in st and isinstance(st["master_param"], Tensor):
                    src_mp = (
                        raw_st["master_param"]
                        if ("master_param" in raw_st and isinstance(raw_st["master_param"], Tensor))
                        else st["master_param"]
                    )
                    st["master_param"] = src_mp.to(device=p.device, dtype=torch.float32).contiguous()
                else:
                    st["master_param"] = p.detach().clone().float().contiguous()
                target_state_dtype = torch.float32 if master_mode == "full" else p.dtype
                for buf_key in ("exp_avg", "exp_avg_sq"):
                    if buf_key in st and isinstance(st[buf_key], Tensor):
                        src_buf = raw_st[buf_key] if (buf_key in raw_st and isinstance(raw_st[buf_key], Tensor)) else st[buf_key]
                        st[buf_key] = src_buf.to(device=p.device, dtype=target_state_dtype).contiguous()
                if "exp_avg" not in st or not isinstance(st["exp_avg"], Tensor):
                    st["exp_avg"] = torch.zeros_like(p, dtype=target_state_dtype, memory_format=torch.contiguous_format)
                if buf_count == 1:
                    st.pop("exp_avg_sq", None)
                elif "exp_avg_sq" not in st or not isinstance(st["exp_avg_sq"], Tensor):
                    st["exp_avg_sq"] = torch.zeros_like(p, dtype=target_state_dtype, memory_format=torch.contiguous_format)
                st["buf_count"] = buf_count

    def _prepare_triton_buffers(
        self,
        p: Tensor,
        grad: Tensor,
        exp_avg: Tensor,
        exp_avg_sq: Optional[Tensor],
        mp: Optional[Tensor],
        param_has_master: bool,
        reuse_contig: bool = False,
    ):
        state = self.state[p]
        n_elements = p.numel()
        curr_dt_mode = 1 if p.dtype == torch.bfloat16 else (2 if p.dtype == torch.float16 else 0)
        curr_s_dt_mode = 1 if exp_avg.dtype == torch.bfloat16 else (2 if exp_avg.dtype == torch.float16 else 0)
        cached_plan = state.get("_plan")
        if (
            cached_plan is None
            or len(cached_plan) != 6
            or cached_plan[0] != n_elements
            or cached_plan[4] != curr_dt_mode
            or cached_plan[5] != curr_s_dt_mode
        ):
            if n_elements >= 2_097_152:
                b_size = 2048
                n_warps = 8
            else:
                b_size = 1024
                n_warps = 4
            c_grid = (triton.cdiv(n_elements, b_size),)
            dt_mode = curr_dt_mode
            s_dt_mode = curr_s_dt_mode
            state["_plan"] = (n_elements, b_size, n_warps, c_grid, dt_mode, s_dt_mode)
        else:
            n_elements, b_size, n_warps, c_grid, dt_mode, s_dt_mode = cached_plan

        is_orig_contig = p.is_contiguous()
        is_grad_contig = grad.is_contiguous()
        is_m_contig = exp_avg.is_contiguous()
        is_v_contig = (exp_avg_sq is None) or exp_avg_sq.is_contiguous()
        is_mp_contig = (not param_has_master) or (mp is not None and mp.is_contiguous())
        is_all_contig = is_orig_contig and is_grad_contig and is_m_contig and is_v_contig and is_mp_contig

        if is_all_contig:
            p_contig = p
            grad_contig = grad
            exp_avg_contig = exp_avg
            exp_avg_sq_contig = exp_avg_sq
            mp_contig = mp if (param_has_master and mp is not None) else p
        else:
            if not self._warned_non_contiguous:
                logger.warning(
                    "LuminaV detected non-contiguous tensor buffer(s). An explicit contiguous copy "
                    "was created for Triton kernel execution."
                )
                self._warned_non_contiguous = True

            curr_step_tag = state.get("step")
            can_reuse = bool(reuse_contig and curr_step_tag is not None and state.get("_contig_step") == curr_step_tag)

            if is_orig_contig:
                p_contig = p
            else:
                need_p_alloc = (
                    "p_contig" not in state
                    or state["p_contig"].shape != p.shape
                    or state["p_contig"].dtype != p.dtype
                )
                if need_p_alloc:
                    state["p_contig"] = torch.empty_like(p, memory_format=torch.contiguous_format)
                p_contig = state["p_contig"]
                if need_p_alloc or not can_reuse:
                    p_contig.copy_(p)

            grad_contig = grad if is_grad_contig else grad.contiguous()

            if is_m_contig:
                exp_avg_contig = exp_avg
            else:
                need_m_alloc = (
                    "exp_avg_contig" not in state
                    or state["exp_avg_contig"].shape != exp_avg.shape
                    or state["exp_avg_contig"].dtype != exp_avg.dtype
                )
                if need_m_alloc:
                    state["exp_avg_contig"] = torch.empty_like(exp_avg, memory_format=torch.contiguous_format)
                exp_avg_contig = state["exp_avg_contig"]
                if need_m_alloc or not can_reuse:
                    exp_avg_contig.copy_(exp_avg)

            if exp_avg_sq is None:
                exp_avg_sq_contig = None
            elif is_v_contig:
                exp_avg_sq_contig = exp_avg_sq
            else:
                need_v_alloc = (
                    "exp_avg_sq_contig" not in state
                    or state["exp_avg_sq_contig"].shape != exp_avg_sq.shape
                    or state["exp_avg_sq_contig"].dtype != exp_avg_sq.dtype
                )
                if need_v_alloc:
                    state["exp_avg_sq_contig"] = torch.empty_like(exp_avg_sq, memory_format=torch.contiguous_format)
                exp_avg_sq_contig = state["exp_avg_sq_contig"]
                if need_v_alloc or not can_reuse:
                    exp_avg_sq_contig.copy_(exp_avg_sq)

            if param_has_master and mp is not None:
                if is_mp_contig:
                    mp_contig = mp
                else:
                    need_mp_alloc = (
                        "mp_contig" not in state
                        or state["mp_contig"].shape != mp.shape
                        or state["mp_contig"].dtype != mp.dtype
                    )
                    if need_mp_alloc:
                        state["mp_contig"] = torch.empty_like(mp, memory_format=torch.contiguous_format)
                    mp_contig = state["mp_contig"]
                    if need_mp_alloc or not can_reuse:
                        mp_contig.copy_(mp)
            else:
                mp_contig = p_contig

            state["_contig_step"] = curr_step_tag

        return (
            n_elements, b_size, n_warps, c_grid, dt_mode, s_dt_mode, is_all_contig,
            p_contig, grad_contig, exp_avg_contig, exp_avg_sq_contig, mp_contig
        )

    @staticmethod
    def _copy_back_non_contig(
        p: Tensor,
        p_contig: Tensor,
        exp_avg: Tensor,
        exp_avg_contig: Tensor,
        exp_avg_sq: Optional[Tensor],
        exp_avg_sq_contig: Optional[Tensor],
        mp: Optional[Tensor],
        mp_contig: Tensor,
        param_has_master: bool,
    ) -> None:
        if not p.is_contiguous():
            p.copy_(p_contig)
        if not exp_avg.is_contiguous():
            exp_avg.copy_(exp_avg_contig)
        if exp_avg_sq is not None and exp_avg_sq_contig is not None and not exp_avg_sq.is_contiguous():
            exp_avg_sq.copy_(exp_avg_sq_contig)
        if param_has_master and mp is not None and not mp.is_contiguous():
            mp.copy_(mp_contig)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self._validate_and_normalize_group(group)
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            master_params = []
            steps = []
            scratches = []
            buf_count = group["buffer"]
            master_mode = group["master_weights"]
            fsdp_sync = group.get("fsdp_sync", False)
            cautious = group["cautious"]
            bound = group.get("bound", True)
            bound_type = group.get("bound_type", "coordinate")
            bound_ratio = group.get("bound_ratio", 0.05)
            centered_defect = group.get("centered_defect", False)

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("LuminaV does not support sparse gradients.")
                if p.numel() == 0 and not fsdp_sync:
                    continue

                state = self.state[p]
                target_state_dtype = torch.float32 if master_mode == "full" else p.dtype
                if len(state) == 0:
                    state["step"] = 0
                    if master_mode != "none" and p.dtype != torch.float32:
                        state["master_param"] = p.detach().clone().float().contiguous()

                    state["exp_avg"] = torch.zeros_like(p, dtype=target_state_dtype, memory_format=torch.contiguous_format)
                    if buf_count == 2:
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=target_state_dtype, memory_format=torch.contiguous_format)

                    scratch = torch.zeros((6,), device=p.device, dtype=torch.float32)
                    scratch[3] = 1.0
                    scratch[4] = 1.0
                    state["scratch"] = scratch
                    state["buf_count"] = buf_count
                else:
                    if master_mode != "none":
                        if p.dtype == torch.float32:
                            state.pop("master_param", None)
                        elif "master_param" not in state:
                            state["master_param"] = p.detach().clone().float().contiguous()
                        elif state["master_param"].device != p.device or state["master_param"].dtype != torch.float32:
                            state["master_param"] = state["master_param"].to(device=p.device, dtype=torch.float32, non_blocking=True)
                    else:
                        state.pop("master_param", None)

                    if "scratch" not in state or state["scratch"].numel() < 6:
                        scratch = torch.zeros((6,), device=p.device, dtype=torch.float32)
                        scratch[3] = 1.0
                        scratch[4] = 1.0
                        if "scratch" in state and isinstance(state["scratch"], Tensor):
                            old_sc = state["scratch"].to(device=p.device, dtype=torch.float32)
                            scratch[:old_sc.numel()].copy_(old_sc)
                        state["scratch"] = scratch
                        state["buf_count"] = buf_count
                    elif state["scratch"].device != p.device or state["scratch"].dtype != torch.float32:
                        state["scratch"] = state["scratch"].to(device=p.device, dtype=torch.float32, non_blocking=True)
                        state["buf_count"] = buf_count
                    elif state.get("buf_count") != buf_count:
                        state["scratch"].zero_()
                        state["scratch"][3] = 1.0
                        state["scratch"][4] = 1.0
                        state["buf_count"] = buf_count

                    if not cautious:
                        state["scratch"][3] = 1.0
                    if not (bound and bound_type == "radial" and bound_ratio > 0.0):
                        state["scratch"][4] = 1.0

                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p, dtype=target_state_dtype, memory_format=torch.contiguous_format)
                    elif state["exp_avg"].device != p.device or state["exp_avg"].dtype != target_state_dtype:
                        state["exp_avg"] = state["exp_avg"].to(device=p.device, dtype=target_state_dtype, non_blocking=True)

                    if buf_count == 2:
                        if "exp_avg_sq" not in state:
                            state["exp_avg_sq"] = torch.zeros_like(p, dtype=target_state_dtype, memory_format=torch.contiguous_format)
                        elif state["exp_avg_sq"].device != p.device or state["exp_avg_sq"].dtype != target_state_dtype:
                            state["exp_avg_sq"] = state["exp_avg_sq"].to(device=p.device, dtype=target_state_dtype, non_blocking=True)
                    else:
                        state.pop("exp_avg_sq", None)

                params_with_grad.append(p)
                grads.append(grad)
                exp_avgs.append(state["exp_avg"])
                scratches.append(state["scratch"])
                if buf_count == 2:
                    exp_avg_sqs.append(state["exp_avg_sq"])

                if master_mode != "none":
                    master_params.append(state["master_param"] if "master_param" in state else p)

                curr_step = state.get("step", 0)
                step_val = int(curr_step.item() if isinstance(curr_step, torch.Tensor) else curr_step) + 1
                state["step"] = step_val
                steps.append(step_val)

            if not params_with_grad:
                continue

            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            tau = group["tau"]
            lam = group.get("lam", 0.3)
            clamp_min = group["cautious_clamp_min"]
            use_sr = group["stochastic_rounding"]
            exec_mode = group["execution"]
            fused_single_pass = group.get("fused_single_pass", True)
            ema_decay = group.get("ema_decay", 0.8)
            seed = group.get("seed", self.seed)

            all_accelerator = all(p.device.type in ("cuda", "xpu") for p in params_with_grad)

            deterministic_mode = False
            if hasattr(torch, "are_deterministic_algorithms_enabled"):
                deterministic_mode = torch.are_deterministic_algorithms_enabled()

            if exec_mode in ("auto", "foreach"):
                if HAS_TRITON and all_accelerator and not deterministic_mode:
                    exec_mode = "triton"
                else:
                    exec_mode = "single"

            if exec_mode == "triton" and not (HAS_TRITON and all_accelerator):
                exec_mode = "single"

            has_master = (master_mode != "none") and any(p.dtype != torch.float32 for p in params_with_grad)
            m_list = master_params if has_master else None

            if exec_mode == "triton" and HAS_TRITON and all_accelerator:
                completed_indices: Set[int] = set()
                try:
                    self._dispatch_triton(
                        params_with_grad, grads, exp_avgs, exp_avg_sqs, m_list, steps, scratches,
                        lr, beta1, beta2, eps, weight_decay, tau, lam,
                        cautious, clamp_min, buf_count, use_sr, bound, bound_type, bound_ratio,
                        fused_single_pass, ema_decay, fsdp_sync, seed, centered_defect,
                        completed_indices=completed_indices
                    )
                except Exception as e:
                    if not self._warned_triton_failure:
                        err_line = str(e).strip().splitlines()[0] if str(e).strip() else "Execution failed"
                        logger.warning(f"LuminaV: Triton dispatch failed ({err_line}), falling back to single.")
                        self._warned_triton_failure = True
                    uncompleted = [i for i in range(len(params_with_grad)) if i not in completed_indices]
                    if uncompleted:
                        for idx_u in uncompleted:
                            scratches[idx_u][:3].zero_()
                            scratches[idx_u][5].zero_()
                            st_u = int(steps[idx_u].item() if isinstance(steps[idx_u], torch.Tensor) else steps[idx_u])
                            if st_u <= 1:
                                scratches[idx_u][3] = 1.0
                                scratches[idx_u][4] = 1.0
                        sub_p = [params_with_grad[i] for i in uncompleted]
                        sub_g = [grads[i] for i in uncompleted]
                        sub_m = [exp_avgs[i] for i in uncompleted]
                        sub_v = [exp_avg_sqs[i] for i in uncompleted] if buf_count == 2 else None
                        sub_mp = [m_list[i] for i in uncompleted] if m_list is not None else None
                        sub_st = [steps[i] for i in uncompleted]
                        sub_sc = [scratches[i] for i in uncompleted]
                        if buf_count == 2:
                            self._single_step_dual(
                                sub_p, sub_g, sub_m, sub_v, sub_mp, sub_st, sub_sc,
                                lr, beta1, beta2, eps, weight_decay, tau, cautious, clamp_min,
                                use_sr, bound, bound_type, bound_ratio,
                                fused_single_pass, ema_decay, fsdp_sync, seed, centered_defect,
                                orig_indices=uncompleted
                            )
                        else:
                            self._single_step_tdclm(
                                sub_p, sub_g, sub_m, sub_mp, sub_st, sub_sc,
                                lr, beta1, eps, weight_decay, lam, cautious, clamp_min,
                                use_sr, bound, bound_type, bound_ratio,
                                fused_single_pass, ema_decay, fsdp_sync, seed, centered_defect,
                                orig_indices=uncompleted
                            )
            else:
                if buf_count == 2:
                    self._single_step_dual(
                        params_with_grad, grads, exp_avgs, exp_avg_sqs, m_list, steps, scratches,
                        lr, beta1, beta2, eps, weight_decay, tau, cautious, clamp_min,
                        use_sr, bound, bound_type, bound_ratio,
                        fused_single_pass, ema_decay, fsdp_sync, seed, centered_defect
                    )
                else:
                    self._single_step_tdclm(
                        params_with_grad, grads, exp_avgs, m_list, steps, scratches,
                        lr, beta1, eps, weight_decay, lam, cautious, clamp_min,
                        use_sr, bound, bound_type, bound_ratio,
                        fused_single_pass, ema_decay, fsdp_sync, seed, centered_defect
                    )

        return loss

    def _dispatch_triton(
        self, params, grads, exp_avgs, exp_avg_sqs, master_params, steps, scratches,
        lr, beta1, beta2, eps, weight_decay, tau, lam,
        cautious, clamp_min, buf_count, use_sr, bound, bound_type, bound_ratio,
        fused_single_pass, ema_decay, fsdp_sync: bool = False, seed: int = 1337,
        centered_defect: bool = False,
        completed_indices: Optional[Set[int]] = None
    ):
        primary_dev = params[0].device
        is_single_device = all(p.device == primary_dev for p in params)

        if is_single_device:
            with _get_device_context(primary_dev):
                if buf_count == 2:
                    self._triton_step_dual(
                        params, grads, exp_avgs, exp_avg_sqs, master_params, steps, scratches,
                        lr, beta1, beta2, eps, weight_decay, tau, cautious, clamp_min,
                        use_sr, bound, bound_type, bound_ratio, fused_single_pass, ema_decay,
                        fsdp_sync, seed, centered_defect,
                        orig_indices=None, completed_indices=completed_indices
                    )
                else:
                    self._triton_step_tdclm(
                        params, grads, exp_avgs, master_params, steps, scratches,
                        lr, beta1, eps, weight_decay, lam, cautious, clamp_min,
                        use_sr, bound, bound_type, bound_ratio,
                        fused_single_pass=fused_single_pass, ema_decay=ema_decay,
                        fsdp_sync=fsdp_sync, seed=seed, centered_defect=centered_defect,
                        orig_indices=None, completed_indices=completed_indices
                    )
            return

        device_groups: Dict[torch.device, List[int]] = {}
        for idx, p in enumerate(params):
            dev = p.device
            if dev not in device_groups:
                device_groups[dev] = []
            device_groups[dev].append(idx)

        for dev, indices in device_groups.items():
            sub_p = [params[i] for i in indices]
            sub_g = [grads[i] for i in indices]
            sub_m = [exp_avgs[i] for i in indices]
            sub_st = [steps[i] for i in indices]
            sub_sc = [scratches[i] for i in indices]
            sub_mp = [master_params[i] for i in indices] if master_params is not None else None

            with _get_device_context(dev):
                if buf_count == 2:
                    sub_v = [exp_avg_sqs[i] for i in indices]
                    self._triton_step_dual(
                        sub_p, sub_g, sub_m, sub_v, sub_mp, sub_st, sub_sc,
                        lr, beta1, beta2, eps, weight_decay, tau, cautious, clamp_min,
                        use_sr, bound, bound_type, bound_ratio, fused_single_pass, ema_decay,
                        fsdp_sync, seed, centered_defect,
                        orig_indices=indices, completed_indices=completed_indices
                    )
                else:
                    self._triton_step_tdclm(
                        sub_p, sub_g, sub_m, sub_mp, sub_st, sub_sc,
                        lr, beta1, eps, weight_decay, lam, cautious, clamp_min,
                        use_sr, bound, bound_type, bound_ratio,
                        fused_single_pass=fused_single_pass, ema_decay=ema_decay,
                        fsdp_sync=fsdp_sync, seed=seed, centered_defect=centered_defect,
                        orig_indices=indices, completed_indices=completed_indices
                    )

    def _precompute_fsdp_diff_triton(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        scratches: List[Tensor],
    ) -> List[float]:
        for p, grad, exp_avg, sc in zip(params, grads, exp_avgs, scratches):
            sc[5].zero_()
            n_el = p.numel()
            if n_el == 0:
                continue
            try:
                g_c = grad if grad.is_contiguous() else grad.contiguous()
                m_c = exp_avg if exp_avg.is_contiguous() else exp_avg.contiguous()
                b_size = 2048 if n_el >= 2_097_152 else 1024
                n_warps = 8 if n_el >= 2_097_152 else 4
                c_grid = (triton.cdiv(n_el, b_size),)
                _lumina_diff_sum_kernel[c_grid](
                    g_c, m_c, sc, n_el, BLOCK_SIZE=b_size, num_warps=n_warps
                )
            except Exception:
                chunk_sz = 2_097_152
                if n_el <= chunk_sz:
                    g_clean = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                    sc[5].copy_((g_clean - exp_avg.float()).clamp(-1e18, 1e18).sum().to(torch.float32))
                else:
                    flat_g = (grad if grad.is_contiguous() else grad.contiguous()).view(-1)
                    flat_m = (exp_avg if exp_avg.is_contiguous() else exp_avg.contiguous()).view(-1)
                    diff_acc = 0.0
                    for offset in range(0, n_el, chunk_sz):
                        cur = min(chunk_sz, n_el - offset)
                        g_c_sub = torch.nan_to_num(flat_g[offset : offset + cur], nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                        m_c_sub = flat_m[offset : offset + cur].float()
                        diff_acc += (g_c_sub - m_c_sub).clamp(-1e18, 1e18).sum().item()
                    sc[5] = diff_acc
        return _sync_fsdp_diff_batch(scratches, [p.numel() for p in params])

    @staticmethod
    def _compute_pass1_single_dual_one(
        p: Tensor,
        grad: Tensor,
        exp_avg: Tensor,
        exp_avg_sq: Tensor,
        ref_p: Tensor,
        sc: Tensor,
        beta1: float,
        beta2: float,
        c1: float,
        c2: float,
        bound_mode: int,
        cautious: bool,
        centered_defect: bool,
        diff_total_elements: float,
        chunk_size: int = 2_097_152,
    ) -> None:
        sc[:3].zero_()
        n_el = p.numel()
        if n_el == 0:
            return
        use_cd = bool(centered_defect and diff_total_elements > 1.0)
        diff_mean_scaled = (sc[5].item() / diff_total_elements) * beta1 if use_cd else 0.0
        clamp_v = (p.dtype == torch.float16) or (exp_avg.dtype == torch.float16) or (exp_avg_sq.dtype == torch.float16)
        if n_el <= chunk_size:
            g_clean = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
            m_f32 = exp_avg.float()
            v_f32 = exp_avg_sq.float()
            m_new = m_f32.mul(beta1).add(g_clean, alpha=1.0 - beta1)
            clm_m = m_new.mul(beta1).add(g_clean, alpha=1.0 - beta1)
            diff = (g_clean - m_new).clamp(-1e18, 1e18)
            if use_cd:
                diff = (diff - diff_mean_scaled).clamp(-1e18, 1e18)
            v_new = v_f32.mul(beta2).addcmul(diff, diff, value=1.0 - beta2)
            if clamp_v:
                v_new = v_new.clamp(0.0, 65000.0)
            sigma = v_new.clamp_min(0.0).sqrt().mul(c1).add(c2)
            u = torch.tanh(clm_m / sigma.clamp_min(1e-12))
            mask = (u * g_clean > 0.0).float() if cautious else None
            if cautious and mask is not None:
                sc[0].copy_(mask.sum().to(torch.float32))
            if bound_mode == 1:
                u_eff = (u * mask) if (cautious and mask is not None) else u
                sc[1].copy_(torch.sum(u_eff.float() ** 2))
                sc[2].copy_(torch.sum(ref_p.float() ** 2))
        else:
            flat_g = (grad if grad.is_contiguous() else grad.contiguous()).view(-1)
            flat_m = (exp_avg if exp_avg.is_contiguous() else exp_avg.contiguous()).view(-1)
            flat_v = (exp_avg_sq if exp_avg_sq.is_contiguous() else exp_avg_sq.contiguous()).view(-1)
            flat_ref = (ref_p if ref_p.is_contiguous() else ref_p.contiguous()).view(-1)
            mask_sum_acc = 0.0
            u_sq_sum_acc = 0.0
            p_sq_sum_acc = 0.0
            for offset in range(0, n_el, chunk_size):
                cur = min(chunk_size, n_el - offset)
                g_c = torch.nan_to_num(flat_g[offset : offset + cur], nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                m_f32 = flat_m[offset : offset + cur].float()
                v_f32 = flat_v[offset : offset + cur].float()
                m_new = m_f32.mul(beta1).add(g_c, alpha=1.0 - beta1)
                clm_m = m_new.mul(beta1).add(g_c, alpha=1.0 - beta1)
                diff = (g_c - m_new).clamp(-1e18, 1e18)
                if use_cd:
                    diff = (diff - diff_mean_scaled).clamp(-1e18, 1e18)
                v_new = v_f32.mul(beta2).addcmul(diff, diff, value=1.0 - beta2)
                if clamp_v:
                    v_new = v_new.clamp(0.0, 65000.0)
                sigma = v_new.clamp_min(0.0).sqrt().mul(c1).add(c2)
                u = torch.tanh(clm_m / sigma.clamp_min(1e-12))
                if cautious:
                    m_mask = (u * g_c > 0.0).float()
                    mask_sum_acc += m_mask.sum().item()
                    u_eff = u * m_mask
                else:
                    u_eff = u
                if bound_mode == 1:
                    u_sq_sum_acc += torch.sum(u_eff ** 2).item()
                    p_sq_sum_acc += torch.sum(flat_ref[offset : offset + cur].float() ** 2).item()
            if cautious:
                sc[0] = mask_sum_acc
            if bound_mode == 1:
                sc[1] = u_sq_sum_acc
                sc[2] = p_sq_sum_acc

    @staticmethod
    def _compute_pass1_single_tdclm_one(
        p: Tensor,
        grad: Tensor,
        exp_avg: Tensor,
        ref_p: Tensor,
        sc: Tensor,
        beta1: float,
        eps: float,
        lam: float,
        bound_mode: int,
        cautious: bool,
        centered_defect: bool,
        diff_total_elements: float,
        chunk_size: int = 2_097_152,
    ) -> None:
        sc[:3].zero_()
        n_el = p.numel()
        if n_el == 0:
            return
        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq
        effective_eps = max(eps, 1e-4) if p.dtype == torch.float16 else eps
        use_cd = bool(centered_defect and diff_total_elements > 1.0)
        diff_mean = (sc[5].item() / diff_total_elements) if use_cd else 0.0
        if n_el <= chunk_size:
            g_clean = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
            m_f32 = exp_avg.float()
            clm_m = m_f32.mul(beta1_sq).add(g_clean, alpha=one_minus_beta1_sq)
            diff_f32 = (g_clean - m_f32).clamp(-1e18, 1e18)
            if use_cd:
                diff_f32 = (diff_f32 - diff_mean).clamp(-1e18, 1e18)
            clm_env = (
                torch.abs(m_f32)
                .mul(beta1_sq)
                .add(torch.abs(g_clean), alpha=one_minus_beta1_sq)
                .add_(torch.abs(diff_f32).mul(lam))
                .add_(effective_eps)
            )
            u = clm_m / clm_env
            mask = (u * g_clean > 0.0).float() if cautious else None
            if cautious and mask is not None:
                sc[0].copy_(mask.sum().to(torch.float32))
            if bound_mode == 1:
                u_eff = (u * mask) if (cautious and mask is not None) else u
                sc[1].copy_(torch.sum(u_eff.float() ** 2))
                sc[2].copy_(torch.sum(ref_p.float() ** 2))
        else:
            flat_g = (grad if grad.is_contiguous() else grad.contiguous()).view(-1)
            flat_m = (exp_avg if exp_avg.is_contiguous() else exp_avg.contiguous()).view(-1)
            flat_ref = (ref_p if ref_p.is_contiguous() else ref_p.contiguous()).view(-1)
            mask_sum_acc = 0.0
            u_sq_sum_acc = 0.0
            p_sq_sum_acc = 0.0
            for offset in range(0, n_el, chunk_size):
                cur = min(chunk_size, n_el - offset)
                g_c = torch.nan_to_num(flat_g[offset : offset + cur], nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                m_f32 = flat_m[offset : offset + cur].float()
                clm_m = m_f32.mul(beta1_sq).add(g_c, alpha=one_minus_beta1_sq)
                diff_f32 = (g_c - m_f32).clamp(-1e18, 1e18)
                if use_cd:
                    diff_f32 = (diff_f32 - diff_mean).clamp(-1e18, 1e18)
                clm_env = (
                    torch.abs(m_f32)
                    .mul(beta1_sq)
                    .add(torch.abs(g_c), alpha=one_minus_beta1_sq)
                    .add_(torch.abs(diff_f32).mul(lam))
                    .add_(effective_eps)
                )
                u = clm_m / clm_env
                if cautious:
                    m_mask = (u * g_c > 0.0).float()
                    mask_sum_acc += m_mask.sum().item()
                    u_eff = u * m_mask
                else:
                    u_eff = u
                if bound_mode == 1:
                    u_sq_sum_acc += torch.sum(u_eff ** 2).item()
                    p_sq_sum_acc += torch.sum(flat_ref[offset : offset + cur].float() ** 2).item()
            if cautious:
                sc[0] = mask_sum_acc
            if bound_mode == 1:
                sc[1] = u_sq_sum_acc
                sc[2] = p_sq_sum_acc

    def _precompute_fsdp_pass1_triton_dual(
        self,
        pass1_indices: List[int],
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        master_params: Optional[List[Tensor]],
        steps: List[int],
        scratches: List[Tensor],
        beta1: float,
        beta2: float,
        eps: float,
        tau: float,
        base_step: int,
        c1_base: float,
        c2_default: float,
        c2_fp16: float,
        bound_mode: int,
        cautious: bool,
        centered_defect: bool,
        batched_diff_totals: Optional[List[float]],
    ) -> Dict[int, float]:
        has_master = master_params is not None
        for idx in pass1_indices:
            p = params[idx]
            sc = scratches[idx]
            sc[:3].zero_()
            n_el = p.numel()
            if n_el == 0:
                continue
            grad = grads[idx]
            exp_avg = exp_avgs[idx]
            exp_avg_sq = exp_avg_sqs[idx]
            step_raw = steps[idx]
            step = int(step_raw.item() if isinstance(step_raw, torch.Tensor) else step_raw)
            safe_step = max(step, 1)
            if step == base_step:
                c1 = c1_base
                c2 = c2_fp16 if p.dtype == torch.float16 else c2_default
            else:
                effective_eps = max(eps, 1e-4) if p.dtype == torch.float16 else eps
                bc1_nes = 1.0 - (beta1 ** (safe_step + 1))
                bc2 = max(1.0 - (beta2 ** safe_step), 1e-15)
                c1 = (bc1_nes * tau) / math.sqrt(bc2)
                c2 = effective_eps * bc1_nes * tau
            param_has_master = has_master and (master_params[idx] is not p)
            mp_ref = master_params[idx] if param_has_master else None
            diff_tot = batched_diff_totals[idx] if batched_diff_totals is not None else float(n_el)
            use_cd = bool(centered_defect and diff_tot > 1.0)
            try:
                (
                    n_elements, b_size, n_warps, c_grid, dt_mode, s_dt_mode, _,
                    p_contig, grad_contig, exp_avg_contig, exp_avg_sq_contig, mp_contig
                ) = self._prepare_triton_buffers(p, grad, exp_avg, exp_avg_sq, mp_ref, param_has_master)
                _lumina_dual_pass1_kernel[c_grid](
                    p_contig, mp_contig, grad_contig, exp_avg_contig, exp_avg_sq_contig,
                    sc,
                    n_elements, diff_tot, beta1, beta2, c1, c2,
                    bound_mode, dt_mode, s_dt_mode,
                    cautious, param_has_master, use_cd,
                    BLOCK_SIZE=b_size, num_warps=n_warps
                )
            except Exception:
                ref_p = mp_ref if (param_has_master and mp_ref is not None) else p
                self._compute_pass1_single_dual_one(
                    p, grad, exp_avg, exp_avg_sq, ref_p, sc,
                    beta1, beta2, c1, c2, bound_mode, cautious, centered_defect, diff_tot
                )
        synced_totals = _sync_fsdp_scratch_batch(
            [scratches[idx] for idx in pass1_indices],
            [params[idx].numel() for idx in pass1_indices],
        )
        return {idx: tot for idx, tot in zip(pass1_indices, synced_totals)}

    def _precompute_fsdp_pass1_triton_tdclm(
        self,
        pass1_indices: List[int],
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        master_params: Optional[List[Tensor]],
        scratches: List[Tensor],
        beta1: float,
        eps: float,
        lam: float,
        bound_mode: int,
        cautious: bool,
        centered_defect: bool,
        batched_diff_totals: Optional[List[float]],
    ) -> Dict[int, float]:
        has_master = master_params is not None
        for idx in pass1_indices:
            p = params[idx]
            sc = scratches[idx]
            sc[:3].zero_()
            n_el = p.numel()
            if n_el == 0:
                continue
            grad = grads[idx]
            exp_avg = exp_avgs[idx]
            param_has_master = has_master and (master_params[idx] is not p)
            mp_ref = master_params[idx] if param_has_master else None
            diff_tot = batched_diff_totals[idx] if batched_diff_totals is not None else float(n_el)
            use_cd = bool(centered_defect and diff_tot > 1.0)
            try:
                (
                    n_elements, b_size, n_warps, c_grid, dt_mode, _, _,
                    p_contig, grad_contig, exp_avg_contig, _, mp_contig
                ) = self._prepare_triton_buffers(p, grad, exp_avg, None, mp_ref, param_has_master)
                _lumina_tdclm_pass1_kernel[c_grid](
                    p_contig, mp_contig, grad_contig, exp_avg_contig,
                    sc,
                    n_elements, diff_tot, beta1, eps, lam,
                    bound_mode, dt_mode,
                    cautious, param_has_master, use_cd,
                    BLOCK_SIZE=b_size, num_warps=n_warps
                )
            except Exception:
                ref_p = mp_ref if (param_has_master and mp_ref is not None) else p
                self._compute_pass1_single_tdclm_one(
                    p, grad, exp_avg, ref_p, sc,
                    beta1, eps, lam, bound_mode, cautious, centered_defect, diff_tot
                )
        synced_totals = _sync_fsdp_scratch_batch(
            [scratches[idx] for idx in pass1_indices],
            [params[idx].numel() for idx in pass1_indices],
        )
        return {idx: tot for idx, tot in zip(pass1_indices, synced_totals)}

    @classmethod
    def _precompute_fsdp_pass1_single_dual(
        cls,
        pass1_indices: List[int],
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        master_params: Optional[List[Tensor]],
        steps: List[int],
        scratches: List[Tensor],
        beta1: float,
        beta2: float,
        eps: float,
        tau: float,
        base_step: int,
        c1_base: float,
        c2_default: float,
        c2_fp16: float,
        bound_mode: int,
        cautious: bool,
        centered_defect: bool,
        batched_diff_totals: Optional[List[float]],
        chunk_size: int = 2_097_152,
    ) -> Dict[int, float]:
        has_master = master_params is not None
        for idx in pass1_indices:
            p = params[idx]
            sc = scratches[idx]
            sc[:3].zero_()
            n_el = p.numel()
            if n_el == 0:
                continue
            step_raw = steps[idx]
            step = int(step_raw.item() if isinstance(step_raw, torch.Tensor) else step_raw)
            safe_step = max(step, 1)
            if step == base_step:
                c1 = c1_base
                c2 = c2_fp16 if p.dtype == torch.float16 else c2_default
            else:
                effective_eps = max(eps, 1e-4) if p.dtype == torch.float16 else eps
                bc1_nes = 1.0 - (beta1 ** (safe_step + 1))
                bc2 = max(1.0 - (beta2 ** safe_step), 1e-15)
                c1 = (bc1_nes * tau) / math.sqrt(bc2)
                c2 = effective_eps * bc1_nes * tau
            param_has_master = has_master and (master_params[idx] is not p)
            ref_p = master_params[idx] if param_has_master else p
            diff_tot = batched_diff_totals[idx] if batched_diff_totals is not None else float(n_el)
            cls._compute_pass1_single_dual_one(
                p, grads[idx], exp_avgs[idx], exp_avg_sqs[idx], ref_p, sc,
                beta1, beta2, c1, c2, bound_mode, cautious, centered_defect, diff_tot, chunk_size
            )
        synced_totals = _sync_fsdp_scratch_batch(
            [scratches[idx] for idx in pass1_indices],
            [params[idx].numel() for idx in pass1_indices],
        )
        return {idx: tot for idx, tot in zip(pass1_indices, synced_totals)}

    @classmethod
    def _precompute_fsdp_pass1_single_tdclm(
        cls,
        pass1_indices: List[int],
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        master_params: Optional[List[Tensor]],
        scratches: List[Tensor],
        beta1: float,
        eps: float,
        lam: float,
        bound_mode: int,
        cautious: bool,
        centered_defect: bool,
        batched_diff_totals: Optional[List[float]],
        chunk_size: int = 2_097_152,
    ) -> Dict[int, float]:
        has_master = master_params is not None
        for idx in pass1_indices:
            p = params[idx]
            sc = scratches[idx]
            sc[:3].zero_()
            n_el = p.numel()
            if n_el == 0:
                continue
            param_has_master = has_master and (master_params[idx] is not p)
            ref_p = master_params[idx] if param_has_master else p
            diff_tot = batched_diff_totals[idx] if batched_diff_totals is not None else float(n_el)
            cls._compute_pass1_single_tdclm_one(
                p, grads[idx], exp_avgs[idx], ref_p, sc,
                beta1, eps, lam, bound_mode, cautious, centered_defect, diff_tot, chunk_size
            )
        synced_totals = _sync_fsdp_scratch_batch(
            [scratches[idx] for idx in pass1_indices],
            [params[idx].numel() for idx in pass1_indices],
        )
        return {idx: tot for idx, tot in zip(pass1_indices, synced_totals)}

    def _finalize_coalesced_fsdp_ema_triton(
        self,
        pending_ema: List[Tuple[Tensor, int, bool]],
        lr: float,
        clamp_min: float,
        ema_decay: float,
        bound_mode: int,
        bound_ratio: float,
        cautious: bool,
    ) -> None:
        if not pending_ema:
            return
        sc_list = [item[0] for item in pending_ema]
        numels = [item[1] for item in pending_ema]
        try:
            totals = _sync_fsdp_scratch_batch(sc_list, numels)
            batch_synced = True
        except Exception:
            totals = [max(float(n), 1.0) for n in numels]
            batch_synced = False
        for (sc_p, n_el_p, allow_triton), tot_el in zip(pending_ema, totals):
            if batch_synced and allow_triton and n_el_p > 0:
                try:
                    _lumina_post_step_ema_kernel[(1,)](
                        sc_p, tot_el, lr, clamp_min, ema_decay,
                        bound_mode, bound_ratio, cautious, num_warps=1
                    )
                    continue
                except Exception:
                    pass
            _fallback_post_step_ema(
                sc_p, n_el_p, lr, clamp_min, ema_decay,
                bound_mode, bound_ratio, cautious, True, batch_synced,
                synced_total_elements=tot_el
            )

    def _triton_step_dual(
        self, params, grads, exp_avgs, exp_avg_sqs, master_params, steps, scratches,
        lr, beta1, beta2, eps, weight_decay, tau, cautious, clamp_min,
        use_sr, bound=True, bound_type="coordinate", bound_ratio=0.05,
        fused_single_pass=False, ema_decay=0.8, fsdp_sync: bool = False, seed: int = 1337,
        centered_defect: bool = False,
        orig_indices: Optional[List[int]] = None,
        completed_indices: Optional[Set[int]] = None
    ):
        bound_mode = 0
        if bound and bound_ratio > 0.0:
            bound_mode = 1 if bound_type == "radial" else 2

        has_master = master_params is not None

        base_step = int(steps[0].item() if isinstance(steps[0], torch.Tensor) else steps[0])
        safe_base_step = max(base_step, 1)
        bc1_nes_base = 1.0 - (beta1 ** (safe_base_step + 1))
        bc2_base = max(1.0 - (beta2 ** safe_base_step), 1e-15)
        c1_base = (bc1_nes_base * tau) / math.sqrt(bc2_base)
        c2_default = eps * bc1_nes_base * tau
        c2_fp16 = max(eps, 1e-4) * bc1_nes_base * tau

        batched_diff_totals: Optional[List[float]] = None
        if centered_defect and fsdp_sync and len(params) > 1:
            batched_diff_totals = self._precompute_fsdp_diff_triton(params, grads, exp_avgs, scratches)

        pending_ema: List[Tuple[Tensor, int, bool]] = []
        needs_ema_sync = cautious or bound_mode == 1

        batched_scratch_totals: Dict[int, float] = {}
        if fsdp_sync and needs_ema_sync and len(params) > 1:
            pass1_indices = [
                idx for idx, st in enumerate(steps)
                if not (fused_single_pass and int(st.item() if isinstance(st, torch.Tensor) else st) > 1)
            ]
            if len(pass1_indices) > 1:
                batched_scratch_totals = self._precompute_fsdp_pass1_triton_dual(
                    pass1_indices, params, grads, exp_avgs, exp_avg_sqs, master_params,
                    steps, scratches, beta1, beta2, eps, tau, base_step, c1_base, c2_default, c2_fp16,
                    bound_mode, cautious, centered_defect, batched_diff_totals
                )

        for i in range(len(params)):
            p, grad, exp_avg, exp_avg_sq, step_raw, scratch_param = (
                params[i], grads[i], exp_avgs[i], exp_avg_sqs[i], steps[i], scratches[i]
            )
            step = int(step_raw.item() if isinstance(step_raw, torch.Tensor) else step_raw)
            global_param_idx = orig_indices[i] if orig_indices is not None else i
            cur_scratch_synced = i in batched_scratch_totals
            cur_scratch_total = batched_scratch_totals[i] if cur_scratch_synced else float(p.numel())

            if p.numel() == 0:
                if fsdp_sync:
                    if not cur_scratch_synced:
                        scratch_param[:3].zero_()
                    if batched_diff_totals is None:
                        scratch_param[5].zero_()
                        if centered_defect:
                            _sync_fsdp_diff_only(scratch_param, 0)
                    if fused_single_pass and step > 1 and needs_ema_sync:
                        pending_ema.append((scratch_param, 0, False))
                    else:
                        _sync_empty_fsdp_shard(
                            scratch_param, step, lr, clamp_min, ema_decay,
                            bound_mode, bound_ratio, cautious, centered_defect, fused_single_pass,
                            diff_synced=True, scratch_synced=cur_scratch_synced,
                            synced_total_elements=cur_scratch_total,
                        )
                if completed_indices is not None:
                    completed_indices.add(global_param_idx)
                continue

            param_has_master = has_master and (master_params[i] is not p)
            mp_ref = master_params[i] if param_has_master else None
            p_contig = p
            exp_avg_contig = exp_avg
            exp_avg_sq_contig = exp_avg_sq
            mp_contig = mp_ref if param_has_master and mp_ref is not None else p
            is_all_contig = True
            diff_synced = batched_diff_totals is not None
            scratch_synced = cur_scratch_synced
            param_updated = False
            ema_updated = False
            copied_back = False
            n_elements = p.numel()
            diff_total_elements = batched_diff_totals[i] if batched_diff_totals is not None else float(n_elements)
            total_elements = cur_scratch_total
            defer_this_ema = bool(fsdp_sync and fused_single_pass and step > 1 and needs_ema_sync)

            try:
                (
                    n_elements, b_size, n_warps, c_grid, dt_mode, s_dt_mode, is_all_contig,
                    p_contig, grad_contig, exp_avg_contig, exp_avg_sq_contig, mp_contig
                ) = self._prepare_triton_buffers(p, grad, exp_avg, exp_avg_sq, mp_ref, param_has_master, reuse_contig=cur_scratch_synced)

                safe_step = max(step, 1)
                if step == base_step:
                    c1 = c1_base
                    c2 = c2_fp16 if p.dtype == torch.float16 else c2_default
                else:
                    effective_eps = max(eps, 1e-4) if p.dtype == torch.float16 else eps
                    bc1_nes = 1.0 - (beta1 ** (safe_step + 1))
                    bc2 = max(1.0 - (beta2 ** safe_step), 1e-15)
                    c1 = (bc1_nes * tau) / math.sqrt(bc2)
                    c2 = effective_eps * bc1_nes * tau

                kernel_seed = (seed + step * 0x9E3779B9 + global_param_idx * 10007) & 0x7FFFFFFF

                use_cd = bool(centered_defect and (n_elements > 1 or fsdp_sync))
                if use_cd:
                    if not diff_synced:
                        scratch_param[5].zero_()
                        _lumina_diff_sum_kernel[c_grid](
                            grad_contig, exp_avg_contig, scratch_param,
                            n_elements, BLOCK_SIZE=b_size, num_warps=n_warps
                        )
                        if fsdp_sync:
                            diff_total_elements = _sync_fsdp_diff_only(scratch_param, n_elements)
                            diff_synced = True
                    if diff_total_elements <= 1.0:
                        use_cd = False

                if not cautious and bound_mode != 1:
                    _lumina_dual_single_pass_kernel[c_grid](
                        p_contig, mp_contig, grad_contig, exp_avg_contig, exp_avg_sq_contig,
                        scratch_param,
                        n_elements, diff_total_elements, beta1, beta2, c1, c2, lr, weight_decay,
                        bound_mode, bound_ratio,
                        kernel_seed, dt_mode, s_dt_mode, use_sr, param_has_master, use_cd,
                        BLOCK_SIZE=b_size, num_warps=n_warps
                    )
                    param_updated = True
                    if use_cd or diff_synced:
                        scratch_param[5].zero_()
                    ema_updated = True
                elif fused_single_pass and step > 1:
                    scratch_param[:3].zero_()
                    _lumina_dual_fused_single_pass_kernel[c_grid](
                        p_contig, mp_contig, grad_contig, exp_avg_contig, exp_avg_sq_contig,
                        scratch_param,
                        n_elements, diff_total_elements, beta1, beta2, c1, c2, lr, weight_decay,
                        bound_mode, bound_ratio,
                        kernel_seed, dt_mode, s_dt_mode, use_sr,
                        cautious, param_has_master, use_cd, BLOCK_SIZE=b_size, num_warps=n_warps
                    )
                    param_updated = True
                    if defer_this_ema:
                        pending_ema.append((scratch_param, n_elements, True))
                        ema_updated = True
                    else:
                        total_elements = float(n_elements)
                        if fsdp_sync and needs_ema_sync:
                            total_elements = _sync_fsdp_scratch(scratch_param, n_elements)
                            scratch_synced = True
                        _lumina_post_step_ema_kernel[(1,)](
                            scratch_param, total_elements, lr, clamp_min, ema_decay,
                            bound_mode, bound_ratio, cautious, num_warps=1
                        )
                        ema_updated = True
                else:
                    if not scratch_synced:
                        scratch_param[:3].zero_()
                        _lumina_dual_pass1_kernel[c_grid](
                            p_contig, mp_contig, grad_contig, exp_avg_contig, exp_avg_sq_contig,
                            scratch_param,
                            n_elements, diff_total_elements, beta1, beta2, c1, c2,
                            bound_mode, dt_mode, s_dt_mode,
                            cautious, param_has_master, use_cd,
                            BLOCK_SIZE=b_size, num_warps=n_warps
                        )
                        total_elements = float(n_elements)
                        if fsdp_sync and needs_ema_sync:
                            total_elements = _sync_fsdp_scratch(scratch_param, n_elements)
                            scratch_synced = True
                    _lumina_dual_pass2_kernel[c_grid](
                        p_contig, mp_contig, grad_contig, exp_avg_contig, exp_avg_sq_contig,
                        scratch_param,
                        n_elements, total_elements, beta1, beta2, c1, c2, lr, weight_decay, clamp_min,
                        bound_mode, bound_ratio,
                        kernel_seed, dt_mode, s_dt_mode, use_sr, cautious, param_has_master, use_cd,
                        BLOCK_SIZE=b_size, num_warps=n_warps
                    )
                    param_updated = True
                    scratch_param[:3].zero_()
                    if use_cd or diff_synced:
                        scratch_param[5].zero_()
                    ema_updated = True

                if not is_all_contig:
                    self._copy_back_non_contig(
                        p, p_contig, exp_avg, exp_avg_contig, exp_avg_sq, exp_avg_sq_contig,
                        mp_ref, mp_contig, param_has_master
                    )
                    copied_back = True

                if completed_indices is not None:
                    completed_indices.add(global_param_idx)
            except Exception as e:
                if not self._warned_triton_failure:
                    err_line = str(e).strip().splitlines()[0] if str(e).strip() else "Unknown error"
                    logger.warning(f"LuminaV: Triton kernel execution failed ({err_line}). Falling back to PyTorch single-step.")
                    self._warned_triton_failure = True

                if p.is_cuda:
                    with contextlib.suppress(Exception):
                        torch.cuda.synchronize(p.device)

                if param_updated:
                    if not is_all_contig and not copied_back:
                        self._copy_back_non_contig(
                            p, p_contig, exp_avg, exp_avg_contig, exp_avg_sq, exp_avg_sq_contig,
                            mp_ref, mp_contig, param_has_master
                        )
                    if (fused_single_pass and step > 1) and needs_ema_sync and not defer_this_ema and not ema_updated:
                        _fallback_post_step_ema(
                            scratch_param, n_elements, lr, clamp_min, ema_decay,
                            bound_mode, bound_ratio, cautious, fsdp_sync, scratch_synced,
                            synced_total_elements=total_elements
                        )
                else:
                    if not scratch_synced:
                        scratch_param[:3].zero_()
                    if not diff_synced:
                        scratch_param[5].zero_()
                    if step <= 1:
                        scratch_param[3] = 1.0
                        scratch_param[4] = 1.0
                    self._single_step_dual(
                        [p], [grad], [exp_avg], [exp_avg_sq],
                        [mp_ref] if param_has_master else None,
                        [step], [scratch_param], lr, beta1, beta2, eps, weight_decay, tau,
                        cautious, clamp_min, use_sr, bound, bound_type, bound_ratio,
                        fused_single_pass, ema_decay, fsdp_sync, seed, centered_defect,
                        diff_synced=diff_synced, scratch_synced=scratch_synced,
                        diff_total_elements=diff_total_elements,
                        scratch_total_elements=total_elements,
                        orig_indices=[global_param_idx],
                        defer_fsdp_post_ema=defer_this_ema,
                    )
                    if defer_this_ema:
                        pending_ema.append((scratch_param, n_elements, False))
                if completed_indices is not None:
                    completed_indices.add(global_param_idx)

        if pending_ema:
            self._finalize_coalesced_fsdp_ema_triton(
                pending_ema, lr, clamp_min, ema_decay, bound_mode, bound_ratio, cautious
            )
            if completed_indices is not None:
                for idx_done in range(len(params)):
                    completed_indices.add(orig_indices[idx_done] if orig_indices is not None else idx_done)

    def _triton_step_tdclm(
        self, params, grads, exp_avgs, master_params, steps, scratches,
        lr, beta1, eps, weight_decay, lam, cautious, clamp_min,
        use_sr, bound=True, bound_type="coordinate", bound_ratio=0.05,
        fused_single_pass: bool = True, ema_decay: float = 0.8,
        fsdp_sync: bool = False, seed: int = 1337, centered_defect: bool = False,
        orig_indices: Optional[List[int]] = None,
        completed_indices: Optional[Set[int]] = None
    ):
        bound_mode = 0
        if bound and bound_ratio > 0.0:
            bound_mode = 1 if bound_type == "radial" else 2

        has_master = master_params is not None

        batched_diff_totals: Optional[List[float]] = None
        if centered_defect and fsdp_sync and len(params) > 1:
            batched_diff_totals = self._precompute_fsdp_diff_triton(params, grads, exp_avgs, scratches)

        pending_ema: List[Tuple[Tensor, int, bool]] = []
        needs_ema_sync = cautious or bound_mode == 1

        batched_scratch_totals: Dict[int, float] = {}
        if fsdp_sync and needs_ema_sync and len(params) > 1:
            pass1_indices = [
                idx for idx, st in enumerate(steps)
                if not (fused_single_pass and int(st.item() if isinstance(st, torch.Tensor) else st) > 1)
            ]
            if len(pass1_indices) > 1:
                batched_scratch_totals = self._precompute_fsdp_pass1_triton_tdclm(
                    pass1_indices, params, grads, exp_avgs, master_params, scratches,
                    beta1, eps, lam, bound_mode, cautious, centered_defect, batched_diff_totals
                )

        for i in range(len(params)):
            p, grad, exp_avg, step_raw, scratch_param = (
                params[i], grads[i], exp_avgs[i], steps[i], scratches[i]
            )
            step = int(step_raw.item() if isinstance(step_raw, torch.Tensor) else step_raw)
            global_param_idx = orig_indices[i] if orig_indices is not None else i
            cur_scratch_synced = i in batched_scratch_totals
            cur_scratch_total = batched_scratch_totals[i] if cur_scratch_synced else float(p.numel())

            if p.numel() == 0:
                if fsdp_sync:
                    if not cur_scratch_synced:
                        scratch_param[:3].zero_()
                    if batched_diff_totals is None:
                        scratch_param[5].zero_()
                        if centered_defect:
                            _sync_fsdp_diff_only(scratch_param, 0)
                    if fused_single_pass and step > 1 and needs_ema_sync:
                        pending_ema.append((scratch_param, 0, False))
                    else:
                        _sync_empty_fsdp_shard(
                            scratch_param, step, lr, clamp_min, ema_decay,
                            bound_mode, bound_ratio, cautious, centered_defect, fused_single_pass,
                            diff_synced=True, scratch_synced=cur_scratch_synced,
                            synced_total_elements=cur_scratch_total,
                        )
                if completed_indices is not None:
                    completed_indices.add(global_param_idx)
                continue

            param_has_master = has_master and (master_params[i] is not p)
            mp_ref = master_params[i] if param_has_master else None
            p_contig = p
            exp_avg_contig = exp_avg
            mp_contig = mp_ref if param_has_master and mp_ref is not None else p
            is_all_contig = True
            diff_synced = batched_diff_totals is not None
            scratch_synced = cur_scratch_synced
            param_updated = False
            ema_updated = False
            copied_back = False
            n_elements = p.numel()
            diff_total_elements = batched_diff_totals[i] if batched_diff_totals is not None else float(n_elements)
            total_elements = cur_scratch_total
            defer_this_ema = bool(fsdp_sync and fused_single_pass and step > 1 and needs_ema_sync)

            try:
                (
                    n_elements, b_size, n_warps, c_grid, dt_mode, s_dt_mode, is_all_contig,
                    p_contig, grad_contig, exp_avg_contig, _, mp_contig
                ) = self._prepare_triton_buffers(p, grad, exp_avg, None, mp_ref, param_has_master, reuse_contig=cur_scratch_synced)

                kernel_seed = (seed + step * 0x9E3779B9 + global_param_idx * 10007) & 0x7FFFFFFF

                use_cd = bool(centered_defect and (n_elements > 1 or fsdp_sync))
                if use_cd:
                    if not diff_synced:
                        scratch_param[5].zero_()
                        _lumina_diff_sum_kernel[c_grid](
                            grad_contig, exp_avg_contig, scratch_param,
                            n_elements, BLOCK_SIZE=b_size, num_warps=n_warps
                        )
                        if fsdp_sync:
                            diff_total_elements = _sync_fsdp_diff_only(scratch_param, n_elements)
                            diff_synced = True
                    if diff_total_elements <= 1.0:
                        use_cd = False

                if (fused_single_pass and step > 1) or (not cautious and bound_mode != 1):
                    if needs_ema_sync:
                        scratch_param[:3].zero_()
                    _lumina_tdclm_single_pass_kernel[c_grid](
                        p_contig, mp_contig, grad_contig, exp_avg_contig,
                        scratch_param,
                        n_elements, diff_total_elements, beta1, lr, weight_decay, eps, lam,
                        bound_mode, bound_ratio,
                        kernel_seed, dt_mode, s_dt_mode,
                        use_sr, cautious, param_has_master, use_cd,
                        BLOCK_SIZE=b_size, num_warps=n_warps
                    )
                    param_updated = True
                    if needs_ema_sync:
                        if defer_this_ema:
                            pending_ema.append((scratch_param, n_elements, True))
                            ema_updated = True
                        else:
                            total_elements = float(n_elements)
                            if fsdp_sync:
                                total_elements = _sync_fsdp_scratch(scratch_param, n_elements)
                                scratch_synced = True
                            _lumina_post_step_ema_kernel[(1,)](
                                scratch_param, total_elements, lr, clamp_min, ema_decay,
                                bound_mode, bound_ratio, cautious, num_warps=1
                            )
                            ema_updated = True
                    else:
                        if use_cd or diff_synced:
                            scratch_param[5].zero_()
                        ema_updated = True
                else:
                    if not scratch_synced:
                        scratch_param[:3].zero_()
                        _lumina_tdclm_pass1_kernel[c_grid](
                            p_contig, mp_contig, grad_contig, exp_avg_contig,
                            scratch_param,
                            n_elements, diff_total_elements, beta1, eps, lam,
                            bound_mode, dt_mode,
                            cautious, param_has_master, use_cd,
                            BLOCK_SIZE=b_size, num_warps=n_warps
                        )
                        total_elements = float(n_elements)
                        if fsdp_sync and needs_ema_sync:
                            total_elements = _sync_fsdp_scratch(scratch_param, n_elements)
                            scratch_synced = True
                    _lumina_tdclm_pass2_kernel[c_grid](
                        p_contig, mp_contig, grad_contig, exp_avg_contig,
                        scratch_param,
                        n_elements, total_elements, beta1, lr, weight_decay, eps, lam, clamp_min,
                        bound_mode, bound_ratio,
                        kernel_seed, dt_mode, s_dt_mode, use_sr,
                        cautious, param_has_master, use_cd,
                        BLOCK_SIZE=b_size, num_warps=n_warps
                    )
                    param_updated = True
                    scratch_param[:3].zero_()
                    if use_cd or diff_synced:
                        scratch_param[5].zero_()
                    ema_updated = True

                if not is_all_contig:
                    self._copy_back_non_contig(
                        p, p_contig, exp_avg, exp_avg_contig, None, None,
                        mp_ref, mp_contig, param_has_master
                    )
                    copied_back = True

                if completed_indices is not None:
                    completed_indices.add(global_param_idx)
            except Exception as e:
                if not self._warned_triton_failure:
                    err_line = str(e).strip().splitlines()[0] if str(e).strip() else "Unknown error"
                    logger.warning(f"LuminaV: Triton kernel execution failed ({err_line}). Falling back to PyTorch single-step.")
                    self._warned_triton_failure = True

                if p.is_cuda:
                    with contextlib.suppress(Exception):
                        torch.cuda.synchronize(p.device)

                if param_updated:
                    if not is_all_contig and not copied_back:
                        self._copy_back_non_contig(
                            p, p_contig, exp_avg, exp_avg_contig, None, None,
                            mp_ref, mp_contig, param_has_master
                        )
                    if (fused_single_pass and step > 1) and needs_ema_sync and not defer_this_ema and not ema_updated:
                        _fallback_post_step_ema(
                            scratch_param, n_elements, lr, clamp_min, ema_decay,
                            bound_mode, bound_ratio, cautious, fsdp_sync, scratch_synced,
                            synced_total_elements=total_elements
                        )
                else:
                    if not scratch_synced:
                        scratch_param[:3].zero_()
                    if not diff_synced:
                        scratch_param[5].zero_()
                    if step <= 1:
                        scratch_param[3] = 1.0
                        scratch_param[4] = 1.0
                    self._single_step_tdclm(
                        [p], [grad], [exp_avg],
                        [mp_ref] if param_has_master else None,
                        [step], [scratch_param], lr, beta1, eps, weight_decay, lam,
                        cautious, clamp_min, use_sr, bound, bound_type, bound_ratio,
                        fused_single_pass=fused_single_pass, ema_decay=ema_decay,
                        fsdp_sync=fsdp_sync, seed=seed, centered_defect=centered_defect,
                        diff_synced=diff_synced, scratch_synced=scratch_synced,
                        diff_total_elements=diff_total_elements,
                        scratch_total_elements=total_elements,
                        orig_indices=[global_param_idx],
                        defer_fsdp_post_ema=defer_this_ema,
                    )
                    if defer_this_ema:
                        pending_ema.append((scratch_param, n_elements, False))
                if completed_indices is not None:
                    completed_indices.add(global_param_idx)

        if pending_ema:
            self._finalize_coalesced_fsdp_ema_triton(
                pending_ema, lr, clamp_min, ema_decay, bound_mode, bound_ratio, cautious
            )
            if completed_indices is not None:
                for idx_done in range(len(params)):
                    completed_indices.add(orig_indices[idx_done] if orig_indices is not None else idx_done)

    @staticmethod
    def _precompute_fsdp_diff_single(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        scratches: List[Tensor],
        chunk_size: int,
    ) -> List[float]:
        for p, grad, exp_avg, sc in zip(params, grads, exp_avgs, scratches):
            sc[5].zero_()
            n_el = p.numel()
            if n_el == 0:
                continue
            if n_el <= chunk_size:
                g_clean = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                sc[5].copy_((g_clean - exp_avg.float()).clamp(-1e18, 1e18).sum().to(torch.float32))
            else:
                flat_g = (grad if grad.is_contiguous() else grad.contiguous()).view(-1)
                flat_m = (exp_avg if exp_avg.is_contiguous() else exp_avg.contiguous()).view(-1)
                diff_acc = 0.0
                for offset in range(0, n_el, chunk_size):
                    cur = min(chunk_size, n_el - offset)
                    g_c = torch.nan_to_num(flat_g[offset : offset + cur], nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                    m_c = flat_m[offset : offset + cur].float()
                    diff_acc += (g_c - m_c).clamp(-1e18, 1e18).sum().item()
                sc[5] = diff_acc
        return _sync_fsdp_diff_batch(scratches, [p.numel() for p in params])

    def _single_step_dual(
        self, params, grads, exp_avgs, exp_avg_sqs, master_params, steps, scratches,
        lr, beta1, beta2, eps, weight_decay, tau, cautious, clamp_min,
        use_sr, bound=True, bound_type="coordinate", bound_ratio=0.05,
        fused_single_pass=False, ema_decay=0.8, fsdp_sync: bool = False, seed: int = 1337,
        centered_defect: bool = False,
        diff_synced: bool = False, scratch_synced: bool = False,
        diff_total_elements: float = 0.0, scratch_total_elements: float = 0.0,
        orig_indices: Optional[List[int]] = None,
        defer_fsdp_post_ema: bool = False,
    ):
        bound_mode = 0
        if bound and bound_ratio > 0.0:
            bound_mode = 1 if bound_type == "radial" else 2

        base_step = int(steps[0].item() if isinstance(steps[0], torch.Tensor) else steps[0])
        safe_base_step = max(base_step, 1)
        bc1_nes_base = 1.0 - (beta1 ** (safe_base_step + 1))
        bc2_base = max(1.0 - (beta2 ** safe_base_step), 1e-15)
        c1_base = (bc1_nes_base * tau) / math.sqrt(bc2_base)
        c2_default = eps * bc1_nes_base * tau
        c2_fp16 = max(eps, 1e-4) * bc1_nes_base * tau

        ELEM_CHUNK_SIZE = 2_097_152

        batched_diff_totals: Optional[List[float]] = None
        if centered_defect and fsdp_sync and not diff_synced and scratches is not None and len(params) > 1:
            batched_diff_totals = self._precompute_fsdp_diff_single(
                params, grads, exp_avgs, scratches, ELEM_CHUNK_SIZE
            )

        needs_ema_sync = cautious or bound_mode == 1
        batch_post_ema = bool(
            fsdp_sync and not scratch_synced and not defer_fsdp_post_ema
            and scratches is not None and len(params) > 1 and needs_ema_sync
        )
        batched_scratch_totals: Dict[int, float] = {}
        if batch_post_ema:
            pass1_indices = [
                idx for idx, st in enumerate(steps)
                if not (fused_single_pass and int(st.item() if isinstance(st, torch.Tensor) else st) > 1)
            ]
            if len(pass1_indices) > 1:
                batched_scratch_totals = self._precompute_fsdp_pass1_single_dual(
                    pass1_indices, params, grads, exp_avgs, exp_avg_sqs, master_params,
                    steps, scratches, beta1, beta2, eps, tau, base_step, c1_base, c2_default, c2_fp16,
                    bound_mode, cautious, centered_defect, batched_diff_totals, ELEM_CHUNK_SIZE
                )
        pending_ema_single: List[Tuple[Tensor, int]] = []

        for i in range(len(params)):
            p, grad, exp_avg, exp_avg_sq, step_raw = params[i], grads[i], exp_avgs[i], exp_avg_sqs[i], steps[i]
            step = int(step_raw.item() if isinstance(step_raw, torch.Tensor) else step_raw)
            global_param_idx = orig_indices[i] if orig_indices is not None else i
            param_has_master = master_params is not None and (master_params[i] is not p)
            ref_p = master_params[i] if param_has_master else p
            sc = scratches[i] if scratches is not None else None
            n_elem = p.numel()
            cur_diff_synced = diff_synced or (batched_diff_totals is not None)
            cur_diff_total = batched_diff_totals[i] if batched_diff_totals is not None else diff_total_elements
            cur_scratch_synced = scratch_synced or (i in batched_scratch_totals)
            cur_scratch_total = batched_scratch_totals[i] if (i in batched_scratch_totals) else scratch_total_elements
            defer_cur_ema = defer_fsdp_post_ema or (batch_post_ema and fused_single_pass and step > 1)

            if n_elem == 0:
                if fsdp_sync and sc is not None:
                    if not cur_scratch_synced:
                        sc[:3].zero_()
                    if not cur_diff_synced:
                        sc[5].zero_()
                        if centered_defect:
                            _sync_fsdp_diff_only(sc, 0)
                    if defer_cur_ema and needs_ema_sync:
                        if batch_post_ema:
                            pending_ema_single.append((sc, 0))
                    else:
                        _sync_empty_fsdp_shard(
                            sc, step, lr, clamp_min, ema_decay,
                            bound_mode, bound_ratio, cautious, centered_defect, fused_single_pass,
                            diff_synced=True, scratch_synced=cur_scratch_synced,
                            synced_total_elements=cur_scratch_total,
                        )
                continue

            safe_step = max(step, 1)
            if step == base_step:
                c1 = c1_base
                c2 = c2_fp16 if p.dtype == torch.float16 else c2_default
            else:
                effective_eps = max(eps, 1e-4) if p.dtype == torch.float16 else eps
                bc1_nes = 1.0 - (beta1 ** (safe_step + 1))
                bc2 = max(1.0 - (beta2 ** safe_step), 1e-15)
                c1 = (bc1_nes * tau) / math.sqrt(bc2)
                c2 = effective_eps * bc1_nes * tau

            if n_elem > ELEM_CHUNK_SIZE:
                self._single_step_dual_element_chunked(
                    p, grad, exp_avg, exp_avg_sq, ref_p, step, sc,
                    c1, c2, lr, beta1, beta2, weight_decay, cautious, clamp_min,
                    use_sr, bound, bound_type, bound_ratio,
                    fused_single_pass, ema_decay, fsdp_sync, seed, ELEM_CHUNK_SIZE,
                    centered_defect,
                    diff_synced=cur_diff_synced, scratch_synced=cur_scratch_synced,
                    diff_total_elements=cur_diff_total,
                    scratch_total_elements=cur_scratch_total,
                    param_index=global_param_idx,
                    defer_fsdp_post_ema=defer_cur_ema,
                )
                if batch_post_ema and defer_cur_ema and sc is not None:
                    pending_ema_single.append((sc, n_elem))
                continue

            need_sr_m = use_sr and (exp_avg.dtype in (torch.bfloat16, torch.float16))
            need_sr_p = use_sr and (not param_has_master) and (p.dtype in (torch.bfloat16, torch.float16))
            gen_m, gen_p = self._get_step_generators(
                p.device, seed, step, global_param_idx, need_sr_m, need_sr_p
            )

            grad_clean = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            g_f32 = grad_clean.float().clamp(-1e18, 1e18)
            m_f32 = exp_avg.float()
            v_f32 = exp_avg_sq.float()

            m_new = m_f32.mul(beta1).add(g_f32, alpha=1.0 - beta1)
            clm_m = m_new.mul(beta1).add(g_f32, alpha=1.0 - beta1)
            diff = (g_f32 - m_new).clamp(-1e18, 1e18)
            if centered_defect:
                if fsdp_sync and sc is not None:
                    if not cur_diff_synced:
                        sc[5].copy_((g_f32 - m_f32).clamp(-1e18, 1e18).sum().to(torch.float32))
                        tot_cd = _sync_fsdp_diff_only(sc, n_elem)
                    else:
                        tot_cd = cur_diff_total if cur_diff_total > 0.0 else float(n_elem)
                    if tot_cd > 1.0:
                        diff = (diff - (sc[5] / tot_cd) * beta1).clamp(-1e18, 1e18)
                    sc[5].zero_()
                elif diff.numel() > 1:
                    diff_mean = (g_f32 - m_f32).clamp(-1e18, 1e18).mean() * beta1
                    diff = (diff - diff_mean).clamp(-1e18, 1e18)

            v_new = v_f32.mul(beta2).addcmul(diff, diff, value=1.0 - beta2)
            if p.dtype == torch.float16 or exp_avg.dtype == torch.float16 or exp_avg_sq.dtype == torch.float16:
                v_new = v_new.clamp(0.0, 65000.0)

            sigma = v_new.clamp_min(0.0).sqrt().mul(c1).add(c2)
            u = torch.tanh(clm_m / sigma.clamp_min(1e-12))

            if exp_avg.dtype in (torch.bfloat16, torch.float16):
                _sr_quantize(exp_avg, m_new, use_sr=use_sr, generator=gen_m)
            else:
                exp_avg.copy_(m_new.to(dtype=exp_avg.dtype))
            exp_avg_sq.copy_(v_new.to(dtype=exp_avg_sq.dtype))

            if fused_single_pass and sc is not None and step > 1:
                cautious_scale = sc[3].clone()
                bound_scale = sc[4].clone()

                if cautious:
                    m_mask = (u * g_f32 > 0).float()
                    delta_theta = u * m_mask * cautious_scale
                else:
                    m_mask = None
                    delta_theta = u

                step_val = _compute_step_val(
                    delta_theta, ref_p, lr, bound, bound_type, bound_ratio,
                    target_dtype=p.dtype,
                    radial_scale=bound_scale if bound_mode == 1 else None,
                )

                if not (fsdp_sync and cur_scratch_synced):
                    sc[:3].zero_()
                    if cautious and m_mask is not None:
                        sc[0].copy_(m_mask.sum().to(torch.float32))
                    if bound_mode == 1:
                        u_eff = (u * m_mask) if (cautious and m_mask is not None) else u
                        sc[1].copy_(torch.sum(u_eff.float() ** 2))
                        sc[2].copy_(torch.sum(ref_p.float() ** 2))

                if defer_cur_ema:
                    if batch_post_ema:
                        pending_ema_single.append((sc, n_elem))
                else:
                    total_elements = float(n_elem)
                    if fsdp_sync and needs_ema_sync:
                        if not cur_scratch_synced:
                            total_elements = _sync_fsdp_scratch(sc, n_elem)
                        else:
                            total_elements = cur_scratch_total if cur_scratch_total > 0.0 else float(n_elem)

                    if cautious and m_mask is not None:
                        m_bar = (sc[0] / total_elements).clamp_(min=clamp_min, max=1.0)
                        target_cautious = 1.0 / m_bar
                        new_cautious = ema_decay * cautious_scale + (1.0 - ema_decay) * target_cautious
                        sc[3].copy_(new_cautious)

                    if bound_mode == 1:
                        c_eff = sc[3] if cautious else 1.0
                        norm_delta = torch.sqrt(torch.clamp_min(sc[1], 0.0)) * c_eff
                        norm_p = torch.sqrt(torch.clamp_min(sc[2], 0.0))
                        R = bound_ratio * torch.clamp_min(norm_p, 1.0)
                        r = (lr * norm_delta) / R
                        target_bound = torch.where(r < 1e-4, torch.ones_like(r), torch.tanh(r) / torch.clamp_min(r, 1e-8))
                        new_bound = ema_decay * bound_scale + (1.0 - ema_decay) * target_bound
                        sc[4].copy_(new_bound)

                    sc[:3].zero_()
                    sc[5].zero_()
            else:
                update = u
                mask: Optional[Tensor] = None
                if sc is not None and not (fsdp_sync and cur_scratch_synced):
                    sc[:3].zero_()

                if cautious:
                    mask = (u * g_f32 > 0).float()
                    if sc is not None and not (fsdp_sync and cur_scratch_synced):
                        sc[0].copy_(mask.sum().to(torch.float32))

                u_sq_sum: Optional[Tensor] = None
                p_sq_sum: Optional[Tensor] = None
                if bound_mode == 1 and not (fsdp_sync and cur_scratch_synced and sc is not None):
                    u_eff = (u * mask) if (cautious and mask is not None) else u
                    u_sq_sum = torch.sum(u_eff.float() ** 2)
                    p_sq_sum = torch.sum(ref_p.float() ** 2)
                    if sc is not None:
                        sc[1].copy_(u_sq_sum)
                        sc[2].copy_(p_sq_sum)

                total_elements = float(n_elem)
                if fsdp_sync and sc is not None and needs_ema_sync:
                    if not cur_scratch_synced:
                        total_elements = _sync_fsdp_scratch(sc, n_elem)
                    else:
                        total_elements = cur_scratch_total if cur_scratch_total > 0.0 else float(n_elem)

                cautious_scale_val: Optional[Tensor] = None
                if cautious and mask is not None:
                    if fsdp_sync and sc is not None:
                        m_bar = (sc[0] / total_elements).clamp_(min=clamp_min, max=1.0)
                    else:
                        m_bar = mask.mean().clamp_(min=clamp_min, max=1.0)
                    cautious_scale_val = 1.0 / m_bar
                    update = u.mul(mask).mul(cautious_scale_val)
                    if sc is not None:
                        sc[3].copy_(cautious_scale_val.to(torch.float32))

                b_scale: Optional[Tensor] = None
                if bound_mode == 1:
                    if sc is not None:
                        c_scale = sc[3] if cautious else 1.0
                        norm_u = torch.sqrt(torch.clamp_min(sc[1], 0.0)) * c_scale
                        norm_p = torch.sqrt(torch.clamp_min(sc[2], 0.0))
                    else:
                        c_scale_f = cautious_scale_val.float() if (cautious and cautious_scale_val is not None) else 1.0
                        norm_u = torch.sqrt(torch.clamp_min(u_sq_sum, 0.0)) * c_scale_f
                        norm_p = torch.sqrt(torch.clamp_min(p_sq_sum, 0.0))
                    R = bound_ratio * torch.clamp_min(norm_p, 1.0)
                    r = (lr * norm_u) / R
                    b_scale = torch.where(r < 1e-4, torch.ones_like(r), torch.tanh(r) / torch.clamp_min(r, 1e-8))
                    if sc is not None:
                        sc[4].copy_(b_scale.to(torch.float32))

                step_val = _compute_step_val(
                    update, ref_p, lr, bound, bound_type, bound_ratio,
                    target_dtype=p.dtype, radial_scale=b_scale,
                )
                if sc is not None:
                    sc[:3].zero_()
                    sc[5].zero_()

            if param_has_master:
                mp = master_params[i]
                if weight_decay != 0.0:
                    mp.mul_(1.0 - lr * weight_decay)
                mp.add_(step_val, alpha=-1.0)
                if p.dtype == torch.float16:
                    p.copy_(mp.clamp(-65504.0, 65504.0).to(dtype=p.dtype))
                elif p.dtype == torch.bfloat16:
                    p.copy_(mp.clamp(-3.3895313892515355e38, 3.3895313892515355e38).to(dtype=p.dtype))
                else:
                    p.copy_(mp.to(dtype=p.dtype))
            else:
                _sr_apply_step_val(p, step_val, lr, weight_decay, use_sr, generator=gen_p)

        if pending_ema_single:
            sc_list = [item[0] for item in pending_ema_single]
            numels = [item[1] for item in pending_ema_single]
            totals = _sync_fsdp_scratch_batch(sc_list, numels)
            for (sc_p, n_el_p), tot_el in zip(pending_ema_single, totals):
                _fallback_post_step_ema(
                    sc_p, n_el_p, lr, clamp_min, ema_decay,
                    bound_mode, bound_ratio, cautious, True, True,
                    synced_total_elements=tot_el
                )

    def _single_step_dual_element_chunked(
        self, p, grad, exp_avg, exp_avg_sq, ref_p, step, sc,
        c1, c2, lr, beta1, beta2, weight_decay, cautious, clamp_min,
        use_sr, bound, bound_type, bound_ratio,
        fused_single_pass, ema_decay, fsdp_sync, seed, chunk_size,
        centered_defect: bool = False,
        diff_synced: bool = False, scratch_synced: bool = False,
        diff_total_elements: float = 0.0, scratch_total_elements: float = 0.0,
        param_index: int = 0,
        defer_fsdp_post_ema: bool = False,
    ):
        n_elem = p.numel()
        has_master = ref_p is not None and ref_p is not p
        p_work = p if p.is_contiguous() else p.contiguous()
        g_work = grad if grad.is_contiguous() else grad.contiguous()
        m_work = exp_avg if exp_avg.is_contiguous() else exp_avg.contiguous()
        v_work = exp_avg_sq if exp_avg_sq.is_contiguous() else exp_avg_sq.contiguous()
        if has_master and ref_p is not None:
            ref_work = ref_p if ref_p.is_contiguous() else ref_p.contiguous()
        else:
            ref_work = p_work

        flat_p = p_work.view(-1)
        flat_g = g_work.view(-1)
        flat_m = m_work.view(-1)
        flat_v = v_work.view(-1)
        flat_ref = ref_work.view(-1)

        need_sr_m = use_sr and (exp_avg.dtype in (torch.bfloat16, torch.float16))
        need_sr_p = use_sr and (not has_master) and (p.dtype in (torch.bfloat16, torch.float16))
        gen_m, gen_p = self._get_step_generators(
            p.device, seed, step, param_index, need_sr_m, need_sr_p
        )

        diff_mean_scaled = 0.0
        use_cd = False
        if centered_defect and (n_elem > 1 or fsdp_sync):
            if fsdp_sync and diff_synced and sc is not None:
                diff_sum_acc = sc[5].item()
                tot_cd = diff_total_elements if diff_total_elements > 0.0 else float(n_elem)
            else:
                diff_sum_acc = 0.0
                for offset in range(0, n_elem, chunk_size):
                    cur = min(chunk_size, n_elem - offset)
                    g_c = torch.nan_to_num(flat_g[offset : offset + cur], nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                    m_c = flat_m[offset : offset + cur].float()
                    diff_sum_acc += (g_c - m_c).clamp(-1e18, 1e18).sum().item()
                tot_cd = float(n_elem)
                if fsdp_sync and not diff_synced and sc is not None:
                    sc[5] = diff_sum_acc
                    tot_cd = _sync_fsdp_diff_only(sc, n_elem)
                    diff_sum_acc = sc[5].item()
            if tot_cd > 1.0:
                diff_mean_scaled = (diff_sum_acc / tot_cd) * beta1
                use_cd = True

        mask_sum_acc = 0.0
        u_sq_sum_acc = 0.0
        p_sq_sum_acc = 0.0
        needs_global_reduction = cautious or (bound and bound_type == "radial" and bound_ratio > 0.0)

        if (fused_single_pass and sc is not None and step > 1) or not needs_global_reduction:
            cautious_scale = sc[3].item() if (cautious and sc is not None and step > 1) else 1.0
            bound_scale = sc[4].item() if (bound and bound_type == "radial" and bound_ratio > 0.0 and sc is not None and step > 1) else 1.0

            for offset in range(0, n_elem, chunk_size):
                cur = min(chunk_size, n_elem - offset)
                p_c = flat_p[offset : offset + cur]
                g_c = flat_g[offset : offset + cur]
                m_c = flat_m[offset : offset + cur]
                v_c = flat_v[offset : offset + cur]
                ref_c = flat_ref[offset : offset + cur]

                g_clean = torch.nan_to_num(g_c, nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                m_f32 = m_c.float()
                v_f32 = v_c.float()

                m_new = m_f32.mul(beta1).add(g_clean, alpha=1.0 - beta1)
                clm_m = m_new.mul(beta1).add(g_clean, alpha=1.0 - beta1)
                diff = (g_clean - m_new).clamp(-1e18, 1e18)
                if use_cd:
                    diff = (diff - diff_mean_scaled).clamp(-1e18, 1e18)
                v_new = v_f32.mul(beta2).addcmul(diff, diff, value=1.0 - beta2)
                if p.dtype == torch.float16 or m_c.dtype == torch.float16 or v_c.dtype == torch.float16:
                    v_new = v_new.clamp(0.0, 65000.0)

                sigma = v_new.clamp_min(0.0).sqrt().mul(c1).add(c2)
                u = torch.tanh(clm_m / sigma.clamp_min(1e-12))

                if m_c.dtype in (torch.bfloat16, torch.float16):
                    _sr_quantize(m_c, m_new, use_sr=use_sr, generator=gen_m)
                else:
                    m_c.copy_(m_new.to(dtype=m_c.dtype))
                v_c.copy_(v_new.to(dtype=v_c.dtype))

                if cautious:
                    m_mask = (u * g_clean > 0.0).float()
                    delta_theta = u * m_mask * cautious_scale
                    mask_sum_acc += m_mask.sum().item()
                else:
                    m_mask = None
                    delta_theta = u

                step_val = lr * delta_theta

                if bound and bound_ratio > 0.0:
                    if bound_type == "radial":
                        step_val = step_val * bound_scale
                    elif bound_type == "coordinate":
                        B_i = bound_ratio * torch.sqrt(ref_c.float().square() + 1e-4)
                        step_val = B_i * torch.tanh(step_val / B_i)

                if p.dtype == torch.float16:
                    b_safe = torch.clamp_min(torch.abs(ref_c.float()) * 0.01, 1.5e-4)
                    step_val = b_safe * torch.tanh(step_val / b_safe)

                if bound and bound_type == "radial" and bound_ratio > 0.0:
                    u_eff = (u * m_mask) if (cautious and m_mask is not None) else u
                    u_sq_sum_acc += torch.sum(u_eff ** 2).item()
                    p_sq_sum_acc += torch.sum(ref_c.float() ** 2).item()

                if has_master:
                    if weight_decay != 0.0:
                        ref_c.mul_(1.0 - lr * weight_decay)
                    ref_c.add_(step_val, alpha=-1.0)
                    if p_c.dtype == torch.float16:
                        p_c.copy_(ref_c.clamp(-65504.0, 65504.0).to(dtype=p_c.dtype))
                    elif p_c.dtype == torch.bfloat16:
                        p_c.copy_(ref_c.clamp(-3.3895313892515355e38, 3.3895313892515355e38).to(dtype=p_c.dtype))
                    else:
                        p_c.copy_(ref_c.to(dtype=p_c.dtype))
                else:
                    p_fp32 = p_c.float()
                    if weight_decay != 0.0:
                        p_fp32.mul_(1.0 - lr * weight_decay)
                    p_fp32.add_(step_val, alpha=-1.0)
                    _sr_quantize(p_c, p_fp32, use_sr=use_sr, generator=gen_p)

            if sc is not None:
                if needs_global_reduction:
                    if not (fsdp_sync and scratch_synced):
                        sc[:3].zero_()
                        if cautious:
                            sc[0] = mask_sum_acc
                        if bound and bound_type == "radial" and bound_ratio > 0.0:
                            sc[1] = u_sq_sum_acc
                            sc[2] = p_sq_sum_acc

                    if not defer_fsdp_post_ema:
                        total_elements = float(n_elem)
                        if fsdp_sync:
                            if not scratch_synced:
                                total_elements = _sync_fsdp_scratch(sc, n_elem)
                            else:
                                total_elements = scratch_total_elements if scratch_total_elements > 0.0 else float(n_elem)
                            mask_sum_acc = sc[0].item()
                            u_sq_sum_acc = sc[1].item()
                            p_sq_sum_acc = sc[2].item()

                        if cautious:
                            m_bar = max(min(mask_sum_acc / total_elements, 1.0), clamp_min)
                            target_cautious = 1.0 / m_bar
                            new_cautious = ema_decay * cautious_scale + (1.0 - ema_decay) * target_cautious
                            sc[3] = new_cautious

                        if bound and bound_type == "radial" and bound_ratio > 0.0:
                            c_eff = sc[3].item() if cautious else 1.0
                            norm_delta = math.sqrt(max(u_sq_sum_acc, 0.0)) * c_eff
                            norm_p = math.sqrt(max(p_sq_sum_acc, 0.0))
                            R = bound_ratio * max(norm_p, 1.0)
                            r = (lr * norm_delta) / R
                            target_bound = 1.0 if r < 1e-4 else math.tanh(r) / max(r, 1e-8)
                            new_bound = ema_decay * bound_scale + (1.0 - ema_decay) * target_bound
                            sc[4] = new_bound

                        sc[:3].zero_()
                        sc[5].zero_()
                else:
                    sc[:3].zero_()
                    sc[5].zero_()
        else:
            if fsdp_sync and scratch_synced and sc is not None:
                mask_sum_acc = sc[0].item()
                u_sq_sum_acc = sc[1].item()
                p_sq_sum_acc = sc[2].item()
                total_elements = scratch_total_elements if scratch_total_elements > 0.0 else float(n_elem)
            else:
                for offset in range(0, n_elem, chunk_size):
                    cur = min(chunk_size, n_elem - offset)
                    g_c = flat_g[offset : offset + cur]
                    m_c = flat_m[offset : offset + cur]
                    v_c = flat_v[offset : offset + cur]
                    ref_c = flat_ref[offset : offset + cur]

                    g_clean = torch.nan_to_num(g_c, nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                    m_f32 = m_c.float()
                    v_f32 = v_c.float()

                    m_new = m_f32.mul(beta1).add(g_clean, alpha=1.0 - beta1)
                    clm_m = m_new.mul(beta1).add(g_clean, alpha=1.0 - beta1)
                    diff = (g_clean - m_new).clamp(-1e18, 1e18)
                    if use_cd:
                        diff = (diff - diff_mean_scaled).clamp(-1e18, 1e18)
                    v_new = v_f32.mul(beta2).addcmul(diff, diff, value=1.0 - beta2)
                    if p.dtype == torch.float16 or m_c.dtype == torch.float16 or v_c.dtype == torch.float16:
                        v_new = v_new.clamp(0.0, 65000.0)

                    sigma = v_new.clamp_min(0.0).sqrt().mul(c1).add(c2)
                    u = torch.tanh(clm_m / sigma.clamp_min(1e-12))

                    if cautious:
                        m_mask = (u * g_clean > 0.0).float()
                        mask_sum_acc += m_mask.sum().item()
                        u_eff = u * m_mask
                    else:
                        u_eff = u

                    if bound and bound_type == "radial" and bound_ratio > 0.0:
                        u_sq_sum_acc += torch.sum(u_eff ** 2).item()
                        p_sq_sum_acc += torch.sum(ref_c.float() ** 2).item()

                total_elements = float(n_elem)
                if sc is not None:
                    sc[:3].zero_()
                    if cautious:
                        sc[0] = mask_sum_acc
                    if bound and bound_type == "radial" and bound_ratio > 0.0:
                        sc[1] = u_sq_sum_acc
                        sc[2] = p_sq_sum_acc

                    if fsdp_sync and not scratch_synced:
                        total_elements = _sync_fsdp_scratch(sc, n_elem)
                        mask_sum_acc = sc[0].item()
                        u_sq_sum_acc = sc[1].item()
                        p_sq_sum_acc = sc[2].item()

            cautious_scale = 1.0
            if cautious:
                m_bar = max(min(mask_sum_acc / total_elements, 1.0), clamp_min)
                cautious_scale = 1.0 / m_bar
                if sc is not None:
                    sc[3] = cautious_scale

            bound_scale = 1.0
            if bound and bound_type == "radial" and bound_ratio > 0.0:
                norm_p = math.sqrt(max(p_sq_sum_acc, 0.0))
                norm_u = math.sqrt(max(u_sq_sum_acc, 0.0)) * cautious_scale
                R = bound_ratio * max(norm_p, 1.0)
                r = (lr * norm_u) / R
                bound_scale = 1.0 if r < 1e-4 else math.tanh(r) / max(r, 1e-8)
                if sc is not None:
                    sc[4] = bound_scale

            for offset in range(0, n_elem, chunk_size):
                cur = min(chunk_size, n_elem - offset)
                p_c = flat_p[offset : offset + cur]
                g_c = flat_g[offset : offset + cur]
                m_c = flat_m[offset : offset + cur]
                v_c = flat_v[offset : offset + cur]
                ref_c = flat_ref[offset : offset + cur]

                g_clean = torch.nan_to_num(g_c, nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                m_f32 = m_c.float()
                v_f32 = v_c.float()

                m_new = m_f32.mul(beta1).add(g_clean, alpha=1.0 - beta1)
                clm_m = m_new.mul(beta1).add(g_clean, alpha=1.0 - beta1)
                diff = (g_clean - m_new).clamp(-1e18, 1e18)
                if use_cd:
                    diff = (diff - diff_mean_scaled).clamp(-1e18, 1e18)
                v_new = v_f32.mul(beta2).addcmul(diff, diff, value=1.0 - beta2)
                if p.dtype == torch.float16 or m_c.dtype == torch.float16 or v_c.dtype == torch.float16:
                    v_new = v_new.clamp(0.0, 65000.0)

                sigma = v_new.clamp_min(0.0).sqrt().mul(c1).add(c2)
                u = torch.tanh(clm_m / sigma.clamp_min(1e-12))

                if m_c.dtype in (torch.bfloat16, torch.float16):
                    _sr_quantize(m_c, m_new, use_sr=use_sr, generator=gen_m)
                else:
                    m_c.copy_(m_new.to(dtype=m_c.dtype))
                v_c.copy_(v_new.to(dtype=v_c.dtype))

                delta_theta = (u * (u * g_clean > 0.0).float() * cautious_scale) if cautious else u
                step_val = lr * delta_theta

                if bound and bound_ratio > 0.0:
                    if bound_type == "radial":
                        step_val = step_val * bound_scale
                    elif bound_type == "coordinate":
                        B_i = bound_ratio * torch.sqrt(ref_c.float().square() + 1e-4)
                        step_val = B_i * torch.tanh(step_val / B_i)

                if p.dtype == torch.float16:
                    b_safe = torch.clamp_min(torch.abs(ref_c.float()) * 0.01, 1.5e-4)
                    step_val = b_safe * torch.tanh(step_val / b_safe)

                if has_master:
                    if weight_decay != 0.0:
                        ref_c.mul_(1.0 - lr * weight_decay)
                    ref_c.add_(step_val, alpha=-1.0)
                    if p_c.dtype == torch.float16:
                        p_c.copy_(ref_c.clamp(-65504.0, 65504.0).to(dtype=p_c.dtype))
                    elif p_c.dtype == torch.bfloat16:
                        p_c.copy_(ref_c.clamp(-3.3895313892515355e38, 3.3895313892515355e38).to(dtype=p_c.dtype))
                    else:
                        p_c.copy_(ref_c.to(dtype=p_c.dtype))
                else:
                    p_fp32 = p_c.float()
                    if weight_decay != 0.0:
                        p_fp32.mul_(1.0 - lr * weight_decay)
                    p_fp32.add_(step_val, alpha=-1.0)
                    _sr_quantize(p_c, p_fp32, use_sr=use_sr, generator=gen_p)

            if sc is not None:
                sc[:3].zero_()
                sc[5].zero_()

        if not p.is_contiguous():
            p.copy_(p_work)
        if not exp_avg.is_contiguous():
            exp_avg.copy_(m_work)
        if not exp_avg_sq.is_contiguous():
            exp_avg_sq.copy_(v_work)
        if has_master and ref_p is not None and not ref_p.is_contiguous():
            ref_p.copy_(ref_work)

    def _single_step_tdclm(
        self, params, grads, exp_avgs, master_params, steps, scratches,
        lr, beta1, eps, weight_decay, lam, cautious, clamp_min,
        use_sr, bound=True, bound_type="coordinate", bound_ratio=0.05,
        fused_single_pass: bool = False, ema_decay: float = 0.8,
        fsdp_sync: bool = False, seed: int = 1337, centered_defect: bool = False,
        diff_synced: bool = False, scratch_synced: bool = False,
        diff_total_elements: float = 0.0, scratch_total_elements: float = 0.0,
        orig_indices: Optional[List[int]] = None,
        defer_fsdp_post_ema: bool = False,
    ):
        bound_mode = 0
        if bound and bound_ratio > 0.0:
            bound_mode = 1 if bound_type == "radial" else 2

        ELEM_CHUNK_SIZE = 2_097_152

        batched_diff_totals: Optional[List[float]] = None
        if centered_defect and fsdp_sync and not diff_synced and scratches is not None and len(params) > 1:
            batched_diff_totals = self._precompute_fsdp_diff_single(
                params, grads, exp_avgs, scratches, ELEM_CHUNK_SIZE
            )

        needs_ema_sync = cautious or bound_mode == 1
        batch_post_ema = bool(
            fsdp_sync and not scratch_synced and not defer_fsdp_post_ema
            and scratches is not None and len(params) > 1 and needs_ema_sync
        )
        batched_scratch_totals: Dict[int, float] = {}
        if batch_post_ema:
            pass1_indices = [
                idx for idx, st in enumerate(steps)
                if not (fused_single_pass and int(st.item() if isinstance(st, torch.Tensor) else st) > 1)
            ]
            if len(pass1_indices) > 1:
                batched_scratch_totals = self._precompute_fsdp_pass1_single_tdclm(
                    pass1_indices, params, grads, exp_avgs, master_params, scratches,
                    beta1, eps, lam, bound_mode, cautious, centered_defect, batched_diff_totals, ELEM_CHUNK_SIZE
                )
        pending_ema_single: List[Tuple[Tensor, int]] = []

        for i in range(len(params)):
            p, grad, exp_avg, step_raw = params[i], grads[i], exp_avgs[i], steps[i]
            step = int(step_raw.item() if isinstance(step_raw, torch.Tensor) else step_raw)
            global_param_idx = orig_indices[i] if orig_indices is not None else i
            param_has_master = master_params is not None and (master_params[i] is not p)
            ref_p = master_params[i] if param_has_master else p
            sc = scratches[i] if scratches is not None else None
            n_elem = p.numel()
            cur_diff_synced = diff_synced or (batched_diff_totals is not None)
            cur_diff_total = batched_diff_totals[i] if batched_diff_totals is not None else diff_total_elements
            cur_scratch_synced = scratch_synced or (i in batched_scratch_totals)
            cur_scratch_total = batched_scratch_totals[i] if (i in batched_scratch_totals) else scratch_total_elements
            defer_cur_ema = defer_fsdp_post_ema or (batch_post_ema and fused_single_pass and step > 1)

            if n_elem == 0:
                if fsdp_sync and sc is not None:
                    if not cur_scratch_synced:
                        sc[:3].zero_()
                    if not cur_diff_synced:
                        sc[5].zero_()
                        if centered_defect:
                            _sync_fsdp_diff_only(sc, 0)
                    if defer_cur_ema and needs_ema_sync:
                        if batch_post_ema:
                            pending_ema_single.append((sc, 0))
                    else:
                        _sync_empty_fsdp_shard(
                            sc, step, lr, clamp_min, ema_decay,
                            bound_mode, bound_ratio, cautious, centered_defect, fused_single_pass,
                            diff_synced=True, scratch_synced=cur_scratch_synced,
                            synced_total_elements=cur_scratch_total,
                        )
                continue

            if n_elem > ELEM_CHUNK_SIZE:
                self._single_step_tdclm_element_chunked(
                    p, grad, exp_avg, ref_p, step, sc,
                    lr, beta1, eps, weight_decay, lam, cautious, clamp_min,
                    use_sr, bound, bound_type, bound_ratio,
                    fused_single_pass, ema_decay, fsdp_sync, seed, ELEM_CHUNK_SIZE,
                    centered_defect,
                    diff_synced=cur_diff_synced, scratch_synced=cur_scratch_synced,
                    diff_total_elements=cur_diff_total,
                    scratch_total_elements=cur_scratch_total,
                    param_index=global_param_idx,
                    defer_fsdp_post_ema=defer_cur_ema,
                )
                if batch_post_ema and defer_cur_ema and sc is not None:
                    pending_ema_single.append((sc, n_elem))
                continue

            need_sr_m = use_sr and (exp_avg.dtype in (torch.bfloat16, torch.float16))
            need_sr_p = use_sr and (not param_has_master) and (p.dtype in (torch.bfloat16, torch.float16))
            gen_m, gen_p = self._get_step_generators(
                p.device, seed, step, global_param_idx, need_sr_m, need_sr_p
            )

            grad_clean = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            g_f32 = grad_clean.float().clamp(-1e18, 1e18)
            m_f32 = exp_avg.float()

            beta1_sq = beta1 * beta1
            one_minus_beta1_sq = 1.0 - beta1_sq
            effective_eps = max(eps, 1e-4) if p.dtype == torch.float16 else eps

            clm_m = m_f32.mul(beta1_sq).add(g_f32, alpha=one_minus_beta1_sq)

            diff_f32 = (g_f32 - m_f32).clamp(-1e18, 1e18)
            if centered_defect:
                if fsdp_sync and sc is not None:
                    if not cur_diff_synced:
                        sc[5].copy_(diff_f32.sum().to(torch.float32))
                        tot_cd = _sync_fsdp_diff_only(sc, n_elem)
                    else:
                        tot_cd = cur_diff_total if cur_diff_total > 0.0 else float(n_elem)
                    if tot_cd > 1.0:
                        diff_f32 = (diff_f32 - (sc[5] / tot_cd)).clamp(-1e18, 1e18)
                    sc[5].zero_()
                elif diff_f32.numel() > 1:
                    diff_f32 = (diff_f32 - diff_f32.mean()).clamp(-1e18, 1e18)

            clm_defect = torch.abs(diff_f32).mul(lam)
            clm_env = (
                torch.abs(m_f32)
                .mul(beta1_sq)
                .add(torch.abs(g_f32), alpha=one_minus_beta1_sq)
                .add_(clm_defect)
                .add_(effective_eps)
            )
            u = clm_m / clm_env
            del clm_m, clm_env, clm_defect, diff_f32

            m_new_f32 = m_f32.mul(beta1).add(g_f32, alpha=1.0 - beta1)
            if exp_avg.dtype in (torch.bfloat16, torch.float16):
                _sr_quantize(exp_avg, m_new_f32, use_sr=use_sr, generator=gen_m)
            else:
                exp_avg.copy_(m_new_f32.to(dtype=exp_avg.dtype))
            del m_new_f32, m_f32

            if fused_single_pass and sc is not None and step > 1:
                cautious_scale = sc[3].clone()
                bound_scale = sc[4].clone()

                if cautious:
                    m_mask = (u * g_f32 > 0).float()
                    delta_theta = u * m_mask * cautious_scale
                else:
                    m_mask = None
                    delta_theta = u

                step_val = _compute_step_val(
                    delta_theta, ref_p, lr, bound, bound_type, bound_ratio,
                    target_dtype=p.dtype,
                    radial_scale=bound_scale if bound_mode == 1 else None,
                )

                if not (fsdp_sync and cur_scratch_synced):
                    sc[:3].zero_()
                    if cautious and m_mask is not None:
                        sc[0].copy_(m_mask.sum().to(torch.float32))
                    if bound_mode == 1:
                        u_eff = (u * m_mask) if (cautious and m_mask is not None) else u
                        sc[1].copy_(torch.sum(u_eff.float() ** 2))
                        sc[2].copy_(torch.sum(ref_p.float() ** 2))

                if defer_cur_ema:
                    if batch_post_ema:
                        pending_ema_single.append((sc, n_elem))
                else:
                    total_elements = float(n_elem)
                    if fsdp_sync and needs_ema_sync:
                        if not cur_scratch_synced:
                            total_elements = _sync_fsdp_scratch(sc, n_elem)
                        else:
                            total_elements = cur_scratch_total if cur_scratch_total > 0.0 else float(n_elem)

                    if cautious and m_mask is not None:
                        m_bar = (sc[0] / total_elements).clamp_(min=clamp_min, max=1.0)
                        target_cautious = 1.0 / m_bar
                        new_cautious = ema_decay * cautious_scale + (1.0 - ema_decay) * target_cautious
                        sc[3].copy_(new_cautious)

                    if bound_mode == 1:
                        c_eff = sc[3] if cautious else 1.0
                        norm_delta = torch.sqrt(torch.clamp_min(sc[1], 0.0)) * c_eff
                        norm_p = torch.sqrt(torch.clamp_min(sc[2], 0.0))
                        R = bound_ratio * torch.clamp_min(norm_p, 1.0)
                        r = (lr * norm_delta) / R
                        target_bound = torch.where(r < 1e-4, torch.ones_like(r), torch.tanh(r) / torch.clamp_min(r, 1e-8))
                        new_bound = ema_decay * bound_scale + (1.0 - ema_decay) * target_bound
                        sc[4].copy_(new_bound)

                    sc[:3].zero_()
                    sc[5].zero_()
            else:
                mask: Optional[Tensor] = None
                if sc is not None and not (fsdp_sync and cur_scratch_synced):
                    sc[:3].zero_()

                if cautious:
                    mask = (u * g_f32 > 0).float()
                    if sc is not None and not (fsdp_sync and cur_scratch_synced):
                        sc[0].copy_(mask.sum().to(torch.float32))

                u_sq_sum: Optional[Tensor] = None
                p_sq_sum: Optional[Tensor] = None
                if bound_mode == 1 and not (fsdp_sync and cur_scratch_synced and sc is not None):
                    u_eff = (u * mask) if (cautious and mask is not None) else u
                    u_sq_sum = torch.sum(u_eff.float() ** 2)
                    p_sq_sum = torch.sum(ref_p.float() ** 2)
                    if sc is not None:
                        sc[1].copy_(u_sq_sum)
                        sc[2].copy_(p_sq_sum)

                total_elements = float(n_elem)
                if fsdp_sync and sc is not None and needs_ema_sync:
                    if not cur_scratch_synced:
                        total_elements = _sync_fsdp_scratch(sc, n_elem)
                    else:
                        total_elements = cur_scratch_total if cur_scratch_total > 0.0 else float(n_elem)

                cautious_scale_val: Optional[Tensor] = None
                if cautious and mask is not None:
                    if fsdp_sync and sc is not None:
                        m_bar = (sc[0] / total_elements).clamp_(min=clamp_min, max=1.0)
                    else:
                        m_bar = mask.mean().clamp_(min=clamp_min, max=1.0)
                    cautious_scale_val = 1.0 / m_bar
                    update = u.mul(mask).mul(cautious_scale_val)
                    if sc is not None:
                        sc[3].copy_(cautious_scale_val.to(torch.float32))
                else:
                    update = u

                b_scale: Optional[Tensor] = None
                if bound_mode == 1:
                    if sc is not None:
                        c_scale = sc[3] if cautious else 1.0
                        norm_u = torch.sqrt(torch.clamp_min(sc[1], 0.0)) * c_scale
                        norm_p = torch.sqrt(torch.clamp_min(sc[2], 0.0))
                    else:
                        c_scale_f = cautious_scale_val.float() if (cautious and cautious_scale_val is not None) else 1.0
                        norm_u = torch.sqrt(torch.clamp_min(u_sq_sum, 0.0)) * c_scale_f
                        norm_p = torch.sqrt(torch.clamp_min(p_sq_sum, 0.0))
                    R = bound_ratio * torch.clamp_min(norm_p, 1.0)
                    r = (lr * norm_u) / R
                    b_scale = torch.where(r < 1e-4, torch.ones_like(r), torch.tanh(r) / torch.clamp_min(r, 1e-8))
                    if sc is not None:
                        sc[4].copy_(b_scale.to(torch.float32))

                step_val = _compute_step_val(
                    update, ref_p, lr, bound, bound_type, bound_ratio,
                    target_dtype=p.dtype, radial_scale=b_scale,
                )
                if sc is not None:
                    sc[:3].zero_()
                    sc[5].zero_()

            if param_has_master:
                mp = master_params[i]
                if weight_decay != 0.0:
                    mp.mul_(1.0 - lr * weight_decay)
                mp.add_(step_val, alpha=-1.0)
                if p.dtype == torch.float16:
                    p.copy_(mp.clamp(-65504.0, 65504.0).to(dtype=p.dtype))
                elif p.dtype == torch.bfloat16:
                    p.copy_(mp.clamp(-3.3895313892515355e38, 3.3895313892515355e38).to(dtype=p.dtype))
                else:
                    p.copy_(mp.to(dtype=p.dtype))
            else:
                _sr_apply_step_val(p, step_val, lr, weight_decay, use_sr, generator=gen_p)
            del step_val, grad_clean, g_f32

        if pending_ema_single:
            sc_list = [item[0] for item in pending_ema_single]
            numels = [item[1] for item in pending_ema_single]
            totals = _sync_fsdp_scratch_batch(sc_list, numels)
            for (sc_p, n_el_p), tot_el in zip(pending_ema_single, totals):
                _fallback_post_step_ema(
                    sc_p, n_el_p, lr, clamp_min, ema_decay,
                    bound_mode, bound_ratio, cautious, True, True,
                    synced_total_elements=tot_el
                )

    def _single_step_tdclm_element_chunked(
        self, p, grad, exp_avg, ref_p, step, sc,
        lr, beta1, eps, weight_decay, lam, cautious, clamp_min,
        use_sr, bound, bound_type, bound_ratio,
        fused_single_pass, ema_decay, fsdp_sync, seed, chunk_size,
        centered_defect: bool = False,
        diff_synced: bool = False, scratch_synced: bool = False,
        diff_total_elements: float = 0.0, scratch_total_elements: float = 0.0,
        param_index: int = 0,
        defer_fsdp_post_ema: bool = False,
    ):
        n_elem = p.numel()
        has_master = ref_p is not None and ref_p is not p
        p_work = p if p.is_contiguous() else p.contiguous()
        g_work = grad if grad.is_contiguous() else grad.contiguous()
        m_work = exp_avg if exp_avg.is_contiguous() else exp_avg.contiguous()
        if has_master and ref_p is not None:
            ref_work = ref_p if ref_p.is_contiguous() else ref_p.contiguous()
        else:
            ref_work = p_work

        flat_p = p_work.view(-1)
        flat_g = g_work.view(-1)
        flat_m = m_work.view(-1)
        flat_ref = ref_work.view(-1)

        need_sr_m = use_sr and (exp_avg.dtype in (torch.bfloat16, torch.float16))
        need_sr_p = use_sr and (not has_master) and (p.dtype in (torch.bfloat16, torch.float16))
        gen_m, gen_p = self._get_step_generators(
            p.device, seed, step, param_index, need_sr_m, need_sr_p
        )

        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq
        effective_eps = max(eps, 1e-4) if p.dtype == torch.float16 else eps

        diff_mean = 0.0
        use_cd = False
        if centered_defect and (n_elem > 1 or fsdp_sync):
            if fsdp_sync and diff_synced and sc is not None:
                diff_sum_acc = sc[5].item()
                tot_cd = diff_total_elements if diff_total_elements > 0.0 else float(n_elem)
            else:
                diff_sum_acc = 0.0
                for offset in range(0, n_elem, chunk_size):
                    cur = min(chunk_size, n_elem - offset)
                    g_c = torch.nan_to_num(flat_g[offset : offset + cur], nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                    m_c = flat_m[offset : offset + cur].float()
                    diff_sum_acc += (g_c - m_c).clamp(-1e18, 1e18).sum().item()
                tot_cd = float(n_elem)
                if fsdp_sync and not diff_synced and sc is not None:
                    sc[5] = diff_sum_acc
                    tot_cd = _sync_fsdp_diff_only(sc, n_elem)
                    diff_sum_acc = sc[5].item()
            if tot_cd > 1.0:
                diff_mean = diff_sum_acc / tot_cd
                use_cd = True

        mask_sum_acc = 0.0
        u_sq_sum_acc = 0.0
        p_sq_sum_acc = 0.0
        needs_global_reduction = cautious or (bound and bound_type == "radial" and bound_ratio > 0.0)

        if (fused_single_pass and sc is not None and step > 1) or not needs_global_reduction:
            cautious_scale = sc[3].item() if (cautious and sc is not None and step > 1) else 1.0
            bound_scale = sc[4].item() if (bound and bound_type == "radial" and bound_ratio > 0.0 and sc is not None and step > 1) else 1.0

            for offset in range(0, n_elem, chunk_size):
                cur = min(chunk_size, n_elem - offset)
                p_c = flat_p[offset : offset + cur]
                g_c = flat_g[offset : offset + cur]
                m_c = flat_m[offset : offset + cur]
                ref_c = flat_ref[offset : offset + cur]

                g_clean = torch.nan_to_num(g_c, nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                m_f32 = m_c.float()

                clm_m = m_f32.mul(beta1_sq).add(g_clean, alpha=one_minus_beta1_sq)
                diff_f32 = (g_clean - m_f32).clamp(-1e18, 1e18)
                if use_cd:
                    diff_f32 = (diff_f32 - diff_mean).clamp(-1e18, 1e18)
                clm_defect = torch.abs(diff_f32).mul(lam)
                clm_env = torch.abs(m_f32).mul(beta1_sq).add(torch.abs(g_clean), alpha=one_minus_beta1_sq).add_(clm_defect).add_(effective_eps)
                u = clm_m / clm_env

                m_new = m_f32.mul(beta1).add(g_clean, alpha=1.0 - beta1)
                if m_c.dtype in (torch.bfloat16, torch.float16):
                    _sr_quantize(m_c, m_new, use_sr=use_sr, generator=gen_m)
                else:
                    m_c.copy_(m_new.to(dtype=m_c.dtype))

                if cautious:
                    m_mask = (u * g_clean > 0.0).float()
                    delta_theta = u * m_mask * cautious_scale
                    mask_sum_acc += m_mask.sum().item()
                else:
                    m_mask = None
                    delta_theta = u

                step_val = lr * delta_theta

                if bound and bound_ratio > 0.0:
                    if bound_type == "radial":
                        step_val = step_val * bound_scale
                    elif bound_type == "coordinate":
                        B_i = bound_ratio * torch.sqrt(ref_c.float().square() + 1e-4)
                        step_val = B_i * torch.tanh(step_val / B_i)

                if p.dtype == torch.float16:
                    b_safe = torch.clamp_min(torch.abs(ref_c.float()) * 0.01, 1.5e-4)
                    step_val = b_safe * torch.tanh(step_val / b_safe)

                if bound and bound_type == "radial" and bound_ratio > 0.0:
                    u_eff = (u * m_mask) if (cautious and m_mask is not None) else u
                    u_sq_sum_acc += torch.sum(u_eff ** 2).item()
                    p_sq_sum_acc += torch.sum(ref_c.float() ** 2).item()

                if has_master:
                    if weight_decay != 0.0:
                        ref_c.mul_(1.0 - lr * weight_decay)
                    ref_c.add_(step_val, alpha=-1.0)
                    if p_c.dtype == torch.float16:
                        p_c.copy_(ref_c.clamp(-65504.0, 65504.0).to(dtype=p_c.dtype))
                    elif p_c.dtype == torch.bfloat16:
                        p_c.copy_(ref_c.clamp(-3.3895313892515355e38, 3.3895313892515355e38).to(dtype=p_c.dtype))
                    else:
                        p_c.copy_(ref_c.to(dtype=p_c.dtype))
                else:
                    p_fp32 = p_c.float()
                    if weight_decay != 0.0:
                        p_fp32.mul_(1.0 - lr * weight_decay)
                    p_fp32.add_(step_val, alpha=-1.0)
                    _sr_quantize(p_c, p_fp32, use_sr=use_sr, generator=gen_p)

            if sc is not None:
                if needs_global_reduction:
                    if not (fsdp_sync and scratch_synced):
                        sc[:3].zero_()
                        if cautious:
                            sc[0] = mask_sum_acc
                        if bound and bound_type == "radial" and bound_ratio > 0.0:
                            sc[1] = u_sq_sum_acc
                            sc[2] = p_sq_sum_acc

                    if not defer_fsdp_post_ema:
                        total_elements = float(n_elem)
                        if fsdp_sync:
                            if not scratch_synced:
                                total_elements = _sync_fsdp_scratch(sc, n_elem)
                            else:
                                total_elements = scratch_total_elements if scratch_total_elements > 0.0 else float(n_elem)
                            mask_sum_acc = sc[0].item()
                            u_sq_sum_acc = sc[1].item()
                            p_sq_sum_acc = sc[2].item()

                        if cautious:
                            m_bar = max(min(mask_sum_acc / total_elements, 1.0), clamp_min)
                            target_cautious = 1.0 / m_bar
                            new_cautious = ema_decay * cautious_scale + (1.0 - ema_decay) * target_cautious
                            sc[3] = new_cautious

                        if bound and bound_type == "radial" and bound_ratio > 0.0:
                            c_eff = sc[3].item() if cautious else 1.0
                            norm_delta = math.sqrt(max(u_sq_sum_acc, 0.0)) * c_eff
                            norm_p = math.sqrt(max(p_sq_sum_acc, 0.0))
                            R = bound_ratio * max(norm_p, 1.0)
                            r = (lr * norm_delta) / R
                            target_bound = 1.0 if r < 1e-4 else math.tanh(r) / max(r, 1e-8)
                            new_bound = ema_decay * bound_scale + (1.0 - ema_decay) * target_bound
                            sc[4] = new_bound

                        sc[:3].zero_()
                        sc[5].zero_()
                else:
                    sc[:3].zero_()
                    sc[5].zero_()
        else:
            if fsdp_sync and scratch_synced and sc is not None:
                mask_sum_acc = sc[0].item()
                u_sq_sum_acc = sc[1].item()
                p_sq_sum_acc = sc[2].item()
                total_elements = scratch_total_elements if scratch_total_elements > 0.0 else float(n_elem)
            else:
                for offset in range(0, n_elem, chunk_size):
                    cur = min(chunk_size, n_elem - offset)
                    g_c = flat_g[offset : offset + cur]
                    m_c = flat_m[offset : offset + cur]
                    ref_c = flat_ref[offset : offset + cur]

                    g_clean = torch.nan_to_num(g_c, nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                    m_f32 = m_c.float()

                    clm_m = m_f32.mul(beta1_sq).add(g_clean, alpha=one_minus_beta1_sq)
                    diff_f32 = (g_clean - m_f32).clamp(-1e18, 1e18)
                    if use_cd:
                        diff_f32 = (diff_f32 - diff_mean).clamp(-1e18, 1e18)
                    clm_defect = torch.abs(diff_f32).mul(lam)
                    clm_env = (
                        torch.abs(m_f32)
                        .mul(beta1_sq)
                        .add(torch.abs(g_clean), alpha=one_minus_beta1_sq)
                        .add_(clm_defect)
                        .add_(effective_eps)
                    )
                    u = clm_m / clm_env

                    if cautious:
                        m_mask = (u * g_clean > 0.0).float()
                        mask_sum_acc += m_mask.sum().item()
                        u_eff = u * m_mask
                    else:
                        u_eff = u

                    if bound and bound_type == "radial" and bound_ratio > 0.0:
                        u_sq_sum_acc += torch.sum(u_eff ** 2).item()
                        p_sq_sum_acc += torch.sum(ref_c.float() ** 2).item()

                total_elements = float(n_elem)
                if sc is not None:
                    sc[:3].zero_()
                    if cautious:
                        sc[0] = mask_sum_acc
                    if bound and bound_type == "radial" and bound_ratio > 0.0:
                        sc[1] = u_sq_sum_acc
                        sc[2] = p_sq_sum_acc

                    if fsdp_sync and not scratch_synced:
                        total_elements = _sync_fsdp_scratch(sc, n_elem)
                        mask_sum_acc = sc[0].item()
                        u_sq_sum_acc = sc[1].item()
                        p_sq_sum_acc = sc[2].item()

            cautious_scale = 1.0
            if cautious:
                m_bar = max(min(mask_sum_acc / total_elements, 1.0), clamp_min)
                cautious_scale = 1.0 / m_bar
                if sc is not None:
                    sc[3] = cautious_scale

            bound_scale = 1.0
            if bound and bound_type == "radial" and bound_ratio > 0.0:
                norm_p = math.sqrt(max(p_sq_sum_acc, 0.0))
                norm_u = math.sqrt(max(u_sq_sum_acc, 0.0)) * cautious_scale
                R = bound_ratio * max(norm_p, 1.0)
                r = (lr * norm_u) / R
                bound_scale = 1.0 if r < 1e-4 else math.tanh(r) / max(r, 1e-8)
                if sc is not None:
                    sc[4] = bound_scale

            for offset in range(0, n_elem, chunk_size):
                cur = min(chunk_size, n_elem - offset)
                p_c = flat_p[offset : offset + cur]
                g_c = flat_g[offset : offset + cur]
                m_c = flat_m[offset : offset + cur]
                ref_c = flat_ref[offset : offset + cur]

                g_clean = torch.nan_to_num(g_c, nan=0.0, posinf=0.0, neginf=0.0).float().clamp(-1e18, 1e18)
                m_f32 = m_c.float()

                clm_m = m_f32.mul(beta1_sq).add(g_clean, alpha=one_minus_beta1_sq)
                diff_f32 = (g_clean - m_f32).clamp(-1e18, 1e18)
                if use_cd:
                    diff_f32 = (diff_f32 - diff_mean).clamp(-1e18, 1e18)
                clm_defect = torch.abs(diff_f32).mul(lam)
                clm_env = (
                    torch.abs(m_f32)
                    .mul(beta1_sq)
                    .add(torch.abs(g_clean), alpha=one_minus_beta1_sq)
                    .add_(clm_defect)
                    .add_(effective_eps)
                )
                u = clm_m / clm_env

                m_new = m_f32.mul(beta1).add(g_clean, alpha=1.0 - beta1)
                if m_c.dtype in (torch.bfloat16, torch.float16):
                    _sr_quantize(m_c, m_new, use_sr=use_sr, generator=gen_m)
                else:
                    m_c.copy_(m_new.to(dtype=m_c.dtype))

                if cautious:
                    m_mask = (u * g_clean > 0.0).float()
                    delta_theta = u * m_mask * cautious_scale
                else:
                    delta_theta = u

                step_val = lr * delta_theta

                if bound and bound_ratio > 0.0:
                    if bound_type == "radial":
                        step_val = step_val * bound_scale
                    elif bound_type == "coordinate":
                        B_i = bound_ratio * torch.sqrt(ref_c.float().square() + 1e-4)
                        step_val = B_i * torch.tanh(step_val / B_i)

                if p.dtype == torch.float16:
                    b_safe = torch.clamp_min(torch.abs(ref_c.float()) * 0.01, 1.5e-4)
                    step_val = b_safe * torch.tanh(step_val / b_safe)

                if has_master:
                    if weight_decay != 0.0:
                        ref_c.mul_(1.0 - lr * weight_decay)
                    ref_c.add_(step_val, alpha=-1.0)
                    if p_c.dtype == torch.float16:
                        p_c.copy_(ref_c.clamp(-65504.0, 65504.0).to(dtype=p_c.dtype))
                    elif p_c.dtype == torch.bfloat16:
                        p_c.copy_(ref_c.clamp(-3.3895313892515355e38, 3.3895313892515355e38).to(dtype=p_c.dtype))
                    else:
                        p_c.copy_(ref_c.to(dtype=p_c.dtype))
                else:
                    p_fp32 = p_c.float()
                    if weight_decay != 0.0:
                        p_fp32.mul_(1.0 - lr * weight_decay)
                    p_fp32.add_(step_val, alpha=-1.0)
                    _sr_quantize(p_c, p_fp32, use_sr=use_sr, generator=gen_p)

            if sc is not None:
                sc[:3].zero_()
                sc[5].zero_()

        if not p.is_contiguous():
            p.copy_(p_work)
        if not exp_avg.is_contiguous():
            exp_avg.copy_(m_work)
        if has_master and ref_p is not None and not ref_p.is_contiguous():
            ref_p.copy_(ref_work)