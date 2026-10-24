from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import torch

HAS_TRITON = False
if TYPE_CHECKING:
    import triton.language as tl  # type: ignore
    triton: Any = None
    HAS_TRITON = True
else:
    triton = None
    tl = None
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
        return getattr(torch.accelerator, "device")(dev)
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
            return tl.math.tanh(x)  # type: ignore
    elif _HAS_TL_ROOT_TANH:
        @triton.jit
        def _triton_tanh_fast(x):
            return tl.tanh(x)  # type: ignore
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
else:
    _prng: Any = None
    _sr_bf16_bits: Any = None
    _sr_fp16_bits: Any = None
    _store_param: Any = None
    _store_state_buffer: Any = None
    _triton_tanh_fast: Any = None
    _lumina_diff_sum_kernel: Any = None
    _lumina_post_step_ema_kernel: Any = None
