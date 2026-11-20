from __future__ import annotations

import contextlib
from typing import Optional

import torch
from torch import Tensor

from .constants import SR_QUANTIZE_CHUNK_SIZE


def _get_device_context(dev: torch.device):
    if dev.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.device(dev)
    if hasattr(torch, "xpu") and dev.type == "xpu" and torch.xpu.is_available():
        return torch.xpu.device(dev)
    if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "device"):
        return getattr(torch.accelerator, "device")(dev)
    return contextlib.nullcontext()


def _tanh_pade_torch(x: Tensor) -> Tensor:
    x_c = x.clamp(-7.9988117, 7.9988117)
    x2 = x_c * x_c
    p = -2.7607684e-16 * x2 + 2.000188e-13
    q = 1.1982584e-6 * x2 + 1.1853471e-4
    p = p * x2 - 8.604672e-11
    q = q * x2 + 2.2684346e-3
    p = p * x2 + 5.122297e-8
    q = q * x2 + 4.8935246e-3
    p = p * x2 + 1.4857224e-5
    p = p * x2 + 6.3726193e-4
    p = p * x2 + 4.8935246e-3
    return (x_c * p) / q


def _prng_torch(seed: int, offsets: Tensor) -> Tensor:
    # Masking right-shifts simulates unsigned 32-bit logical shift on signed int32 tensors.
    z = (offsets.to(torch.int64) + int(seed) - 1640531527).to(torch.int32)
    z = (((z ^ ((z >> 16) & 0xFFFF)).to(torch.int64) * 0x85EB5A6B) & 0xFFFFFFFF).to(torch.int32)
    z = (((z ^ ((z >> 13) & 0x7FFFF)).to(torch.int64) * 0xC2B2AE35) & 0xFFFFFFFF).to(torch.int32)
    return (z ^ ((z >> 16) & 0xFFFF)).to(torch.int32)


def _prng_weyl_torch(seed: int, offsets: Tensor, step_offset: int = 0) -> Tensor:
    h0 = _prng_torch(seed, offsets)
    if step_offset == 0:
        return h0
    return ((h0.to(torch.int64) + int(step_offset & 0xFFFFFFFF)) & 0xFFFFFFFF).to(torch.int32)


def _apply_bf16_sr_inplace(
    flat_view: Tensor,
    target_device: torch.device,
    seed: int,
    offset_base: int = 0,
    step_offset: int = 0,
) -> None:
    n_elem = flat_view.numel()
    mask_bits = 0xFFFF
    for offset in range(0, n_elem, SR_QUANTIZE_CHUNK_SIZE):
        cur = min(SR_QUANTIZE_CHUNK_SIZE, n_elem - offset)
        sub_slice = flat_view[offset : offset + cur]
        idx = torch.arange(offset_base + offset, offset_base + offset + cur, device=target_device, dtype=torch.int32)
        noise = (_prng_weyl_torch(seed, idx, step_offset) >> 16) & mask_bits
        abs_int = sub_slice & 0x7FFFFFFF
        is_nan = abs_int > 0x7F800000
        sign_bit = sub_slice & (-2147483648)
        rounded_abs = torch.clamp_max((abs_int + noise) & (~mask_bits), 0x7F7F0000)
        out_int = torch.where(is_nan, (sub_slice | 0x00400000) & (~mask_bits), sign_bit | rounded_abs)
        sub_slice.copy_(out_int)


def _apply_bound_scalar(
    step_val: Tensor,
    p_ref: Tensor,
    bound: bool,
    bound_type: str,
    bound_ratio: float,
    target_dtype: torch.dtype,
    radial_scale: Optional[Tensor] = None,
    is_triangle: bool = False,
) -> Tensor:
    if bound and bound_ratio > 0.0:
        if bound_type == "radial" and radial_scale is not None:
            step_val = step_val * radial_scale
        elif bound_type == "coordinate":
            if is_triangle:
                b_i = bound_ratio * (torch.abs(p_ref) + 1e-2)
                denom = b_i * (b_i + torch.abs(step_val))
                step_val = (denom * step_val) / (denom + step_val.square())
            else:
                b_i = bound_ratio * torch.sqrt(p_ref.square() + 1e-4)
                step_val = b_i * _tanh_pade_torch(step_val / b_i)

    if target_dtype == torch.float16:
        b_safe = torch.clamp_min(torch.abs(p_ref) * 0.01, 1.5e-4)
        if is_triangle:
            denom = b_safe * (b_safe + torch.abs(step_val))
            step_val = (denom * step_val) / (denom + step_val.square())
        else:
            step_val = b_safe * _tanh_pade_torch(step_val / b_safe)
    return step_val


def _compute_step_val(
    update: Tensor,
    p: Tensor,
    lr: float,
    bound: bool,
    bound_type: str,
    bound_ratio: float,
    target_dtype: Optional[torch.dtype] = None,
    radial_scale: Optional[Tensor] = None,
    is_triangle: bool = False,
) -> Tensor:
    u_float = torch.nan_to_num(update.float(), nan=0.0, posinf=1.0, neginf=-1.0)
    if lr == 0.0:
        return torch.zeros_like(u_float)

    effective_dtype = target_dtype if target_dtype is not None else p.dtype
    step_val = lr * u_float
    p_ref: Optional[Tensor] = None

    if bound and bound_type == "radial" and bound_ratio > 0.0 and radial_scale is None:
        p_ref = p.float()
        norm_u = torch.linalg.vector_norm(u_float)
        norm_p = torch.linalg.vector_norm(p_ref)
        radius = bound_ratio * torch.clamp_min(norm_p, 1.0)
        r = (lr * norm_u) / radius
        radial_scale = (
            (1.0 + r) / (1.0 + r + r * r)
            if is_triangle
            else torch.where(r < 1e-4, torch.ones_like(r), _tanh_pade_torch(r) / torch.clamp_min(r, 1e-8))
        )

    if p_ref is None and ((bound and bound_type == "coordinate") or effective_dtype == torch.float16):
        p_ref = p.float()

    if p_ref is not None or (bound and bound_type == "radial" and radial_scale is not None):
        step_val = _apply_bound_scalar(
            step_val,
            p_ref if p_ref is not None else p.float(),
            bound,
            bound_type,
            bound_ratio,
            effective_dtype,
            radial_scale=radial_scale,
            is_triangle=is_triangle,
        )
    return step_val


def _sr_quantize(
    target: Tensor,
    val_f32: Tensor,
    use_sr: bool,
    seed: int = 1337,
    offset_base: int = 0,
    step_offset: int = 0,
) -> None:
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
    flat_view = int_view.view(-1)

    if target.dtype == torch.bfloat16:
        _apply_bf16_sr_inplace(
            flat_view, target.device, seed, offset_base=offset_base, step_offset=step_offset
        )
    else:
        n_elem = flat_view.numel()
        for offset in range(0, n_elem, SR_QUANTIZE_CHUNK_SIZE):
            cur = min(SR_QUANTIZE_CHUNK_SIZE, n_elem - offset)
            sub_slice = flat_view[offset : offset + cur]
            idx = torch.arange(offset_base + offset, offset_base + offset + cur, device=target.device, dtype=torch.int32)
            raw_rand = _prng_weyl_torch(seed, idx, step_offset)
            abs_int = sub_slice & 0x7FFFFFFF
            is_nan = abs_int > 0x7F800000
            sign_bit = sub_slice & (-2147483648)
            exp_f32 = (abs_int >> 23) & 0xFF
            sub_shift = torch.clamp(113 - exp_f32, min=0, max=11)
            total_shift = 13 + sub_shift
            dyn_mask = (1 << total_shift) - 1
            noise = (raw_rand >> (32 - total_shift)) & dyn_mask
            rounded_abs = torch.where(exp_f32 < 103, torch.zeros_like(abs_int), (abs_int + noise) & (~dyn_mask))
            rounded_abs = torch.clamp_max(rounded_abs, 0x477FE000)
            out_int = torch.where(is_nan, sub_slice | 0x00400000, sign_bit | rounded_abs)
            sub_slice.copy_(out_int)

    target.copy_(int_view.view(torch.float32).to(dtype=target.dtype))


def _decode_state_f32(state_tensor: Tensor) -> Tensor:
    if state_tensor.dtype == torch.float16:
        raw_i32 = state_tensor.view(torch.int16).to(torch.int32)
        return (raw_i32 << 16).view(torch.float32)
    return state_tensor.float()


def _encode_state_from_f32(
    target: Tensor,
    val_f32: Tensor,
    use_sr: bool,
    seed: int = 1337,
    offset_base: int = 0,
    step_offset: int = 0,
) -> None:
    if target.dtype != torch.float16:
        _sr_quantize(
            target, val_f32, use_sr=use_sr, seed=seed, offset_base=offset_base, step_offset=step_offset
        )
        return

    if not use_sr:
        clamped = val_f32.clamp(-3.3895313892515355e38, 3.3895313892515355e38)
        work_f32 = clamped.clone() if clamped.is_contiguous() else clamped.contiguous()
        int_view = work_f32.view(torch.int32)
        abs_int = int_view & 0x7FFFFFFF
        is_nan = abs_int > 0x7F800000
        sign_bit = int_view & (-2147483648)
        lsb = (abs_int >> 16) & 1
        rounded_abs = torch.clamp_max((abs_int + 0x7FFF + lsb) & (-65536), 0x7F7F0000)
        out_int = torch.where(is_nan, (int_view | 0x00400000) & (-65536), sign_bit | rounded_abs)
        out_i16 = (out_int >> 16).to(torch.int16)
        target.view(torch.int16).copy_(out_i16.view(target.shape))
        return

    work_f32 = val_f32.clone() if val_f32.is_contiguous() else val_f32.contiguous()
    int_view = work_f32.view(torch.int32)
    _apply_bf16_sr_inplace(
        int_view.view(-1), target.device, seed, offset_base=offset_base, step_offset=step_offset
    )
    out_i16 = (int_view >> 16).to(torch.int16)
    target.view(torch.int16).copy_(out_i16)


def _apply_master_update(mp: Tensor, p: Tensor, step_val: Tensor, lr: float, weight_decay: float) -> None:
    if weight_decay != 0.0:
        mp.mul_(1.0 - lr * weight_decay)
    mp.add_(step_val, alpha=-1.0)
    if p.dtype == torch.float16:
        p.copy_(mp.clamp(-65504.0, 65504.0).to(dtype=p.dtype))
    elif p.dtype == torch.bfloat16:
        p.copy_(mp.clamp(-3.3895313892515355e38, 3.3895313892515355e38).to(dtype=p.dtype))
    else:
        p.copy_(mp.to(dtype=p.dtype))


def _sr_apply_step_val(
    p: Tensor,
    step_val: Tensor,
    lr: float,
    weight_decay: float,
    use_sr: bool,
    seed: int = 1337,
    offset_base: int = 0,
    step_offset: int = 0,
) -> None:
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
    _sr_quantize(p, p_fp32, use_sr, seed=seed, offset_base=offset_base, step_offset=step_offset)
