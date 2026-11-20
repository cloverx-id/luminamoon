from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from .constants import SR_QUANTIZE_CHUNK_SIZE


def _get_generator(opt: Any, device: torch.device, seed: int, slot: int) -> torch.Generator:
    effective_seed = int(seed) & 0x7FFFFFFF
    dev_type = device.type
    if device.index is not None:
        dev_idx = device.index
    elif dev_type == "cuda" and torch.cuda.is_available():
        dev_idx = torch.cuda.current_device()
    elif dev_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        dev_idx = torch.xpu.current_device()
    elif (
        hasattr(torch, "accelerator")
        and hasattr(torch.accelerator, "current_device_index")
        and torch.accelerator.is_available()
    ):
        dev_idx = getattr(torch.accelerator, "current_device_index")()
    else:
        dev_idx = 0

    generators: Dict[Tuple[str, int, int], torch.Generator] = opt._generators
    key = (dev_type, dev_idx, slot)
    if key not in generators:
        try:
            gen = torch.Generator(device=device)
        except (RuntimeError, ValueError, TypeError):
            gen = torch.Generator(device="cpu")
        gen.manual_seed(effective_seed)
        generators[key] = gen
    else:
        generators[key].manual_seed(effective_seed)
    return generators[key]


def _step_generators(
    opt: Any,
    device: torch.device,
    seed: int,
    step: int,
    param_idx: int,
    need_m: bool,
    need_p: bool,
    need_v: bool,
) -> Tuple[Optional[torch.Generator], Optional[torch.Generator], Optional[torch.Generator]]:
    base = (seed + step * 0x9E3779B9 + param_idx * 10007) & 0x7FFFFFFF
    gen_m = _get_generator(opt, device, base, slot=0) if need_m else None
    gen_p = _get_generator(opt, device, (base + 0x55555555) & 0x7FFFFFFF, slot=1) if need_p else None
    gen_v = _get_generator(opt, device, (base + 0x2A6D365B) & 0x7FFFFFFF, slot=2) if need_v else None
    return gen_m, gen_p, gen_v


def _get_device_context(dev: torch.device):
    if dev.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.device(dev)
    if hasattr(torch, "xpu") and dev.type == "xpu" and torch.xpu.is_available():
        return torch.xpu.device(dev)
    if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "device"):
        return getattr(torch.accelerator, "device")(dev)
    return contextlib.nullcontext()


def _sample_int32_noise(
    length: int,
    high: int,
    target_device: torch.device,
    generator: Optional[torch.Generator],
) -> Tensor:
    cpu_gen = generator if (generator is not None and generator.device.type == "cpu") else None
    use_cpu_rng = (target_device.type == "cpu") or (cpu_gen is not None)
    if use_cpu_rng:
        noise = torch.empty(length, dtype=torch.int32, device="cpu").random_(0, high, generator=cpu_gen)
        return noise.to(device=target_device, non_blocking=True) if target_device.type != "cpu" else noise
    try:
        return torch.empty(length, dtype=torch.int32, device=target_device).random_(0, high, generator=generator)
    except RuntimeError:
        noise = torch.empty(length, dtype=torch.int32, device="cpu").random_(0, high, generator=cpu_gen)
        return noise.to(device=target_device, non_blocking=True)


def _apply_bf16_sr_inplace(
    flat_view: Tensor,
    target_device: torch.device,
    generator: Optional[torch.Generator],
) -> None:
    n_elem = flat_view.numel()
    mask_bits = 0xFFFF
    shift_range = 1 << 16
    for offset in range(0, n_elem, SR_QUANTIZE_CHUNK_SIZE):
        cur = min(SR_QUANTIZE_CHUNK_SIZE, n_elem - offset)
        sub_slice = flat_view[offset : offset + cur]
        noise = _sample_int32_noise(cur, shift_range, target_device, generator)
        abs_int = sub_slice & 0x7FFFFFFF
        sign_bit = sub_slice & (-2147483648)
        rounded_abs = torch.clamp_max((abs_int + noise) & (~mask_bits), 0x7F7F0000)
        sub_slice.copy_(sign_bit | rounded_abs)


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
                step_val = b_i * torch.tanh(step_val / b_i)

    if target_dtype == torch.float16:
        b_safe = torch.clamp_min(torch.abs(p_ref) * 0.01, 1.5e-4)
        if is_triangle:
            denom = b_safe * (b_safe + torch.abs(step_val))
            step_val = (denom * step_val) / (denom + step_val.square())
        else:
            step_val = b_safe * torch.tanh(step_val / b_safe)
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
            else torch.where(r < 1e-4, torch.ones_like(r), torch.tanh(r) / torch.clamp_min(r, 1e-8))
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
    generator: Optional[torch.Generator] = None,
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
        _apply_bf16_sr_inplace(flat_view, target.device, generator)
    else:
        n_elem = flat_view.numel()
        for offset in range(0, n_elem, SR_QUANTIZE_CHUNK_SIZE):
            cur = min(SR_QUANTIZE_CHUNK_SIZE, n_elem - offset)
            sub_slice = flat_view[offset : offset + cur]
            raw_rand = _sample_int32_noise(cur, 1 << 24, target.device, generator)
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


def _decode_state_f32(state_tensor: Tensor) -> Tensor:
    if state_tensor.dtype == torch.float16:
        raw_i32 = state_tensor.view(torch.int16).to(torch.int32)
        return (raw_i32 << 16).view(torch.float32)
    return state_tensor.float()


def _encode_state_from_f32(
    target: Tensor,
    val_f32: Tensor,
    use_sr: bool,
    generator: Optional[torch.Generator] = None,
) -> None:
    if target.dtype != torch.float16:
        _sr_quantize(target, val_f32, use_sr=use_sr, generator=generator)
        return

    if not use_sr:
        clamped = val_f32.clamp(-3.3895313892515355e38, 3.3895313892515355e38)
        work_f32 = clamped.clone() if clamped.is_contiguous() else clamped.contiguous()
        int_view = work_f32.view(torch.int32)
        abs_int = int_view & 0x7FFFFFFF
        sign_bit = int_view & (-2147483648)
        lsb = (abs_int >> 16) & 1
        rounded_abs = torch.clamp_max((abs_int + 0x7FFF + lsb) & (-65536), 0x7F7F0000)
        out_i16 = ((sign_bit | rounded_abs) >> 16).to(torch.int16)
        target.view(torch.int16).copy_(out_i16.view(target.shape))
        return

    work_f32 = val_f32.clone() if val_f32.is_contiguous() else val_f32.contiguous()
    int_view = work_f32.view(torch.int32)
    _apply_bf16_sr_inplace(int_view.view(-1), target.device, generator)
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
    generator: Optional[torch.Generator] = None,
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
    _sr_quantize(p, p_fp32, use_sr, generator)
