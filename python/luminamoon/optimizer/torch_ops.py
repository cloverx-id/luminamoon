from __future__ import annotations

import contextlib
from typing import Optional
import torch
from torch import Tensor


def _get_device_context(dev: torch.device):
    if dev.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.device(dev)
    elif hasattr(torch, "xpu") and dev.type == "xpu" and torch.xpu.is_available():
        return torch.xpu.device(dev)
    elif hasattr(torch, "accelerator") and hasattr(torch.accelerator, "device"):
        return getattr(torch.accelerator, "device")(dev)
    return contextlib.nullcontext()


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
