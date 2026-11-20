from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

from ..config import StepOptions
from ..validation import _decode_state_numpy, _quantize_numpy, _quantize_state_numpy


def _is_fp16_array(arr: Any) -> bool:
    if not hasattr(arr, "dtype"):
        return False
    ds = str(arr.dtype).lower()
    return "float16" in ds or ds in ("f2", "<f2", "=f2")


def _tanh_pade_numpy(x_in: Any) -> Any:
    x = np.clip(np.asarray(x_in, dtype=np.float32), np.float32(-7.9988117), np.float32(7.9988117))
    x2 = x * x
    p = np.float32(-2.7607684e-16) * x2 + np.float32(2.000188e-13)
    q = np.float32(1.1982584e-6) * x2 + np.float32(1.1853471e-4)
    p = p * x2 - np.float32(8.604672e-11)
    q = q * x2 + np.float32(2.2684346e-3)
    p = p * x2 + np.float32(5.122297e-8)
    q = q * x2 + np.float32(4.8935246e-3)
    p = p * x2 + np.float32(1.4857224e-5)
    p = p * x2 + np.float32(6.3726193e-4)
    p = p * x2 + np.float32(4.8935246e-3)
    return ((x * p) / q).astype(np.float32, copy=False)


def _radial_scale(r: float, is_triangle: bool) -> float:
    if is_triangle:
        return (1.0 + r) / (1.0 + r + r * r)
    if r < 1e-4:
        return 1.0
    x = max(min(r, 7.9988117), -7.9988117)
    x2 = x * x
    p = -2.7607684e-16 * x2 + 2.000188e-13
    q = 1.1982584e-6 * x2 + 1.1853471e-4
    p = p * x2 - 8.604672e-11
    q = q * x2 + 2.2684346e-3
    p = p * x2 + 5.122297e-8
    q = q * x2 + 4.8935246e-3
    p = p * x2 + 1.4857224e-5
    p = p * x2 + 6.3726193e-4
    p = p * x2 + 4.8935246e-3
    return ((x * p) / q) / max(r, 1e-8)


def _apply_numpy_bounds(
    step_val: Any,
    p_init: Any,
    opts: StepOptions,
    bound_scale: float,
    is_fp16: bool,
) -> Any:
    if opts.bound_mode == 1:
        step_val = step_val * bound_scale
    elif opts.bound_mode == 2:
        if opts.is_triangle:
            b_i = opts.bound_ratio * (np.abs(p_init) + 1e-2)
            denom = b_i * (b_i + np.abs(step_val))
            step_val = (denom * step_val) / (denom + step_val * step_val)
        else:
            b_i = opts.bound_ratio * np.sqrt(p_init * p_init + 1e-4)
            step_val = b_i * _tanh_pade_numpy(step_val / b_i)

    if is_fp16:
        b_safe = np.maximum(np.abs(p_init) * 0.01, 1.5e-4)
        if opts.is_triangle:
            denom = b_safe * (b_safe + np.abs(step_val))
            step_val = (denom * step_val) / (denom + step_val * step_val)
        else:
            step_val = b_safe * _tanh_pade_numpy(step_val / b_safe)
    return step_val


def step_numpy_group(
    opt: Any,
    group: Dict[str, Any],
    params: list[Any],
    grads: list[Any],
    exp_avgs: list[Any],
    exp_avg_sqs: list[Any],
    master_params: list[Any],
    scratches: list[Any],
    steps: list[int],
    param_indices: Optional[List[int]] = None,
) -> None:
    if not HAS_NUMPY or np is None:
        raise RuntimeError("NumPy is required for pure-python CPU fallback execution.")

    opts = StepOptions.from_group(group, fallback_seed=getattr(opt, "seed", 1337))

    for i, (p, g, m, sq, mp, sc, step) in enumerate(
        zip(params, grads, exp_avgs, exp_avg_sqs, master_params, scratches, steps)
    ):
        n_elem = int(p.size)
        if n_elem == 0:
            continue

        p_idx = param_indices[i] if param_indices is not None else i
        safe_step = max(int(step), 1)
        seed_m = int((opts.seed + p_idx * 10007) & 0x7FFFFFFF)
        seed_p = int((seed_m + 0x55555555) & 0x7FFFFFFF)
        seed_v = int((seed_m + 0x2A6D365B) & 0x7FFFFFFF)
        weyl_m = int((safe_step * 0x6A09E667) & 0xFFFFFFFF)
        weyl_v = int((safe_step * 0xBB67AE85) & 0xFFFFFFFF)
        weyl_p = int((safe_step * 0x9E3779B9) & 0xFFFFFFFF)

        is_fp16 = _is_fp16_array(p)
        has_master = mp is not None
        p_init = np.array(mp if has_master else p, dtype=np.float32, copy=True)

        g_raw = np.asarray(g, dtype=np.float32)
        valid_g = np.isfinite(g_raw) & (np.abs(g_raw) < 1.0e38)
        g_f32 = np.where(valid_g, np.clip(g_raw, -1.0e18, 1.0e18), 0.0).astype(np.float32, copy=False)
        m_f32 = _decode_state_numpy(m)

        if opts.buffer == 2:
            sq_f32 = _decode_state_numpy(sq)
            m_new = (opts.beta1 * m_f32 + (1.0 - opts.beta1) * g_f32).astype(np.float32)
            clm_m = (opts.beta1 * m_new + (1.0 - opts.beta1) * g_f32).astype(np.float32)
            diff = np.clip(g_f32 - m_new, -1.0e18, 1.0e18)
            v_new = (opts.beta2 * sq_f32 + (1.0 - opts.beta2) * (diff * diff)).astype(np.float32)
            if is_fp16 or _is_fp16_array(m) or _is_fp16_array(sq):
                v_new = np.minimum(v_new, 65000.0)
            c1, c2 = opts.dual_coefficients(safe_step)
            sigma = np.sqrt(np.maximum(v_new, 0.0)) * c1 + c2
            u = _tanh_pade_numpy(clm_m / np.maximum(sigma, 1e-12))
            _quantize_state_numpy(m, m_new, opts.use_sr, seed_m, step_offset=weyl_m)
            _quantize_state_numpy(sq, v_new, opts.use_sr, seed_v, step_offset=weyl_v)
        elif opts.buffer == "t2":
            sq_f32 = _decode_state_numpy(sq)
            beta1_sq = opts.beta1 * opts.beta1
            one_minus_beta1_sq = 1.0 - beta1_sq
            m_new = (opts.beta1 * m_f32 + (1.0 - opts.beta1) * g_f32).astype(np.float32)
            clm_m = (beta1_sq * m_f32 + one_minus_beta1_sq * g_f32).astype(np.float32)
            diff = np.clip(g_f32 - m_f32, -1.0e18, 1.0e18)
            v_new = (opts.beta2 * sq_f32 + (1.0 - opts.beta2) * (diff * diff)).astype(np.float32)
            if is_fp16 or _is_fp16_array(m) or _is_fp16_array(sq):
                v_new = np.minimum(v_new, 65000.0)
            c_sigma, c_defect = opts.dual_coefficients(safe_step)
            clm_env = (
                beta1_sq * np.abs(m_f32)
                + one_minus_beta1_sq * np.abs(g_f32)
                + c_defect * np.abs(diff)
                + c_sigma * np.sqrt(np.maximum(v_new, 0.0))
                + opts.eps
            )
            u = (clm_m / clm_env).astype(np.float32)
            _quantize_state_numpy(m, m_new, opts.use_sr, seed_m, step_offset=weyl_m)
            _quantize_state_numpy(sq, v_new, opts.use_sr, seed_v, step_offset=weyl_v)
        else:
            beta1_sq = opts.beta1 * opts.beta1
            one_minus_beta1_sq = 1.0 - beta1_sq
            m_new = (opts.beta1 * m_f32 + (1.0 - opts.beta1) * g_f32).astype(np.float32)
            clm_m = (beta1_sq * m_f32 + one_minus_beta1_sq * g_f32).astype(np.float32)
            diff = np.clip(g_f32 - m_f32, -1.0e18, 1.0e18)
            clm_env = (
                beta1_sq * np.abs(m_f32)
                + one_minus_beta1_sq * np.abs(g_f32)
                + opts.lam * np.abs(diff)
                + opts.eps
            )
            u = (clm_m / clm_env).astype(np.float32)
            _quantize_state_numpy(m, m_new, opts.use_sr, seed_m, step_offset=weyl_m)

        m_mask = (u * g_f32 > 0.0).astype(np.float32) if opts.cautious else None
        mask_sum = float(np.sum(m_mask)) if (opts.cautious and m_mask is not None) else 0.0
        u_eff = (u * m_mask) if (opts.cautious and m_mask is not None) else u

        use_single_pass = (opts.fused_single_pass and safe_step > 1) or not opts.needs_ema_sync

        if use_single_pass and sc is not None and len(sc) >= 6:
            cautious_scale = float(sc[3]) if opts.cautious else 1.0
            bound_scale = float(sc[4]) if opts.bound_mode == 1 else 1.0
        else:
            if opts.cautious:
                m_bar = max(min(mask_sum / max(float(n_elem), 1.0), 1.0), opts.clamp_min)
                cautious_scale = 1.0 / m_bar
                if sc is not None and len(sc) >= 4:
                    sc[3] = cautious_scale
            else:
                cautious_scale = 1.0

            bound_scale = 1.0
            if opts.bound_mode == 1:
                u_sq_sum = float(np.sum(u_eff * u_eff))
                p_sq_sum = float(np.sum(p_init * p_init))
                norm_delta = math.sqrt(max(u_sq_sum, 0.0)) * cautious_scale
                norm_p = math.sqrt(max(p_sq_sum, 0.0))
                radius = opts.bound_ratio * max(norm_p, 1.0)
                bound_scale = _radial_scale((opts.lr * norm_delta) / radius, opts.is_triangle)
                if sc is not None and len(sc) >= 5:
                    sc[4] = bound_scale

        delta_theta = u_eff * cautious_scale if opts.cautious else u
        step_val = _apply_numpy_bounds(
            (opts.lr * delta_theta).astype(np.float32), p_init, opts, bound_scale, is_fp16
        )

        if sc is not None and len(sc) >= 6:
            if use_single_pass:
                if opts.cautious:
                    m_bar = max(min(mask_sum / max(float(n_elem), 1.0), 1.0), opts.clamp_min)
                    sc[3] = opts.ema_decay * cautious_scale + (1.0 - opts.ema_decay) * (1.0 / m_bar)
                if opts.bound_mode == 1:
                    c_eff = float(sc[3]) if opts.cautious else 1.0
                    norm_delta = math.sqrt(max(float(np.sum(u_eff * u_eff)), 0.0)) * c_eff
                    norm_p = math.sqrt(max(float(np.sum(p_init * p_init)), 0.0))
                    radius = opts.bound_ratio * max(norm_p, 1.0)
                    target_b = _radial_scale((opts.lr * norm_delta) / radius, opts.is_triangle)
                    sc[4] = opts.ema_decay * bound_scale + (1.0 - opts.ema_decay) * target_b
            sc[:3] = 0.0
            sc[5] = 0.0

        p_decayed = p_init * (1.0 - opts.lr * opts.weight_decay) if opts.weight_decay != 0.0 else p_init
        p_updated = (p_decayed - step_val).astype(np.float32)

        if has_master:
            mp[:] = p_updated
            _quantize_numpy(p, p_updated, False, seed_p, step_offset=weyl_p)
        else:
            _quantize_numpy(p, p_updated, opts.use_sr, seed_p, step_offset=weyl_p)
