from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

from .validation import _decode_state_numpy, _prng_numpy, _quantize_numpy, _quantize_state_numpy


def step_numpy_group(
    self,
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
    lr = float(group["lr"])
    beta1 = float(group["betas"][0])
    beta2 = float(group["betas"][1])
    eps = float(group["eps"])
    weight_decay = float(group["weight_decay"])
    tau = float(group["tau"])
    lam = float(group["lam"])
    cautious = bool(group.get("cautious", True))
    clamp_min = float(group.get("cautious_clamp_min", 0.5))
    buf_count = int(group["buffer"])
    bound = bool(group.get("bound", True))
    bound_type = group.get("bound_type", 2)
    if isinstance(bound_type, str):
        bound_mode = 1 if bound_type == "radial" else (2 if bound_type == "coordinate" else 0)
    else:
        bound_mode = int(bound_type)
    bound_ratio = float(group.get("bound_ratio", 0.05))
    if not bound or bound_ratio <= 0.0:
        bound_mode = 0
    fused_single_pass = bool(group.get("fused_single_pass", True))
    ema_decay = float(group.get("ema_decay", 0.95 if buf_count == 1 else 0.80))
    centered_defect = bool(group.get("centered_defect", False))
    use_sr = bool(group.get("stochastic_rounding", True))
    seed = int(group.get("seed", getattr(self, "seed", 1337)))

    for i, (p, g, m, sq, mp, sc, step) in enumerate(
        zip(params, grads, exp_avgs, exp_avg_sqs, master_params, scratches, steps)
    ):
        n_elem = int(p.size)
        if n_elem == 0:
            continue

        p_idx = param_indices[i] if param_indices is not None else i
        safe_step = max(int(step), 1)
        seed_m = int((seed + safe_step * 0x9E3779B9 + p_idx * 10007) & 0x7FFFFFFF)
        seed_p = int((seed_m + 0x55555555) & 0x7FFFFFFF)

        is_fp16 = hasattr(p, "dtype") and ("float16" in str(p.dtype).lower() or str(p.dtype) in ("f2", "<f2", "=f2"))
        effective_eps = eps

        has_master = mp is not None
        p_init = np.array(mp if has_master else p, dtype=np.float32, copy=True)

        g_raw = np.asarray(g, dtype=np.float32)
        valid_g = np.isfinite(g_raw) & (np.abs(g_raw) < 1.0e38)
        g_f32 = np.where(valid_g, np.clip(g_raw, -1.0e18, 1.0e18), 0.0).astype(np.float32, copy=False)
        m_f32 = _decode_state_numpy(m)

        if buf_count == 2:
            sq_f32 = _decode_state_numpy(sq)
            m_new = (beta1 * m_f32 + (1.0 - beta1) * g_f32).astype(np.float32)
            clm_m = (beta1 * m_new + (1.0 - beta1) * g_f32).astype(np.float32)
            diff = np.clip(g_f32 - m_new, -1.0e18, 1.0e18)
            if centered_defect and n_elem > 1:
                diff_mean = float(np.mean(np.clip(g_f32 - m_f32, -1.0e18, 1.0e18))) * beta1
                diff = np.clip(diff - diff_mean, -1.0e18, 1.0e18)
            v_new = (beta2 * sq_f32 + (1.0 - beta2) * (diff * diff)).astype(np.float32)
            is_m_fp16 = hasattr(m, "dtype") and ("float16" in str(m.dtype).lower() or str(m.dtype) in ("f2", "<f2", "=f2"))
            is_sq_fp16 = hasattr(sq, "dtype") and ("float16" in str(sq.dtype).lower() or str(sq.dtype) in ("f2", "<f2", "=f2"))
            if is_fp16 or is_m_fp16 or is_sq_fp16:
                v_new = np.minimum(v_new, 65000.0)

            bc1_nes = 1.0 - (beta1 ** (safe_step + 1))
            bc2 = max(1.0 - (beta2 ** safe_step), 1e-15)
            c1 = (bc1_nes * tau) / math.sqrt(bc2)
            c2 = effective_eps * bc1_nes * tau
            sigma = np.sqrt(np.maximum(v_new, 0.0)) * c1 + c2
            u = np.tanh(clm_m / np.maximum(sigma, 1e-12)).astype(np.float32)

            _quantize_state_numpy(m, m_new, use_sr, seed_m)
            _quantize_state_numpy(sq, v_new, False, seed_m)
        else:
            beta1_sq = beta1 * beta1
            one_minus_beta1_sq = 1.0 - beta1_sq
            m_new = (beta1 * m_f32 + (1.0 - beta1) * g_f32).astype(np.float32)
            clm_m = (beta1_sq * m_f32 + one_minus_beta1_sq * g_f32).astype(np.float32)
            diff = np.clip(g_f32 - m_f32, -1.0e18, 1.0e18)
            if centered_defect and n_elem > 1:
                diff_mean = float(np.mean(diff))
                diff = np.clip(diff - diff_mean, -1.0e18, 1.0e18)
            clm_env = (
                beta1_sq * np.abs(m_f32)
                + one_minus_beta1_sq * np.abs(g_f32)
                + lam * np.abs(diff)
                + effective_eps
            )
            u = (clm_m / clm_env).astype(np.float32)
            _quantize_state_numpy(m, m_new, use_sr, seed_m)

        m_mask = (u * g_f32 > 0.0).astype(np.float32) if cautious else None
        mask_sum = float(np.sum(m_mask)) if (cautious and m_mask is not None) else 0.0
        u_eff = (u * m_mask) if (cautious and m_mask is not None) else u

        use_single_pass = (fused_single_pass and safe_step > 1) or (not cautious and bound_mode != 1)
        if use_single_pass and sc is not None and len(sc) >= 6:
            cautious_scale = float(sc[3]) if cautious else 1.0
            bound_scale = float(sc[4]) if bound_mode == 1 else 1.0
            delta_theta = u_eff * cautious_scale if cautious else u
            step_val = (lr * delta_theta).astype(np.float32)

            if bound_mode == 1:
                step_val = step_val * bound_scale
            elif bound_mode == 2:
                b_i = bound_ratio * np.sqrt(p_init * p_init + 1e-4)
                step_val = b_i * np.tanh(step_val / b_i)

            if is_fp16:
                b_safe = np.maximum(np.abs(p_init) * 0.01, 1.5e-4)
                step_val = b_safe * np.tanh(step_val / b_safe)

            if cautious:
                m_bar = max(min(mask_sum / max(float(n_elem), 1.0), 1.0), clamp_min)
                target_c = 1.0 / m_bar
                sc[3] = ema_decay * cautious_scale + (1.0 - ema_decay) * target_c
            if bound_mode == 1:
                c_eff = float(sc[3]) if cautious else 1.0
                u_sq_sum = float(np.sum(u_eff * u_eff))
                p_sq_sum = float(np.sum(p_init * p_init))
                norm_delta = math.sqrt(max(u_sq_sum, 0.0)) * c_eff
                norm_p = math.sqrt(max(p_sq_sum, 0.0))
                radius = bound_ratio * max(norm_p, 1.0)
                r = (lr * norm_delta) / radius
                target_b = 1.0 if r < 1e-4 else (math.tanh(r) / max(r, 1e-8))
                sc[4] = ema_decay * bound_scale + (1.0 - ema_decay) * target_b
            sc[:3] = 0.0
            sc[5] = 0.0
        else:
            if cautious:
                m_bar = max(min(mask_sum / max(float(n_elem), 1.0), 1.0), clamp_min)
                cautious_scale = 1.0 / m_bar
                if sc is not None and len(sc) >= 4:
                    sc[3] = cautious_scale
            else:
                cautious_scale = 1.0

            delta_theta = u_eff * cautious_scale if cautious else u
            step_val = (lr * delta_theta).astype(np.float32)

            if bound_mode == 1:
                u_sq_sum = float(np.sum(u_eff * u_eff))
                p_sq_sum = float(np.sum(p_init * p_init))
                norm_delta = math.sqrt(max(u_sq_sum, 0.0)) * cautious_scale
                norm_p = math.sqrt(max(p_sq_sum, 0.0))
                radius = bound_ratio * max(norm_p, 1.0)
                r = (lr * norm_delta) / radius
                bound_scale = 1.0 if r < 1e-4 else (math.tanh(r) / max(r, 1e-8))
                if sc is not None and len(sc) >= 5:
                    sc[4] = bound_scale
                step_val = step_val * bound_scale
            elif bound_mode == 2:
                b_i = bound_ratio * np.sqrt(p_init * p_init + 1e-4)
                step_val = b_i * np.tanh(step_val / b_i)

            if is_fp16:
                b_safe = np.maximum(np.abs(p_init) * 0.01, 1.5e-4)
                step_val = b_safe * np.tanh(step_val / b_safe)

            if sc is not None and len(sc) >= 6:
                sc[:3] = 0.0
                sc[5] = 0.0

        if weight_decay != 0.0:
            p_decayed = p_init * (1.0 - lr * weight_decay)
        else:
            p_decayed = p_init
        p_updated = (p_decayed - step_val).astype(np.float32)

        if has_master:
            mp[:] = p_updated
            _quantize_numpy(p, p_updated, False, seed_p)
        else:
            _quantize_numpy(p, p_updated, use_sr, seed_p)
