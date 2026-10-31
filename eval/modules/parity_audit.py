try:
    from . import common as _common
except (ImportError, ValueError):
    import common as _common

import numpy as np


def run_ema_comparison_standard(buffer_mode, use_ema, ema_decay=0.8, lr=0.02, max_steps=1000, tol=1e-4):
    w = np.array([10.0, 1.0], dtype=np.float64)
    m = np.zeros_like(w)
    v = np.zeros_like(w)

    beta1, beta2 = 0.9, 0.999
    lam, tau, bound_ratio, c_min, eps = 0.3, 0.8, 0.05, 0.5, 1e-8

    cautious_scale_ema = 1.0
    bound_scale_ema = 1.0

    for t in range(1, max_steps + 1):
        g = np.array([w[0], 1000.0 * w[1]], dtype=np.float64)

        if buffer_mode == 1:
            w1 = beta1**2
            w2 = 1.0 - w1
            clm_m = w1 * m + w2 * g
            clm_defect = lam * np.abs(g - m)
            clm_env = w1 * np.abs(m) + w2 * np.abs(g) + clm_defect + eps
            u = clm_m / clm_env
            m = beta1 * m + (1.0 - beta1) * g
        else:
            m_new = beta1 * m + (1.0 - beta1) * g
            clm_m = beta1 * m_new + (1.0 - beta1) * g
            diff = g - m_new
            v = beta2 * v + (1.0 - beta2) * (diff**2)

            bc1 = 1.0 - beta1 ** (t + 1)
            bc2 = max(1.0 - beta2**t, 1e-15)
            c1 = (bc1 * tau) / np.sqrt(bc2)
            c2 = eps * bc1 * tau
            sigma = np.sqrt(np.maximum(v, 0.0)) * c1 + c2
            u = np.tanh(clm_m / np.maximum(sigma, 1e-12))
            m = m_new

        mask = (u * g > 0.0).astype(np.float64)
        m_bar = np.clip(np.mean(mask), c_min, 1.0)
        target_cautious_scale = 1.0 / m_bar

        norm_p = np.linalg.norm(w)
        u_eff = u * mask
        R = bound_ratio * max(norm_p, 1.0)

        if use_ema and t > 1:
            active_cautious_scale = cautious_scale_ema
            active_bound_scale = bound_scale_ema

            cautious_scale_ema = ema_decay * cautious_scale_ema + (1.0 - ema_decay) * target_cautious_scale
            norm_delta = np.linalg.norm(u_eff) * cautious_scale_ema
            r = (lr * norm_delta) / R
            target_bound_scale = 1.0 if r < 1e-4 else np.tanh(r) / max(r, 1e-8)
            bound_scale_ema = ema_decay * bound_scale_ema + (1.0 - ema_decay) * target_bound_scale
        else:
            active_cautious_scale = target_cautious_scale
            norm_delta = np.linalg.norm(u_eff) * active_cautious_scale
            r = (lr * norm_delta) / R
            target_bound_scale = 1.0 if r < 1e-4 else np.tanh(r) / max(r, 1e-8)
            active_bound_scale = target_bound_scale
            cautious_scale_ema = target_cautious_scale
            bound_scale_ema = target_bound_scale

        delta = u_eff * active_cautious_scale
        step_val = lr * delta * active_bound_scale
        w = w - step_val

        loss = 0.5 * (w[0] ** 2 + 1000.0 * w[1] ** 2)
        if loss < tol:
            return t, loss, "CONVERGED"

    return max_steps, loss, "MAX_STEPS"


def run_parity_audit() -> bool:
    print("\n1-Pass Fused Scratchpad EMA vs 2-Pass Instantaneous Parity:")
    ema_configs = [
        (1, False, 0.0, "Buffer 1 (Instantaneous 2-Pass)"),
        (1, True, 0.95, "Buffer 1 (Fused EMA decay=0.95 Default)"),
        (2, False, 0.0, "Buffer 2 (Instantaneous 2-Pass)"),
        (2, True, 0.80, "Buffer 2 (Fused EMA decay=0.80 Default)"),
    ]

    base_steps = {}
    base_losses = {}
    ema_ok = True
    for buf, use_e, decay, desc in ema_configs:
        steps, loss, status = run_ema_comparison_standard(buf, use_e, decay)
        if not use_e:
            base_steps[buf] = steps
            base_losses[buf] = loss
            delta_str = "Baseline"
        else:
            step_delta_pct = ((steps - base_steps[buf]) / float(base_steps[buf])) * 100.0
            delta_str = f"{step_delta_pct:+.2f}%"
            if status != "CONVERGED" or step_delta_pct > 1.0:
                ema_ok = False
        print(f"  {desc:<40} | Steps: {steps:<4} | Loss: {loss:.2e} | Step Delta: {delta_str:<8} | {status}")

    return ema_ok


if __name__ == "__main__":
    run_parity_audit()
