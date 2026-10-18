try:
    from .common import *
except (ImportError, ValueError):
    from common import *

import numpy as np

def run_param_audit(param_name, val, buffer_mode=2):
    np.random.seed(42)
    w = np.array([10.0, 1.0], dtype=np.float64)
    m = np.zeros_like(w)
    v = np.zeros_like(w)

    lr = val if param_name == "lr" else 0.02
    c_min = val if param_name == "c_min" else 0.5
    wd = val if param_name == "wd" else 0.08

    beta1, beta2 = 0.9, 0.999
    lam = 0.3
    tau = 0.8
    bound_ratio = 0.05
    eps = 1e-8
    steps = 1000
    noise_std = 0.5

    converged_step = steps
    for t in range(1, steps + 1):
        g_clean = np.array([w[0], 1000.0 * w[1]], dtype=np.float64)
        g = g_clean + np.random.normal(0.0, noise_std, size=g_clean.shape)

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

            bc1 = 1.0 - beta1**(t + 1)
            bc2 = max(1.0 - beta2**t, 1e-15)
            c1 = (bc1 * tau) / np.sqrt(bc2)
            c2 = eps * bc1 * tau
            sigma = np.sqrt(np.maximum(v, 0.0)) * c1 + c2
            u = np.tanh(clm_m / np.maximum(sigma, 1e-12))
            m = m_new

        mask = (u * g > 0.0).astype(np.float64)
        m_bar = np.clip(np.mean(mask), c_min, 1.0)
        cautious_scale = 1.0 / m_bar
        delta = u * mask * cautious_scale
        step_val = lr * delta

        norm_p = np.linalg.norm(w)
        norm_delta = np.linalg.norm(delta)
        R = bound_ratio * max(norm_p, 1.0)
        r = (lr * norm_delta) / R
        scale = 1.0 if r < 1e-4 else np.tanh(r) / max(r, 1e-8)
        step_val = step_val * scale

        if wd > 0.0:
            w = w * (1.0 - lr * wd)
        w = w - step_val

        loss = 0.5 * (w[0]**2 + 1000.0 * w[1]**2)
        if loss < 1e-4 and converged_step == steps:
            converged_step = t

    final_loss = 0.5 * (w[0]**2 + 1000.0 * w[1]**2)
    return converged_step, final_loss

def run_parameter_audit():
    print("\n--- [LSO-Eval] Parameter Sensitivity Audit (Buffer 2) ---")

    c_min_list = [0.2, 0.4, 0.5, 0.7, 1.0]
    wd_list = [0.0, 0.02, 0.05, 0.08, 0.12]

    print("\n1. Cautious Clamp Min Sensitivity (c_min)")
    print(f"{'c_min':<8} | {'Max Boost':<12} | {'Steps (1e-4)':<14} | {'Final Loss':<12}")
    for val in c_min_list:
        step, loss = run_param_audit("c_min", val, buffer_mode=2)
        boost = f"{1.0/val:.2f}x"
        step_str = f"{step}" if step < 1000 else "Max Steps"
        print(f"{val:<8.2f} | {boost:<12} | {step_str:<14} | {loss:<12.2e}")

    print("\n2. Decoupled Weight Decay Sensitivity (wd)")
    print(f"{'WD':<8} | {'Steps (1e-4)':<14} | {'Final Loss':<12}")
    for val in wd_list:
        step, loss = run_param_audit("wd", val, buffer_mode=2)
        step_str = f"{step}" if step < 1000 else "Max Steps"
        print(f"{val:<8.2f} | {step_str:<14} | {loss:<12.2e}")

if __name__ == "__main__":
    run_parameter_audit()
