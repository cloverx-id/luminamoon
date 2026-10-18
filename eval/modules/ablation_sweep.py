try:
    from .common import *
except (ImportError, ValueError):
    from common import *
import numpy as np

def run_combination_test(buffer_mode, bound_mode, use_cautious, lr=0.02, steps=1000, noise_std=0.5):
    np.random.seed(42)
    w = np.array([10.0, 1.0], dtype=np.float64)
    m = np.zeros_like(w)
    v = np.zeros_like(w)

    beta1, beta2 = 0.9, 0.999
    lam = 0.3
    tau = 0.8
    bound_ratio = 0.05
    c_min = 0.5
    eps = 1e-8

    converged_step = None
    for t in range(1, steps + 1):
        g_clean = np.array([w[0], 1000.0 * w[1]], dtype=np.float64)
        g = g_clean + np.random.normal(0.0, noise_std, size=g_clean.shape) if noise_std > 0 else g_clean

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

        if use_cautious:
            mask = (u * g > 0.0).astype(np.float64)
            m_bar = np.clip(np.mean(mask), c_min, 1.0)
            cautious_scale = 1.0 / m_bar
            delta = u * mask * cautious_scale
        else:
            delta = u

        step_val = lr * delta

        if bound_mode == "coordinate":
            B = bound_ratio * np.sqrt(w**2 + 1e-4)
            step_val = B * np.tanh(step_val / B)
        elif bound_mode == "radial":
            norm_p = np.linalg.norm(w)
            norm_delta = np.linalg.norm(delta)
            R = bound_ratio * max(norm_p, 1.0)
            r = (lr * norm_delta) / R
            scale = 1.0 if r < 1e-4 else np.tanh(r) / max(r, 1e-8)
            step_val = step_val * scale

        w = w - step_val

        loss = 0.5 * (w[0]**2 + 1000.0 * w[1]**2)
        if loss < 1e-4 and converged_step is None:
            converged_step = t

    final_loss = 0.5 * (w[0]**2 + 1000.0 * w[1]**2)
    return converged_step, final_loss

def run_ablation_sweep():
    print("\n--- [LSO-Eval] Combinatorial Feature Ablation (Full Factorial) ---")
    combinations = [
        (1, "radial",     True,  "Buffer 1 + Radial + Cautious"),
        (1, "coordinate", True,  "Buffer 1 + Coordinate + Cautious"),
        (1, "none",       True,  "Buffer 1 + Unbounded + Cautious"),
        (1, "radial",     False, "Buffer 1 + Radial + No Cautious"),
        (1, "coordinate", False, "Buffer 1 + Coordinate + No Cautious"),
        (1, "none",       False, "Buffer 1 + Unbounded + No Cautious"),
        (2, "radial",     True,  "Buffer 2 + Radial + Cautious"),
        (2, "coordinate", True,  "Buffer 2 + Coordinate + Cautious"),
        (2, "none",       True,  "Buffer 2 + Unbounded + Cautious"),
        (2, "radial",     False, "Buffer 2 + Radial + No Cautious"),
        (2, "coordinate", False, "Buffer 2 + Coordinate + No Cautious"),
        (2, "none",       False, "Buffer 2 + Unbounded + No Cautious"),
    ]

    print(f"{'Configuration':<35} | {'Steps (1e-4)':<14} | {'Final Loss'}")
    for buf, b_type, c_flag, desc in combinations:
        step, loss = run_combination_test(buf, b_type, c_flag)
        step_str = f"{step}" if step is not None else "Max Steps"
        print(f"{desc:<35} | {step_str:<14} | {loss:.2e}")

if __name__ == "__main__":
    run_ablation_sweep()
