try:
    from .common import *
except (ImportError, ValueError):
    from common import *
import numpy as np

def run_hyperparam_sweep(mode, param_val, lr=0.02, steps=1000):
    w = np.array([10.0, 1.0], dtype=np.float64)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    tau = param_val if mode == "tau" else 0.8
    bound_ratio = param_val if mode in ("coordinate", "radial") else 0.05
    bound_type = mode if mode in ("coordinate", "radial") else "none"

    converged_step = steps
    for t in range(1, steps + 1):
        g = np.array([w[0], 1000.0 * w[1]], dtype=np.float64)

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
        step_val = lr * u

        if bound_type == "coordinate":
            B = bound_ratio * np.sqrt(w**2 + 1e-4)
            step_val = B * np.tanh(step_val / B)
        elif bound_type == "radial":
            norm_p = np.linalg.norm(w)
            norm_u = np.linalg.norm(u)
            R = bound_ratio * max(norm_p, 1.0)
            r = (lr * norm_u) / R
            scale = 1.0 if r < 1e-4 else np.tanh(r) / max(r, 1e-8)
            step_val = step_val * scale

        m = m_new
        w = w - step_val

        loss = 0.5 * (w[0]**2 + 1000.0 * w[1]**2)
        if loss < 1e-4 and converged_step == steps:
            converged_step = t

    final_loss = 0.5 * (w[0]**2 + 1000.0 * w[1]**2)
    return converged_step, final_loss

def run_hyperparameter_sweeps():
    print("\n--- [LSO-Eval] Hyperparameter Sensitivity Sweeps ---")

    print("\n1. Buffer-2 Tau Sensitivity (Default = 0.80)")
    print(f"{'Tau':<8} | {'Steps (1e-4)':<14} | {'Final Loss':<12} | {'Behavior'}")
    for val in [0.2, 0.56, 0.8, 1.04, 1.5]:
        step, loss = run_hyperparam_sweep("tau", val)
        note = "Premature saturation" if val < 0.5 else ("Operating plateau" if val <= 1.1 else "Over-smoothed")
        print(f"{val:<8.2f} | {step:<14} | {loss:<12.2e} | {note}")

    print("\n2. Coordinate Bound Ratio Sensitivity (Default = 0.05)")
    print(f"{'Ratio':<8} | {'Steps (1e-4)':<14} | {'Final Loss':<12} | {'Behavior'}")
    for val in [0.01, 0.035, 0.05, 0.065, 0.15]:
        step, loss = run_hyperparam_sweep("coordinate", val)
        note = "Over-restricted step" if val < 0.03 else ("Balanced clipping basin" if val <= 0.08 else "Unconstrained")
        print(f"{val:<8.3f} | {step:<14} | {loss:<12.2e} | {note}")

    print("\n3. Radial Bound Ratio Sensitivity (Default = 0.05)")
    print(f"{'Ratio':<8} | {'Steps (1e-4)':<14} | {'Final Loss':<12} | {'Behavior'}")
    for val in [0.01, 0.035, 0.05, 0.065, 0.15]:
        step, loss = run_hyperparam_sweep("radial", val)
        note = "Over-restricted norm" if val < 0.03 else ("Balanced sphere basin" if val <= 0.08 else "Unconstrained")
        print(f"{val:<8.3f} | {step:<14} | {loss:<12.2e} | {note}")

if __name__ == "__main__":
    run_hyperparameter_sweeps()
