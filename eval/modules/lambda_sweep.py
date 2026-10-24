try:
    from .common import *
except (ImportError, ValueError):
    from common import *
import numpy as np

def calc_theoretical_limits(lam, beta1=0.9):
    w1 = beta1**2
    w2 = 1.0 - w1
    u_spike = w2 / (w2 + lam)
    u_reversal = (w1 - w2) / (1.0 + 2.0 * lam)
    return u_spike, u_reversal

def calc_equilibrium_lambda(beta1=0.9):
    return (2.0 * (1.0 - beta1**2)**2) / (4.0 * beta1**2 - 3.0)

def run_lambda_sweep(lam, beta1=0.9, lr=0.02, steps=1000, tol=1e-5):
    w = np.array([10.0, 1.0], dtype=np.float64)
    m = np.zeros_like(w)

    w1 = beta1**2
    w2 = 1.0 - w1
    eps = 1e-8

    converged_step = steps
    for t in range(1, steps + 1):
        g = np.array([w[0], 1000.0 * w[1]], dtype=np.float64)

        clm_m = w1 * m + w2 * g
        clm_defect = lam * np.abs(g - m)
        clm_env = w1 * np.abs(m) + w2 * np.abs(g) + clm_defect + eps
        u = clm_m / clm_env

        m = beta1 * m + (1.0 - beta1) * g
        w = w - lr * u

        loss = 0.5 * (w[0]**2 + 1000.0 * w[1]**2)
        if loss < tol and converged_step == steps:
            converged_step = t

    final_loss = 0.5 * (w[0]**2 + 1000.0 * w[1]**2)
    return converged_step, final_loss

def describe_lambda_behavior(lam, u_spk, u_rev, c_step, f_loss, max_steps=1000):
    if c_step >= max_steps:
        return f"Stalled above 1e-5 (final {f_loss:.2e}, resonance w2=lam)"
    if lam == 0.0:
        return f"Unregularized B-CLM (hits 1e-5 at {c_step}, rebounds to {f_loss:.2e})"
    if abs(u_spk - u_rev) < 1e-3:
        return f"Spike-reversal equilibrium (u_spk~=u_rev, hits 1e-5 at {c_step})"
    if f_loss < 1e-4:
        return f"Strong steady-state damping (hits 1e-5 at {c_step}, final {f_loss:.2e})"
    if u_spk > u_rev:
        return f"Under-damped spike ceiling (hits 1e-5 at {c_step}, rebounds to {f_loss:.2e})"
    return f"Over-damped velocity (hits 1e-5 at {c_step}, rebounds to {f_loss:.2e})"

def run_defect_factor_sweep():
    lam_star = calc_equilibrium_lambda(0.9)
    print(f"\n--- [LSO-Eval] TD-CLM Defect Factor (Lambda) Sweep (Equilibrium lambda* = {lam_star:.4f}) ---")
    print(f"{'Lambda':<8} | {'Spike Limit':<12} | {'Reversal Limit':<15} | {'Steps (1e-5)':<14} | {'Final Loss':<12} | {'Behavior'}")

    lambda_values = [0.0, 0.1, 0.19, 0.3, 0.5, 0.8, 1.0]
    for lam in lambda_values:
        u_spk, u_rev = calc_theoretical_limits(lam)
        c_step, f_loss = run_lambda_sweep(lam)
        note = describe_lambda_behavior(lam, u_spk, u_rev, c_step, f_loss)
        step_str = f"{c_step}" if c_step < 1000 else "Max Steps"
        print(f"{lam:<8.2f} | {u_spk:<12.4f} | {u_rev:<15.4f} | {step_str:<14} | {f_loss:<12.2e} | {note}")

if __name__ == "__main__":
    run_defect_factor_sweep()
