try:
    from . import common as _common
except (ImportError, ValueError):
    import common as _common

import numpy as np


def calc_theoretical_limits(lam, beta1=0.9):
    w1 = beta1**2
    w2 = 1.0 - w1
    u_spike = w2 / (w2 + lam)
    u_reversal = (w1 - w2) / (1.0 + 2.0 * lam)
    return u_spike, u_reversal


def calc_equilibrium_lambda(beta1=0.9):
    return (2.0 * (1.0 - beta1**2) ** 2) / (4.0 * beta1**2 - 3.0)


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

        loss = 0.5 * (w[0] ** 2 + 1000.0 * w[1] ** 2)
        if loss < tol and converged_step == steps:
            converged_step = t

    final_loss = 0.5 * (w[0] ** 2 + 1000.0 * w[1] ** 2)
    return converged_step, final_loss


def run_defect_factor_sweep() -> bool:
    lam_star = calc_equilibrium_lambda(0.9)
    print(f"\nStage II.2: Equilibrium Calibration & Sensitivity Sweeps (lambda* = {lam_star:.4f})")

    lambda_values = [0.0, 0.1, 0.19, 0.3, 0.5, 0.8, 1.0]
    step_at_03 = 1000
    for lam in lambda_values:
        u_spk, u_rev = calc_theoretical_limits(lam)
        c_step, f_loss = run_lambda_sweep(lam)
        if abs(lam - 0.3) < 1e-9:
            step_at_03 = c_step
        step_str = f"{c_step}" if c_step < 1000 else "Max"
        print(f"  lambda={lam:<4.2f} | u_spike={u_spk:.4f}, u_rev={u_rev:.4f} | Steps: {step_str:<4} | Loss: {f_loss:.2e}")

    return step_at_03 < 1000


if __name__ == "__main__":
    run_defect_factor_sweep()
