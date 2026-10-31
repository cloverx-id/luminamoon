try:
    from . import common as _common
except (ImportError, ValueError):
    import common as _common

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

        bc1 = 1.0 - beta1 ** (t + 1)
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

        loss = 0.5 * (w[0] ** 2 + 1000.0 * w[1] ** 2)
        if loss < 1e-4 and converged_step == steps:
            converged_step = t

    final_loss = 0.5 * (w[0] ** 2 + 1000.0 * w[1] ** 2)
    return converged_step, final_loss


def run_hyperparameter_sweeps() -> bool:
    ref_tau_step, _ = run_hyperparam_sweep("tau", 0.8)
    ref_coord_step, _ = run_hyperparam_sweep("coordinate", 0.05)
    ref_rad_step, _ = run_hyperparam_sweep("radial", 0.05)

    tau_basin_steps = [run_hyperparam_sweep("tau", v)[0] for v in [0.56, 0.8, 1.04]]
    coord_basin_steps = [run_hyperparam_sweep("coordinate", v)[0] for v in [0.035, 0.05, 0.065]]
    rad_basin_steps = [run_hyperparam_sweep("radial", v)[0] for v in [0.035, 0.05, 0.065]]

    tau_stable = all(abs(s - ref_tau_step) / float(ref_tau_step) <= 0.15 for s in tau_basin_steps)
    coord_stable = all(abs(s - ref_coord_step) / float(ref_coord_step) <= 0.15 for s in coord_basin_steps)
    rad_stable = all(abs(s - ref_rad_step) / float(ref_rad_step) <= 0.15 for s in rad_basin_steps)

    print(f"  Buffer-2 tau +-30% stability [0.56, 1.04]        : {'PASSED' if tau_stable else 'FAILED'} (ref {ref_tau_step} steps)")
    print(f"  Coordinate bound +-30% stability [0.035, 0.065]  : {'PASSED' if coord_stable else 'FAILED'} (ref {ref_coord_step} steps)")
    print(f"  Radial bound +-30% stability [0.035, 0.065]      : {'PASSED' if rad_stable else 'FAILED'} (ref {ref_rad_step} steps)")

    return tau_stable and coord_stable and rad_stable


if __name__ == "__main__":
    run_hyperparameter_sweeps()
