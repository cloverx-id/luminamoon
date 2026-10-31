try:
    from .common import _quantize_bf16_rne, _step_optimizer
except (ImportError, ValueError):
    from common import _quantize_bf16_rne, _step_optimizer

import numpy as np


def quad_loss(w):
    return 0.5 * (w[0] ** 2 + 1000.0 * w[1] ** 2)


def quad_grad(w):
    return np.array([w[0], 1000.0 * w[1]], dtype=np.float64)


def rosen_loss(w):
    return (1.0 - w[0]) ** 2 + 100.0 * (w[1] - w[0] ** 2) ** 2


def rosen_grad(w):
    dw0 = -2.0 * (1.0 - w[0]) - 400.0 * w[0] * (w[1] - w[0] ** 2)
    dw1 = 200.0 * (w[1] - w[0] ** 2)
    return np.array([dw0, dw1], dtype=np.float64)


def run_2d_benchmark(problem_type, opt_name, lr=0.01, wd=0.0, max_steps=2000, tol=1e-5):
    if problem_type == "Quadratic":
        loss_fn, grad_fn = quad_loss, quad_grad
        w = np.array([10.0, 1.0], dtype=np.float64)
        f_star = 0.0
    elif problem_type == "Rosenbrock":
        loss_fn, grad_fn = rosen_loss, rosen_grad
        w = np.array([-1.5, 2.0], dtype=np.float64)
        f_star = 0.0
    else:
        n_dim = 100
        lambdas = np.linspace(-1.0, 1000.0, n_dim, dtype=np.float64)
        neg_mask = lambdas < 0.0
        f_star = float(np.sum(-0.25 * (lambdas[neg_mask] ** 2)))

        def loss_fn(x):
            return float(0.5 * np.sum(lambdas * (x**2)) + 0.25 * np.sum(x**4))

        def grad_fn(x):
            return lambdas * x + (x**3)

        w = np.where(neg_mask, 1e-2, 1.0).astype(np.float64)

    if opt_name == "AdamW_Native_BF16":
        w = _quantize_bf16_rne(w)

    m = np.zeros_like(w)
    v = np.zeros_like(w)

    for t in range(1, max_steps + 1):
        g = grad_fn(w)
        w, m, v = _step_optimizer(opt_name, w, g, m, v, t, lr, wd)
        curr_gap = abs(loss_fn(w) - f_star)

        if np.isnan(w).any() or np.isinf(w).any() or curr_gap > 1e8:
            return t, curr_gap, "DIVERGED"
        if curr_gap < tol:
            return t, curr_gap, "CONVERGED"

    return max_steps, abs(loss_fn(w) - f_star), "MAX_STEPS"


def run_benchmarks_2d() -> bool:
    print("\nStage II.1: Synthetic Benchmark Surfaces (B1 - B5)")
    optimizers = [
        "Legacy_BCLM",
        "LuminaV_Buffer1_TDCLM",
        "LuminaV_Buffer2_Dual",
        "AdamW_FP32_Master",
        "AdamW_Native_BF16",
    ]

    test_cases = [
        ("B1. Quadratic Bowl (kappa=1000)", "Quadratic", 0.02, 1000),
        ("B2. Rosenbrock Curved Valley", "Rosenbrock", 0.002, 2000),
        ("B5. High-Dim Saddle Escape (N=100)", "Saddle100", 0.02, 1500),
    ]

    b1_ok = False
    b2_ok = False
    b5_ok = False

    for title, prob, lr_val, steps_val in test_cases:
        print(f"\n{title} | LR: {lr_val}")
        for opt in optimizers:
            steps, final_gap, status = run_2d_benchmark(prob, opt, lr=lr_val, max_steps=steps_val)
            print(f"  {opt:<24} : {status:<9} | Steps: {steps:<4} | Loss Gap: {final_gap:.6e}")
            if prob == "Quadratic" and opt == "LuminaV_Buffer1_TDCLM" and status == "CONVERGED":
                b1_ok = True
            if prob == "Rosenbrock" and opt == "LuminaV_Buffer1_TDCLM" and final_gap < 1e-2:
                b2_ok = True
            if prob == "Saddle100" and opt in ("LuminaV_Buffer1_TDCLM", "LuminaV_Buffer2_Dual") and status == "CONVERGED":
                b5_ok = True

    return all([b1_ok, b2_ok, b5_ok])


if __name__ == "__main__":
    run_benchmarks_2d()
