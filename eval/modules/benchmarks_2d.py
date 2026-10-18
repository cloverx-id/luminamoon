try:
    from .common import *
except (ImportError, ValueError):
    from common import *
import numpy as np

def quad_loss(w):
    return 0.5 * (w[0]**2 + 1000.0 * w[1]**2)

def quad_grad(w):
    return np.array([w[0], 1000.0 * w[1]], dtype=np.float64)

def rosen_loss(w):
    return (1.0 - w[0])**2 + 100.0 * (w[1] - w[0]**2)**2

def rosen_grad(w):
    dw0 = -2.0 * (1.0 - w[0]) - 400.0 * w[0] * (w[1] - w[0]**2)
    dw1 = 200.0 * (w[1] - w[0]**2)
    return np.array([dw0, dw1], dtype=np.float64)

def run_2d_benchmark(problem_type, opt_name, lr=0.01, wd=0.0, max_steps=2000, tol=1e-5):
    if problem_type == "Quadratic":
        loss_fn, grad_fn = quad_loss, quad_grad
        w = np.array([10.0, 1.0], dtype=np.float64)
    else:
        loss_fn, grad_fn = rosen_loss, rosen_grad
        w = np.array([-1.5, 2.0], dtype=np.float64)

    m = np.zeros_like(w)
    v = np.zeros_like(w)
    beta1, beta2 = 0.9, 0.999
    lam, tau, eps = 0.3, 0.8, 1e-8

    for t in range(1, max_steps + 1):
        g = grad_fn(w)

        if opt_name == "Legacy_BCLM":
            w1 = beta1**2
            w2 = 1.0 - w1
            clm_m = w1 * m + w2 * g
            clm_env = w1 * np.abs(m) + w2 * np.abs(g) + eps
            u = clm_m / clm_env
            m = beta1 * m + (1.0 - beta1) * g
            w = w * (1.0 - lr * wd) - lr * u

        elif opt_name == "LuminaV_Buffer1_TDCLM":
            w1 = beta1**2
            w2 = 1.0 - w1
            clm_m = w1 * m + w2 * g
            clm_defect = lam * np.abs(g - m)
            clm_env = w1 * np.abs(m) + w2 * np.abs(g) + clm_defect + eps
            u = clm_m / clm_env
            m = beta1 * m + (1.0 - beta1) * g
            w = w * (1.0 - lr * wd) - lr * u

        elif opt_name == "LuminaV_Buffer2_Dual":
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
            w = w * (1.0 - lr * wd) - lr * u

        elif opt_name == "AdamW":
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g**2)
            m_hat = m / (1.0 - beta1**t)
            v_hat = v / (1.0 - beta2**t)
            u = m_hat / (np.sqrt(v_hat) + eps)
            w = w * (1.0 - lr * wd) - lr * u

        curr_loss = loss_fn(w)
        if np.isnan(w).any() or np.isinf(w).any() or curr_loss > 1e8:
            return t, curr_loss, "DIVERGED"
        if curr_loss < tol:
            return t, curr_loss, "CONVERGED"

    return max_steps, loss_fn(w), "MAX_STEPS"

def run_benchmarks_2d():
    print("\n--- [LSO-Eval] 2D Synthetic Benchmark (Quadratic & Rosenbrock) ---")
    test_cases = [
        ("Quadratic (kappa=1000)", "Quadratic", 0.02, 1000),
        ("Rosenbrock (Banana Valley)", "Rosenbrock", 0.002, 2000),
    ]
    optimizers = ["Legacy_BCLM", "LuminaV_Buffer1_TDCLM", "LuminaV_Buffer2_Dual", "AdamW"]

    for title, prob, lr_val, steps_val in test_cases:
        print(f"\nTask: {title} | LR: {lr_val}")
        print(f"{'Optimizer':<24} | {'Status':<10} | {'Steps':<6} | {'Final Loss'}")
        for opt in optimizers:
            steps, final_loss, status = run_2d_benchmark(prob, opt, lr=lr_val, max_steps=steps_val)
            print(f"{opt:<24} | {status:<10} | {steps:<6} | {final_loss:.6e}")

if __name__ == "__main__":
    run_benchmarks_2d()
