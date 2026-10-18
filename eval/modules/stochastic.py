try:
    from .common import *
except (ImportError, ValueError):
    from common import *
import numpy as np

def rastrigin_loss(w):
    return 20.0 + (w[0]**2 - 10.0 * np.cos(2.0 * np.pi * w[0])) + (w[1]**2 - 10.0 * np.cos(2.0 * np.pi * w[1]))

def rastrigin_grad(w):
    dw0 = 2.0 * w[0] + 20.0 * np.pi * np.sin(2.0 * np.pi * w[0])
    dw1 = 2.0 * w[1] + 20.0 * np.pi * np.sin(2.0 * np.pi * w[1])
    return np.array([dw0, dw1], dtype=np.float64)

def run_scenario_test(scenario_type, opt_name, lr=0.01, wd=0.0, steps=1000, noise_std=0.0):
    np.random.seed(42)
    if scenario_type == "Stochastic_Noise":
        def loss_fn(w): return 0.5 * (w[0]**2 + 1000.0 * w[1]**2)
        def grad_fn(w): return np.array([w[0], 1000.0 * w[1]], dtype=np.float64)
        w = np.array([10.0, 1.0], dtype=np.float64)
    else:
        loss_fn, grad_fn = rastrigin_loss, rastrigin_grad
        w = np.array([2.5, 2.5], dtype=np.float64)

    m = np.zeros_like(w)
    v = np.zeros_like(w)
    beta1, beta2 = 0.9, 0.999
    lam, tau, eps = 0.3, 0.8, 1e-8

    for t in range(1, steps + 1):
        g = grad_fn(w)
        if noise_std > 0.0:
            g = g + np.random.normal(0.0, noise_std, size=g.shape)

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

    return loss_fn(w)

def run_stochastic_scenarios():
    print("\n--- [LSO-Eval] Stochastic Noise & Multimodal Escape ---")
    opts = ["Legacy_BCLM", "LuminaV_Buffer1_TDCLM", "LuminaV_Buffer2_Dual", "AdamW"]

    print("\nScenario 1: Stochastic Noise (Gaussian std=1.0, LR=0.01)")
    print(f"{'Optimizer':<24} | {'Final Loss'}")
    for opt in opts:
        loss = run_scenario_test("Stochastic_Noise", opt, lr=0.01, steps=1000, noise_std=1.0)
        print(f"{opt:<24} | {loss:.6e}")

    print("\nScenario 2: Multimodal Escape (Rastrigin, LR=0.02)")
    print(f"{'Optimizer':<24} | {'Final Loss'}")
    for opt in opts:
        loss = run_scenario_test("Rastrigin", opt, lr=0.02, steps=1000, noise_std=0.0)
        print(f"{opt:<24} | {loss:.6e}")

if __name__ == "__main__":
    run_stochastic_scenarios()
