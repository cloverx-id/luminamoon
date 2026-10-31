try:
    from .common import _quantize_bf16_rne, _step_optimizer
except (ImportError, ValueError):
    from common import _quantize_bf16_rne, _step_optimizer

import numpy as np


def quad_loss(w):
    return 0.5 * (w[0] ** 2 + 1000.0 * w[1] ** 2)


def quad_grad(w):
    return np.array([w[0], 1000.0 * w[1]], dtype=np.float64)


def rastrigin_loss(w):
    return 20.0 + (w[0] ** 2 - 10.0 * np.cos(2.0 * np.pi * w[0])) + (w[1] ** 2 - 10.0 * np.cos(2.0 * np.pi * w[1]))


def rastrigin_grad(w):
    dw0 = 2.0 * w[0] + 20.0 * np.pi * np.sin(2.0 * np.pi * w[0])
    dw1 = 2.0 * w[1] + 20.0 * np.pi * np.sin(2.0 * np.pi * w[1])
    return np.array([dw0, dw1], dtype=np.float64)


def run_scenario_test(scenario_type, opt_name, lr=0.01, wd=0.0, steps=1000, noise_std=0.0):
    np.random.seed(42)
    if scenario_type == "Stochastic_Noise":
        loss_fn, grad_fn = quad_loss, quad_grad
        w = np.array([10.0, 1.0], dtype=np.float64)
    else:
        loss_fn, grad_fn = rastrigin_loss, rastrigin_grad
        w = np.array([2.5, 2.5], dtype=np.float64)

    if opt_name == "AdamW_Native_BF16":
        w = _quantize_bf16_rne(w)

    m = np.zeros_like(w)
    v = np.zeros_like(w)

    for t in range(1, steps + 1):
        g = grad_fn(w)
        if noise_std > 0.0:
            g = g + np.random.normal(0.0, noise_std, size=g.shape)
        w, m, v = _step_optimizer(opt_name, w, g, m, v, t, lr, wd)

    return float(loss_fn(w))


def run_stochastic_scenarios() -> bool:
    optimizers = [
        "Legacy_BCLM",
        "LuminaV_Buffer1_TDCLM",
        "LuminaV_Buffer2_Dual",
        "AdamW_FP32_Master",
        "AdamW_Native_BF16",
    ]

    print("\nB3. Stochastic Gradient Noise (std=1.0) | LR: 0.01")
    b3_losses = {}
    for opt in optimizers:
        loss = run_scenario_test("Stochastic_Noise", opt, lr=0.01, steps=1000, noise_std=1.0)
        b3_losses[opt] = loss
        print(f"  {opt:<24} : Final Loss: {loss:.6e}")

    print("\nB4. Multimodal Rastrigin Escape | LR: 0.02")
    b4_losses = {}
    for opt in optimizers:
        loss = run_scenario_test("Rastrigin", opt, lr=0.02, steps=1000, noise_std=0.0)
        b4_losses[opt] = loss
        print(f"  {opt:<24} : Final Loss: {loss:.6e}")

    b3_ok = (
        b3_losses["LuminaV_Buffer1_TDCLM"] < b3_losses["AdamW_FP32_Master"]
        and b3_losses["LuminaV_Buffer2_Dual"] < b3_losses["AdamW_FP32_Master"]
    )
    b4_ok = min(b4_losses["LuminaV_Buffer1_TDCLM"], b4_losses["LuminaV_Buffer2_Dual"]) <= b4_losses["AdamW_FP32_Master"] + 1e-5
    return b3_ok and b4_ok


if __name__ == "__main__":
    run_stochastic_scenarios()
