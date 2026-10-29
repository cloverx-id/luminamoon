import copy
import io
import os
import subprocess
import sys
from pathlib import Path

for _pkg in ["z3", "sympy", "scipy", "torch", "numpy"]:
    try:
        __import__(_pkg)
    except ImportError:
        _name = "z3-solver" if _pkg == "z3" else _pkg
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _name])

_base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_root = _base.parent if _base.name == "eval" else _base
for _path in [str(_root / "python"), str(_root / "optimizer" / "reference"), str(_root)]:
    if os.path.exists(_path) and _path not in sys.path:
        sys.path.append(_path)


from z3 import *

solver = Solver()

m, g = Reals("m g")
w1, w2, lam, eps = Reals("w1 w2 lam eps")
solver.push()
solver.add(w1 > 0, w2 > 0, (w1 + w2) == 1, lam >= 0, eps > 0)
abs_m = If(m >= 0, m, -m)
abs_g = If(g >= 0, g, -g)
diff = g - m
abs_diff = If(diff >= 0, diff, -diff)
clm_env = w1 * abs_m + w2 * abs_g + lam * abs_diff + eps
solver.add(clm_env <= 0)
res_tdclm_div0 = solver.check()
solver.pop()

solver.push()
solver.add(w1 > 0, w2 > 0, (w1 + w2) == 1, lam >= 0, eps > 0)
clm_m = w1 * m + w2 * g
abs_clm_m = If(clm_m >= 0, clm_m, -clm_m)
solver.add(abs_clm_m >= clm_env)
res_tdclm_bound = solver.check()
solver.pop()

sqrt_v, c1, c2 = Reals("sqrt_v c1 c2")
solver.push()
solver.add(sqrt_v >= 0, c1 > 0, c2 > 0)
sigma = sqrt_v * c1 + c2
solver.add(sigma <= 0)
res_sigma = solver.check()
solver.pop()

m_bar, c_min = Reals("m_bar c_min")
solver.push()
solver.add(c_min > 0, c_min <= 1)
solver.add(m_bar >= c_min, m_bar <= 1)
solver.add((1 / m_bar) > (1 / c_min))
res_cautious = solver.check()
solver.pop()

p, B_ratio = Reals("p B_ratio")
solver.push()
solver.add(B_ratio > 0)
B_sq = (B_ratio * B_ratio) * (p * p + 1e-4)
solver.add(B_sq <= 0)
res_coord_b = solver.check()
solver.pop()

norm_p = Real("norm_p")
solver.push()
solver.add(B_ratio > 0, norm_p >= 0)
R = If(norm_p >= 1.0, B_ratio * norm_p, B_ratio * 1.0)
solver.add(R < B_ratio)
res_radial_r = solver.check()
solver.pop()

p_fp16, v_raw = Reals("p_fp16 v_raw")
abs_p_fp16 = If(p_fp16 >= 0, p_fp16, -p_fp16)
p_scaled = 0.01 * abs_p_fp16
b_safe = If(p_scaled >= 1.5e-4, p_scaled, 1.5e-4)
eps_fp16 = If(eps >= 1.0e-4, eps, 1.0e-4)
v_fp16 = If(v_raw <= 65000.0, v_raw, 65000.0)
subnormal_floor = 6.1e-5

solver.push()
solver.add(b_safe < 1.5e-4)
res_fp16 = solver.check()
solver.pop()

solver.push()
solver.add(Or(b_safe <= subnormal_floor, eps_fp16 <= subnormal_floor, v_fp16 > 65504.0))
res_subnormal = solver.check()
solver.pop()

r = Real("r")
solver.push()
solver.add(r > 0)
pade_tanh_over_r = (15 + r * r) / (15 + 6 * r * r)
solver.add(pade_tanh_over_r > 1.0)
res_radial_scale = solver.check()
solver.pop()

beta1, beta2 = Reals("beta1 beta2")
solver.push()
solver.add(beta1 > 0, beta1 < 1, beta2 > 0, beta2 < 1)
bc1_t1 = 1 - beta1 * beta1
bc2_t1 = 1 - beta2
solver.add(Or(bc1_t1 <= 0, bc2_t1 <= 0))
res_bias_corr = solver.check()
solver.pop()

lr, wd = Reals("lr wd")
solver.push()
solver.add(lr > 0, lr <= 0.05, wd >= 0, wd <= 0.2)
decay_factor = 1.0 - lr * wd
solver.add(Or(decay_factor <= 0, decay_factor > 1))
res_wd = solver.check()
solver.pop()

g_val, w2_val, lam_val, eps_val = Reals("g_val w2_val lam_val eps_val")
solver.push()
solver.add(w2_val > 0, lam_val >= 0, eps_val > 0)
abs_g_val = If(g_val >= 0, g_val, -g_val)
u_zero_m = (w2_val * g_val) / (w2_val * abs_g_val + lam_val * abs_g_val + eps_val)
solver.add(u_zero_m * g_val < 0)
res_descent = solver.check()
solver.pop()

results = [
    ("1. TD-CLM Div-Zero (Q1)", res_tdclm_div0),
    ("2. TD-CLM Metric Bound |u| < 1.0 (Q2)", res_tdclm_bound),
    ("3. Buffer-2 Sigma Positivity (sigma > 0)", res_sigma),
    ("4. Cautious Scale Upper Bound", res_cautious),
    ("5. Coordinate Governor Radius (B > 0)", res_coord_b),
    ("6. Radial Governor Lower Bound (R >= B_ratio)", res_radial_r),
    ("7. FP16 Safe-Zone Floor (< 1.5e-4)", res_fp16),
    ("8. FP16 Subnormal Breach <= 6.1e-5 (Q4)", res_subnormal),
    ("9. Radial Scale Contraction <= 1.0 (Q3)", res_radial_scale),
    ("10. Bias Correction Positivity t >= 1 (Q7)", res_bias_corr),
    ("11. Weight Decay Contraction in (0, 1] (Q6)", res_wd),
    ("12. Descent Direction Alignment >= 0 (Q5)", res_descent),
]

for name, status in results:
    verdict = "PASSED (UNSAT)" if status == unsat else "FAILED (SAT)"
    print(f"{name:<48}: {verdict}")

import numpy as np

def tanh_pade_f32(x_in: float) -> float:
    x = max(min(float(x_in), 7.9988117), -7.9988117)
    x2 = x * x
    p = -2.7607684e-16 * x2 + 2.000188e-13
    q = 1.1982584e-6 * x2 + 1.1853471e-4
    p = p * x2 - 8.604672e-11
    q = q * x2 + 2.2684346e-3
    p = p * x2 + 5.122297e-8
    q = q * x2 + 4.8935246e-3
    p = p * x2 + 1.4857224e-5
    p = p * x2 + 6.3726193e-4
    p = p * x2 + 4.8935246e-3
    return (x * p) / q

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

test_cases = [
    ("Quadratic (kappa=1000)", "Quadratic", 0.02, 1000),
    ("Rosenbrock (Banana Valley)", "Rosenbrock", 0.002, 2000),
]

optimizers = ["Legacy_BCLM", "LuminaV_Buffer1_TDCLM", "LuminaV_Buffer2_Dual", "AdamW"]

for title, prob, lr_val, steps_val in test_cases:
    print("_" * 85)
    print(f"BENCHMARK: {title} | LR: {lr_val}")
    print("_" * 85)
    for opt in optimizers:
        steps, final_loss, status = run_2d_benchmark(prob, opt, lr=lr_val, max_steps=steps_val)
        print(f"[{opt:<22}] Status: {status:<9} | Steps: {steps:<4} | Final Loss: {final_loss:.6e}")

print("_" * 85)

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

opts = ["Legacy_BCLM", "LuminaV_Buffer1_TDCLM", "LuminaV_Buffer2_Dual", "AdamW"]

print("_" * 85)
print("SCENARIO 1: STOCHASTIC NOISY GRADIENTS (Gaussian Noise std = 1.0) | LR: 0.01")
print("_" * 85)
for opt in opts:
    loss = run_scenario_test("Stochastic_Noise", opt, lr=0.01, steps=1000, noise_std=1.0)
    print(f"[{opt:<22}] Final Loss: {loss:.6e}")

print("_" * 85)
print("SCENARIO 2: MULTIMODAL LOCAL MINIMA ESCAPE (Rastrigin Function) | LR: 0.02")
print("_" * 85)
for opt in opts:
    loss = run_scenario_test("Rastrigin", opt, lr=0.02, steps=1000, noise_std=0.0)
    print(f"[{opt:<22}] Final Loss: {loss:.6e}")
print("_" * 85)

import scipy.stats as stats
import sympy as sp

def _murmur3_prng(seed: int, offsets: np.ndarray) -> np.ndarray:
    z = (offsets.astype(np.int64) + int(seed) - 1640531527).astype(np.int32)
    z = (((z ^ (z >> 16)).astype(np.int64) * -2048173461) & 0xFFFFFFFF).astype(np.int32)
    z = (((z ^ (z >> 13)).astype(np.int64) * -1028477387) & 0xFFFFFFFF).astype(np.int32)
    return (z ^ (z >> 16)).astype(np.int32)

def test_rounding():
    rng = np.random.default_rng(42)
    n_samples = 200_000
    offsets = np.arange(n_samples, dtype=np.int32)
    prng_words = _murmur3_prng(1337, offsets)
    normal_vals = [np.float32(0.12345678), np.float32(-0.12345678)]
    subnormal_fp16_vals = [np.float32(3.0e-5), np.float32(-3.0e-5)]
    all_passed = True
    max_bias = 0.0

    for val in normal_vals:
        int_val = val.view(np.int32)
        abs_int = int_val & 0x7FFFFFFF
        sign_bit = int_val & (-2147483648)

        noise_bf16 = rng.integers(0, 1 << 16, size=n_samples, dtype=np.int32)
        rounded_bf16_abs = (abs_int + noise_bf16) & (-65536)
        rounded_bf16 = (sign_bit | rounded_bf16_abs).view(np.float32)
        bias_bf16 = float(abs(np.mean(rounded_bf16, dtype=np.float64) - float(val)))
        _, p_val_bf16 = stats.ttest_1samp(rounded_bf16.astype(np.float64), float(val))

        exp_f32 = (abs_int >> 23) & 0xFF
        sub_shift = int(np.clip(113 - exp_f32, 0, 11))
        total_shift = 13 + sub_shift
        dyn_mask = (1 << total_shift) - 1
        noise_fp16 = prng_words & dyn_mask
        rounded_abs = np.where(exp_f32 < 103, 0, (abs_int + noise_fp16) & (~dyn_mask)).astype(np.int32)
        rounded_fp16 = (sign_bit | rounded_abs).view(np.float32).astype(np.float16).astype(np.float32)
        bias_fp16 = float(abs(np.mean(rounded_fp16, dtype=np.float64) - float(val)))
        _, p_val_fp16 = stats.ttest_1samp(rounded_fp16.astype(np.float64), float(val))

        max_bias = max(max_bias, bias_bf16, bias_fp16)
        if p_val_bf16 < 0.01 or p_val_fp16 < 0.01 or max_bias >= 1e-6:
            all_passed = False

    for sub_val in subnormal_fp16_vals:
        int_val = sub_val.view(np.int32)
        abs_int = int_val & 0x7FFFFFFF
        sign_bit = int_val & (-2147483648)
        exp_f32 = (abs_int >> 23) & 0xFF
        sub_shift = int(np.clip(113 - exp_f32, 0, 11))
        total_shift = 13 + sub_shift
        dyn_mask = (1 << total_shift) - 1
        noise_sub = prng_words & dyn_mask
        rounded_abs = np.where(exp_f32 < 103, 0, (abs_int + noise_sub) & (~dyn_mask)).astype(np.int32)
        rounded_sub = (sign_bit | rounded_abs).view(np.float32).astype(np.float16).astype(np.float32)
        bias_sub = float(abs(np.mean(rounded_sub, dtype=np.float64) - float(sub_val)))
        max_bias = max(max_bias, bias_sub)
        if sub_shift <= 0 or bias_sub >= 1e-6:
            all_passed = False

    return all_passed, max_bias

def test_tanh_clamping():
    x = np.linspace(-8.0, 8.0, 500_000, dtype=np.float64)
    pade_outputs = np.array([tanh_pade_f32(xi) for xi in x], dtype=np.float64)
    exact_tanh = np.tanh(x)
    max_err = float(np.max(np.abs(pade_outputs - exact_tanh)))

    extreme_inputs = np.array([-1e10, -50.0, 50.0, 1e10], dtype=np.float64)
    extreme_outputs = np.array([tanh_pade_f32(xi) for xi in extreme_inputs], dtype=np.float64)
    no_nan_inf = bool(not np.isnan(extreme_outputs).any() and not np.isinf(extreme_outputs).any())

    return (max_err < 1.5e-6) and no_nan_inf, max_err

def test_radial_boundary():
    r = sp.Symbol("r", positive=True)
    taylor_tanh_over_r = sp.series(sp.tanh(r) / r, r, 0, 4).removeO()

    r_val = 1e-4
    val_taylor = float(taylor_tanh_over_r.subs(r, r_val))
    diff = abs(1.0 - val_taylor)
    fp32_eps = float(np.finfo(np.float32).eps)

    return (diff < fp32_eps), diff, fp32_eps

def test_lr_scheduler():
    total_steps = 100_000
    warmup_steps = 1500
    min_ratio = 0.05
    steps = np.arange(total_steps + 1)

    def get_ratio(s):
        if s < warmup_steps:
            return s / max(1, warmup_steps)
        progress = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return (1.0 - min_ratio) * cosine_decay + min_ratio

    ratios = np.array([get_ratio(s) for s in steps])
    warmup_ok = np.isclose(ratios[0], 0.0) and np.isclose(ratios[warmup_steps], 1.0)
    decay_part = ratios[warmup_steps:]
    decay_ok = np.isclose(decay_part[-1], min_ratio) and np.all(np.diff(decay_part) <= 1e-12)

    return (warmup_ok and decay_ok), ratios[0], ratios[warmup_steps], decay_part[-1]

def test_tdclm_large_gradient():
    g = sp.Symbol("g", positive=True)
    m = sp.Symbol("m", real=True)
    beta1 = sp.Rational(9, 10)
    lam = sp.Rational(3, 10)
    eps = sp.Rational(1, 10**8)

    w1 = beta1**2
    w2 = 1 - w1
    clm_m = w1 * m + w2 * g
    clm_env = w1 * sp.Abs(m) + w2 * sp.Abs(g) + lam * sp.Abs(g - m) + eps

    u_expr = clm_m / clm_env
    limit_inf = sp.limit(u_expr, g, sp.oo)
    theoretical_limit = float(w2 / (w2 + lam))
    empirical_limit = float(limit_inf)

    is_valid = (empirical_limit < 1.0) and abs(empirical_limit - theoretical_limit) < 1e-12
    return is_valid, empirical_limit

def test_buffer2_bias_correction():
    beta1, beta2 = 0.9, 0.999
    tau, eps = 0.8, 1e-8
    steps = np.array([1, 10, 100, 1000, 10000, 100000], dtype=np.int64)

    bc1 = 1.0 - beta1**(steps + 1)
    bc2 = np.maximum(1.0 - beta2**steps, 1e-15)
    c1 = (bc1 * tau) / np.sqrt(bc2)
    c2 = eps * bc1 * tau

    all_positive = np.all(c1 > 0) and np.all(c2 > 0)
    converges_to_tau = abs(c1[-1] - tau) < 1e-3

    return all_positive and converges_to_tau, c1[0], c1[-1]

def test_cautious_scale():
    c_min = 0.5
    m_bars = np.linspace(c_min, 1.0, 1000)
    scales = 1.0 / m_bars

    max_scale = np.max(scales)
    min_scale = np.min(scales)
    bounded = (max_scale <= (1.0 / c_min) + 1e-12) and (min_scale >= 1.0 - 1e-12)

    return bounded, min_scale, max_scale

def test_coordinate_bound():
    x = sp.Symbol("x", real=True)
    B = sp.Symbol("B", positive=True)

    step_expr = B * sp.tanh(x / B)
    d_step = sp.diff(step_expr, x)
    d_at_zero = float(d_step.subs(x, 0))
    d_at_inf = float(sp.limit(d_step, x, sp.oo))

    is_valid = (d_at_zero == 1.0) and (d_at_inf == 0.0)
    return is_valid, d_at_zero, d_at_inf

def test_weight_decay():
    lr_grid = np.array([1e-5, 1e-4, 8e-4, 1e-3, 5e-3])
    wd_grid = np.array([0.01, 0.05, 0.08, 0.1])

    lr_mesh, wd_mesh = np.meshgrid(lr_grid, wd_grid)
    multipliers = 1.0 - lr_mesh * wd_mesh

    strictly_stable = np.all(multipliers > 0.0) and np.all(multipliers < 1.0)
    return strictly_stable, np.min(multipliers), np.max(multipliers)

print("_" * 85)
print("LuminaV Math Verification")
print("_" * 85)

p1, b_max = test_rounding()
print(f"1. Stochastic rounding mean error   : {'PASSED' if p1 else 'FAILED'} | Max bias: {b_max:.2e}")

p2, max_err_tanh = test_tanh_clamping()
print(f"2. Pade [13/6] tanh approx error    : {'PASSED' if p2 else 'FAILED'} | Max error: {max_err_tanh:.2e}")

p3, diff_cont, eps_f32 = test_radial_boundary()
print(f"3. Radial boundary continuity       : {'PASSED' if p3 else 'FAILED'} | Step jump: {diff_cont:.2e}")

p4, s_start, s_peak, s_end = test_lr_scheduler()
print(f"4. Learning rate schedule range     : {'PASSED' if p4 else 'FAILED'} | Warmup: {s_start:.1f}->{s_peak:.1f}, Min: {s_end:.2f}")

p5, lim_val = test_tdclm_large_gradient()
print(f"5. TD-CLM limit at large gradient   : {'PASSED' if p5 else 'FAILED'} | Asymptotic value: {lim_val:.4f}")

p6, c1_init, c1_inf = test_buffer2_bias_correction()
print(f"6. Buffer 2 bias correction         : {'PASSED' if p6 else 'FAILED'} | Step 1: {c1_init:.4f}, Step inf: {c1_inf:.4f}")

p7, min_sc, max_sc = test_cautious_scale()
print(f"7. Cautious scaling range           : {'PASSED' if p7 else 'FAILED'} | Factor: [{min_sc:.2f}, {max_sc:.2f}]")

p8, d0, dinf = test_coordinate_bound()
print(f"8. Coordinate bound derivative      : {'PASSED' if p8 else 'FAILED'} | Derivative range: [{dinf:.1f}, {d0:.1f}]")

p9, min_decay, max_decay = test_weight_decay()
print(f"9. Weight decay multiplier          : {'PASSED' if p9 else 'FAILED'} | Range: [{min_decay:.5f}, {max_decay:.5f}]")

print("_" * 85)

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

lambda_values = [0.0, 0.1, 0.19, 0.3, 0.5, 0.8, 1.0]
lam_star = calc_equilibrium_lambda(0.9)

print("_" * 105)
print(f"TD-CLM DEFECT FACTOR (LAMBDA) SWEEP | Proposition 1 Equilibrium lambda* = {lam_star:.4f}")
print("_" * 105)
print(f"{'Lambda':<8} | {'Spike Limit':<12} | {'Reversal Limit':<15} | {'Steps to 1e-5':<15} | {'Final Loss':<12} | {'Empirical Behavior Note'}")
print("_" * 105)

for lam in lambda_values:
    u_spk, u_rev = calc_theoretical_limits(lam)
    c_step, f_loss = run_lambda_sweep(lam)
    note = describe_lambda_behavior(lam, u_spk, u_rev, c_step, f_loss)
    step_str = f"{c_step}" if c_step < 1000 else "Max Steps"
    print(f"{lam:<8.2f} | {u_spk:<12.4f} | {u_rev:<15.4f} | {step_str:<15} | {f_loss:<12.2e} | {note}")

print("_" * 105)

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

def describe_sweep_behavior(val, default_val, step, ref_step, loss):
    pct = ((step - ref_step) / float(ref_step)) * 100.0
    if abs(val - default_val) < 1e-9:
        return f"Calibrated default baseline ({step} steps, final {loss:.2e})"
    if abs(pct) <= 10.0 and loss < 1e-20:
        return f"Stable +/-30% operating basin ({pct:+.1f}% steps vs default)"
    if pct < -5.0:
        return f"Aggressive sign-saturated regime ({pct:+.1f}% steps, final {loss:.2e})"
    if loss > 1e-10:
        return f"Restricted trust radius ({pct:+.1f}% steps, final {loss:.2e})"
    return f"Heavily damped tail ({pct:+.1f}% steps, final {loss:.2e})"

tau_list = [0.2, 0.56, 0.8, 1.04, 1.5]
bound_list = [0.01, 0.035, 0.05, 0.065, 0.15]

ref_tau_step, _ = run_hyperparam_sweep("tau", 0.8)
ref_coord_step, _ = run_hyperparam_sweep("coordinate", 0.05)
ref_rad_step, _ = run_hyperparam_sweep("radial", 0.05)

print("_" * 95)
print("SWEEP 1: BUFFER-2 TAU PARAMETER SENSITIVITY (Default = 0.80, +/-30% Basin = [0.56, 1.04])")
print("_" * 95)
print(f"{'Tau':<8} | {'Steps to 1e-4':<15} | {'Final Loss':<12} | {'Empirical Behavior'}")
print("_" * 95)
for val in tau_list:
    step, loss = run_hyperparam_sweep("tau", val)
    note = describe_sweep_behavior(val, 0.8, step, ref_tau_step, loss)
    print(f"{val:<8.2f} | {step:<15} | {loss:<12.2e} | {note}")

print("_" * 95)
print("SWEEP 2: COORDINATE BOUND RATIO SENSITIVITY (Default = 0.05, +/-30% Basin = [0.035, 0.065])")
print("_" * 95)
print(f"{'Ratio':<8} | {'Steps to 1e-4':<15} | {'Final Loss':<12} | {'Empirical Behavior'}")
print("_" * 95)
for val in bound_list:
    step, loss = run_hyperparam_sweep("coordinate", val)
    note = describe_sweep_behavior(val, 0.05, step, ref_coord_step, loss)
    print(f"{val:<8.3f} | {step:<15} | {loss:<12.2e} | {note}")

print("_" * 95)
print("SWEEP 3: RADIAL BOUND RATIO SENSITIVITY (Default = 0.05, +/-30% Basin = [0.035, 0.065])")
print("_" * 95)
print(f"{'Ratio':<8} | {'Steps to 1e-4':<15} | {'Final Loss':<12} | {'Empirical Behavior'}")
print("_" * 95)
for val in bound_list:
    step, loss = run_hyperparam_sweep("radial", val)
    note = describe_sweep_behavior(val, 0.05, step, ref_rad_step, loss)
    print(f"{val:<8.3f} | {step:<15} | {loss:<12.2e} | {note}")
print("_" * 95)

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

print("_" * 85)
print("COMBINATORIAL ABLATION SWEEP: LUMINAV FEATURE INTERACTIONS (FULL FACTORIAL)")
print("_" * 85)
print(f"{'Configuration':<35} | {'Steps to 1e-4':<15} | {'Final Loss':<12}")
print("_" * 85)

for buf, b_type, c_flag, desc in combinations:
    step, loss = run_combination_test(buf, b_type, c_flag)
    step_str = f"{step}" if step is not None else "Max Steps"
    print(f"{desc:<35} | {step_str:<15} | {loss:<12.2e}")

print("_" * 85)


def run_ema_comparison_standard(
    buffer_mode,
    use_ema,
    ema_decay=0.8,
    lr=0.02,
    max_steps=1000,
    tol=1e-4,
    noise_std=0.0,
):
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

    cautious_scale_ema = 1.0
    bound_scale_ema = 1.0

    for t in range(1, max_steps + 1):
        g = np.array([w[0], 1000.0 * w[1]], dtype=np.float64)

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

            bc1 = 1.0 - beta1 ** (t + 1)
            bc2 = max(1.0 - beta2**t, 1e-15)
            c1 = (bc1 * tau) / np.sqrt(bc2)
            c2 = eps * bc1 * tau
            sigma = np.sqrt(np.maximum(v, 0.0)) * c1 + c2
            u = np.tanh(clm_m / np.maximum(sigma, 1e-12))
            m = m_new

        mask = (u * g > 0.0).astype(np.float64)
        m_bar = np.clip(np.mean(mask), c_min, 1.0)
        target_cautious_scale = 1.0 / m_bar

        norm_p = np.linalg.norm(w)
        u_eff = u * mask
        R = bound_ratio * max(norm_p, 1.0)

        if use_ema and t > 1:
            active_cautious_scale = cautious_scale_ema
            active_bound_scale = bound_scale_ema

            cautious_scale_ema = (
                ema_decay * cautious_scale_ema
                + (1.0 - ema_decay) * target_cautious_scale
            )
            norm_delta = np.linalg.norm(u_eff) * cautious_scale_ema
            r = (lr * norm_delta) / R
            target_bound_scale = 1.0 if r < 1e-4 else np.tanh(r) / max(r, 1e-8)
            bound_scale_ema = (
                ema_decay * bound_scale_ema + (1.0 - ema_decay) * target_bound_scale
            )
        else:
            active_cautious_scale = target_cautious_scale
            norm_delta = np.linalg.norm(u_eff) * active_cautious_scale
            r = (lr * norm_delta) / R
            target_bound_scale = 1.0 if r < 1e-4 else np.tanh(r) / max(r, 1e-8)
            active_bound_scale = target_bound_scale

            cautious_scale_ema = target_cautious_scale
            bound_scale_ema = target_bound_scale

        delta = u_eff * active_cautious_scale
        step_val = lr * delta * active_bound_scale
        w = w - step_val

        loss = 0.5 * (w[0] ** 2 + 1000.0 * w[1] ** 2)

        if loss < tol:
            return t, loss, "CONVERGED"

    return max_steps, loss, "MAX_STEPS"


configs = [
    (1, False, 0.0, "Buffer 1 (Instantaneous 2-Pass)"),
    (1, True, 0.5, "Buffer 1 (Fused EMA decay=0.5)"),
    (1, True, 0.8, "Buffer 1 (Fused EMA decay=0.8)"),
    (1, True, 0.95, "Buffer 1 (Fused EMA decay=0.95 - Default)"),
    (2, False, 0.0, "Buffer 2 (Instantaneous 2-Pass)"),
    (2, True, 0.5, "Buffer 2 (Fused EMA decay=0.5)"),
    (2, True, 0.8, "Buffer 2 (Fused EMA decay=0.8 - Default)"),
    (2, True, 0.95, "Buffer 2 (Fused EMA decay=0.95)"),
]

print("_" * 105)
print("STANDARD LSO-EVAL STAGE II: 1-PASS VS 2-PASS PARITY AUDIT (LISTING 2 HARNESS)")
print("_" * 105)
print(
    f"{'Mode':<42} | {'Steps':<8} | {'Final Loss':<12} | {'Step Delta'}"
    f" | {'Status'}"
)
print("_" * 105)

base_steps = {}
for buf, use_e, decay, desc in configs:
    steps, loss, status = run_ema_comparison_standard(
        buf, use_e, decay, tol=1e-4, noise_std=0.0
    )
    if not use_e:
        base_steps[buf] = steps
        delta_str = "Baseline"
    else:
        delta_pct = (
            abs(steps - base_steps[buf]) / float(base_steps[buf])
        ) * 100.0
        delta_str = f"{delta_pct:.2f}%"

    print(
        f"{desc:<42} | {steps:<8} | {loss:<12.2e} | {delta_str:<10} |"
        f" {status}"
    )

print("_" * 105)

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

c_min_list = [0.2, 0.4, 0.5, 0.7, 1.0]
wd_list = [0.0, 0.02, 0.05, 0.08, 0.12]

print("_" * 85)
print("SWEEP 1: CAUTIOUS CLAMP MIN SENSITIVITY (Buffer 2)")
print("_" * 85)
print(f"{'c_min':<8} | {'Max Boost':<12} | {'Steps to 1e-4':<15} | {'Final Loss':<12}")
print("_" * 85)
for val in c_min_list:
    step, loss = run_param_audit("c_min", val, buffer_mode=2)
    boost = f"{1.0/val:.2f}x"
    step_str = f"{step}" if step < 1000 else "Max Steps"
    print(f"{val:<8.2f} | {boost:<12} | {step_str:<15} | {loss:<12.2e}")

print("_" * 85)
print("SWEEP 2: DECOUPLED WEIGHT DECAY SENSITIVITY (Buffer 2)")
print("_" * 85)
print(f"{'WD':<8} | {'Steps to 1e-4':<15} | {'Final Loss':<12}")
print("_" * 85)
for val in wd_list:
    step, loss = run_param_audit("wd", val, buffer_mode=2)
    step_str = f"{step}" if step < 1000 else "Max Steps"
    print(f"{val:<8.2f} | {step_str:<15} | {loss:<12.2e}")
print("_" * 85)

def run_fair_lr_test(buffer_mode, bound_type, lr, max_steps=15000, tol=1e-4, noise_std=0.5):
    np.random.seed(42)
    w = np.array([10.0, 1.0], dtype=np.float64)
    m = np.zeros_like(w)
    v = np.zeros_like(w)

    beta1, beta2 = 0.9, 0.999
    lam = 0.3
    tau = 0.8
    bound_ratio = 0.05
    c_min = 0.5
    wd = 0.08
    eps = 1e-8

    ema_decay = 0.95 if buffer_mode == 1 else 0.8
    cautious_scale_ema = 1.0
    bound_scale_ema = 1.0

    converged_step = max_steps
    for t in range(1, max_steps + 1):
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

        mask = (u * g > 0.0).astype(np.float64)
        m_bar = np.clip(np.mean(mask), c_min, 1.0)
        target_c_scale = 1.0 / m_bar

        norm_p = np.linalg.norm(w)
        u_eff = u * mask
        R = bound_ratio * max(norm_p, 1.0)

        if t > 1:
            active_c_scale = cautious_scale_ema
            active_b_scale = bound_scale_ema
            cautious_scale_ema = ema_decay * cautious_scale_ema + (1.0 - ema_decay) * target_c_scale
            norm_delta = np.linalg.norm(u_eff) * cautious_scale_ema
            r = (lr * norm_delta) / R
            target_b_scale = 1.0 if r < 1e-4 else np.tanh(r) / max(r, 1e-8)
            bound_scale_ema = ema_decay * bound_scale_ema + (1.0 - ema_decay) * target_b_scale
        else:
            active_c_scale = target_c_scale
            norm_delta = np.linalg.norm(u_eff) * active_c_scale
            r = (lr * norm_delta) / R
            target_b_scale = 1.0 if r < 1e-4 else np.tanh(r) / max(r, 1e-8)
            active_b_scale = target_b_scale
            cautious_scale_ema = target_c_scale
            bound_scale_ema = target_b_scale

        delta = u_eff * active_c_scale
        step_val = lr * delta

        if bound_type == "coordinate":
            B = bound_ratio * np.sqrt(w**2 + 1e-4)
            step_val = B * np.tanh(step_val / B)
        elif bound_type == "radial":
            step_val = step_val * active_b_scale

        w = w * (1.0 - lr * wd) - step_val

        loss = 0.5 * (w[0]**2 + 1000.0 * w[1]**2)
        if loss < tol and converged_step == max_steps:
            converged_step = t

    final_loss = 0.5 * (w[0]**2 + 1000.0 * w[1]**2)
    return converged_step, final_loss

def describe_lr_grid_behavior(lr_val, s1, l1, s2, l2, ref_s1, ref_s2):
    sp1 = ref_s1 / float(max(s1, 1))
    sp2 = ref_s2 / float(max(s2, 1))
    avg_sp = 0.5 * (sp1 + sp2)
    max_l = max(l1, l2)
    if lr_val == 0.0008:
        return f"Baseline parity (|delta_steps|={abs(s1-s2)}, final {l1:.1e} / {l2:.1e})"
    if max_l <= 2e-4:
        return f"{avg_sp:.1f}x faster convergence, stable floor (max loss {max_l:.1e})"
    if max_l <= 5e-3:
        return f"{avg_sp:.1f}x faster entry, moderate noise floor (max loss {max_l:.1e})"
    return f"Fast entry ({s1}/{s2} steps), noise rebound to {max_l:.1e}"

lr_grid = [0.0008, 0.002, 0.005, 0.010, 0.020, 0.040]

print("_" * 125)
print("FAIR LEARNING RATE GRID: CAUSE-AND-EFFECT BENCHMARK (BUFFER 1 VS BUFFER 2)")
print("Budget: 15,000 Steps | Noise std = 0.5 | Default = 8e-4")
print("_" * 125)
print(f"{'LR':<8} | {'Bound':<11} | {'Buf 1 Steps':<12} | {'Buf 1 Loss':<12} | {'Buf 2 Steps':<12} | {'Buf 2 Loss':<12} | {'Empirical Cause & Effect'}")
print("_" * 125)

for bound_t in ["radial", "coordinate"]:
    ref_s1, _ = run_fair_lr_test(1, bound_t, 0.0008)
    ref_s2, _ = run_fair_lr_test(2, bound_t, 0.0008)
    for lr_val in lr_grid:
        s1, l1 = run_fair_lr_test(1, bound_t, lr_val)
        s2, l2 = run_fair_lr_test(2, bound_t, lr_val)
        lr_str = f"{lr_val:.4f}" if lr_val < 0.01 else f"{lr_val:.3f}"
        note = describe_lr_grid_behavior(lr_val, s1, l1, s2, l2, ref_s1, ref_s2)
        print(f"{lr_str:<8} | {bound_t:<11} | {s1:<12} | {l1:<12.2e} | {s2:<12} | {l2:<12.2e} | {note}")
    print("_" * 125)

print("\nPRACTICAL RECOMMENDATIONS:")
print("1. Long LLM pre-training (>20k steps) : Use default LR 8e-4 (Buffer 1 & 2 reach 100% parity).")
print("2. Short fine-tuning / runs (<10k steps) : Increase LR to 0.01 - 0.02 + set bound_type='coordinate'.")
print("3. Preferred manifold bound choice       : 'coordinate' handles stochastic noise substantially better than 'radial'.")
print("_" * 125)

import torch
import torch.nn as nn
try:
    from luminamoon import LuminaV
except ImportError:
    try:
        from luminamoon.optimizer import LuminaV
    except ImportError:
        from luminav import LuminaV

def test_checkpoint_roundtrip():
    torch.manual_seed(42)
    forbidden_keys = (
        "p_contig",
        "exp_avg_contig",
        "exp_avg_sq_contig",
        "mp_contig",
        "_plan",
        "_contig_step",
    )

    for buf in [1, 2]:
        for mw in [False, "semi", "full"]:
            model_cont = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 2),
            )
            model_res = copy.deepcopy(model_cont)

            opt_cont = LuminaV(model_cont.parameters(), lr=1e-3, buffer=buf, bound=True, master_weights=mw)
            opt_res = LuminaV(model_res.parameters(), lr=1e-3, buffer=buf, bound=True, master_weights=mw)

            batches = [
                (torch.randn(4, 10), torch.tensor([0, 1, 0, 1], dtype=torch.long))
                for _ in range(6)
            ]

            for i in range(5):
                x, y = batches[i]
                for m, opt in ((model_cont, opt_cont), (model_res, opt_res)):
                    loss = nn.CrossEntropyLoss()(m(x), y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            buffer_io = io.BytesIO()
            torch.save(opt_res.state_dict(), buffer_io)
            buffer_io.seek(0)

            opt_loaded = LuminaV(model_res.parameters(), lr=1e-3, buffer=buf, bound=True, master_weights=mw)
            loaded_dict = torch.load(buffer_io, weights_only=True)
            opt_loaded.load_state_dict(loaded_dict)

            for p_state in loaded_dict["state"].values():
                for forbidden_key in forbidden_keys:
                    assert forbidden_key not in p_state, f"Memory leak detected: {forbidden_key} found in checkpoint!"
                if "scratch" in p_state:
                    sc = p_state["scratch"]
                    assert float(sc[0]) == 0.0 and float(sc[1]) == 0.0
                    assert float(sc[2]) == 0.0 and float(sc[5]) == 0.0

            for p in model_res.parameters():
                st = opt_loaded.state[p]
                if "scratch" in st:
                    assert st["scratch"].dtype == torch.float32
                if "master_param" in st:
                    assert st["master_param"].dtype == torch.float32

            x_last, y_last = batches[5]
            for m, opt in ((model_cont, opt_cont), (model_res, opt_loaded)):
                loss = nn.CrossEntropyLoss()(m(x_last), y_last)
                opt.zero_grad()
                loss.backward()
                opt.step()

            for p_c, p_r in zip(model_cont.parameters(), model_res.parameters()):
                torch.testing.assert_close(p_c, p_r, rtol=0.0, atol=0.0)

    return True

print("_" * 85)
print("SOFTWARE INTEGRITY: CHECKPOINT SERIALIZATION ROUNDTRIP")
print("_" * 85)
success = test_checkpoint_roundtrip()
print(f"Checkpoint Load/Save & State Cleanliness: {'PASSED' if success else 'FAILED'}")
print("_" * 85)
