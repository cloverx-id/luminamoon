try:
    from .common import *
except (ImportError, ValueError):
    from common import *
import numpy as np
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

def run_math_verification():
    print("\n--- [LSO-Eval] Mathematical Verification Suite ---")

    p1, b_max = test_rounding()
    print(f"1. Stochastic Rounding Unbiasedness : {'PASSED' if p1 else 'FAILED'} (Max Bias: {b_max:.2e})")

    p2, max_err_tanh = test_tanh_clamping()
    print(f"2. Pade [13/6] Tanh Error          : {'PASSED' if p2 else 'FAILED'} (Max Error: {max_err_tanh:.2e})")

    p3, diff_cont, _ = test_radial_boundary()
    print(f"3. Radial Boundary Continuity       : {'PASSED' if p3 else 'FAILED'} (Step Jump: {diff_cont:.2e})")

    p4, s_start, s_peak, s_end = test_lr_scheduler()
    print(f"4. Learning Rate Schedule Range     : {'PASSED' if p4 else 'FAILED'} ({s_start:.1f} -> {s_peak:.1f}, Min: {s_end:.2f})")

    p5, lim_val = test_tdclm_large_gradient()
    print(f"5. TD-CLM Large-Gradient Limit      : {'PASSED' if p5 else 'FAILED'} (Asymptotic: {lim_val:.4f})")

    p6, c1_init, c1_inf = test_buffer2_bias_correction()
    print(f"6. Buffer-2 Bias Correction         : {'PASSED' if p6 else 'FAILED'} (Step 1: {c1_init:.4f}, Step inf: {c1_inf:.4f})")

    p7, min_sc, max_sc = test_cautious_scale()
    print(f"7. Cautious Scaling Range           : {'PASSED' if p7 else 'FAILED'} ([{min_sc:.2f}, {max_sc:.2f}])")

    p8, d0, dinf = test_coordinate_bound()
    print(f"8. Coordinate Bound Derivative      : {'PASSED' if p8 else 'FAILED'} ([{dinf:.1f}, {d0:.1f}])")

    p9, min_decay, max_decay = test_weight_decay()
    print(f"9. Weight Decay Multiplier          : {'PASSED' if p9 else 'FAILED'} ([{min_decay:.5f}, {max_decay:.5f}])")

if __name__ == "__main__":
    run_math_verification()
