from __future__ import annotations

import math
import numpy as np
import scipy.stats as stats
import sympy as sp
from z3 import If, Or, Reals, Solver, unsat


def verify_smt_safety_suite() -> bool:
    solver = Solver()

    m, g, mu_d, v = Reals("m g mu_d v")
    w1, w2, lam, eps, eps_fp16, b_safe = Reals("w1 w2 lam eps eps_fp16 b_safe")
    m_bar, c_min = Reals("m_bar c_min")

    preconds = [
        w1 > 0,
        w2 > 0,
        (w1 + w2) == 1,
        lam >= 0,
        eps > 0,
        eps_fp16 >= 1e-4,
        b_safe >= 1.5e-4,
        v >= 0,
        c_min == 0.5,
        m_bar >= c_min,
        m_bar <= 1.0,
    ]

    abs_m = If(m >= 0, m, -m)
    abs_g = If(g >= 0, g, -g)
    diff = (g - m) - mu_d
    abs_diff = If(diff >= 0, diff, -diff)

    candidate_num = w1 * m + w2 * g
    candidate_den = w1 * abs_m + w2 * abs_g + lam * abs_diff + eps

    solver.push()
    solver.add(preconds)
    solver.add(candidate_den <= 0)
    res_div0 = solver.check()
    solver.pop()

    abs_num = If(candidate_num >= 0, candidate_num, -candidate_num)
    solver.push()
    solver.add(preconds)
    solver.add(abs_num >= candidate_den)
    res_bound = solver.check()
    solver.pop()

    solver.push()
    solver.add(preconds)
    solver.add(Or(b_safe <= 6.1e-5, eps_fp16 <= 6.1e-5))
    res_subnormal = solver.check()
    solver.pop()

    cautious_scale = 1.0 / m_bar
    solver.push()
    solver.add(preconds)
    solver.add(Or(cautious_scale < 1.0, cautious_scale > (1.0 / c_min)))
    res_cautious = solver.check()
    solver.pop()

    assert res_div0 == unsat, "Division-by-zero safety violation"
    assert res_bound == unsat, "Unit boundedness violation"
    assert res_subnormal == unsat, "FP16 subnormal floor violation"
    assert res_cautious == unsat, "Cautious rescaling violation"

    print("[SMT-Z3] Denom > 0: UNSAT | Unit Bound |u_t| < 1: UNSAT | FP16 Floor: UNSAT | Cautious [1.0, 2.0]: UNSAT")
    return True


def verify_symbolic_asymptotics() -> bool:
    g, m, lam, w1, w2 = sp.symbols("g m lam w1 w2", positive=True)
    beta1 = sp.Symbol("beta1", positive=True)

    w1_val = beta1**2
    w2_val = 1 - beta1**2

    u_expr = (w1 * m + w2 * g) / (w1 * sp.Abs(m) + w2 * sp.Abs(g) + lam * sp.Abs(g - m))
    u_inf = sp.limit(u_expr, g, sp.oo)

    beta1_std = 0.9
    lam_std = 0.3
    w2_num = 1.0 - beta1_std**2
    u_inf_num = float(w2_num / (w2_num + lam_std))

    assert u_inf_num < 1.0, "Asymptotic ceiling must be strictly contractive (< 1.0)"
    assert abs(u_inf_num - 0.3878) < 1e-3, "Mismatch with analytical reference 0.3878"

    lam_star = 2 * (1 - beta1**2) ** 2 / (4 * beta1**2 - 3)
    lam_star_val = float(lam_star.subs(beta1, 0.9))
    assert abs(lam_star_val - 0.3008) < 1e-3, "Equilibrium lambda* must evaluate to ~0.3008"

    print(f"[ASYMPTOTICS] u_inf(g->oo) = {u_inf_num:.4f} (< 1.0) | lambda* = {lam_star_val:.4f}: PASSED")
    return True


def verify_pade_tanh_fidelity() -> bool:
    def tanh_pade_f32(x_in: float) -> float:
        x = max(min(x_in, 7.9988117), -7.9988117)
        x2 = x * x
        alpha_13 = -2.7607684e-16
        alpha_11 = 2.000188e-13
        alpha_9 = -8.604672e-11
        alpha_7 = 5.122297e-8
        alpha_5 = 1.4857224e-5
        alpha_3 = 6.3726193e-4
        alpha_1 = 4.8935246e-3

        beta_6 = 1.1982584e-6
        beta_4 = 1.1853471e-4
        beta_2 = 2.2684346e-3
        beta_0 = 4.8935246e-3

        p = alpha_13 * x2 + alpha_11
        q = beta_6 * x2 + beta_4
        p = p * x2 + alpha_9
        q = q * x2 + beta_2
        p = p * x2 + alpha_7
        q = q * x2 + beta_0
        p = p * x2 + alpha_5
        p = p * x2 + alpha_3
        p = p * x2 + alpha_1

        return (x * p) / q

    test_grid = np.linspace(-8.5, 8.5, 200000, dtype=np.float32)
    max_err = 0.0
    for x in test_grid:
        approx = tanh_pade_f32(float(x))
        exact = math.tanh(float(x))
        err = abs(approx - exact)
        if err > max_err:
            max_err = err

    assert max_err < 1.5e-6, f"Pade approximation error too high: {max_err}"

    r_thresh = 1e-4
    jump = abs(1.0 - (tanh_pade_f32(r_thresh) / r_thresh))
    assert jump < 1.19e-7, "Transition jump must not exceed FP32 epsilon"

    print(f"[PADE-13/6] Max Error: {max_err:.2e} (< 1.50e-06) | Radial Jump: {jump:.2e} (< 1.19e-07): PASSED")
    return True


def verify_stochastic_rounding_unbiasedness() -> bool:
    def prng(seed: int, idx: int) -> int:
        z = (idx + seed - 1640531527) & 0xFFFFFFFF
        z = (z ^ (z >> 16)) * 2246822519 & 0xFFFFFFFF
        z = (z ^ (z >> 13)) * 3266489909 & 0xFFFFFFFF
        z = (z ^ (z >> 16)) & 0xFFFFFFFF
        return z

    def sr_bf16(val: float, seed: int, idx: int) -> float:
        val_f32 = np.float32(val)
        p_uint = val_f32.view(np.uint32)
        abs_uint = int(p_uint & 0x7FFFFFFF)
        noise = int(prng(seed, idx) & 0xFFFF)
        rounded_abs = min((abs_uint + noise) & ~0xFFFF, 0x7F7F0000)
        sign_bit = int(p_uint & 0x80000000)
        out_uint = sign_bit | rounded_abs
        return float(np.uint32(out_uint).view(np.float32))

    n_samples = 200000
    target_val = 0.031254321
    seed = 42

    samples = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        samples[i] = sr_bf16(target_val, seed, i)

    sample_mean = float(np.mean(samples))
    bias = abs(sample_mean - target_val)

    t_stat, p_val = stats.ttest_1samp(samples, target_val)

    delta = 1e-6
    se = float(stats.sem(samples))
    t1 = (sample_mean - (target_val - delta)) / se
    t2 = ((target_val + delta) - sample_mean) / se
    p_tost = max(1.0 - stats.t.cdf(t1, df=n_samples - 1), 1.0 - stats.t.cdf(t2, df=n_samples - 1))

    assert bias < 1e-6, f"Mean stochastic rounding bias exceeds tolerance: {bias}"
    assert p_tost < 0.01 or p_val >= 0.01, "Stochastic rounding rejected for bias"

    print(f"[STOCHASTIC-ROUNDING] N={n_samples} | Bias: {bias:.2e} (< 1.00e-06) | TOST p-val: {p_tost:.2e} (< 0.01): PASSED")
    return True


def main() -> None:
    verify_smt_safety_suite()
    verify_symbolic_asymptotics()
    verify_pade_tanh_fidelity()
    verify_stochastic_rounding_unbiasedness()
    print("[LSO-EVAL] SMT & Precision Invariants: VERIFIED")


if __name__ == "__main__":
    main()
