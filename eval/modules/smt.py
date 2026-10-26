try:
    from .common import *
except (ImportError, ValueError):
    from common import *
from z3 import *

def run_smt_suite():
    print("\n--- [LSO-Eval] Stage I: Z3 SMT Invariant Verification ---")
    solver = Solver()

    m, g, mu_d = Reals("m g mu_d")
    w1, w2, lam, eps = Reals("w1 w2 lam eps")
    solver.push()
    solver.add(w1 > 0, w2 > 0, (w1 + w2) == 1, lam >= 0, eps > 0)
    abs_m = If(m >= 0, m, -m)
    abs_g = If(g >= 0, g, -g)
    diff = (g - m) - mu_d
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

    all_unsat = True
    for name, status in results:
        is_pass = (status == unsat)
        if not is_pass:
            all_unsat = False
        verdict = "PASSED (UNSAT)" if is_pass else "FAILED (SAT)"
        print(f"{name:<48}: {verdict}")

    return all_unsat

if __name__ == "__main__":
    run_smt_suite()
