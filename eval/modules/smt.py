try:
    from . import common as _common
except (ImportError, ValueError):
    import common as _common

from z3 import If, Or, Real, Reals, Solver, unsat


def run_smt_suite() -> bool:
    solver = Solver()

    m, g = Reals("m g")
    w1, w2, lam, eps = Reals("w1 w2 lam eps")
    base_conds = [w1 > 0, w2 > 0, (w1 + w2) == 1, lam >= 0, eps > 0]

    abs_m = If(m >= 0, m, -m)
    abs_g = If(g >= 0, g, -g)
    diff = g - m
    abs_diff = If(diff >= 0, diff, -diff)

    clm_m = w1 * m + w2 * g
    clm_env = w1 * abs_m + w2 * abs_g + lam * abs_diff + eps
    abs_clm_m = If(clm_m >= 0, clm_m, -clm_m)

    solver.push()
    solver.add(*base_conds, clm_env <= 0)
    res_tdclm_div0 = solver.check()
    solver.pop()

    solver.push()
    solver.add(*base_conds, abs_clm_m >= clm_env)
    res_tdclm_bound = solver.check()
    solver.pop()

    sqrt_v, c1, c2 = Reals("sqrt_v c1 c2")
    sigma = sqrt_v * c1 + c2
    sigma_clamped = If(sigma >= 1e-12, sigma, 1e-12)
    solver.push()
    solver.add(sqrt_v >= 0, c1 > 0, c2 > 0)
    solver.add(Or(sigma <= 0, sigma_clamped <= 0))
    res_sigma = solver.check()
    solver.pop()

    m_bar, c_min = Reals("m_bar c_min")
    cautious_scale = 1.0 / m_bar
    solver.push()
    solver.add(c_min == 0.5, m_bar >= c_min, m_bar <= 1.0)
    solver.add(Or(cautious_scale < 1.0, cautious_scale > (1.0 / c_min)))
    res_cautious = solver.check()
    solver.pop()

    p, B_ratio, norm_p = Reals("p B_ratio norm_p")
    solver.push()
    solver.add(B_ratio > 0)
    B_sq = (B_ratio * B_ratio) * (p * p + 1e-4)
    solver.add(B_sq <= 0)
    res_coord_b = solver.check()
    solver.pop()

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
    v_fp16 = If(v_raw <= 65000.0, v_raw, 65000.0)
    subnormal_floor = 6.1035e-5

    solver.push()
    solver.add(b_safe < 1.5e-4)
    res_fp16 = solver.check()
    solver.pop()

    solver.push()
    solver.add(Or(b_safe <= subnormal_floor, v_fp16 > 65504.0))
    res_subnormal = solver.check()
    solver.pop()

    r = Real("r")
    solver.push()
    solver.add(r >= 0)
    s_radial = If(r == 0, 1.0, (15.0 + r * r) / (15.0 + 6.0 * r * r))
    solver.add(Or(s_radial <= 0.0, s_radial > 1.0))
    res_radial_scale = solver.check()
    solver.pop()

    beta1, beta2, b1_tp1, b2_t = Reals("beta1 beta2 b1_tp1 b2_t")
    solver.push()
    solver.add(
        beta1 > 0, beta1 < 1,
        beta2 > 0, beta2 < 1,
        b1_tp1 > 0, b1_tp1 <= beta1 * beta1,
        b2_t > 0, b2_t <= beta2,
    )
    bc1_nes = 1.0 - b1_tp1
    bc2_clamped = If(1.0 - b2_t >= 1e-15, 1.0 - b2_t, 1e-15)
    solver.add(Or(bc1_nes <= 0, bc2_clamped <= 0))
    res_bias_corr = solver.check()
    solver.pop()

    lr, wd = Reals("lr wd")
    solver.push()
    solver.add(lr > 0, lr <= 0.05, wd >= 0, wd <= 0.2)
    decay_factor = 1.0 - lr * wd
    solver.add(Or(decay_factor <= 0, decay_factor > 1))
    res_wd = solver.check()
    solver.pop()

    g_val, u_val, w2_val, lam_val, eps_val, gamma_c = Reals(
        "g_val u_val w2_val lam_val eps_val gamma_c"
    )
    solver.push()
    solver.add(w2_val > 0, lam_val >= 0, eps_val > 0, gamma_c >= 1.0)
    abs_g_val = If(g_val >= 0, g_val, -g_val)
    u_zero_m = (w2_val * g_val) / (w2_val * abs_g_val + lam_val * abs_g_val + eps_val)
    m_mask = If((u_val * g_val) > 0, 1.0, 0.0)
    delta_cautious = u_val * m_mask * gamma_c
    solver.add(Or(u_zero_m * g_val < 0, delta_cautious * g_val < 0))
    res_descent = solver.check()
    solver.pop()

    results = [
        ("1. TD-CLM Div-Zero (Q1)", res_tdclm_div0),
        ("2. TD-CLM Metric Bound |u| < 1.0 (Q2)", res_tdclm_bound),
        ("3. Buffer-2 Sigma Positivity (Q1/Q2)", res_sigma),
        ("4. Cautious Scale Bounds [1, 1/c_min] (Q3)", res_cautious),
        ("5. Coordinate Governor Radius B > 0 (Q2)", res_coord_b),
        ("6. Radial Governor Lower Bound R >= B_ratio (Q2)", res_radial_r),
        ("7. FP16 Safe-Zone Floor >= 1.5e-4 (Q4)", res_fp16),
        ("8. FP16 Subnormal & Variance Safety (Q4)", res_subnormal),
        ("9. Radial Scale Contraction in (0, 1] (Q3)", res_radial_scale),
        ("10. Bias Correction Positivity t >= 1 (Q7)", res_bias_corr),
        ("11. Weight Decay Contraction in (0, 1] (Q6)", res_wd),
        ("12. Descent Direction Alignment >= 0 (Q5)", res_descent),
    ]

    print("Stage I.1: Z3 SMT Safety Verification")
    all_passed = True
    for name, status in results:
        ok = status == unsat
        if not ok:
            all_passed = False
        verdict = "PASSED (UNSAT)" if ok else "FAILED (SAT)"
        print(f"{name:<50}: {verdict}")

    return all_passed


run_z3_smt_audit = run_smt_suite

if __name__ == "__main__":
    run_smt_suite()
