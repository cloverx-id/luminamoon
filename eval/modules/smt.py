try:
    from . import common as _common
except (ImportError, ValueError):
    import common as _common

from z3 import (
    And,
    BitVec,
    BitVecVal,
    Concat,
    Extract,
    FP,
    FPVal,
    Float16,
    Float32,
    If,
    LShR,
    Not,
    Or,
    RNE,
    Real,
    RealVal,
    Reals,
    Solver,
    UGE,
    ULE,
    ULT,
    ZeroExt,
    fpAbs,
    fpBVToFP,
    fpGEQ,
    fpGT,
    fpIsInf,
    fpIsNaN,
    fpIsZero,
    fpLEQ,
    fpLT,
    fpMul,
    fpToFP,
    fpToIEEEBV,
    unsat,
)


def _prove(solver: Solver, *constraints):
    solver.push()
    solver.add(*constraints)
    res = solver.check()
    solver.pop()
    return res


def _run_bitvec_and_ieee_queries() -> list[tuple[str, object]]:
    def _check(build_fn):
        s = Solver()
        s.set("timeout", 30000)
        build_fn(s)
        return s.check()

    def q_b1(s: Solver) -> None:
        raw = BitVec("raw", 32)
        m31 = BitVecVal(0x7FFFFFFF, 32)
        sm = raw & m31
        sp_ = (sm + BitVecVal(0x55555555, 32)) & m31
        sv = (sm + BitVecVal(0x2A6D365B, 32)) & m31
        z0 = BitVecVal(0, 32)
        s.add(Or(sm < z0, sp_ < z0, sv < z0, sm == sp_, sm == sv, sp_ == sv))

    def q_b2(s: Solver) -> None:
        z, weyl = BitVec("z", 32), BitVec("weyl", 32)
        c1, c2 = 0x85EB5A6B, 0xC2B2AE35
        c1_inv = pow(c1, -1, 1 << 32)
        c2_inv = pow(c2, -1, 1 << 32)
        y16 = z ^ LShR(z, 16)
        inv16 = y16 ^ LShR(y16, 16)
        y13 = z ^ LShR(z, 13)
        inv13 = y13 ^ LShR(y13, 13) ^ LShR(y13, 26)
        inv_c1 = (z * BitVecVal(c1, 32)) * BitVecVal(c1_inv, 32)
        inv_c2 = (z * BitVecVal(c2, 32)) * BitVecVal(c2_inv, 32)
        inv_w = (z + weyl) - weyl
        s.add(Or(inv16 != z, inv13 != z, inv_c1 != z, inv_c2 != z, inv_w != z))

    def q_b3(s: Solver) -> None:
        B, nr = BitVec("B_bf16", 32), BitVec("nr_bf16", 32)
        S, A = B & BitVecVal(0x80000000, 32), B & BitVecVal(0x7FFFFFFF, 32)
        s.add(ULT(Extract(30, 23, B), BitVecVal(0xFF, 8)))
        noise = LShR(nr, 16)
        masked = (A + noise) & BitVecVal(0xFFFF0000, 32)
        bmax = BitVecVal(0x7F7F0000, 32)
        R_abs = If(ULE(masked, bmax), masked, bmax)
        out = S | R_abs
        low = A & BitVecVal(0xFFFF0000, 32)
        high = low + BitVecVal(0x00010000, 32)
        s.add(
            Or(
                Not(ULT(Extract(30, 23, out), BitVecVal(0xFF, 8))),
                (out & BitVecVal(0x80000000, 32)) != S,
                And(A == BitVecVal(0, 32), R_abs != BitVecVal(0, 32)),
                And(ULE(A, bmax), (A & BitVecVal(0xFFFF, 32)) == BitVecVal(0, 32), R_abs != A),
                And(ULT(A, bmax), Not(Or(R_abs == low, R_abs == high))),
            )
        )

    def q_b4(s: Solver) -> None:
        B, nr = BitVec("B_fp16", 32), BitVec("nr_fp16", 32)
        S, A = B & BitVecVal(0x80000000, 32), B & BitVecVal(0x7FFFFFFF, 32)
        exp_f32 = ZeroExt(24, Extract(30, 23, B))
        s.add(ULT(exp_f32, BitVecVal(255, 32)))
        diff113 = BitVecVal(113, 32) - exp_f32
        sub_shift = If(
            ULT(exp_f32, BitVecVal(113, 32)),
            If(ULE(diff113, BitVecVal(11, 32)), diff113, BitVecVal(11, 32)),
            BitVecVal(0, 32),
        )
        tot = BitVecVal(13, 32) + sub_shift
        mask_b = (BitVecVal(1, 32) << tot) - BitVecVal(1, 32)
        noise = LShR(nr, BitVecVal(32, 32) - tot)
        masked = (A + noise) & (~mask_b)
        fmax = BitVecVal(0x477FE000, 32)
        R_abs = If(ULT(exp_f32, BitVecVal(103, 32)), BitVecVal(0, 32), If(ULE(masked, fmax), masked, fmax))
        out = S | R_abs
        flr = A & (~mask_b)
        hi = flr + (BitVecVal(1, 32) << tot)
        s.add(
            Or(
                Not(ULE(R_abs, fmax)),
                (out & BitVecVal(0x80000000, 32)) != S,
                And(ULT(exp_f32, BitVecVal(103, 32)), R_abs != BitVecVal(0, 32)),
                And(UGE(exp_f32, BitVecVal(103, 32)), ULE(A, fmax), (A & mask_b) == BitVecVal(0, 32), R_abs != A),
                And(UGE(exp_f32, BitVecVal(103, 32)), ULT(A, fmax), Not(Or(R_abs == flr, R_abs == hi))),
            )
        )

    def q_b5(s: Solver) -> None:
        raw_u16 = BitVec("raw_u16", 16)
        roundtrip_u16 = Extract(31, 16, Concat(raw_u16, BitVecVal(0, 16)))
        B = BitVec("B_state", 32)
        S, A = B & BitVecVal(0x80000000, 32), B & BitVecVal(0x7FFFFFFF, 32)
        bmax = BitVecVal(0x7F7F0000, 32)
        masked = (A + LShR(BitVec("ns", 32), 16)) & BitVecVal(0xFFFF0000, 32)
        R_abs = If(ULE(masked, bmax), masked, bmax)
        dec = Concat(Extract(31, 16, S | R_abs), BitVecVal(0, 16))
        bad_exp = And(ULT(Extract(30, 23, B), BitVecVal(0xFF, 8)), Not(ULT(Extract(30, 23, dec), BitVecVal(0xFF, 8))))
        s.add(Or(roundtrip_u16 != raw_u16, bad_exp))

    def q_f1(s: Solver) -> None:
        v = FP("v_f32", Float32())
        s.add(fpGEQ(v, FPVal(2.0**-126, Float32())), fpLT(v, FPVal(2.0**-25, Float32())))
        v_fp16 = fpToFP(RNE(), v, Float16())
        vb = fpToIEEEBV(v)
        v_S, v_A = vb & BitVecVal(0x80000000, 32), vb & BitVecVal(0x7FFFFFFF, 32)
        v_m = (v_A + LShR(BitVec("vn", 32), 16)) & BitVecVal(0xFFFF0000, 32)
        v_r = If(ULE(v_m, BitVecVal(0x7F7F0000, 32)), v_m, BitVecVal(0x7F7F0000, 32))
        v_pr = fpBVToFP(Concat(Extract(31, 16, v_S | v_r), BitVecVal(0, 16)), Float32())
        s.add(Or(Not(fpIsZero(v_fp16)), fpIsZero(v_pr), fpIsNaN(v_pr), fpIsInf(v_pr)))

    def q_f2(s: Solver) -> None:
        g = FP("g_raw", Float32())
        l38, l18, n18, z0 = FPVal(1e38, Float32()), FPVal(1e18, Float32()), FPVal(-1e18, Float32()), FPVal(0.0, Float32())
        safe = And(Not(fpIsNaN(g)), Not(fpIsInf(g)), fpLT(fpAbs(g), l38))
        gs = If(safe, If(fpGT(g, l18), l18, If(fpLT(g, n18), n18, g)), z0)
        s.add(Or(fpIsNaN(gs), fpIsInf(gs), fpGT(fpAbs(gs), l18)))

    def q_f3(s: Solver) -> None:
        d = FP("diff_f32", Float32())
        s.add(Not(fpIsNaN(d)), Not(fpIsInf(d)), fpLEQ(fpAbs(d), FPVal(1e18, Float32())))
        d2 = fpMul(RNE(), d, d)
        s.add(Or(fpIsNaN(d2), fpIsInf(d2), fpLT(d2, FPVal(0.0, Float32()))))

    return [
        ("B1. 3-Salt 31-bit Seed Independence (QF_BV)", _check(q_b1)),
        ("B2. Unsigned LShR Murmur3 + 3D Weyl Bijectivity", _check(q_b2)),
        ("B3. BF16 Top-16 SR Safety & 1-ULP Bracket", _check(q_b3)),
        ("B4. FP16 Subnormal-Aware Top-k SR Safety", _check(q_b4)),
        ("B5. Bitwise-BF16 Container Roundtrip Safety", _check(q_b5)),
        ("F1. IEEE FP16 Underflow vs Bitwise-BF16 Retention", _check(q_f1)),
        ("F2. IEEE Float32 Gradient Sanitization (2^32)", _check(q_f2)),
        ("F3. IEEE Float32 Innovation Squaring Safety", _check(q_f3)),
    ]


def run_smt_checks() -> bool:
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

    res_tdclm_div0 = _prove(solver, *base_conds, clm_env <= 0)
    res_tdclm_bound = _prove(solver, *base_conds, abs_clm_m >= clm_env)

    sqrt_v, c1, c2 = Reals("sqrt_v c1 c2")
    sigma = sqrt_v * c1 + c2
    sigma_clamped = If(sigma >= 1e-12, sigma, 1e-12)
    res_sigma = _prove(solver, sqrt_v >= 0, c1 > 0, c2 > 0, Or(sigma <= 0, sigma_clamped < 1e-12))

    m_bar, c_min = Reals("m_bar c_min")
    cautious_scale = 1.0 / m_bar
    res_cautious = _prove(solver, c_min == 0.5, m_bar >= c_min, m_bar <= 1.0, Or(cautious_scale < 1.0, cautious_scale > 2.0))

    p, B_ratio, norm_p, f_floor = Reals("p B_ratio norm_p f_floor")
    B_sq = (B_ratio * B_ratio) * (p * p + 1e-4)
    res_coord_b = _prove(solver, B_ratio > 0, B_sq < (B_ratio * B_ratio) * 1e-4)

    R = If(norm_p >= 1.0, B_ratio * norm_p, B_ratio * 1.0)
    res_radial_r = _prove(solver, B_ratio > 0, norm_p >= 0, R < B_ratio)

    p_fp16, v_raw = Reals("p_fp16 v_raw")
    abs_p_fp16 = If(p_fp16 >= 0, p_fp16, -p_fp16)
    b_safe = If(0.01 * abs_p_fp16 >= 1.5e-4, 0.01 * abs_p_fp16, 1.5e-4)
    v_fp16 = If(v_raw <= 65000.0, v_raw, 65000.0)
    res_fp16 = _prove(solver, b_safe < 1.5e-4)
    res_subnormal = _prove(solver, Or(b_safe <= 6.1035e-5, v_fp16 > 65504.0))

    r, tanh_r = Reals("r tanh_r")
    denom_r = If(r >= 1e-8, r, 1e-8)
    s_radial = If(r < 1e-4, 1.0, tanh_r / denom_r)
    res_radial_scale = _prove(
        solver,
        r >= 0,
        If(r == 0, tanh_r == 0, And(tanh_r > 0, tanh_r < r, tanh_r <= 1.0)),
        Or(s_radial <= 0.0, s_radial > 1.0),
    )

    beta1, beta2, b1_tp1, b2_t = Reals("beta1 beta2 b1_tp1 b2_t")
    bc1_nes = 1.0 - b1_tp1
    bc2_clamped = If(1.0 - b2_t >= 1e-15, 1.0 - b2_t, 1e-15)
    res_bias_corr = _prove(
        solver,
        beta1 > 0, beta1 < 1, beta2 > 0, beta2 < 1,
        b1_tp1 > 0, b1_tp1 <= beta1 * beta1, b2_t > 0, b2_t <= beta2,
        Or(bc1_nes <= 0, bc2_clamped <= 0),
    )

    lr, wd = Reals("lr wd")
    decay_factor = 1.0 - lr * wd
    res_wd = _prove(solver, lr > 0, lr <= 0.05, wd >= 0, wd <= 1.0, Or(decay_factor <= 0, decay_factor > 1))

    g_val, u_val, w2_val, lam_val, eps_val, gamma_c = Reals("g_val u_val w2_val lam_val eps_val gamma_c")
    abs_g_val = If(g_val >= 0, g_val, -g_val)
    u_zero_m = (w2_val * g_val) / (w2_val * abs_g_val + lam_val * abs_g_val + eps_val)
    delta_cautious = u_val * If((u_val * g_val) > 0, 1.0, 0.0) * gamma_c
    res_descent = _prove(
        solver,
        w2_val > 0, lam_val >= 0, eps_val > 0, gamma_c >= 1.0,
        Or(u_zero_m * g_val < 0, delta_cautious * g_val < 0),
    )

    # Pade [13/6] non-singularity & strict unit bound
    s_pade = Solver()
    s_pade.set("timeout", 30000)
    x_p = Real("x_pade")
    x2 = x_p * x_p
    b0 = RealVal("0.0048935246")
    q_p = ((RealVal("0.0000011982584") * x2 + RealVal("0.00011853471")) * x2 + RealVal("0.0022684346")) * x2 + b0
    p_p = (
        ((((RealVal("-0.00000000000000027607684") * x2 + RealVal("0.00000000000020001880")) * x2
           + RealVal("-0.000000000086046720")) * x2 + RealVal("0.000000051222970")) * x2
         + RealVal("0.000014857224")) * x2 + RealVal("0.00063726193")
    ) * x2 + b0
    res_pade_denom = _prove(s_pade, Or(q_p <= 0, q_p < b0))
    res_pade_range = _prove(s_pade, x_p >= 0, x_p <= RealVal("7.9988117"), Or(p_p <= 0, x_p * p_p >= q_p))

    # v2.0.0 invariants: T2 envelope, T1/T2 rational bounds, bound_floor, RTV, CWD
    sigma_t2 = Real("sigma_t2")
    env_t2 = w1 * abs_m + w2 * abs_g + 0.5 * lam * abs_diff + 0.5 * lam * sigma_t2 + eps
    res_t2_env = _prove(solver, *base_conds, sigma_t2 >= 0, Or(env_t2 <= 0, abs_clm_m >= env_t2))

    b_rat, s_rat = Reals("b_rat s_rat")
    abs_s = If(s_rat >= 0, s_rat, -s_rat)
    den_rat = b_rat * (b_rat + abs_s) + s_rat * s_rat
    num_rat = b_rat * (b_rat + abs_s) * s_rat
    abs_num_rat = If(num_rat >= 0, num_rat, -num_rat)
    res_rat_coord = _prove(solver, b_rat > 0, Or(den_rat <= 0, abs_num_rat >= b_rat * den_rat, num_rat * s_rat < 0))

    r_tri = Real("r_tri")
    res_rat_rad = _prove(solver, r_tri >= 0, Or(1.0 + r_tri + r_tri * r_tri <= 0, 1.0 + r_tri <= 0, 1.0 + r_tri > 1.0 + r_tri + r_tri * r_tri))

    abs_p = If(p >= 0, p, -p)
    res_bfloor = _prove(
        solver,
        B_ratio > 0, f_floor > 0,
        Or(B_ratio * (abs_p + f_floor) < B_ratio * f_floor, (B_ratio * B_ratio) * (p * p + f_floor * f_floor) < (B_ratio * f_floor) * (B_ratio * f_floor)),
    )

    d_abs, s_ref, c_temper = Reals("d_abs s_ref c_temper")
    b_rtv = c_temper * s_ref + eps
    den_rtv = b_rtv * (b_rtv + d_abs) + d_abs * d_abs
    num_rtv = b_rtv * (b_rtv + d_abs) * d_abs
    res_rtv = _prove(solver, d_abs >= 0, s_ref >= 0, c_temper > 0, eps > 0, Or(den_rtv <= 0, num_rtv < 0, num_rtv > d_abs * den_rtv, num_rtv >= b_rtv * den_rtv))

    p_init, step_v = Reals("p_init step_v")
    abs_pi = If(p_init >= 0, p_init, -p_init)
    p_cwd = If(step_v * p_init >= 0, p_init * (1.0 - lr * wd), p_init)
    abs_pcwd = If(p_cwd >= 0, p_cwd, -p_cwd)
    res_cwd = _prove(solver, lr > 0, lr <= 0.05, wd >= 0, wd <= 1.0, abs_pcwd > abs_pi)

    results = [
        ("1. TD-CLM Div-Zero (Q1)", res_tdclm_div0),
        ("2. TD-CLM Metric Bound |u| < 1.0 (Q2)", res_tdclm_bound),
        ("3. Buffer-2 Sigma Positivity (Q3)", res_sigma),
        ("4. Cautious Scale Bounds [1, 1/c_min] (Q4)", res_cautious),
        ("5. Coordinate Clipping Bound B > 0 (Q5)", res_coord_b),
        ("6. Radial Clipping Lower Bound R >= B_ratio (Q6)", res_radial_r),
        ("7. FP16 Safe-Zone Floor >= 1.5e-4 (Q7)", res_fp16),
        ("8. FP16 Subnormal & Variance Safety (Q8)", res_subnormal),
        ("9. Radial Scale Contraction in (0, 1] (Q9)", res_radial_scale),
        ("10. Bias Correction Positivity t >= 1 (Q10)", res_bias_corr),
        ("11. Weight Decay Contraction in (0, 1] (Q11)", res_wd),
        ("12. Descent Direction Alignment >= 0 (Q12)", res_descent),
        ("13. Pade [13/6] Denominator Positivity (Q13)", res_pade_denom),
        ("14. Pade [13/6] Strict Unit Bound (Q14)", res_pade_range),
        ("15. T2 Triangle Envelope |u_t| < 1.0 (Q15)", res_t2_env),
        ("16. T1/T2 Rational Coordinate Bound (Q16)", res_rat_coord),
        ("17. T1/T2 Rational Radial Contraction (Q17)", res_rat_rad),
        ("18. Configurable bound_floor Positivity (Q18)", res_bfloor),
        ("19. RTV Tempered Residual Safety (Q19)", res_rtv),
        ("20. CWD Weight Decay Contraction (Q20)", res_cwd),
    ] + _run_bitvec_and_ieee_queries()

    print("Stage I.1 & I.5: Z3 SMT Formal Verification (Real, QF_BV & IEEE-754)")
    all_passed = True
    for name, status in results:
        ok = status == unsat
        if not ok:
            all_passed = False
        verdict = "PASSED (UNSAT)" if ok else f"FAILED ({status})"
        print(f"{name:<52}: {verdict}")

    return all_passed


run_z3_smt_audit = run_smt_checks

if __name__ == "__main__":
    run_smt_checks()
