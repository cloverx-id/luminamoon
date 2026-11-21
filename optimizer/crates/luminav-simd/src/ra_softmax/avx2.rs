use std::arch::x86_64::*;

#[target_feature(enable = "avx2,fma")]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let lo128 = _mm256_castps256_ps128(v);
    let hi128 = _mm256_extractf128_ps::<1>(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_movehdup_ps(sum64);
    _mm_cvtss_f32(_mm_add_ss(sum64, hi32))
}

#[target_feature(enable = "avx2,fma")]
unsafe fn find_max_and_idx_avx2(scores: &[f32], eff_len: usize) -> (f32, i64) {
    let chunks = eff_len / 8;
    let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
    for c in 0..chunks {
        let v = _mm256_loadu_ps(scores.as_ptr().add(c * 8));
        max_vec = _mm256_max_ps(max_vec, v);
    }
    let lo128 = _mm256_castps256_ps128(max_vec);
    let hi128 = _mm256_extractf128_ps::<1>(max_vec);
    let max128 = _mm_max_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(max128, max128);
    let max64 = _mm_max_ps(max128, hi64);
    let hi32 = _mm_movehdup_ps(max64);
    let mut max_val = _mm_cvtss_f32(_mm_max_ss(max64, hi32));
    for &s in &scores[chunks * 8..eff_len] {
        if s > max_val {
            max_val = s;
        }
    }

    let max_broadcast = _mm256_set1_ps(max_val);
    let mut m_idx: i64 = 0;
    let mut found_max = false;
    for c in 0..chunks {
        let v = _mm256_loadu_ps(scores.as_ptr().add(c * 8));
        let eq = _mm256_movemask_ps(_mm256_cmp_ps(v, max_broadcast, _CMP_EQ_OQ));
        if eq != 0 {
            m_idx = (c * 8 + eq.trailing_zeros() as usize) as i64;
            found_max = true;
            break;
        }
    }
    if !found_max {
        for (j, &s) in scores[chunks * 8..eff_len].iter().enumerate() {
            if s == max_val {
                m_idx = (chunks * 8 + j) as i64;
                break;
            }
        }
    }
    (max_val, m_idx)
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn ra_softmax_row_with_stats_avx2(
    scores: &[f32],
    output: &mut [f32],
    scale: f32,
    eff_len: usize,
) -> (f32, f32, i64) {
    let (max_val, m_idx) = find_max_and_idx_avx2(scores, eff_len);
    let chunks = eff_len / 8;
    let max_broadcast = _mm256_set1_ps(max_val);
    let sign_mask = _mm256_set1_ps(-0.0_f32);
    let inf_vec = _mm256_set1_ps(f32::INFINITY);
    let scale_vec = _mm256_set1_ps(scale);
    let one_vec = _mm256_set1_ps(1.0_f32);
    let zero_vec = _mm256_setzero_ps();
    let mut sum_phi_vec = _mm256_setzero_ps();

    for c in 0..chunks {
        let offset = c * 8;
        let s = _mm256_loadu_ps(scores.as_ptr().add(offset));
        let finite = _mm256_cmp_ps(_mm256_andnot_ps(sign_mask, s), inf_vec, _CMP_LT_OQ);
        let is_max = _mm256_cmp_ps(s, max_broadcast, _CMP_EQ_OQ);
        let delta = _mm256_blendv_ps(_mm256_sub_ps(max_broadcast, s), zero_vec, is_max);
        let u = _mm256_mul_ps(delta, scale_vec);
        let active = _mm256_and_ps(finite, _mm256_cmp_ps(u, one_vec, _CMP_LT_OQ));
        let safe_u = _mm256_and_ps(active, u);
        let q = _mm256_div_ps(
            _mm256_sub_ps(one_vec, safe_u),
            _mm256_add_ps(one_vec, safe_u),
        );
        let q2 = _mm256_mul_ps(q, q);
        let q4 = _mm256_mul_ps(q2, q2);
        let phi = _mm256_and_ps(active, _mm256_mul_ps(q4, q4));

        sum_phi_vec = _mm256_add_ps(sum_phi_vec, phi);
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), phi);
    }

    let mut total_phi = hsum256_ps(sum_phi_vec);
    for i in (chunks * 8)..eff_len {
        let s = scores[i];
        let phi = if s.is_finite() {
            let delta = if s == max_val { 0.0 } else { max_val - s };
            let u = delta * scale;
            if u < 1.0 {
                let q = (1.0 - u) / (1.0 + u);
                let q2 = q * q;
                let q4 = q2 * q2;
                q4 * q4
            } else {
                0.0
            }
        } else {
            0.0
        };
        output[i] = phi;
        total_phi += phi;
    }

    let l_val = total_phi.max(1e-38);
    let inv_sum = 1.0_f32 / l_val;
    let inv_sum_vec = _mm256_set1_ps(inv_sum);
    for c in 0..chunks {
        let offset = c * 8;
        let p = _mm256_loadu_ps(output.as_ptr().add(offset));
        _mm256_storeu_ps(
            output.as_mut_ptr().add(offset),
            _mm256_mul_ps(p, inv_sum_vec),
        );
    }
    for x in output[chunks * 8..eff_len].iter_mut() {
        *x *= inv_sum;
    }
    (max_val, l_val, m_idx)
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn ra_softmax_row_with_stats_inplace_avx2(
    buf: &mut [f32],
    scale: f32,
    eff_len: usize,
) -> (f32, f32, i64) {
    let (max_val, m_idx) = find_max_and_idx_avx2(buf, eff_len);
    let chunks = eff_len / 8;
    let max_broadcast = _mm256_set1_ps(max_val);
    let sign_mask = _mm256_set1_ps(-0.0_f32);
    let inf_vec = _mm256_set1_ps(f32::INFINITY);
    let scale_vec = _mm256_set1_ps(scale);
    let one_vec = _mm256_set1_ps(1.0_f32);
    let zero_vec = _mm256_setzero_ps();
    let mut sum_phi_vec = _mm256_setzero_ps();

    for c in 0..chunks {
        let offset = c * 8;
        let ptr = buf.as_mut_ptr().add(offset);
        let s = _mm256_loadu_ps(ptr);
        let finite = _mm256_cmp_ps(_mm256_andnot_ps(sign_mask, s), inf_vec, _CMP_LT_OQ);
        let is_max = _mm256_cmp_ps(s, max_broadcast, _CMP_EQ_OQ);
        let delta = _mm256_blendv_ps(_mm256_sub_ps(max_broadcast, s), zero_vec, is_max);
        let u = _mm256_mul_ps(delta, scale_vec);
        let active = _mm256_and_ps(finite, _mm256_cmp_ps(u, one_vec, _CMP_LT_OQ));
        let safe_u = _mm256_and_ps(active, u);
        let q = _mm256_div_ps(
            _mm256_sub_ps(one_vec, safe_u),
            _mm256_add_ps(one_vec, safe_u),
        );
        let q2 = _mm256_mul_ps(q, q);
        let q4 = _mm256_mul_ps(q2, q2);
        let phi = _mm256_and_ps(active, _mm256_mul_ps(q4, q4));

        sum_phi_vec = _mm256_add_ps(sum_phi_vec, phi);
        _mm256_storeu_ps(ptr, phi);
    }

    let mut total_phi = hsum256_ps(sum_phi_vec);
    for x in buf[chunks * 8..eff_len].iter_mut() {
        let s = *x;
        let phi = if s.is_finite() {
            let delta = if s == max_val { 0.0 } else { max_val - s };
            let u = delta * scale;
            if u < 1.0 {
                let q = (1.0 - u) / (1.0 + u);
                let q2 = q * q;
                let q4 = q2 * q2;
                q4 * q4
            } else {
                0.0
            }
        } else {
            0.0
        };
        *x = phi;
        total_phi += phi;
    }

    let l_val = total_phi.max(1e-38);
    let inv_sum = 1.0_f32 / l_val;
    let inv_sum_vec = _mm256_set1_ps(inv_sum);
    for c in 0..chunks {
        let offset = c * 8;
        let ptr = buf.as_mut_ptr().add(offset);
        let p = _mm256_loadu_ps(ptr);
        _mm256_storeu_ps(ptr, _mm256_mul_ps(p, inv_sum_vec));
    }
    for x in buf[chunks * 8..eff_len].iter_mut() {
        *x *= inv_sum;
    }
    (max_val, l_val, m_idx)
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn ra_softmax_row_bwd_avx2(
    scores: &[f32],
    grad_p: &[f32],
    out_ds: &mut [f32],
    mut out_p: Option<&mut [f32]>,
    m_val: f32,
    l_val: f32,
    m_idx: i64,
    d_val: f32,
    scale: f32,
    eff_len: usize,
) {
    let chunks = eff_len / 8;
    let inv_l = 1.0_f32 / l_val.max(1e-38);
    let coeff = (16.0_f32 * scale) * inv_l;

    let max_v = _mm256_set1_ps(m_val);
    let inv_l_v = _mm256_set1_ps(inv_l);
    let coeff_v = _mm256_set1_ps(coeff);
    let d_v = _mm256_set1_ps(d_val);
    let scale_v = _mm256_set1_ps(scale);
    let one_v = _mm256_set1_ps(1.0_f32);
    let zero_v = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0_f32);
    let inf_v = _mm256_set1_ps(f32::INFINITY);
    let mut sum_ds_v = _mm256_setzero_ps();

    for c in 0..chunks {
        let offset = c * 8;
        let s = _mm256_loadu_ps(scores.as_ptr().add(offset));
        let gp = _mm256_loadu_ps(grad_p.as_ptr().add(offset));

        let finite = _mm256_cmp_ps(_mm256_andnot_ps(sign_mask, s), inf_v, _CMP_LT_OQ);
        let is_max = _mm256_cmp_ps(s, max_v, _CMP_EQ_OQ);
        let delta = _mm256_blendv_ps(_mm256_sub_ps(max_v, s), zero_v, is_max);
        let u = _mm256_mul_ps(delta, scale_v);
        let active = _mm256_and_ps(finite, _mm256_cmp_ps(u, one_v, _CMP_LT_OQ));
        let safe_u = _mm256_and_ps(active, u);

        // Single reciprocal 1 / (1 + u) reused for both q and pw.
        let inv_1pu = _mm256_div_ps(one_v, _mm256_add_ps(one_v, safe_u));
        let q = _mm256_mul_ps(_mm256_sub_ps(one_v, safe_u), inv_1pu);
        let q2 = _mm256_mul_ps(q, q);
        let q4 = _mm256_mul_ps(q2, q2);
        let q7 = _mm256_mul_ps(_mm256_mul_ps(q, q2), q4);

        if let Some(ref mut p_buf) = out_p {
            let p_val = _mm256_and_ps(active, _mm256_mul_ps(_mm256_mul_ps(q4, q4), inv_l_v));
            _mm256_storeu_ps(p_buf.as_mut_ptr().add(offset), p_val);
        }

        let inv_1pu_sq = _mm256_mul_ps(inv_1pu, inv_1pu);
        let pw = _mm256_mul_ps(_mm256_mul_ps(coeff_v, q7), inv_1pu_sq);
        let ds = _mm256_and_ps(active, _mm256_mul_ps(pw, _mm256_sub_ps(gp, d_v)));
        _mm256_storeu_ps(out_ds.as_mut_ptr().add(offset), ds);
        sum_ds_v = _mm256_add_ps(sum_ds_v, ds);
    }

    let mut sum_ds = hsum256_ps(sum_ds_v);
    for j in (chunks * 8)..eff_len {
        let s = scores[j];
        let mut p_val = 0.0_f32;
        let mut ds_val = 0.0_f32;
        if s.is_finite() {
            let delta = if s == m_val { 0.0 } else { m_val - s };
            let u = delta * scale;
            if u < 1.0 {
                let inv_1pu = 1.0_f32 / (1.0_f32 + u);
                let q = (1.0_f32 - u) * inv_1pu;
                let q2 = q * q;
                let q4 = q2 * q2;
                let q7 = q * q2 * q4;
                p_val = (q4 * q4) * inv_l;
                ds_val = (coeff * q7 * inv_1pu * inv_1pu) * (grad_p[j] - d_val);
            }
        }
        if let Some(ref mut p_buf) = out_p {
            p_buf[j] = p_val;
        }
        out_ds[j] = ds_val;
        sum_ds += ds_val;
    }

    if m_idx >= 0 && (m_idx as usize) < eff_len {
        out_ds[m_idx as usize] -= sum_ds;
    }
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn ra_softmax_row_bwd_inplace_avx2(
    scores_to_p: &mut [f32],
    dp_to_ds: &mut [f32],
    m_val: f32,
    l_val: f32,
    m_idx: i64,
    d_val: f32,
    scale: f32,
    eff_len: usize,
) {
    let chunks = eff_len / 8;
    let inv_l = 1.0_f32 / l_val.max(1e-38);
    let coeff = (16.0_f32 * scale) * inv_l;

    let max_v = _mm256_set1_ps(m_val);
    let inv_l_v = _mm256_set1_ps(inv_l);
    let coeff_v = _mm256_set1_ps(coeff);
    let d_v = _mm256_set1_ps(d_val);
    let scale_v = _mm256_set1_ps(scale);
    let one_v = _mm256_set1_ps(1.0_f32);
    let zero_v = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0_f32);
    let inf_v = _mm256_set1_ps(f32::INFINITY);
    let mut sum_ds_v = _mm256_setzero_ps();

    for c in 0..chunks {
        let offset = c * 8;
        let sp_ptr = scores_to_p.as_mut_ptr().add(offset);
        let dd_ptr = dp_to_ds.as_mut_ptr().add(offset);
        let s = _mm256_loadu_ps(sp_ptr);
        let gp = _mm256_loadu_ps(dd_ptr);

        let finite = _mm256_cmp_ps(_mm256_andnot_ps(sign_mask, s), inf_v, _CMP_LT_OQ);
        let is_max = _mm256_cmp_ps(s, max_v, _CMP_EQ_OQ);
        let delta = _mm256_blendv_ps(_mm256_sub_ps(max_v, s), zero_v, is_max);
        let u = _mm256_mul_ps(delta, scale_v);
        let active = _mm256_and_ps(finite, _mm256_cmp_ps(u, one_v, _CMP_LT_OQ));
        let safe_u = _mm256_and_ps(active, u);

        let inv_1pu = _mm256_div_ps(one_v, _mm256_add_ps(one_v, safe_u));
        let q = _mm256_mul_ps(_mm256_sub_ps(one_v, safe_u), inv_1pu);
        let q2 = _mm256_mul_ps(q, q);
        let q4 = _mm256_mul_ps(q2, q2);
        let q7 = _mm256_mul_ps(_mm256_mul_ps(q, q2), q4);

        let p_val = _mm256_and_ps(active, _mm256_mul_ps(_mm256_mul_ps(q4, q4), inv_l_v));
        _mm256_storeu_ps(sp_ptr, p_val);

        let inv_1pu_sq = _mm256_mul_ps(inv_1pu, inv_1pu);
        let pw = _mm256_mul_ps(_mm256_mul_ps(coeff_v, q7), inv_1pu_sq);
        let ds = _mm256_and_ps(active, _mm256_mul_ps(pw, _mm256_sub_ps(gp, d_v)));
        _mm256_storeu_ps(dd_ptr, ds);
        sum_ds_v = _mm256_add_ps(sum_ds_v, ds);
    }

    let mut sum_ds = hsum256_ps(sum_ds_v);
    for j in (chunks * 8)..eff_len {
        let s = scores_to_p[j];
        let gp = dp_to_ds[j];
        let mut p_val = 0.0_f32;
        let mut ds_val = 0.0_f32;
        if s.is_finite() {
            let delta = if s == m_val { 0.0 } else { m_val - s };
            let u = delta * scale;
            if u < 1.0 {
                let inv_1pu = 1.0_f32 / (1.0_f32 + u);
                let q = (1.0_f32 - u) * inv_1pu;
                let q2 = q * q;
                let q4 = q2 * q2;
                let q7 = q * q2 * q4;
                p_val = (q4 * q4) * inv_l;
                ds_val = (coeff * q7 * inv_1pu * inv_1pu) * (gp - d_val);
            }
        }
        scores_to_p[j] = p_val;
        dp_to_ds[j] = ds_val;
        sum_ds += ds_val;
    }

    if m_idx >= 0 && (m_idx as usize) < eff_len {
        dp_to_ds[m_idx as usize] -= sum_ds;
    }
}
