use std::arch::x86_64::*;

#[target_feature(enable = "avx512f,avx512vl,fma")]
unsafe fn find_max_and_idx_avx512(scores: &[f32], eff_len: usize) -> (f32, i64) {
    let chunks = eff_len / 16;
    let mut max_vec = _mm512_set1_ps(f32::NEG_INFINITY);
    for c in 0..chunks {
        let v = _mm512_loadu_ps(scores.as_ptr().add(c * 16));
        max_vec = _mm512_max_ps(max_vec, v);
    }
    let mut max_val = _mm512_reduce_max_ps(max_vec);
    for &s in &scores[chunks * 16..eff_len] {
        if s > max_val {
            max_val = s;
        }
    }

    let max_broadcast = _mm512_set1_ps(max_val);
    let mut m_idx: i64 = 0;
    let mut found_max = false;
    for c in 0..chunks {
        let v = _mm512_loadu_ps(scores.as_ptr().add(c * 16));
        let eq = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(v, max_broadcast);
        if eq != 0 {
            m_idx = (c * 16 + eq.trailing_zeros() as usize) as i64;
            found_max = true;
            break;
        }
    }
    if !found_max {
        for (j, &s) in scores[chunks * 16..eff_len].iter().enumerate() {
            if s == max_val {
                m_idx = (chunks * 16 + j) as i64;
                break;
            }
        }
    }
    (max_val, m_idx)
}

#[target_feature(enable = "avx512f,avx512vl,fma")]
pub(super) unsafe fn ra_softmax_row_with_stats_avx512(
    scores: &[f32],
    output: &mut [f32],
    scale: f32,
    eff_len: usize,
) -> (f32, f32, i64) {
    let (max_val, m_idx) = find_max_and_idx_avx512(scores, eff_len);
    let chunks = eff_len / 16;
    let max_v = _mm512_set1_ps(max_val);
    let inf_v = _mm512_set1_ps(f32::INFINITY);
    let scale_v = _mm512_set1_ps(scale);
    let one_v = _mm512_set1_ps(1.0_f32);
    let zero_v = _mm512_setzero_ps();
    let mut sum_phi_v = _mm512_setzero_ps();

    for c in 0..chunks {
        let off = c * 16;
        let s = _mm512_loadu_ps(scores.as_ptr().add(off));
        let finite_m = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(_mm512_abs_ps(s), inf_v);
        let is_max_m = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(s, max_v);
        let delta = _mm512_mask_blend_ps(is_max_m, _mm512_sub_ps(max_v, s), zero_v);
        let u = _mm512_mul_ps(delta, scale_v);
        let active_m = finite_m & _mm512_cmp_ps_mask::<_CMP_LT_OQ>(u, one_v);
        let safe_u = _mm512_maskz_mov_ps(active_m, u);
        let q = _mm512_div_ps(_mm512_sub_ps(one_v, safe_u), _mm512_add_ps(one_v, safe_u));
        let q2 = _mm512_mul_ps(q, q);
        let q4 = _mm512_mul_ps(q2, q2);
        let phi = _mm512_maskz_mul_ps(active_m, q4, q4);
        sum_phi_v = _mm512_add_ps(sum_phi_v, phi);
        _mm512_storeu_ps(output.as_mut_ptr().add(off), phi);
    }

    let mut total_phi = _mm512_reduce_add_ps(sum_phi_v);
    for i in (chunks * 16)..eff_len {
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
    let inv_sum_v = _mm512_set1_ps(inv_sum);
    for c in 0..chunks {
        let off = c * 16;
        let ptr = output.as_mut_ptr().add(off);
        _mm512_storeu_ps(ptr, _mm512_mul_ps(_mm512_loadu_ps(ptr), inv_sum_v));
    }
    for x in output[chunks * 16..eff_len].iter_mut() {
        *x *= inv_sum;
    }
    (max_val, l_val, m_idx)
}

#[target_feature(enable = "avx512f,avx512vl,fma")]
pub(super) unsafe fn ra_softmax_row_with_stats_inplace_avx512(
    buf: &mut [f32],
    scale: f32,
    eff_len: usize,
) -> (f32, f32, i64) {
    let (max_val, m_idx) = find_max_and_idx_avx512(buf, eff_len);
    let chunks = eff_len / 16;
    let max_v = _mm512_set1_ps(max_val);
    let inf_v = _mm512_set1_ps(f32::INFINITY);
    let scale_v = _mm512_set1_ps(scale);
    let one_v = _mm512_set1_ps(1.0_f32);
    let zero_v = _mm512_setzero_ps();
    let mut sum_phi_v = _mm512_setzero_ps();

    for c in 0..chunks {
        let ptr = buf.as_mut_ptr().add(c * 16);
        let s = _mm512_loadu_ps(ptr);
        let finite_m = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(_mm512_abs_ps(s), inf_v);
        let is_max_m = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(s, max_v);
        let delta = _mm512_mask_blend_ps(is_max_m, _mm512_sub_ps(max_v, s), zero_v);
        let u = _mm512_mul_ps(delta, scale_v);
        let active_m = finite_m & _mm512_cmp_ps_mask::<_CMP_LT_OQ>(u, one_v);
        let safe_u = _mm512_maskz_mov_ps(active_m, u);
        let q = _mm512_div_ps(_mm512_sub_ps(one_v, safe_u), _mm512_add_ps(one_v, safe_u));
        let q2 = _mm512_mul_ps(q, q);
        let q4 = _mm512_mul_ps(q2, q2);
        let phi = _mm512_maskz_mul_ps(active_m, q4, q4);
        sum_phi_v = _mm512_add_ps(sum_phi_v, phi);
        _mm512_storeu_ps(ptr, phi);
    }

    let mut total_phi = _mm512_reduce_add_ps(sum_phi_v);
    for x in buf[chunks * 16..eff_len].iter_mut() {
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
    let inv_sum_v = _mm512_set1_ps(inv_sum);
    for c in 0..chunks {
        let ptr = buf.as_mut_ptr().add(c * 16);
        _mm512_storeu_ps(ptr, _mm512_mul_ps(_mm512_loadu_ps(ptr), inv_sum_v));
    }
    for x in buf[chunks * 16..eff_len].iter_mut() {
        *x *= inv_sum;
    }
    (max_val, l_val, m_idx)
}

#[target_feature(enable = "avx512f,avx512vl,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn ra_softmax_row_bwd_avx512(
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
    let chunks = eff_len / 16;
    let inv_l = 1.0_f32 / l_val.max(1e-38);
    let coeff = (16.0_f32 * scale) * inv_l;

    let max_v = _mm512_set1_ps(m_val);
    let inv_l_v = _mm512_set1_ps(inv_l);
    let coeff_v = _mm512_set1_ps(coeff);
    let d_v = _mm512_set1_ps(d_val);
    let scale_v = _mm512_set1_ps(scale);
    let one_v = _mm512_set1_ps(1.0_f32);
    let zero_v = _mm512_setzero_ps();
    let inf_v = _mm512_set1_ps(f32::INFINITY);
    let mut sum_ds_v = _mm512_setzero_ps();

    for c in 0..chunks {
        let off = c * 16;
        let s = _mm512_loadu_ps(scores.as_ptr().add(off));
        let gp = _mm512_loadu_ps(grad_p.as_ptr().add(off));

        let finite_m = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(_mm512_abs_ps(s), inf_v);
        let is_max_m = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(s, max_v);
        let delta = _mm512_mask_blend_ps(is_max_m, _mm512_sub_ps(max_v, s), zero_v);
        let u = _mm512_mul_ps(delta, scale_v);
        let active_m = finite_m & _mm512_cmp_ps_mask::<_CMP_LT_OQ>(u, one_v);
        let safe_u = _mm512_maskz_mov_ps(active_m, u);

        let inv_1pu = _mm512_div_ps(one_v, _mm512_add_ps(one_v, safe_u));
        let q = _mm512_mul_ps(_mm512_sub_ps(one_v, safe_u), inv_1pu);
        let q2 = _mm512_mul_ps(q, q);
        let q4 = _mm512_mul_ps(q2, q2);
        let q7 = _mm512_mul_ps(_mm512_mul_ps(q, q2), q4);

        if let Some(ref mut p_buf) = out_p {
            let p_val = _mm512_maskz_mul_ps(active_m, _mm512_mul_ps(q4, q4), inv_l_v);
            _mm512_storeu_ps(p_buf.as_mut_ptr().add(off), p_val);
        }

        let pw = _mm512_mul_ps(_mm512_mul_ps(coeff_v, q7), _mm512_mul_ps(inv_1pu, inv_1pu));
        let ds = _mm512_maskz_mul_ps(active_m, pw, _mm512_sub_ps(gp, d_v));
        _mm512_storeu_ps(out_ds.as_mut_ptr().add(off), ds);
        sum_ds_v = _mm512_add_ps(sum_ds_v, ds);
    }

    let mut sum_ds = _mm512_reduce_add_ps(sum_ds_v);
    for j in (chunks * 16)..eff_len {
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

#[target_feature(enable = "avx512f,avx512vl,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn ra_softmax_row_bwd_inplace_avx512(
    scores_to_p: &mut [f32],
    dp_to_ds: &mut [f32],
    m_val: f32,
    l_val: f32,
    m_idx: i64,
    d_val: f32,
    scale: f32,
    eff_len: usize,
) {
    let chunks = eff_len / 16;
    let inv_l = 1.0_f32 / l_val.max(1e-38);
    let coeff = (16.0_f32 * scale) * inv_l;

    let max_v = _mm512_set1_ps(m_val);
    let inv_l_v = _mm512_set1_ps(inv_l);
    let coeff_v = _mm512_set1_ps(coeff);
    let d_v = _mm512_set1_ps(d_val);
    let scale_v = _mm512_set1_ps(scale);
    let one_v = _mm512_set1_ps(1.0_f32);
    let zero_v = _mm512_setzero_ps();
    let inf_v = _mm512_set1_ps(f32::INFINITY);
    let mut sum_ds_v = _mm512_setzero_ps();

    for c in 0..chunks {
        let off = c * 16;
        let sp_ptr = scores_to_p.as_mut_ptr().add(off);
        let dd_ptr = dp_to_ds.as_mut_ptr().add(off);
        let s = _mm512_loadu_ps(sp_ptr);
        let gp = _mm512_loadu_ps(dd_ptr);

        let finite_m = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(_mm512_abs_ps(s), inf_v);
        let is_max_m = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(s, max_v);
        let delta = _mm512_mask_blend_ps(is_max_m, _mm512_sub_ps(max_v, s), zero_v);
        let u = _mm512_mul_ps(delta, scale_v);
        let active_m = finite_m & _mm512_cmp_ps_mask::<_CMP_LT_OQ>(u, one_v);
        let safe_u = _mm512_maskz_mov_ps(active_m, u);

        let inv_1pu = _mm512_div_ps(one_v, _mm512_add_ps(one_v, safe_u));
        let q = _mm512_mul_ps(_mm512_sub_ps(one_v, safe_u), inv_1pu);
        let q2 = _mm512_mul_ps(q, q);
        let q4 = _mm512_mul_ps(q2, q2);
        let q7 = _mm512_mul_ps(_mm512_mul_ps(q, q2), q4);

        let p_val = _mm512_maskz_mul_ps(active_m, _mm512_mul_ps(q4, q4), inv_l_v);
        _mm512_storeu_ps(sp_ptr, p_val);

        let pw = _mm512_mul_ps(_mm512_mul_ps(coeff_v, q7), _mm512_mul_ps(inv_1pu, inv_1pu));
        let ds = _mm512_maskz_mul_ps(active_m, pw, _mm512_sub_ps(gp, d_v));
        _mm512_storeu_ps(dd_ptr, ds);
        sum_ds_v = _mm512_add_ps(sum_ds_v, ds);
    }

    let mut sum_ds = _mm512_reduce_add_ps(sum_ds_v);
    for j in (chunks * 16)..eff_len {
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
