#[inline]
pub(super) fn ra_softmax_row_with_stats_scalar(
    scores: &[f32],
    output: &mut [f32],
    scale: f32,
    eff_len: usize,
) -> (f32, f32, i64) {
    let mut max_val = f32::NEG_INFINITY;
    let mut m_idx: i64 = 0;
    for (j, &s) in scores[..eff_len].iter().enumerate() {
        if s > max_val {
            max_val = s;
            m_idx = j as i64;
        }
    }

    let mut sum_phi = 0.0_f32;
    for i in 0..eff_len {
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
        sum_phi += phi;
    }

    let l_val = sum_phi.max(1e-38);
    let inv_sum = 1.0_f32 / l_val;
    for x in output[..eff_len].iter_mut() {
        *x *= inv_sum;
    }
    (max_val, l_val, m_idx)
}

#[inline]
pub(super) fn ra_softmax_row_with_stats_inplace_scalar(
    buf: &mut [f32],
    scale: f32,
    eff_len: usize,
) -> (f32, f32, i64) {
    let mut max_val = f32::NEG_INFINITY;
    let mut m_idx: i64 = 0;
    for (j, &s) in buf[..eff_len].iter().enumerate() {
        if s > max_val {
            max_val = s;
            m_idx = j as i64;
        }
    }
    let mut sum_phi = 0.0_f32;
    for x in buf[..eff_len].iter_mut() {
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
        sum_phi += phi;
    }
    let l_val = sum_phi.max(1e-38);
    let inv_sum = 1.0_f32 / l_val;
    for x in buf[..eff_len].iter_mut() {
        *x *= inv_sum;
    }
    (max_val, l_val, m_idx)
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn ra_softmax_row_bwd_scalar(
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
    let inv_l = 1.0_f32 / l_val.max(1e-38);
    let coeff = (16.0_f32 * scale) * inv_l;
    let mut sum_ds = 0.0_f32;

    for j in 0..eff_len {
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

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn ra_softmax_row_bwd_inplace_scalar(
    scores_to_p: &mut [f32],
    dp_to_ds: &mut [f32],
    m_val: f32,
    l_val: f32,
    m_idx: i64,
    d_val: f32,
    scale: f32,
    eff_len: usize,
) {
    let inv_l = 1.0_f32 / l_val.max(1e-38);
    let coeff = (16.0_f32 * scale) * inv_l;
    let mut sum_ds = 0.0_f32;
    for j in 0..eff_len {
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
