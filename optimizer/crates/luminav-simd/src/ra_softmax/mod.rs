#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
mod batch;
mod scalar;

pub use batch::{
    ra_attn_scores_bwd_batch_f32, ra_attn_scores_bwd_inplace_batch_f32,
    ra_attn_weights_fwd_batch_f32, ra_attn_weights_fwd_inplace_batch_f32, ra_softmax_batch_f32,
    ra_softmax_bwd_batch_f32,
};

#[inline]
pub fn ra_softmax_row(scores: &[f32], output: &mut [f32], scale: f32) {
    let len = scores.len();
    assert_eq!(
        output.len(),
        len,
        "Output slice length mismatch in ra_softmax"
    );
    if len == 0 {
        return;
    }
    let _ = ra_softmax_row_with_stats(scores, output, scale, len);
}

#[inline]
pub fn ra_softmax_row_with_stats(
    scores: &[f32],
    output: &mut [f32],
    scale: f32,
    active_len: usize,
) -> (f32, f32, i64) {
    let len = scores.len();
    assert_eq!(
        output.len(),
        len,
        "Output slice length mismatch in ra_softmax"
    );
    let eff_len = active_len.min(len);
    if eff_len < len {
        output[eff_len..len].fill(0.0);
    }
    if eff_len == 0 {
        return (f32::NEG_INFINITY, 1e-38, 0);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe {
                avx512::ra_softmax_row_with_stats_avx512(scores, output, scale, eff_len)
            };
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { avx2::ra_softmax_row_with_stats_avx2(scores, output, scale, eff_len) };
        }
    }
    scalar::ra_softmax_row_with_stats_scalar(scores, output, scale, eff_len)
}

#[inline]
pub fn ra_softmax_row_with_stats_inplace(
    buf: &mut [f32],
    scale: f32,
    active_len: usize,
) -> (f32, f32, i64) {
    let len = buf.len();
    let eff_len = active_len.min(len);
    if eff_len < len {
        buf[eff_len..len].fill(0.0);
    }
    if eff_len == 0 {
        return (f32::NEG_INFINITY, 1e-38, 0);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe {
                avx512::ra_softmax_row_with_stats_inplace_avx512(buf, scale, eff_len)
            };
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { avx2::ra_softmax_row_with_stats_inplace_avx2(buf, scale, eff_len) };
        }
    }
    scalar::ra_softmax_row_with_stats_inplace_scalar(buf, scale, eff_len)
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn ra_softmax_row_bwd_factored(
    scores: &[f32],
    grad_p: &[f32],
    out_ds: &mut [f32],
    mut out_p: Option<&mut [f32]>,
    m_val: f32,
    l_val: f32,
    m_idx: i64,
    d_val: f32,
    scale: f32,
    active_len: usize,
) {
    let len = scores.len();
    assert!(
        grad_p.len() >= len,
        "grad_p slice shorter than scores in ra_softmax_row_bwd_factored"
    );
    assert!(
        out_ds.len() >= len,
        "out_ds slice shorter than scores in ra_softmax_row_bwd_factored"
    );
    if let Some(ref p_buf) = out_p {
        assert!(
            p_buf.len() >= len,
            "out_p slice shorter than scores in ra_softmax_row_bwd_factored"
        );
    }
    let eff_len = active_len.min(len);
    if eff_len < len {
        out_ds[eff_len..len].fill(0.0);
        if let Some(ref mut p_buf) = out_p {
            p_buf[eff_len..len].fill(0.0);
        }
    }
    if eff_len == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe {
                avx512::ra_softmax_row_bwd_avx512(
                    scores, grad_p, out_ds, out_p, m_val, l_val, m_idx, d_val, scale, eff_len,
                );
            }
            return;
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe {
                avx2::ra_softmax_row_bwd_avx2(
                    scores, grad_p, out_ds, out_p, m_val, l_val, m_idx, d_val, scale, eff_len,
                );
            }
            return;
        }
    }
    scalar::ra_softmax_row_bwd_scalar(
        scores, grad_p, out_ds, out_p, m_val, l_val, m_idx, d_val, scale, eff_len,
    );
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn ra_softmax_row_bwd_inplace_factored(
    scores_to_p: &mut [f32],
    dp_to_ds: &mut [f32],
    m_val: f32,
    l_val: f32,
    m_idx: i64,
    d_val: f32,
    scale: f32,
    active_len: usize,
) {
    let len = scores_to_p.len();
    assert!(
        dp_to_ds.len() >= len,
        "dp_to_ds slice shorter than scores_to_p in ra_softmax_row_bwd_inplace_factored"
    );
    let eff_len = active_len.min(len);
    if eff_len < len {
        scores_to_p[eff_len..len].fill(0.0);
        dp_to_ds[eff_len..len].fill(0.0);
    }
    if eff_len == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe {
                avx512::ra_softmax_row_bwd_inplace_avx512(
                    scores_to_p,
                    dp_to_ds,
                    m_val,
                    l_val,
                    m_idx,
                    d_val,
                    scale,
                    eff_len,
                );
            }
            return;
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe {
                avx2::ra_softmax_row_bwd_inplace_avx2(
                    scores_to_p,
                    dp_to_ds,
                    m_val,
                    l_val,
                    m_idx,
                    d_val,
                    scale,
                    eff_len,
                );
            }
            return;
        }
    }
    scalar::ra_softmax_row_bwd_inplace_scalar(
        scores_to_p,
        dp_to_ds,
        m_val,
        l_val,
        m_idx,
        d_val,
        scale,
        eff_len,
    );
}
