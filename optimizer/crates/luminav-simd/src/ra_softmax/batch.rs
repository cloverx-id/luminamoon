use rayon::prelude::*;

use super::{
    ra_softmax_row_bwd_factored, ra_softmax_row_bwd_inplace_factored, ra_softmax_row_with_stats,
    ra_softmax_row_with_stats_inplace,
};

const PARALLEL_ELEM_THRESHOLD: usize = 16_384;

#[inline]
fn row_active_len(r: usize, cols: usize, n_q: usize, q_start: usize, is_causal: bool) -> usize {
    if is_causal && n_q > 0 {
        ((q_start + (r % n_q)) + 1).min(cols)
    } else {
        cols
    }
}

pub fn ra_softmax_batch_f32(
    scores: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    scale: f32,
) {
    if rows == 0 || cols == 0 {
        return;
    }
    if rows * cols >= PARALLEL_ELEM_THRESHOLD && rows > 1 {
        scores
            .par_chunks_exact(cols)
            .zip(output.par_chunks_exact_mut(cols))
            .for_each(|(s_row, o_row)| {
                let _ = ra_softmax_row_with_stats(s_row, o_row, scale, cols);
            });
    } else {
        for r in 0..rows {
            let off = r * cols;
            let _ = ra_softmax_row_with_stats(
                &scores[off..off + cols],
                &mut output[off..off + cols],
                scale,
                cols,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn ra_attn_weights_fwd_batch_f32(
    scores: &[f32],
    out_p: &mut [f32],
    out_m: &mut [f32],
    out_l: &mut [f32],
    out_m_idx: &mut [i64],
    rows: usize,
    cols: usize,
    n_q: usize,
    q_start: usize,
    is_causal: bool,
    scale: f32,
) {
    if rows == 0 || cols == 0 {
        return;
    }
    if rows * cols >= PARALLEL_ELEM_THRESHOLD && rows > 1 {
        scores
            .par_chunks_exact(cols)
            .zip(out_p.par_chunks_exact_mut(cols))
            .zip(
                out_m
                    .par_iter_mut()
                    .zip(out_l.par_iter_mut())
                    .zip(out_m_idx.par_iter_mut()),
            )
            .enumerate()
            .for_each(|(r, ((s_row, p_row), ((m_slot, l_slot), idx_slot)))| {
                let active_len = row_active_len(r, cols, n_q, q_start, is_causal);
                let (m_val, l_val, m_idx) =
                    ra_softmax_row_with_stats(s_row, p_row, scale, active_len);
                *m_slot = m_val;
                *l_slot = l_val;
                *idx_slot = m_idx;
            });
    } else {
        for r in 0..rows {
            let off = r * cols;
            let active_len = row_active_len(r, cols, n_q, q_start, is_causal);
            let (m_val, l_val, m_idx) = ra_softmax_row_with_stats(
                &scores[off..off + cols],
                &mut out_p[off..off + cols],
                scale,
                active_len,
            );
            out_m[r] = m_val;
            out_l[r] = l_val;
            out_m_idx[r] = m_idx;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn ra_attn_weights_fwd_inplace_batch_f32(
    buf: &mut [f32],
    out_m: &mut [f32],
    out_l: &mut [f32],
    out_m_idx: &mut [i64],
    rows: usize,
    cols: usize,
    n_q: usize,
    q_start: usize,
    is_causal: bool,
    scale: f32,
) {
    if rows == 0 || cols == 0 {
        return;
    }
    if rows * cols >= PARALLEL_ELEM_THRESHOLD && rows > 1 {
        buf.par_chunks_exact_mut(cols)
            .zip(
                out_m
                    .par_iter_mut()
                    .zip(out_l.par_iter_mut())
                    .zip(out_m_idx.par_iter_mut()),
            )
            .enumerate()
            .for_each(|(r, (row, ((m_slot, l_slot), idx_slot)))| {
                let active_len = row_active_len(r, cols, n_q, q_start, is_causal);
                let (m_val, l_val, m_idx) =
                    ra_softmax_row_with_stats_inplace(row, scale, active_len);
                *m_slot = m_val;
                *l_slot = l_val;
                *idx_slot = m_idx;
            });
    } else {
        for r in 0..rows {
            let off = r * cols;
            let active_len = row_active_len(r, cols, n_q, q_start, is_causal);
            let (m_val, l_val, m_idx) =
                ra_softmax_row_with_stats_inplace(&mut buf[off..off + cols], scale, active_len);
            out_m[r] = m_val;
            out_l[r] = l_val;
            out_m_idx[r] = m_idx;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn ra_attn_scores_bwd_batch_f32(
    scores: &[f32],
    dp: &[f32],
    m_vals: &[f32],
    l_vals: &[f32],
    m_idxs: &[i64],
    d_vals: &[f32],
    out_p: &mut [f32],
    out_ds: &mut [f32],
    rows: usize,
    cols: usize,
    n_q: usize,
    q_start: usize,
    is_causal: bool,
    scale: f32,
) {
    if rows == 0 || cols == 0 {
        return;
    }
    if rows * cols >= PARALLEL_ELEM_THRESHOLD && rows > 1 {
        scores
            .par_chunks_exact(cols)
            .zip(dp.par_chunks_exact(cols))
            .zip(out_p.par_chunks_exact_mut(cols))
            .zip(out_ds.par_chunks_exact_mut(cols))
            .enumerate()
            .for_each(|(r, (((s_row, dp_row), p_row), ds_row))| {
                let active_len = row_active_len(r, cols, n_q, q_start, is_causal);
                ra_softmax_row_bwd_factored(
                    s_row,
                    dp_row,
                    ds_row,
                    Some(p_row),
                    m_vals[r],
                    l_vals[r],
                    m_idxs[r],
                    d_vals[r],
                    scale,
                    active_len,
                );
            });
    } else {
        for r in 0..rows {
            let off = r * cols;
            let active_len = row_active_len(r, cols, n_q, q_start, is_causal);
            ra_softmax_row_bwd_factored(
                &scores[off..off + cols],
                &dp[off..off + cols],
                &mut out_ds[off..off + cols],
                Some(&mut out_p[off..off + cols]),
                m_vals[r],
                l_vals[r],
                m_idxs[r],
                d_vals[r],
                scale,
                active_len,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn ra_attn_scores_bwd_inplace_batch_f32(
    scores_to_p: &mut [f32],
    dp_to_ds: &mut [f32],
    m_vals: &[f32],
    l_vals: &[f32],
    m_idxs: &[i64],
    d_vals: &[f32],
    rows: usize,
    cols: usize,
    n_q: usize,
    q_start: usize,
    is_causal: bool,
    scale: f32,
) {
    if rows == 0 || cols == 0 {
        return;
    }
    if rows * cols >= PARALLEL_ELEM_THRESHOLD && rows > 1 {
        scores_to_p
            .par_chunks_exact_mut(cols)
            .zip(dp_to_ds.par_chunks_exact_mut(cols))
            .enumerate()
            .for_each(|(r, (sp_row, dd_row))| {
                let active_len = row_active_len(r, cols, n_q, q_start, is_causal);
                ra_softmax_row_bwd_inplace_factored(
                    sp_row, dd_row, m_vals[r], l_vals[r], m_idxs[r], d_vals[r], scale, active_len,
                );
            });
    } else {
        for r in 0..rows {
            let off = r * cols;
            let active_len = row_active_len(r, cols, n_q, q_start, is_causal);
            ra_softmax_row_bwd_inplace_factored(
                &mut scores_to_p[off..off + cols],
                &mut dp_to_ds[off..off + cols],
                m_vals[r],
                l_vals[r],
                m_idxs[r],
                d_vals[r],
                scale,
                active_len,
            );
        }
    }
}

pub fn ra_softmax_bwd_batch_f32(
    scores: &[f32],
    p_in: &[f32],
    grad_out: &[f32],
    out_ds: &mut [f32],
    rows: usize,
    cols: usize,
    scale: f32,
) {
    if rows == 0 || cols == 0 {
        return;
    }
    let process_row = |s_row: &[f32], p_row: &[f32], g_row: &[f32], ds_row: &mut [f32]| {
        let mut max_val = f32::NEG_INFINITY;
        let mut m_idx: i64 = 0;
        let mut sum_gp = 0.0_f32;
        for j in 0..cols {
            let s = s_row[j];
            if s > max_val {
                max_val = s;
                m_idx = j as i64;
            }
            if s.is_finite() {
                sum_gp += g_row[j] * p_row[j];
            }
        }
        let mut sum_phi = 0.0_f32;
        for &s in s_row {
            if s.is_finite() {
                let delta = if s == max_val { 0.0 } else { max_val - s };
                let u = delta * scale;
                if u < 1.0 {
                    let q = (1.0 - u) / (1.0 + u);
                    let q2 = q * q;
                    let q4 = q2 * q2;
                    sum_phi += q4 * q4;
                }
            }
        }
        let l_val = sum_phi.max(1e-38);
        ra_softmax_row_bwd_factored(
            s_row, g_row, ds_row, None, max_val, l_val, m_idx, sum_gp, scale, cols,
        );
    };

    if rows * cols >= PARALLEL_ELEM_THRESHOLD && rows > 1 {
        scores
            .par_chunks_exact(cols)
            .zip(p_in.par_chunks_exact(cols))
            .zip(grad_out.par_chunks_exact(cols))
            .zip(out_ds.par_chunks_exact_mut(cols))
            .for_each(|(((s_r, p_r), g_r), ds_r)| process_row(s_r, p_r, g_r, ds_r));
    } else {
        for r in 0..rows {
            let off = r * cols;
            process_row(
                &scores[off..off + cols],
                &p_in[off..off + cols],
                &grad_out[off..off + cols],
                &mut out_ds[off..off + cols],
            );
        }
    }
}
