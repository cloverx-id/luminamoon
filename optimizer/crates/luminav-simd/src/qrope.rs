use rayon::prelude::*;

const PARALLEL_QROPE_ROWS: usize = 256;

#[inline(always)]
fn qrope_row_f32(
    x_row: &[f32],
    out_row: &mut [f32],
    cl_row: &[f32],
    sl_row: &[f32],
    cr_row: &[f32],
    sr_row: &[f32],
    sign: f32,
) {
    let nq = cl_row.len();
    for q in 0..nq {
        let base = q * 4;
        let w = x_row[base];
        let i = x_row[base + 1];
        let j = x_row[base + 2];
        let k = x_row[base + 3];

        let cl = cl_row[q];
        let sl = sl_row[q] * sign;
        let cr = cr_row[q];
        let sr = sr_row[q] * sign;

        let yw = cl * w - sl * (i + j + k);
        let yi = cl * i + sl * (w + k - j);
        let yj = cl * j + sl * (w + i - k);
        let yk = cl * k + sl * (w + j - i);

        out_row[base] = cr * yw - sr * (yi - yj);
        out_row[base + 1] = cr * yi + sr * (yw + yk);
        out_row[base + 2] = cr * yj + sr * (yk - yw);
        out_row[base + 3] = cr * yk - sr * (yi + yj);
    }
}

/// Applies Spin(4) double-quaternion rotary embedding over contiguous `(B*H, S, 4*Q)` float32 buffers.
/// `sign = +1.0` for forward rotation, `sign = -1.0` for orthogonal inverse (backward).
#[allow(clippy::too_many_arguments)]
pub fn qrope_batch_f32(
    x: &[f32],
    out: &mut [f32],
    cos_l: &[f32],
    sin_l: &[f32],
    cos_r: &[f32],
    sin_r: &[f32],
    batch_heads: usize,
    seq_len: usize,
    num_quaternions: usize,
    sign: f32,
) {
    let d_head = num_quaternions
        .checked_mul(4)
        .expect("num_quaternions * 4 overflow");
    let total_rows = batch_heads
        .checked_mul(seq_len)
        .expect("batch_heads * seq_len overflow");
    let expected = total_rows
        .checked_mul(d_head)
        .expect("total_rows * d_head overflow");
    assert_eq!(x.len(), expected);
    assert_eq!(out.len(), expected);
    let trig_len = seq_len
        .checked_mul(num_quaternions)
        .expect("seq_len * num_quaternions overflow");
    assert_eq!(cos_l.len(), trig_len);
    assert_eq!(sin_l.len(), trig_len);
    assert_eq!(cos_r.len(), trig_len);
    assert_eq!(sin_r.len(), trig_len);
    if total_rows == 0 || num_quaternions == 0 {
        return;
    }

    if total_rows >= PARALLEL_QROPE_ROWS {
        x.par_chunks_exact(d_head)
            .zip(out.par_chunks_exact_mut(d_head))
            .enumerate()
            .for_each(|(row_idx, (x_row, out_row))| {
                let pos = row_idx % seq_len;
                let t_off = pos * num_quaternions;
                qrope_row_f32(
                    x_row,
                    out_row,
                    &cos_l[t_off..t_off + num_quaternions],
                    &sin_l[t_off..t_off + num_quaternions],
                    &cos_r[t_off..t_off + num_quaternions],
                    &sin_r[t_off..t_off + num_quaternions],
                    sign,
                );
            });
    } else {
        for (row_idx, (x_row, out_row)) in x
            .chunks_exact(d_head)
            .zip(out.chunks_exact_mut(d_head))
            .enumerate()
        {
            let pos = row_idx % seq_len;
            let t_off = pos * num_quaternions;
            qrope_row_f32(
                x_row,
                out_row,
                &cos_l[t_off..t_off + num_quaternions],
                &sin_l[t_off..t_off + num_quaternions],
                &cos_r[t_off..t_off + num_quaternions],
                &sin_r[t_off..t_off + num_quaternions],
                sign,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qrope_orthogonality_and_invertibility() {
        let bh = 8usize;
        let seq_len = 64usize;
        let nq = 8usize;
        let d = nq * 4;
        let x: Vec<f32> = (0..bh * seq_len * d)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.15)
            .collect();
        let inv_sqrt3 = 1.0_f32 / 3.0_f32.sqrt();
        let inv_sqrt2 = 1.0_f32 / 2.0_f32.sqrt();
        let mut cl = vec![0.0_f32; seq_len * nq];
        let mut sl = vec![0.0_f32; seq_len * nq];
        let mut cr = vec![0.0_f32; seq_len * nq];
        let mut sr = vec![0.0_f32; seq_len * nq];
        for p in 0..seq_len {
            for q in 0..nq {
                let tl = (p as f32) * 0.05 / ((q + 1) as f32);
                let tr = tl * 0.618_034;
                cl[p * nq + q] = tl.cos();
                sl[p * nq + q] = tl.sin() * inv_sqrt3;
                cr[p * nq + q] = tr.cos();
                sr[p * nq + q] = tr.sin() * inv_sqrt2;
            }
        }
        let mut y = vec![0.0_f32; x.len()];
        let mut x_rec = vec![0.0_f32; x.len()];
        qrope_batch_f32(&x, &mut y, &cl, &sl, &cr, &sr, bh, seq_len, nq, 1.0);
        qrope_batch_f32(&y, &mut x_rec, &cl, &sl, &cr, &sr, bh, seq_len, nq, -1.0);
        for i in 0..x.len() {
            assert!((x[i] - x_rec[i]).abs() < 1e-5);
        }
    }
}
