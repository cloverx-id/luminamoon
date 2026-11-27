use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn ra_norm_bwd_row_avx2(
    gy: &[f32],
    x: &[f32],
    weight: Option<&[f32]>,
    dx: &mut [f32],
    mut dw_acc: Option<&mut [f32]>,
    mut db_acc: Option<&mut [f32]>,
    eps: f32,
) {
    use std::arch::x86_64::*;
    let len = x.len();
    let chunks = len / 8;
    let sign_mask = _mm256_set1_ps(-0.0_f32);
    let zero_vec = _mm256_setzero_ps();
    let one_vec = _mm256_set1_ps(1.0_f32);

    let hsum = |v: __m256| -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps::<1>(v);
        let s128 = _mm_add_ps(lo, hi);
        let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
        let s32 = _mm_add_ss(s64, _mm_movehdup_ps(s64));
        _mm_cvtss_f32(s32)
    };

    // Pass 1: accumulate sum(|x|) and sum(dy_w * x) together so c = inv_denom * sum(dy_w * x) / len.
    let mut sum_abs_v = _mm256_setzero_ps();
    let mut sum_dywx_v = _mm256_setzero_ps();
    for c in 0..chunks {
        let off = c * 8;
        let xv = _mm256_loadu_ps(x.as_ptr().add(off));
        let gyv = _mm256_loadu_ps(gy.as_ptr().add(off));
        sum_abs_v = _mm256_add_ps(sum_abs_v, _mm256_andnot_ps(sign_mask, xv));
        let dyw = match weight {
            Some(w) => _mm256_mul_ps(gyv, _mm256_loadu_ps(w.as_ptr().add(off))),
            None => gyv,
        };
        sum_dywx_v = _mm256_fmadd_ps(dyw, xv, sum_dywx_v);
    }
    let mut sum_abs = hsum(sum_abs_v);
    let mut sum_dywx = hsum(sum_dywx_v);
    for i in (chunks * 8)..len {
        let xi = x[i];
        sum_abs += xi.abs();
        let dyw = match weight {
            Some(w) => gy[i] * w[i],
            None => gy[i],
        };
        sum_dywx += dyw * xi;
    }

    let inv_len = 1.0_f32 / (len as f32);
    let inv_denom = 1.0_f32 / (sum_abs * inv_len + eps);
    let c_val = (sum_dywx * inv_denom) * inv_len;
    let inv_d_vec = _mm256_set1_ps(inv_denom);
    let c_vec = _mm256_set1_ps(c_val);

    for c in 0..chunks {
        let off = c * 8;
        let xv = _mm256_loadu_ps(x.as_ptr().add(off));
        let gyv = _mm256_loadu_ps(gy.as_ptr().add(off));
        let dyw = match weight {
            Some(w) => _mm256_mul_ps(gyv, _mm256_loadu_ps(w.as_ptr().add(off))),
            None => gyv,
        };
        let gt0 = _mm256_and_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(xv, zero_vec), one_vec);
        let lt0 = _mm256_and_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(xv, zero_vec), one_vec);
        let sgn = _mm256_sub_ps(gt0, lt0);
        let dxv = _mm256_mul_ps(inv_d_vec, _mm256_fnmadd_ps(sgn, c_vec, dyw));
        _mm256_storeu_ps(dx.as_mut_ptr().add(off), dxv);

        if let Some(ref mut dw) = dw_acc {
            let x_hat = _mm256_mul_ps(xv, inv_d_vec);
            let cur = _mm256_loadu_ps(dw.as_ptr().add(off));
            _mm256_storeu_ps(dw.as_mut_ptr().add(off), _mm256_fmadd_ps(gyv, x_hat, cur));
        }
        if let Some(ref mut db) = db_acc {
            let cur = _mm256_loadu_ps(db.as_ptr().add(off));
            _mm256_storeu_ps(db.as_mut_ptr().add(off), _mm256_add_ps(cur, gyv));
        }
    }
    for i in (chunks * 8)..len {
        let xi = x[i];
        let gyi = gy[i];
        let dyw = match weight {
            Some(w) => gyi * w[i],
            None => gyi,
        };
        let sgn = if xi > 0.0_f32 {
            1.0_f32
        } else if xi < 0.0_f32 {
            -1.0_f32
        } else {
            0.0_f32
        };
        dx[i] = inv_denom * (dyw - sgn * c_val);
        if let Some(ref mut dw) = dw_acc {
            dw[i] += gyi * (xi * inv_denom);
        }
        if let Some(ref mut db) = db_acc {
            db[i] += gyi;
        }
    }
}

fn ra_norm_bwd_row_scalar(
    gy: &[f32],
    x: &[f32],
    weight: Option<&[f32]>,
    dx: &mut [f32],
    mut dw_acc: Option<&mut [f32]>,
    mut db_acc: Option<&mut [f32]>,
    eps: f32,
) {
    let len = x.len();
    let mut sum_abs = 0.0_f32;
    let mut sum_dywx = 0.0_f32;
    for i in 0..len {
        let xi = x[i];
        sum_abs += xi.abs();
        let dyw = match weight {
            Some(w) => gy[i] * w[i],
            None => gy[i],
        };
        sum_dywx += dyw * xi;
    }
    let inv_len = 1.0_f32 / (len as f32);
    let inv_denom = 1.0_f32 / (sum_abs * inv_len + eps);
    let c_val = (sum_dywx * inv_denom) * inv_len;
    for i in 0..len {
        let xi = x[i];
        let gyi = gy[i];
        let dyw = match weight {
            Some(w) => gyi * w[i],
            None => gyi,
        };
        let sgn = if xi > 0.0_f32 {
            1.0_f32
        } else if xi < 0.0_f32 {
            -1.0_f32
        } else {
            0.0_f32
        };
        dx[i] = inv_denom * (dyw - sgn * c_val);
        if let Some(ref mut dw) = dw_acc {
            dw[i] += gyi * (xi * inv_denom);
        }
        if let Some(ref mut db) = db_acc {
            db[i] += gyi;
        }
    }
}

#[inline]
fn ra_norm_bwd_row(
    gy: &[f32],
    x: &[f32],
    weight: Option<&[f32]>,
    dx: &mut [f32],
    dw_acc: Option<&mut [f32]>,
    db_acc: Option<&mut [f32]>,
    eps: f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe {
                return ra_norm_bwd_row_avx2(gy, x, weight, dx, dw_acc, db_acc, eps);
            }
        }
    }
    ra_norm_bwd_row_scalar(gy, x, weight, dx, dw_acc, db_acc, eps);
}

#[allow(clippy::too_many_arguments)]
pub fn ra_norm_bwd_batch_f32(
    grad_y: &[f32],
    input: &[f32],
    weight: Option<&[f32]>,
    dx: &mut [f32],
    mut dw: Option<&mut [f32]>,
    mut db: Option<&mut [f32]>,
    rows: usize,
    cols: usize,
    eps: f32,
) {
    let expected = rows
        .checked_mul(cols)
        .expect("rows * cols overflow in ra_norm_bwd");
    assert_eq!(grad_y.len(), expected);
    assert_eq!(input.len(), expected);
    assert_eq!(dx.len(), expected);
    if let Some(w) = weight {
        assert!(
            w.len() >= cols,
            "weight slice shorter than cols in ra_norm_bwd"
        );
    }
    if let Some(ref w) = dw {
        assert!(w.len() >= cols, "dw slice shorter than cols in ra_norm_bwd");
    }
    if let Some(ref b) = db {
        assert!(b.len() >= cols, "db slice shorter than cols in ra_norm_bwd");
    }
    if cols == 0 {
        return;
    }
    if let Some(ref mut w) = dw {
        w[..cols].fill(0.0);
    }
    if let Some(ref mut b) = db {
        b[..cols].fill(0.0);
    }
    if rows == 0 {
        return;
    }

    let need_dw = dw.is_some();
    let need_db = db.is_some();

    if rows >= 32 && expected >= 8192 {
        let tile_rows = (rows / rayon::current_num_threads().max(1)).clamp(16, 128);
        let tile_elems = tile_rows
            .checked_mul(cols)
            .expect("tile_rows * cols overflow in ra_norm_bwd");
        let partials: Vec<(Vec<f32>, Vec<f32>)> = grad_y
            .par_chunks(tile_elems)
            .zip(input.par_chunks(tile_elems))
            .zip(dx.par_chunks_mut(tile_elems))
            .map(|((gy_t, x_t), dx_t)| {
                let mut local_dw = if need_dw {
                    vec![0.0_f32; cols]
                } else {
                    Vec::new()
                };
                let mut local_db = if need_db {
                    vec![0.0_f32; cols]
                } else {
                    Vec::new()
                };
                for ((gy_r, x_r), dx_r) in gy_t
                    .chunks_exact(cols)
                    .zip(x_t.chunks_exact(cols))
                    .zip(dx_t.chunks_exact_mut(cols))
                {
                    ra_norm_bwd_row(
                        gy_r,
                        x_r,
                        weight,
                        dx_r,
                        if need_dw {
                            Some(local_dw.as_mut_slice())
                        } else {
                            None
                        },
                        if need_db {
                            Some(local_db.as_mut_slice())
                        } else {
                            None
                        },
                        eps,
                    );
                }
                (local_dw, local_db)
            })
            .collect();
        for (p_dw, p_db) in partials {
            if let Some(ref mut dw_out) = dw {
                for j in 0..cols {
                    dw_out[j] += p_dw[j];
                }
            }
            if let Some(ref mut db_out) = db {
                for j in 0..cols {
                    db_out[j] += p_db[j];
                }
            }
        }
    } else {
        for ((gy_r, x_r), dx_r) in grad_y
            .chunks_exact(cols)
            .zip(input.chunks_exact(cols))
            .zip(dx.chunks_exact_mut(cols))
        {
            ra_norm_bwd_row(
                gy_r,
                x_r,
                weight,
                dx_r,
                dw.as_deref_mut(),
                db.as_deref_mut(),
                eps,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ra_norm_bwd_matches_scalar() {
        let rows = 40usize;
        let cols = 19usize;
        let x: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.3)
            .collect();
        let gy: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.2)
            .collect();
        let w: Vec<f32> = (0..cols).map(|i| 0.7 + (i % 5) as f32 * 0.1).collect();

        let mut dx_par = vec![0.0_f32; rows * cols];
        let mut dw_par = vec![0.0_f32; cols];
        let mut db_par = vec![0.0_f32; cols];
        ra_norm_bwd_batch_f32(
            &gy,
            &x,
            Some(&w),
            &mut dx_par,
            Some(&mut dw_par),
            Some(&mut db_par),
            rows,
            cols,
            1e-6,
        );

        let mut dx_ref = vec![0.0_f32; rows * cols];
        let mut dw_ref = vec![0.0_f32; cols];
        let mut db_ref = vec![0.0_f32; cols];
        for r in 0..rows {
            let s = r * cols;
            ra_norm_bwd_row_scalar(
                &gy[s..s + cols],
                &x[s..s + cols],
                Some(&w),
                &mut dx_ref[s..s + cols],
                Some(&mut dw_ref),
                Some(&mut db_ref),
                1e-6,
            );
        }
        for i in 0..dx_par.len() {
            assert!((dx_par[i] - dx_ref[i]).abs() < 1e-5);
        }
        for j in 0..cols {
            assert!((dw_par[j] - dw_ref[j]).abs() < 1e-4);
            assert!((db_par[j] - db_ref[j]).abs() < 1e-4);
        }
    }
}
