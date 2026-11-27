use rayon::prelude::*;

use crate::ra::{ra_glu_slice, ra_swiglu_slice};

const PARALLEL_ELEM_THRESHOLD: usize = 32_768;

#[inline(always)]
fn ra_tanh_bwd_scalar_elem(g: f32, x: f32, b: f32) -> f32 {
    let abs_x = x.abs();
    let inv_m = 1.0_f32 / b.max(abs_x);
    let ub = b * inv_m;
    let ua = abs_x * inv_m;
    let inv_dn = 1.0_f32 / (ub * (ub + ua) + ua * ua);
    let deriv = (ub * ub * ub) * (ub + 2.0_f32 * ua) * (inv_dn * inv_dn);
    g * deriv
}

#[inline(always)]
fn ra_swiglu_bwd_scalar_elem(g: f32, z: f32, u: f32, b: f32) -> (f32, f32) {
    let abs_z = z.abs();
    let inv_m = 1.0_f32 / b.max(abs_z);
    let ub = b * inv_m;
    let ua = abs_z * inv_m;
    let uz = z * inv_m;
    let inv_dn = 1.0_f32 / (ub * (ub + ua) + ua * ua);
    let q = (0.5_f32 * ub * ub) * inv_dn;
    let (s, ds_dz) = if z >= 0.0_f32 {
        let sigma = 1.0_f32 - q;
        (z * sigma, sigma + q * ua * (ub + 2.0_f32 * ua) * inv_dn)
    } else {
        (
            (0.5_f32 * b * ub) * uz * inv_dn,
            q * (ub * ub - ua * ua) * inv_dn,
        )
    };
    (g * u * ds_dz, g * s)
}

#[inline(always)]
fn ra_glu_bwd_scalar_elem(g: f32, z: f32, u: f32, b: f32) -> (f32, f32) {
    let abs_z = z.abs();
    let inv_m = 1.0_f32 / b.max(abs_z);
    let ub = b * inv_m;
    let ua = abs_z * inv_m;
    let uz = z * inv_m;
    let inv_dn = 1.0_f32 / (ub * (ub + ua) + ua * ua);
    let ra_t = b * (ub + ua) * uz * inv_dn;
    let deriv = (ub * ub * ub) * (ub + 2.0_f32 * ua) * (inv_dn * inv_dn);
    (g * u * deriv, g * ra_t)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn ra_tanh_bwd_chunk_avx2(grad_out: &[f32], input: &[f32], dx: &mut [f32], b: f32) {
    use std::arch::x86_64::*;
    let len = input.len();
    let chunks = len / 8;
    let b_vec = _mm256_set1_ps(b);
    let one_vec = _mm256_set1_ps(1.0_f32);
    let two_vec = _mm256_set1_ps(2.0_f32);
    let sign_mask = _mm256_set1_ps(-0.0_f32);

    for c in 0..chunks {
        let off = c * 8;
        let g = _mm256_loadu_ps(grad_out.as_ptr().add(off));
        let x = _mm256_loadu_ps(input.as_ptr().add(off));
        let abs_x = _mm256_andnot_ps(sign_mask, x);
        let inv_m = _mm256_div_ps(one_vec, _mm256_max_ps(b_vec, abs_x));
        let ub = _mm256_mul_ps(b_vec, inv_m);
        let ua = _mm256_mul_ps(abs_x, inv_m);
        let dn = _mm256_fmadd_ps(ua, ua, _mm256_mul_ps(ub, _mm256_add_ps(ub, ua)));
        let inv_dn = _mm256_div_ps(one_vec, dn);
        let ub2 = _mm256_mul_ps(ub, ub);
        let ub3 = _mm256_mul_ps(ub2, ub);
        let factor = _mm256_fmadd_ps(two_vec, ua, ub);
        let inv_dn2 = _mm256_mul_ps(inv_dn, inv_dn);
        let deriv = _mm256_mul_ps(_mm256_mul_ps(ub3, factor), inv_dn2);
        _mm256_storeu_ps(dx.as_mut_ptr().add(off), _mm256_mul_ps(g, deriv));
    }
    for i in (chunks * 8)..len {
        dx[i] = ra_tanh_bwd_scalar_elem(grad_out[i], input[i], b);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn ra_swiglu_bwd_chunk_avx2(
    grad_out: &[f32],
    gate: &[f32],
    up: &[f32],
    dgate: &mut [f32],
    dup: &mut [f32],
    b: f32,
) {
    use std::arch::x86_64::*;
    let len = gate.len();
    let chunks = len / 8;
    let b_vec = _mm256_set1_ps(b);
    let half_vec = _mm256_set1_ps(0.5_f32);
    let half_b_vec = _mm256_set1_ps(0.5_f32 * b);
    let one_vec = _mm256_set1_ps(1.0_f32);
    let two_vec = _mm256_set1_ps(2.0_f32);
    let zero_vec = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0_f32);

    for c in 0..chunks {
        let off = c * 8;
        let g = _mm256_loadu_ps(grad_out.as_ptr().add(off));
        let z = _mm256_loadu_ps(gate.as_ptr().add(off));
        let u = _mm256_loadu_ps(up.as_ptr().add(off));

        let abs_z = _mm256_andnot_ps(sign_mask, z);
        let inv_m = _mm256_div_ps(one_vec, _mm256_max_ps(b_vec, abs_z));
        let ub = _mm256_mul_ps(b_vec, inv_m);
        let ua = _mm256_mul_ps(abs_z, inv_m);
        let uz = _mm256_mul_ps(z, inv_m);
        let ub2 = _mm256_mul_ps(ub, ub);
        let ua2 = _mm256_mul_ps(ua, ua);
        let dn = _mm256_fmadd_ps(ub, _mm256_add_ps(ub, ua), ua2);
        let inv_dn = _mm256_div_ps(one_vec, dn);
        let q = _mm256_mul_ps(_mm256_mul_ps(half_vec, ub2), inv_dn);
        let sigma_pos = _mm256_sub_ps(one_vec, q);

        let pos_s = _mm256_mul_ps(z, sigma_pos);
        let neg_s = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(half_b_vec, ub), uz), inv_dn);
        let pos_ds = _mm256_fmadd_ps(
            _mm256_mul_ps(_mm256_mul_ps(q, ua), _mm256_fmadd_ps(two_vec, ua, ub)),
            inv_dn,
            sigma_pos,
        );
        let neg_ds = _mm256_mul_ps(_mm256_mul_ps(q, _mm256_sub_ps(ub2, ua2)), inv_dn);

        let ge_zero = _mm256_cmp_ps::<_CMP_GE_OQ>(z, zero_vec);
        let s = _mm256_blendv_ps(neg_s, pos_s, ge_zero);
        let ds_dz = _mm256_blendv_ps(neg_ds, pos_ds, ge_zero);

        _mm256_storeu_ps(
            dgate.as_mut_ptr().add(off),
            _mm256_mul_ps(_mm256_mul_ps(g, u), ds_dz),
        );
        _mm256_storeu_ps(dup.as_mut_ptr().add(off), _mm256_mul_ps(g, s));
    }
    for i in (chunks * 8)..len {
        let (dg, du) = ra_swiglu_bwd_scalar_elem(grad_out[i], gate[i], up[i], b);
        dgate[i] = dg;
        dup[i] = du;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn ra_glu_bwd_chunk_avx2(
    grad_out: &[f32],
    gate: &[f32],
    up: &[f32],
    dgate: &mut [f32],
    dup: &mut [f32],
    b: f32,
) {
    use std::arch::x86_64::*;
    let len = gate.len();
    let chunks = len / 8;
    let b_vec = _mm256_set1_ps(b);
    let one_vec = _mm256_set1_ps(1.0_f32);
    let two_vec = _mm256_set1_ps(2.0_f32);
    let sign_mask = _mm256_set1_ps(-0.0_f32);

    for c in 0..chunks {
        let off = c * 8;
        let g = _mm256_loadu_ps(grad_out.as_ptr().add(off));
        let z = _mm256_loadu_ps(gate.as_ptr().add(off));
        let u = _mm256_loadu_ps(up.as_ptr().add(off));

        let abs_z = _mm256_andnot_ps(sign_mask, z);
        let inv_m = _mm256_div_ps(one_vec, _mm256_max_ps(b_vec, abs_z));
        let ub = _mm256_mul_ps(b_vec, inv_m);
        let ua = _mm256_mul_ps(abs_z, inv_m);
        let uz = _mm256_mul_ps(z, inv_m);
        let ub_plus_ua = _mm256_add_ps(ub, ua);
        let dn = _mm256_fmadd_ps(ua, ua, _mm256_mul_ps(ub, ub_plus_ua));
        let inv_dn = _mm256_div_ps(one_vec, dn);

        let ra_t = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(b_vec, ub_plus_ua), uz), inv_dn);
        let ub2 = _mm256_mul_ps(ub, ub);
        let ub3 = _mm256_mul_ps(ub2, ub);
        let deriv = _mm256_mul_ps(
            _mm256_mul_ps(ub3, _mm256_fmadd_ps(two_vec, ua, ub)),
            _mm256_mul_ps(inv_dn, inv_dn),
        );
        _mm256_storeu_ps(
            dgate.as_mut_ptr().add(off),
            _mm256_mul_ps(_mm256_mul_ps(g, u), deriv),
        );
        _mm256_storeu_ps(dup.as_mut_ptr().add(off), _mm256_mul_ps(g, ra_t));
    }
    for i in (chunks * 8)..len {
        let (dg, du) = ra_glu_bwd_scalar_elem(grad_out[i], gate[i], up[i], b);
        dgate[i] = dg;
        dup[i] = du;
    }
}

#[inline]
fn ra_tanh_bwd_chunk(grad_out: &[f32], input: &[f32], dx: &mut [f32], b: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe {
                return ra_tanh_bwd_chunk_avx2(grad_out, input, dx, b);
            }
        }
    }
    for i in 0..input.len() {
        dx[i] = ra_tanh_bwd_scalar_elem(grad_out[i], input[i], b);
    }
}

#[inline]
fn ra_swiglu_bwd_chunk(
    grad_out: &[f32],
    gate: &[f32],
    up: &[f32],
    dgate: &mut [f32],
    dup: &mut [f32],
    b: f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe {
                return ra_swiglu_bwd_chunk_avx2(grad_out, gate, up, dgate, dup, b);
            }
        }
    }
    for i in 0..gate.len() {
        let (dg, du) = ra_swiglu_bwd_scalar_elem(grad_out[i], gate[i], up[i], b);
        dgate[i] = dg;
        dup[i] = du;
    }
}

#[inline]
fn ra_glu_bwd_chunk(
    grad_out: &[f32],
    gate: &[f32],
    up: &[f32],
    dgate: &mut [f32],
    dup: &mut [f32],
    b: f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe {
                return ra_glu_bwd_chunk_avx2(grad_out, gate, up, dgate, dup, b);
            }
        }
    }
    for i in 0..gate.len() {
        let (dg, du) = ra_glu_bwd_scalar_elem(grad_out[i], gate[i], up[i], b);
        dgate[i] = dg;
        dup[i] = du;
    }
}

pub fn ra_tanh_bwd_slice(grad_out: &[f32], input: &[f32], dx: &mut [f32], b: f32) {
    let len = input.len();
    assert_eq!(grad_out.len(), len);
    assert_eq!(dx.len(), len);
    if len == 0 {
        return;
    }
    if len >= PARALLEL_ELEM_THRESHOLD {
        let chunk = 16_384;
        grad_out
            .par_chunks(chunk)
            .zip(input.par_chunks(chunk))
            .zip(dx.par_chunks_mut(chunk))
            .for_each(|((g_c, x_c), dx_c)| ra_tanh_bwd_chunk(g_c, x_c, dx_c, b));
    } else {
        ra_tanh_bwd_chunk(grad_out, input, dx, b);
    }
}

fn ra_gated_bwd_slice_impl(
    grad_out: &[f32],
    gate: &[f32],
    up: &[f32],
    dgate: &mut [f32],
    dup: &mut [f32],
    b: f32,
    is_swiglu: bool,
) {
    let len = gate.len();
    assert_eq!(grad_out.len(), len);
    assert_eq!(up.len(), len);
    assert_eq!(dgate.len(), len);
    assert_eq!(dup.len(), len);
    if len == 0 {
        return;
    }
    let f = if is_swiglu {
        ra_swiglu_bwd_chunk
    } else {
        ra_glu_bwd_chunk
    };
    if len >= PARALLEL_ELEM_THRESHOLD {
        let chunk = 16_384;
        grad_out
            .par_chunks(chunk)
            .zip(gate.par_chunks(chunk))
            .zip(up.par_chunks(chunk))
            .zip(dgate.par_chunks_mut(chunk))
            .zip(dup.par_chunks_mut(chunk))
            .for_each(|((((g_c, z_c), u_c), dg_c), du_c)| f(g_c, z_c, u_c, dg_c, du_c, b));
    } else {
        f(grad_out, gate, up, dgate, dup, b);
    }
}

pub fn ra_swiglu_bwd_slice(
    grad_out: &[f32],
    gate: &[f32],
    up: &[f32],
    dgate: &mut [f32],
    dup: &mut [f32],
    b: f32,
) {
    ra_gated_bwd_slice_impl(grad_out, gate, up, dgate, dup, b, true);
}

pub fn ra_glu_bwd_slice(
    grad_out: &[f32],
    gate: &[f32],
    up: &[f32],
    dgate: &mut [f32],
    dup: &mut [f32],
    b: f32,
) {
    ra_gated_bwd_slice_impl(grad_out, gate, up, dgate, dup, b, false);
}

pub fn ra_gated_packed_fwd_batch_f32(
    gu: &[f32],
    output: &mut [f32],
    rows: usize,
    hidden: usize,
    b: f32,
    is_swiglu: bool,
) {
    let two_h = hidden.checked_mul(2).expect("2 * hidden overflow");
    let total_out = rows.checked_mul(hidden).expect("rows * hidden overflow");
    let total_gu = rows.checked_mul(two_h).expect("rows * 2 * hidden overflow");
    assert_eq!(gu.len(), total_gu);
    assert_eq!(output.len(), total_out);
    if rows == 0 || hidden == 0 {
        return;
    }
    let row_fn = |gu_row: &[f32], out_row: &mut [f32]| {
        let (gate_row, up_row) = gu_row.split_at(hidden);
        if is_swiglu {
            ra_swiglu_slice(gate_row, up_row, out_row, b);
        } else {
            ra_glu_slice(gate_row, up_row, out_row, b);
        }
    };
    if total_out >= PARALLEL_ELEM_THRESHOLD && rows > 1 {
        gu.par_chunks_exact(two_h)
            .zip(output.par_chunks_exact_mut(hidden))
            .for_each(|(gu_row, out_row)| row_fn(gu_row, out_row));
    } else {
        for (gu_row, out_row) in gu.chunks_exact(two_h).zip(output.chunks_exact_mut(hidden)) {
            row_fn(gu_row, out_row);
        }
    }
}

pub fn ra_gated_packed_bwd_batch_f32(
    grad_out: &[f32],
    gu: &[f32],
    dgu: &mut [f32],
    rows: usize,
    hidden: usize,
    b: f32,
    is_swiglu: bool,
) {
    let two_h = hidden.checked_mul(2).expect("2 * hidden overflow");
    let total_out = rows.checked_mul(hidden).expect("rows * hidden overflow");
    let total_gu = rows.checked_mul(two_h).expect("rows * 2 * hidden overflow");
    assert_eq!(grad_out.len(), total_out);
    assert_eq!(gu.len(), total_gu);
    assert_eq!(dgu.len(), total_gu);
    if rows == 0 || hidden == 0 {
        return;
    }
    let row_fn = |g_row: &[f32], gu_row: &[f32], dgu_row: &mut [f32]| {
        let (gate_row, up_row) = gu_row.split_at(hidden);
        let (dg_row, du_row) = dgu_row.split_at_mut(hidden);
        if is_swiglu {
            ra_swiglu_bwd_chunk(g_row, gate_row, up_row, dg_row, du_row, b);
        } else {
            ra_glu_bwd_chunk(g_row, gate_row, up_row, dg_row, du_row, b);
        }
    };
    if total_out >= PARALLEL_ELEM_THRESHOLD && rows > 1 {
        grad_out
            .par_chunks_exact(hidden)
            .zip(gu.par_chunks_exact(two_h))
            .zip(dgu.par_chunks_exact_mut(two_h))
            .for_each(|((g_row, gu_row), dgu_row)| row_fn(g_row, gu_row, dgu_row));
    } else {
        for ((g_row, gu_row), dgu_row) in grad_out
            .chunks_exact(hidden)
            .zip(gu.chunks_exact(two_h))
            .zip(dgu.chunks_exact_mut(two_h))
        {
            row_fn(g_row, gu_row, dgu_row);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ra_swiglu_and_glu_bwd_parity() {
        let gate: Vec<f32> = (-20..20).map(|i| i as f32 * 0.4).collect();
        let up: Vec<f32> = (0..40).map(|i| (i as f32 - 15.0) * 0.25).collect();
        let g: Vec<f32> = (0..40).map(|i| ((i % 9) as f32 - 4.0) * 0.3).collect();
        let mut dg = vec![0.0_f32; 40];
        let mut du = vec![0.0_f32; 40];
        ra_swiglu_bwd_slice(&g, &gate, &up, &mut dg, &mut du, 1.25);
        for i in 0..40 {
            let (eg, eu) = ra_swiglu_bwd_scalar_elem(g[i], gate[i], up[i], 1.25);
            assert!((dg[i] - eg).abs() < 1e-5);
            assert!((du[i] - eu).abs() < 1e-5);
        }
        ra_glu_bwd_slice(&g, &gate, &up, &mut dg, &mut du, 1.25);
        for i in 0..40 {
            let (eg, eu) = ra_glu_bwd_scalar_elem(g[i], gate[i], up[i], 1.25);
            assert!((dg[i] - eg).abs() < 1e-5);
            assert!((du[i] - eu).abs() < 1e-5);
        }
    }
}
