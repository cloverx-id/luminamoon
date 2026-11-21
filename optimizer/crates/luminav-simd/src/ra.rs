#[inline]
pub fn ra_norm_slice(
    input: &[f32],
    output: &mut [f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
) {
    let len = input.len();
    assert_eq!(output.len(), len, "Output slice length mismatch in ra_norm");
    if len == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe {
                return ra_norm_slice_avx2(input, output, weight, bias, eps);
            }
        }
    }

    ra_norm_slice_scalar(input, output, weight, bias, eps);
}

#[inline]
pub fn ra_tanh_slice(input: &[f32], output: &mut [f32], b: f32) {
    let len = input.len();
    assert_eq!(output.len(), len, "Output slice length mismatch in ra_tanh");
    if len == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe {
                return ra_tanh_slice_avx2(input, output, b);
            }
        }
    }

    ra_tanh_slice_scalar(input, output, b);
}

#[inline]
pub fn pos_ra_glu_slice(gate: &[f32], up: &[f32], output: &mut [f32], b: f32) {
    let len = gate.len();
    assert_eq!(up.len(), len, "Up slice length mismatch in pos_ra_glu");
    assert_eq!(
        output.len(),
        len,
        "Output slice length mismatch in pos_ra_glu"
    );
    if len == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe {
                return pos_ra_glu_slice_avx2(gate, up, output, b);
            }
        }
    }

    pos_ra_glu_slice_scalar(gate, up, output, b);
}

#[inline]
pub fn ra_norm_slice_scalar(
    input: &[f32],
    output: &mut [f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
) {
    let len = input.len();
    let mut sum_abs = 0.0_f32;
    for &x in input {
        sum_abs += x.abs();
    }
    let l1_mean = sum_abs / (len as f32);
    let inv_scale = 1.0_f32 / (l1_mean + eps);

    match (weight, bias) {
        (Some(w), Some(b)) => {
            for i in 0..len {
                output[i] = input[i] * inv_scale * w[i] + b[i];
            }
        }
        (Some(w), None) => {
            for i in 0..len {
                output[i] = input[i] * inv_scale * w[i];
            }
        }
        (None, Some(b)) => {
            for i in 0..len {
                output[i] = input[i] * inv_scale + b[i];
            }
        }
        (None, None) => {
            for i in 0..len {
                output[i] = input[i] * inv_scale;
            }
        }
    }
}

#[inline]
pub fn ra_tanh_slice_scalar(input: &[f32], output: &mut [f32], b: f32) {
    for i in 0..input.len() {
        let x = input[i];
        let abs_x = x.abs();
        let b_term = b * (b + abs_x);
        output[i] = (b_term * x) / (b_term + x * x);
    }
}

#[inline]
pub fn pos_ra_glu_slice_scalar(gate: &[f32], up: &[f32], output: &mut [f32], b: f32) {
    for i in 0..gate.len() {
        let z = gate[i];
        let abs_z = z.abs();
        let b_term = b * (b + abs_z);
        let rat_tanh = (b_term * z) / (b_term + z * z);
        let sigma = 0.5_f32 * (1.0_f32 + rat_tanh);
        output[i] = up[i] * sigma;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn ra_norm_slice_avx2(
    input: &[f32],
    output: &mut [f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
) {
    use std::arch::x86_64::*;
    let len = input.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let sign_mask = _mm256_set1_ps(-0.0_f32);
    let mut sum_vec = _mm256_setzero_ps();

    for c in 0..chunks {
        let offset = c * 8;
        let v = _mm256_loadu_ps(input.as_ptr().add(offset));
        let abs_v = _mm256_andnot_ps(sign_mask, v);
        sum_vec = _mm256_add_ps(sum_vec, abs_v);
    }

    // Horizontal reduction of sum_vec
    let lo128 = _mm256_castps256_ps128(sum_vec);
    let hi128 = _mm256_extractf128_ps::<1>(sum_vec);
    let sum128 = _mm_add_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_movehdup_ps(sum64);
    let sum32 = _mm_add_ss(sum64, hi32);
    let mut total_sum = _mm_cvtss_f32(sum32);

    for &x in input.iter().skip(chunks * 8) {
        total_sum += x.abs();
    }

    let l1_mean = total_sum / (len as f32);
    let inv_scale = 1.0_f32 / (l1_mean + eps);
    let inv_scale_vec = _mm256_set1_ps(inv_scale);

    for c in 0..chunks {
        let offset = c * 8;
        let x = _mm256_loadu_ps(input.as_ptr().add(offset));
        let mut scaled = _mm256_mul_ps(x, inv_scale_vec);

        if let Some(w) = weight {
            let w_vec = _mm256_loadu_ps(w.as_ptr().add(offset));
            scaled = _mm256_mul_ps(scaled, w_vec);
        }
        if let Some(b) = bias {
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(offset));
            scaled = _mm256_add_ps(scaled, b_vec);
        }

        _mm256_storeu_ps(output.as_mut_ptr().add(offset), scaled);
    }

    if remainder > 0 {
        let start = chunks * 8;
        for i in start..len {
            let mut val = input[i] * inv_scale;
            if let Some(w) = weight {
                val *= w[i];
            }
            if let Some(b) = bias {
                val += b[i];
            }
            output[i] = val;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn ra_tanh_slice_avx2(input: &[f32], output: &mut [f32], b: f32) {
    use std::arch::x86_64::*;
    let len = input.len();
    let chunks = len / 8;
    let b_vec = _mm256_set1_ps(b);
    let sign_mask = _mm256_set1_ps(-0.0_f32);

    for c in 0..chunks {
        let offset = c * 8;
        let x = _mm256_loadu_ps(input.as_ptr().add(offset));
        let abs_x = _mm256_andnot_ps(sign_mask, x);
        let b_plus_abs = _mm256_add_ps(b_vec, abs_x);
        let b_term = _mm256_mul_ps(b_vec, b_plus_abs);
        let num = _mm256_mul_ps(b_term, x);
        let x_sq = _mm256_mul_ps(x, x);
        let den = _mm256_add_ps(b_term, x_sq);
        let res = _mm256_div_ps(num, den);
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), res);
    }

    for i in (chunks * 8)..len {
        let x = input[i];
        let abs_x = x.abs();
        let b_term = b * (b + abs_x);
        output[i] = (b_term * x) / (b_term + x * x);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pos_ra_glu_slice_avx2(gate: &[f32], up: &[f32], output: &mut [f32], b: f32) {
    use std::arch::x86_64::*;
    let len = gate.len();
    let chunks = len / 8;
    let b_vec = _mm256_set1_ps(b);
    let sign_mask = _mm256_set1_ps(-0.0_f32);
    let half_vec = _mm256_set1_ps(0.5_f32);
    let one_vec = _mm256_set1_ps(1.0_f32);

    for c in 0..chunks {
        let offset = c * 8;
        let z = _mm256_loadu_ps(gate.as_ptr().add(offset));
        let u = _mm256_loadu_ps(up.as_ptr().add(offset));

        let abs_z = _mm256_andnot_ps(sign_mask, z);
        let b_term = _mm256_mul_ps(b_vec, _mm256_add_ps(b_vec, abs_z));
        let num = _mm256_mul_ps(b_term, z);
        let den = _mm256_fmadd_ps(z, z, b_term);
        let ra_tanh = _mm256_div_ps(num, den);
        let sigma = _mm256_mul_ps(half_vec, _mm256_add_ps(one_vec, ra_tanh));

        let out = _mm256_mul_ps(u, sigma);
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), out);
    }

    for i in (chunks * 8)..len {
        let z = gate[i];
        let abs_z = z.abs();
        let b_term = b * (b + abs_z);
        let rat_tanh = (b_term * z) / (b_term + z * z);
        let sigma = 0.5_f32 * (1.0_f32 + rat_tanh);
        output[i] = up[i] * sigma;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ra_softmax::ra_softmax_row;

    #[test]
    fn test_ra_norm_scale_homogeneity() {
        let x = [1.0_f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0];
        let mut out1 = vec![0.0_f32; x.len()];
        let mut out2 = vec![0.0_f32; x.len()];

        ra_norm_slice(&x, &mut out1, None, None, 1e-6);

        let alpha = 3.5_f32;
        let x_scaled: Vec<f32> = x.iter().map(|&v| v * alpha).collect();
        ra_norm_slice(&x_scaled, &mut out2, None, None, 1e-6);

        for i in 0..x.len() {
            assert!((out1[i] - out2[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_ra_tanh_bounds_and_symmetry() {
        let b = 1.2_f32;
        let inputs = [-500.0_f32, -10.0, -1.0, 0.0, 1.0, 10.0, 500.0];
        let mut outputs = vec![0.0_f32; inputs.len()];

        ra_tanh_slice(&inputs, &mut outputs, b);

        for &y in &outputs {
            assert!(y > -b && y < b);
        }

        // Test symmetry f(-x) == -f(x)
        assert_eq!(outputs[3], 0.0); // f(0) == 0
        assert!((outputs[2] + outputs[4]).abs() < 1e-6); // f(-1) + f(1) == 0
        assert!((outputs[1] + outputs[5]).abs() < 1e-6); // f(-10) + f(10) == 0
    }

    #[test]
    fn test_pos_ra_glu_linear_bounds() {
        let gate = [-10.0_f32, -2.0, 0.0, 2.0, 10.0];
        let up = [5.0_f32, 5.0, 5.0, 5.0, 5.0];
        let mut out = vec![0.0_f32; gate.len()];

        pos_ra_glu_slice(&gate, &up, &mut out, 1.0);

        for &h in &out {
            assert!((0.0..=5.0).contains(&h));
        }
    }

    #[test]
    fn test_ra_softmax_compact_support() {
        // Token 0 is max (10.0), Token 1 is 10.0 - 5.0 = 5.0 (Delta = 5 < 16), Token 2 is 10.0 - 30.0 = -20.0 (Delta = 30 >= 16)
        let scores = [10.0_f32, 5.0, -20.0];
        let mut p = vec![0.0_f32; scores.len()];

        ra_softmax_row(&scores, &mut p, 0.0625);

        // Sum must be 1.0
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Token 2 has Delta = 30 >= 16 -> must be exactly 0.0
        assert_eq!(p[2], 0.0);
        assert!(p[0] > p[1]);
        assert!(p[1] > 0.0);
    }
}
