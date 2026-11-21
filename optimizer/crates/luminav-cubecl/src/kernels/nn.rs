use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn ra_tanh_fwd_kernel(x: &[f32], out: &mut [f32], n_elements: usize, b_val: f32) {
    let idx = ABSOLUTE_POS;
    if idx < n_elements {
        let xi = x[idx];
        let b_term = b_val * (b_val + f32::abs(xi));
        out[idx] = (b_term * xi) / (b_term + xi * xi);
    }
}

#[cube(launch_unchecked)]
pub fn ra_swiglu_fwd_kernel(
    gate: &[f32],
    up: &[f32],
    out: &mut [f32],
    n_elements: usize,
    b_val: f32,
) {
    let idx = ABSOLUTE_POS;
    if idx < n_elements {
        let z = gate[idx];
        let u = up[idx];
        let d = b_val * (b_val + f32::abs(z)) + z * z;
        let q = (0.5_f32 * b_val * b_val) / d;
        let sigma = if z >= 0.0_f32 { 1.0_f32 - q } else { q };
        out[idx] = (z * sigma) * u;
    }
}

#[cube(launch_unchecked)]
pub fn ra_glu_fwd_kernel(gate: &[f32], up: &[f32], out: &mut [f32], n_elements: usize, b_val: f32) {
    let idx = ABSOLUTE_POS;
    if idx < n_elements {
        let z = gate[idx];
        let u = up[idx];
        let b_term = b_val * (b_val + f32::abs(z));
        let ra_t = (b_term * z) / (b_term + z * z);
        out[idx] = ra_t * u;
    }
}

#[cube(launch_unchecked)]
pub fn ra_norm_fwd_kernel(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    out: &mut [f32],
    cols: usize,
    eps: f32,
    #[comptime] has_weight: bool,
    #[comptime] has_bias: bool,
    #[comptime] cube_dim: usize,
) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let row_base = row * cols;

    let mut smem = Shared::<[f32]>::new_slice(cube_dim);
    let mut local_abs_sum = f32::new(0.0_f32);
    let mut c = tid;
    while c < cols {
        local_abs_sum += f32::abs(x[row_base + c]);
        c += cube_dim;
    }
    smem[tid] = local_abs_sum;
    sync_cube();

    let mut stride = (CUBE_DIM_X as usize) / 2;
    while stride > 0 {
        if tid < stride {
            let a = smem[tid];
            let b = smem[tid + stride];
            smem[tid] = a + b;
        }
        sync_cube();
        stride /= 2;
    }

    let inv_denom = 1.0_f32 / (smem[0] / f32::cast_from(cols as u32) + eps);
    let mut c2 = tid;
    while c2 < cols {
        let mut y = x[row_base + c2] * inv_denom;
        if comptime!(has_weight) {
            y *= weight[c2];
        }
        if comptime!(has_bias) {
            y += bias[c2];
        }
        out[row_base + c2] = y;
        c2 += cube_dim;
    }
}

#[cube(launch_unchecked)]
pub fn ra_softmax_fwd_kernel(
    scores: &[f32],
    out: &mut [f32],
    cols: usize,
    scale: f32,
    #[comptime] cube_dim: usize,
) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let row_base = row * cols;

    let mut smem = Shared::<[f32]>::new_slice(cube_dim);
    let mut local_max = f32::new(-3.402_823_4e38_f32);
    let mut c = tid;
    while c < cols {
        let s = scores[row_base + c];
        if s > local_max {
            local_max = s;
        }
        c += cube_dim;
    }
    smem[tid] = local_max;
    sync_cube();

    let mut stride = (CUBE_DIM_X as usize) / 2;
    while stride > 0 {
        if tid < stride {
            let cur = smem[tid];
            let other = smem[tid + stride];
            if other > cur {
                smem[tid] = other;
            }
        }
        sync_cube();
        stride /= 2;
    }
    let max_s = smem[0];

    let mut local_sum = f32::new(0.0_f32);
    let mut c2 = tid;
    while c2 < cols {
        let s = scores[row_base + c2];
        let delta = max_s - s;
        let u = delta * scale;
        let phi = if u < 1.0_f32 {
            let q = (1.0_f32 - u) / (1.0_f32 + u);
            let q2 = q * q;
            let q4 = q2 * q2;
            q4 * q4
        } else {
            0.0_f32
        };
        out[row_base + c2] = phi;
        local_sum += phi;
        c2 += cube_dim;
    }
    smem[tid] = local_sum;
    sync_cube();

    let mut stride2 = (CUBE_DIM_X as usize) / 2;
    while stride2 > 0 {
        if tid < stride2 {
            let a = smem[tid];
            let b = smem[tid + stride2];
            smem[tid] = a + b;
        }
        sync_cube();
        stride2 /= 2;
    }
    let safe_sum = f32::max(smem[0], 1.0e-38_f32);
    let inv_sum = 1.0_f32 / safe_sum;

    let mut c3 = tid;
    while c3 < cols {
        let v = out[row_base + c3];
        out[row_base + c3] = v * inv_sum;
        c3 += cube_dim;
    }
}
