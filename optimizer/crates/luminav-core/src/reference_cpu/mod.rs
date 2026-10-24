pub mod dual;
pub mod tdclm;

use crate::math::{reduce_sum_16, sanitize_gradient};

pub use dual::single_step_dual_scalar;
pub use tdclm::single_step_tdclm_scalar;

pub const ELEM_CHUNK_SIZE: usize = 2_097_152;

#[inline]
pub(crate) fn compute_diff_sum_scalar(grad: &[f32], exp_avg: &[f32]) -> f32 {
    let n = grad.len();
    let mut total_diff = 0.0_f32;
    let mut chunk_start = 0;
    while chunk_start < n {
        let chunk_end = (chunk_start + ELEM_CHUNK_SIZE).min(n);
        let chunk_len = chunk_end - chunk_start;
        let simd_len = chunk_len & !15;
        let mut lanes = [0.0_f32; 16];
        let mut i = 0;
        while i < simd_len {
            for (lane, acc) in lanes.iter_mut().enumerate() {
                let idx = chunk_start + i + lane;
                let g = sanitize_gradient(grad[idx]);
                let m = exp_avg[idx];
                let diff_gm = (g - m).clamp(-1.0e18_f32, 1.0e18_f32);
                *acc += diff_gm;
            }
            i += 16;
        }
        let mut chunk_sum = reduce_sum_16(&lanes);
        for idx in (chunk_start + simd_len)..chunk_end {
            let g = sanitize_gradient(grad[idx]);
            let m = exp_avg[idx];
            let diff_gm = (g - m).clamp(-1.0e18_f32, 1.0e18_f32);
            chunk_sum += diff_gm;
        }
        total_diff += chunk_sum;
        chunk_start = chunk_end;
    }
    total_diff
}
