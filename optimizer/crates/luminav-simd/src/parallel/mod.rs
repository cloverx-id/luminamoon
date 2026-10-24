pub mod dual;
pub mod tdclm;

use crate::SimdBackend;
use luminav_core::{reduce_sum_16, sanitize_gradient};

pub use dual::{dual_step_parallel, dual_step_parallel_chunked};
pub use tdclm::{tdclm_step_parallel, tdclm_step_parallel_chunked};

pub const ELEM_CHUNK_SIZE: usize = 2_097_152;

#[inline]
pub(crate) fn compute_diff_sum_chunk_dispatch(
    grad: &[f32],
    exp_avg: &[f32],
    backend: SimdBackend,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    match backend {
        SimdBackend::Avx512 => unsafe {
            crate::avx512::compute_diff_sum_chunk_avx512(grad, exp_avg)
        },
        SimdBackend::Avx2 => unsafe { crate::avx2::compute_diff_sum_chunk_avx2(grad, exp_avg) },
        SimdBackend::Scalar => compute_diff_sum_chunk_scalar(grad, exp_avg),
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = backend;
        compute_diff_sum_chunk_scalar(grad, exp_avg)
    }
}

#[inline]
fn compute_diff_sum_chunk_scalar(grad: &[f32], exp_avg: &[f32]) -> f32 {
    let n = grad.len();
    let simd_len = n & !15;
    let mut lanes = [0.0_f32; 16];
    let mut i = 0;
    while i < simd_len {
        for (lane, acc) in lanes.iter_mut().enumerate() {
            let idx = i + lane;
            *acc += (sanitize_gradient(grad[idx]) - exp_avg[idx]).clamp(-1.0e18_f32, 1.0e18_f32);
        }
        i += 16;
    }
    let mut sum = reduce_sum_16(&lanes);
    for idx in simd_len..n {
        sum += (sanitize_gradient(grad[idx]) - exp_avg[idx]).clamp(-1.0e18_f32, 1.0e18_f32);
    }
    sum
}
