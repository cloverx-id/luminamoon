pub mod dual;
pub mod tdclm;

pub use dual::{dual_step_parallel, dual_step_parallel_chunked};
pub use tdclm::{tdclm_step_parallel, tdclm_step_parallel_chunked};

pub const ELEM_CHUNK_SIZE: usize = 2_097_152;
