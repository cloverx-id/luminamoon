pub mod dual;
pub mod t2;
pub mod tdclm;

pub use dual::single_step_dual_scalar;
pub use t2::single_step_t2_scalar;
pub use tdclm::{single_step_t1_scalar, single_step_tdclm_scalar};

pub const ELEM_CHUNK_SIZE: usize = 2_097_152;
