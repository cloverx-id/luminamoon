use crate::LuminaError;
use crate::config::{BufferMode, DTypeMode};

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(32))]
pub struct ScratchBuffer {
    data: [f32; 6],
}

impl Default for ScratchBuffer {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ScratchBuffer {
    #[inline]
    pub const fn new() -> Self {
        Self {
            data: [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        }
    }

    #[inline]
    pub const fn from_array(data: [f32; 6]) -> Self {
        Self { data }
    }

    #[inline]
    pub const fn as_slice(&self) -> &[f32; 6] {
        &self.data
    }

    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [f32; 6] {
        &mut self.data
    }

    #[inline]
    pub const fn mask_sum(&self) -> f32 {
        self.data[0]
    }

    #[inline]
    pub fn set_mask_sum(&mut self, val: f32) {
        self.data[0] = val;
    }

    #[inline]
    pub fn add_mask_sum(&mut self, delta: f32) {
        self.data[0] += delta;
    }

    #[inline]
    pub const fn u_sq_sum(&self) -> f32 {
        self.data[1]
    }

    #[inline]
    pub fn set_u_sq_sum(&mut self, val: f32) {
        self.data[1] = val;
    }

    #[inline]
    pub fn add_u_sq_sum(&mut self, delta: f32) {
        self.data[1] += delta;
    }

    #[inline]
    pub const fn p_sq_sum(&self) -> f32 {
        self.data[2]
    }

    #[inline]
    pub fn set_p_sq_sum(&mut self, val: f32) {
        self.data[2] = val;
    }

    #[inline]
    pub fn add_p_sq_sum(&mut self, delta: f32) {
        self.data[2] += delta;
    }

    #[inline]
    pub const fn cautious_scale(&self) -> f32 {
        self.data[3]
    }

    #[inline]
    pub fn set_cautious_scale(&mut self, val: f32) {
        self.data[3] = val;
    }

    #[inline]
    pub const fn bound_scale(&self) -> f32 {
        self.data[4]
    }

    #[inline]
    pub fn set_bound_scale(&mut self, val: f32) {
        self.data[4] = val;
    }

    #[inline]
    pub fn reset_accumulators(&mut self) {
        self.data[0] = 0.0;
        self.data[1] = 0.0;
        self.data[2] = 0.0;
        self.data[5] = 0.0;
    }

    #[inline]
    pub fn reset_all(&mut self) {
        self.data = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    }
}

#[derive(Debug, Clone)]
pub struct ParamState {
    pub step: u64,
    pub buf_count: BufferMode,
    pub scratch: ScratchBuffer,
    pub exp_avg: Vec<f32>,
    pub exp_avg_sq: Option<Vec<f32>>,
    pub master_param: Option<Vec<f32>>,
}

impl ParamState {
    pub fn new(initial_param: &[f32], buffer_mode: BufferMode, has_master: bool) -> Self {
        let n = initial_param.len();
        let exp_avg = vec![0.0_f32; n];
        let exp_avg_sq = match buffer_mode {
            BufferMode::Dual => Some(vec![0.0_f32; n]),
            BufferMode::TdClm => None,
        };
        let master_param = if has_master {
            Some(initial_param.to_vec())
        } else {
            None
        };

        Self {
            step: 0,
            buf_count: buffer_mode,
            scratch: ScratchBuffer::new(),
            exp_avg,
            exp_avg_sq,
            master_param,
        }
    }

    #[inline]
    pub fn increment_step(&mut self) -> u64 {
        self.step = self.step.saturating_add(1);
        self.step
    }

    pub fn ensure_buffer_mode(&mut self, mode: BufferMode) {
        if self.buf_count != mode {
            self.scratch.reset_all();
            self.buf_count = mode;
        }
        if mode == BufferMode::Dual && self.exp_avg_sq.is_none() {
            self.exp_avg_sq = Some(vec![0.0_f32; self.exp_avg.len()]);
        }
    }
}

#[derive(Debug)]
pub struct TensorViewMut<'a> {
    pub param: &'a mut [f32],
    pub master_param: Option<&'a mut [f32]>,
    pub grad: &'a [f32],
    pub exp_avg: &'a mut [f32],
    pub exp_avg_sq: Option<&'a mut [f32]>,
    pub scratch: &'a mut ScratchBuffer,
    pub dtype_mode: DTypeMode,
    pub state_dtype_mode: DTypeMode,
}

impl<'a> TensorViewMut<'a> {
    pub fn validate(
        &self,
        buffer_mode: BufferMode,
        require_master: bool,
    ) -> Result<usize, LuminaError> {
        let n = self.param.len();
        if self.grad.len() != n {
            return Err(LuminaError::ShapeMismatch {
                expected: n,
                actual: self.grad.len(),
            });
        }
        if self.exp_avg.len() != n {
            return Err(LuminaError::ShapeMismatch {
                expected: n,
                actual: self.exp_avg.len(),
            });
        }
        if buffer_mode == BufferMode::Dual {
            match &self.exp_avg_sq {
                Some(v) if v.len() == n => {}
                Some(v) => {
                    return Err(LuminaError::ShapeMismatch {
                        expected: n,
                        actual: v.len(),
                    });
                }
                None => return Err(LuminaError::MissingSecondaryStateBuffer),
            }
        }
        if require_master {
            match &self.master_param {
                Some(mp) if mp.len() == n => {}
                Some(mp) => {
                    return Err(LuminaError::ShapeMismatch {
                        expected: n,
                        actual: mp.len(),
                    });
                }
                None => return Err(LuminaError::MissingMasterParameterBuffer),
            }
        } else if let Some(mp) = &self.master_param
            && mp.len() != n
        {
            return Err(LuminaError::ShapeMismatch {
                expected: n,
                actual: mp.len(),
            });
        }
        Ok(n)
    }
}
