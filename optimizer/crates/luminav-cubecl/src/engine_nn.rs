use cubecl::prelude::*;

use crate::engine::CubeClLuminaEngine;
use crate::error::CubeClError;
use crate::kernels::nn::{
    ra_glu_fwd_kernel, ra_norm_fwd_kernel, ra_softmax_fwd_kernel, ra_swiglu_fwd_kernel,
    ra_tanh_fwd_kernel,
};

impl CubeClLuminaEngine {
    pub fn ra_tanh_f32(&self, x: &[f32], b: f32) -> Result<Vec<f32>, CubeClError> {
        let n = x.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let h_in = self.client.create_from_slice(f32::as_bytes(x));
        let h_out = self.client.empty(core::mem::size_of_val(x));
        let num_cubes = n.div_ceil(self.cube_dim_1d);
        unsafe {
            ra_tanh_fwd_kernel::launch_unchecked(
                &self.client,
                CubeCount::Static(num_cubes as u32, 1, 1),
                CubeDim::new_1d(self.cube_dim_1d as u32),
                BufferArg::from_raw_parts(h_in, n),
                BufferArg::from_raw_parts(h_out.clone(), n),
                n,
                b,
            );
        }
        let mut out = vec![0.0_f32; n];
        self.read_into_slice(h_out, &mut out);
        Ok(out)
    }

    pub fn ra_swiglu_f32(&self, gate: &[f32], up: &[f32], b: f32) -> Result<Vec<f32>, CubeClError> {
        self.launch_ra_gated_f32(gate, up, b, true)
    }

    pub fn ra_glu_f32(&self, gate: &[f32], up: &[f32], b: f32) -> Result<Vec<f32>, CubeClError> {
        self.launch_ra_gated_f32(gate, up, b, false)
    }

    fn launch_ra_gated_f32(
        &self,
        gate: &[f32],
        up: &[f32],
        b: f32,
        is_swiglu: bool,
    ) -> Result<Vec<f32>, CubeClError> {
        if gate.len() != up.len() {
            return Err(CubeClError::InvalidShape(format!(
                "gate len {} != up len {}",
                gate.len(),
                up.len()
            )));
        }
        let n = gate.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let h_gate = self.client.create_from_slice(f32::as_bytes(gate));
        let h_up = self.client.create_from_slice(f32::as_bytes(up));
        let h_out = self.client.empty(core::mem::size_of_val(gate));
        let num_cubes = n.div_ceil(self.cube_dim_1d);
        let grid = CubeCount::Static(num_cubes as u32, 1, 1);
        let dim = CubeDim::new_1d(self.cube_dim_1d as u32);
        unsafe {
            if is_swiglu {
                ra_swiglu_fwd_kernel::launch_unchecked(
                    &self.client,
                    grid,
                    dim,
                    BufferArg::from_raw_parts(h_gate, n),
                    BufferArg::from_raw_parts(h_up, n),
                    BufferArg::from_raw_parts(h_out.clone(), n),
                    n,
                    b,
                );
            } else {
                ra_glu_fwd_kernel::launch_unchecked(
                    &self.client,
                    grid,
                    dim,
                    BufferArg::from_raw_parts(h_gate, n),
                    BufferArg::from_raw_parts(h_up, n),
                    BufferArg::from_raw_parts(h_out.clone(), n),
                    n,
                    b,
                );
            }
        }
        let mut out = vec![0.0_f32; n];
        self.read_into_slice(h_out, &mut out);
        Ok(out)
    }

    pub fn ra_norm_f32(
        &self,
        x: &[f32],
        weight: Option<&[f32]>,
        bias: Option<&[f32]>,
        cols: usize,
        eps: f32,
    ) -> Result<Vec<f32>, CubeClError> {
        if cols == 0 || !x.len().is_multiple_of(cols) {
            return Err(CubeClError::InvalidShape(format!(
                "x len {} not divisible by cols {}",
                x.len(),
                cols
            )));
        }
        let rows = x.len() / cols;
        if rows == 0 {
            return Ok(Vec::new());
        }
        let h_x = self.client.create_from_slice(f32::as_bytes(x));
        let h_w = match weight {
            Some(w) => self.client.create_from_slice(f32::as_bytes(w)),
            None => self.client.empty(core::mem::size_of::<f32>()),
        };
        let h_b = match bias {
            Some(b) => self.client.create_from_slice(f32::as_bytes(b)),
            None => self.client.empty(core::mem::size_of::<f32>()),
        };
        let h_out = self.client.empty(core::mem::size_of_val(x));
        let row_cube = cols.next_power_of_two().min(self.cube_dim_1d).max(1);
        unsafe {
            ra_norm_fwd_kernel::launch_unchecked(
                &self.client,
                CubeCount::Static(rows as u32, 1, 1),
                CubeDim::new_1d(row_cube as u32),
                BufferArg::from_raw_parts(h_x, x.len()),
                BufferArg::from_raw_parts(h_w, weight.map_or(1, |w| w.len())),
                BufferArg::from_raw_parts(h_b, bias.map_or(1, |b| b.len())),
                BufferArg::from_raw_parts(h_out.clone(), x.len()),
                cols,
                eps,
                weight.is_some(),
                bias.is_some(),
                row_cube,
            );
        }
        let mut out = vec![0.0_f32; x.len()];
        self.read_into_slice(h_out, &mut out);
        Ok(out)
    }

    pub fn ra_softmax_f32(
        &self,
        scores: &[f32],
        cols: usize,
        scale: f32,
    ) -> Result<Vec<f32>, CubeClError> {
        if cols == 0 || !scores.len().is_multiple_of(cols) {
            return Err(CubeClError::InvalidShape(format!(
                "scores len {} not divisible by cols {}",
                scores.len(),
                cols
            )));
        }
        let rows = scores.len() / cols;
        if rows == 0 {
            return Ok(Vec::new());
        }
        let h_s = self.client.create_from_slice(f32::as_bytes(scores));
        let h_out = self.client.empty(core::mem::size_of_val(scores));
        let row_cube = cols.next_power_of_two().min(self.cube_dim_1d).max(1);
        unsafe {
            ra_softmax_fwd_kernel::launch_unchecked(
                &self.client,
                CubeCount::Static(rows as u32, 1, 1),
                CubeDim::new_1d(row_cube as u32),
                BufferArg::from_raw_parts(h_s, scores.len()),
                BufferArg::from_raw_parts(h_out.clone(), scores.len()),
                cols,
                scale,
                row_cube,
            );
        }
        let mut out = vec![0.0_f32; scores.len()];
        self.read_into_slice(h_out, &mut out);
        Ok(out)
    }
}
