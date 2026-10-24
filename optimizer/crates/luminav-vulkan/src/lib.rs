#![deny(warnings)]

pub mod context;
pub mod dual;
pub mod error;
pub mod pipeline;
pub mod tdclm;
pub mod uma_memory;

pub use context::VulkanContext;
pub use error::VulkanError;
pub use pipeline::{DiffSumPush, DualPush, PostStepEmaPush, TdclmPush, VulkanPipelines};
pub use uma_memory::UmaBuffer;

use luminav_core::{BufferMode, LuminaVConfig, TensorViewMut};

pub struct UmaWorkspace {
    pub capacity: usize,
    pub param: UmaBuffer<f32>,
    pub master: UmaBuffer<f32>,
    pub grad: UmaBuffer<f32>,
    pub exp_avg: UmaBuffer<f32>,
    pub exp_avg_sq: UmaBuffer<f32>,
    pub scratch: UmaBuffer<f32>,
}

pub struct VulkanLuminaEngine {
    workspace: Option<UmaWorkspace>,
    pub pipelines: VulkanPipelines,
    pub context: VulkanContext,
}

impl VulkanLuminaEngine {
    pub fn new() -> Result<Self, VulkanError> {
        let context = VulkanContext::new()?;
        let pipelines = VulkanPipelines::new(&context)?;
        Ok(Self {
            workspace: None,
            pipelines,
            context,
        })
    }

    pub fn allocate_buffer<T: Copy>(&self, len: usize) -> Result<UmaBuffer<T>, VulkanError> {
        UmaBuffer::allocate(&self.context, len)
    }

    pub fn step(
        &mut self,
        view: &mut TensorViewMut<'_>,
        config: &LuminaVConfig,
        step: u64,
        param_index: usize,
    ) -> Result<(), VulkanError> {
        let n = view.param.len();
        if n == 0 {
            return Ok(());
        }

        let mut ws = match self.workspace.take() {
            Some(ws) if ws.capacity >= n => ws,
            _ => {
                let cap = n.max(262_144).next_power_of_two();
                UmaWorkspace {
                    capacity: cap,
                    param: self.allocate_buffer::<f32>(cap)?,
                    master: self.allocate_buffer::<f32>(cap)?,
                    grad: self.allocate_buffer::<f32>(cap)?,
                    exp_avg: self.allocate_buffer::<f32>(cap)?,
                    exp_avg_sq: self.allocate_buffer::<f32>(cap)?,
                    scratch: self.allocate_buffer::<f32>(6)?,
                }
            }
        };

        ws.param.set_len(n);
        ws.grad.set_len(n);
        ws.exp_avg.set_len(n);
        ws.scratch.set_len(6);

        ws.param.copy_from_slice(view.param);
        ws.grad.copy_from_slice(view.grad);
        ws.exp_avg.copy_from_slice(view.exp_avg);
        ws.scratch.copy_from_slice(view.scratch.as_slice());

        let has_master = if let Some(mp) = &view.master_param {
            ws.master.set_len(n);
            ws.master.copy_from_slice(mp);
            true
        } else {
            false
        };

        let res = match config.buffer {
            BufferMode::TdClm => self.step_tdclm_uma_with_state_dtype(
                &mut ws.param,
                if has_master {
                    Some(&mut ws.master)
                } else {
                    None
                },
                &ws.grad,
                &mut ws.exp_avg,
                &mut ws.scratch,
                config,
                view.dtype_mode,
                view.state_dtype_mode,
                step,
                param_index,
            ),
            BufferMode::Dual => {
                let exp_avg_sq = match view.exp_avg_sq.as_mut() {
                    Some(sq) => sq,
                    None => {
                        self.workspace = Some(ws);
                        return Err(VulkanError::InvalidShape(
                            "Missing exp_avg_sq in Dual mode".to_string(),
                        ));
                    }
                };
                ws.exp_avg_sq.set_len(n);
                ws.exp_avg_sq.copy_from_slice(exp_avg_sq);

                let r = self.step_dual_uma_with_state_dtype(
                    &mut ws.param,
                    if has_master {
                        Some(&mut ws.master)
                    } else {
                        None
                    },
                    &ws.grad,
                    &mut ws.exp_avg,
                    &mut ws.exp_avg_sq,
                    &mut ws.scratch,
                    config,
                    view.dtype_mode,
                    view.state_dtype_mode,
                    step,
                    param_index,
                );
                if r.is_ok() {
                    ws.exp_avg_sq.copy_to_slice(exp_avg_sq);
                }
                r
            }
        };

        if res.is_ok() {
            ws.param.copy_to_slice(view.param);
            if let Some(mp_view) = &mut view.master_param
                && has_master
            {
                ws.master.copy_to_slice(mp_view);
            }
            ws.exp_avg.copy_to_slice(view.exp_avg);
            ws.scratch.copy_to_slice(view.scratch.as_mut_slice());
        }

        self.workspace = Some(ws);
        res
    }
}
