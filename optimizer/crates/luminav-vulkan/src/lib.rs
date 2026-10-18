#![deny(warnings)]

pub mod context;
pub mod error;
pub mod pipeline;
pub mod uma_memory;

use ash::vk;
use std::sync::Arc;

pub use context::VulkanContext;
pub use error::VulkanError;
pub use pipeline::{DiffSumPush, DualPush, PostStepEmaPush, TdclmPush, VulkanPipelines};
pub use uma_memory::UmaBuffer;

use luminav_core::{
    BoundType, BufferMode, DTypeMode, LuminaVConfig, TensorViewMut, compute_bias_correction,
    derive_kernel_seeds,
};

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

    #[allow(clippy::too_many_arguments)]
    pub fn step_tdclm_uma(
        &mut self,
        param: &mut UmaBuffer<f32>,
        master_param: Option<&mut UmaBuffer<f32>>,
        grad: &UmaBuffer<f32>,
        exp_avg: &mut UmaBuffer<f32>,
        scratch: &mut UmaBuffer<f32>,
        config: &LuminaVConfig,
        dtype_mode: DTypeMode,
        step: u64,
        param_index: usize,
    ) -> Result<(), VulkanError> {
        self.step_tdclm_uma_with_state_dtype(
            param,
            master_param,
            grad,
            exp_avg,
            scratch,
            config,
            dtype_mode,
            DTypeMode::Float32,
            step,
            param_index,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_tdclm_uma_with_state_dtype(
        &mut self,
        param: &mut UmaBuffer<f32>,
        master_param: Option<&mut UmaBuffer<f32>>,
        grad: &UmaBuffer<f32>,
        exp_avg: &mut UmaBuffer<f32>,
        scratch: &mut UmaBuffer<f32>,
        config: &LuminaVConfig,
        dtype_mode: DTypeMode,
        state_dtype_mode: DTypeMode,
        step: u64,
        param_index: usize,
    ) -> Result<(), VulkanError> {
        config.validate()?;
        let n = param.len();
        if n == 0 {
            return Ok(());
        }

        let device = Arc::clone(&self.context.device);
        let cb = self.pipelines.command_buffer;
        let fence = self.pipelines.fence;
        let queue = self.context.compute_queue;

        let has_master = config.master_weights.is_enabled() && master_param.is_some();
        let master_ptr = match &master_param {
            Some(mp) if has_master => mp.device_address(),
            _ => 0,
        };

        let (seed_m, seed_p) = derive_kernel_seeds(config.seed, step, param_index);

        let total_elements = n as f32;
        let use_cd = config.centered_defect && n > 1;

        {
            let sc_mut = scratch.as_slice_mut();
            sc_mut[0] = 0.0;
            sc_mut[1] = 0.0;
            sc_mut[2] = 0.0;
            sc_mut[5] = 0.0;
        }

        let diff_mean = if use_cd {
            unsafe {
                device.wait_for_fences(&[fence], true, u64::MAX)?;
                device.reset_fences(&[fence])?;
                device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;

                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                device.begin_command_buffer(cb, &begin_info)?;

                device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines.pipe_diff_sum,
                );

                let diff_push = DiffSumPush {
                    grad_ptr: grad.device_address(),
                    exp_avg_ptr: exp_avg.device_address(),
                    scratch_ptr: scratch.device_address(),
                    n_elements: n as u32,
                    _padding: 0,
                };

                let push_bytes = std::slice::from_raw_parts(
                    &diff_push as *const _ as *const u8,
                    std::mem::size_of::<DiffSumPush>(),
                );
                device.cmd_push_constants(
                    cb,
                    self.pipelines.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_bytes,
                );

                let wg_count = n.div_ceil(256) as u32;
                device.cmd_dispatch(cb, wg_count, 1, 1);
                device.end_command_buffer(cb)?;

                let submit_info =
                    vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cb));
                device.queue_submit(queue, &[submit_info], fence)?;
                device.wait_for_fences(&[fence], true, u64::MAX)?;
            }

            let diff_sum = scratch.as_slice()[5];
            diff_sum / total_elements
        } else {
            0.0_f32
        };

        let bound_type_val = match config.effective_bound_type() {
            BoundType::None => 0u32,
            BoundType::Radial => 1u32,
            BoundType::Coordinate => 2u32,
        };
        let dtype_mode_val = match dtype_mode {
            DTypeMode::Float32 => 0u32,
            DTypeMode::BFloat16 => 1u32,
            DTypeMode::Float16 => 2u32,
        };
        let state_dtype_val = match state_dtype_mode {
            DTypeMode::Float32 => 0u32,
            DTypeMode::BFloat16 => 1u32,
            DTypeMode::Float16 => 2u32,
        };

        let mut base_flags = 0u32;
        if has_master {
            base_flags |= 1;
        }
        base_flags |= bound_type_val << 1;
        if config.stochastic_rounding {
            base_flags |= 8;
        }
        base_flags |= dtype_mode_val << 4;
        if config.cautious {
            base_flags |= 64;
        }
        base_flags |= state_dtype_val << 8;

        let scratch_slice = scratch.as_slice();
        let cautious_scale = scratch_slice[3];
        let bound_scale = scratch_slice[4];

        let beta1 = config.betas.0;
        let beta1_sq = beta1 * beta1;
        let one_minus_beta1_sq = 1.0 - beta1_sq;
        let effective_eps = if dtype_mode == DTypeMode::Float16 {
            config.eps.max(1.0e-4)
        } else {
            config.eps
        };

        let two_pass =
            (!config.fused_single_pass || step <= 1) && (config.cautious || bound_type_val == 1u32);

        let mut base_ema_flags = 0u32;
        if config.cautious {
            base_ema_flags |= 1;
        }
        if bound_type_val == 1u32 {
            base_ema_flags |= 2;
        }

        let wg_count = n.div_ceil(256) as u32;

        unsafe {
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.reset_fences(&[fence])?;
            device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(cb, &begin_info)?;

            let mem_barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
            let dep_info =
                vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));

            if two_pass {
                device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines.pipe_tdclm,
                );

                let pass1_push = TdclmPush {
                    param_ptr: param.device_address(),
                    grad_ptr: grad.device_address(),
                    exp_avg_ptr: exp_avg.device_address(),
                    master_ptr,
                    scratch_ptr: scratch.device_address(),
                    n_elements: n as u32,
                    lr: config.lr,
                    weight_decay: config.weight_decay,
                    beta1,
                    beta1_sq,
                    one_minus_beta1_sq,
                    eps: effective_eps,
                    lam: config.lam,
                    bound_ratio: config.bound_ratio,
                    cautious_scale,
                    bound_scale,
                    diff_mean,
                    seed_m,
                    seed_p,
                    flags: base_flags | 128 | 1024,
                };
                let pass1_bytes = std::slice::from_raw_parts(
                    &pass1_push as *const _ as *const u8,
                    std::mem::size_of::<TdclmPush>(),
                );
                device.cmd_push_constants(
                    cb,
                    self.pipelines.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    pass1_bytes,
                );
                device.cmd_dispatch(cb, wg_count, 1, 1);

                device.cmd_pipeline_barrier2(cb, &dep_info);

                device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines.pipe_post_step_ema,
                );
                let post_push = PostStepEmaPush {
                    scratch_ptr: scratch.device_address(),
                    ema_decay: config.ema_decay,
                    n_elements: total_elements,
                    cautious_clamp_min: config.cautious_clamp_min,
                    lr: config.lr,
                    bound_ratio: config.bound_ratio,
                    flags: base_ema_flags | 4,
                };
                let post_bytes = std::slice::from_raw_parts(
                    &post_push as *const _ as *const u8,
                    std::mem::size_of::<PostStepEmaPush>(),
                );
                device.cmd_push_constants(
                    cb,
                    self.pipelines.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    post_bytes,
                );
                device.cmd_dispatch(cb, 1, 1, 1);

                device.cmd_pipeline_barrier2(cb, &dep_info);

                device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines.pipe_tdclm,
                );
                let pass2_push = TdclmPush {
                    flags: base_flags | 2048,
                    ..pass1_push
                };
                let pass2_bytes = std::slice::from_raw_parts(
                    &pass2_push as *const _ as *const u8,
                    std::mem::size_of::<TdclmPush>(),
                );
                device.cmd_push_constants(
                    cb,
                    self.pipelines.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    pass2_bytes,
                );
                device.cmd_dispatch(cb, wg_count, 1, 1);
            } else {
                device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines.pipe_tdclm,
                );

                let tdclm_push = TdclmPush {
                    param_ptr: param.device_address(),
                    grad_ptr: grad.device_address(),
                    exp_avg_ptr: exp_avg.device_address(),
                    master_ptr,
                    scratch_ptr: scratch.device_address(),
                    n_elements: n as u32,
                    lr: config.lr,
                    weight_decay: config.weight_decay,
                    beta1,
                    beta1_sq,
                    one_minus_beta1_sq,
                    eps: effective_eps,
                    lam: config.lam,
                    bound_ratio: config.bound_ratio,
                    cautious_scale,
                    bound_scale,
                    diff_mean,
                    seed_m,
                    seed_p,
                    flags: base_flags | 128,
                };

                let push_bytes = std::slice::from_raw_parts(
                    &tdclm_push as *const _ as *const u8,
                    std::mem::size_of::<TdclmPush>(),
                );
                device.cmd_push_constants(
                    cb,
                    self.pipelines.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_bytes,
                );
                device.cmd_dispatch(cb, wg_count, 1, 1);

                device.cmd_pipeline_barrier2(cb, &dep_info);

                device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines.pipe_post_step_ema,
                );

                let post_push = PostStepEmaPush {
                    scratch_ptr: scratch.device_address(),
                    ema_decay: config.ema_decay,
                    n_elements: total_elements,
                    cautious_clamp_min: config.cautious_clamp_min,
                    lr: config.lr,
                    bound_ratio: config.bound_ratio,
                    flags: base_ema_flags,
                };

                let post_bytes = std::slice::from_raw_parts(
                    &post_push as *const _ as *const u8,
                    std::mem::size_of::<PostStepEmaPush>(),
                );
                device.cmd_push_constants(
                    cb,
                    self.pipelines.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    post_bytes,
                );
                device.cmd_dispatch(cb, 1, 1, 1);
            }

            device.end_command_buffer(cb)?;

            let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cb));
            device.queue_submit(queue, &[submit_info], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_dual_uma(
        &mut self,
        param: &mut UmaBuffer<f32>,
        master_param: Option<&mut UmaBuffer<f32>>,
        grad: &UmaBuffer<f32>,
        exp_avg: &mut UmaBuffer<f32>,
        exp_avg_sq: &mut UmaBuffer<f32>,
        scratch: &mut UmaBuffer<f32>,
        config: &LuminaVConfig,
        dtype_mode: DTypeMode,
        step: u64,
        param_index: usize,
    ) -> Result<(), VulkanError> {
        self.step_dual_uma_with_state_dtype(
            param,
            master_param,
            grad,
            exp_avg,
            exp_avg_sq,
            scratch,
            config,
            dtype_mode,
            DTypeMode::Float32,
            step,
            param_index,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_dual_uma_with_state_dtype(
        &mut self,
        param: &mut UmaBuffer<f32>,
        master_param: Option<&mut UmaBuffer<f32>>,
        grad: &UmaBuffer<f32>,
        exp_avg: &mut UmaBuffer<f32>,
        exp_avg_sq: &mut UmaBuffer<f32>,
        scratch: &mut UmaBuffer<f32>,
        config: &LuminaVConfig,
        dtype_mode: DTypeMode,
        state_dtype_mode: DTypeMode,
        step: u64,
        param_index: usize,
    ) -> Result<(), VulkanError> {
        config.validate()?;
        let n = param.len();
        if n == 0 {
            return Ok(());
        }

        let device = Arc::clone(&self.context.device);
        let cb = self.pipelines.command_buffer;
        let fence = self.pipelines.fence;
        let queue = self.context.compute_queue;

        let has_master = config.master_weights.is_enabled() && master_param.is_some();
        let master_ptr = match &master_param {
            Some(mp) if has_master => mp.device_address(),
            _ => 0,
        };

        let coeffs = compute_bias_correction(
            step,
            config.betas.0,
            config.betas.1,
            config.eps,
            config.tau,
            dtype_mode,
        );
        let (seed_m, seed_p) = derive_kernel_seeds(config.seed, step, param_index);

        let total_elements = n as f32;
        let use_cd = config.centered_defect && n > 1;

        {
            let sc_mut = scratch.as_slice_mut();
            sc_mut[0] = 0.0;
            sc_mut[1] = 0.0;
            sc_mut[2] = 0.0;
            sc_mut[5] = 0.0;
        }

        let diff_mean_scaled = if use_cd {
            unsafe {
                device.wait_for_fences(&[fence], true, u64::MAX)?;
                device.reset_fences(&[fence])?;
                device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;

                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                device.begin_command_buffer(cb, &begin_info)?;

                device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines.pipe_diff_sum,
                );

                let diff_push = DiffSumPush {
                    grad_ptr: grad.device_address(),
                    exp_avg_ptr: exp_avg.device_address(),
                    scratch_ptr: scratch.device_address(),
                    n_elements: n as u32,
                    _padding: 0,
                };

                let push_bytes = std::slice::from_raw_parts(
                    &diff_push as *const _ as *const u8,
                    std::mem::size_of::<DiffSumPush>(),
                );
                device.cmd_push_constants(
                    cb,
                    self.pipelines.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_bytes,
                );

                let wg_count = n.div_ceil(256) as u32;
                device.cmd_dispatch(cb, wg_count, 1, 1);
                device.end_command_buffer(cb)?;

                let submit_info =
                    vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cb));
                device.queue_submit(queue, &[submit_info], fence)?;
                device.wait_for_fences(&[fence], true, u64::MAX)?;
            }

            let diff_sum = scratch.as_slice()[5];
            (diff_sum / total_elements) * config.betas.0
        } else {
            0.0_f32
        };

        let bound_type_val = match config.effective_bound_type() {
            BoundType::None => 0u32,
            BoundType::Radial => 1u32,
            BoundType::Coordinate => 2u32,
        };
        let dtype_mode_val = match dtype_mode {
            DTypeMode::Float32 => 0u32,
            DTypeMode::BFloat16 => 1u32,
            DTypeMode::Float16 => 2u32,
        };
        let state_dtype_val = match state_dtype_mode {
            DTypeMode::Float32 => 0u32,
            DTypeMode::BFloat16 => 1u32,
            DTypeMode::Float16 => 2u32,
        };

        let mut base_flags = 0u32;
        if has_master {
            base_flags |= 1;
        }
        base_flags |= bound_type_val << 1;
        if config.stochastic_rounding {
            base_flags |= 8;
        }
        base_flags |= dtype_mode_val << 4;
        if config.cautious {
            base_flags |= 64;
        }
        base_flags |= state_dtype_val << 8;

        let scratch_slice = scratch.as_slice();
        let cautious_scale = scratch_slice[3];
        let bound_scale = scratch_slice[4];

        let two_pass =
            (!config.fused_single_pass || step <= 1) && (config.cautious || bound_type_val == 1u32);

        let mut base_ema_flags = 0u32;
        if config.cautious {
            base_ema_flags |= 1;
        }
        if bound_type_val == 1u32 {
            base_ema_flags |= 2;
        }

        let wg_count = n.div_ceil(256) as u32;

        unsafe {
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.reset_fences(&[fence])?;
            device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(cb, &begin_info)?;

            let mem_barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
            let dep_info =
                vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mem_barrier));

            if two_pass {
                device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines.pipe_dual,
                );

                let pass1_push = DualPush {
                    param_ptr: param.device_address(),
                    grad_ptr: grad.device_address(),
                    exp_avg_ptr: exp_avg.device_address(),
                    exp_avg_sq_ptr: exp_avg_sq.device_address(),
                    master_ptr,
                    scratch_ptr: scratch.device_address(),
                    n_elements: n as u32,
                    lr: config.lr,
                    weight_decay: config.weight_decay,
                    beta1: config.betas.0,
                    beta2: config.betas.1,
                    c1: coeffs.c1,
                    c2: coeffs.c2,
                    bound_ratio: config.bound_ratio,
                    cautious_scale,
                    bound_scale,
                    diff_mean_scaled,
                    seed_m,
                    seed_p,
                    flags: base_flags | 128 | 1024,
                };
                let pass1_bytes = std::slice::from_raw_parts(
                    &pass1_push as *const _ as *const u8,
                    std::mem::size_of::<DualPush>(),
                );
                device.cmd_push_constants(
                    cb,
                    self.pipelines.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    pass1_bytes,
                );
                device.cmd_dispatch(cb, wg_count, 1, 1);

                device.cmd_pipeline_barrier2(cb, &dep_info);

                device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines.pipe_post_step_ema,
                );
                let post_push = PostStepEmaPush {
                    scratch_ptr: scratch.device_address(),
                    ema_decay: config.ema_decay,
                    n_elements: total_elements,
                    cautious_clamp_min: config.cautious_clamp_min,
                    lr: config.lr,
                    bound_ratio: config.bound_ratio,
                    flags: base_ema_flags | 4,
                };
                let post_bytes = std::slice::from_raw_parts(
                    &post_push as *const _ as *const u8,
                    std::mem::size_of::<PostStepEmaPush>(),
                );
                device.cmd_push_constants(
                    cb,
                    self.pipelines.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    post_bytes,
                );
                device.cmd_dispatch(cb, 1, 1, 1);

                device.cmd_pipeline_barrier2(cb, &dep_info);

                device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines.pipe_dual,
                );
                let pass2_push = DualPush {
                    flags: base_flags | 2048,
                    ..pass1_push
                };
                let pass2_bytes = std::slice::from_raw_parts(
                    &pass2_push as *const _ as *const u8,
                    std::mem::size_of::<DualPush>(),
                );
                device.cmd_push_constants(
                    cb,
                    self.pipelines.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    pass2_bytes,
                );
                device.cmd_dispatch(cb, wg_count, 1, 1);
            } else {
                device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines.pipe_dual,
                );

                let dual_push = DualPush {
                    param_ptr: param.device_address(),
                    grad_ptr: grad.device_address(),
                    exp_avg_ptr: exp_avg.device_address(),
                    exp_avg_sq_ptr: exp_avg_sq.device_address(),
                    master_ptr,
                    scratch_ptr: scratch.device_address(),
                    n_elements: n as u32,
                    lr: config.lr,
                    weight_decay: config.weight_decay,
                    beta1: config.betas.0,
                    beta2: config.betas.1,
                    c1: coeffs.c1,
                    c2: coeffs.c2,
                    bound_ratio: config.bound_ratio,
                    cautious_scale,
                    bound_scale,
                    diff_mean_scaled,
                    seed_m,
                    seed_p,
                    flags: base_flags | 128,
                };

                let push_bytes = std::slice::from_raw_parts(
                    &dual_push as *const _ as *const u8,
                    std::mem::size_of::<DualPush>(),
                );
                device.cmd_push_constants(
                    cb,
                    self.pipelines.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_bytes,
                );
                device.cmd_dispatch(cb, wg_count, 1, 1);

                device.cmd_pipeline_barrier2(cb, &dep_info);

                device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines.pipe_post_step_ema,
                );

                let post_push = PostStepEmaPush {
                    scratch_ptr: scratch.device_address(),
                    ema_decay: config.ema_decay,
                    n_elements: total_elements,
                    cautious_clamp_min: config.cautious_clamp_min,
                    lr: config.lr,
                    bound_ratio: config.bound_ratio,
                    flags: base_ema_flags,
                };

                let post_bytes = std::slice::from_raw_parts(
                    &post_push as *const _ as *const u8,
                    std::mem::size_of::<PostStepEmaPush>(),
                );
                device.cmd_push_constants(
                    cb,
                    self.pipelines.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    post_bytes,
                );
                device.cmd_dispatch(cb, 1, 1, 1);
            }

            device.end_command_buffer(cb)?;

            let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cb));
            device.queue_submit(queue, &[submit_info], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
        }

        Ok(())
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
