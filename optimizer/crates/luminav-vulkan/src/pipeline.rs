use ash::vk;
use std::sync::Arc;

use crate::context::VulkanContext;
use crate::error::VulkanError;

const DIFF_SUM_SPV: &[u8] = include_bytes!("../shaders/diff_sum.spv");
const TDCLM_SPV: &[u8] = include_bytes!("../shaders/tdclm_single_pass.spv");
const DUAL_SPV: &[u8] = include_bytes!("../shaders/dual_fused_single_pass.spv");
const POST_STEP_EMA_SPV: &[u8] = include_bytes!("../shaders/post_step_ema.spv");

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DiffSumPush {
    pub grad_ptr: u64,
    pub exp_avg_ptr: u64,
    pub scratch_ptr: u64,
    pub n_elements: u32,
    pub _padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TdclmPush {
    pub param_ptr: u64,
    pub grad_ptr: u64,
    pub exp_avg_ptr: u64,
    pub master_ptr: u64,
    pub scratch_ptr: u64,
    pub n_elements: u32,
    pub lr: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta1_sq: f32,
    pub one_minus_beta1_sq: f32,
    pub eps: f32,
    pub lam: f32,
    pub bound_ratio: f32,
    pub cautious_scale: f32,
    pub bound_scale: f32,
    pub diff_mean: f32,
    pub seed_m: i32,
    pub seed_p: i32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DualPush {
    pub param_ptr: u64,
    pub grad_ptr: u64,
    pub exp_avg_ptr: u64,
    pub exp_avg_sq_ptr: u64,
    pub master_ptr: u64,
    pub scratch_ptr: u64,
    pub n_elements: u32,
    pub lr: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub c1: f32,
    pub c2: f32,
    pub bound_ratio: f32,
    pub cautious_scale: f32,
    pub bound_scale: f32,
    pub diff_mean_scaled: f32,
    pub seed_m: i32,
    pub seed_p: i32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PostStepEmaPush {
    pub scratch_ptr: u64,
    pub ema_decay: f32,
    pub n_elements: f32,
    pub cautious_clamp_min: f32,
    pub lr: f32,
    pub bound_ratio: f32,
    pub flags: u32,
}

pub struct VulkanPipelines {
    device: Arc<ash::Device>,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipe_diff_sum: vk::Pipeline,
    pub pipe_tdclm: vk::Pipeline,
    pub pipe_dual: vk::Pipeline,
    pub pipe_post_step_ema: vk::Pipeline,
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub fence: vk::Fence,
}

impl VulkanPipelines {
    pub fn new(ctx: &VulkanContext) -> Result<Self, VulkanError> {
        let device = Arc::clone(&ctx.device);

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(128);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        let sm_diff_sum = create_shader_module(&device, DIFF_SUM_SPV)?;
        let sm_tdclm = create_shader_module(&device, TDCLM_SPV)?;
        let sm_dual = create_shader_module(&device, DUAL_SPV)?;
        let sm_post_step = create_shader_module(&device, POST_STEP_EMA_SPV)?;

        let pipe_diff_sum =
            create_compute_pipeline(&device, pipeline_layout, sm_diff_sum, ctx.wave32_supported)?;
        let pipe_tdclm =
            create_compute_pipeline(&device, pipeline_layout, sm_tdclm, ctx.wave32_supported)?;
        let pipe_dual =
            create_compute_pipeline(&device, pipeline_layout, sm_dual, ctx.wave32_supported)?;
        let pipe_post_step_ema =
            create_compute_pipeline(&device, pipeline_layout, sm_post_step, false)?;

        unsafe {
            device.destroy_shader_module(sm_diff_sum, None);
            device.destroy_shader_module(sm_tdclm, None);
            device.destroy_shader_module(sm_dual, None);
            device.destroy_shader_module(sm_post_step, None);
        }

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(ctx.queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe { device.create_command_pool(&pool_info, None)? };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info)? };
        let command_buffer = command_buffers[0];

        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let fence = unsafe { device.create_fence(&fence_info, None)? };

        Ok(Self {
            device,
            pipeline_layout,
            pipe_diff_sum,
            pipe_tdclm,
            pipe_dual,
            pipe_post_step_ema,
            command_pool,
            command_buffer,
            fence,
        })
    }
}

impl Drop for VulkanPipelines {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_pipeline(self.pipe_diff_sum, None);
            self.device.destroy_pipeline(self.pipe_tdclm, None);
            self.device.destroy_pipeline(self.pipe_dual, None);
            self.device.destroy_pipeline(self.pipe_post_step_ema, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

fn create_shader_module(
    device: &ash::Device,
    spv_bytes: &[u8],
) -> Result<vk::ShaderModule, VulkanError> {
    let mut cursor = std::io::Cursor::new(spv_bytes);
    let code = ash::util::read_spv(&mut cursor)
        .map_err(|e| VulkanError::InvalidShape(format!("Failed to parse SPIR-V bytes: {e}")))?;

    let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
    let sm = unsafe { device.create_shader_module(&create_info, None)? };
    Ok(sm)
}

fn create_compute_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    wave32: bool,
) -> Result<vk::Pipeline, VulkanError> {
    let entry_name = c"main";
    let mut stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(entry_name);

    let mut required_subgroup =
        vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(32);

    if wave32 {
        stage_info = stage_info.push_next(&mut required_subgroup);
    }

    let pipe_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage_info)
        .layout(layout);

    let pipelines = unsafe {
        device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipe_info], None)
            .map_err(|(_, r)| VulkanError::Vk(r))?
    };

    Ok(pipelines[0])
}
