use ash::vk;
use std::ffi::CStr;
use std::sync::Arc;

use crate::error::VulkanError;

pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: Arc<ash::Device>,
    pub queue_family_index: u32,
    pub compute_queue: vk::Queue,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub device_name: String,
    pub wave32_supported: bool,
}

impl VulkanContext {
    pub fn new() -> Result<Self, VulkanError> {
        let entry = unsafe { ash::Entry::load()? };

        let supported_instance_version = unsafe {
            entry
                .try_enumerate_instance_version()?
                .unwrap_or(vk::API_VERSION_1_3)
        };
        let target_api_version = if supported_instance_version >= vk::make_api_version(0, 1, 4, 0) {
            vk::make_api_version(0, 1, 4, 0)
        } else {
            vk::make_api_version(0, 1, 3, 0)
        };

        let app_name = c"luminav-vulkan";
        let engine_name = c"LuminaMoon";
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 1, 5, 3))
            .engine_name(engine_name)
            .engine_version(vk::make_api_version(0, 1, 5, 3))
            .api_version(target_api_version);

        let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);

        let instance = unsafe { entry.create_instance(&instance_info, None)? };

        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        if physical_devices.is_empty() {
            unsafe { instance.destroy_instance(None) };
            return Err(VulkanError::NoPhysicalDevice);
        }

        let mut candidates = Vec::new();
        let uma_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;

        for (idx, &pdev) in physical_devices.iter().enumerate() {
            let q_props = unsafe { instance.get_physical_device_queue_family_properties(pdev) };
            let mut compute_q = None;
            for (q_idx, q) in q_props.iter().enumerate() {
                if q.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    compute_q = Some(q_idx as u32);
                    break;
                }
            }

            let Some(compute_queue) = compute_q else {
                continue;
            };

            let props = unsafe { instance.get_physical_device_properties(pdev) };
            let raw_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
            let name = raw_name.to_string_lossy().into_owned();

            let mem_props = unsafe { instance.get_physical_device_memory_properties(pdev) };
            let has_uma_memory = mem_props.memory_types[..mem_props.memory_type_count as usize]
                .iter()
                .any(|mt| mt.property_flags.contains(uma_flags));
            let is_uma =
                has_uma_memory || props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU;

            candidates.push((idx, pdev, name, compute_queue, is_uma, mem_props));
        }

        if candidates.is_empty() {
            unsafe { instance.destroy_instance(None) };
            return Err(VulkanError::NoComputeQueue);
        }

        let selected = if let Ok(env_val) = std::env::var("LUMINAV_VULKAN_DEVICE") {
            let env_spec = env_val.trim();
            let requested_idx = env_spec.parse::<usize>().ok();
            candidates
                .iter()
                .find(|(idx, _, name, ..)| {
                    requested_idx.is_some_and(|req_idx| *idx == req_idx)
                        || name.to_lowercase().contains(&env_spec.to_lowercase())
                })
                .or_else(|| candidates.iter().find(|(.., is_uma, _)| *is_uma))
                .or_else(|| candidates.first())
        } else {
            candidates
                .iter()
                .find(|(.., is_uma, _)| *is_uma)
                .or_else(|| candidates.first())
        };

        let Some(&(_, pdevice, ref selected_name, queue_family_index, _, memory_properties)) =
            selected
        else {
            unsafe { instance.destroy_instance(None) };
            return Err(VulkanError::NoPhysicalDevice);
        };
        let selected_name = selected_name.clone();

        let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut features12)
            .push_next(&mut features13);

        unsafe { instance.get_physical_device_features2(pdevice, &mut features2) };

        if features12.buffer_device_address == vk::FALSE {
            unsafe { instance.destroy_instance(None) };
            return Err(VulkanError::BdaNotSupported);
        }

        let mut props13 = vk::PhysicalDeviceVulkan13Properties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut props13);
        unsafe { instance.get_physical_device_properties2(pdevice, &mut props2) };

        let wave32_supported = features13.subgroup_size_control == vk::TRUE
            && props13.min_subgroup_size <= 32
            && props13.max_subgroup_size >= 32
            && props13
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE);

        let queue_priorities = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);
        let queue_infos = [queue_info];

        let mut enable_features12 =
            vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true);

        let mut enable_features13 = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(features13.synchronization2 == vk::TRUE);

        if wave32_supported {
            enable_features13 = enable_features13
                .subgroup_size_control(true)
                .compute_full_subgroups(features13.compute_full_subgroups == vk::TRUE);
        }

        let mut enable_features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut enable_features12)
            .push_next(&mut enable_features13);

        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .push_next(&mut enable_features2);

        let device = unsafe { instance.create_device(pdevice, &device_info, None)? };
        let compute_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        Ok(Self {
            entry,
            instance,
            physical_device: pdevice,
            device: Arc::new(device),
            queue_family_index,
            compute_queue,
            memory_properties,
            device_name: selected_name,
            wave32_supported,
        })
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
