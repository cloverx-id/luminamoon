use ash::vk;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::context::VulkanContext;
use crate::error::VulkanError;

pub struct UmaBuffer<T> {
    device: Arc<ash::Device>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped_ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    size_bytes: vk::DeviceSize,
    device_address: u64,
}

unsafe impl<T: Send> Send for UmaBuffer<T> {}
unsafe impl<T: Sync> Sync for UmaBuffer<T> {}

impl<T: Copy> UmaBuffer<T> {
    pub fn allocate(ctx: &VulkanContext, len: usize) -> Result<Self, VulkanError> {
        let size_bytes = (len * std::mem::size_of::<T>()) as vk::DeviceSize;
        let size_bytes = size_bytes.max(256);

        let buffer_info = vk::BufferCreateInfo::default()
            .size(size_bytes)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { ctx.device.create_buffer(&buffer_info, None)? };
        let mem_reqs = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };

        let cached_device_uma = vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            | vk::MemoryPropertyFlags::HOST_CACHED;

        let cached_host_uma = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            | vk::MemoryPropertyFlags::HOST_CACHED;

        let wc_device_uma = vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;

        let fallback_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let mem_type_index = find_memory_type_index(
            &ctx.memory_properties,
            mem_reqs.memory_type_bits,
            cached_device_uma,
        )
        .or_else(|| {
            find_memory_type_index(
                &ctx.memory_properties,
                mem_reqs.memory_type_bits,
                cached_host_uma,
            )
        })
        .or_else(|| {
            find_memory_type_index(
                &ctx.memory_properties,
                mem_reqs.memory_type_bits,
                wc_device_uma,
            )
        })
        .or_else(|| {
            find_memory_type_index(
                &ctx.memory_properties,
                mem_reqs.memory_type_bits,
                fallback_flags,
            )
        })
        .ok_or(VulkanError::NoSuitableMemoryType(fallback_flags))?;

        let mut alloc_flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type_index)
            .push_next(&mut alloc_flags);

        let memory = unsafe { ctx.device.allocate_memory(&alloc_info, None)? };
        unsafe { ctx.device.bind_buffer_memory(buffer, memory, 0)? };

        let raw_ptr = unsafe {
            ctx.device
                .map_memory(memory, 0, mem_reqs.size, vk::MemoryMapFlags::empty())
                .map_err(VulkanError::MapFailed)?
        };

        let mapped_ptr = NonNull::new(raw_ptr as *mut T)
            .ok_or(VulkanError::MapFailed(vk::Result::ERROR_MEMORY_MAP_FAILED))?;

        let addr_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let device_address = unsafe { ctx.device.get_buffer_device_address(&addr_info) };

        Ok(Self {
            device: Arc::clone(&ctx.device),
            buffer,
            memory,
            mapped_ptr,
            len,
            capacity: len,
            size_bytes: mem_reqs.size,
            device_address,
        })
    }

    #[inline]
    pub fn set_len(&mut self, new_len: usize) {
        assert!(new_len <= self.capacity);
        self.len = new_len;
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.mapped_ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.mapped_ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn device_address(&self) -> u64 {
        self.device_address
    }

    #[inline]
    pub fn size_bytes(&self) -> vk::DeviceSize {
        self.size_bytes
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn copy_from_slice(&mut self, src: &[T]) {
        assert_eq!(self.len, src.len());
        self.as_slice_mut().copy_from_slice(src);
    }

    pub fn copy_to_slice(&self, dst: &mut [T]) {
        assert_eq!(self.len, dst.len());
        dst.copy_from_slice(self.as_slice());
    }
}

impl<T> Drop for UmaBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            self.device.unmap_memory(self.memory);
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

fn find_memory_type_index(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    property_flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..mem_props.memory_type_count).find(|&i| {
        (type_filter & (1 << i)) != 0
            && (mem_props.memory_types[i as usize].property_flags & property_flags)
                == property_flags
    })
}
