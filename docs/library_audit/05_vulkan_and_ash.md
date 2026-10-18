# 5. Vulkan Compute (1.3 / 1.4) dan Crate `ash`

## 5.1 Versi Stabil Terbaru
- **Spesifikasi Vulkan**: `Vulkan 1.4` (dengan kompatibilitas penuh `Vulkan 1.3`)
- **`ash`**: `0.38.0+1.3.281`

---

## 5.2 Riwayat Perubahan Vulkan (`1.2` - `1.4`) & `ash` (`0.37` - `0.38.0`)

### A. Promosi `VK_EXT_subgroup_size_control` dan `VK_KHR_synchronization2` ke Inti Vulkan 1.3
- **Apa yang berubah**:
  - Ekstensi `VK_EXT_subgroup_size_control` dipromosikan ke inti Vulkan 1.3 di dalam struktur `vk::PhysicalDeviceVulkan13Features` (`subgroup_size_control` dan `compute_full_subgroups`) serta `vk::PhysicalDeviceVulkan13Properties` (`min_subgroup_size`, `max_subgroup_size`, `required_subgroup_size_stages`).
  - Struktur pipeline `VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT` dipromosikan menjadi tipe inti `vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo` tanpa sufiks `EXT`.
  - Ekstensi `VK_KHR_synchronization2` (`vkCmdPipelineBarrier2`, `VkMemoryBarrier2`, `VkDependencyInfo`) dipromosikan ke inti Vulkan 1.3.

### B. Aturan Validasi Spesifikasi `VUID-VkPipelineShaderStageCreateInfo-pNext-02756`
- **Apa yang berubah**:
  Menurut spesifikasi Vulkan 1.3 dan 1.4, menyematkan `vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo` dengan `required_subgroup_size = 32` pada *compute pipeline* hanya sah apabila ketiga syarat berikut terpenuhi secara bersamaan:
  1. `features13.subgroup_size_control == vk::TRUE`
  2. `props13.min_subgroup_size <= 32 && props13.max_subgroup_size >= 32`
  3. `props13.required_subgroup_size_stages` memiliki bit `vk::ShaderStageFlags::COMPUTE`
- **Mengapa penting**: Memeriksa `features13.subgroup_size_control == vk::TRUE` saja tanpa memeriksa `PhysicalDeviceVulkan13Properties` dapat menyebabkan kegagalan pembuatan pipeline (`VK_ERROR_INITIALIZATION_FAILED` atau *validation error*) pada GPU yang mendukung `subgroup_size_control` tetapi memiliki `min_subgroup_size = 64` atau tidak mengizinkan pengaturan ukuran subgroup pada tahap `COMPUTE`.

### C. Negosiasi Versi Instance Dinamis (`Vulkan 1.4` vs `1.3`)
- **Apa yang berubah**: Pada driver modern yang telah mendukung `Vulkan 1.4`, pemanggilan `entry.try_enumerate_instance_version()` mengembalikan versi runtime loader `>= 1.4.0`.

---

## 5.3 Adaptasi pada Kode Sumber LuminaMoon

### 1. Negosiasi API `1.4.0` / `1.3.0` dan Validasi Properti Subgroup
- **Berkas**: `optimizer/crates/luminav-vulkan/src/context.rs`
- **Implementasi**:
  ```rust
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
  ```
  Serta pengecekan properti perangkat Vulkan 1.3 untuk mode Wave32:
  ```rust
  let mut props13 = vk::PhysicalDeviceVulkan13Properties::default();
  let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut props13);
  unsafe { instance.get_physical_device_properties2(pdevice, &mut props2) };

  let wave32_supported = features13.subgroup_size_control == vk::TRUE
      && props13.min_subgroup_size <= 32
      && props13.max_subgroup_size >= 32
      && props13
          .required_subgroup_size_stages
          .contains(vk::ShaderStageFlags::COMPUTE);
  ```

### 2. Pembersihan `Drop` pada `VulkanContext`
- **Berkas**: `optimizer/crates/luminav-vulkan/src/context.rs`
- **Implementasi**: Menyederhanakan blok penghancuran perangkat logis (`self.device.destroy_device(None)`) setelah `device_wait_idle()` tanpa percabangan `Arc::get_mut` yang redundan.
