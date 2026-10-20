# 5. Vulkan Compute (1.3 / 1.4) and the `ash` Crate

## 5.1 Latest Stable Versions
- **Vulkan Specification**: `Vulkan 1.4` (with full `Vulkan 1.3` compatibility)
- **`ash`**: `0.38.0+1.3.281`

---

## 5.2 Vulkan (`1.2` - `1.4`) & `ash` (`0.37` - `0.38.0`) Changelog

### A. Promotion of `VK_EXT_subgroup_size_control` and `VK_KHR_synchronization2` to Vulkan 1.3 Core
- **What changed**:
  - `VK_EXT_subgroup_size_control` was promoted to Vulkan 1.3 core inside `vk::PhysicalDeviceVulkan13Features` (`subgroup_size_control` and `compute_full_subgroups`) and `vk::PhysicalDeviceVulkan13Properties` (`min_subgroup_size`, `max_subgroup_size`, `required_subgroup_size_stages`).
  - `VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT` was promoted to the core type `vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo` without the `EXT` suffix.
  - `VK_KHR_synchronization2` (`vkCmdPipelineBarrier2`, `VkMemoryBarrier2`, `VkDependencyInfo`) was promoted to Vulkan 1.3 core.

### B. Specification Validation Rule `VUID-VkPipelineShaderStageCreateInfo-pNext-02756`
- **What changed**:
  Under the Vulkan 1.3 and 1.4 specifications, chaining `vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo` with `required_subgroup_size = 32` onto a compute pipeline stage is only valid when all three conditions hold simultaneously:
  1. `features13.subgroup_size_control == vk::TRUE`
  2. `props13.min_subgroup_size <= 32 && props13.max_subgroup_size >= 32`
  3. `props13.required_subgroup_size_stages` contains `vk::ShaderStageFlags::COMPUTE`
- **Why it matters**: Checking `features13.subgroup_size_control == vk::TRUE` alone without inspecting `PhysicalDeviceVulkan13Properties` can cause pipeline creation failure (`VK_ERROR_INITIALIZATION_FAILED` or validation errors) on GPUs that support `subgroup_size_control` but have `min_subgroup_size = 64` or do not permit subgroup size overrides in the `COMPUTE` stage.

### C. Dynamic Instance Version Negotiation (`Vulkan 1.4` vs `1.3`)
- **What changed**: On modern drivers supporting `Vulkan 1.4`, `entry.try_enumerate_instance_version()` returns a runtime loader version `>= 1.4.0`.

---

## 5.3 Adaptations in LuminaMoon Source Code

### 1. Dynamic `1.4.0` / `1.3.0` API Negotiation and Subgroup Property Validation
- **File**: `optimizer/crates/luminav-vulkan/src/context.rs`
- **Implementation**:
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
  And Vulkan 1.3 device property verification for Wave32 mode:
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

### 2. Clean `Drop` Implementation on `VulkanContext`
- **File**: `optimizer/crates/luminav-vulkan/src/context.rs`
- **Implementation**: Streamlined logical device destruction (`self.device.destroy_device(None)`) after `device_wait_idle()` without redundant `Arc::get_mut` branching.
