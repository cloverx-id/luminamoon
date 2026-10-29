# `luminav-vulkan`: Ash Vulkan Compute Pipeline & Wave32 UMA

`luminav-vulkan` delivers cross-platform GPU compute acceleration via standard Vulkan 1.3 / 1.4 compute shaders, targeting non-NVIDIA GPUs (AMD Radeon, Intel Arc/Iris, Apple via MoltenVK) and Unified Memory Architecture (UMA) APUs.

---

## 1. Vulkan Compute Architecture

The compute pipeline is orchestrated via low-level Rust `ash` bindings without large runtime dependencies:

```mermaid
flowchart LR
    Context["Ash Vulkan 1.3/1.4 Context"] --> Map["Zero-Copy UMA Buffer Map (HOST_VISIBLE)"]
    Context --> Shaders["Load SPIR-V Shaders (TD-CLM / Dual-Fused)"]
    Map --> Dispatch["Submit Compute Command Buffer"]
    Shaders --> Dispatch
    Dispatch --> Subgroup["Wave32 Subgroup Reduction (subgroupAdd)"]
    Subgroup --> Sync["In-Place Host Memory Update"]
```

---

## 2. SPIR-V Compute Shaders (`shaders/`)

The crate compiles high-performance GLSL compute shaders into portable SPIR-V bytecode:

1. **`tdclm_single_pass.comp`**: Single-buffer TD-CLM kernel computing temporal difference defect compensation in a single dispatch pass.
2. **`dual_fused_single_pass.comp`**: Dual-buffer CLM compute shader fusing momentum updates and EMA boundary calculations.
3. **`post_step_ema.comp`**: Asynchronous post-step EMA smoothing for cautious and bounding coefficients.

---

## 3. Wave32 Subgroup Reductions & UMA Zero-Copy

### 3.1 Wave32 / Wave64 Subgroups
- Leverages Vulkan `VK_KHR_shader_subgroup_basic` and `subgroup_arithmetic` extensions.
- Executes fast intra-warp reductions using `subgroupAdd()` and `subgroupElect()`, bypassing shared memory fences where available.

### 3.2 Unified Memory Architecture (UMA) Buffer Mapping
On modern APUs (AMD Ryzen with Radeon 780M/890M or Intel Core Ultra):

- Persistent Unified Memory (`UmaWorkspace`) buffers are mapped using `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`.
- The GPU compute engine reads and mutates persistent host-visible UMA workspace slices (`copy_from_slice` / `copy_to_slice`) without discrete PCIe staging buffer allocations.
