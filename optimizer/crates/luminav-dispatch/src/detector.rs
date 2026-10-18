use std::sync::OnceLock;

use luminav_simd::{SimdBackend, detect_simd_backend};

use crate::router::get_global_dispatcher;

#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    pub simd_backend: SimdBackend,
    pub triton_aot_available: bool,
    pub triton_arch: Option<String>,
    pub vulkan_available: bool,
    pub gpu_device_name: Option<String>,
    pub wave32_supported: bool,
}

static HARDWARE_CAPABILITIES: OnceLock<HardwareCapabilities> = OnceLock::new();

pub fn detect_hardware() -> &'static HardwareCapabilities {
    HARDWARE_CAPABILITIES.get_or_init(|| {
        let simd_backend = detect_simd_backend();
        if let Ok(guard) = get_global_dispatcher().lock() {
            let (triton_aot_available, triton_arch) = match &guard.triton_engine {
                Some(engine) => (true, Some(engine.arch_tag().to_string())),
                None => (false, None),
            };
            let (vulkan_available, gpu_device_name, wave32_supported) = match &guard.vulkan_engine {
                Some(vk_eng) => (
                    true,
                    Some(vk_eng.context.device_name.clone()),
                    vk_eng.context.wave32_supported,
                ),
                None => (false, None, false),
            };
            return HardwareCapabilities {
                simd_backend,
                triton_aot_available,
                triton_arch,
                vulkan_available,
                gpu_device_name,
                wave32_supported,
            };
        }

        HardwareCapabilities {
            simd_backend,
            triton_aot_available: false,
            triton_arch: None,
            vulkan_available: false,
            gpu_device_name: None,
            wave32_supported: false,
        }
    })
}
