use std::sync::OnceLock;

use luminav_simd::{SimdBackend, detect_simd_backend};

use crate::router::get_global_dispatcher;

#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    pub simd_backend: SimdBackend,
    pub triton_aot_available: bool,
    pub triton_arch: Option<String>,
    pub cubecl_available: bool,
    pub cubecl_backend: Option<String>,
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
            #[cfg(feature = "cubecl")]
            let (cubecl_available, cubecl_backend) = match &guard.cubecl_engine {
                Some(engine) => (true, Some(engine.backend().name().to_string())),
                None => (false, None),
            };
            #[cfg(not(feature = "cubecl"))]
            let (cubecl_available, cubecl_backend) = (false, None);
            let gpu_device_name = cubecl_backend.as_ref().map(|b| format!("cubecl:{b}"));
            return HardwareCapabilities {
                simd_backend,
                triton_aot_available,
                triton_arch,
                cubecl_available,
                cubecl_backend,
                vulkan_available: false,
                gpu_device_name,
                wave32_supported: false,
            };
        }

        HardwareCapabilities {
            simd_backend,
            triton_aot_available: false,
            triton_arch: None,
            cubecl_available: false,
            cubecl_backend: None,
            vulkan_available: false,
            gpu_device_name: None,
            wave32_supported: false,
        }
    })
}
