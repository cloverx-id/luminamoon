pub(crate) mod dual;
pub(crate) mod tdclm;

use luminav_core::{
    BoundType, LuminaVConfig, compute_cautious_scale, compute_radial_bound_scale,
    compute_radial_bound_scale_t2,
};

use crate::common::ChunkMetrics;

#[inline]
pub(crate) fn resolve_pass2_scales(
    reduced: ChunkMetrics,
    total_elements: f32,
    bound_mode: BoundType,
    config: &LuminaVConfig,
) -> (f32, f32) {
    let cautious_scale = if config.cautious {
        compute_cautious_scale(
            reduced.mask_count as f32,
            total_elements,
            config.cautious_clamp_min,
        )
    } else {
        1.0_f32
    };

    let bound_scale = if bound_mode == BoundType::Radial {
        if config.buffer.is_triangle_family() {
            compute_radial_bound_scale_t2(
                reduced.u_sq_sum,
                reduced.p_sq_sum,
                cautious_scale,
                config.lr,
                config.bound_ratio,
            )
        } else {
            compute_radial_bound_scale(
                reduced.u_sq_sum,
                reduced.p_sq_sum,
                cautious_scale,
                config.lr,
                config.bound_ratio,
            )
        }
    } else {
        1.0_f32
    };

    (cautious_scale, bound_scale)
}
