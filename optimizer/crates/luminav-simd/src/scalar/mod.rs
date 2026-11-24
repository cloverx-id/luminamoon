pub(crate) mod dual;
pub(crate) mod tdclm;

use luminav_core::{
    BoundType, DTypeMode, LuminaVConfig, apply_decoupled_weight_decay, apply_step_bounds,
};

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn finalize_param_step(
    u: f32,
    m_mask: f32,
    p_init: f32,
    config: &LuminaVConfig,
    bound_mode: BoundType,
    is_triangle: bool,
    dtype_mode: DTypeMode,
    cautious_enabled: bool,
    cautious_scale: f32,
    bound_scale: f32,
    lr_wd: f32,
) -> f32 {
    let delta_theta = if cautious_enabled {
        (u * m_mask) * cautious_scale
    } else {
        u
    };
    let step_val = apply_step_bounds(
        config.lr * delta_theta,
        p_init,
        bound_mode,
        config.bound_ratio,
        config.bound_floor,
        bound_scale,
        dtype_mode,
        is_triangle,
    );

    let p_decayed =
        apply_decoupled_weight_decay(p_init, step_val, lr_wd, config.cautious_weight_decay);
    p_decayed - step_val
}
