pub(crate) mod dual;
pub(crate) mod tdclm;

use luminav_core::{
    BoundType, DTypeMode, LuminaVConfig, apply_coordinate_bound, apply_coordinate_bound_t2,
    apply_fp16_safety_bound, apply_fp16_safety_bound_t2,
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
    let mut step_val = config.lr * delta_theta;

    match bound_mode {
        BoundType::Radial => step_val *= bound_scale,
        BoundType::Coordinate => {
            step_val = if is_triangle {
                apply_coordinate_bound_t2(step_val, p_init, config.bound_ratio)
            } else {
                apply_coordinate_bound(step_val, p_init, config.bound_ratio)
            };
        }
        BoundType::None => {}
    }

    if dtype_mode == DTypeMode::Float16 {
        step_val = if is_triangle {
            apply_fp16_safety_bound_t2(step_val, p_init)
        } else {
            apply_fp16_safety_bound(step_val, p_init)
        };
    }

    let p_decayed = if config.weight_decay != 0.0_f32 {
        (-lr_wd).mul_add(p_init, p_init)
    } else {
        p_init
    };
    p_decayed - step_val
}
