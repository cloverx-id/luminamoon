use cubecl::prelude::*;

use super::math::{
    apply_rpm_cube, coord_bound_cube, coord_bound_t2_cube, decoupled_weight_decay_cube,
    fp16_safety_bound_cube, fp16_safety_bound_t2_cube, quantize_value_weyl_cube, reduce_cube_sum,
    sanitize_gradient_cube, tanh_pade_cube, temper_belief_defect_cube,
};

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn lumina_dual_buffer_stats_kernel(
    param: &[f32],
    master: &[f32],
    grad: &[f32],
    exp_avg: &[f32],
    exp_avg_sq: &[f32],
    partial_mask: &mut [f32],
    partial_u_sq: &mut [f32],
    partial_p_sq: &mut [f32],
    n: usize,
    beta1: f32,
    one_minus_beta1: f32,
    beta2: f32,
    one_minus_beta2: f32,
    beta1_sq: f32,
    one_minus_beta1_sq: f32,
    c1: f32,
    c2: f32,
    eps: f32,
    variance_temper: f32,
    inv_sqrt_bc2_prev: f32,
    #[comptime] is_t2: bool,
    #[comptime] use_rtv: bool,
    #[comptime] use_rpm: bool,
    #[comptime] has_master: bool,
    #[comptime] cautious: bool,
    #[comptime] radial: bool,
    #[comptime] cube_dim: usize,
) {
    let idx = ABSOLUTE_POS;
    let mut mask_v = 0.0_f32;
    let mut u_sq_v = 0.0_f32;
    let mut p_sq_v = 0.0_f32;

    if idx < n {
        let g = sanitize_gradient_cube(grad[idx]);
        let m_raw = exp_avg[idx];
        let m = if comptime!(use_rpm) {
            let p_idx = idx ^ 1_usize;
            if p_idx < n {
                apply_rpm_cube(
                    m_raw,
                    exp_avg[p_idx],
                    g,
                    sanitize_gradient_cube(grad[p_idx]),
                    one_minus_beta1,
                    eps,
                )
            } else {
                m_raw
            }
        } else {
            m_raw
        };
        let v = exp_avg_sq[idx];
        let u = if comptime!(is_t2) {
            let clm_m = beta1_sq * m + one_minus_beta1_sq * g;
            let diff = f32::clamp(g - m, -1.0e18_f32, 1.0e18_f32);
            let abs_d = f32::abs(diff);
            let d_mem = if comptime!(use_rtv) {
                temper_belief_defect_cube(abs_d, v, m, inv_sqrt_bc2_prev, variance_temper, eps)
            } else {
                abs_d
            };
            let v_new = beta2 * v + one_minus_beta2 * (d_mem * d_mem);
            let env = beta1_sq * f32::abs(m)
                + one_minus_beta1_sq * f32::abs(g)
                + c2 * abs_d
                + c1 * f32::sqrt(f32::max(v_new, 0.0_f32))
                + eps;
            clm_m / env
        } else {
            let m_new = beta1 * m + one_minus_beta1 * g;
            let clm_m = beta1 * m_new + one_minus_beta1 * g;
            let diff = f32::clamp(g - m_new, -1.0e18_f32, 1.0e18_f32);
            let abs_d = f32::abs(diff);
            let d_mem = if comptime!(use_rtv) {
                temper_belief_defect_cube(abs_d, v, m, inv_sqrt_bc2_prev, variance_temper, eps)
            } else {
                abs_d
            };
            let v_new = beta2 * v + one_minus_beta2 * (d_mem * d_mem);
            let sigma = f32::sqrt(f32::max(v_new, 0.0_f32)) * c1 + c2;
            tanh_pade_cube(clm_m / f32::max(sigma, 1.0e-12_f32))
        };

        let active = (u * g) > 0.0_f32;
        let m_mask = if active { 1.0_f32 } else { 0.0_f32 };
        if comptime!(cautious) {
            mask_v = m_mask;
        }
        if comptime!(radial) {
            let u_eff = if comptime!(cautious) { u * m_mask } else { u };
            let p_val = if comptime!(has_master) {
                master[idx]
            } else {
                param[idx]
            };
            u_sq_v = u_eff * u_eff;
            p_sq_v = p_val * p_val;
        }
    }

    let s_mask = reduce_cube_sum(mask_v, cube_dim);
    let s_u = reduce_cube_sum(u_sq_v, cube_dim);
    let s_p = reduce_cube_sum(p_sq_v, cube_dim);
    if UNIT_POS_X == 0 {
        let cid = CUBE_POS_X as usize;
        partial_mask[cid] = s_mask;
        partial_u_sq[cid] = s_u;
        partial_p_sq[cid] = s_p;
    }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn lumina_dual_buffer_step_kernel(
    param: &mut [f32],
    master: &mut [f32],
    grad: &[f32],
    exp_avg: &mut [f32],
    exp_avg_sq: &mut [f32],
    partial_mask: &mut [f32],
    partial_u_sq: &mut [f32],
    partial_p_sq: &mut [f32],
    n: usize,
    lr: f32,
    beta1: f32,
    one_minus_beta1: f32,
    beta2: f32,
    one_minus_beta2: f32,
    beta1_sq: f32,
    one_minus_beta1_sq: f32,
    c1: f32,
    c2: f32,
    eps: f32,
    lr_wd: f32,
    bound_ratio: f32,
    bound_floor: f32,
    variance_temper: f32,
    inv_sqrt_bc2_prev: f32,
    cautious_scale: f32,
    bound_scale: f32,
    seed_m: u32,
    seed_v: u32,
    seed_p: u32,
    weyl_m: u32,
    weyl_v: u32,
    weyl_p: u32,
    dtype_mode: u32,
    state_dtype_mode: u32,
    stochastic: u32,
    #[comptime] is_t2: bool,
    #[comptime] use_rtv: bool,
    #[comptime] use_rpm: bool,
    #[comptime] has_master: bool,
    #[comptime] cautious: bool,
    #[comptime] cautious_wd: bool,
    #[comptime] bound_mode: u32,
    #[comptime] collect_stats: bool,
    #[comptime] cube_dim: usize,
) {
    let idx = ABSOLUTE_POS;
    let mut mask_v = 0.0_f32;
    let mut u_sq_v = 0.0_f32;
    let mut p_sq_v = 0.0_f32;
    let mut g_loaded = 0.0_f32;
    let mut m_loaded = 0.0_f32;

    if idx < n {
        g_loaded = sanitize_gradient_cube(grad[idx]);
        let m_raw = exp_avg[idx];
        m_loaded = if comptime!(use_rpm) {
            let p_idx = idx ^ 1_usize;
            if p_idx < n {
                apply_rpm_cube(
                    m_raw,
                    exp_avg[p_idx],
                    g_loaded,
                    sanitize_gradient_cube(grad[p_idx]),
                    one_minus_beta1,
                    eps,
                )
            } else {
                m_raw
            }
        } else {
            m_raw
        };
    }
    if comptime!(use_rpm) {
        sync_cube();
    }

    if idx < n {
        let p_init = if comptime!(has_master) {
            master[idx]
        } else {
            param[idx]
        };
        let g = g_loaded;
        let m = m_loaded;
        let v = exp_avg_sq[idx];
        let m_new = beta1 * m + one_minus_beta1 * g;

        let (u, v_new) = if comptime!(is_t2) {
            let clm_m = beta1_sq * m + one_minus_beta1_sq * g;
            let diff = f32::clamp(g - m, -1.0e18_f32, 1.0e18_f32);
            let abs_d = f32::abs(diff);
            let d_mem = if comptime!(use_rtv) {
                temper_belief_defect_cube(abs_d, v, m, inv_sqrt_bc2_prev, variance_temper, eps)
            } else {
                abs_d
            };
            let v_val = beta2 * v + one_minus_beta2 * (d_mem * d_mem);
            let env = beta1_sq * f32::abs(m)
                + one_minus_beta1_sq * f32::abs(g)
                + c2 * abs_d
                + c1 * f32::sqrt(f32::max(v_val, 0.0_f32))
                + eps;
            (clm_m / env, v_val)
        } else {
            let clm_m = beta1 * m_new + one_minus_beta1 * g;
            let diff = f32::clamp(g - m_new, -1.0e18_f32, 1.0e18_f32);
            let abs_d = f32::abs(diff);
            let d_mem = if comptime!(use_rtv) {
                temper_belief_defect_cube(abs_d, v, m, inv_sqrt_bc2_prev, variance_temper, eps)
            } else {
                abs_d
            };
            let v_val = beta2 * v + one_minus_beta2 * (d_mem * d_mem);
            let sigma = f32::sqrt(f32::max(v_val, 0.0_f32)) * c1 + c2;
            (tanh_pade_cube(clm_m / f32::max(sigma, 1.0e-12_f32)), v_val)
        };

        let active = (u * g) > 0.0_f32;
        let m_mask = if active { 1.0_f32 } else { 0.0_f32 };

        exp_avg[idx] = quantize_value_weyl_cube(
            m_new,
            seed_m,
            idx as u32,
            weyl_m,
            state_dtype_mode,
            stochastic,
        );
        exp_avg_sq[idx] = quantize_value_weyl_cube(
            v_new,
            seed_v,
            idx as u32,
            weyl_v,
            state_dtype_mode,
            stochastic,
        );

        let delta_theta = if comptime!(cautious) {
            (u * m_mask) * cautious_scale
        } else {
            u
        };
        let mut step_val = lr * delta_theta;
        if comptime!(bound_mode == 1_u32) {
            step_val *= bound_scale;
        } else if comptime!(bound_mode == 2_u32) {
            step_val = if comptime!(is_t2) {
                coord_bound_t2_cube(step_val, p_init, bound_ratio, bound_floor, dtype_mode)
            } else {
                coord_bound_cube(step_val, p_init, bound_ratio, bound_floor, dtype_mode)
            };
        }
        if comptime!(bound_mode != 2_u32) && dtype_mode == 2_u32 {
            step_val = if comptime!(is_t2) {
                fp16_safety_bound_t2_cube(step_val, p_init)
            } else {
                fp16_safety_bound_cube(step_val, p_init)
            };
        }

        let p_decayed = decoupled_weight_decay_cube(p_init, step_val, lr_wd, cautious_wd);
        let p_updated = p_decayed - step_val;

        if comptime!(has_master) {
            master[idx] = p_updated;
            param[idx] =
                quantize_value_weyl_cube(p_updated, seed_p, idx as u32, weyl_p, dtype_mode, 0_u32);
        } else {
            param[idx] = quantize_value_weyl_cube(
                p_updated, seed_p, idx as u32, weyl_p, dtype_mode, stochastic,
            );
        }

        if comptime!(collect_stats) {
            if comptime!(cautious) {
                mask_v = m_mask;
            }
            if comptime!(bound_mode == 1_u32) {
                let u_eff = if comptime!(cautious) { u * m_mask } else { u };
                u_sq_v = u_eff * u_eff;
                p_sq_v = p_init * p_init;
            }
        }
    }

    if comptime!(collect_stats) {
        let s_mask = reduce_cube_sum(mask_v, cube_dim);
        let s_u = reduce_cube_sum(u_sq_v, cube_dim);
        let s_p = reduce_cube_sum(p_sq_v, cube_dim);
        if UNIT_POS_X == 0 {
            let cid = CUBE_POS_X as usize;
            partial_mask[cid] = s_mask;
            partial_u_sq[cid] = s_u;
            partial_p_sq[cid] = s_p;
        }
    }
}
