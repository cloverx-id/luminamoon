/*
 * LuminaMoon: LuminaV Optimizer Suite v1.0.0
 * Standalone C/C++ ABI Header for Triton AOT GPU Acceleration
 *
 * Enables native CUDA and ROCm HIP execution of LuminaV optimizer kernels
 * directly from C, C++, or low-level systems languages without Python or PyTorch.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#ifndef LUMINAV_TRITON_STANDALONE_H
#define LUMINAV_TRITON_STANDALONE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque pointer to Triton AOT Engine instance */
typedef struct LuminaVTritonEngine LuminaVTritonEngine;

/* Status codes */
#define LUMINAV_STATUS_SUCCESS 0
#define LUMINAV_STATUS_NULL_POINTER -1
#define LUMINAV_STATUS_NO_DRIVER -2
#define LUMINAV_STATUS_INVALID_CONFIG -3
#define LUMINAV_STATUS_EXECUTION_FAILED -4

/* Buffer Modes */
#define LUMINAV_BUFFER_TDCLM 1  /* Single-Buffer TD-CLM */
#define LUMINAV_BUFFER_DUAL 2   /* Dual-Buffer CLM */

/* Bound Modes */
#define LUMINAV_BOUND_NONE 0
#define LUMINAV_BOUND_RADIAL 1
#define LUMINAV_BOUND_COORDINATE 2

/**
 * Creates and initializes a standalone Triton AOT engine.
 * Probes and dynamically loads NVIDIA CUDA (nvcuda) or AMD ROCm HIP (amdhip64) directly.
 * Returns NULL if neither driver is available.
 */
LuminaVTritonEngine* luminav_triton_engine_create(void);

/**
 * Destroys a standalone Triton AOT engine instance and releases driver resources.
 */
void luminav_triton_engine_destroy(LuminaVTritonEngine* engine);

/**
 * Returns the active platform tag: "cuda", "rocm", or "none".
 */
const char* luminav_triton_get_platform(const LuminaVTritonEngine* engine);

/**
 * Returns the active GPU architecture tag (e.g., "sm_80", "sm_90", "sm_100", "gfx942", "gfx1100").
 */
const char* luminav_triton_get_arch(const LuminaVTritonEngine* engine);

/**
 * Executes a single optimization step on FP32 parameter buffer using GPU Triton AOT kernel.
 * Bypasses Python and PyTorch completely.
 *
 * @param engine          Active engine pointer from luminav_triton_engine_create()
 * @param param           Pointer to model parameter buffer (in-out)
 * @param grad            Pointer to parameter gradient buffer (in)
 * @param exp_avg         Pointer to 1st momentum buffer (in-out)
 * @param exp_avg_sq      Pointer to 2nd momentum buffer (in-out, required for Dual buffer, NULL for TD-CLM)
 * @param master_param    Pointer to master FP32 weights (in-out, optional, pass NULL if disabled)
 * @param scratch         6-element scratch buffer: [mask_sum, u_sq_sum, p_sq_sum, c_scale, b_scale, diff_sum]
 * @param num_elements    Number of scalar elements in the parameter tensor
 * @param lr              Learning rate
 * @param beta1           1st momentum decay (e.g. 0.9)
 * @param beta2           2nd momentum decay (e.g. 0.999)
 * @param eps             Epsilon stability term (e.g. 1e-8)
 * @param weight_decay    Weight decay coefficient (e.g. 8e-2)
 * @param tau             CLM temperature parameter (e.g. 0.8)
 * @param lam             TD-CLM defect scaling parameter (e.g. 0.3)
 * @param buffer_mode     1 = TD-CLM, 2 = Dual-Buffer
 * @param cautious        1 = enabled, 0 = disabled
 * @param bound_mode      0 = none, 1 = radial, 2 = coordinate
 * @param bound_ratio     Bound ratio scalar (e.g. 0.05)
 * @param step            Current optimization step (1-indexed)
 * @param seed            PRNG seed for deterministic updates
 * @return                0 on success, negative status code on failure
 */
int luminav_triton_step_f32(
    LuminaVTritonEngine* engine,
    float* param,
    const float* grad,
    float* exp_avg,
    float* exp_avg_sq,
    float* master_param,
    float* scratch,
    size_t num_elements,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float tau,
    float lam,
    int buffer_mode,
    int cautious,
    int bound_mode,
    float bound_ratio,
    uint64_t step,
    uint32_t seed
);

#ifdef __cplusplus
}
#endif

#endif /* LUMINAV_TRITON_STANDALONE_H */
