/*
 * LuminaMoon: LuminaV Optimizer Suite v2.0.0
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
#define LUMINAV_STATUS_UNSUPPORTED_AOT_CONFIG -5 /* Runtime flags do not match embedded AOT constexprs (FP32, coordinate bound, master_param=NULL, cautious=1 for all modes or cautious=0 for Dual/T2) */

/* Buffer Modes */
#define LUMINAV_BUFFER_TDCLM 1  /* Single-Buffer TD-CLM */
#define LUMINAV_BUFFER_DUAL 2   /* Dual-Buffer CLM */
#define LUMINAV_BUFFER_T1 3     /* Single-Buffer Triangle Family (tanh-free) */
#define LUMINAV_BUFFER_T2 4     /* Dual-Buffer Triangle Family (tanh-free) */

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
 * Returns 1 if the requested FP32 configuration matches an embedded GPU AOT kernel profile, else 0.
 */
int luminav_triton_supports_config(
    const LuminaVTritonEngine* engine,
    int buffer_mode,
    int cautious,
    int bound_mode,
    int has_master
);

/**
 * Executes a single optimization step on host FP32 buffers (uses pooled device staging buffers;
 * automatically falls back to CPU scalar reference if runtime flags are outside embedded AOT profiles).
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

/**
 * Extended host FP32 step with explicit `param_index` and configurable CPU fallback
 * (`allow_cpu_fallback=0` returns LUMINAV_STATUS_UNSUPPORTED_AOT_CONFIG (-5) on AOT profile mismatch).
 */
int luminav_triton_step_f32_ex(
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
    uint32_t seed,
    size_t param_index,
    int allow_cpu_fallback
);

/**
 * Zero-copy GPU-resident FP32 step operating directly on CUDA/ROCm device pointers (zero PCIe transfer).
 * Pass 0 for optional `d_exp_avg_sq` (when buffer_mode is 1/3) and `d_master_param`.
 */
int luminav_triton_step_f32_device(
    LuminaVTritonEngine* engine,
    uint64_t d_param,
    uint64_t d_grad,
    uint64_t d_exp_avg,
    uint64_t d_exp_avg_sq,
    uint64_t d_master_param,
    uint64_t d_scratch,
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
    uint32_t seed,
    size_t param_index
);

#ifdef __cplusplus
}
#endif

#endif /* LUMINAV_TRITON_STANDALONE_H */
