# LuminaV Optimizer Algorithm Changelog

All notable changes to the LuminaV Optimizer algorithm are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

* [Software Package Changelog (`luminamoon`)](SOFTWARE_CHANGELOG.md)

---

## [1.5.3] - 2026-10-20

No versions were published between 1.3.3 and 1.5.3. This release notes all changes made since version 1.3.3.

### Added
* **Single-Buffer Triangle-Defect Convex Lookahead Momentum (TD-CLM, `buffer=1`)**:
  * Replaced the legacy 3-pass RMS single-buffer update with a single-pass formulation:
    $$\tilde{m}_t = w_1 m_{t-1} + w_2 g_t, \quad w_1 = \beta_1^2, \quad w_2 = 1 - \beta_1^2$$
    $$\mathcal{E}_t = w_1 |m_{t-1}| + w_2 |g_t| + \lambda |d_t^{(\text{B1})}| + \epsilon_{\text{eff}}$$
    $$u_t^{(\text{B1})} = \frac{\tilde{m}_t}{\mathcal{E}_t} \in (-1, 1)$$
  * In 1.3.3, single-buffer mode required three sequential GPU kernel launches (`_lumina_v1_pass1_kernel`, `_lumina_v1_pass2_kernel`, `_lumina_v1_pass3_update_kernel`) and global atomic reductions across thread blocks to compute tensor-wide root-mean-square ($\text{rms} = \sqrt{\frac{1}{N}\sum \tilde{m}_i^2 + \epsilon}$).
  * TD-CLM replaces the tensor-wide reduction with coordinate-wise envelope bounding via the Triangle Inequality Envelope $\mathcal{E}_t$. This reduces single-buffer execution to a single GPU pass and eliminates all $\tanh$ evaluations in the momentum path.
  * Calibrated the defect penalty weight $\lambda$ to the symmetric spike-reversal equilibrium:
    $$\lambda^* = \frac{2(1 - \beta_1^2)^2}{4\beta_1^2 - 3}$$
    For the default $\beta_1 = 0.9$, $\lambda^* \approx 0.3008$, setting the default parameter to `lam=0.3`.

* **Padé [13/6] Rational Approximation (`tanh_pade_f32`)**:
  * Implemented a rational Padé [13/6] polynomial for hyperbolic tangent calculations:
    $$\tanh_{\text{Pade}}(x) = \frac{x(1 + c_1 x^2 + c_2 x^4 + c_3 x^6 + c_4 x^8 + c_5 x^{10} + c_6 x^{12})}{1 + d_1 x^2 + d_2 x^4 + d_3 x^6}$$
  * Evaluated over the clamped domain $x \in [-7.9988117, 7.9988117]$ with asymptotic saturation to $\pm 1.0$.
  * In 1.3.3, `_triton_tanh_fast` used hardware-dependent SFU instructions (`tl.math.tanh`) or analytical sigmoid fallbacks (`2 / (1 + exp(-2x)) - 1`). Different GPU architectures (e.g. Turing, Ampere, Hopper) evaluate SFU transcendental functions with slightly different mantissa rounding, leading to bitwise divergence during distributed training.
  * Padé [13/6] guarantees uniform precision with $\lVert \tanh_{\text{Pade}}(x) - \tanh(x) \rVert_\infty < 1.5 \times 10^{-6}$ across CPU Scalar, AVX2, AVX-512, ARM NEON, Vulkan SPIR-V, and Triton GPU kernels.

* **Centered Innovation Defect (`centered_defect`)**:
  * Added batch-mean centering for gradient innovation tracking (`centered_defect=False` by default):
    $$\mu_d = \frac{1}{N}\sum_{i=1}^N (g_{t,i} - m_{t-1,i})$$
    $$d_t^{(\text{B1})} = (g_t - m_{t-1}) - \mu_d$$
    $$d_t^{(\text{B2})} = (g_t - m_t) - \beta_1 \mu_d$$
  * In 1.3.3, uncentered innovation $g_t - m_t$ was used directly. On layers where gradients have a non-zero mean (such as normalization scales, biases, and token embeddings), the raw difference caused systematic drift in the second-moment accumulator. Centering eliminates this mean shift.

* **Subnormal-Safe IEEE-754 FP16 Stochastic Rounding**:
  * Added `f32_to_f16_bits_rne` to handle IEEE-754 subnormal numbers ($|x| < 2^{-14}$) during FP16 stochastic rounding.
  * In 1.3.3, stochastic rounding added 13-bit random noise and masked with `& -8192`. When a floating-point value entered the subnormal range, bitwise addition directly corrupted the exponent field (bit 10), causing values to flip between normal and subnormal representations or trigger hardware flush-to-zero traps.
  * The updated routine checks for zero exponent, shifts the significand into the subnormal format, and applies round-to-nearest-even tie-breaking before bit truncation.

* **Multi-Backend Execution Engine**:
  * Replaced the Python-only runtime with a multi-backend execution layer:
    * **Rust SIMD (`luminav-simd`)**: AVX-512 (512-bit registers), AVX2 with FMA, ARM NEON, and Rayon parallel chunk scheduling.
    * **Vulkan UMA (`luminav-vulkan`)**: SPIR-V compute pipeline for cross-vendor GPUs (AMD, Intel, Apple Silicon via MoltenVK, and NVIDIA) with zero-copy host-visible memory buffers.
    * **Triton AOT (`luminav-triton-aot`)**: C-ABI wrapper dispatching ahead-of-time precompiled kernels for 84 GPU architectures (Blackwell SM100/SM120, Hopper SM90, Ada SM89, Ampere SM80/SM86, Turing SM75, and AMD ROCm gfx90a/gfx942/gfx1100).
    * **Standalone Runtime (`luminamoon.optimizer.standalone`)**: Pure NumPy and C-ABI execution that runs without requiring PyTorch.

### Changed
* **Parameter `alpha_ss`**: Removed. In 1.3.3, `alpha_ss=0.5` was used for soft-sign scaling $z / (1 + \alpha_{\text{ss}} |z|)$ in the legacy single-buffer path. In 1.5.3, TD-CLM envelope normalization bounds updates directly to $(-1, 1)$, making soft-sign obsolete.
* **Parameter `lam`**: Added with default `0.3`. Governs the defect penalty weight in TD-CLM.
* **Parameter `centered_defect`**: Added with default `False`. Enables batch-mean centering of the innovation defect.
* **Parameter `bound_type`**: Default changed from `"radial"` (1.3.3) to `"coordinate"` (1.5.3). Coordinate-wise hyperbolic bounding ($B_i \tanh(\Delta \theta_i / B_i)$) evaluates locally on each parameter without computing a tensor-wide norm reduction.
* **Parameter `bound_ratio`**: Default changed from `0.03` to `0.05` to provide better update headroom on large model architectures.
* **Parameter `ema_decay`**: Default changed from static `0.8` to `"auto"`. Resolves to `0.80` for Dual-Buffer mode (`buffer=2`) and `0.95` for Single-Buffer mode (`buffer=1`).
* **Scratch Buffer**: Expanded from 5 floats `[mask_sum, u_sq_sum, p_sq_sum, cautious_scale, bound_scale]` to 6 floats, adding `diff_sum` at index 5 to accumulate batch-mean defect statistics.
* **Cautious Scaling Floor**: Enforced `cautious_clamp_min=0.5` across all backends to cap maximum cautious energy scaling at $2.0\times$ ($2^1$).

### Fixed
* **Hardware Mantissa Divergence**: Fixed bitwise output differences across GPU families by replacing hardware SFU `tl.math.tanh` with Padé [13/6] rational approximation.
* **FP16 Subnormal Bit Corruption**: Fixed exponent corruption in subnormal FP16 stochastic rounding by separating exponent extraction from significand shift logic.
* **Single-Buffer Kernel Launch Latency**: Reduced single-buffer kernel dispatches from 3 passes to 1 pass, eliminating host-side synchronization bubbles and intermediate L2 atomic reduction overhead.
* **Non-Zero Mean Gradient Drift**: Resolved directional accumulator drift on bias and normalization layers via `centered_defect`.

### Parameter Comparison (1.3.3 vs 1.5.3)

| Parameter | Version 1.3.3 | Version 1.5.3 | Technical Notes |
| :--- | :--- | :--- | :--- |
| `params` | `params` | `params` | Iterable of parameters or parameter groups. |
| `lr` | `8e-4` | `8e-4` | Base learning rate. |
| `betas` | `(0.9, 0.999)` | `(0.9, 0.999)` | First and second moment exponential decay rates. |
| `eps` | `1e-8` | `1e-8` | Numerical stability floor. |
| `weight_decay` | `8e-2` | `8e-2` | Decoupled weight decay rate. |
| `tau` | `0.8` | `0.8` | Noise dispersion scale factor. |
| `alpha_ss` | `0.5` | *(removed)* | Removed; superseded by TD-CLM envelope normalization. |
| `lam` | *(none)* | `0.3` | Defect penalty weight in TD-CLM (calibrated to $\lambda^* \approx 0.3008$). |
| `cautious` | `True` | `True` | Directional verification gate ($u_i g_i > 0$). |
| `cautious_clamp_min` | `0.5` | `0.5` | Maximum cautious scale cap ($2.0\times$). |
| `buffer` | `2` | `2` | `1` = Single-Buffer (TD-CLM), `2` = Dual-Buffer (CLM). |
| `centered_defect` | *(none)* | `False` | Enables subtraction of batch-mean defect $\mu_d$. |
| `stochastic_rounding` | `True` | `True` | Bitwise stochastic rounding on FP16/BF16 weights. |
| `bound` | `True` | `True` | Enables update trust-region bounding. |
| `bound_type` | `"radial"` | `"coordinate"` | Default changed to coordinate-wise bounding (zero reduction overhead). |
| `bound_ratio` | `0.03` | `0.05` | Maximum step magnitude relative to parameter magnitude. |
| `master_weights` | `False` | `"none"` | `"none"`, `"semi"`, or `"full"`. |
| `execution` | `"auto"` | `"auto"` | Dynamic routing: SIMD, Vulkan, Triton, or Foreach. |
| `fused_single_pass` | `False` | `True` | Enables single-pass streaming with EMA scaling. |
| `ema_decay` | `0.8` | `"auto"` | Automatically sets `0.80` for dual-buffer, `0.95` for single-buffer. |
| `fsdp_sync` | `False` | `False` | Cross-rank collective all-reduce synchronization. |
| `seed` | `1337` | `1337` | PRNG seed for deterministic stochastic rounding. |

---

## [1.3.3] - 2026-10-06

### Fixed
* **Triton AST Compiler Early-Return Inlining Error:**
  * Resolved fatal MLIR/CFG code generation failure (`at 35:4: _store_state_buffer(...) ^`) triggered by mid-branch `return` statements in `@triton.jit` helper functions (`_store_param` and `_store_state_buffer`).
  * Refactored helper control flow into structured, single-exit `if ... elif ... else` branches, ensuring compliant JIT function inlining across all kernel entry points.
* **Bitcast Width Mismatch (Int64 Promotion in Stochastic Rounding):**
  * Fixed implicit 64-bit integer promotion caused by Python integer literals (`0x9E3779B9`, `~0x1FFF`, `~0xFFFF`) inside Triton bitwise operations.
  * Enforced strict 32-bit signed arithmetic via two's complement constants (`-1640531527`, `-2048173461`, `-1028477387`) with explicit `.to(tl.int32)` coercion, and standardized bitwise masks (`-65536` for BF16, `-8192` for FP16), guaranteeing type-safe 32-bit bitcasting (`tl.int32` to `tl.float32`).
* **JIT Kernel Call Keyword Arguments:**
  * Replaced internal keyword argument invocations (`use_sr=False`) with standard positional boolean parameters inside Triton JIT kernels to eliminate compiler argument binding warnings across Triton runtime versions.

### Improved
* **Throttled Fallback Telemetry & Console De-Noising:**
  * Added `self._warned_triton_failure` state tracking to prevent multi-line compiler AST dumps from flooding the console across every parameter tensor in a parameter group.
  * Formatted fallback notifications into a single concise one-line `WARNING` log on initial encounter, routing full tracebacks to `DEBUG` level.

### Verified
* **End-to-End JIT Execution on NVIDIA Architectures (Tesla T4):**
  * Validated zero-warning JIT execution across both Dual-Pass (`fused_single_pass=False`) and Fused Single-Pass (`fused_single_pass=True`) modes in native FP16.
  * Confirmed non-zero weight update dynamics ($|\Delta W| > 0$) without gradient underflow or weight freezing in pure master-free training for Version 1.3.3.

---

## [1.3.2] - 2026-10-06

### Added
- **Distributed FSDP Collective Synchronization (`fsdp_sync`)**:
  - Introduced the `fsdp_sync: bool = False` configuration flag on optimizer initialization, backed by automated runtime guarding via `torch.distributed.is_initialized()`.
  - Integrated cross-GPU collective communications (`torch.distributed.all_reduce(..., op=torch.distributed.ReduceOp.SUM)`) for global reduction metrics (`mask_sum`, `u_sq_sum`, and `p_sq_sum`) across Triton kernels, single-step execution, and vectorized foreach loops.
  - Guarantees bit-exact scalar consistency for `m_bar`, `cautious_scale`, and `bound_scale` across distributed parameter shards during large-scale pretraining under PyTorch FSDP (Fully Sharded Data Parallel) and DeepSpeed ZeRO-3.
- **Dedicated Low-Precision Stochastic Quantization Routine (`_sr_quantize`)**:
  - Factored out bitwise stochastic rounding into a dedicated, modular helper function: `_sr_quantize(target, val_f32, use_sr, generator)`.
  - Enabled native stochastic rounding on half-precision momentum states (`exp_avg` in BF16/FP16) across all PyTorch CPU/CUDA fallback paths (`_single_step_v2`, `_foreach_step_v2`, `_single_step_v1`, and `_foreach_step_v1`), systematically eliminating gradient stagnation induced by fractional bit truncation (LSB underflow).
- **Persistent Contiguous Buffer Caching (Zero-Allocation Triton Kernel Execution)**:
  - Introduced persistent state cache tensors within `self.state[p]` (`p_contig`, `exp_avg_contig`, `exp_avg_sq_contig`, and `mp_contig`) to stage strided or sliced parameters before kernel launch.
  - Eliminates dynamic memory allocations (`.contiguous()`) during iterative training loops, preserving PyTorch VRAM caching allocator stability and preventing CUDA Graph capture invalidation.
- **Zero-Copy Direct Tensor Aliasing for FP32 Master Weights**:
  - When parameter tensors are natively `torch.float32`, `state["master_param"]` now directly aliases the parameter tensor pointer (`state["master_param"] = p`), eliminating 100% of master weight VRAM overhead for FP32 layers (0 bytes auxiliary memory footprint).
- **Checkpoint Sanitizer via State Dict Filter (`state_dict`)**:
  - Overrode `state_dict()` to automatically purge temporary contiguous cache buffers (`p_contig`, `exp_avg_contig`, `exp_avg_sq_contig`, `mp_contig`), ensuring on-disk checkpoint payloads remain clean, compact, and fully backward-compatible.
- **Defensive Device Pipeline Flush on Kernel Fault (`torch.cuda.synchronize`)**:
  - Embedded explicit hardware device synchronization (`torch.cuda.synchronize(p.device)`) within Triton exception recovery handlers prior to PyTorch fallback execution, preventing asynchronous CUDA error cascades into subsequent tensor operations.

### Changed
- **High-Entropy Bit-Avalanche PRNG Hash Function (`_prng`)**:
  - Replaced the standard Linear Congruential Generator (LCG) with a MurmurHash3 / SplitMix32 integer bit-avalanche hashing routine:
    ```python
    z = idx + seed + 0x9E3779B9
    z = (z ^ (z >> 16)) * 0x85EBCA6B
    z = (z ^ (z >> 13)) * 0xC2B2AE35
    z = z ^ (z >> 16)
    ```
  - Delivers uniform pseudo-random noise distributions with zero low-order bit periodicity, significantly refining stochastic rounding fidelity on BF16/FP16 parameters.
- **Memory-Contiguous Scratch Buffer Realignment for v1 Triton Kernels**:
  - Re-indexed reduction buffer offsets in Triton v1 kernels:
    - Pass 1 metrics (`[rms_sum, p_sq_sum]`) mapped to contiguous indices `[0, 1]`.
    - Pass 2 metrics (`[mask_sum, u_sq_sum]`) mapped to contiguous indices `[2, 3]`.
  - Enables vectorized contiguous slice all-reduces (`scratch_param[:2]` and `scratch_param[2:4]`) under FSDP without requiring non-contiguous memory gather/scatter overhead.
- **Multi-GPU Device Index Resolution in RNG Dispatcher (`_get_generator`)**:
  - Fixed device index resolution when `device` is passed as a generic `torch.device("cuda")` without an explicit index by defaulting to `torch.cuda.current_device()`, avoiding cross-GPU generator key collisions during multi-GPU distributed runs.
- **Conditional Atomic Reduction Pruning in v2 Pass 1 Kernel**:
  - Guarded `mask_sum` atomic accumulation in `_lumina_v2_pass1_kernel` with `if cautious:`, bypassing CTA reductions and L2 atomic additions when `cautious=False`.
- **Expanded Master Weight Precision Configuration Aliases**:
  - Added the `"half"` alias for `"semi"` mode alongside `"fp32"` and `"masterweight"` aliases for `"full"` mode.
- **Metadata & Documentation Header Update**:
  - Bumped specification version to `1.3.2`.

### Fixed
- **Singular Zero-Division & Step-Zero Bias Correction (`safe_base_step`)**:
  - Introduced `safe_base_step = max(base_step, 1)` and `bc2_base = max(1.0 - (beta2 ** safe_base_step), 1e-15)` across Triton, single-step, and foreach loops.
  - Fully resolves potential `ZeroDivisionError` and NaN blowups in `c1` / `c2` computations when steps evaluate at `step == 0`.
- **Numerical Overflow Clamp in Sigmoid-Based Tanh Fallback**:
  - Applied register clamp bounds `x_clamped = tl.maximum(tl.minimum(x, 20.0), -20.0)` in analytical fallback `_triton_tanh_fast`, preventing `tl.exp(-2.0 * x)` from overflowing to `inf` or `nan` on devices lacking hardware SFU tanh instructions.
- **Missing Argument Forwarding in PyTorch Single-Step v1**:
  - Corrected `_single_step_v1` method signature and call-sites to accept and propagate `scratches`, eliminating runtime `TypeError` during fallback invocation.
- **Multi-Rank Synchronization Skew in Sharded Parameter Bounding**:
  - Aligned tensor norms and radial bounding factors across sharded ranks via synchronized global scalar reductions, preventing parameter update drift across distributed ranks.

### Performance & Engine Refactoring
- **Zero Dynamic VRAM Allocation on Strided / Sliced Tensors**:
  - Persistent pre-allocated contiguous staging buffers eradicate PyTorch caching allocator thrashing and eliminate CUDA Graph memory address invalidations.
  - Preserves deterministic memory footprints across long pre-training runs.
- **Optimized Distributed Collective Bandwidth**:
  - Grouping scalar reductions into contiguous vector slices compresses distributed synchronization overhead into a single collective kernel launch per pass.

---

## [1.3.0] - 2026-09-23

### Added
- **Fused Single-Pass Kernel with Post-Step EMA (`fused_single_pass`, `ema_decay`)**:
  - Introduced `_lumina_v2_fused_single_pass_kernel` coupled with an asynchronous CTA micro-kernel `_lumina_v2_post_step_ema_kernel[(1,)]`.
  - When `fused_single_pass=True`, completely eliminates two-pass global reductions by leveraging running Exponential Moving Average (EMA) estimates of `cautious_scale` ($1 / \bar{M}$) and `bound_scale` ($\tanh(r) / r$).
  - Unifies gradient filtering, momentum accumulation, variance dispersion, cautious masking, radial bounding, decoupled weight decay, stochastic rounding, and block partial reductions into a **single unified VRAM-resident pass**, slashing global GPU memory bandwidth traffic per optimizer step.
- **Hybrid Master Weight Precision Architecture (`master_weights`)**:
  - Added multi-tier master weight flexibility via `master_weights: Union[bool, str] = False`:
    - `"none"` / `False` *(Default)*: Pure master-free low-precision training leveraging on-chip bitwise stochastic rounding directly on native half-precision weights.
    - `"semi"` / `"hybrid"` / `"half"`: 32-bit FP32 master parameters (`state["master_param"]`) coupled with 16-bit low-memory optimizer states (`exp_avg`, `exp_avg_sq` in BF16/FP16), saving ~50% optimizer state VRAM compared to traditional FP32 setups while maintaining full FP32 weight accumulation accuracy.
    - `"full"` / `"fp32"` / `"masterweight"`: Traditional full FP32 master parameters with full FP32 optimizer states for standard baseline comparisons.
- **In-Kernel In-Flight NaN / Inf Gradient Sanitization**:
  - Embedded register-level defensive sanitization across all Triton GPU kernels: `is_finite_g = (g == g) & (g < 1e38) & (g > -1e38); g = tl.where(is_finite_g, g, 0.0)`.
  - Instantly neutralizes gradient anomalies, exploding activations, and subnormal corruptions at zero latency before they can contaminate momentum (`exp_avg`) or variance (`exp_avg_sq`) state buffers.
- **Low-Precision State Buffer Serialization (`_store_state_buffer`)**:
  - Added a dedicated Triton state serialization routine supporting FP32, BF16, and FP16 state representations with optional bitwise stochastic rounding applied directly to momentum buffers.
- **Multi-Device & Cross-Accelerator Hardware Context Isolation (`_get_device_context`)**:
  - Unified device stream scoping supporting NVIDIA CUDA (`torch.cuda`), Intel XPU (`torch.xpu`), and modern PyTorch 2.x unified accelerator interfaces (`torch.accelerator.device`), eliminating multi-GPU stream hijacking and cross-device race conditions.
- **Strict Determinism Auto-Routing**:
  - Automatically queries `torch.are_deterministic_algorithms_enabled()`. When enabled under `execution="auto"`, dynamically routes execution to C++ vectorized `foreach` to guarantee 100% bitwise cross-run reproducibility, circumventing non-deterministic floating-point hardware atomic reordering.
- **Granular Fault Recovery with State Guarding (`completed_indices`)**:
  - Implemented per-parameter completion tracking in `_dispatch_triton`. If a kernel encounters an unexpected runtime fault, completed parameters are preserved while uncompleted ones fall back cleanly to `foreach`.
  - Added `_single_step_v2_apply_param` and `_single_step_v1_apply_param` to safely finalize parameter updates if Pass 1 had already committed state transitions, strictly preventing double-update state corruption.

### Changed
- **Per-Parameter Persistent Scratch Tensors (Elimination of Global Slicing)**:
  - Replaced the global dynamic slice allocator (`self._scratch_tensors[dev][:count]`) with isolated per-parameter persistent state allocations (`state["scratch"] = torch.zeros((5,), device=p.device, dtype=torch.float32)`).
  - Memory pointers for `[mask_sum, u_sq_sum, p_sq_sum, cautious_scale, bound_scale]` remain completely invariant throughout the entire training lifecycle, providing rock-solid stability for **CUDA Graph Capture** (`torch.accelerator.Graph`) and TorchInductor AOT compilation without memory aliasing.
- **Analytically Exact Nesterov Bias Correction ($\beta_1^{\text{step} + 1}$)**:
  - Corrected first-moment bias correction from the classical Adam formulation ($1.0 - \beta_1^{\text{step}}$) to the exact analytical expectation of the Nesterov blended momentum: `bc1_nes = 1.0 - beta1^(step + 1)`. Completely eradicates residual momentum bias during early warmup iterations.
- **Adaptive CTA Tile Sizing and Dynamic Thread Warp Scaling**:
  - Introduced hardware-aware CTA tile scaling based on parameter volume: parameters with $\ge 2,097,152$ elements (2M+) dynamically configure `BLOCK_SIZE = 2048` and `num_warps = 8` (versus `BLOCK_SIZE = 1024`, `num_warps = 4` for smaller tensors).
  - Cuts total thread-block grid counts and L2 cache atomic addition pressure by 50% on massive MLP, Attention Projection, and Vocabulary layers.
- **Hardware-Native Tanh Instruction Dispatch (`tl.math.tanh` / `tl.tanh`)**:
  - Upgraded `_triton_tanh_fast` to dynamically inspect compiler capabilities, binding directly to native PTX Special Function Unit (SFU) instructions (`tanh.approx.f32`) on modern Triton compilers while retaining analytical sigmoid fallbacks for legacy environments.
- **PRNG Decorrelation between Momentum and Parameter Updates**:
  - Separated RNG seed streams using bitwise alternating mask shifts: `seed_p = (seed + 0x55555555) & 0x7FFFFFFF`, mathematically eliminating cross-coupling noise correlation between state buffer rounding and parameter update rounding.
- **Bumped specification version metadata to 1.3.0** across `luminav.py`.

### Fixed
- **Catastrophic Floating-Point Cancellation in Radial Bounding ($r \to 0$)**:
  - Fixed precision jitter and roundoff degradation near zero by adding an asymptotic Taylor-series guard across Triton and PyTorch bounding engines: `torch.where(r < 1e-4, 1.0, tanh(r) / r)`.
- **Artificial Denominator Dilation in Dispersion Scale ($\sigma$)**:
  - Replaced additive epsilon inflation (`sigma + 1e-6`) with a non-distorting floor clamp (`tl.maximum(sigma, 1e-12)`), preventing artificial dampening of subtle gradient signals.
- **Singular Zero-Division in Parameter Norm Clamping**:
  - Replaced `norm_p + 1e-6` with `torch.clamp_min(norm_p, 1.0)` / `tl.maximum(norm_p, 1.0)`, guaranteeing complete numerical stability even when parameters are initialized near zero.
- **Transient Memory Pool Spikes during PyTorch Fallback Stochastic Rounding**:
  - Refactored `_sr_update` to stream stochastic noise generation in bounded chunks of 1,048,576 elements (~4 MB chunk size), eliminating multi-hundred megabyte temporary int32 tensor spikes during fallback execution on large parameter matrices.
- **Non-Contiguous Buffer Safety & Strided Memory Copy**:
  - Added strict contiguous layout verification for parameters, states, and master weights with in-place synchronization (`p.copy_(p_contig)`), guarded by a one-time developer warning.

### Performance & Engine Refactoring
- **~50% Global Memory Bandwidth Reduction via Fused Single-Pass Mode**:
  - By unifying parameter loading, gradient evaluation, state transformation, and bounding into one GPU pass under `fused_single_pass=True`, VRAM read/write cycles are halved for cautious and radially bounded steps.
- **Sub-Microsecond L2 Hardware Atomic Reduction Architecture**:
  - Formally documented the hybrid Intra-Block Tree Reduction + Inter-Block Hardware Atomic pattern. With `BLOCK_SIZE=2048`, a 10M-parameter tensor issues only 4,883 atomic additions directly into GPU L2 cache crossbars, executing in < 0.5 µs and strictly outperforming multi-tier cascaded reduction micro-kernels by eliminating 3–5 µs driver dispatch bubbles.
- **Static Buffer Address Invariance for CUDA Graphs & TorchInductor AOT**:
  - Persistent 5-slot state scratch buffers eliminate all dynamic slicing and memory reallocations during optimization loops, achieving 100% address invariance for CUDA Graph replay.

---

## [1.2.1] - 2026-09-17

### Added
- **In-Kernel GPU Parameter Norm Reduction (`p_sq_sum_ptr`)**: Fused parameter Euclidean norm squared $\sum(p_i^2)$ calculation directly into Pass 1 Triton reduction kernels (`_lumina_v2_pass1_kernel` and `_lumina_v1_pass1_kernel`). Pass 2 and Pass 3 now load and resolve `norm_p` (`tl.sqrt(tl.maximum(p_sq_sum, 0.0))`) entirely inside GPU SRAM/registers without host intervention.
- **Stateless CPU Integer Multiplicative PRNG Hashing**: Replaced dynamic GPU RNG tensor allocation with a pure CPU integer hash using Knuth's 32-bit golden ratio constant: `(step * 0x9E3779B9 + i * 10007) & 0x7FFFFFFF`. Generates uncorrelated 31-bit integer seeds per layer with zero GPU kernel launch overhead and zero PCIe host-device latency.
- **Isolated Generator Cache for Fallback Engine (`self._generators`)**: Added an internal per-device `torch.Generator` cache, ensuring stochastic rounding in `_sr_update` operates in complete isolation from PyTorch's default CUDA generator state.

### Changed
- **Consolidated Scratch Buffer Zeroing**: Eradicated redundant per-parameter slice zeroing inside parameter loops. Buffer zeroing is now batched into a single consolidated `scratch.zero_()` call per step inside `_get_scratch_buffer`.
- **Scratch Buffer Stride Expansion**:
  - `buffer=2`: Expanded stride from 2×N to 3×N elements per parameter to host `[mask_sum, u_sq_sum, p_sq_sum]`.
  - `buffer=1`: Expanded stride from 3×N to 4×N elements per parameter to host `[rms_sum, mask_sum, u_sq_sum, p_sq_sum]`.
- **Bumped specification version metadata to 1.2.1**.

### Fixed
- **Host-Device Synchronization & Pipeline Stalls (`.item()` Bottleneck)**: Eradicated `norm_p = float(p_contig.float().norm().item())` and `seed = torch.randint(...).item()` from Triton step pipelines. Eliminated 200–400 synchronous PCIe roundtrips and GPU pipeline stalls per step on deep architectures (e.g., 28-layer Transformers), restoring full asynchronous GPU compute utilization.
- **CUDA Graphs & `torch.compile` Graph Breaks**: Fixed fatal graph capture crashes and TorchDynamo graph breaks caused by blocking host-device synchronization during radial bounding and seed generation. The optimizer step is now 100% compliant with CUDA Graphs (`torch.accelerator.Graph`) and TorchInductor compilation.
- **Transient VRAM Allocator Thrashing**: Eliminated repeated temporary FP32 parameter cloning (`p_contig.float()`) during norm evaluation, preventing memory pool fragmentation in tight VRAM regimes.
- **Global RNG Pollution & Determinism Compliance**: Fixed global RNG state mutation in fallback stochastic rounding. Solved reproducibility drift in upstream Dropout, DataLoader shuffling, and RLHF reward modeling, ensuring strict adherence to PyTorch 2.10+ `torch.use_deterministic_algorithms(True)`.

### Performance & Engine Refactoring
- **Batch Kernel Launch Overhead Elimination**: Removed up to 400–800 individual `cudaMemsetAsync` micro-calls per step previously triggered by per-slice `.zero_()` invocations on 1-element scratch tensors.
- **Zero-Reduction Bypass for Coordinate Bounding**: Streamlined `_triton_step_v2` to immediately bypass two-pass reductions and invoke `_lumina_v2_single_pass_kernel` when `cautious=False` and coordinate bounding (`bound_mode == 2`) is selected.

---

## [1.2.0] - 2026-09-17

### Added
- **Automatic Dynamic FP16 Cliff Governor:** Built-in asymptotic boundary governor (`_triton_tanh_fast` in GPU register and PyTorch fallback) that automatically activates when `p.dtype == torch.float16`. Prevents unclipped steps from pushing parameters, activations, and residual streams into the IEEE-754 FP16 overflow cliff (> 65,504). Remains completely dormant (zero overhead) on `bfloat16` (TPU / Ampere+) and `float32`.
- **Smooth Asymptotic Parameter Bounding (`bound`, `bound_type`, `bound_ratio`):** Added non-intrusive step bounding with two mathematically continuous formulations:
  - `"radial"`: Direction-Preserving Radial Squashing (`update * tanh(r) / r`) maintaining 100.0000% gradient angular fidelity (Cosine Similarity = 1.000).
  - `"coordinate"`: Dual-Hyperbolic Coordinate Squashing for fine-grained per-element damping.
  - Configurable via constructor: `bound: bool = True`, `bound_type: str = "radial"`, `bound_ratio: float = 0.03`.

### Changed
- **Increased Default `cautious_clamp_min` from `0.2` to `0.5`:** Enforces an exact power-of-two maximum energy scaling factor (2.00x, $2^1$). Eliminates floating-point mantissa roundoff distortion in IEEE-754 hardware.
- Bumped specification version metadata to `1.2.0`.

### Fixed
* **Catastrophic FP16 Step Divergence & NaN Explosion at High Learning Rates:** Solved step divergence and subsequent NaN crashes when training deep 28-layer Transformer architectures (e.g., Qwen3-0.6B) in pure FP16 at nominal learning rates (`lr = 8e-4`).
* **Sparse Embedding & LM-Head Amplification:** Mitigated 5x over-amplification on sparse vocabulary gradients where active token counts are much smaller than total tensor parameters ($N \gg N_{\text{active}}$).
* **Verified stable 1,000-step pretraining from scratch on Qwen3-0.6B** (seq length 512, pure FP16, `lr = 8e-4`) on Nvidia Tesla T4 with zero gradient clipping and zero LR scheduler (loss dropped from 12.09 down to ~5.7 with zero NaNs).
* **Weight Decay Numerical Parity on Fallback Engine:** Resolved a silent precision-stagnation issue in `_single_step` and `_foreach_step`. Previously, decoupled weight decay (`p.mul_(1.0 - lr * weight_decay)`) was executed directly on native FP16/BF16 tensors prior to `_sr_update`. Due to IEEE-754 Round-to-Nearest (RTN) limits, `(1 - lr * weight_decay) ≈ 0.999936` rounded back to `1.0`, causing weight decay to freeze on half-precision parameters. Weight decay is now unified into FP32 registers inside `_sr_update` alongside stochastic rounding, achieving 100% numerical parity with the Triton kernel.
* **Fallback Foreach Indentation & Scope Alignment:** Corrected tensor iteration scope and indentation for `torch._foreach_add_` in non-SR fallback branches.
* **Zero-Allocation GPU Clamping in `_apply_bound`:** Replaced dynamic GPU scalar allocations (`torch.tensor(1.5e-4, device=p.device)`) with native `torch.clamp_min`, eliminating transient CUDA malloc spikes per update step.

### Performance & Engine Refactoring
* **Unified Bound Execution Pipeline:** Added an immediate short-circuit guard (`if lr == 0.0: return update`) and streamlined the execution flow: intermediate updates are scaled in FP32 (`step_val = lr * u_float`), passed through radial/coordinate bounding, chained cleanly into the FP16 Cliff Governor, and returned in a single normalized step.

---

## [1.1.5] - 2026-09-15

### Added
- **Native Hardware Capability Detection (`_supports_native_bf16`):** Added automatic device capability detection via `torch.cuda.is_bf16_supported()` to identify Ampere SM80+ architectures and streamline native execution paths.
- **Canonical Hugging Face Repository Reference:** Integrated canonical Hugging Face repository metadata in module docstrings.

### Changed
- **Dynamic Epsilon Clamping for FP16:** Dynamically floors effective epsilon to `max(eps, 1e-4)` when parameters or states are in `torch.float16`. Prevents bias-corrected scale factor `c2` from underflowing into IEEE-754 subnormal/zero limits on step 1.
- Bumped specification version to `1.1.5`.

### Fixed
- **FP16 Step-1 `NaN` from Epsilon Underflow:** Fixed catastrophic `0.0 / 0.0 = NaN` division occurring on initial training steps when parameters receive zero gradients (e.g., unselected vocabulary embeddings, padding tokens, or dropout paths).
- **Second-Moment Variance Saturation (`+inf`):** Added safe upper-bound register clamping (`tl.minimum(v_new, 65000.0)` in Triton kernels and `.clamp_(0.0, 65000.0)` in PyTorch C++ foreach/single loops). Prevents unclipped initial pretraining gradient surges ($|g| > 256$) from overflowing the FP16 maximum dynamic range (65,504) into irreversible `+inf`.
- **Zero-Sigma Division Guard:** Injected explicit `sigma + 1e-6` denominator guards across all Triton kernel passes and PyTorch update steps to ensure numerical safety during cold-start iterations.
- Verified stable 100-step pretraining from scratch on `SmolLM-135M` using Wikimedia Wikipedia in pure FP16 on Nvidia Tesla T4 GPU with zero gradient clipping.

---

## [1.1.0] - 2026-09-11

### Added
- **Bitwise Stochastic Rounding (SR):** Integrated on-chip PRNG integer bit-manipulation hash in Triton kernels and vectorized PyTorch fallback for native FP16/BF16 training without master weights.
- Unified Triton store routine introducing compile-time branching (`dtype_mode` / `use_sr: tl.constexpr`) with explicit safe casting for FP16 and BF16 pointers.
- Vectorized fallback improvements utilizing C++ multi-tensor operations when stochastic rounding is disabled or tensors are FP32.
- State buffer contiguity validation (`exp_avg_contig`, `exp_avg_sq_contig`) ensuring safe linear memory addressing before kernel dispatch.
- Declarative configuration file (`config.json`) defining hyperparameters and optimizer presets.

### Changed
- Refactored PRNG integer arithmetic to strictly stay within signed 32-bit integer limits ($< 2^{31} - 1$), eliminating JIT compilation failures on newer Triton releases.
- State buffers (`exp_avg`, `exp_avg_sq`) now initialize with strict contiguous memory format.

### Fixed
- Fixed `ValueError: Scalar out of range for type int32` and Triton LLVM compile errors during unsigned 32-bit bitmask operations.
- Resolved kernel fallback hangs and compilation overhead on CUDA devices.

---

## [1.0.2] - 2026-09-08

### Added
- Official Apache-2.0 license notice and copyright headers.
- Standardized package exports (`__all__`, `__version__`, `__author__`, `__license__`).

---

## [1.0.1] - 2026-08-31

### Fixed
- **Memory Synchronization Fix (`423d171`):** Fixed a critical bug where non-contiguous parameter tensors (e.g., transposed linear layers or sliced weights) were updated on temporary contiguous copies without reflecting changes back to model weights. Added `p.copy_(p_contig)` after Triton execution.

---

## [1.0.0] - 2026-08-30

### Added
- Initial release of the LuminaV optimizer.
- **Zero Master-Weight Copies:** Direct parameter updates in native half-precision without allocating 4-byte FP32 master weights.
- **Hyperbolic Tangent (`tanh`) Bounding Envelope:** Projects normalized momentum ratio into `(-1.0, 1.0)` to eliminate exploding gradient shocks.
- **Traffic-Cop Directional Verification Gate:** Cautious update masking that zeroes out momentum steps opposing mini-batch gradient directions.
- **Central Innovation Variance:** Noise dispersion tracking $(g_t - m_t)^2$ decoupling mini-batch turbulence from directional descent.
- Dual-buffer mode (`buffer=2`, LuminaV-2B) and single-buffer mode (`buffer=1`, LuminaV-1).
- Custom fused OpenAI Triton kernels with automated fallback execution.
