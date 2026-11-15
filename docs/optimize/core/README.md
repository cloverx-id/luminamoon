# `luminav-core`: Mathematical Foundations & State Invariants

`luminav-core` is the foundational dependency of the LuminaV optimizer workspace. It contains pure mathematical specifications, fixed-size parameter state buffers, precision conversions, and unbiased bitwise stochastic rounding routines without platform-specific SIMD or GPU code.

---

## 1. Mathematical Formulations

For step $t = \max(\mathrm{step}, 1)$, parameter vector $p _ {\text{init}} \in \mathbb{R}^N$, and raw gradient $g _ {\text{raw}} \in \mathbb{R}^N$, `luminav-core` sanitizes inputs and residuals before computing updates (`sanitize_gradient`, `clamp_defect`):

$$
g_t = \begin{cases} \mathrm{clamp}(g _ {\text{raw}}, -10^{18}, 10^{18}) & \text{if } \mathrm{isfinite}(g _ {\text{raw}}) \text{ and } \lvert g _ {\text{raw}} \rvert < 10^{38} \\ 0 & \text{otherwise} \end{cases}
$$

### 1.1 Dual-Buffer Convex Lookahead Momentum (`buffer=2` / CLM)

Dual-buffer mode tracks first-order momentum $m_t$, convex lookahead numerator $\tilde{m}_t$, and second-moment belief variance $v_t$ over the clamped gradient-momentum residual $d_t^{(\text{B2})} = \mathrm{clamp}(g_t - m_t, -10^{18}, 10^{18})$:

$$
m_t = \beta_1 m _ {t-1} + (1 - \beta_1) g_t, \qquad \tilde{m}_t = \beta_1 m_t + (1 - \beta_1) g_t = \beta_1^2 m _ {t-1} + (1 - \beta_1^2) g_t
$$

$$
d_t^{(\text{B2})} = \mathrm{clamp}(g_t - m_t, -10^{18}, 10^{18}), \qquad v_t = \beta_2 v _ {t-1} + (1 - \beta_2) \left(d_t^{(\text{B2})}\right)^2
$$

*(Under FP16 parameter or state dtypes, $v_t$ is capped at $65000.0$ to stay inside the finite FP16 magnitude range.)*

Bias-corrected scaling and rational Padé $[13/6]$ hyperbolic tangent projection (`tanh_pade_f32`):

$$
\hat{\beta} _ {1, t} = 1 - \beta_1^{t+1}, \qquad \hat{\beta} _ {2, t} = \max(1 - \beta_2^t, 10^{-15})
$$

$$
\sigma_t = \left(\frac{\hat{\beta} _ {1, t} \cdot \tau}{\sqrt{\hat{\beta} _ {2, t}}}\right) \sqrt{\max(v_t, 0)} + \epsilon _ {\text{eff}} \cdot \hat{\beta} _ {1, t} \cdot \tau, \qquad u_t^{(\text{B2})} = \tanh _ {\mathrm{Pade}[13/6]}\left(\frac{\tilde{m}_t}{\max(\sigma_t, 10^{-12})}\right)
$$

In `tanh_pade_f32`, $x$ is clamped to $[-7.9988117, 7.9988117]$ and evaluated via Horner's method as $\frac{x P(x^2)}{Q(x^2)} = \frac{x \sum _ {k=0}^6 a_k x^{2k}}{\sum _ {k=0}^3 b_k x^{2k}}$ using the following single-precision constants:

| Numerator Coefficient | Value (`f32`) | Denominator Coefficient | Value (`f32`) |
| :--- | :--- | :--- | :--- |
| $a_0$ (`ALPHA_1`) | `4.8935246e-3` | $b_0$ (`BETA_0`) | `4.8935246e-3` |
| $a_1$ (`ALPHA_3`) | `6.3726193e-4` | $b_1$ (`BETA_2`) | `2.2684346e-3` |
| $a_2$ (`ALPHA_5`) | `1.4857224e-5` | $b_2$ (`BETA_4`) | `1.1853471e-4` |
| $a_3$ (`ALPHA_7`) | `5.122297e-8` | $b_3$ (`BETA_6`) | `1.1982584e-6` |
| $a_4$ (`ALPHA_9`) | `-8.604672e-11` | | |
| $a_5$ (`ALPHA_11`) | `2.000188e-13` | | |
| $a_6$ (`ALPHA_13`) | `-2.7607684e-16` | | |

### 1.2 Single-Buffer Triangle-Defect Convex Lookahead Momentum (`buffer=1` / TD-CLM)

Single-buffer mode drops the second-order variance buffer $v_t$ (cutting optimizer state memory by 50%) and bounds the convex lookahead numerator $\tilde{m}_t = w_1 m _ {t-1} + w_2 g_t$ ($w_1 + w_2 = 1$) via the **Triangle Inequality** $\lvert w_1 m _ {t-1} + w_2 g_t \rvert \le w_1 \lvert m _ {t-1} \rvert + w_2 \lvert g_t \rvert$ and the **triangle-defect penalty** $\delta_t = \lambda \lvert d_t^{(\text{B1})} \rvert$ over the gradient-momentum residual $d_t^{(\text{B1})} = \mathrm{clamp}(g_t - m _ {t-1}, -10^{18}, 10^{18})$:

$$
w_1 = \beta_1^2, \qquad w_2 = 1 - \beta_1^2, \qquad \tilde{m}_t = w_1 m _ {t-1} + w_2 g_t
$$

$$
d_t^{(\text{B1})} = \mathrm{clamp}(g_t - m _ {t-1}, -10^{18}, 10^{18}), \qquad \delta_t = \lambda \left\lvert d_t^{(\text{B1})} \right\rvert
$$

$$
\mathcal{E}_t = w_1 \lvert m _ {t-1} \rvert + w_2 \lvert g_t \rvert + \delta_t + \epsilon _ {\text{eff}}, \qquad u_t^{(\text{B1})} = \frac{\tilde{m}_t}{\mathcal{E}_t} \in (-1, 1), \qquad m_t = \beta_1 m _ {t-1} + (1 - \beta_1) g_t
$$

**Symmetric Penalty Calibration (Proposition 1)**: When $\beta_1 \in (\sqrt{3}/2, 1)$ (so $w_2 < 1/4$ and $w_1 > w_2$), choosing $\lambda^*$ balances two transient regimes relative to $\lambda = 0$ (ignoring $\epsilon _ {\text{eff}}$):
- *Isolated spike* ($m _ {t-1} = 0$, $g_t \ne 0$): $\lvert d_t \rvert = \lvert g_t \rvert$, so $u _ {\text{spike}} = \frac{w_2}{w_2 + \lambda} \mathrm{sgn}(g_t)$, yielding magnitude $\lvert u _ {\text{spike}} \rvert = \frac{w_2}{w_2 + \lambda}$.
- *Sign reversal* ($g_t = -m _ {t-1}$, $m _ {t-1} \ne 0$): $\lvert d_t \rvert = 2\lvert m _ {t-1} \rvert$, so $u _ {\text{rev}} = \frac{w_1 - w_2}{1 + 2\lambda} \mathrm{sgn}(m _ {t-1})$, yielding magnitude $\lvert u _ {\text{rev}} \rvert = \frac{w_1 - w_2}{1 + 2\lambda}$.

Equating the two magnitudes $\lvert u _ {\text{spike}} \rvert = \lvert u _ {\text{rev}} \rvert$, i.e., $\frac{w_2}{w_2 + \lambda} = \frac{w_1 - w_2}{1 + 2\lambda}$ (with $w_1 = 1 - w_2$), gives $\lambda^* = \frac{2 w_2^2}{1 - 4 w_2} = \frac{2(1 - \beta_1^2)^2}{4\beta_1^2 - 3}$, which evaluates to $\lambda^* \approx 0.3008$ at $\beta_1 = 0.9$ (hence default `lam=0.3`).

### 1.3 Triangle Family (T-Family, `buffer="T2"` & `buffer="T1"`)

The **Triangle Family (T-Family)** modes use tanh-free rational envelopes in both the step direction and update bounds:

- **Two-Buffer Triangle Family (`buffer="T2"`)**: Uses the lagged residual $d_t = \mathrm{clamp}(g_t - m _ {t-1}, -10^{18}, 10^{18})$ (unlike `buffer=2`, which uses $m_t$) to update $v_t = \beta_2 v _ {t-1} + (1 - \beta_2) d_t^2$ (capped at $65000.0$ under FP16 parameter or state dtypes), with bias-corrected scale $\sigma_t^{(\text{T2})} = \sqrt{\frac{\max(v_t, 0)}{\hat{\beta} _ {2, t}}}$:

$$
d_t = \mathrm{clamp}(g_t - m _ {t-1}, -10^{18}, 10^{18}), \qquad \mathcal{E}_t^{(\text{T2})} = w_1 \lvert m _ {t-1} \rvert + w_2 \lvert g_t \rvert + 0.5\lambda \lvert d_t \rvert + 0.5\lambda \sigma_t^{(\text{T2})} + \epsilon _ {\text{eff}}, \qquad u_t^{(\text{T2})} = \frac{\tilde{m}_t}{\mathcal{E}_t^{(\text{T2})}} \in (-1, 1)
$$

- **Single-Buffer Triangle Family (`buffer="T1"`)**: Omits the variance buffer $v_t$ (50% state memory reduction) and uses the same update quotient as TD-CLM ($u_t^{(\text{T1})} = u_t^{(\text{B1})}$), paired with rational coordinate and radial bounds:

$$
\mathcal{E}_t^{(\text{T1})} = w_1 \lvert m _ {t-1} \rvert + w_2 \lvert g_t \rvert + \lambda \lvert d_t \rvert + \epsilon _ {\text{eff}}, \qquad u_t^{(\text{T1})} = \frac{\tilde{m}_t}{\mathcal{E}_t^{(\text{T1})}} \in (-1, 1)
$$

Both T-Family modes replace $\tanh$ in coordinate and radial update bounds with rational bounds (`apply_coordinate_bound_t2`, `compute_radial_bound_scale_t2`), where $s_i = s _ {t, i}$ is the scaled update coordinate before bounding and $r = r_t$ is the normalized radial ratio:

$$
b_i = \rho \cdot (\lvert p _ {\text{init}, i} \rvert + 10^{-2}), \qquad s _ {\text{bounded}, i} = \frac{b_i (b_i + \lvert s_i \rvert) s_i}{b_i (b_i + \lvert s_i \rvert) + s_i^2}, \qquad \mathrm{scale} _ {\text{radial}} = \frac{1 + r}{1 + r + r^2}
$$

### 1.4 Cautious Masking, Update Bounds, 1-Pass EMA & Final Update

- **Cautious Update Condition**: When `cautious=True`, updates apply only along aligned coordinates ($u _ {\text{eff}, t} = u_t \odot \mathcal{M}_t$); otherwise $u _ {\text{eff}, t} = u_t$ and $\gamma_t = 1.0$:

$$
\mathcal{M}_t = \mathbb{I}(u_t \odot g_t > 0), \qquad \bar{m}_t = \mathrm{clip}\left(\frac{\sum _ i \mathcal{M} _ {t,i}}{\max(N, 1)}, c _ {\text{min}}, 1.0\right), \qquad \gamma _ {\text{target}, t} = \frac{1}{\bar{m}_t}
$$

With `cautious=True`, the applied alignment scalar is $\gamma_t = \gamma _ {\text{target}, t}$ in 2-Pass mode or at $t = 1$, and $\gamma_t = \mathrm{sc}[3] _ {t-1}$ in 1-Pass mode for $t > 1$.

- **Coordinate-Wise Bound (`bound_type="coordinate"`)** for `buffer=1` and `buffer=2` (with $\rho = \mathrm{bound} _ {\text{ratio}}$ and $s_t = \eta \cdot u _ {\text{eff}, t} \cdot \gamma_t$; T-Family uses the rational bound in Section 1.3):

$$
b_i = \rho \cdot \sqrt{p _ {\text{init}, i}^2 + 10^{-4}}, \qquad s _ {\text{bounded}, i} = b_i \cdot \tanh _ {\mathrm{Pade}[13/6]}\left(\frac{s _ {t, i}}{b_i}\right)
$$

- **Radial ($L_2$-Norm) Bound (`bound_type="radial"`)** (T-Family guards $R$ with $\max(R, 10^{-12})$ and computes $\beta _ {\text{target}, t} = \frac{1 + r_t}{1 + r_t + r_t^2}$):

$$
R = \rho \cdot \max(\lVert p _ {\text{init}} \rVert_2, 1.0), \qquad r_t = \frac{\eta \cdot \gamma _ {\text{eff}} \lVert u _ {\text{eff}, t} \rVert_2}{R}, \qquad \beta _ {\text{target}, t} = \begin{cases} 1.0 & \text{if } r_t < 10^{-4} \\ \dfrac{\tanh _ {\mathrm{Pade}[13/6]}(r_t)}{\max(r_t, 10^{-8})} & \text{otherwise} \end{cases}
$$

Here $\gamma _ {\text{eff}} = 1.0$ when `cautious=False`, $\gamma _ {\text{target}, t}$ in 2-Pass mode (or $t = 1$), and the updated $\mathrm{sc}[3]_t$ when computing the post-step radial target in 1-Pass mode ($t > 1$). The applied radial factor scales the step as $s _ {\text{bounded}} = s_t \cdot \beta_t$, where $\beta_t = \beta _ {\text{target}, t}$ in 2-Pass mode (or $t = 1$) and $\beta_t = \mathrm{sc}[4] _ {t-1}$ in 1-Pass mode for $t > 1$.

- **1-Pass EMA Schedule (`fused_single_pass=True`)**: Step $t = 1$ runs a two-pass reduction so $\gamma_1 = \gamma _ {\text{target}, 1}$ and $\beta_1 = \beta _ {\text{target}, 1}$. For $t > 1$, the kernel applies lagged scalars $\mathrm{sc}[3] _ {t-1}$ and $\mathrm{sc}[4] _ {t-1}$ in one pass, then updates the EMA accumulators ($\alpha = 0.95$ for `buffer=1`/`"T1"`, $\alpha = 0.80$ for `buffer=2`/`"T2"`):

$$
\mathrm{sc}[3]_t = \alpha\,\mathrm{sc}[3] _ {t-1} + (1 - \alpha)\,\gamma _ {\text{target}, t}, \qquad \mathrm{sc}[4]_t = \alpha\,\mathrm{sc}[4] _ {t-1} + (1 - \alpha)\,\beta _ {\text{target}, t}
$$

- **FP16 Safety Bound & Decoupled Weight Decay**: When the parameter dtype is `Float16`, $s _ {\text{bounded}}$ passes through a nested safety bound with $b _ {\text{safe}, i} = \max(0.01 \lvert p _ {\text{init}, i} \rvert, 1.5 \times 10^{-4})$ (`apply_fp16_safety_bound` or `apply_fp16_safety_bound_t2`), followed by decoupled weight decay $p_t = p _ {\text{init}} (1 - \eta\,\mathrm{wd}) - s _ {\text{bounded}}$ (or $p _ {\text{init}} - s _ {\text{bounded}}$ when $\mathrm{wd} = 0$).

---

## 2. Unbiased Bitwise Stochastic Rounding

When operating in BF16 or FP16 without FP32 master weights, standard Round-to-Nearest-Even (RNE) causes gradient stagnation under small learning rates $\lvert \Delta w \rvert < 2^{-13} \lvert w \rvert$. `luminav-core` implements bitwise stochastic rounding:

$$
x_{\text{sr}} = \mathcal{Q}\left(x + \mathrm{Uniform}[0, 1) \cdot \Delta_{\text{ULP}}\right)
$$

Generated using a stateless 32-bit MurmurHash3 bit-avalanche PRNG (`_prng`) seeded from the tensor parameter index and training step (`derive_kernel_seeds` and `derive_variance_seed`):

- **Per-Step Seed Derivation (`src/prng.rs`)**:
  - `seed_m = (seed + step * 0x9E3779B9 + param_index * 10007) & 0x7FFFFFFF`
  - `seed_p = (seed_m + 0x55555555) & 0x7FFFFFFF`
  - `seed_v = (seed_m + 0x2A6D365B) & 0x7FFFFFFF`
- **Signed 32-Bit Two's-Complement Finalizer (`_prng(seed, idx)`)**:
  - `z = idx + seed - 1640531527` (`-1640531527 = 0x9E3779B9` in signed `i32`)
  - `z = (z ^ (z >> 16)) * -2048173461` (`-2048173461 = 0x85EB5A6B` in signed `i32`, a variant of `0x85EBCA6B`)
  - `z = (z ^ (z >> 13)) * -1028477387` (`-1028477387 = 0xC2B2AE35` in signed `i32`)
  - `z = z ^ (z >> 16)` (executed with signed 32-bit arithmetic shifts across Rust `i32`, GLSL `int`, Triton `tl.int32`, and NumPy `np.int32`)
- **Target Precision Bitwise Quantization**:
  - **BF16 (`stochastic_round_bf16_bits`)**: Adds the lower 16 random bits `noise = _prng(seed, idx) & 0xFFFF` to the FP32 magnitude bits, clears the truncated 16 bits, clamps magnitudes above `0x7F7F0000`, and restores the sign bit (passing NaNs through as quiet NaNs).
  - **FP16 (`stochastic_round_fp16_bits`)**: Adapts the subnormal shift width from the FP32 biased exponent `exp_f32`: `shift = 13 + clip(113 - exp_f32, 0, 11)`. Values with `exp_f32 < 103` ($\lvert x \rvert < 2^{-24}$) flush magnitude to zero; remaining values add `noise = _prng(seed, idx) & ((1 << shift) - 1)`, truncate low bits, clamp at `0x477FE000` ($65504.0$), restore sign, and encode via `f32_to_f16_bits_rne`.
  - **Bitwise-BF16 State Promotion**: Whenever state containers are `float16` (`state_dtype_mode == Float16`, which covers both `master_weights="none"` and `master_weights="semi"` on `float16` parameters), state quantization maps `state_dtype_mode` to `BFloat16`, storing BF16 bit patterns inside the existing 16-bit state containers to keep an 8-bit exponent dynamic range without extra state memory.

---

## 3. Data Structures (`src/state.rs` & `src/config.rs`)

```rust
pub struct ParamState {
    pub step: u64,
    pub buf_count: BufferMode,
    pub scratch: ScratchBuffer,       // 6-element [mask_sum, u_sq_sum, p_sq_sum, cautious_scale, bound_scale, _reserved]
    pub exp_avg: Vec<f32>,
    pub exp_avg_sq: Option<Vec<f32>>, // None when buf_count is TdClm or T1
    pub master_param: Option<Vec<f32>>,
}
```

- When `buffer == BufferMode::TdClm` or `BufferMode::T1`, `exp_avg_sq` is never allocated, reducing memory allocation pressure to a single vector per parameter (50% state memory reduction).

---

## 4. References

- L. Maligranda, "Some remarks on the triangle inequality for norms," *Banach J. Math. Anal.*, vol. 2, no. 2, pp. 31–41, 2008.
- T. Dozat, "Incorporating Nesterov momentum into Adam," *ICLR Workshop*, pp. 1–4, 2016.
- M. R. Zhang, J. Lucas, G. E. Hinton, and J. Ba, "Lookahead optimizer: $k$ steps forward, 1 step back," *NeurIPS*, vol. 32, pp. 9593–9604, 2019.
- J. Zhuang *et al.*, "AdaBelief optimizer: Adapting stepsizes by the belief in observed gradients," *NeurIPS*, vol. 33, pp. 18795–18806, 2020.
- S. Taniguchi *et al.*, "ADOPT: Modified Adam can converge with any $\beta_2$ with the optimal rate," *NeurIPS*, vol. 37, pp. 72438–72474, 2024.
- W. Jiang *et al.*, "Adaptive variance reduction for stochastic optimization under weaker assumptions," *NeurIPS*, vol. 37, pp. 22047–22080, 2024.
- K. Liang, L. Chen, B. Liu, and Q. Liu, "Cautious optimizers: Improving training with one line of code," *ICLR*, pp. 106538–106563, 2026.
- A. Brock, S. De, S. L. Smith, and K. Simonyan, "High-performance large-scale image recognition without normalization," *ICML*, vol. 139, pp. 1059–1071, 2021.
- A. Molina, P. Schramowski, and K. Kersting, "Padé activation units: End-to-end learning of flexible activation functions in deep networks," *ICLR*, 2020.
- S. Gupta, A. Agrawal, K. Gopalakrishnan, and P. Narayanan, "Deep learning with limited numerical precision," *ICML*, vol. 37, pp. 1737–1746, 2015.
- T. Yu *et al.*, "COLLAGE: Light-weight low-precision strategy for LLM training," *ICML*, vol. 235, pp. 57459–57479, 2024.
- K. Ozkara, T. Yu, and Y. Park, "Stochastic rounding for LLM training: Theory and practice," *AISTATS*, vol. 258, pp. 4402–4410, 2025.
- T. Dao *et al.*, "FlashAttention: Fast and memory-efficient exact attention with IO-awareness," *NeurIPS*, vol. 35, pp. 16344–16359, 2022.
