# `luminav-core`: Mathematical Foundations & State Invariants

`luminav-core` is the foundational dependency of the LuminaV optimizer workspace. It contains pure mathematical specifications, fixed-size parameter state buffers, precision conversions, and unbiased bitwise stochastic rounding routines without platform-specific SIMD or GPU code.

---

## 1. Mathematical Formulations

### 1.1 Dual-Buffer Convex Lookahead Momentum (`buffer=2` / CLM)

Dual-buffer mode tracks first-order momentum $m_t$, convex lookahead numerator $\tilde{m}_t$, and centered innovation variance $v_t$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$\tilde{m}_t = \beta_1 m_t + (1 - \beta_1) g_t$$
$$d_t^{(\text{B2})} = (g_t - m_t) - \mathbb{I}_{\text{cd}} \beta_1 \mu_d, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) \left(d_t^{(\text{B2})}\right)^2$$

Bias-corrected denominator scaling and rational Padé $[13/6]$ hyperbolic tangent bounding (`tanh_pade_f32` clamped on $[-7.9988117, 7.9988117]$):
$$\hat{\beta}_{1, t} = 1 - \beta_1^{t+1}, \quad \hat{\beta}_{2, t} = \max(1 - \beta_2^t, 10^{-15})$$
$$\sigma_t = \left(\frac{\hat{\beta}_{1, t} \cdot \tau}{\sqrt{\hat{\beta}_{2, t}}}\right) \sqrt{\max(v_t, 0)} + \epsilon_{\text{eff}} \cdot \hat{\beta}_{1, t} \cdot \tau$$
$$u_t^{(\text{B2})} = \tanh_{\text{Padé}[13/6]}\left(\frac{\tilde{m}_t}{\max(\sigma_t, 10^{-12})}\right)$$

### 1.2 Single-Buffer Triangle-Defect Convex Lookahead Momentum (`buffer=1` / TD-CLM)

Single-buffer mode eliminates the second-order buffer $v_t$ (slashing state VRAM by 50%) by bounding the convex lookahead numerator $\tilde{m}_t = w_1 m_{t-1} + w_2 g_t$ ($w_1 + w_2 = 1$) directly via the **Triangle Inequality** ($\lvert w_1 m_{t-1} + w_2 g_t \rvert \le w_1 \lvert m_{t-1} \rvert + w_2 \lvert g_t \rvert$) and the **Triangle Defect** residual $d_t^{(\text{B1})}$:

$$w_1 = \beta_1^2, \quad w_2 = 1 - \beta_1^2$$
$$\tilde{m}_t = w_1 m_{t-1} + w_2 g_t$$
$$d_t^{(\text{B1})} = (g_t - m_{t-1}) - \mathbb{I}_{\text{cd}} \mu_d, \quad \delta_t = \lambda \left|d_t^{(\text{B1})}\right|$$
$$\mathcal{E}_t = w_1 |m_{t-1}| + w_2 |g_t| + \delta_t + \epsilon_{\text{eff}}$$
$$u_t^{(\text{B1})} = \frac{\tilde{m}_t}{\mathcal{E}_t} \in (-1, 1)$$
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

The default triangle defect ratio $\lambda = 0.30$ is calibrated to Proposition 1's analytical symmetric spike–reversal equilibrium $\lambda^* = \frac{2(1 - \beta_1^2)^2}{4\beta_1^2 - 3} \approx 0.3008$ (for $\beta_1 = 0.9$).

### 1.3 Cautious Masking & Boundary Control

- **Cautious Update Condition**: Updates are applied only along dimensions where the candidate step aligns with the gradient:
  $$\mathcal{M}_t = \mathbb{I}(u_t \odot g_t > 0)$$
  $$\bar{m}_t = \text{clip}\left(\text{mean}(\mathcal{M}_t), c_{\text{min}}, 1.0\right), \quad s_{\text{cautious}} = \frac{1}{\bar{m}_t}$$
- **Coordinate-wise Bound**:
  $$B = \text{bound}_{\text{ratio}} \cdot \sqrt{w^2 + \epsilon}$$
  $$\Delta w = B \cdot \tanh\left(\frac{\eta \cdot u_t \odot \mathcal{M}_t \cdot s_{\text{cautious}}}{B}\right)$$
- **Radial Manifold Bound**:
  $$R = \text{bound}_{\text{ratio}} \cdot \max(\|w\|_2, 1.0), \quad r = \frac{\eta \|\Delta w\|_2}{R}$$
  $$s_{\text{radial}} = \begin{cases} 1.0 & \text{if } r < 10^{-4} \\ \frac{\tanh(r)}{\max(r, 10^{-8})} & \text{otherwise} \end{cases}$$

---

## 2. Unbiased Bitwise Stochastic Rounding

When operating in BF16 or FP16 without FP32 master weights, standard Round-to-Nearest-Even (RNE) causes gradient stagnation under small learning rates ($|\Delta w| < 2^{-13} |w|$). `luminav-core` implements bitwise stochastic rounding:

$$x_{\text{sr}} = \mathcal{Q}\left(x + \text{Uniform}[0, 1) \cdot \Delta_{\text{ULP}}\right)$$

Generated using high-throughput XORShift32 / PCG PRNG generators directly seeded from tensor index and training step counter.

---

## 3. Data Structures (`src/state.rs` & `src/config.rs`)

```rust
pub struct ParamState {
    pub exp_avg: Vec<f32>,
    pub exp_avg_sq: Option<Vec<f32>>, // None when buffer == 1 (TD-CLM)
    pub scratch: [f32; 6],           // [cautious_scale_ema, bound_scale_ema, step, ...]
}
```

- When `buffer == BufferMode::TdClm`, `exp_avg_sq` is never allocated, reducing memory allocation pressure to a single vector per parameter.
