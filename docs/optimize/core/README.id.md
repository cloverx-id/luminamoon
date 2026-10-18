# `luminav-core`: Fondasi Matematika & Invarian Status

`luminav-core` adalah fondasi dasar dari seluruh crate di workspace LuminaV. Crate ini memuat formulasi matematika murni, buffer status parameter berukuran tetap, konversi presisi numerik, dan algoritma *bitwise stochastic rounding* tanpa ketergantungan pada intrinsik SIMD atau driver GPU spesifik.

[English Version (README.md)](README.md)

---

## 1. Formulasi Matematika

### 1.1 Dual-Buffer Cautious Learning Momentum (Buffer = 2)

Mode *dual-buffer* melacak momen gradien orde pertama $m_t$ dan varians orde kedua $v_t$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$\tilde{m}_t = \beta_1 m_t + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (g_t - m_t)^2$$

Koreksi bias dan penskalaan penyebut:
$$\hat{\beta}_{1, t} = 1 - \beta_1^{t+1}, \quad \hat{\beta}_{2, t} = \max(1 - \beta_2^t, 10^{-15})$$
$$\sigma_t = \left(\frac{\hat{\beta}_{1, t} \cdot \tau}{\sqrt{\hat{\beta}_{2, t}}}\right) \sqrt{v_t} + \epsilon \cdot \hat{\beta}_{1, t} \cdot \tau$$
$$u_t = \tanh\left(\frac{\tilde{m}_t}{\max(\sigma_t, 10^{-12})}\right)$$

### 1.2 Single-Buffer Temporal-Difference CLM (Buffer = 1)

Mode *single-buffer* mengeliminasi buffer momen kedua $v_t$, memangkas 50% alokasi memori status melalui kompensasi selisih gradien temporal (*temporal difference defect*):

$$w_1 = \beta_1^2, \quad w_2 = 1 - w_1$$
$$\tilde{m}_t = w_1 m_{t-1} + w_2 g_t$$
$$\delta_t = \lambda |g_t - m_{t-1}|$$
$$\mathcal{E}_t = w_1 |m_{t-1}| + w_2 |g_t| + \delta_t + \epsilon$$
$$u_t = \frac{\tilde{m}_t}{\mathcal{E}_t}$$
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

### 1.3 Cautious Masking & Kendali Batas Manifold

- **Kondisi Cautious Update**: Pembaruan bobot hanya diterapkan pada dimensi di mana arah gerak selaras dengan gradien instan:
  $$\mathcal{M}_t = \mathbb{I}(u_t \odot g_t > 0)$$
  $$\bar{m}_t = \text{clip}\left(\text{mean}(\mathcal{M}_t), c_{\text{min}}, 1.0\right), \quad s_{\text{cautious}} = \frac{1}{\bar{m}_t}$$
- **Pembatasan Koordinat (Coordinate-wise Bound)**:
  $$B = \text{bound\_ratio} \cdot \sqrt{w^2 + \epsilon}$$
  $$\Delta w = B \cdot \tanh\left(\frac{\eta \cdot u_t \odot \mathcal{M}_t \cdot s_{\text{cautious}}}{B}\right)$$
- **Pembatasan Norm Radial (Radial Manifold Bound)**:
  $$R = \text{bound\_ratio} \cdot \max(\|w\|_2, 1.0), \quad r = \frac{\eta \|\Delta w\|_2}{R}$$
  $$s_{\text{radial}} = \begin{cases} 1.0 & \text{jika } r < 10^{-4} \\ \frac{\tanh(r)}{\max(r, 10^{-8})} & \text{lainnya} \end{cases}$$

---

## 2. Unbiased Bitwise Stochastic Rounding

Dalam pelatihan BF16 atau FP16 tanpa master weights FP32, pembulatan standar Round-to-Nearest-Even (RNE) memicu stagnasi gradien ketika step pembaruan sangat kecil ($|\Delta w| < 2^{-13} |w|$). `luminav-core` mengimplementasikan pembulatan stokastik:

$$x_{\text{sr}} = \mathcal{Q}\left(x + \text{Uniform}[0, 1) \cdot \Delta_{\text{ULP}}\right)$$

Dihasilkan melalui generator PRNG XORShift32 / PCG berbasis bitwise yang di-seed secara deterministik per indeks parameter dan nomor step pelatihan.

---

## 3. Struktur Data Status (`src/state.rs`)

```rust
pub struct ParamState {
    pub exp_avg: Vec<f32>,
    pub exp_avg_sq: Option<Vec<f32>>, // None saat buffer == 1 (TD-CLM)
    pub scratch: [f32; 6],           // [cautious_scale_ema, bound_scale_ema, step, ...]
}
```

Ketika `buffer == BufferMode::TdClm`, alokasi `exp_avg_sq` ditiadakan secara permanen, menghemat ruang memori secara instan.
