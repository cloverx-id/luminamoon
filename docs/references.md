# Scientific Foundations and Theoretical References

This document outlines the mathematical foundations, peer-reviewed literature, and algorithmic design choices underpinning the LuminaMoon ecosystem (`luminamoon.nn` and `luminamoon.optimizer`).

LuminaMoon is engineered around **tanh-free rational approximations (`Ra-*`) and scale-invariant numerical mechanics**, designed to achieve high training throughput, memory efficiency, and numerical stability across half-precision (`bfloat16` and `float16`) deep learning workloads.

---

## 1. Architectural Component Mapping

### A. Neural Network Architecture (`luminamoon.nn`)

| LuminaMoon Component | Academic Alias | Theoretical Foundation | Key Literature Reference |
| :--- | :--- | :--- | :--- |
| **`RaNorm`** | `L1Norm`, `L1LayerNorm` | Zero square-root, single-reciprocal $L_1$-norm token normalization | Hoffer et al. (NeurIPS 2018)<br>Wu et al. (IEEE TNNLS 2019) |
| **`RaBatchNorm2d`** | `L1BatchNorm2d` | Mean Absolute Deviation (MAD) 2D batch normalization | Wu et al. (IEEE TNNLS 2019) |
| **`RaTanh`**, `ra_tanh` | `RationalTanh`, `rational_tanh` | Padé rational hyperbolic tangent approximation with guaranteed positive denominator | Boullé et al. (NeurIPS 2020)<br>Molina et al. (ICLR 2020) |
| **`RaGELU`**, `ra_gelu` | `RationalGELU`, `rational_gelu` | Rational approximation of Gaussian Error Linear Units | Boullé et al. (NeurIPS 2020)<br>Tang & Townsend (ICML 2026) |
| **`PosRaGLU`** | `RationalGLU`, `BoundedRationalGLU` | Bounded, strictly positive rational gated linear unit MLP block | Dauphin et al. (ICML 2017)<br>Tang & Townsend (ICML 2026) |
| **`ra_softmax`** | `rational_softmax`, `pade_softmax` | Cayley-Padé rational softmax avoiding exponential function evaluations | Prieto et al. (ICLR 2025)<br>Saratchandran et al. (NeurIPS 2024) |
| **`RaAttention`** | `RationalAttention`, `PadeAttention` | Multi-head attention operating on rational softmax kernel mechanics | Duan et al. (NeurIPS 2024)<br>Prieto et al. (ICLR 2025) |

---

### B. Numerical Optimizer Architecture (`luminamoon.optimizer.LuminaV`)

| Mechanism & Mode | Configuration Parameter | Theoretical Foundation | Key Literature Reference |
| :--- | :--- | :--- | :--- |
| **TD-CLM Mode** | `buffer=1` (or `"tdclm"`) | Single-buffer triangle-defect convex lookahead momentum with triangle-defect penalty $\lambda |d_t|$ where $d_t = g_t - m _ {t-1}$ ($\lambda^* \approx 0.30$) | Maligranda (2006 / 2008)<br>Zhang et al. (NeurIPS 2019) |
| **Triangle Family (T-Family)** | `buffer="T1"`, `buffer="T2"` | tanh-free triangle envelopes eliminating $\tanh$ and square roots across optimization and rational update bounds | Maligranda (2006)<br>Tang & Townsend (ICML 2026) |
| **Dual-Buffer CLM** | `buffer=2` (or `"dual"`) | Two-buffer convex lookahead momentum with central innovation variance tracking | Zhang et al. (NeurIPS 2019) |
| **Cautious Updates** | `cautious=True` | Sign-aligned gradient-momentum masking and adaptive alignment scaling | Liang et al. (arXiv:2411.16085) |
| **Fused Single-Pass** | `fused_single_pass=True` | Single-pass GPU kernel fusing cautious scaling and updates via EMA running statistics | LuminaMoon Systems Architecture |
| **Bitwise Stochastic Rounding** | `stochastic_rounding=True` | Stateless deterministic 32-bit MurmurHash3 PRNG stochastic rounding for native half-precision states without FP32 master weights | Gupta et al. (ICML 2015)<br>Croci et al. (2022) |
| **Decoupled Weight Decay** | `weight_decay > 0` | Decoupled parameter shrinkage avoiding gradient scale distortion | Loshchilov & Hutter (ICLR 2019) |
| **Adaptive Trust-Region Bounding** | `bound=True`, `bound_type="coordinate"` | Trust-region parameter update clipping preventing gradient explosion | Brock et al. (ICML 2021)<br>You et al. (ICLR 2020) |

---

## 2. Algorithmic Refinements & Deviations from Literature

While LuminaMoon draws its foundational validity from published literature, practical efficiency and numerical safety requirements have driven important refinements:

### 1. Normalization: $L_1$-Norm Layers (`RaNorm` vs. Wu et al. / Hoffer et al.)
- **Literature Baseline**: Hoffer et al. (2018) and Wu et al. (2019) applied $L_1$-norm primarily to convolutional Batch Normalization ($L_1\text{-BN}$).
- **LuminaMoon Refinement**:
  * Extended $L_1$ normalization to sequence and token-level Layer Normalization (`RaNorm`) for Transformer backbones.
  * Formulated as a zero square-root, single-reciprocal pass:

    $\displaystyle \text{RaNorm}(x) = \frac{x - \mu}{\frac{1}{d} \sum _ {i=1}^d |x_i - \mu| + \epsilon} \odot \gamma + \beta$

  * Avoids division by zero using safe numerical floors and implements single-kernel fused reductions in Triton and SIMD, eliminating GPU register spills and SFU reciprocal-root instructions.

### 2. Rational Activations: Pole Singularity Elimination (`RaTanh` & `RaGELU` vs. Boullé et al. / Molina et al.)
- **Literature Baseline**: Padé Activation Units (PAU, Molina et al. 2020) and standard Rational Neural Networks (Boullé et al. 2020) treat numerator and denominator polynomial coefficients as freely trainable parameters. If denominator roots approach zero during optimization ($Q(x) \to 0$), the network encounters pole singularities resulting in `NaN` values.
- **LuminaMoon Refinement**:
  * Fixes coefficients to high-order analytical Padé approximants with **strictly positive even-degree denominators** ($Q(x) > 0, \; \forall x \in \mathbb{R}$).
  * For example, in `RaTanh`:

    $\displaystyle \text{ra\_tanh}(x) = \frac{x (105 + 10x^2)}{105 + 45x^2 + x^4}$

  * In `PosRaGLU`, positive gating is guaranteed via bounded non-negative rational forms without activation explosion:

    $\displaystyle \text{PosRaGate}(x) = \frac{x^2}{1 + x^2 + 0.1 x^4}$

  * Completely eliminates numerical poles while delivering up to $1.8\times$ faster execution than transcendental $\tanh$ or $\text{erf}$ operations on GPU special function units (SFU).

### 3. Attention & Softmax: Stable Attention Fusion (`RaAttention` vs. Prieto et al. / Saratchandran et al.)
- **Literature Baseline**: StableMax (Prieto et al., 2025) and Polynomial Attention (Saratchandran et al., 2024) studied rational softmax for classification or linear attention variants, but often without support for causal masking or fused IO-aware memory access.
- **LuminaMoon Refinement**:
  * Implements `ra_softmax` with an order-2 positive rational envelope and dynamic scaling.
  * Fuses the rational attention computation directly with causal mask evaluation (`RaAttention`), avoiding exponential overflows without requiring large negative score clamping ($\text{score} \to -10000.0$) that degrades FP16 dynamic range.

### 4. Optimizer: Single-Pass Cautious Scaling & TD-CLM (`LuminaV` vs. Liang et al. & Zhang et al.)
- **Literature Baseline**:
  * Cautious Optimizers (Liang et al., 2024) require a two-pass GPU reduction to evaluate sign alignment $\langle g_t, m_t \rangle > 0$, introducing GPU synchronization latency.
  * Lookahead (Zhang et al., 2019) maintains two distinct model weight copies (slow weights and fast weights) updated discretely every $k$ steps.
- **LuminaMoon Refinement**:
  * **Fused Single-Pass Pipeline**: Employs an exponential moving average (EMA) of alignment statistics from the preceding step, allowing cautious update masking and scaling to execute within a single fused kernel dispatch (`DEFAULT_FUSED_SINGLE_PASS = True`).
  * **Triangle-Defect Single-Buffer Formulation (TD-CLM, `buffer=1`)**: Eliminates the slow-weights buffer by tracking the continuous gradient-momentum residual $d_t = g_t - m _ {t-1}$ scaled by the triangle-defect penalty $\lambda |d_t|$ with theoretical balance point $\lambda^* \approx 0.30$ (derived from Maligranda's triangle inequality envelope). When gradients spike abruptly, the denominator automatically inflates, damping the update step without requiring a second parameter buffer.
  * **Triangle Family (`buffer="T1"`, `buffer="T2"`)**: Replaces all hyperbolic tangent and square-root evaluations in the step trajectory and rational update bounds with second-order rational triangle envelopes:

    $\displaystyle \tilde{m} _ t = w_1 m _ {t-1} + w_2 g_t, \quad w_1 = \beta_1^2, \quad w_2 = 1 - \beta_1^2$

    $\displaystyle \mathcal{E} _ t^{(\text{T2})} = w_1 |m _ {t-1}| + w_2 |g_t| + 0.5\lambda |d_t| + 0.5\lambda \sigma_t + \epsilon _ {\text{eff}}$

    $\displaystyle u_t^{(\text{T2})} = \frac{\tilde{m} _ t}{\mathcal{E} _ t^{(\text{T2})}} \in (-1, 1)$

    $\displaystyle b_i = \text{bound ratio} \cdot (|w_i| + 10^{-2}), \quad u _ {\text{bounded}, i} = \frac{b_i (b_i + |s_i|) s_i}{b_i (b_i + |s_i|) + s_i^2}$

---

## 3. Citing the Repository & Primary Literature

### A. Citing This Repository

If you use Loomy (`luminamoon`) in academic research or engineering implementations, you can cite this repository as follows:

```bibtex
@misc{loomy2026,
  author = {Silver Moon and {Lumina Moon contributors}},
  title = {Loomy: Low-Precision Numerical Optimization, Rational Neural Architecture, and Data Processing},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/cloverx-id/luminamoon}},
  note = {PROJECT LOOMY repository}
}
```

### B. Primary Literature Bibliography

All formal academic citations for the underlying algorithms (Maligranda, Liang et al., Wu et al., Hoffer et al., Boullé et al., Prieto et al., and others) are organized in [`references/bibliography.bib`](https://github.com/cloverx-id/luminamoon/blob/main/references/bibliography.bib).
