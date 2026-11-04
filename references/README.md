# Scientific Foundations and Theoretical References

This directory documents the mathematical foundations, peer-reviewed literature, and algorithmic design choices underpinning the LuminaMoon ecosystem (`luminamoon.nn` and `luminamoon.optimizer`).

LuminaMoon is engineered around **tanh-free rational approximations (`Ra-*`) and scale-invariant numerical mechanics**, designed to achieve high training throughput, memory efficiency, and numerical stability across half-precision (`bfloat16` and `float16`) deep learning workloads.

---

## 1. Architectural Component Mapping

### A. Neural Network Architecture (`luminamoon.nn`)

| LuminaMoon Component | Academic Alias | Theoretical Foundation | Key Literature Reference |
| :--- | :--- | :--- | :--- |
| **`RaNorm`** | `L1Norm`, `L1LayerNorm` | Zero square-root, single-reciprocal mean-free $L_1$-norm token normalization | Hoffer et al. (NeurIPS 2018)<br>Wu et al. (IEEE TNNLS 2019)<br>Zhang & Sennrich (NeurIPS 2019) |
| **`RaBatchNorm2d`** | `L1BatchNorm2d` | Mean Absolute Deviation (MAD) 2D batch normalization with batch-mean centering | Wu et al. (IEEE TNNLS 2019) |
| **`RaTanh`**, `ra_tanh` | `RationalTanh`, `rational_tanh` | Algebraic rational hyperbolic tangent with unit derivative and non-vanishing tail | Boullé et al. (NeurIPS 2020)<br>Molina et al. (ICLR 2020) |
| **`RaGELU`**, `ra_gelu` | `RationalGELU`, `rational_gelu` | Rational approximation of Gaussian Error Linear Units | Boullé et al. (NeurIPS 2020)<br>Tang & Townsend (ICML 2026) |
| **`PosRaGLU`** | `RationalGLU`, `BoundedRationalGLU` | Bounded, strictly positive rational gated linear unit MLP block ($0 < \sigma_{\text{Rat}} < 1$) | Dauphin et al. (ICML 2017)<br>Tang & Townsend (ICML 2026) |
| **`ra_softmax`** | `rational_softmax`, `pade_softmax` | Cayley-Padé rational softmax with compact support, avoiding exponential function evaluations | Prieto et al. (ICLR 2025)<br>Saratchandran et al. (arXiv 2024) |
| **`RaAttention`** | `RationalAttention`, `PadeAttention` | Multi-head attention operating on rational softmax kernel mechanics | Duan et al. (NeurIPS 2024)<br>Prieto et al. (ICLR 2025) |

---

### B. Numerical Optimizer Architecture (`luminamoon.optimizer.LuminaV`)

| Mechanism & Mode | Configuration Parameter | Theoretical Foundation | Key Literature Reference |
| :--- | :--- | :--- | :--- |
| **TD-CLM Mode** | `buffer=1` (or `"tdclm"`) | Single-buffer triangle-defect convex lookahead momentum with triangle-defect penalty $\lambda \lvert d_t \rvert$ where $d_t = g_t - m _ {t-1}$ ($\lambda^* \approx 0.30$) | Maligranda (2008)<br>Zhang et al. (NeurIPS 2019)<br>Zhuang et al. (NeurIPS 2020) |
| **Triangle Family (T-Family)** | `buffer="T1"`, `buffer="T2"` | tanh-free triangle envelopes eliminating $\tanh$ and square roots across optimization and rational update bounds | Maligranda (2008)<br>Chen et al. (Google Brain 2023) |
| **Dual-Buffer CLM** | `buffer=2` (or `"dual"`) | Two-buffer convex lookahead momentum with central innovation variance tracking | Zhang et al. (NeurIPS 2019)<br>Shoham et al. (NeurIPS 2024) |
| **Cautious Updates** | `cautious=True` | Sign-aligned gradient-momentum masking and adaptive alignment scaling | Liang et al. (ICLR 2026) |
| **Fused Single-Pass** | `fused_single_pass=True` | Single-pass GPU kernel fusing cautious scaling and updates via EMA running statistics | Dao et al. (NeurIPS 2022 / FlashAttention IO-awareness) |
| **Bitwise Stochastic Rounding** | `stochastic_rounding=True` | Stateless deterministic 32-bit MurmurHash3 PRNG stochastic rounding for native half-precision states without FP32 master weights | Gupta et al. (ICML 2015)<br>Zamirai et al. (2020 / 2021) |
| **Decoupled Weight Decay** | `weight_decay > 0` | Decoupled parameter shrinkage avoiding gradient scale distortion | Loshchilov & Hutter (ICLR 2019) |
| **Adaptive Trust-Region Bounding** | `bound=True`, `bound_type="coordinate"` | Trust-region parameter update clipping preventing gradient explosion | Brock et al. (ICML 2021 / AGC)<br>You et al. (ICLR 2020 / LAMB) |

---

## 2. Algorithmic Refinements & Deviations from Literature

While LuminaMoon draws its foundational validity from published literature, practical efficiency and numerical safety requirements have driven important refinements:

### 1. Normalization: Mean-Free $L_1$-Norm Layers (`RaNorm` vs. Wu et al. / Hoffer et al. / Zhang & Sennrich)
- **Literature Baseline**: Hoffer et al. (2018) and Wu et al. (2019) applied $L_1$-norm primarily to convolutional Batch Normalization ($L_1\text{-BN}$) with explicit mean subtraction.
- **LuminaMoon Refinement**:
  * In `RaNorm`, sequence and token-level normalization for Transformer backbones is formulated as a **zero square-root, mean-free** $L_1$ pass (analogous to the RMSNorm discovery by Zhang & Sennrich that mean-centering is redundant for stability):

    $\displaystyle \mu_{|x|} = \frac{1}{d} \sum _ {i=1}^d |x_i|$

    $\displaystyle \mathrm{RaNorm}(x) = \frac{x}{\mu_{|x|} + \epsilon} \odot \gamma + \beta$

  * In `RaBatchNorm2d`, cross-batch Mean Absolute Deviation (MAD) is retained with batch-mean centering:

    $\displaystyle \mu_B = \frac{1}{m} \sum _ {i=1}^m x_i, \quad \mathrm{MAD}_B = \frac{1}{m} \sum _ {i=1}^m |x_i - \mu_B|, \quad y = \frac{x - \mu_B}{\mathrm{MAD}_B + \epsilon} \odot \gamma + \beta$

  * Completely eliminates GPU square-root and reciprocal-square-root instructions (`MUFU.RSQ`), amortizing a single scalar reciprocal per token.

### 2. Rational Activations: Pole Singularity Elimination (`RaTanh` & `RaGELU` vs. Boullé et al. / Molina et al.)
- **Literature Baseline**: Padé Activation Units (PAU, Molina et al. 2020) and standard Rational Neural Networks (Boullé et al. 2020) treat numerator and denominator polynomial coefficients as freely trainable parameters. If denominator roots approach zero during optimization ($Q(x) \to 0$), the network encounters pole singularities resulting in `NaN` values.
- **LuminaMoon Refinement**:
  * Implements an algebraic rational hyperbolic form with guaranteed positive denominator ($b(b+|x|) + x^2 > 0, \; \forall x \in \mathbb{R}$):

    $\displaystyle \mathrm{RaTanh}(x; b) = \frac{b (b + |x|) x}{b (b + |x|) + x^2}$

  * In `PosRaGLU`, positive gating is guaranteed via bounded rational sigmoid mapping without activation explosion:

    $\displaystyle \sigma_{\text{Rat}}(z; b) = 0.5 \cdot (1 + \mathrm{RaTanh}(z; b)) \in (0, 1)$

    $\displaystyle \mathrm{PosRaGLU}(\text{gate}, \text{up}) = \text{up} \odot \sigma_{\text{Rat}}(\text{gate}; b)$

  * Because $\sigma_{\text{Rat}}(z)$ is strictly bounded in $(0, 1)$, the hidden representation is bounded linearly by $\|z_{\text{up}}\|_\infty$. This eliminates the quadratic activation outlier explosion observed with SwiGLU ($x \cdot \text{swish}(x) \sim x^2$), stabilizing low-precision FP8 and INT4 quantization.

### 3. Attention & Softmax: Compact Support Rational Kernel (`ra_softmax` vs. Prieto et al. / Saratchandran et al.)
- **Literature Baseline**: Stablemax (Prieto et al., 2025) and Polynomial Attention (Saratchandran et al., 2024) studied rational alternatives to softmax to prevent exponential saturation and enable integer-friendly computation, but without exact compact support.
- **LuminaMoon Refinement**:
  * Implements `ra_softmax` using a Cayley-Padé rational kernel with compact support:

    $\displaystyle M_i = \max_m S_{im}, \quad \Delta_{ij} = M_i - S_{ij} \ge 0, \quad u_{ij} = \Delta_{ij} \cdot \text{scale}$

    $\displaystyle \phi_{\text{Ra}}(\Delta_{ij}) = \left(\frac{\max(0, 1 - u_{ij})}{1 + u_{ij}}\right)^8$

    $\displaystyle \mathrm{RaSoftmax}(S)_{ij} = \frac{\phi_{\text{Ra}}(\Delta_{ij})}{\sum_k \phi_{\text{Ra}}(\Delta_{ik})}$

  * Matches the Maclaurin expansion of $\exp(-\Delta)$ up to second order, while guaranteeing $\phi_{\text{Ra}} \equiv 0$ for $u \ge 1$ ($\Delta \ge 16$). This eliminates background noise accumulation across long context sequences and removes exponential `MUFU.EX2` instructions.

### 4. Optimizer: Single-Pass Cautious Scaling & TD-CLM (`LuminaV` vs. Liang et al. & Zhang et al.)
- **Literature Baseline**:
  * Cautious Optimizers (Liang et al., 2026) require a two-pass GPU reduction to evaluate global mean tensor alignment $\langle u_t, g_t \rangle > 0$, introducing GPU synchronization latency.
  * Lookahead (Zhang et al., 2019) maintains two distinct model weight copies (slow weights and fast weights) updated discretely every $k$ steps.
- **LuminaMoon Refinement**:
  * **Fused Single-Pass Pipeline**: Employs an exponential moving average (EMA) of alignment statistics from the preceding step, allowing cautious update masking and scaling to execute within a single fused kernel dispatch (`fused_single_pass=True`), eliminating GPU synchronization bottlenecks.
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

All formal academic citations for the underlying algorithms (Maligranda, Liang et al., Wu et al., Hoffer et al., Boullé et al., Prieto et al., Zamirai et al., Chen et al., and others) are organized in [`bibliography.bib`](bibliography.bib).
