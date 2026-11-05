# Theoretical References

This document outlines the mathematical formulas, peer-reviewed literature, and design choices behind `luminamoon.nn`, `luminamoon.optimizer`, and `luminamoon.data`.

LuminaMoon is built around **tanh-free rational approximations (`Ra-*`), scale-invariant normalization, triangle-inequality optimization envelopes, and SIMD streaming data structures**, designed for stable half-precision (`bfloat16` and `float16`) training and fast data preprocessing.

---

## 1. Component Mapping

### A. Neural Network Components (`luminamoon.nn`)

| Component | Academic Alias | Mathematical Basis | Primary Literature |
| :--- | :--- | :--- | :--- |
| **`RaNorm`** | `L1Norm`, `L1LayerNorm` | Square-root-free, single-reciprocal mean-free $L_1$-norm token normalization without the Gaussian $C _ {L_1} = \sqrt{\pi/2}$ factor | Hoffer et al. (NeurIPS 2018)<br>Wu et al. (IEEE TNNLS 2019)<br>Zhang & Sennrich (NeurIPS 2019)<br>Zhu et al. (CVPR 2025) |
| **`RaBatchNorm2d`** | `L1BatchNorm2d` | Cross-batch Mean Absolute Deviation (MAD) 2D batch normalization with batch-mean centering and `Conv2d` parameter folding | Wu et al. (IEEE TNNLS 2019)<br>Hoffer et al. (NeurIPS 2018) |
| **`RaTanh`**, `ra_tanh` | `RationalTanh`, `rational_tanh` | Fixed-shape / single-anchor $C^2$-smooth algebraic rational hyperbolic tangent with strictly positive denominator $b(b+\lvert x\rvert)+x^2 > 0$ and $\mathcal{O}(1/\lvert x\rvert^3)$ gradient tail | Boullé et al. (NeurIPS 2020)<br>Molina et al. (ICLR 2020)<br>Tang & Townsend (ICML 2026) |
| **`RaGELU`**, `ra_gelu` | `RationalGELU`, `rational_gelu` | Rational Gaussian Error Linear Unit $0.5 x (1 + \mathrm{RaTanh}(\sqrt{2/\pi}\,x))$ avoiding `erf` and $\tanh$ evaluations | Boullé et al. (NeurIPS 2020)<br>Kim et al. (ICML 2021 / I-BERT)<br>Tang & Townsend (ICML 2026) |
| **`PosRaGLU`**, **`RaGLU`**, **`TriGLU`** | `RationalGLU`, `BoundedRationalGLU` | Rational gated linear unit MLP blocks with strictly positive gate $\sigma _ {\text{Rat}}(z; b) \in (0, 1)$, symmetric gate $\mathrm{RaTanh}(z; b)$, and triple-path gating | Dauphin et al. (ICML 2017)<br>Tang & Townsend (ICML 2026) |
| **`ra_softmax`** | `rational_softmax`, `pade_softmax` | Shift-invariant Cayley-Padé rational softmax with exact compact support ($\phi _ {\text{Ra}}(\Delta _ {ij}) \equiv 0$ for $\Delta _ {ij} \ge 16$), preserving probability axioms and bounded Frobenius norm without $\exp$ | Prieto et al. (ICLR 2025)<br>Saratchandran et al. (arXiv 2024/2026)<br>Kim et al. (ICML 2021 / I-BERT) |
| **`RaAttention`** (`RaAttn`) | `RationalAttention`, `PadeAttention` | Multi-head attention using `ra_softmax` compact-support locality, optional `RaNorm` query-key normalization (`qk_norm=True`), and additive attention bias masking | Duan et al. (NeurIPS 2024)<br>Yang et al. (ICML 2024 / GLA)<br>Su et al. (arXiv 2023 / RoFormer)<br>Dao et al. (NeurIPS 2022) |

---

### B. Optimizer Components (`luminamoon.optimizer.LuminaV`)

| Mechanism & Mode | Configuration Parameter | Mathematical Basis | Primary Literature |
| :--- | :--- | :--- | :--- |
| **TD-CLM Mode** | `buffer=1` (or `"tdclm"`) | Single-buffer triangle-defect convex lookahead momentum with gradient-momentum residual $d_t = g_t - m _ {t-1}$ and triangle-defect penalty $\lambda \lvert d_t \rvert$ ($\lambda^* \approx 0.30$) | Maligranda (2008)<br>Zhang et al. (NeurIPS 2019)<br>Zhuang et al. (NeurIPS 2020)<br>Dettmers et al. (ICLR 2022)<br>Zhao et al. (ICML 2024 / GaLore) |
| **Triangle Family (T-Family)** | `buffer="T1"`, `buffer="T2"` | `tanh`-free triangle inequality envelopes removing $\tanh$ and square roots in `T1` (`buffer="T1"`) and removing $\tanh$ across step updates and coordinate/radial bounds in `T2` (`buffer="T2"`) | Maligranda (2008)<br>Chen et al. (NeurIPS 2023 / Lion) |
| **Dual-Buffer CLM** | `buffer=2` (or `"dual"`) | Two-buffer convex lookahead momentum with innovation residual variance tracking $d_t^{(\text{B2})} = g_t - m_t$ and Padé $[13/6]$ $\tanh$ bounding | Zhang et al. (NeurIPS 2019)<br>Zhuang et al. (NeurIPS 2020 / AdaBelief)<br>Taniguchi et al. (NeurIPS 2024 / ADOPT)<br>Molina et al. (ICLR 2020) |
| **Cautious Updates** | `cautious=True` | Sign-aligned gradient-momentum masking and adaptive alignment scaling | Liang et al. (ICLR 2026) |
| **Fused Single-Pass** | `fused_single_pass=True` | Single-pass kernel fusing cautious/radial scaling and parameter updates via lagged EMA running statistics | Dao et al. (NeurIPS 2022 / FlashAttention) |
| **Bitwise Stochastic Rounding** | `stochastic_rounding=True` | Stateless deterministic 32-bit MurmurHash3 PRNG stochastic rounding and Bitwise-BF16 state encoding without FP32 master weights | Gupta et al. (ICML 2015)<br>Zamirai et al. (arXiv 2021) |
| **Decoupled Weight Decay** | `weight_decay > 0` | Decoupled parameter shrinkage avoiding gradient scale distortion | Loshchilov & Hutter (ICLR 2019 / AdamW) |
| **Rational Update Bounds** | `bound=True`, `bound_type="coordinate"` or `"radial"` | Element-wise or radial norm update bounding relative to parameter magnitude | Brock et al. (ICML 2021 / AGC)<br>You et al. (ICLR 2020 / LAMB) |

---

### C. Data Engine Components (`luminamoon.data`)

| Component | Configuration / Class | Mathematical & Systems Basis | Primary Literature |
| :--- | :--- | :--- | :--- |
| **Quality Filtering & SIMD Cleaning** | `FusedPipeline`, `TextCleaner`, `QualityFilter` | Parallel Rayon text normalization (`AVX-512BW` / `AVX2` / `NEON` SIMD) and heuristic Shannon character entropy, repetition, and symbol-ratio filtering | Penedo et al. (NeurIPS 2024 / FineWeb) |
| **Two-Stage & Fuzzy Deduplication** | `StreamDeduplicator`, `MinHashDeduplicator` | Exact 64-bit `xxh3_64` SwissTable deduplication with `.dedup` write-ahead log (WAL) and 64-permutation MinHash LSH Jaccard shingling | Penedo et al. (NeurIPS 2024 / FineWeb) |
| **Ping-Pong Checkpoint Recovery** | `StateTracker`, `StreamCoordinate`, `ResumablePipeline` | 128-byte Dual-Slot Ping-Pong (`Slot A` / `Slot B`) `.meta` header verified by IEEE 802.3 CRC32 for atomic stream resumption | Penedo et al. (NeurIPS 2024 / FineWeb) |

---

## 2. Comparison with Literature

### 1. Normalization: Mean-Free $L_1$-Norm Layers (`RaNorm` & `RaBatchNorm2d` vs. Hoffer et al. / Wu et al. / Zhang & Sennrich / Zhu et al.)
- **Literature Baseline**:
  * Hoffer et al. (NeurIPS 2018, Section 4.1) and Wu et al. (IEEE TNNLS 2019, Section IV) studied $L_1$-norm Batch Normalization and $L_1$ Layer Normalization on ResNets and WMT14 Transformers. Both formulations retained explicit mean subtraction $(x - \mu)$ and multiplied the denominator by the Gaussian half-normal scaling constant $C _ {L_1} = \sqrt{\pi/2} \approx 1.2533$.
  * Zhang & Sennrich (NeurIPS 2019, `RMSNorm`) showed that mean-centering is redundant for $L_2$ LayerNorm stability, but retained the square-root operation $\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}$.
  * Zhu et al. (CVPR 2025, `DyT`) replaced normalization layers with element-wise $\gamma \tanh(\alpha x) + \beta$. While effective in vision models, `DyT` drops token-wise scale invariance ($\mathrm{DyT}(\alpha x) \ne \mathrm{DyT}(x)$), evaluates transcendental $\tanh$ instructions, and requires manual per-model-size and per-block (attention vs. FFN) tuning of the initial scalar $\alpha_0$ in large language models (Zhu et al., 2025, Section 7.2, Table 9).
- **LuminaMoon Design**:
  * `RaNorm` combines the mean-free property of `RMSNorm` with $L_1$ geometry, dropping both mean subtraction $\mu$ and the Gaussian constant $C _ {L_1} = \sqrt{\pi/2}$:

    $\displaystyle \mu _ {|x|} = \frac{1}{d} \sum _ {i=1}^d |x_i|$

    $\displaystyle \mathrm{RaNorm}(x) = \frac{x}{\mu _ {|x|} + \epsilon} \odot \gamma + \beta$

  * Preserves exact positive scale homogeneity ($\mathrm{RaNorm}(c x) = \mathrm{RaNorm}(x)$ for $c > 0$) without square-root (`MUFU.RSQ`) or $\tanh$ instructions, amortizing a single scalar reciprocal per token vector and requiring zero $\alpha_0$ tuning.
  * `RaBatchNorm2d` retains cross-batch Mean Absolute Deviation (MAD) with batch-mean centering and provides `fold_into_conv(conv)` to fold running statistics $(\mu_B, \mathrm{MAD} _ B, \gamma, \beta)$ directly into preceding `Conv2d` weights and biases for zero-overhead inference:

    $\displaystyle \mu_B = \frac{1}{m} \sum _ {i=1}^m x_i, \quad \mathrm{MAD} _ B = \frac{1}{m} \sum _ {i=1}^m |x_i - \mu_B|, \quad y = \frac{x - \mu_B}{\mathrm{MAD} _ B + \epsilon} \odot \gamma + \beta$

### 2. Rational Activations: Smoothness & Normalization Compatibility (`RaTanh`, `RaGELU`, & `PosRaGLU` vs. Boullé et al. / Molina et al. / Tang & Townsend / Kim et al.)
- **Literature Baseline**:
  * Unconstrained trainable rational activations $P(x)/Q(x)$ (type $(3,2)$ in Boullé et al., NeurIPS 2020) can develop real poles during training if denominator roots cross the real axis.
  * To exclude real poles, Molina et al. (ICLR 2020) introduced Safe Padé Activation Units (PAU) with an absolute-value polynomial denominator $Q(x) = 1 + \lvert\sum _ {j=1}^{r_Q} b_j x^j\rvert \ge 1$. However, as noted by Boullé et al. (2020, Section 2), $\lvert\sum _ {j=1}^{r_Q} b_j x^j\rvert$ introduces non-smooth $C^0$ kinks (discontinuous first derivatives) at every root of $\sum b_j x^j = 0$ and increases backward-pass cost.
  * Crucially, Tang & Townsend (ICML 2026, Section 4.5, *"Normalization and Adaptive Rational Activations"*) proved that adaptive rational activations with trainable polynomial coefficients $(a_i, b_j)$ conflict with normalization layers (`LayerNorm`, `BatchNorm`, `GroupNorm`): the affine scale and shift $(\gamma, \beta)$ of normalization can be absorbed into $(a_i, b_j)$, creating parameter non-identifiability, poor conditioning, and amplified gradient noise in Transformer blocks.
  * Meanwhile, integer-only polynomial GELU approximations such as `I-BERT` (Kim et al., ICML 2021, Section 3.4) clip inputs to $[-1.769, 1.769]$, resulting in identically zero gradients in the saturation tails.
- **LuminaMoon Design**:
  * `RaTanh` uses a fixed-shape (or single-anchor $b > 0$) algebraic rational formulation:

    $\displaystyle \mathrm{RaTanh}(x; b) = \frac{b (b + |x|) x}{b (b + |x|) + x^2}$

  * Because $\frac{d}{dx}[b(b+|x|)x] = b^2 + 2b|x|$ and $\frac{d}{dx}[b(b+|x|)+x^2] = b\,\mathrm{sgn}(x) + 2x$, `RaTanh` has unit slope $\mathrm{RaTanh}'(0) = 1$ and vanishing second derivative $\mathrm{RaTanh}''(0) = 0$ at the origin ($C^2$-smooth across $x = 0$), a strictly positive single-term denominator $b(b+|x|) + x^2 \ge b^2 > 0$, and algebraic $\mathcal{O}(1/|x|^3)$ non-vanishing gradient tails.
  * Because `RaTanh`, `RaGELU`, and `PosRaGLU` do not train free numerator/denominator polynomial coefficients $(a_i, b_j)$, they avoid the normalization-activation non-identifiability pathology identified by Tang & Townsend (2026) and train stably alongside `RaNorm` and `RaBatchNorm2d`.
  * In `PosRaGLU`, positive gating is bounded strictly within $(0, 1)$:

    $\displaystyle \sigma _ {\text{Rat}}(z; b) = 0.5 \cdot (1 + \mathrm{RaTanh}(z; b)) \in (0, 1)$

    $\displaystyle \mathrm{PosRaGLU}(\text{gate}, \text{up}) = \text{up} \odot \sigma _ {\text{Rat}}(\text{gate}; b)$

  * Because $\sigma _ {\text{Rat}}(z) \in (0, 1)$, the gated hidden state is bounded linearly by $\|z _ {\text{up}}\| _ \infty$, preventing the quadratic outlier growth of SwiGLU ($x \cdot \mathrm{swish}(x) \sim x^2$) in FP8 and INT4 quantization. `luminamoon.nn` also provides symmetric `RaGLU` ($W _ {\text{down}}(W _ {\text{up}}(x) \odot \mathrm{RaTanh}(W _ {\text{gate}}(x); b))$) and three-way `TriGLU` ($W _ {\text{down}}(W _ {\text{val}}(x) \odot \sigma _ {\text{Rat}}(W _ {g1}(x)) \odot (1 + \mathrm{RaTanh}(W _ {g2}(x))))$).

### 3. Attention & Softmax: Shift-Invariant Compact-Support Rational Kernel (`ra_softmax` & `RaAttention` vs. Prieto et al. / Saratchandran et al. / Kim et al. / Duan et al.)
- **Literature Baseline**:
  * **Prieto et al. (ICLR 2025, Section 3.3)** analyzed floating-point Softmax Collapse in the final classification Cross-Entropy loss (`SCE`) on grokking tasks and proposed piecewise `StableMax` ($s(x) = x + 1$ for $x \ge 0$, $s(x) = \frac{1}{1-x}$ for $x < 0$). However, `StableMax` was formulated for output logits rather than self-attention, is not shift-invariant ($\mathrm{StableMax}(x + c) \ne \mathrm{StableMax}(x)$), and has a heavy $\mathcal{O}(1/|x|)$ tail without compact support.
  * **Saratchandran et al. (arXiv:2410.18613v3, 2024/2026, Section 4)** replaced softmax attention with unnormalized monomial polynomials $\phi(x) = \frac{1}{\sqrt{N}}x^3$ that deliberately violate non-negativity, row normalization, and sparsity, proving that bounding the expected Frobenius norm $\|\mathrm{softmax}(A)\| _ F \le \sqrt{N}$ is sufficient to stabilize Transformer gradients.
  * **Kim et al. (ICML 2021, `I-BERT`, Section 3.5)** approximated $\exp(\tilde{x})$ on $(-\ln 2, 0]$ with a quadratic polynomial $0.3585(p+1.353)^2 + 0.344$ and integer bit-shifts `>> z`, which retains dense non-zero tails across all tokens.
  * **Duan et al. (NeurIPS 2024 MATH-AI)** demonstrated via Attention Bias Scaffolding (`ABS`) and Calibration (`ABC`) that suppressing distant irrelevant tokens via localized attention bias masks over standard exponential softmax is essential for length generalization on arithmetic tasks; **Yang et al. (ICML 2024, `GLA`)** used data-dependent forget gates for linear attention.
- **LuminaMoon Design**:
  * `ra_softmax` uses a shift-invariant Cayley-Padé rational kernel with exact compact support:

    $\displaystyle M_i = \max_m S _ {im}, \quad \Delta _ {ij} = M_i - S _ {ij} \ge 0, \quad u _ {ij} = \Delta _ {ij} \cdot \text{scale}$

    $\displaystyle \phi _ {\text{Ra}}(\Delta _ {ij}) = \left(\frac{\max(0, 1 - u _ {ij})}{1 + u _ {ij}}\right)^8$

    $\displaystyle \mathrm{RaSoftmax}(S) _ {ij} = \frac{\phi _ {\text{Ra}}(\Delta _ {ij})}{\sum _ k \phi _ {\text{Ra}}(\Delta _ {ik})}$

  * Unlike Saratchandran et al.'s $\frac{1}{\sqrt{N}}x^3$, `ra_softmax` preserves all three probabilistic attention axioms (**strict non-negativity** $\phi _ {\text{Ra}} \ge 0$, **exact row normalization** $\sum_j p _ {ij} = 1$, and **shift invariance** via $\Delta _ {ij} = M_i - S _ {ij}$) while automatically satisfying the Frobenius norm bound $\|\mathrm{RaSoftmax}(S)\| _ F \le \sqrt{N}$ since $\sum_j p _ {ij}^2 \le (\sum_j p _ {ij})^2 = 1$ for each of the $N$ rows.
  * Matches the Maclaurin expansion of $\exp(-\Delta)$ up to second order ($\phi _ {\text{Ra}}(\Delta) = 1 - \Delta + \frac{1}{2}\Delta^2 - \mathcal{O}(\Delta^3)$ for $\text{scale} = 1/16$), while enforcing **exact algebraic compact support** $\phi _ {\text{Ra}}(\Delta _ {ij}) \equiv 0$ for $u _ {ij} \ge 1$ ($\Delta _ {ij} \ge 16$). In `RaAttention`, this provides intrinsic algebraic locality without manual `ABS` scaffolding, alongside optional `RaNorm` query-key normalization (`qk_norm=True`) and additive `attn_mask` support.

### 4. Optimizer: Single-Pass Cautious Scaling, TD-CLM & Triangle Family (`LuminaV` vs. Liang et al. / Zhang et al. / Zhuang et al. / Taniguchi et al. / Dettmers et al. / Zhao et al.)
- **Literature Baseline**:
  * Cautious Optimizers (Liang et al., ICLR 2026) apply coordinate-wise alignment masking $\mathbb{I}(u_t \odot g_t > 0)$ paired with a global mask-mean rescaling reduction $\mathrm{dim}(u_t) / (\mathrm{nnz}(u_t \odot g_t > 0) + \xi)$, requiring a two-pass reduction before applying the step.
  * Lookahead (Zhang et al., NeurIPS 2019) maintains a separate slow-weights buffer updated every $k$ steps, while AdaBelief (Zhuang et al., NeurIPS 2020) and ADOPT (Taniguchi et al., NeurIPS 2024) track a second-moment variance buffer $v_t$ of gradient-momentum residuals $g_t - m_t$ normalized via inverse square roots.
  * Memory-efficient optimizers such as 8-bit Adam (Dettmers et al., ICLR 2022) and GaLore (Zhao et al., ICML 2024) reduce state memory via block-wise dynamic quantization or SVD low-rank projections.
- **LuminaMoon Design**:
  * **Fused Single-Pass Pipeline (`fused_single_pass=True`)**: Uses an exponential moving average (EMA) of cautious and radial alignment scalars from step $t-1$, fusing alignment scaling and parameter updates into a single kernel pass (inspired by IO-aware kernel fusion in Dao et al., NeurIPS 2022).
  * **Triangle-Defect Single-Buffer Mode (TD-CLM, `buffer=1`)**: Replaces the second-moment variance buffer $v_t$ and square-root denominator with the triangle inequality envelope (Maligranda, 2008) over the gradient-momentum residual $d_t = g_t - m _ {t-1}$ and triangle-defect penalty $\lambda |d_t|$ ($\lambda^* \approx 0.30$). This cuts optimizer state memory by 50% (1 state tensor per parameter, or 2 bytes/param in native `bfloat16` with stateless MurmurHash3 stochastic rounding following Gupta et al., 2015 and Zamirai et al., 2021) without block dequantization overhead or SVD computations.
  * **Triangle Family (`buffer="T1"`, `buffer="T2"`)**: Replaces all $\tanh$ evaluations in the step update and rational update bounds with rational triangle envelopes (eliminating both $\tanh$ and square roots in single-buffer `T1`, while retaining $\sigma_t = \sqrt{\hat{v}_t}$ in dual-buffer `T2` and removing the square root from the coordinate bound $b_i$):

    $\displaystyle \tilde{m} _ t = w_1 m _ {t-1} + w_2 g_t, \quad w_1 = \beta_1^2, \quad w_2 = 1 - \beta_1^2$

    $\displaystyle \mathcal{E} _ t^{(\text{T2})} = w_1 |m _ {t-1}| + w_2 |g_t| + 0.5\lambda |d_t| + 0.5\lambda \sigma_t + \epsilon _ {\text{eff}}$

    $\displaystyle u_t^{(\text{T2})} = \frac{\tilde{m} _ t}{\mathcal{E} _ t^{(\text{T2})}} \in (-1, 1)$

    $\displaystyle b_i = \text{bound ratio} \cdot (|w_i| + 10^{-2}), \quad u _ {\text{bounded}, i} = \frac{b_i (b_i + |s_i|) s_i}{b_i (b_i + |s_i|) + s_i^2}$

### 5. Data Engine: Fused SIMD Filtering, Deduplication & Ping-Pong Checkpoints (`luminamoon.data` vs. Penedo et al.)
- **Literature Baseline**:
  * Penedo et al. (NeurIPS 2024, `FineWeb`) showed that LLM pretraining quality depends heavily on heuristic document filtering (character/line repetition, symbol-to-word ratios) combined with URL/hash and MinHash deduplication, typically executed as multi-stage offline CPU cluster pipelines.
- **LuminaMoon Design**:
  * `FusedPipeline` executes SIMD whitespace/newline normalization (`AVX-512BW` / `AVX2` / `NEON`), Shannon character entropy and symbol/repetition filtering (`QualityFilter`), exact 64-bit `xxh3_64` SwissTable deduplication (`StreamDeduplicator`), and PII masking in a single multi-threaded Rust Rayon pass.
  * `StateTracker` pairs `.dedup` write-ahead log (WAL) persistence with a 128-byte Dual-Slot Ping-Pong (`Slot A` / `Slot B`) `.meta` header verified by IEEE 802.3 CRC32, allowing interrupted streaming runs to resume from the exact byte and row coordinate without corrupting the previous checkpoint slot.

---

## 3. Citing the Repository & Literature

If you use Loomy (`luminamoon`) in research or engineering projects, you can cite this repository as follows (`references/bibliography.bib` contains the full BibTeX database for all 29 referenced papers):

```bibtex
@misc{loomy2026,
  author = {Silver Moon and {Lumina Moon contributors}},
  title = {Loomy: Low-Precision Numerical Optimization, Rational Neural Architecture, and Data Processing Repository},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/cloverx-id/luminamoon}},
  doi = {10.5281/zenodo.23973326},
  note = {PROJECT LOOMY repository}
}
```
