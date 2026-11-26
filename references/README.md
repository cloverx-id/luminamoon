# Theoretical References & Architectural Positioning

This document explains how `luminamoon` (`v2.0.0rc1`, LuminaV Math `v2.0.0`) differs from prior implementations across `luminamoon.nn`, `luminamoon.optimizer`, and `luminamoon.data`, maps each component to its mathematical basis and primary literature, and records current scope and limitations.

---

## 1. Architectural Positioning & Differences from Prior Work

Modern LLM training stacks face four coupled bottlenecks documented in 2024–2026 literature:

1. **Asymmetric GPU hardware scaling**: On NVIDIA Blackwell B200 relative to Hopper H100, FP16/BF16 Tensor Core throughput more than doubles (`2.25 PFLOPS` vs. `1 PFLOPS`), while shared-memory bandwidth and multi-function exponential units (`MUFU` / `MUFU.EX2`) scale far more slowly—causing shared-memory traffic and exponential evaluations to exceed MMA compute by `25–60%` in attention workloads (Zadouri et al., MLSys 2026, Section 1 & 3).
2. **Large-vocabulary loss memory**: Materializing the `[B * N, V]` cross-entropy logit matrix can account for up to `90%` of training memory on large-vocabulary models (Wijmans et al., ICLR 2025).
3. **Half-precision optimizer state overhead and rounding drift**: Standard AdamW stores two FP32 moments alongside FP32 master weights, while naive FP16/BF16 updates stall under round-to-nearest truncation or lose accuracy under quantization or low-rank compression.
4. **Fragmented offline data preparation**: Heuristic filtering, exact/fuzzy deduplication, and non-truncating document packing (Penedo et al., NeurIPS 2024; Ding et al., ICML 2024) are typically split across disconnected batch jobs that lack resumable byte/token crash recovery.

Most prior methods address one layer in isolation while keeping standard exponential or square-root primitives everywhere else. LuminaMoon (`2.0` series, currently `v2.0.0rc1`) instead explores **algebraic, low-precision co-design across layers, optimization, and data streaming**, focusing on fixed rational operators, atomic-free kernels, and bounded memory layouts.

### Summary of Architectural Differences

| Pillar | Prior Approaches | What LuminaMoon `2.0` Implements | Documented Scope & Trade-offs |
| :--- | :--- | :--- | :--- |
| **1. Rational Layers (`Ra-Family`)** | `FlashAttention-4` emulates `exp` on FMA units; `DyT` drops scale invariance and tunes $\alpha_0$; adaptive rationals (`PAU` / Tang & Townsend, ICML 2026) interact with normalization; `Saratchandran et al.` use unnormalized $\frac{1}{\sqrt{N}}x^3$; `AdaSplash` runs 3-iteration Halley-bisection. | Uses **fixed-shape / single-anchor** rational kernels across norm (`RaNorm`, scale-homogeneous, no `sqrt`), FFNs (`RaSwiGLU`/`RaGLU`, no `exp`), and attention (`ra_softmax`, **0-iteration closed-form compact support** $\Delta _ {ij} \ge 16$). Preserves non-negativity, row normalization, and deterministic Frobenius bound without trainable rational polynomial coefficients. | `ra_attention` skips fully zero $V$ tiles without a forward-to-backward block-sparsity bitmask yet. Comparative training benchmarks against external baselines are not published in this release. |
| **2. Triangle-Defect Optimizer (`LuminaV`)** | `8-bit Adam` uses block quantization; `GaLore` uses periodic SVD; `Cautious Optimizers` run a 2-pass reduction; `CWD` is studied standalone on AdamW/Lion/Muon; standard SR uses single-stream PRNGs or FP32 states. | **`TD-CLM` / `T1` / `T2`** cut state memory by 50% algebraically via triangle-defect damping $\lambda \lvert g_t - m _ {t-1} \rvert$ (`T1` drops `tanh` and `sqrt`). Combines lagged **1-pass EMA scaling** ($t > 1$), opt-in **CWD**, **RTV**, `bound_floor`, **3-salt Weyl-Kronecker SR**, **Bitwise-BF16 state in `float16`**, and **FSDP2 / HSDP** coalesced sync. | Lagged EMA introduces a 1-step lag for $t > 1$ (step $t=1$ runs a 2-pass init). Comparative empirical training benchmarks against external optimizers are outside the scope of this documentation. |
| **3. cuBLAS Loss & 3-Tier MoE** | `Cut Cross-Entropy` (`CCE`, ICLR 2025) fuses projection and LSE in custom Triton/CUDA SRAM kernels (`24 GB -> 1 MB`); `MegaBlocks` uses block-sparse grouped GEMMs; `DeepSeekMoE` pairs routed and shared experts. | Uses **Adaptive 400 MiB Chunked LCE** (cuBLAS `torch.mm` into a reusable `<= 400 MiB` chunk buffer + Triton online LSE, 0 `atomic_add`). `RaMoE` uses packed 3D weights, on-device `counts`, and 3-tier grouped GEMM dispatch. | Materializes a bounded `<= 400 MiB` native `BF16`/`FP16` chunk buffer rather than an SRAM-only projection. `torch._grouped_mm` activates only on `sm_90+` (PyTorch `>= 2.7`) with 8-element alignment. |
| **4. Streaming Data Engine** | `FineWeb` runs offline cluster stages; `Best-Fit Packing` (ICML 2024) packs offline corpus batches; `Umbra`/`Velox` 16B `StringView` lives in SQL engines. | Unifies SIMD cleaning, Shannon/trigram filtering, `XXH3-128` dedup, double-hashed MinHash LSH (adapted from Kirsch-Mitzenmacher), PII masking, 16B Arrow `Utf8View` PyCapsule `StringTensor`, and streaming Best-Fit `SequencePacker` behind one `DataPipeline` with **exact-once crash/resume** (`emitted_above`, `is_final`, $O(\Delta)$ WAL, 256B Ping-Pong `LMDL v2`). | Streaming Best-Fit Packing operates over a bounded lookahead window (`pack_window`, default `64`) rather than a global full-corpus sort, trading bin optimality for bounded RAM and online resumability. |

### Scope & Honest Limitations

To keep all documentation grounded in implemented code and formulas:

- **Empirical Benchmark Scope**: Formal comparative training benchmarks, throughput measurements, and large-scale ablation results against external baselines are not yet published in this documentation. Statements here describe the implemented mathematical definitions, algebraic properties, memory layouts, and architectural differences neutrally without claiming empirical superiority over prior methods.
- **Formal vs. Empirical Scope**: The 28 Z3 SMT solver queries in `eval/` verify boundedness, non-division-by-zero, and algebraic invariants of `LuminaV` update formulas over real/floating-point abstractions; they do not constitute a general non-convex convergence proof across arbitrary data distributions.
- **Hardware Coverage**: Pre-compiled Triton AOT binaries currently ship **182 ELF64 kernels** across **13 targets** (`13 × 14`): 8 NVIDIA (`sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`, `sm_103`, `sm_120`) and 5 AMD (`gfx908`, `gfx90a`, `gfx942`, `gfx1100`, `gfx1103`). Newer targets (such as `sm_107`, `sm_110`, `sm_121`, `gfx950`, `gfx1200`/`gfx1201`) fall back to Triton JIT, CubeCL, or CPU SIMD until added in future AOT builds.

---

## 2. Component Mapping

### A. Neural Network Components (`luminamoon.nn`)

| Component | Mathematical Basis | Primary Literature |
| :--- | :--- | :--- |
| **`RaNorm`** | Square-root-free, single-reciprocal mean-free $L_1$-norm token normalization without the Gaussian $C _ {L_1} = \sqrt{\pi/2}$ factor | Hoffer et al. (NeurIPS 2018)<br>Wu et al. (IEEE TNNLS 2019)<br>Zhang & Sennrich (NeurIPS 2019)<br>Zhu et al. (CVPR 2025) |
| **`RaBatchNorm2d`** | Cross-batch Mean Absolute Deviation (MAD) 2D batch normalization with batch-mean centering and `Conv2d` parameter folding | Wu et al. (IEEE TNNLS 2019)<br>Hoffer et al. (NeurIPS 2018) |
| **`RaTanh`**, `ra_tanh` | Fixed-shape / single-anchor $C^2$-smooth algebraic rational hyperbolic tangent with strictly positive denominator $b(b+\lvert x\rvert)+x^2 > 0$ and $\mathcal{O}(1/\lvert x\rvert^3)$ gradient tail | Boullé et al. (NeurIPS 2020)<br>Molina et al. (ICLR 2020)<br>Tang & Townsend (ICML 2026) |
| **`RaGELU`**, `ra_gelu` | Rational Gaussian Error Linear Unit $0.5 x (1 + \mathrm{RaTanh}(\sqrt{2/\pi}\,x))$ avoiding `erf` and $\tanh$ evaluations | Boullé et al. (NeurIPS 2020)<br>Kim et al. (ICML 2021 / I-BERT)<br>Tang & Townsend (ICML 2026) |
| **`RaSwiGLU`**, `ra_swiglu`, **`RaGLU`**, `ra_glu`, `pos_ra_gate` | Rational gated linear unit MLP blocks: `RaSwiGLU` (`(gate * pos_ra_gate(gate; b)) * up`, cancellation-free forward/backward) without per-element `exp()`, and hard-bounded `RaGLU` (`RaTanh(gate; b) * up`) | Dauphin et al. (ICML 2017)<br>Shazeer (arXiv 2020)<br>Tang & Townsend (ICML 2026) |
| **`ra_softmax`** | Shift-invariant Cayley-Padé rational softmax with exact compact support ($\phi _ {\text{Ra}}(\Delta _ {ij}) \equiv 0$ for $\Delta _ {ij} \ge 16$) and closed-form analytical derivative, preserving probability axioms and bounded Frobenius norm without $\exp$ | Prieto et al. (ICLR 2025)<br>Saratchandran et al. (arXiv 2024/2026)<br>Kim et al. (ICML 2021 / I-BERT)<br>Gonçalves et al. (ICML 2025 / AdaSplash) |
| **`RaAttention`**, `ra_attention` | Fused tiled multi-head and grouped-query attention (`num_kv_heads`) using `ra_softmax` compact-support locality with 2-pass SRAM tiling and block skipping, optional `RaNorm` query-key normalization (`qk_norm=True`), and additive or boolean attention masking | Dao et al. (NeurIPS 2022)<br>Zadouri et al. (MLSys 2026 / FlashAttention-4)<br>Ainslie et al. (EMNLP 2023 / GQA)<br>Dehghani et al. (ICML 2023 / QK-Norm)<br>Duan et al. (NeurIPS 2024)<br>Su et al. (Neurocomputing 2024 / RoFormer) |
| **`LinearCrossEntropy`**, `linear_cross_entropy` | Adaptive VRAM-budget chunked linear projection (`400 MiB` native `BF16`/`FP16` `z_buf` via cuBLAS + Triton online LSE, 0 `atomic_add`) avoiding full `[B * N, V]` logit tensor materialization, with FP16 underflow protection and optional `save_logits` | Dao et al. (NeurIPS 2022)<br>Wijmans et al. (ICLR 2025 / CCE comparison) |
| **`RaMoE`**, `permute_tokens_by_expert` | Mixture-of-Experts layer using `ra_softmax` routing, packed 3D weights with 3-tier grouped GEMM (`ra_swiglu`, `ra_glu`), optional shared experts (`num_shared_experts`), optional `router_logit_bound`, zero-sync on-device `counts`, and contiguous token permutation | Dauphin et al. (ICML 2017)<br>Gale et al. (MLSys 2023 / MegaBlocks)<br>Dai et al. (ACL 2024 / DeepSeekMoE)<br>Prieto et al. (ICLR 2025) |

---

### B. Optimizer Components (`luminamoon.optimizer.LuminaV`)

| Mechanism & Mode | Configuration Parameter | Mathematical Basis | Primary Literature |
| :--- | :--- | :--- | :--- |
| **TD-CLM Mode** | `buffer=1` | Single-buffer triangle-defect convex lookahead momentum with gradient-momentum residual $d_t = g_t - m _ {t-1}$ and triangle-defect penalty $\lambda \lvert d_t \rvert$ ($\lambda^* \approx 0.30$) | Maligranda (2008)<br>Dozat (ICLR Workshop 2016 / Nadam)<br>Zhang et al. (NeurIPS 2019)<br>Zhuang et al. (NeurIPS 2020)<br>Jiang et al. (NeurIPS 2024 / Ada-STORM)<br>Dettmers et al. (ICLR 2022)<br>Zhao et al. (ICML 2024 / GaLore) |
| **Triangle Family (T-Family)** | `buffer="T1"`, `buffer="T2"` | `tanh`-free triangle inequality envelopes removing $\tanh$ and square roots in `T1` (`buffer="T1"`) and removing $\tanh$ across step updates and rational update bounds in `T2` (`buffer="T2"`) | Maligranda (2008)<br>Chen et al. (NeurIPS 2023 / Lion) |
| **Dual-Buffer CLM** | `buffer=2` | Two-buffer convex lookahead momentum with innovation residual variance tracking $d_t^{(\text{B2})} = g_t - m_t$ and Padé $[13/6]$ $\tanh$ projection | Dozat (ICLR Workshop 2016 / Nadam)<br>Zhang et al. (NeurIPS 2019)<br>Zhuang et al. (NeurIPS 2020 / AdaBelief)<br>Taniguchi et al. (NeurIPS 2024 / ADOPT, variance normalization comparison)<br>Molina et al. (ICLR 2020) |
| **Cautious Updates** | `cautious=True` | Sign-aligned gradient-momentum masking and adaptive alignment scaling | Liang et al. (ICLR 2026) |
| **Fused Single-Pass** | `fused_single_pass=True` | Single-pass kernel fusing cautious/radial scaling and parameter updates via lagged EMA running statistics | Dao et al. (NeurIPS 2022 / FlashAttention) |
| **Bitwise Stochastic Rounding** | `stochastic_rounding=True` | Stateless 32-bit MurmurHash3 spatial hashing with 3D Weyl-Kronecker step offsets (`WEYL_C_PHI`, `WEYL_C_SQRT2`, `WEYL_C_SQRT3`), unsigned logical shifts, and independent salts (`seed_p`, `seed_m`, `seed_v`), plus Bitwise-BF16 state encoding inside `float16` containers (`master_weights="none"` or `"semi"`) without FP32 state buffers | Gupta et al. (ICML 2015)<br>Zamirai et al. (arXiv 2021)<br>Yu et al. (ICML 2024 / COLLAGE)<br>Ozkara et al. (AISTATS 2025) |
| **Decoupled & Cautious Weight Decay** | `weight_decay > 0`, `cautious_weight_decay` | Decoupled parameter shrinkage, optionally gated to coordinates where `step_val * param >= 0` (CWD) | Loshchilov & Hutter (ICLR 2019 / AdamW)<br>Chen et al. (ICLR 2026 / CWD) |
| **Rational-Tempered Variance (RTV)** | `variance_type="rtv"`, `variance_temper=3.0` | Outlier-damped 2-buffer innovation variance update relative to $\max(\sigma _ {\text{prev}}, \lvert m _ {\text{prev}}\rvert)$ | Zhuang et al. (NeurIPS 2020 / AdaBelief) |
| **Rational Update Bounds** | `bound=True`, `bound_type="coordinate"` or `"radial"`, `bound_floor=1e-2` | Element-wise (with configurable floor $f$) or radial norm update bounding relative to parameter magnitude | Brock et al. (ICML 2021 / AGC)<br>You et al. (ICLR 2020 / LAMB) |
| **Distributed FSDP1 / FSDP2 / HSDP Sync** | `fsdp_sync=True` | Coalesced shard-dimension `all_reduce` over `DTensor` / `DeviceMesh` keeping DCP-safe `DTensor` optimizer states | Zhao et al. (PVLDB 2023 / PyTorch FSDP)<br>Liang et al. (ICLR 2025 / TorchTitan) |

---

### C. Data Engine Components (`luminamoon.data`)

| Component | Configuration / Class | Mathematical & Systems Basis | Primary Literature |
| :--- | :--- | :--- | :--- |
| **Streaming Pipeline & Document Packing** | `DataPipeline`, `SequencePacker`, `TokenBatch` | Multi-source mixing, rank sharding, document-isolated streaming Best-Fit / Concat / Pad packing, and exact-once resumable checkpointing | Ding et al. (ICML 2024 / Best-Fit Packing)<br>Penedo et al. (NeurIPS 2024 / FineWeb) |
| **Quality Filtering & SIMD Cleaning** | `FusedPipeline`, `TextCleaner`, `QualityFilter` | Parallel Rayon text normalization (`AVX-512BW` / `AVX2` / `NEON` SIMD) and heuristic Shannon character entropy, trigram repetition, and symbol-ratio filtering | Penedo et al. (NeurIPS 2024 / FineWeb) |
| **Two-Stage & Fuzzy Deduplication** | `StreamDeduplicator`, `MinHashDeduplicator` | Exact 128-bit `XXH3-128` SwissTable deduplication with $O(\Delta)$ `LMWL v2` WAL (`LMWP v2` in Python fallback) and double-hashed MinHash LSH (heuristic adaptation of Kirsch-Mitzenmacher) | Kirsch & Mitzenmacher (RSA 2008)<br>Penedo et al. (NeurIPS 2024 / FineWeb) |
| **Columnar `StringTensor` & Checkpoints** | `StringTensor`, `StateTracker`, `StreamCoordinate`, `ResumablePipeline` | 16-byte Arrow `Utf8View`-aligned `StringView16` (12B inline / 4B prefix) + PyCapsule export, and 256-byte `LMDL v2` Dual-Slot Ping-Pong (`Slot A` / `Slot B`) CRC32 checkpoints | Neumann & Freitag (CIDR 2020 / Umbra)<br>Pedreira et al. (PVLDB 2022 / Velox)<br>Penedo et al. (NeurIPS 2024 / FineWeb) |

---

## 3. Detailed Comparison with Literature

### 3.1 Normalization: Mean-Free L1-Norm Layers

*Components: `RaNorm`, `RaBatchNorm2d` · Compared works: Hoffer et al. (2018), Wu et al. (2019), Zhang & Sennrich (2019), Zhu et al. (2025).*

- **Literature Baseline**:
    - **Hoffer et al. (NeurIPS 2018, Sec. 4.1)** studied $L_1$-norm Batch Normalization on ResNets and $L_1$ Layer Normalization on WMT14 Transformers, retaining explicit mean subtraction $(x - \mu)$ and the Gaussian half-normal factor $C _ {L_1} = \sqrt{\pi/2} \approx 1.2533$; **Wu et al. (IEEE TNNLS 2019, Sec. IV–V)** studied $L_1$-norm Batch Normalization on CNNs and GANs with explicit mean subtraction, derived $\sqrt{\pi/2}$ theoretically, and explicitly omitted it whenever a learnable scale $\gamma$ is enabled (Alg. 1).
    - **Zhang & Sennrich (NeurIPS 2019, `RMSNorm`)** showed mean-centering is redundant for $L_2$ LayerNorm stability, while keeping the square-root operation $\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}$.
    - **Zhu et al. (CVPR 2025, `DyT`)** replaced normalization layers with $\gamma \tanh(\alpha x) + \beta$. `DyT` drops token-wise scale invariance ($\mathrm{DyT}(\alpha x) \ne \mathrm{DyT}(x)$), evaluates transcendental $\tanh$, and requires manual per-model-size and per-block $\alpha_0$ tuning in LLMs (Section 7.2, Table 9).
- **LuminaMoon Design**:
    - `RaNorm` combines the mean-free property of `RMSNorm` with $L_1$ geometry, dropping both mean subtraction $\mu$ and the Gaussian constant $C _ {L_1} = \sqrt{\pi/2}$:

      $\displaystyle \mu _ {|x|} = \frac{1}{d} \sum _ {i=1}^d |x_i|$

      $\displaystyle \mathrm{RaNorm}(x) = \frac{x}{\mu _ {|x|} + \epsilon} \odot \gamma + \beta$

    - Preserves exact positive scale homogeneity ($\mathrm{RaNorm}(c x) = \mathrm{RaNorm}(x)$ for $c > 0$) without `MUFU.RSQ` or $\tanh$ instructions, amortizing a single scalar reciprocal per token vector with zero $\alpha_0$ tuning. Its Triton backward uses a 2-stage atomic-free reduction (`_ra_norm_bwd_kernel` -> `_ra_norm_bwd_reduce_kernel`).
    - `RaBatchNorm2d` retains cross-batch Mean Absolute Deviation (MAD) with batch-mean centering and provides `fold_into_conv(conv)` to fold running statistics directly into preceding `Conv2d` weights and biases for inference:

      $\displaystyle \mu_B = \frac{1}{m} \sum _ {i=1}^m x_i, \quad \mathrm{MAD} _ B = \frac{1}{m} \sum _ {i=1}^m |x_i - \mu_B|, \quad y = \frac{x - \mu_B}{\mathrm{MAD} _ B + \epsilon} \odot \gamma + \beta$

### 3.2 Rational Activations and Normalization

*Components: `RaTanh`, `RaGELU`, `RaSwiGLU`, `RaGLU` · Compared works: Boullé et al. (2020), Molina et al. (2020), Kim et al. (2021), Tang & Townsend (2026).*

- **Literature Baseline**:
    - **Boullé et al. (NeurIPS 2020)** showed unconstrained trainable rational activations $P(x)/Q(x)$ (type $(3,2)$) can develop real poles if denominator roots cross the real axis.
    - **Molina et al. (ICLR 2020, `PAU`)** excluded real poles via $Q(x) = 1 + \lvert\sum _ {j=1}^{r_Q} b_j x^j\rvert \ge 1$, which introduces non-smooth $C^0$ kinks at every root of $\sum b_j x^j = 0$ and higher backward cost (Boullé et al., 2020, Sec. 2).
    - **Tang & Townsend (ICML 2026, `arXiv:2602.12390`, Sec. 4.5 p. 5 & App. D p. 31)** showed two conditioning pathologies specific to **adaptive** (trainable-coefficient) rational activations coupled with normalization (`BatchNorm`, `LayerNorm`, `GroupNorm`): learned affine scales $(\gamma, \beta)$ can be absorbed into $(a_i, b_j)$ causing non-identifiability, and data-dependent statistics amplify coefficient gradient noise.
    - **Kim et al. (ICML 2021, `I-BERT`, Sec. 3.4, Eq. 8–9 & Alg. 2)** clipped the integer-only `erf` argument $x/\sqrt{2}$ to $[-1.769, 1.769]$ (corresponding to GELU input $[-2.502, 2.502]$), yielding identically zero derivative for `I-ERF` in both tails and zero gradient for `i-GELU` in the negative saturation tail (slope $1$ in the positive tail).
- **LuminaMoon Design**:
    - `RaTanh` uses a fixed-shape (or single-anchor $b > 0$) algebraic rational formulation:

      $\displaystyle \mathrm{RaTanh}(x; b) = \frac{b (b + |x|) x}{b (b + |x|) + x^2}$

    - Because $\mathrm{RaTanh}'(0) = 1$ and $\mathrm{RaTanh}''(0) = 0$, it is $C^2$-smooth across $x = 0$, keeps a strictly positive denominator $b(b+|x|) + x^2 \ge b^2 > 0$, and retains algebraic $\mathcal{O}(1/|x|^3)$ non-vanishing gradient tails.
    - Because `RaTanh`, `RaGELU`, `RaSwiGLU`, and `RaGLU` do not train free numerator/denominator polynomial coefficients $(a_i, b_j)$, they avoid the adaptive-rational normalization non-identifiability described by Tang & Townsend (2026, Sec. 4.5).
    - In `RaSwiGLU`, `pos_ra_gate` replaces $\mathrm{sigmoid}(z)$ with a $(0, 1)$ rational gate evaluated via a cancellation-free identity ($d = b(b + |z|) + z^2$, $q = \frac{0.5 b^2}{d}$, $\sigma = 1 - q$ for $z \ge 0$ and $\sigma = q$ for $z < 0$):

      $\displaystyle \sigma _ {\text{Rat}}(z; b) = 0.5 \cdot \left(1 + \frac{1}{b}\mathrm{RaTanh}(z; b)\right) \in (0, 1)$

      $\displaystyle \mathrm{RaSwiGLU}(\text{gate}, \text{up}) = \left(\text{gate} \odot \sigma _ {\text{Rat}}(\text{gate}; b)\right) \odot \text{up}$

    - For hard-bounded activation ranges under FP8/INT4 quantization, `RaGLU` applies $\mathrm{RaTanh}(\text{gate}; b) \in (-b, b)$ directly so hidden outputs stay bounded linearly by $b \lVert W _ {\text{up}}(x)\rVert _ \infty$.

### 3.3 Compact-Support Rational Softmax and Attention

*Components: `ra_softmax`, `ra_attention`, `RaAttention` · Compared works: Prieto et al. (2025), Saratchandran et al. (2024/2026), Kim et al. (2021), Duan et al. (2024), Gonçalves et al. (2025), Zadouri et al. (2026), Yang et al. (2024).*

- **Literature Baseline**:
    - **Prieto et al. (ICLR 2025, Sec. 3.3)** proposed piecewise `StableMax` for classification cross-entropy (`StCE`), which is not shift-invariant ($\mathrm{StableMax}(x + c) \ne \mathrm{StableMax}(x)$) and has a dense $\mathcal{O}(1/|x|)$ tail without compact support.
    - **Saratchandran et al. (arXiv:2410.18613v3, 2024/2026, Sec. 4, Theorems 4.1, 4.2, 4.4 and Corollaries 4.3, 4.5, p. 4; Remark 4.6 & Sec. 5, p. 5)** proved $\|\mathrm{softmax}(A)\| _ F \le \sqrt{N}$ and $\|\nabla \mathrm{softmax}(A)\| _ F \le 2\sqrt{N}$, and proposed scaled monomials $\phi(x) = \frac{1}{\sqrt{N}}x^p$ (evaluated with $\frac{1}{k}x^3$ in Sec. 5) that control the Frobenius norm only *in expectation* under i.i.d. Gaussian assumptions while dropping non-negativity, row normalization, and sparsity.
    - **Kim et al. (ICML 2021, `I-BERT`, Sec. 3.5)** approximated $\exp(\tilde{x})$ with a quadratic polynomial and integer bit-shifts, keeping dense non-zero tails across all tokens.
    - **Duan et al. (NeurIPS 2024 MATH-AI)** showed via `ABS`/`ABC` that suppressing distant tokens improves arithmetic length generalization over standard softmax.
    - **Gonçalves et al. (ICML 2025, `AdaSplash`, `arXiv:2502.12082v2`, Sec. 3, p. 4)** cut $\alpha$-entmax root-finding from `23` standard bisection iterations to `3` hybrid Halley-bisection iterations (`~15x` faster than Torch bisection), yet still requires iterative on-chip threshold solving ($\tau_i$) and fractional powers $1/(\alpha-1)$.
    - **Zadouri et al. (MLSys 2026, `arXiv:2603.05451`, `FlashAttention-4`, p. 2)** quantified *asymmetric hardware scaling* on NVIDIA Blackwell B200 (`2.25 PFLOPS` vs. H100 `1 PFLOPS` for FP16/BF16), showing shared-memory traffic and exponential evaluations (`MUFU.EX2`) exceed MMA compute by `25–60%` in attention and mitigating it by emulating `2^x` on FMA pipelines.
    - **Yang et al. (ICML 2024, `GLA`)** explored data-dependent forget gates for sub-quadratic linear attention.
- **LuminaMoon Design**:
    - `ra_softmax` uses a shift-invariant Cayley-Padé rational kernel with **0-iteration closed-form compact support** and analytical autograd:

      $\displaystyle M_i = \max_m S _ {im}, \quad \Delta _ {ij} = M_i - S _ {ij} \ge 0, \quad u _ {ij} = \Delta _ {ij} \cdot \text{scale}$

      $\displaystyle \phi _ {\text{Ra}}(\Delta _ {ij}) = \left(\frac{\max(0, 1 - u _ {ij})}{1 + u _ {ij}}\right)^8$

      $\displaystyle \mathrm{RaSoftmax}(S) _ {ij} = \frac{\phi _ {\text{Ra}}(\Delta _ {ij})}{\sum _ k \phi _ {\text{Ra}}(\Delta _ {ik})}, \qquad \frac{\partial \phi _ {\text{Ra}}}{\partial u _ {ij}} = -\frac{16}{1 - u _ {ij}^2} \phi _ {\text{Ra}}(\Delta _ {ij})\,\mathbb{I}(u _ {ij} < 1)$

    - Unlike Saratchandran et al.'s $\frac{1}{\sqrt{N}}x^p$, `ra_softmax` preserves non-negativity ($\phi _ {\text{Ra}} \ge 0$), row normalization ($\sum_j p _ {ij} = 1$), and shift invariance, providing the deterministic Frobenius norm bound $\|\mathrm{RaSoftmax}(S)\| _ F \le \sqrt{N}$ for every finite matrix $S \in \mathbb{R}^{N \times N}$ without distributional assumptions.
    - Matches the Maclaurin expansion of $\exp(-\Delta)$ to second order ($\phi _ {\text{Ra}}(\Delta) = 1 - \Delta + \frac{1}{2}\Delta^2 - \mathcal{O}(\Delta^3)$ for $\text{scale} = 1/16$), while enforcing exact compact support $\phi _ {\text{Ra}}(\Delta _ {ij}) \equiv 0$ for $\Delta _ {ij} \ge 16$ in 0 iterations. `RaAttention` and `ra_attention` use 2-pass Triton SRAM tiling without materializing $[B, H, N, N]$, skipping $V$ tiles when $\min u \ge 1$ and using an atomic-free 2-stage backward (`bwd_dq` -> `bwd_dkdv`).

### 3.4 Chunked Cross-Entropy Loss and Sparse MoE

*Components: `LinearCrossEntropy`, `RaMoE` · Compared works: Wijmans et al. (2025), Gale et al. (2023), Dai et al. (2024).*

- **Literature Baseline**:
    - **Wijmans et al. (ICLR 2025 Oral, `Cut Cross-Entropy`, `arXiv:2411.09009v2`, pp. 1–3)** showed cross-entropy logits consume up to `90%` of training memory on large-vocabulary LLMs (reducing Gemma 2 2B loss memory from `24 GB` to `1 MB` and classifier head memory from `28 GB` to `1 GB`) using custom Triton/CUDA SRAM kernels.
    - **Gale et al. (MLSys 2023, `MegaBlocks`)** formulated sparse MoE as block-sparse grouped matrix multiplications avoiding token dropping.
    - **Dai et al. (ACL 2024, `DeepSeekMoE`)** combined fine-grained routed experts with always-active shared experts.
- **LuminaMoon Design**:
    - LuminaMoon uses **Adaptive VRAM-Budget Chunked LCE** (`_CUDA_TARGET_Z_BYTES = 400 MiB`, chunk size rounded to multiples of `256` in `[512, 4096]`): native `BF16`/`FP16` `torch.mm` (cuBLAS Tensor Cores) into a reusable `z_buf` paired with Triton online LSE and an atomic-free in-place backward pass (zero `atomic_add`), bounding peak logit buffer allocation without materializing the full `[B * N, V]` tensor.
    - `RaMoE` stores packed 3D expert tensors (`w_gate_up [E, D, 2H]`, `w_down [E, H, D_out]`), supports optional `num_shared_experts` and `router_logit_bound`, keeps `counts` on-device (zero GPU-to-CPU sync), and dispatches across 3 tiers (`sm_90+` `torch._grouped_mm` -> persistent Triton `triton_grouped_gemm` with zero-copy `HAS_GATHER` and fused packed activation -> `torch.mm` fallback).

### 3.5 LuminaV Optimizer: CLM, TD-CLM, and Triangle Family

*Components: `LuminaV` (`buffer=2`, `1`, `"T1"`, `"T2"`) · Compared works: Dozat (2016), Zhang et al. (2019), Zhuang et al. (2020), Taniguchi et al. (2024), Jiang et al. (2024), Liang et al. (2026), Chen et al. (2026), Dettmers et al. (2022), Zhao et al. (2024), Yu et al. (2024), Ozkara et al. (2025), Liang et al. (2025 / TorchTitan).*

- **Literature Baseline**:
    - **Nadam** (Dozat, ICLR Workshop 2016) computes a one-step Nesterov lookahead numerator normalized against raw second-moment gradients $g_t^2$; **Lookahead** (Zhang et al., NeurIPS 2019) maintains separate slow weights synced every $k$ steps.
    - **AdaBelief** (Zhuang et al., NeurIPS 2020) tracks innovation variance $(g_t - m_t)^2$; **ADOPT** (Taniguchi et al., NeurIPS 2024) decorrelates raw gradient squares via $\max(\sqrt{v _ {t-1}}, \epsilon)$; **Ada-STORM** (Jiang et al., NeurIPS 2024) analyzes adaptive variance reduction from gradient differences.
    - **Cautious Optimizers & CWD**: Liang et al. (ICLR 2026) apply alignment masking $\mathbb{I}(u_t \odot g_t > 0)$ with a two-pass global mask-mean rescaling; Chen et al. (ICLR 2026, `arXiv:2510.12402v2`, Sec. 3, p. 4) introduce Cautious Weight Decay, $\mathbf{x} _ {t+1} = \mathbf{x}_t - \eta_t (\mathbf{u}_t + \lambda \mathbb{I}(\mathbf{u}_t \odot \mathbf{x}_t \ge \mathbf{0}) \odot \mathbf{x}_t)$, preserving unbiased stationary points ($\nabla f(\mathbf{x}^\star) = \mathbf{0}$) with sliding-mode dynamics toward smaller parameter magnitudes.
    - **Memory-efficient, low-precision & distributed training**: 8-bit Adam (Dettmers et al., ICLR 2022) and GaLore (Zhao et al., ICML 2024) compress optimizer states via block quantization or SVD; COLLAGE (Yu et al., ICML 2024) and Ozkara et al. (AISTATS 2025) study stochastic rounding (Gupta et al., 2015; Zamirai et al., 2021); TorchTitan (Liang et al., ICLR 2025, `arXiv:2410.06511v3`, pp. 4, 16) advances PyTorch FSDP (Zhao et al., 2023) to FSDP2 dim-0 `DTensor` sharding, communication-free DCP state dicts, and 2D `DeviceMesh` HSDP.
- **LuminaMoon Design**:
    - **One-Step Convex Lookahead (`buffer=2`)**: Replaces separate Lookahead slow weights with an in-place one-step convex lookahead numerator $\tilde{m}_t = \beta_1^2 m _ {t-1} + (1 - \beta_1^2) g_t$, normalized by innovation residual variance $d_t^{(\text{B2})} = g_t - m_t$ (with optional `"rtv"` outlier tempering) and projected into $(-1, 1)$ via a Padé $[13/6]$ $\tanh$ approximation.
    - **TD-CLM (`buffer=1`)**: Applies the triangle inequality $|w_1 m _ {t-1} + w_2 g_t| \le w_1 |m _ {t-1}| + w_2 |g_t|$ ($w_1 + w_2 = 1$, Maligranda, 2008) to keep $u_t \in (-1, 1)$ without $\tanh$ or a $v_t$ buffer, cutting optimizer state memory by 50%, with triangle-defect penalty $\lambda |d_t|$ ($\lambda^* \approx 0.30$) over $d_t = g_t - m _ {t-1}$.
    - **Triangle Family (`buffer="T1"`, `buffer="T2"`)**: Uses `tanh`-free rational envelopes in both the step update and rational update bounds ($p _ {\text{init}, i}$ is the parameter coordinate, $s_i = s _ {t, i}$ is the scaled update coordinate before bounding, and $f = \mathrm{bound} _ {\text{floor}}$, default $10^{-2}$). `T1` drops both $\tanh$ and square roots; `T2` keeps $\sigma_t$ in the envelope while dropping $\tanh$ and removing the square root from $b_i$:

      $\displaystyle \tilde{m} _ t = w_1 m _ {t-1} + w_2 g_t, \quad w_1 = \beta_1^2, \quad w_2 = 1 - \beta_1^2, \quad d_t = g_t - m _ {t-1}$

      $\displaystyle \sigma _ t^{(\text{T2})} = \sqrt{\frac{\max(v_t, 0)}{\max(1 - \beta_2^t, 10^{-15})}}, \qquad \mathcal{E} _ t^{(\text{T2})} = w_1 |m _ {t-1}| + w_2 |g_t| + 0.5\lambda |d_t| + 0.5\lambda \sigma _ t^{(\text{T2})} + \epsilon _ {\text{eff}}$

      $\displaystyle u_t^{(\text{T2})} = \frac{\tilde{m} _ t}{\mathcal{E} _ t^{(\text{T2})}} \in (-1, 1)$

      $\displaystyle b_i = \mathrm{bound} _ {\text{ratio}} \cdot (\lvert p _ {\text{init}, i} \rvert + f), \qquad s _ {\text{bounded}, i} = \frac{b_i (b_i + |s_i|) s_i}{b_i (b_i + |s_i|) + s_i^2}$

    - **Fused Single-Pass Pipeline (`fused_single_pass=True`)**: Uses a two-pass reduction at $t=1$, then updates weights for $t > 1$ with lagged EMA alignment/radial scalars from $t-1$ in one kernel pass (Dao et al., NeurIPS 2022), while keeping FSDP2 `DTensor` / 2D HSDP shard-group reductions coalesced and DCP-safe.
    - **3-Salt Stateless SR & Bitwise-BF16 State**: Applies stateless 32-bit MurmurHash3 spatial hashing with 3D Weyl-Kronecker step offsets (`WEYL_C_PHI`, `WEYL_C_SQRT2`, `WEYL_C_SQRT3`), unsigned logical shifts, and independent salts (`seed_p`, `seed_m`, `seed_v`), storing `float16` optimizer states as Bitwise-BF16 inside 16-bit containers (`master_weights="none"` or `"semi"`) without allocating FP32 state buffers.

### 3.6 Data Engine: Filtering, Packing, and Checkpoints

*Components: `DataPipeline`, `FusedPipeline`, `SequencePacker`, `StringTensor`, `StateTracker` · Compared works: Penedo et al. (2024), Ding et al. (2024), Kirsch & Mitzenmacher (2008), Neumann & Freitag (2020), Pedreira et al. (2022).*

- **Literature Baseline**:
    - **Penedo et al. (NeurIPS 2024, `FineWeb`)** showed LLM pretraining quality depends on heuristic document filtering combined with exact and MinHash deduplication, typically executed as multi-stage offline CPU cluster pipelines.
    - **Ding et al. (ICML 2024, `arXiv:2404.10830v2`, p. 1)** introduced *Best-Fit Packing* (via Best-Fit-Decreasing) to eliminate unnecessary document truncations—reporting relative gains on reading comprehension (`+4.7%`), context following (`+16.8%`), and program synthesis (`+9.2%`), while reducing closed-domain hallucination by up to `58.3%`.
    - **Kirsch & Mitzenmacher (RSA 2008)** proved double hashing $g_i(x) = h_1(x) + i h_2(x)$ preserves the asymptotic false-positive probability of Bloom filters using only two base hashes (adapted heuristically in `MinHashDeduplicator` to derive $P$ signature hashes from two base hashes); **Neumann & Freitag (CIDR 2020, `Umbra`)** and **Pedreira et al. (PVLDB 2022, `Velox`)** introduced 16-byte prefix-inlined `StringView` layouts minimizing heap pointer chasing on SIMD CPUs.
- **LuminaMoon Design**:
    - `FusedPipeline` and `DataPipeline` unify SIMD whitespace/newline normalization (`AVX-512BW` / `AVX2` / `NEON`), Shannon character entropy and symbol/trigram-repetition filtering (`QualityFilter`), exact 128-bit `XXH3-128` SwissTable deduplication (`StreamDeduplicator`), double-hashed MinHash LSH (`MinHashDeduplicator`, adapted from Kirsch-Mitzenmacher), code-safe PII masking (`PIIMasker`), 16-byte Arrow `Utf8View`-aligned `StringTensor`, and streaming document-isolated Best-Fit packing (`SequencePacker` + `TokenBatch`).
    - `StateTracker` pairs $O(\Delta)$ append-only `LMWL v2` `.dedup` WAL persistence (with isolated `LMWP v2` magic in the pure-Python `BLAKE2b-128` fallback) with a 256-byte Dual-Slot Ping-Pong `LMDL v2` (`Slot A` / `Slot B`) `.meta` header verified by IEEE 802.3 CRC32 and exact-once packing markers (`emitted_above`, `_PackedSegment.is_final`), allowing interrupted streaming runs to resume from the recorded byte, row, and token coordinate.

---

## 4. Citing the Repository & Literature

If you use Loomy (`luminamoon`) in research or engineering projects, you can cite this repository as follows (see [references/bibliography.bib](bibliography.bib) for the full BibTeX database):

```bibtex
@misc{loomy2026,
  author = {Silver Moon and {Lumina Moon contributors}},
  title = {Loomy: Low-Precision Numerical Optimization, Rational Neural Architecture, and Data Processing Repository},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/cloverx-id/luminamoon}},
  doi = {10.5281/zenodo.23973325},
  note = {Loomy repository}
}
```
