# LuminaMoon Software Changelog

All notable software, packaging, FFI, and infrastructure changes to the `luminamoon` monorepo and PyPI distribution (`pip install luminamoon`) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For the mathematical and kernel evolution of the **LuminaV Optimizer** specification (`v1.0.0` to `v1.6.0`), see [`LUMINAV_ALGORITHM_CHANGELOG.md`](LUMINAV_ALGORITHM_CHANGELOG.md).

---

## [1.1.1.post1] - 2026-11-09

### Added
- **Ra-Family Neural Architecture Extensions (`luminamoon.nn`)**:
  * **Fused Linear Cross-Entropy (`FusedLinearCrossEntropy`, `ChunkedLinearCrossEntropy`, `fused_linear_cross_entropy`, `chunked_linear_cross_entropy`)**: Computes output projection $z = (x W^\top + b) \cdot \text{scale}$ and cross-entropy loss across token chunks (`chunk_size=2048` on CUDA, `512` on CPU) with Triton online LogSumExp and in-place analytical backward kernels (`_fused_ce_chunk_fwd_kernel`, `_fused_ce_chunk_bwd_inplace_kernel`, `_fused_linear_ce_fwd_kernel`) without materializing the full $[B \cdot N, V]$ vocabulary logit tensor in memory. Supports both stateful LM-head mode (`in_features`, `vocab_size`) and stateless criterion mode, `label_smoothing`, `logit_scale`, `ignore_index`, and native low-precision (`bfloat16`/`float16`) Tensor Core GEMM.
  * **Rational Mixture of Experts (`RationalMoE`, `HybridRationalMoE`, `RaMoE`, `permute_tokens_by_expert`)**: Top-$k$ sparse Mixture-of-Experts layer using `ra_softmax` router gating (`ra_scale`, `aux_loss_coef`, `return_aux_loss`) paired with `PosRaGLU`, `RaGLU`, or `TriGLU` experts, optional always-active shared experts (`num_shared_experts`), and a contiguous token permutation dispatcher (`torch.argsort` + `index_add_`).
  * **Grouped-Query & Multi-Query Rational Attention (`RaGQA`, `RationalGQA`, `num_kv_heads`)**: Added native GQA/MQA support (`H_q \bmod H _ {\text{kv}} = 0`) in `RaAttention` and `ra_attention` without expanding KV heads in memory when executing the Triton kernel.
  * **TensorCore Dot-Product Triton Kernels for `ra_attention`**: Updated `_flash_ra_attn_fwd_kernel`, `_flash_ra_attn_bwd_dq_kernel`, and `_flash_ra_attn_bwd_dkdv_kernel` to use matrix-multiply tile accumulation (`tl.dot`) for $D \ge 16$ power-of-two head dimensions while retaining scalar fallback for $D < 16$.
  * **Fused Multi-Head Rational EMA (`RaEMA`, `FusedRaEMA`, `RationalEMA`, `ra_ema`, `fused_ra_ema`)**: Added multi-head exponential moving average recurrence $h_t = a_t h _ {t-1} + (1 - a_t) x_t$ with positive rational decay gating (`pos_ra_gate`) and single-launch Triton forward/backward scan kernels (`_fused_ra_ema_fwd_kernel`, `_fused_ra_ema_bwd_kernel`).
- **Asynchronous Token Ring Buffer (`luminamoon.data`)**:
  * **`TokenRingBuffer` / `LumiRingBuffer`**: Added a thread-safe circular token buffer (`push`, `pop_tokens`, `next_batch`, `pop_batch`, `wait_for_prefill`, `close`) backed by contiguous `int64` storage with background producer prefill (`auto_prefill_tokens`) and optional page-locked host memory (`pin_memory=True`) for streaming `(input_ids, labels)` batches (`[batch_size, seq_len]`).
- **Stochastic Rounding for Innovation Variance (`exp_avg_sq`) Across All Backends**:
  * Added independent SplitMix64 seed derivation (`seed_v` via `0xD1B54A32D192ED03`) in `derive_kernel_seeds` and enabled stochastic rounding on low-precision (`bfloat16` / `float16`) innovation variance state updates ($v_t = \beta_2 v _ {t-1} + (1 - \beta_2) d_t^2$) across Triton JIT/AOT kernels, Rust AVX-512/AVX2/Scalar kernels, Vulkan SPIR-V shaders, and PyTorch/NumPy fallbacks. Prevents `bfloat16` round-to-nearest-even stagnation during variance decay when $\beta_2 = 0.999$.

### Changed
- **Default Learning Rate (`lr = 3e-4`)**:
  * Updated the default `lr` in `LuminaV` across Python (`DEFAULT_LR = 3e-4`) and Rust (`LuminaConfig::default()`, `ParamGroup::new()`) from `8e-4` to `3e-4` for out-of-the-box stability across standard transformer and vision workloads.

### Fixed
- **NumPy `state_dict()` Indexing & Float16 State Decoding**:
  * Replaced memory-address pointer keys (`id(p)`) in NumPy `state_dict()` / `load_state_dict()` with deterministic `(group_idx, param_idx)` integer indices so checkpoints survive process restarts.
  * Updated `fuse_tensors=True` `state_dict()` export to clone per-parameter slices into standalone storage rather than exporting views pinned to the full contiguous bucket.
  * Tagged raw 16-bit integer state buffers with `_fp16_state_format = "bf16_bits"` so IEEE `float16` checkpoints are distinguished from bit-cast `bfloat16` integer buffers during `load_state_dict()`.
- **Triangle Family (`T1` / `T2`) Kernel Dispatch & Prewarming**:
  * Fixed Triton `T2` single-pass wrapper to pass the configured `eps` scalar to `_lumina_t2_fused_single_pass_kernel`.
  * Added `is_triangle` branching to `_lumina_tdclm_pass2_kernel` and `_lumina_post_step_ema_kernel` so `buffer="T1"` at `step=1` with `cautious=True` uses the rational update bound $\tilde{m}_t / \mathcal{E}_t^{(\text{T1})}$ instead of $\tanh$.
  * Updated `_prewarm_all_configs` to pre-compile `_lumina_t1_single_pass_kernel` and `_lumina_t2_fused_single_pass_kernel` when `buffer="T1"` or `"T2"`, and routed `ExecutionMode::Vulkan` for `T1`/`T2` to the SIMD triangle implementation instead of the `tanh` shader.
- **Contiguous Bucket Fusion (`fuse_tensors=True`) & Hot-Path Overhead**:
  * Deduplicated shared/tied parameters during bucket construction, skipped weight decay and parameter mutation on sub-slices where `p.grad is None`, and disabled in-place `p.data` rebinding when `torch.compiler.is_compiling()` is active.
  * Guarded inner Rayon chunk parallelism in `luminav-simd` with `rayon::current_thread_index().is_none()` to avoid nested thread-pool contention during multi-parameter FFI steps, and cached validated parameter-group signatures (`_validated_sig`) in `LuminaV.step()`.

---

## [1.1.1] - 2026-11-07

### Added
- **Tiled Rational Attention (`ra_attention`, `flash_ra_attention`, `rational_attention`, `pade_attention`)**:
  * Added functional `ra_attention` in `luminamoon.nn` (`luminamoon.nn.ra_attention`) and updated `RaAttention.forward` to evaluate Cayley-Padé rational attention without materializing full $[B, H, N_Q, N_K]$ intermediate tensor chains in GPU global memory.
  * **Triton 2-Pass SRAM Tiled Kernels (`_flash_ra_attn_fwd_kernel`, `_flash_ra_attn_bwd_dq_kernel`, `_flash_ra_attn_bwd_dkdv_kernel`)**: Computes row-wise maximum $M_i = \max_k S _ {ik}$ and first-argmax index $m_i^*$ in Pass 1, then evaluates the denominator sum $Z_i = \sum_k \phi _ {\text{Ra}}(M_i - S _ {ik})$ and weighted value accumulator $\sum_k \phi _ {\text{Ra}}(M_i - S _ {ik}) V_k$ in Pass 2 to form $O_i = Z_i^{-1} \sum_k \phi _ {\text{Ra}}(M_i - S _ {ik}) V_k$ while skipping key/value tiles outside the compact support window ($u _ {\text{min}} \ge 1$, i.e. $\Delta _ {ij} \ge 16$).
  * **Query-Chunked PyTorch Fallback**: Automatically partitions queries into bounded chunks (`q_chunk_size=128`) with tile-level backward recomputation when running on CPU, with non-power-of-two head dimensions, with attention dropout (`dropout_p > 0`), or when an explicit additive `attn_mask` is provided, supporting broadcastable mask shapes and asymmetric value head dimensions ($D_v \neq D _ {qk}$).
  * **`torch.compile` (`fullgraph=True`) Compatibility (`_is_torch_compiling`)**: Added TorchDynamo compilation detection across `RaNorm`, `ra_tanh`, `PosRaGLU`, `ra_softmax`, and `ra_attention` to bypass untraceable PyO3 C-extension calls during graph capture and deduplicate aliased $(Q, K, V)$ tensor inputs before `autograd.Function.apply`.

### Changed
- **Closed-Form Analytical Autograd for `ra_softmax` (`_RaSoftmaxFunction`, `_ra_softmax_bwd_kernel`)**:
  * Replaced standard PyTorch autograd graph construction across the 8 intermediate operations (`delta`, `u`, `num`, `den`, `q`, `q2`, `q4`, `phi`) with a custom `torch.autograd.Function` (with row-chunked fallback at `_MAX_CHUNK_ELEMENTS = 4_194_304`) that saves only the input tensor $x$ and output probabilities $P$, reducing backward-pass activation memory by $8\times$.
  * Computes exact gradients in closed form using the analytical derivative of the Cayley-Padé kernel together with the shift-invariance correction at the first-argmax index $m_i^*$:

    $$\displaystyle w _ {ij} = -\frac{16 \cdot \mathrm{scale}}{1 - u _ {ij}^2} \mathbb{I}(u _ {ij} < 1), \qquad dS _ {ij}^{\text{direct}} = -P _ {ij} \, w _ {ij} \left(dP _ {ij} - \sum_k P _ {ik} \, dP _ {ik}\right)$$

    $$\displaystyle \frac{\partial \mathcal{L}}{\partial S _ {ij}} = dS _ {ij}^{\text{direct}} - \mathbb{I}(j = m_i^*) \sum_k dS _ {ik}^{\text{direct}}$$

---

## [1.1.0.post3] - 2026-11-07

### Changed
- **Academic Citation Consolidation**:
  * Removed redundant `CITATION.cff` metadata file to consolidate citation records directly into Zenodo (DOI: [`10.5281/zenodo.23978801`](https://doi.org/10.5281/zenodo.23978801)) and repository `README.md`.
- **Data Engine Version Introspection (`python/luminamoon/data/version.py`)**:
  * Streamlined data version checking so `get_version()` directly returns the active package version string without disk or mutable state side effects.
  * Maintained full backward compatibility for `list_versions()` and `set_version()` with informative `DeprecationWarning` notices explaining that the data engine operates on the unified native SIMD implementation.
  * Removed internal cache directory cloning and dynamic module loading routines from `~/.cache/luminamoon/versions`.
  * Updated `data/README.md` to document direct version inspection via `luminamoon.data.get_version()`.
- **Documentation & Technical Guides**:
  * Rewrote technical guides, component documentation, and reference overviews across `docs/` and `data/docs/` in direct, humble, and accessible language for human readability.
  * Preserved core algorithmic identities including the Triangle Family, Triangle Inequality Envelope, Gradient-Momentum Residual, and Triangle-Defect Penalty without inflated descriptors.
  * Normalized data engine documentation in `python/luminamoon/data` and `data/` to consistent English terminology.
  * Aligned the MinHash documentation in `docs/dataloader/sources_and_streaming/README.md` with the Rust implementation using XXH3-64 seed multipliers.

### Refined
- **Evaluation Specification Artifacts (`eval/`)**:
  * Updated `eval/lso_eval_specification.tex` with refined layout and mathematical notation.
  * Added updated compiled specification manuscript `eval/LSO_Eval.pdf` and removed legacy PDF files.
  * Aligned `tools/sync_huggingface.py` and `eval/README.md` to track `eval/LSO_Eval.pdf`.
- **PosRaGLU Activation Gating (`luminamoon.nn`)**:
  * Added explicit positive anchor support ($b > 0$) for `pos_ra_gate` and `PosRaGLU`, preserving unit derivative at the origin while bounding output in $(0, 1)$.

### Fixed
- **Markdown Parameter Table Pipe Escaping**:
  * Escaped pipe characters in inline code notation (`\|d_t\| = \|g_t - m_{t-1}\|`) within the README parameter table to prevent GitHub Flavored Markdown (GFM) table parsers from prematurely splitting table cells.
- **Static Type Analysis (`luminamoon.nn._compat`)**:
  * Declared `_rs_engine` at module scope with multi-tier import fallbacks to eliminate Pyright unresolved import warnings.

### Infrastructure
- **Continuous Integration & Release Automation**:
  * Added Python runtime setup to the Rust test workflow in `.github/workflows/ci.yml` so linkers reliably locate shared Python libraries during PyO3 FFI tests on updated runner images.
  * Streamlined GitHub Release note generation by eliminating redundant boilerplate headers and prioritizing direct changelog entries.

---

## [1.1.0.post2] - 2026-11-05

### Fixed
- **Rust Arrow & Parquet Dependencies (`RUSTSEC-2026-0442`)**:
  * Updated `parquet`, `arrow-array`, `arrow-schema`, and `arrow-ipc` to `60.1.0` in workspace manifests (`Cargo.toml` and `Cargo.lock`).
  * Resolves advisory **RUSTSEC-2026-0442** (CVE-2026-111959) in `arrow-data`, preventing potential out-of-bounds reads when validating RunEndEncoded arrays during dataset ingestion.

---

## [1.1.0] - 2026-11-04

### Added
- **Hardware-Accelerated Triangle Family Fused Triton Kernels (GPU JIT & Rust AOT)**:
  * **Python JIT (`_lumina_t1_single_pass_kernel`, `_lumina_t2_fused_single_pass_kernel`)**: Added dedicated fused register kernels for the Triangle Family (`buffer="T1"` and `buffer="T2"`), replacing eager PyTorch chunking and eliminating CPU-GPU `.item()` sync stalls, bringing T2 execution latency from 20.7 ms down to native GPU speed (~0.9–1.0 ms).
  * **Rust Triton AOT Crate (`luminav-triton-aot`)**: Added Ahead-Of-Time compiled kernels for `T1` and `T2` across 12 architectures (Turing, Ampere, Ada, Hopper, Blackwell, CDNA, RDNA3), expanding the embedded manifest to 96 binary artifacts.
- **Strict Version Capability Enforcement**:
  * Enforced specification capabilities across historical LuminaV algorithm versions (`1.6.0`, `1.5.5`, `1.5.4`, `1.5.3`): locking to `1.5.4` or `1.5.5` strictly rejects features introduced in later versions (such as `buffer="T2"` or `fuse_tensors=True`), guaranteeing reproducibility for academic benchmarks.

### Added
- **Ra-Family (Rational, Tanh-Free & Square-Root-Free Neural Architecture)**:
  * **Zero Mandatory PyTorch Dependency**: The `luminamoon` package remains lightweight (~2.4 MB) with zero required dependencies (`dependencies = []`). The new `luminamoon.nn` module automatically detects the runtime environment (`_compat.py`), inheriting from `torch.nn.Module` when PyTorch is installed, or falling back cleanly to standalone NumPy buffers when PyTorch is absent.
  * **`RaNorm` (`luminamoon.nn.RaNorm`)**: Token normalization using $L_1$ mean absolute deviation instead of root-mean-square. Eliminates square-root evaluations and amortizes a single scalar reciprocal per token vector.
  * **`RaBatchNorm2d` (`luminamoon.nn.RaBatchNorm2d`)**: 2D batch normalization using cross-batch mean absolute deviation (MAD). Includes inference weight folding (`fold_into_conv`) that merges running statistics into preceding `Conv2d` layers with zero additional inference latency.
  * **`RaTanh` & `ra_tanh`**: Rational hyperbolic tangent with non-vanishing algebraic gradient tails $O(1 / |x|^3)$ and unit derivative at origin.
  * **`RaGELU` & `ra_gelu`**: Rational GELU activation using rational polynomial approximations.
  * **`PosRaGLU` (`luminamoon.nn.PosRaGLU`)**: Gated Linear Unit with strictly positive rational gating bounded in $(0, 1)$. Eliminates quadratic activation outlier explosion for stable low-precision (INT4/FP8) quantization.
  * **`RaGLU` & `TriGLU`**: Standard and triple rational gated linear unit variants.
  * **`RaAttention` & `RaAttn` (`luminamoon.nn.RaAttention`, `luminamoon.nn.RaAttn`)**: Multi-head attention using Cayley-Padé rational softmax. Supports causal masking, QK-Norm (`qk_norm=True`), dropout, and custom head dimension scaling.
  * **`ra_softmax` (`luminamoon.nn.ra_softmax`)**: Rational softmax with exact compact support for $\Delta \ge 16$, guaranteeing sum of numerators $\ge 1.0$ without requiring an epsilon floor.
  * **Dual Hardware Acceleration (Triton JIT GPU & Rust AVX-512/AVX2 SIMD CPU)**:
    - Added Triton GPU kernels (`_ra_norm_fwd_kernel`, `_pos_ra_glu_fwd_kernel`, `_ra_softmax_fwd_kernel`) in `luminamoon.nn.triton_kernels` for register-fused execution without intermediate global memory roundtrips.
    - Added native Rust SIMD kernels in `luminav-simd` with AVX2/FMA vectorization and PyO3 bindings for high-throughput CPU token normalization and rational softmax.
- **Academic Aliases (`luminamoon.nn`)**:
  * Exported standardized academic aliases across neural components: `L1Norm`, `L1LayerNorm`, `L1BatchNorm2d`, `RationalTanh`, `rational_tanh`, `RationalGELU`, `rational_gelu`, `RationalGLU`, `BoundedRationalGLU`, `rational_softmax`, `pade_softmax`, `RationalAttention`, and `PadeAttention`.
- **Theoretical References & Literature Documentation**:
  * Added `references/README.md` and `references/bibliography.bib` documenting literature mappings and rational approximations, paired with `docs/references.md` in the documentation.
- **Empirical Evaluation**:
  * Updated `eval/LSO-Eval.pdf` with the latest benchmark convergence findings and empirical evaluation report.
- **Unit Tests**: Added 32 unit tests in `tests/test_nn_ra_family.py` covering mathematical invariants, convolution folding, isolated zero-torch subprocess execution, and hardware introspection.

### Changed
- **Security Policy**: Streamlined `SECURITY.md` supported version matrix (`luminamoon >= 1.1.0`, `luminav >= 1.6.0`) with simplified reporting guidelines.

---

## [1.0.9] - 2026-11-03

### Added
- **LuminaV v1.6.0 — T-FAMILY (Triangle Family) tanh-free Modes (`buffer="T1"` & `buffer="T2"`)**:
  * **`buffer="T2"` (2-Buffer Triangle Family Mode)**: Keeps the 2-buffer state (`exp_avg` and `exp_avg_sq`) without using $\tanh$. Computes the gradient-momentum residual $d_t = g_t - m _ {t-1}$ and bounds the lookahead update via the **Triangle Inequality Envelope**:

    $\displaystyle \mathcal{E} _ t^{(\text{T2})} = w_1 |m _ {t-1}| + w_2 |g_t| + 0.5\lambda |d_t| + 0.5\lambda \sigma_t + \epsilon _ {\text{eff}}, \qquad u_t^{(\text{T2})} = \frac{\tilde{m} _ t}{\mathcal{E} _ t^{(\text{T2})}} \in (-1, 1)$

    By the Triangle Inequality, $|w_1 m _ {t-1} + w_2 g_t| \le w_1 |m _ {t-1}| + w_2 |g_t| \le \mathcal{E} _ t^{(\text{T2})}$, which guarantees the update step $u_t^{(\text{T2})}$ strictly remains within $(-1, 1)$ without evaluating $\tanh$. The term $0.5\lambda |d_t|$ acts as a **triangle-defect penalty**, while $0.5\lambda \sigma_t$ dampens belief-variance dispersion.
  * **`buffer="T1"` (1-Buffer Triangle Family Mode)**: Pairs the single-buffer **Triangle-Defect** denominator ($\mathcal{E} _ t^{(\text{T1})} = w_1 |m _ {t-1}| + w_2 |g_t| + \lambda |d_t| + \epsilon _ {\text{eff}}$ with residual $d_t = g_t - m _ {t-1}$) with the rational update bounds below via the Triangle Inequality, eliminating the second-moment accumulator ($v_t$) and reducing optimizer state memory by 50%.
  * **Rational Update Bounds (for `T1` & `T2`)**: Replaces $\tanh$ and $\sqrt{\cdot}$ in the coordinate, radial, and FP16 update bounds with simple rational formulas across Rust, PyTorch, FSDP, and NumPy:

    $\displaystyle b_i = \mathrm{bound} _ {\text{ratio}} \cdot (|w_i| + 10^{-2}), \qquad \Delta w_i = \frac{b_i (b_i + |s_i|) s_i}{b_i (b_i + |s_i|) + s_i^2}, \qquad \mathrm{scale} _ {\text{radial}} = \frac{1 + r}{1 + r + r^2}$
- **100% Backward Compatibility & Zero Breaking Changes**:
  * Existing code and checkpoints using `buffer=2` (default), `buffer=1`, or `version="1.5.5"` continue to work as before with identical outputs and `state_dict()` compatibility.

---

## [1.0.8.post1] - 2026-10-30

### Fixed
- **PyPI Discovery Description & Package Metadata**:
  * Updated `pyproject.toml` distribution `description` to prominently highlight LuminaV low-precision BF16/FP16 optimization, zero-copy multimodal dataset streaming, and native Rust SIMD, Vulkan, and Triton acceleration.
- **Un-fused Bucket Cap Configuration Warning**:
  * Added protective `UserWarning` in `LuminaV.__init__` when `bucket_cap_mb` is explicitly configured while `fuse_tensors=False`, preventing developer confusion regarding memory bounding scope on uncoalesced parameter tensors.

---

## [1.0.8] - 2026-10-29

### Added
- **LuminaV v1.5.5 Contiguous Bucket Fusion (`fuse_tensors=False`) & Triton JIT Pre-Warmup (`prewarm_jit=True`)**:
  * Added opt-in `fuse_tensors: bool = False` in `LuminaV` to coalesce fragmented parameter groups into contiguous 1D buffers (`flat_p`, `flat_g`, `flat_m`, `flat_v`, `flat_mp`) with zero-copy tensor views, in-place `zero_grad()` pointer preservation, and LSO-Eval Stage II compliant `state_dict()` / `load_state_dict()` round-tripping.
  * Added `prewarm_jit: bool = True` (enabled by default) in `LuminaVGpuMixin._prewarm_all_configs` to pre-compile Triton GPU kernels (`step=1` two-pass and `step=2+` fused single-pass + post-step EMA) on 64-element dummy tensors (`lr=0.0`) during `__init__()`.
  * Isolated `_prepare_triton_buffers` state lookup via `self.state.get(p)` to prevent contiguous bucket or warmup buffers from polluting `self.state`.
- **Bounded Bucket Chunking (`bucket_cap_mb = 500.0`) & Sequential In-Place Migration**:
  * Added `bucket_cap_mb: float = 500.0` in `LuminaV` to partition contiguous parameter coalescing into bounded chunk buckets (maximum 500 MB per bucket), preventing monolithic VRAM allocation spikes on large models (e.g. 4B+).
  * Implemented `_partition_into_bounded_chunks` with isolated single-tensor handling for huge layers (e.g. large embedding matrices).
  * Implemented sequential in-place migration per parameter inside chunks with immediate dereferencing (`del old_data`, `del old_grad`, `del conv_m`, `del conv_v`) and `gc.collect()`, ensuring that peak transient memory spikes during initialization never exceed 1 chunk (max 500 MB).

---

## [1.0.7] - 2026-10-29

### Added
- **LuminaV v1.5.4 Specification Upgrade**: Upgraded the bundled LuminaV Optimizer specification to `v1.5.4`, removing `centered_defect` and its global reduction kernels (`_lumina_diff_sum_kernel`, `diff_sum.comp`, `reduce_diff_sum_simd`) across Rust core, SIMD, Vulkan 1.3/1.4, Triton AOT/JIT, PyTorch, and NumPy backends to reduce per-step latency.
- **In-Container Native Bitwise-BF16 State Promotion**: Added native Bitwise-BF16 state promotion for pure `FP16` execution (`dtype_mode == 2`, `master_weights=False`) across Rust core, AVX-512/AVX2 SIMD, Vulkan 1.3/1.4 compute shaders, Triton kernels, PyTorch fallback, and standalone NumPy, preserving the 16-bit container memory footprint (zero extra VRAM) while expanding state dynamic range to 8-bit exponent ($1.18 \times 10^{-38}$).
- **Cross-Precision Checkpoint Handshake**: Added `_convert_torch_state_buffer` in `luminamoon.optimizer.luminav` for bit-exact state conversions during `load_state_dict` and `step` between `master_weights=False` (`torch.float16` Bitwise-BF16) and `master_weights="full"` (`torch.float32` / `torch.bfloat16`).
- **Comprehensive Hardware Parity & Regression Tests**: Added `test_vulkan_parity_dual_fp16_native_bf16_state_131072_5steps` and `test_vulkan_parity_tdclm_fp16_native_bf16_state_131072_5steps` in `vulkan_parity.rs` and `run_pure_fp16_native_bf16_state_tests` in `test_end_to_end_training.py`.

### Fixed
- **FP16 Convergence Stability & Loss Reset Mitigation**: Resolved sudden loss reset to $\ln(V)$ during convergence in pure `FP16` mode by eliminating second-moment underflow and unifying adaptive denominator epsilon (`effective_eps = eps = 1e-8`) across all execution backends (fixes #65).
- **Scalar Reference Parity Synchronization**: Fixed state quantization mode propagation in CPU scalar reference `fused_single_pass` path (`dual.rs` and `tdclm.rs`), maintaining mathematical parity between CPU scalar and GPU Vulkan shaders.
- **Headless CI Vulkan Detection Guard**: Added `LuminaV.detect_hardware()["vulkan_available"]` check in end-to-end tests to prevent false failures on headless CPU runners.

---

## [1.0.6.post1] - 2026-10-26

### Fixed
- **Prefetcher Batch ID Synchronization**: Fixed batch ID drift in `ResumablePipeline` during async prefetching (`prefetch_batches > 0`) by tagging each queue item with `(batch_payload, batch_id)` and synchronizing `self._active_batch_id` on consumption, eliminating premature coordinate commits and `UnknownBatchId` errors.
- **Data Engine Fallback Parity**: Fixed `StreamDeduplicator.count` callable signature in `luminamoon.data.fallbacks.dedup` (and high-level wrapper polymorphic dispatch), restored RAM staging in `StateTracker.update(..., batch_id=1)`, and aligned `fused_pipeline_process_batch` with two-stage staging and `commit_dedup` gating.
- **Optimizer Hyperparameter Constants**: Explicitly populated complete default hyperparameter constants in `luminamoon.optimizer.constants`.
- **Version Synchronization Tooling**: Added `--check` mode and PEP 440 validation in `tools/sync_version.py` to prevent accidental version string corruption.

---

## [1.0.6] - 2026-10-26

### Added
- **Python 3.15 Classifier Support**: Added `Programming Language :: Python :: 3.15` to package metadata in `pyproject.toml`.
- **O(1) Deduplication Membership Check**: Added `TwoStageDeduplicator::is_seen` in `data/src/dedup.rs` and wired `PyStreamDeduplicator::is_seen` to avoid $O(N)$ hash-vector exports.

### Changed
- **PROJECT LOOMY Rebranding & Author Standardization**: Standardized project branding to **PROJECT LOOMY** (`Loomy(luminamoon)`) with **Silver Moon (`@cloverxion`)** listed first as lead author followed by **Lumina Moon (`@cloverx-id`) contributors** across documentation and manifests.
- **Optional Dependency Import Diagnostics**: Wrapped `numpy`, `torch`, `pyarrow`, and `polars` imports in `StringTensor` export methods with actionable installation hints.
- **Fallback Signature Parity**: Aligned `suggest_closest_matches` (with Damerau-Levenshtein edit-distance ranking) and `postprocess_and_dedup_records` return tuple `(cleaned_unique, newly_committed)` in `luminamoon.data.fallbacks` with Rust FFI and `_stubs.py`.

### Fixed
- **WAL & GPU Buffer Corruption Hardening**: Added byte-alignment validation (`% 8 == 0` for `.dedup` WAL files, `% 4 == 0` for GPU `uint32` offset buffers), exact pre-commit offset seeking in `DualMetadataManager::commit_gpu_batch`, and `StringView16` field encapsulation (`pub(crate)`), inline length clamping, and saturating `u32` offset arithmetic in `data/src/string_tensor.rs`.
- **Data Pipeline & Collate Hardening**: Fixed WAL double-commit skip in `fused_process_batch`, added missing column typo suggestions and `bytes` decoding in `_extract_text`, auto-populated `tokenizer.pad_token = tokenizer.eos_token` when missing, restored resume coordinates in `ResumablePipeline`, and eliminated background prefetcher queue timeout drops.

---

## [1.0.5.post1] - 2026-10-25

### Changed
- **Modular Data Processing Engine**: Decomposed the monolithic data processing fallback layer in `luminamoon.data` into dedicated submodules under `fallbacks/` (`cleaner.py`, `quality.py`, `dedup.py`, `minhash.py`, `masker.py`, `coordinate.py`, `tracker.py`, `string_tensor.py`, `fused.py`, `serialization.py`, `helpers.py`), preserving backward compatibility.
- **Updated Algorithm Changelog**: Updated `LUMINAV_ALGORITHM_CHANGELOG.md` with detailed mathematical formulation and parameter comparisons between 1.3.3 and 1.5.3.
- **PEP 440 Post-Release Synchronization**: Bumped package distribution version to `1.0.5.post1`.

---

## [1.0.5] - 2026-10-24

### Added
- **Comprehensive In-Editor Docstrings (`luminav.py`)**: Added Google-style docstrings across `LuminaV.__init__`, `step`, `zero_grad`, `state_dict`, `load_state_dict`, and `detect_hardware`, providing full IDE autocomplete tooltips and explicit default hyperparameter values.
- **Hyperparameter Quick Reference (`README.md`)**: Added a structured default parameter reference table directly to the main repository README.
- **Dedicated Dual-Track Changelog Directory (`changelogs/`)**: Separated release documentation into `SOFTWARE_CHANGELOG.md` (package releases) and `LUMINAV_ALGORITHM_CHANGELOG.md` (algorithm specification history).
- **Explicit Algorithm Specification Constant (`LuminaV.ALGORITHM_SPEC`)**: Added `LuminaV.ALGORITHM_SPEC = "1.5.3"` to clearly distinguish the mathematical specification version from the software package version.

### Changed
- **Unified Software Version Synchronization (`1.0.5`)**: Synchronized `luminamoon.__version__`, `luminamoon.data.get_version()`, `pyproject.toml`, and `Cargo.toml` to a single source of truth (`1.0.5`).
- **Offline Version Resolution (`version.py`)**: Removed runtime HTTP GitHub file fetching from `luminamoon.optimizer.version` and `luminamoon.data.version`, allowing offline execution in air-gapped HPC clusters.

### Security
- **Upgraded `setuptools` (`84.0.0` -> `84.1.0`)**: Patched development dependency `setuptools` in `requirements.txt` to resolve `GHSA-xmj3-9gf7-h52w` (`CWE-22`).

---

## [1.0.4] - 2026-10-24

### Added
- **Free-Threaded CPython (No-GIL / PEP 703) Support**: Verified compilation and runtime execution of native `abi3t` / free-threaded wheels on Python `3.14t` and `3.15t`.
- **Rust Core Invariant Tests**: Added `optimizer/crates/luminav-core/tests/core_tests.rs` covering PRNG seed determinism, Padé rational $\tanh$ bounds, scratch buffer EMA smoothing, and Levenshtein typo suggestions.
- **PyPI Project URLs & Contributors Metadata**: Added `Homepage`, `Repository`, `Issues`, `Documentation`, and `Changelog` links alongside `Lumina Moon (cloverx-id) Contributors` to `pyproject.toml`, `Cargo.toml`, and `LICENSE`.

### Changed
- **Rust Workspace Modularization**:
  - Decomposed `luminav-core/src/reference_cpu.rs` into `reference_cpu/{mod, dual, tdclm}.rs`.
  - Decomposed `luminav-simd` (`avx2.rs`, `avx512.rs`, `parallel.rs`) into dedicated submodule directories (`dual.rs`, `tdclm.rs`, `mod.rs`).
  - Extracted Vulkan UMA step dispatches into `luminav-vulkan/src/{dual, tdclm}.rs`.
  - Decomposed `optimizer/src/python_ffi.rs` into `python_ffi/{config_parser, extract, worker, mod}.rs`.
- **Python Package Modularization**:
  - Split `luminav.py` into `standalone.py`, `validation.py`, and `numpy_backend.py`.
  - Split `gpu_backend.py` into `fsdp_sync.py`, `torch_ops.py`, and `triton_kernels.py`.
  - Added cross-platform cache directory resolution (`LOCALAPPDATA` on Windows, `XDG_CACHE_HOME` on POSIX).
- **Reference Implementation Restructuring**:
  - Decomposed monolithic `optimizer/reference/luminav_v153.py` (3,996 lines) into `core/`, `kernels/`, `single_backend/`, `triton_backend/`, and `utils/`.

---

## [1.0.3] - 2026-10-21

### Added
- **Two-Stage Deferred Release Pipeline**: Implemented draft GitHub release artifact staging on tag push (`v*`) with automated PyPI publication triggered upon release publication.

### Fixed
- **SIMD Finite BF16 Clamps**: Hardened AVX-512 and AVX2 finite clamp boundaries prior to BFloat16 stochastic rounding.
- **Build Wheel Privacy Sanitizer**: Added automated local path scrubbing in `tools/build_wheel.py` without modifying compiled ELF/PE/Mach-O relocation tables.

---

## [1.0.2.post2] - 2026-10-20

### Fixed
- **Dynamic Tag Versioning**: Added `tools/sync_version.py` to synchronize `pyproject.toml` and `_version.py` directly from Git release tags during CI wheel builds.

---

## [1.0.2.post1] - 2026-10-20

### Fixed
- **PEP 440 Post-Release Compliance**: Standardized version regex and enabled `skip-existing: true` on PyPI publishing workflows.

---

## [1.0.2] - 2026-10-20

### Added
- **Modular `luminamoon.data` Engine**: Added SIMD text normalization (`TextCleaner`), Shannon entropy & symbol heuristic filtering (`QualityFilter`), two-stage XXH3-64 exact deduplication (`StreamDeduplicator`), 64-permutation MinHash LSH fuzzy deduplication (`MinHashDeduplicator`), PII masking (`PIIMasker`), 128-byte Ping-Pong crash recovery (`StateTracker`), and dual-layout `StringTensor` (CPU 16-byte Umbra `StringView` & GPU contiguous offset+payload buffers).
- **LuminaV v1.5.3 Engine Integration**: Integrated the Rust AVX-512/AVX2 SIMD, Vulkan 1.3/1.4 UMA compute, and Triton AOT/JIT backends.

---

## [1.0.1] - 2026-10-20

### Added
- Initial monorepo release unifying the `luminamoon` Universal DataLoader and `LuminaV` Optimizer under a single Rust + Python workspace.
