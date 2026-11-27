# LuminaMoon Software Changelog

All notable software, packaging, FFI, and infrastructure changes to the `luminamoon` monorepo and PyPI distribution (`pip install luminamoon`) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For the mathematical and kernel evolution of the **LuminaV Optimizer** specification (`v1.0.0` to `v2.0.0`), see [`LUMINAV_ALGORITHM_CHANGELOG.md`](LUMINAV_ALGORITHM_CHANGELOG.md).

---

## [2.0.0rc1] - 2026-11-24

### Added
- **`DataPipeline`, `TokenBatch`, and `SequencePacker` (`python/luminamoon/data/{loader,_sources,packing}.py`)**: Added a streaming data pipeline supporting multi-source mixing, rank/worker sharding, cleaning, quality filtering, exact/MinHash deduplication, PII masking, custom `filter_fn`, document-isolated sequence packing (`best_fit`, `concat`, `pad`), resumable checkpointing (`commit()`, `state_dict()`, `load_state_dict()`), and multi-epoch iteration (`reset()`, `set_epoch()`).
- **Interactive CLI & In-Python Help (`python/luminamoon/cli.py`)**: Added the `luminamoon` console script (`luminamoon --help`, `--topic`, `--output`, `--quiet`, `--no-hardware`, `--version`) and `luminamoon.help()` / `LuminaV.help()` reference helpers.
- **CubeCL GPU Backend (`optimizer/crates/luminav-cubecl`)**: Replaced the legacy `ash` Vulkan crate with an optional CubeCL GPU backend (`wgpu`, `vulkan`, `metal`, `cuda`, `hip`, `cpu` features) supporting all four `LuminaV` buffer modes (`2`, `1`, `"T1"`, `"T2"`) and `luminamoon.nn` ops (`is_cubecl_available()`).
- **LuminaV v2.0.0 Optimizer Controls & FSDP2 Support**: Added Cautious Weight Decay (`cautious_weight_decay=False`), Rational-Tempered Variance (`variance_type="rtv"`, `variance_temper=3.0`), opt-in Rotational Pre-Conditioned Momentum (`momentum_type="standard" | "rpm"`, default `"standard"`, pairwise Cayley rotation across all 4 buffers on PyTorch eager, Triton JIT, CubeCL, Rust SIMD, Scalar, and NumPy backends—precompiled Triton AOT binaries target `momentum_type="standard"`—with Z3 SMT checks in `tests/test_rpm_v2.py`), configurable coordinate bound floor (`bound_floor=1e-2`), and PyTorch `FSDP2` / `DTensor` / `HSDP` rank sync alongside FSDP1 and DDP.
- **Triton `doc_ids` Block-Sparse Attention & Fused Vision Kernels (`python/luminamoon/nn/`)**: Added `doc_ids: Optional[Tensor] = None` document isolation with block-sparse tile skipping and a fast path for non-differentiable `attn_mask` to `RaAttention` / `ra_attention` (`tests/test_nn_doc_ids_v2.py`), plus fused Triton CUDA forward/backward kernels for `RaGELU` and `RaBatchNorm2d` in `python/luminamoon/nn/kernels/vision_act.py` (`tests/test_nn_vision_triton_v2.py`).
  * *Internal Test Note*: During internal development tests on an NVIDIA L4 GPU (no public benchmarks are published yet), passing `doc_ids` directly to Triton `RaAttention` increased packed-LM throughput from ~13.9K tok/s (dense mask) to ~49.9K tok/s eager and ~70.4K tok/s with `torch.compile` (~1.03 GiB VRAM), block-sparse `doc_ids` tile skipping reduced packed attention latency by ~2.0x–2.3x, and fused Triton `RaBatchNorm2d` / `RaGELU` improved ResNet-34 128x128 throughput from ~293 img/s (9.44 GiB) to ~2,020 img/s (1.22 GiB). In separate internal ablations under `buffer="t1"`, opt-in `momentum_type="rpm"` gave lower validation perplexity on SmolLM2-135M and higher Top-1 accuracy on ConvNeXt-Tiny and pure-BF16 ViT-Base/16 than `momentum_type="standard"` with no extra optimizer memory.
- **Position, Latent Attention, Reversible Drift-Kick-Drift (DKD) Blocks, and Memory Modules (`python/luminamoon/nn/`)**:
  * **Position (`position.py`, `kernels/position.py`)**: Added `QRoPE` / `qrope` (Double-Quaternion Rotary Position Embedding via $\mathrm{Spin}(4)$ on 4-element head blocks with fused Triton CUDA forward/backward kernels), `DistanceDecayBias` (linear and rational per-head distance decay with optional prefix hub damping), and `Sinusoidal3DEmbedding`.
  * **Latent & Slot Attention (`attention.py`, `_latent_attention.py`, `_attention_dispatch.py`, `_attention_bias_ops.py`, `_custom_ops_bias.py`)**: Added `LatentKVAttention` (low-rank Key-Value latent projection with `cache_mode="latent"` or `"kv"`, GPU shared-memory SRAM-fused non-learnable distance decay and hub causal masking in Triton `Flash-RaAttention` via `torch.library.custom_op` and compiled CUDA `flex_attention`), `SlotCrossAttention` (with Feature-wise Linear Modulation / FiLM), and thread-safe `use_attention_backend` / `get_attention_backend` context helpers (`"auto"`, `"sdpa"` = Scaled Dot-Product Attention, `"ra"`, `"flash"`, `"cudnn"`, `"fa3"`, `"flex"`).
  * **Reversible Blocks (`blocks.py`, `_reversible_ops.py`)**: Added `ReversibleDKDBlock`, `ReversibleDKDStack`, `resolve_selective_reversible`, and `ResidualRollout`, supporting selective reversibility (`"none"`, `"all"`, `"every"`, `"even"`, `"odd"`, `"first_half"`, `"second_half"`, `"indices"`), custom `kick_attn_fn` / `drift_ffn_fn` factories, per-block or whole-stack `torch.compile` (keeping `state_dict` keys unchanged), and CUDA Graph boundary guards for modes such as `mode="reduce-overhead"`.
  * **Memory & Hub (`memory.py`)**: Added `OrthogonalHub` ($\mathrm{SO}(H)$ float32 Cayley prefix hub with eval caching), `DirectiveGate` (chunk cosine/FiLM gating), and `TwoTierMemory` (two-tier slot memory combining Tier-1 `L1` leading slots and Tier-2 `L2` multi-head token-max LogSumExp selector with `encode_keys(...)` / `cached_keys` reuse).
- **Two-Mode `AutoTune` & Checkpoint Metadata (`python/luminamoon/optimizer/{autotune,_model_profile,_lr_probe}.py`)**: Added standalone `luminamoon.optimizer.AutoTune` (`LuminaV(..., tuning=tune)`) combining architecture, precision, and VRAM starting hyperparameters with an optional `loss_fn` LR and gradient-spike probe that restores parameters, module buffers, gradients, training mode, and RNG states unchanged, selective `tune_params`, `for_optimizer()` / `build()` helpers for other PyTorch optimizers (`AdamW`, `SGD`, etc.), and `weights_only=True`-safe settings saved under `state_dict()["luminav_tuning"]`.
- **Runtime `execution="auto"` Benchmarking & CLI Diagnostics (`luminamoon doctor` / `luminamoon setup`)**: Updated `execution="auto"` to benchmark available backends on the first active step (timing only the last few steps to skip warmup), enforced fail-fast `RuntimeError` / `DispatchError::BackendUnavailable` with install instructions for the current OS and PyTorch build when explicit `execution="triton"` or `"cubecl"` is unavailable, and added `luminamoon doctor [--benchmark]` plus opt-in `luminamoon setup <triton|gpu|cubecl> [--dry-run] [--yes]`.
- **Lazy Root Package Loading (`python/luminamoon/__init__.py`)**: Adopted PEP 562 lazy imports (`__getattr__`, `__dir__`, `TYPE_CHECKING`) so `import luminamoon` and `import luminamoon.data` do not eagerly import `luminamoon.optimizer` or `torch`, with migration messages for removed v1 symbols.
- **Direct Sibling Extension Loader (`python/luminamoon/data/_ffi.py`)**: Loads the sibling native extension binary directly without importing `luminamoon.optimizer` or scanning unrelated `sys.path` locations.
- **ARM64 `NEON` SIMD Backend (`optimizer/crates/luminav-simd`)**: Added AArch64 `NEON` vectorized kernels alongside deduplicated x86_64 `AVX-512` and `AVX2` paths.
- **Precompiled Triton AOT GPU Binaries (`optimizer/crates/luminav-triton-aot`)**: Replaced synthetic stubs with 182 compiled ELF64 GPU kernels (14 kernels across 13 CUDA and ROCm architectures), supporting `Dual` (`buffer=2`), `TdClm` (`buffer=1`), `T1`, and `T2`.
- **256-Byte `LMDL v2` Checkpoints, 128-Bit `XXH3-128` WAL & Arrow PyCapsule (`data/src/`)**: Upgraded `StateTracker` to 256-byte dual-slot Ping-Pong checkpoints (with `consumed_tokens` tracking), `StreamDeduplicator` to 128-bit `XXH3-128` hashes with $O(\Delta)$ append-only `LMWL v2` WAL (separated from pure-Python `LMWP v2` BLAKE2b-128 WAL), double-hashed MinHash LSH (adapted from Kirsch-Mitzenmacher), and `StringTensor.__arrow_c_array__` Arrow C-Data Interface export.
- **Tests & CI Coverage**: Added `tests/test_version.py`, `tests/test_root_lazy_and_ffi_v2.py`, `tests/test_autotune_v2.py`, `tests/test_backend_dispatch_v2.py`, `tests/test_rpm_v2.py`, `tests/test_optimizer_{api,parity,fsdp}_v2.py`, `tests/test_nn_{core,attention,loss_moe,compile,precision,doc_ids,vision_triton}_v2.py`, `tests/test_nn_reversible_latent_modules.py`, and `tests/test_data_{core_v2,pipeline_v2,v2_unified,fallbacks_modular}.py`, wired into `.github/workflows/ci.yml` with file line-budget checks.

### Changed
- **Reorganized Python & Rust Layout**:
  * Split `python/luminamoon/optimizer/` into `config.py`, `migration.py`, `buckets.py`, `state.py`, `luminav.py`, and `backends/{eager,fsdp,gpu,numpy_backend,triton/}`.
  * Split `python/luminamoon/nn/triton_kernels.py` into separate kernel files under `python/luminamoon/nn/kernels/`, separated `attention.py` / `loss.py` / `blocks.py` internals into `_attention_ops.py`, `_attention_bias_ops.py`, `_attention_dispatch.py`, `_latent_attention.py`, `_reversible_ops.py`, `_loss_ops.py`, and `_custom_ops{,_bias,_moe}.py`, and moved CLI help section text into `python/luminamoon/_cli_sections.py`.
  * Split `data/src/py_bridge.rs` into `data/src/py_bridge/{mod,state,text,pipeline,wal}.rs`, and organized `python/luminamoon/data/` around `loader.py`, `_loader_ops.py`, `_sources.py`, `packing.py`, `collate.py`, `pipeline.py`, `ring_buffer.py`, `_ring_codec.py`, and `fallbacks/`.
- **Updated v2 Public API**:
  * `LuminaV`: Updated `buffer` (`1`, `2`, `"T1"`, `"T2"`), `execution` (`"auto"`, `"triton"`, `"cubecl"`, `"simd"`, `"eager"`, `"scalar"`), read-only `MATH_VERSION = "2.0.0"` / `get_math_version()`, and `seed` reproducibility.
  * `luminamoon.nn`: Updated exports to `RaNorm`, `RaBatchNorm2d`, `RaTanh`, `RaGELU`, `RaSwiGLU` (`ra_swiglu`), `RaGLU` (`ra_glu`), `pos_ra_gate`, `ra_softmax`, `RaAttention` (`ra_attention`), `LinearCrossEntropy` (`linear_cross_entropy`, with `save_logits`), `RaMoE` (packed 3D weights, grouped GEMM, and optional `router_logit_bound`), and the PyTorch position, latent/slot attention, reversible DKD block, and memory modules (`QRoPE`/`qrope`, `DistanceDecayBias`, `Sinusoidal3DEmbedding`, `LatentKVAttention`, `SlotCrossAttention`, `use_attention_backend`/`get_attention_backend`, `ReversibleDKDBlock`, `ReversibleDKDStack`, `resolve_selective_reversible`, `ResidualRollout`, `OrthogonalHub`, `DirectiveGate`, `TwoTierMemory`).
  * `luminamoon.data`: Aligned FFI function signatures and defaults across Rust, `_stubs.py`, and `fallbacks/`, added batched tokenization (`tokenizer_batch_size`), configurable `token_dtype` (`"int64"` or `"int32"`), and double-buffered pinned staging to `TokenRingBuffer`, and upgraded Parquet/Arrow serialization to 64-bit `LargeUtf8`/`LargeBinary` with `data = ["pyarrow>=15.0.0", "polars>=1.0.0"]` optional extra.
- **Pre-Release Fixes & Guards (`v2.0.0rc1`)**:
  * Fixed long-document tail loss across `DataPipeline.commit()` + resume, fixed prefetch deduplication / PII-mask ordering and `DataPipeline.close()` shutdown spin, removed orphan `.meta.dedup` file creation, fixed `ResumablePipeline.load_state_dict()` off-by-one cursor handling without `checkpoint_path`, and forwarded `filter_fn` in `DataPipeline.as_collate_fn()`.
  * Added upfront shape, boolean mask, empty/0-D, and FP16 safety guards across `RaNorm`, `ra_attention`, `ra_softmax`, `RaMoE`, and `luminav-cubecl`.
  * Added FFI scratch buffer contiguity checks, byte-range pointer anti-aliasing, thread-local workspace capacity capping, and `DTensor` bucket-fusion guarding.
  * Passed explicit `enable_fp_fusion=True` across all Triton optimizer and `luminamoon.nn` kernel launches and PTX compilation tests for compatibility without deprecation warnings across Triton 3.9 and 3.10+.
  * Updated documentation, reference bibliography entries (`references/bibliography.bib`, `references/README.md`, `eval/lso_eval_specification.tex`), and MkDocs/GitHub raw-tree links to match the `v2.0.0rc1` API.
  * Updated `tools/sync_version.py` for PEP 440 prerelease/Cargo SemVer syncing (`2.0.0rc1` <-> `2.0.0-rc.1`) and `tools/build_universal_wheel.py` to emit console script `entry_points.txt`.
- **Informational OSV Advisory Handling (`tools/security_audit.py`)**: Classified non-security RustSec `unmaintained`/`notice` advisories as non-blocking warnings while still blocking on all CVEs, GHSAs, CVSS scores, and `unsound` advisories.

### Removed
- Removed `optimizer/crates/luminav-vulkan` (replaced by `luminav-cubecl`), dead `optimizer/reference/` monolithic scripts, legacy runtime version setters (`set_version`, `ModularVersionResolver`), `python/luminamoon/data/version.py`, `python/luminamoon/data/adapters.py`, `LumiRingBuffer`, and v1 NN layers/aliases (`PosRaGLU`, `TriGLU`, `RaEMA`, `ra_ema`, and academic aliases).

---

## [1.1.1.post2] - 2026-11-09

### Fixed
- **NN Attention GQA Mixed-Precision**: Enforced automatic dtype alignment (`v_match = v.to(do.dtype)`) across Grouped-Query Attention operations (`_gqa_matmul_do_vt`, `_gqa_matmul_pv`, `_gqa_matmul_qk`, `_gqa_matmul_pt_do`, `_gqa_matmul_ds_k`, `_gqa_matmul_dst_q`), systematically preventing scalar type mismatches (`Float != Half`) during mixed-precision autograd backward passes.
- **NN Attention QK-Norm Casting**: Preserved the input activation tensor precision (`q = q_norm(q).to(x.dtype)`) under AMP autocast, preventing asymmetric FP32 weight promotion on query and key vectors.
- **NN MoE Parameter Alias**: Added `dim` keyword parameter alias to `RationalMoE` and `HybridRationalMoE` (`eff_in = in_features if in_features is not None else dim`) for compatibility with standard transformer block naming conventions.

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
  * Added independent variance seed derivation (`seed_v = (seed_m + 0x2A6D365B) & 0x7FFFFFFF` via `derive_variance_seed`) alongside `derive_kernel_seeds` and enabled stochastic rounding on low-precision (`bfloat16` / `float16`) innovation variance state updates ($v_t = \beta_2 v _ {t-1} + (1 - \beta_2) d_t^2$) across Triton JIT/AOT kernels, Rust AVX-512/AVX2/Scalar kernels, Vulkan SPIR-V shaders, and PyTorch/NumPy fallbacks. Prevents `bfloat16` round-to-nearest-even stagnation during variance decay when $\beta_2 = 0.999$.

### Changed
- **Default Learning Rate (`lr = 3e-4`)**:
  * Updated the default `lr` in `LuminaV` across Python (`DEFAULT_LR = 3e-4`) and Rust (`LuminaConfig::default()`, `ParamGroup::new()`) from `8e-4` to `3e-4` for broader stability across standard transformer and vision workloads.

### Fixed
- **NumPy `state_dict()` Indexing & Float16 State Decoding**:
  * Replaced memory-address pointer keys (`id(p)`) in NumPy `state_dict()` / `load_state_dict()` with deterministic `(group_idx, param_idx)` integer indices so checkpoints survive process restarts.
  * Updated `fuse_tensors=True` `state_dict()` export to clone per-parameter slices into standalone storage rather than exporting views pinned to the full contiguous bucket.
  * Tagged raw 16-bit integer state buffers with `_fp16_state_format = "bf16_bits"` so IEEE `float16` checkpoints are distinguished from bit-cast `bfloat16` integer buffers during `load_state_dict()`.
- **Triangle Family (`T1` / `T2`) Kernel Dispatch & Prewarming**:
  * Fixed Triton `T2` single-pass wrapper to pass the configured `eps` scalar to `_lumina_t2_fused_single_pass_kernel`.
  * Added `is_triangle` branching to `_lumina_tdclm_pass2_kernel` and `_lumina_post_step_ema_kernel` so `buffer="T1"` at `step=1` with `cautious=True` uses the rational update bound $\tilde{m} _ t / \mathcal{E} _ t^{(\text{T1})}$ instead of $\tanh$.
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
  * Computes closed-form gradients using the analytical derivative of the Cayley-Padé kernel together with the shift-invariance correction at the first-argmax index $m_i^*$:

    $\displaystyle w _ {ij} = -\frac{16 \cdot \mathrm{scale}}{1 - u _ {ij}^2} \mathbb{I}(u _ {ij} < 1), \qquad dS _ {ij}^{\text{direct}} = -P _ {ij} \, w _ {ij} \left(dP _ {ij} - \sum_k P _ {ik} \, dP _ {ik}\right)$

    $\displaystyle \frac{\partial \mathcal{L}}{\partial S _ {ij}} = dS _ {ij}^{\text{direct}} - \mathbb{I}(j = m_i^*) \sum_k dS _ {ik}^{\text{direct}}$

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
  * **Python JIT (`_lumina_t1_single_pass_kernel`, `_lumina_t2_fused_single_pass_kernel`)**: Added dedicated fused register kernels for the Triangle Family (`buffer="T1"` and `buffer="T2"`), replacing eager PyTorch chunking and removing CPU-GPU `.item()` sync stalls (in internal development tests, reducing T2 step latency from ~20.7 ms to ~0.9–1.0 ms; no public benchmarks are published yet).
  * **Rust Triton AOT Crate (`luminav-triton-aot`)**: Added Ahead-Of-Time compiled kernels for `T1` and `T2` across 12 architectures (Turing, Ampere, Ada, Hopper, Blackwell, CDNA, RDNA3), expanding the embedded manifest to 96 binary artifacts.
- **Version Capability Checks**:
  * Enforced specification capabilities across historical LuminaV algorithm versions (`1.6.0`, `1.5.5`, `1.5.4`, `1.5.3`): locking to `1.5.4` or `1.5.5` rejected features introduced in later versions (such as `buffer="T2"` or `fuse_tensors=True`) to keep older runs reproducible.

### Added
- **Ra-Family (Rational, Tanh-Free & Square-Root-Free Neural Architecture)**:
  * **Zero Mandatory PyTorch Dependency**: The `luminamoon` package remains lightweight (~2.4 MB) with zero required dependencies (`dependencies = []`). The new `luminamoon.nn` module automatically detects the runtime environment (`_compat.py`), inheriting from `torch.nn.Module` when PyTorch is installed, or falling back to standalone NumPy buffers when PyTorch is absent.
  * **`RaNorm` (`luminamoon.nn.RaNorm`)**: Token normalization using $L_1$ mean absolute deviation instead of root-mean-square. Eliminates square-root evaluations and amortizes a single scalar reciprocal per token vector.
  * **`RaBatchNorm2d` (`luminamoon.nn.RaBatchNorm2d`)**: 2D batch normalization using cross-batch mean absolute deviation (MAD). Includes inference weight folding (`fold_into_conv`) that merges running statistics into preceding `Conv2d` layers with zero additional inference latency.
  * **`RaTanh` & `ra_tanh`**: Rational hyperbolic tangent with non-vanishing algebraic gradient tails $O(1 / |x|^3)$ and unit derivative at origin.
  * **`RaGELU` & `ra_gelu`**: Rational GELU activation using rational polynomial approximations.
  * **`PosRaGLU` (`luminamoon.nn.PosRaGLU`)**: Gated Linear Unit with strictly positive rational gating bounded in $(0, 1)$. Eliminates quadratic activation outlier explosion for stable low-precision (INT4/FP8) quantization.
  * **`RaGLU` & `TriGLU`**: Standard and triple rational gated linear unit variants.
  * **`RaAttention` & `RaAttn` (`luminamoon.nn.RaAttention`, `luminamoon.nn.RaAttn`)**: Multi-head attention using Cayley-Padé rational softmax. Supports causal masking, QK-Norm (`qk_norm=True`), dropout, and custom head dimension scaling.
  * **`ra_softmax` (`luminamoon.nn.ra_softmax`)**: Rational softmax with compact support for $\Delta \ge 16$, keeping the sum of numerators $\ge 1.0$ without requiring an epsilon floor.
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

    By the Triangle Inequality, $|w_1 m _ {t-1} + w_2 g_t| \le w_1 |m _ {t-1}| + w_2 |g_t| \le \mathcal{E} _ t^{(\text{T2})}$, which keeps the update step $u_t^{(\text{T2})}$ within $(-1, 1)$ without evaluating $\tanh$. The term $0.5\lambda |d_t|$ acts as a **triangle-defect penalty**, while $0.5\lambda \sigma_t$ dampens belief-variance dispersion.
  * **`buffer="T1"` (1-Buffer Triangle Family Mode)**: Pairs the single-buffer **Triangle-Defect** denominator ($\mathcal{E} _ t^{(\text{T1})} = w_1 |m _ {t-1}| + w_2 |g_t| + \lambda |d_t| + \epsilon _ {\text{eff}}$ with residual $d_t = g_t - m _ {t-1}$) with the rational update bounds below via the Triangle Inequality, eliminating the second-moment accumulator ($v_t$) and reducing optimizer state memory by 50%.
  * **Rational Update Bounds (for `T1` & `T2`)**: Replaces $\tanh$ and $\sqrt{\cdot}$ in the coordinate, radial, and FP16 update bounds with simple rational formulas across Rust, PyTorch, FSDP, and NumPy:

    $\displaystyle b_i = \mathrm{bound} _ {\text{ratio}} \cdot (|w_i| + 10^{-2}), \qquad \Delta w_i = \frac{b_i (b_i + |s_i|) s_i}{b_i (b_i + |s_i|) + s_i^2}, \qquad \mathrm{scale} _ {\text{radial}} = \frac{1 + r}{1 + r + r^2}$
- **Backward Compatibility**:
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
- **Cross-Precision Checkpoint Handshake**: Added `_convert_torch_state_buffer` in `luminamoon.optimizer.luminav` for state conversions during `load_state_dict` and `step` between `master_weights=False` (`torch.float16` Bitwise-BF16) and `master_weights="full"` (`torch.float32` / `torch.bfloat16`).
- **Hardware Parity & Regression Tests**: Added `test_vulkan_parity_dual_fp16_native_bf16_state_131072_5steps` and `test_vulkan_parity_tdclm_fp16_native_bf16_state_131072_5steps` in `vulkan_parity.rs` and `run_pure_fp16_native_bf16_state_tests` in `test_end_to_end_training.py`.

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
- **Optional Dependency Import Diagnostics**: Wrapped `numpy`, `torch`, `pyarrow`, and `polars` imports in `StringTensor` export methods with clear installation hints.
- **Fallback Signature Parity**: Aligned `suggest_closest_matches` (with Damerau-Levenshtein edit-distance ranking) and `postprocess_and_dedup_records` return tuple `(cleaned_unique, newly_committed)` in `luminamoon.data.fallbacks` with Rust FFI and `_stubs.py`.

### Fixed
- **WAL & GPU Buffer Validation**: Added byte-alignment validation (`% 8 == 0` for `.dedup` WAL files, `% 4 == 0` for GPU `uint32` offset buffers), pre-commit offset seeking in `DualMetadataManager::commit_gpu_batch`, and `StringView16` field encapsulation (`pub(crate)`), inline length clamping, and saturating `u32` offset arithmetic in `data/src/string_tensor.rs`.
- **Data Pipeline & Collate Fixes**: Fixed WAL double-commit skip in `fused_process_batch`, added missing column typo suggestions and `bytes` decoding in `_extract_text`, auto-populated `tokenizer.pad_token = tokenizer.eos_token` when missing, restored resume coordinates in `ResumablePipeline`, and eliminated background prefetcher queue timeout drops.

---

## [1.0.5.post1] - 2026-10-25

### Changed
- **Modular Data Processing Engine**: Decomposed the monolithic data processing fallback layer in `luminamoon.data` into dedicated submodules under `fallbacks/` (`cleaner.py`, `quality.py`, `dedup.py`, `minhash.py`, `masker.py`, `coordinate.py`, `tracker.py`, `string_tensor.py`, `fused.py`, `serialization.py`, `helpers.py`), preserving backward compatibility.
- **Updated Algorithm Changelog**: Updated `LUMINAV_ALGORITHM_CHANGELOG.md` with detailed mathematical formulation and parameter comparisons between 1.3.3 and 1.5.3.
- **PEP 440 Post-Release Synchronization**: Bumped package distribution version to `1.0.5.post1`.

---

## [1.0.5] - 2026-10-24

### Added
- **In-Editor Docstrings (`luminav.py`)**: Added Google-style docstrings across `LuminaV.__init__`, `step`, `zero_grad`, `state_dict`, `load_state_dict`, and `detect_hardware`, providing IDE autocomplete tooltips and default hyperparameter values.
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
- **SIMD Finite BF16 Clamps**: Tightened AVX-512 and AVX2 finite clamp boundaries prior to BFloat16 stochastic rounding.
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
