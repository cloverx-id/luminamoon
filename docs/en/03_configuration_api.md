---
search:
  boost: 1.5
---

# Configuration & API Reference

Class and function signatures, parameter tables, public methods, and validation rules for **LuminaV (`luminamoon.optimizer`)**, **LuminaMoon Data Engine (`luminamoon.data`)**, **Ra-Family (`luminamoon.nn`)**, and the **CLI (`luminamoon.cli`)**.

---

## API 2.0 Stability and Deprecation Policy

> **Pre-Release Notice (`v2.0.0rc1`)**: This release is a **Release Candidate (`rc1`)** prior to `2.0.0` GA. While the core structure is nearing completion, **public APIs, argument names, and defaults may still change across `2.0.0rc*` builds** without a deprecation window. The full `5–8` minor-release stability contract below takes effect starting from **`2.0.0` GA**.

The **LuminaMoon 2.0** public API aims to keep public APIs stable across the `2.x.y` release line once `2.0.0` GA ships. Breaking changes to documented public signatures, default behaviors, or return contracts are avoided across minor and patch updates.

### Deprecation Window (`5–8` Minor Releases, Effective at `2.0.0` GA)

Whenever a documented public class, function, keyword argument, or option value needs to be retired or replaced during the stable `2.x` series:

1. **Explicit `DeprecationWarning`**: The deprecated interface emits a standard Python `DeprecationWarning` stating which version deprecated it, what replaces it, and the earliest target minor version for removal.
2. **Compatibility Window**: Deprecated public APIs remain functional for **5 to 8 minor releases (`2.<minor>.x`)** before removal (for example, an argument deprecated in `2.1.0` remains supported at least through `2.6.0`–`2.9.0`), giving users time to migrate.
3. **Documented Migration Notes**: Every deprecation and its replacement are recorded in [`changelogs/SOFTWARE_CHANGELOG.md`](../../changelogs/SOFTWARE_CHANGELOG.md) alongside a concrete before/after migration snippet.

### Stability Tiers

| Surface Tier | Scope | Compatibility Commitment (`2.0.0+` GA) |
| :--- | :--- | :--- |
| **Stable Public API (`2.0.0+` GA)** | Top-level exports in `luminamoon`, `luminamoon.optimizer` (`LuminaV`, `AutoTune`), `luminamoon.nn` (`RaNorm`, `RaBatchNorm2d`, `RaTanh`, `RaGELU`, `RaSwiGLU`, `RaGLU`, `ra_softmax`, `RaAttention`, `LinearCrossEntropy`, `RaMoE`, functional counterparts, and PyTorch position/latent-attention/reversible-block/memory modules such as `QRoPE`, `LatentKVAttention`, `SlotCrossAttention`, `ReversibleDKDBlock`, `ReversibleDKDStack`, `OrthogonalHub`, `DirectiveGate`, `TwoTierMemory`, `use_attention_backend`), `luminamoon.data` (`DataPipeline`, `TokenBatch`, `SequencePacker`, `FusedPipeline`, `StringTensor`, `StateTracker`, `StreamCoordinate`, `ResumablePipeline`, `TokenRingBuffer`, filters, deduplicators), and `luminamoon.cli`. | Backward compatibility across `2.x.y`; any removal requires a `DeprecationWarning` kept active for **5–8 minor releases (`2.<minor>.x`)**. *(May still change during `2.0.0rc*` pre-releases.)* |
| **Opt-in & Hardware-Gated Backends** | Feature-gated crates (`cubecl`), pre-compiled AOT GPU targets, and upstream runtime hooks (such as `torch._grouped_mm` on `sm_90+`). | Public selection flags (`execution="auto"`, etc.) remain stable; internal dispatch automatically falls back to standard kernels if an optional runtime or driver is unavailable. |
| **Private Internals (`_*`)** | Underscore-prefixed modules, helpers, and raw FFI bindings (`_luminamoon_*`, `_compat`, `_sources`, `_loader_ops`, `_ring_codec`, `_attention_ops`, `_attention_bias_ops`, `_attention_dispatch`, `_reversible_ops`, `_custom_ops*`, `_loss_ops`, `_cli_sections`). | Internal implementation details; may change between minor releases without deprecation warnings. |

---

## Part I: LuminaV Optimizer API

### 1. Constructor Signature (`LuminaV`)

```python
from luminamoon import LuminaV

optimizer = LuminaV(
    model.parameters(),
    lr=3.0e-4,
    betas=(0.9, 0.999),
    eps=1.0e-8,
    weight_decay=8.0e-2,
    cautious_weight_decay=False,
    variance_type="standard",
    variance_temper=3.0,
    momentum_type="standard",
    tau=0.8,
    lam=0.3,
    cautious=True,
    cautious_clamp_min=0.5,
    buffer=2,
    stochastic_rounding=True,
    bound=True,
    bound_type="coordinate",
    bound_ratio=0.05,
    bound_floor=1.0e-2,
    master_weights="none",
    execution="auto",
    fused_single_pass=True,
    fuse_tensors=False,
    bucket_cap_mb=500.0,
    prewarm_jit=True,
    ema_decay="auto",
    fsdp_sync=False,
    seed=1337,
    tuning=None,
)
```

### 2. Optimizer Parameter Reference Table

| Parameter | Type | Default | Allowed Values | Technical Description |
| :--- | :--- | :--- | :--- | :--- |
| `params` | `Iterable` | *(Required)* | Tensor list, NumPy array list, or parameter group dicts | Model parameter weights to optimize. |
| `lr` | `float` | `3.0e-4` | $\ge 0.0$ | Primary learning rate. |
| `betas` | `Tuple[float, float]` | `(0.9, 0.999)` | $0.0 \le \beta < 1.0$ | First-order ($\beta_1$) and second-order ($\beta_2$) momentum decay coefficients. |
| `eps` | `float` | `1.0e-8` | $> 0.0$ | Numerical stability constant preventing division by zero. |
| `weight_decay` | `float` | `8.0e-2` | $\ge 0.0$ | Decoupled weight decay coefficient (AdamW style). |
| `cautious_weight_decay` | `bool` | `False` | `True`, `False` | Applies weight decay only where `step_val * param >= 0.0` (CWD). |
| `variance_type` | `str` | `"standard"` | `"standard"`, `"rtv"` | 2-buffer variance update: standard L2 or Rational-Tempered Variance (RTV). |
| `variance_temper` | `float` | `3.0` | $> 0.0$ | RTV outlier-tempering ratio relative to $\max(\sigma _ {\text{prev}}, \lvert m _ {\text{prev}}\rvert)$. |
| `momentum_type` | `str` | `"standard"` | `"standard"`, `"rpm"` | Momentum update: standard EMA or opt-in pairwise Cayley rotation (RPM) on adjacent coordinates. |
| `tau` | `float` | `0.8` | $> 0.0$ | Temperature parameter for Dual-Buffer Convex Lookahead Momentum (CLM) scaling. |
| `lam` | `float` | `0.3` | $\ge 0.0$ | Triangle-defect penalty weight ($\lambda$) on $\lvert d_t \rvert = \lvert g_t - m _ {t-1} \rvert$ in 1-Buffer TD-CLM (`buffer=1`) and Triangle Family (`buffer="T1"`, `"T2"`). |
| `cautious` | `bool` | `True` | `True`, `False` | Enables cautious masking (updates only applied when $u \cdot g > 0$). |
| `cautious_clamp_min` | `float` | `0.5` | $0.0 < x \le 1.0$ | Floor clamp for cautious mask ratio to prevent over-amplification. |
| `buffer` | `int \| str` | `2` | `2`, `1`, `"T2"`, `"T1"` (`"t2"`, `"t1"`) | Momentum buffer mode: `2` (2-Buffer CLM), `1` (1-Buffer TD-CLM), `"T2"` (2-Buffer Triangle Family `tanh`-free), or `"T1"` (1-Buffer Triangle Family `tanh`-free). |
| `stochastic_rounding` | `bool` | `True` | `True`, `False` | Enables MurmurHash3 + 3D Weyl-Kronecker stochastic rounding for FP16/BF16 weights and states. |
| `bound` | `bool` | `True` | `True`, `False` | Enables gradient step bounding. |
| `bound_type` | `str` | `"coordinate"` | `"coordinate"`, `"radial"` | Bounding mode: `"coordinate"` (per-element) or `"radial"` (tensor $L_2$ norm). |
| `bound_ratio` | `float` | `0.05` | $> 0.0$ | Maximum update step ratio relative to parameter scale (to disable bounding, set `bound=False`). |
| `bound_floor` | `float` | `1.0e-2` | $> 0.0$ | Coordinate bound floor radius $f$: $\rho\sqrt{p^2 + f^2}$ (`buffer=1,2`) or $\rho(\lvert p\rvert + f)$ (`"T1","T2"`). |
| `master_weights` | `str` | `"none"` | `"none"`, `"semi"`, `"full"` | Mixed-precision master weight strategy: `"none"` (low-precision states), `"semi"` (FP32 master weights with 16-bit states), or `"full"` (FP32 master weights and states). |
| `execution` | `str` | `"auto"` | `"auto"`, `"triton"`, `"cubecl"`, `"simd"`, `"eager"`, `"scalar"` | Target hardware backend route. Under `"auto"`, `LuminaV` benchmarks available backends on the first step (timing only the last few steps, cached per model shape and device) and routes to the fastest working backend. Explicit `"triton"` / `"cubecl"` fail fast with OS-specific install instructions when unavailable. |
| `fused_single_pass` | `bool` | `True` | `True`, `False` | Blends parameter update and EMA tracking into a single kernel pass. |
| `fuse_tensors` | `bool` | `False` | `True`, `False` | Coalesces parameter groups into contiguous 1D buffers with views to reduce launch overhead on Vision models (CNN, ViT). Not recommended for large LLMs or FSDP sharding. |
| `bucket_cap_mb` | `float` | `500.0` | $> 0.0$ | Maximum memory capacity per contiguous chunk in MB when `fuse_tensors=True`. Emits a `UserWarning` if set to a non-default value (`!= 500.0`) or specified in a param group while `fuse_tensors=False`. |
| `prewarm_jit` | `bool` | `True` | `True`, `False` | Pre-compiles Triton launch kernels during initialization before the first training step. |
| `ema_decay` | `float \| str` | `"auto"` | `"auto"` or float $(0.0, 1.0)$ | EMA tracking coefficient for mask and bounding. `"auto"` sets `0.95` (`buffer=1`, `"T1"`) or `0.80` (`buffer=2`, `"T2"`). |
| `fsdp_sync` | `bool` | `False` | `True`, `False` | Enables global sync across distributed GPU ranks (FSDP1, FSDP2/`DTensor`, HSDP, DDP). |
| `seed` | `int` | `1337` | `0 <= seed <= 4294967295` | Base seed for deterministic MurmurHash3 + 3D Weyl-Kronecker stochastic rounding. |
| `tuning` | `AutoTune \| dict \| None` | `None` | `AutoTune(model[, loss_fn=...])`, dict, `None` | Overrides default hyperparameters (`lr`, `betas`, `weight_decay`, `cautious_weight_decay`, `variance_type`, `momentum_type`, `buffer`, `master_weights`, `stochastic_rounding`, `bound_ratio`, `fuse_tensors`, `prewarm_jit`) without clobbering explicit user arguments, and saves resolved settings under `state_dict()["luminav_tuning"]`. |

### 3. Optimizer Public Methods & `AutoTune`

1. **`AutoTune(model_or_params, *, loss_fn=None, optimizer_cls=None, tune_params=None, memory_priority="balanced", tune_hardware=None, steps=7, measure_last=2, lr_steps=5, strategy="auto", max_memory_mb=128.0, tune_fuse_tensors=True, verbose=True)`**: Standalone 2-mode tuner (`from luminamoon.optimizer import AutoTune`).
    - **Mode 1 (`AutoTune(model)`)**: Checks model shape, precision (`fp32`/`bf16`/`fp16`), and free VRAM to pick starting hyperparameters (`[prior]`), and benchmarks available backends for `steps` iterations while timing only the last `measure_last` steps (`strategy="auto"` benchmarks all layers of the real model when it fits in memory and restores weights unchanged).
    - **Mode 2 (`AutoTune(model, loss_fn=...)`)**: Runs a short multi-step LR and gradient-spike probe (`[empirical]`), restoring parameters, module buffers (`BatchNorm` running stats), gradients, training mode, and RNG states unchanged before returning.
    - **Summary & Checkpointing**: Prints the active parameter table and setting sources when `verbose=True`, exposes `tune.summary()` / `tune.to_metadata()`, and saves the effective setup into `LuminaV.state_dict()["luminav_tuning"]` (`weights_only=True`-safe).
    - **Use with Other Optimizers**: Pass `optimizer_cls=torch.optim.AdamW` (or `SGD`, etc.) or call `tune.for_optimizer(cls)` / `tune.build(cls, model.parameters())` to filter tuned hyperparameters for external PyTorch optimizers while `LuminaV` remains the default.
2. **`step(closure=None, grads=None)`**: Executes a single optimization step. `grads` is required when operating on standalone NumPy arrays.
3. **`zero_grad(set_to_none=True)`**: Resets gradients of all optimized parameters.
4. **`state_dict()`**: Returns a `weights_only=True`-safe dict containing `state`, `param_groups`, and `luminav_tuning` (resolved settings, sources, and backend timings) with transient hardware buffers stripped.
5. **`load_state_dict(state_dict)`**: Restores optimizer state and `tuning_info` metadata from a saved checkpoint while resetting host-local `execution="auto"` resolution for cross-hardware portability.
6. **`detect_hardware()`**: Static method returning host CPU SIMD, CubeCL, and Triton AOT capability metadata.
7. **`probe_backends(*, force=False)`**: Class method executing cached live micro-probes across all optimizer and data backends.
8. **`benchmark_backends(params, *, steps=7, measure_last=2, max_memory_mb=128.0, use_cache=False, **optimizer_kwargs)`**: Functional benchmark helper returning timings (`winner`, `timings_ms`, `per_step_ms`, `strategy`).
9. **`last_backend` / `resolved_execution` / `tuning_info` / `benchmark_results` / `tuning`**: Inspect the active backend, benchmark-resolved `"auto"` execution target, and full tuning metadata.
10. **`get_math_version()`**: Read-only class method returning the active optimizer mathematical version (`"2.0.0"`).
11. **`help(topic="optimizer", *, output=None, print_out=True, include_hardware=True)`**: Prints and optionally saves the optimizer parameter reference.

### 4. Parameter Validation & Typo Suggestions

If an unrecognized string option is passed, `LuminaV` computes the Levenshtein distance and raises a `ValueError` with a suggestion:

```python
LuminaV(model.parameters(), bound_type="radal")
# ValueError: Invalid bound_type: 'radal'. Expected 'coordinate' or 'radial'. Did you mean 'radial'?
```

---

## Part II: LuminaMoon Data Engine API (`luminamoon.data`)

### 1. `DataPipeline`, `TokenBatch`, and `SequencePacker`

`DataPipeline` is the streaming pipeline. It handles source mixing, sharding, filtering and cleaning, document-isolated packing, and resumable checkpointing:

```python
from luminamoon.data import DataPipeline, SequencePacker, TokenBatch

pipe = DataPipeline(
    source=raw_dataset,
    tokenizer=my_tokenizer,
    batch_size=4,
    seq_len=2048,
    packing="best_fit",           # "best_fit" | "concat" | "pad"
    pack_window=64,
    quality="pretraining",        # None | "pretraining" | QualityFilter
    dedup="exact",                # None | "exact" | "minhash" | "hybrid" | deduplicator
    mask_pii=False,
    filter_fn=None,
    checkpoint="./cache/train",
)
for batch in pipe:
    # Prefer passing batch.doc_ids directly to RaAttention / ra_attention on CUDA
    # for Triton SRAM block-sparse document isolation; use batch.causal_doc_mask()
    # when targeting dense attention layers.
    attn_mask = batch.causal_doc_mask()
    pipe.commit()
```

Key methods on `DataPipeline`:
- `commit()`: Saves consumed document and token positions to `.state.json` and flushes $O(\Delta)$ `.dedup` WAL entries.
- `state_dict()` / `load_state_dict(state)`: PyTorch `Stateful` / DCP-compatible checkpoint dict.
- `reset()` / `set_epoch(epoch)`: Resets stream cursors (and updates the deterministic shuffle/mix seed) for multi-epoch training.
- `as_collate_fn()`: Exports the pipeline's filtering and cleaning stages (including `filter_fn`) as a PyTorch `DataLoader` `collate_fn`.

### 2. `FusedPipeline`

Fused multi-stage text normalization, heuristic filtering, deduplication, and PII masking executed in parallel via Rayon and SIMD kernels:

```python
from luminamoon.data import FusedPipeline, StringTensor

# Standard Instantiation:
pipeline = FusedPipeline(
    trim_whitespace=True,
    normalize_newlines=True,
    filter_quality=True,
    min_chars=10,
    min_entropy=2.0,
    max_repetition_ratio=0.3,
    max_symbol_ratio=0.2,
)

# Process batch -> returns dual-layout StringTensor
tensor: StringTensor = pipeline(["Sample text line 1", "Sample text line 2"])
```

| Parameter | Type | Default | Technical Description |
| :--- | :--- | :--- | :--- |
| `trim_whitespace` | `bool` | `True` | Strips leading and trailing Unicode whitespace using SIMD byte scanning. |
| `normalize_newlines` | `bool` | `True` | Normalizes CRLF (`\r\n`) and lone CR (`\r`) sequences into standard `\n`. |
| `filter_quality` | `bool \| None` | `None` | Enables heuristic quality filtering (`None` auto-enables when any quality threshold such as `min_chars` or `min_entropy` is set). |
| `min_chars` | `int \| None` | `None` | Minimum string character length boundary required to retain a record. |
| `max_chars` | `int \| None` | `None` | Maximum string character length boundary (`None` = unbounded). |
| `min_words` | `int \| None` | `None` | Minimum whitespace-separated word count threshold. |
| `min_entropy` | `float \| None` | `None` | Minimum Shannon character entropy threshold to reject low-variety text. |
| `max_repetition_ratio` | `float \| None` | `None` | Maximum allowed repetitive n-gram fraction threshold. |
| `max_symbol_ratio` | `float \| None` | `None` | Maximum allowed ratio of non-alphanumeric non-whitespace symbols to total non-whitespace characters (`symbols / visible`). |
| `min_byte_len` | `int \| None` | `None` | Minimum UTF-8 byte length boundary. |
| `max_byte_len` | `int \| None` | `None` | Maximum UTF-8 byte length boundary (`None` = unbounded). |
| `truncate_over_max` | `bool` | `False` | Truncates strings exceeding `max_byte_len` instead of rejecting them. |
| `json_extract_key` | `str \| None` | `None` | Extracts a specific JSON field from raw string records before processing. |
| `deduplicator` | `StreamDeduplicator \| None` | `None` | Active exact deduplication SwissTable instance with WAL persistence. |
| `commit_dedup` | `bool` | `False` | Automatically commits SwissTable deduplication state to WAL upon batch processing. |

---

### 3. `StringTensor`

Dual-layout string container bridging 16-byte `StringView16` CPU processing, Apache Arrow C-Data Interface (`__arrow_c_array__`), and flat GPU offset/payload buffers:

```python
tensor = pipeline(["Sample text line 1", "Sample text line 2"])

length: int = len(tensor)
first_item: str = tensor[0]

polars_series = tensor.to_polars()
arrow_array = tensor.to_arrow()
offsets_cpu, payload_cpu = tensor.to_numpy_gpu()
# offsets_gpu, payload_gpu = tensor.to_torch_gpu(device="cuda")
```

| Method | Return Type | Description |
| :--- | :--- | :--- |
| `__arrow_c_array__(requested_schema=None)` | `tuple[PyCapsule, PyCapsule]` | Arrow C-Data Interface export (`Utf8View`) consumed directly by PyArrow and Polars. |
| `to_polars()` | `polars.Series` | Exports to a Polars Series via Arrow PyCapsule (`pip install --pre "luminamoon[data]"`). |
| `to_arrow(zero_copy=True)` | `pyarrow.Array` | Exports to a PyArrow array via Arrow PyCapsule (`zero_copy=True`) or offset/payload buffers. |
| `to_numpy_gpu()` | `tuple[np.ndarray, np.ndarray]` | Exports flat `uint32` offset array and contiguous `uint8` UTF-8 byte buffer. |
| `to_torch_gpu(device="cpu")` | `tuple[torch.Tensor, torch.Tensor]` | Converts contiguous offsets (`torch.int32`) and UTF-8 payload (`torch.uint8`) into PyTorch tensors on the requested `device` (defaults to `"cpu"`; pass `"cuda"` or `"mps"` to transfer to GPU). |

---

### 4. `StateTracker` (256-Byte `LMDL v2` Ping-Pong Checkpoint Recovery)

Dual-slot metadata management with IEEE CRC32 verification, `consumed_tokens` tracking, and $O(\Delta)$ `.dedup` WAL persistence:

```python
from luminamoon.data import StateTracker

tracker = StateTracker("./cache/train.meta")
tracker.consumed_tokens = 65536
tracker.update(
    shard_id=0,
    byte_offset=4096,
    row_group_idx=0,
    page_idx=0,
    row_offset_in_page=150,
    global_row_idx=4800,
    batch_id=150,
)
coord = tracker.commit(batch_id=150)
coord = tracker.read()
```

---

### 5. Modular Preprocessing & Heuristic Quality Components

```python
from luminamoon.data import (
    TextCleaner,
    QualityFilter,
    StreamDeduplicator,
    MinHashDeduplicator,
    PIIMasker,
    data_cpu_isa,
)

isa: str = data_cpu_isa()  # 'AVX-512', 'AVX2', 'NEON', or 'SCALAR'
cleaner = TextCleaner(trim_whitespace=True, normalize_newlines=True)
qfilter = QualityFilter.for_pretraining()
dedup = StreamDeduplicator(wal_path="./cache/dedup.dedup")
minhash = MinHashDeduplicator(threshold=0.8, k_shingle=5, num_perm=64)
masker = PIIMasker()
```

---

### 6. Low-Level Resumable Pipeline, Token Ring Buffer & Collate Adapters

```python
from luminamoon.data import ResumablePipeline, TokenRingBuffer, create_resumable_collate

pipe = ResumablePipeline(
    source=raw_data_iterable,
    fused_pipeline=pipeline,
    batch_size=64,
    prefetch_batches=4,
    checkpoint_path="./cache/train.meta",
)
for batch in pipe:
    pipe.commit()

collate_fn = create_resumable_collate(
    fused_pipeline=pipeline,
    return_string_tensor=True,
)

ring = TokenRingBuffer(
    source=raw_data_iterable,
    tokenizer=my_tokenizer,
    batch_size=8,
    seq_len=1024,
    tokenizer_batch_size=256,
    token_dtype="int32",  # or "int64" (default)
)
```

---

## Part III: Ra-Family Neural Modules (`luminamoon.nn`) & CLI (`luminamoon.cli`)

### 1. `luminamoon.nn` Exports

| Component / Function | Primary Parameters & Behavior |
| :--- | :--- |
| `RaNorm(normalized_shape, eps=1e-6, elementwise_affine=True, bias=False)` | Square-root-free $L_1$ token normalization ($x / (\mathrm{mean}(\lvert x \rvert) + \epsilon)$). |
| `RaBatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)` | 2D MAD batch normalization with `fold_into_conv(conv)`. |
| `RaTanh(b=1.0)` / `ra_tanh(x, b=1.0)` | Rational hyperbolic tangent bounded on $(-b, b)$. |
| `RaGELU()` / `ra_gelu(x)` / `pos_ra_gate(z, b=1.0)` | Rational GELU and strictly positive rational gate on $(0, 1)$. |
| `RaSwiGLU(in_features, hidden_features=None, out_features=None, bias=False, b=1.0)` / `ra_swiglu(gate, up, b=1.0)` | Rational SwiGLU FFN block (default `hidden_features = int(in_features * 8 / 3)`). |
| `RaGLU(in_features, hidden_features=None, out_features=None, bias=False, b=1.0)` / `ra_glu(gate, up, b=1.0)` | Hard-bounded Rational GLU FFN block (`RaTanh(gate; b) * up`). |
| `RaAttention(dim, num_heads=8, num_kv_heads=None, qkv_bias=False, qk_norm=False, attn_drop=0.0, proj_drop=0.0, is_causal=False, scale=None, ra_scale=0.0625, q_chunk_size=None)` / `ra_attention(...)` | MHA/GQA tiled rational attention supporting packed-sequence `doc_ids` (`forward(x, attn_mask=None, doc_ids=None)` / `ra_attention(..., doc_ids=None)`), additive float masks, and boolean masks (`True`=attend, `False`=mask). |
| `ra_softmax(scores, dim=-1, scale=0.0625)` | Cayley-Padé rational softmax with compact support ($\Delta < 1 / \mathrm{scale}$). |
| `LinearCrossEntropy(in_features=None, vocab_size=None, *, bias=False, ignore_index=-100, reduction="mean", label_smoothing=0.0, logit_scale=1.0, chunk_size=None, save_logits=False)` / `linear_cross_entropy(...)` | Chunked fused linear projection + cross-entropy loss (`save_logits=True` trades memory for faster backward). |
| `RaMoE(in_features, hidden_features=None, out_features=None, num_experts=8, top_k=2, *, num_shared_experts=0, expert_type="ra_swiglu", b=1.0, ra_scale=0.0625, router_logit_bound=None, bias=False, aux_loss_coef=0.01, return_aux_loss=False)` / `permute_tokens_by_expert(...)` | Packed 3D-weight Mixture-of-Experts with grouped GEMM and optional `router_logit_bound` (via `ra_tanh`). |
| `QRoPE(head_dim=32, max_seq_len=4096, base=10000.0, freq_ratio=0.6180339887498949)` / `qrope(...)` | PyTorch Double-Quaternion Rotary Position Embedding ($\mathrm{Spin}(4)$ rotation on 4-element head blocks, `head_dim % 4 == 0`), with fused Triton CUDA kernels (`apply_qk` single-launch QK path) and float32 CPU Rust SIMD. |
| `DistanceDecayBias(num_heads=8, hub_size=0, ..., mode="linear" \| "rational", learnable=False)` / `Sinusoidal3DEmbedding(dim, theta=10000.0)` | PyTorch additive per-head distance decay bias (with optional prefix hub damping and bounded LRU caching) and 3D `(z, y, x)` sinusoidal coordinate embeddings. |
| `LatentKVAttention(dim, num_heads=8, head_dim=None, kv_latent_dim=None, hub_size=0, ..., bias_mode="linear", attention_backend="auto", attention_fn=None, cache_mode="latent")` | PyTorch low-rank Key-Value (KV) latent attention with `QK-RaNorm`, full `QRoPE` after up-projection, `DistanceDecayBias`, incremental `"latent"` or `"kv"` caching, and GPU shared-memory (SRAM) bias/hub fusion on Triton `"ra"` (`torch.library.custom_op`) or compiled CUDA `"flex"`. |
| `SlotCrossAttention(dim, num_heads=8, num_terminals=16, slots_per_terminal=4, attention_backend="auto", attention_fn=None)` / `use_attention_backend(backend)` / `get_attention_backend()` | PyTorch slot cross-attention with Feature-wise Linear Modulation (FiLM), returning zeros on empty `context_states`, and thread-safe `ContextVar` helpers for `"auto"` attention backend selection (`"sdpa"` = Scaled Dot-Product Attention, `"ra"`, `"flash"`, `"cudnn"`, `"fa3"`, `"flex"`). |
| `ReversibleDKDBlock(...)` / `ReversibleDKDStack(...)` / `resolve_selective_reversible(...)` / `ResidualRollout(dim, num_steps=2, activation="ra_tanh")` | PyTorch Drift-Kick-Drift (DKD) split blocks and stacks supporting selective reversibility (`"none"`, `"all"`, `"every"`, `"even"`, `"odd"`, `"first_half"`, `"second_half"`, `"indices"`), per-block or whole-stack `torch.compile` (keeping `state_dict` keys unchanged, with CUDA Graph boundary guards for modes such as `mode="reduce-overhead"`), custom `kick_attn_fn`/`drift_ffn_fn` factories, and weight-tied residual rollout. |
| `OrthogonalHub(hub_size=128, dim=512, use_context_gate=False)` / `DirectiveGate(dim, init_logit_scale=4.0)` / `TwoTierMemory(dim, max_l1=16, max_l2=32, num_heads=4, ..., memory_dropout=0.0)` / `CausalMemoryTap(...)` / `assert_causal_no_leak(...)` | PyTorch $\mathrm{SO}(H)$ Cayley prefix hub, chunk cosine/FiLM gate, two-tier slot memory (`L1` + multi-head `L2`, `encode_keys(...)` / `cached_keys`, `query=None` past-only default, `chunks=None` cold-start, `memory_dropout`), combined `CausalMemoryTap` over past chunks, and `assert_causal_no_leak` helper for detecting future-token leakage. |
| `is_torch_available`, `is_numpy_available`, `is_triton_available`, `is_rust_simd_available`, `is_cubecl_available` | Runtime backend introspection helpers. |

See [Ra-Family Neural Architecture](../nn/README.md) for architecture diagrams, standalone NumPy behavior, and PyTorch pipeline examples.

### 2. Command-Line Interface (`luminamoon` / `luminamoon.cli`)

```bash
luminamoon --help                                  # Full API, parameter, and hardware reference
luminamoon --topic=optimizer --output=help.txt     # Save a specific section (all|optimizer|luminav|nn|data|migration|hardware)
luminamoon --output=help.txt --quiet --no-hardware # Write silently without hardware probe
luminamoon --version                               # Print package version and LuminaV MATH_VERSION
luminamoon doctor [--benchmark --steps 7 --measure-last 2]  # Live micro-probes + optional backend benchmark
luminamoon setup triton [--dry-run] [--yes]        # Opt-in OS-aware installer helper (triton-windows on Windows)
```

In Python, call `luminamoon.help(topic="all", *, output=None, print_out=True, include_hardware=True)` or `LuminaV.help()`.
