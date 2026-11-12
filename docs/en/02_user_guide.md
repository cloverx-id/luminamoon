# User Guide

User guide for the **LuminaV Optimizer** (buffer modes, mixed precision, step bounding, checkpointing, distributed training) and the **LuminaMoon Data Engine** (parallel text preprocessing, `StringTensor`, deduplication, and resumable state tracking).

## 1. Selecting Optimizer Buffer Modes (`2`, `1`, `"T2"`, `"T1"`)

LuminaV offers four optimization modes controlled via the `buffer` argument:

```python
# Mode 1: Dual-Buffer CLM (Default)
optimizer = LuminaV(model.parameters(), buffer=2)

# Mode 2: Single-Buffer TD-CLM (Memory-Efficient)
optimizer = LuminaV(model.parameters(), buffer=1)

# Mode 3: 2-Buffer Triangle Family Mode (T2, tanh-free)
optimizer = LuminaV(model.parameters(), buffer="T2")

# Mode 4: 1-Buffer Triangle Family Mode (T1, tanh-free, 50% Memory Savings)
optimizer = LuminaV(model.parameters(), buffer="T1")
```

### Technical Comparison

| Characteristic | Dual-Buffer CLM (`buffer=2`) | Single-Buffer TD-CLM (`buffer=1`) |
| :--- | :--- | :--- |
| **State Buffers** | Maintains both `exp_avg` ($m$) and `exp_avg_sq` ($v$). | Maintains only `exp_avg` ($m$). |
| **Memory Footprint** | Identical to AdamW ($2 \times$ parameter size). | **50% state memory savings** ($1 \times$ parameter size). |
| **Scaling Mechanism** | Computes $\sigma_t = \sqrt{v_t} \cdot c_1 + c_2$ with two-step lookahead bias correction ($\hat{\beta}_{1,t} = 1 - \beta_1^{t+1}$) and Padé $[13/6]$ $\tanh$ bounding. | Computes Triangle-Defect Envelope $\mathcal{E}_t = \beta_1^2 \lvert m_{t-1} \rvert + (1-\beta_1^2)\lvert g_t \rvert + \lambda \lvert d_t \rvert + \epsilon_{\text{eff}}$ via the Triangle Inequality with $\lvert u_t \rvert < 1$. |
| **Typical Use Case** | Pre-training Transformers, large foundation models, and Vision models (ViT/CNN) where second-moment variance tracking is used. | LLM fine-tuning (LoRA / Full), memory-constrained GPUs, edge hardware, and single-buffer training. |

For tanh-free execution (**Triangle Family / T-Family**), set `buffer="T2"` (2-buffer state with gradient-momentum residual $d_t = g_t - m _ {t-1}$ and Triangle Inequality envelope $\mathcal{E} _ t^{(\text{T2})} = w_1 |m _ {t-1}| + w_2 |g_t| + 0.5\lambda |d_t| + 0.5\lambda \sigma_t + \epsilon _ {\text{eff}}$) or `buffer="T1"` (1-buffer state with Triangle-Defect denominator $\mathcal{E} _ t^{(\text{T1})}$ and rational update bounds).

---

## 2. Mixed-Precision & Stochastic Rounding

LuminaV natively supports low-precision parameter updates (`torch.float16` and `torch.bfloat16`) with internal Bitwise-BF16 state promotion and stochastic rounding.

### Master Weights Configuration

The `master_weights` argument controls how the optimizer manages FP32 parameter copies:

- `"none"` or `0` (`False`): No master weights. Low-precision parameters and state accumulators are updated directly using *Stochastic Rounding* (with 16-bit states stored using Bitwise-BF16 layout when parameters are `float16`).
- `"semi"` or `1` (aliases `"hybrid"`, `"half"`): Maintains FP32 master weights for FP16 and BF16 parameters while keeping state accumulators (`exp_avg`, `exp_avg_sq`) in 16-bit precision to conserve memory.
- `"full"` or `2` (`True`, alias `"fp32"`): Master weights and internal state accumulators (`exp_avg`, `exp_avg_sq`) are maintained in full FP32.

```python
optimizer = LuminaV(
    model.parameters(),
    lr=1e-3,
    master_weights="full",  # Options: 'none', 'semi', or 'full' (alias 'fp32')
    stochastic_rounding=True,
)
```

### Stochastic Rounding (SR)

When parameters or momentum accumulators are stored in low precision, standard deterministic rounding (Round-to-Nearest-Even / RNE) can cause small gradient updates to round to zero (*gradient stagnation*).

LuminaV integrates a stateless 32-bit MurmurHash3 bit-avalanche PRNG that performs *Stochastic Rounding* deterministically on each step:

- The probability of rounding up is proportional to the sub-representation numerical fraction.
- Prevents parameter stagnation without host-device synchronization.

### FP16 Training & `GradScaler` (Language & Vision Models)

Although LuminaV (`v1.5.4+`) protects its internal 16-bit state buffers (`exp_avg`, `exp_avg_sq`) from underflow by storing `float16` states in a Bitwise-BF16 layout (8-bit exponent), this protection only applies **inside** `optimizer.step()`.

The forward pass and `loss.backward()` across external model layers run outside the optimizer boundary:

- **`torch.float16` (5-bit exponent)**: Normal values underflow below $2^{-14} \approx 6.10 \times 10^{-5}$ (and subnormals flush to zero below $2^{-24} \approx 5.96 \times 10^{-8}$), while values above $65504$ overflow to `Inf`. When training language models (Transformers/LLMs) or image/vision models (ViT, CNN, Diffusion) in `float16`, layer-to-layer gradients—such as gradients averaged across spatial $(B \times H \times W)$ dimensions or deep residual blocks—can underflow to `0.0` or overflow during `loss.backward()` before reaching the optimizer. Enable `torch.amp.GradScaler` when training in `float16` so gradients stay within the representable `float16` range during backpropagation.
- **`torch.bfloat16` (8-bit exponent)**: Provides the same exponent dynamic range as `float32` throughout both the model graph and the optimizer, so `GradScaler` is not needed.

```python
scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

optimizer.zero_grad(set_to_none=True)
with torch.amp.autocast("cuda", dtype=dtype):
    loss = criterion(model(inputs), targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 3. Parameter Update Bounding

To limit large update steps and loss spikes, LuminaV includes parameter update bounding:

```python
optimizer = LuminaV(
    model.parameters(),
    bound=True,
    bound_type="coordinate",  # Options: 'coordinate' or 'radial'
    bound_ratio=0.05,         # Maximum update step is 5% of parameter norm
)
```

### Bounding Modes: Coordinate vs Radial

#### 1. Coordinate Bounding (`bound_type="coordinate"` or `2`)
Bounds each element independently using a hyperbolic tangent envelope:

$$
B_i = \mathrm{bound}_{\text{ratio}} \cdot \sqrt{\theta_i^2 + 10^{-4}}, \qquad \Delta\theta_i = B_i \cdot \tanh\left(\frac{\Delta\theta_i}{B_i}\right)
$$

Limits sparse outliers and large gradients in embedding layers.

#### 2. Radial Bounding (`bound_type="radial"` or `1`)
Bounds the update vector norm relative to the parameter $L_2$ norm (AGC-style relative norm clipping):

$$
R = \mathrm{bound}_{\text{ratio}} \cdot \max(\lVert \theta \rVert_2, 1.0), \qquad r = \frac{\eta \lVert \Delta\theta \rVert_2}{R}, \qquad \mathrm{scale} = \frac{\tanh(r)}{\max(r, 10^{-8})}
$$

Preserves the direction of the update vector without angular distortion.

---

## 4. Cautious Alignment Masking (`cautious=True`)

When `cautious=True` is enabled, the optimizer only updates parameters whose direction aligns with the current gradient:

$$
\mathrm{mask}_i = \mathbb{I}(u_i \cdot g_i > 0), \qquad \bar{m} = \max\left(\mathrm{clamp}_{\text{min}}, \frac{1}{N}\sum_{i=1}^N \mathrm{mask}_i\right), \qquad \Delta\theta_i = u_i \cdot \mathrm{mask}_i \cdot \frac{1}{\bar{m}}
$$

Dynamic scaling $\frac{1}{\bar{m}}$ maintains effective update magnitude when a fraction of parameters is masked out.

---

## 5. Fused Single-Pass Execution

By default, `fused_single_pass=True` is enabled:

- Starting from step 2, cautious mask reduction and radial scale tracking are updated asynchronously via Exponential Moving Average (EMA).
- Avoids a separate pre-step global reduction pass, reducing global memory read/write traffic compared to two-pass execution.

---

## 6. Contiguous Bucket Fusion, Bounded Chunking & JIT Pre-Warmup

For GPU training on architectures with many parameter tensors (Transformers, ConvNets, YOLO, and multi-layer LoRA adapters), LuminaV provides opt-in contiguous memory coalescing and initialization-time kernel compilation:

```python
optimizer = LuminaV(
    model.parameters(),
    lr=8e-4,
    fuse_tensors=True,    # Coalesces parameters into flat 1D memory buffers
    bucket_cap_mb=500.0,  # Caps maximum contiguous chunk allocation to 500 MB
    prewarm_jit=True,     # Pre-compiles Triton launch kernels during initialization
)
```

### Technical Capabilities & Invariants

1. **Contiguous Bucket Fusion (`fuse_tensors=True`)**:
   - Groups parameters sharing the same `(param_group, device, dtype)` into flat 1D memory buffers (`flat_p`, `flat_g`, `flat_m`, `flat_v`, `flat_mp`) with zero-copy tensor views (`flat_*[offset : offset + n].view(p.shape)`).
   - Updates an entire bucket in a single fused GPU kernel dispatch instead of launching separate kernels across individual parameter tensors, reducing per-step kernel launch overhead.
   - **Pointer Persistence**: In-place `zero_grad(set_to_none=True)` resets gradients via `flat_g.zero_()` and restores tensor view pointers without redundant GPU reallocations.
   - **Workload Guidance**: Contiguous fusion is designed for **Vision and Image models** (CNN, ViT, ConvNeXt) with many small fragmented layers where kernel launch overhead dominates. It is **not recommended for Large Language Models (LLMs)** or distributed FSDP sharding, where individual matrix blocks are already large enough to saturate the GPU and contiguous coalescing increases transient peak memory overhead.

2. **Bounded Bucket Chunking (`bucket_cap_mb=500.0`)**:
   - Automatically partitions large parameter collections into deterministic bucket chunks capped at `bucket_cap_mb` (default 500 MB). Huge individual layers exceeding `bucket_cap_mb` (e.g. giant vocabulary embedding matrices) are isolated into dedicated single-tensor buckets.
   - **Sequential In-Place Migration**: Parameter storage is copied into the contiguous chunk slice-by-slice with immediate source buffer dereferencing (`del old_data`, `del old_grad`) and garbage collection (`gc.collect()`). Peak transient initialization memory overhead is constrained to at most $\le 500\text{ MB}$.

3. **Triton JIT Pre-Warmup (`prewarm_jit=True`)**:
   - Pre-compiles active Triton launch kernels during optimizer `__init__` using 64-element dummy buffers (`lr=0.0`) across Step 1 (two-pass) and Step 2+ (fused single-pass + post-step EMA) paths.
   - Compiles kernels ahead of the first training step.
   - **Decoupled State Lookup**: State isolation via `self.state.get(p)` ensures that dummy warmup buffers do not pollute user-visible state dictionaries.

4. **Protective Configuration Warnings**:
   - **Workload Guidance Warning on Contiguous Fusion**: When `fuse_tensors=True` is enabled, LuminaV emits an advisory `UserWarning` clarifying that contiguous bucket coalescing is designed for fragmented Vision architectures (CNN, ViT) and is not recommended for large LLMs or distributed FSDP training.
   - **Un-fused Bucket Cap Warning**: If `bucket_cap_mb` is explicitly configured while `fuse_tensors=False`, LuminaV emits a `UserWarning` clarifying that chunk bounding only applies to coalesced buffers.
   - **Distributed FSDP Guard**: If `fsdp_sync=True` is enabled when `torch.distributed` is uninitialized or world size is 1, a warning is emitted and the optimizer falls back to local reduction mode.

---

## 7. State Checkpointing (`state_dict` & `load_state_dict`)

LuminaV provides serialization support for PyTorch and standalone NumPy:

```python
checkpoint = {
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
}
torch.save(checkpoint, "checkpoint.pt")

loaded_checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(loaded_checkpoint["model_state"])
optimizer.load_state_dict(loaded_checkpoint["optimizer_state"])
```

> [!NOTE]
> Transient scratch buffers (such as `p_contig`, `_plan`, and `_contig_step`) are automatically stripped during `state_dict()` serialization to produce clean, compact checkpoints.

---

## 8. Distributed Training & FSDP / DDP

In distributed environments such as Fully Sharded Data Parallel (FSDP) or Distributed Data Parallel (DDP):

```python
optimizer = LuminaV(
    model.parameters(),
    lr=1e-3,
    fsdp_sync=True,  # Activates collective synchronization across GPU ranks
    cautious=True,
    bound=True,
    bound_type="radial",
)
```

When `fsdp_sync=True`:

- Global reductions for `mask.sum()`, $\|u\|_2^2$, and $\|\theta\|_2^2$ are batched into a single collective `all_reduce` call.
- Empty shards across ranks are handled deterministically without stalling FSDP collective synchronization barriers.

---

## 9. Fused Preprocessing Pipeline (`FusedPipeline` & `StringTensor`)

The `luminamoon.data` library combines text cleaning, heuristic filtering, deduplication, and PII masking into a single parallel pass in Rust, returning zero-copy `StringTensor` buffers for CPU or GPU ingestion:

```python
from luminamoon.data import FusedPipeline, StreamDeduplicator

pipeline = FusedPipeline(
    trim_whitespace=True,           # AVX-512 / AVX2 / NEON SIMD whitespace trimming
    normalize_newlines=True,        # CRLF to LF newline conversion
    filter_quality=True,            # Enable heuristic quality evaluation
    min_chars=10,                   # Minimum character length boundary
    min_entropy=2.0,                # Shannon character entropy threshold
    max_repetition_ratio=0.3,       # Reject degenerate repetitive patterns
    deduplicator=StreamDeduplicator(),  # XXH3-64 exact deduplication
    commit_dedup=True,
)

# Process batch into dual-layout StringTensor
raw_texts = ["  Sample training text line\r\n", "Repeated text block...", "API_KEY=sk-abc123456789"]
tensor = pipeline(raw_texts)

# Export without memory duplication
polars_series = tensor.to_polars()
arrow_array = tensor.to_arrow()
# offsets_gpu, payload_gpu = tensor.to_torch_gpu(device="cuda")
```

---

## 10. Resumable State Tracking & 128-Byte Ping-Pong Checkpoints

To support deterministic data resumption across distributed training, spot-instance preemption, and node restarts, `StateTracker` provides 128-byte Dual-Slot Ping-Pong metadata with CRC32 integrity validation and write-ahead deduplication logs:

```python
from luminamoon.data import StateTracker, ResumablePipeline, FusedPipeline

tracker = StateTracker("./cache/train_checkpoint.meta")

# Stage in-flight coordinate and atomically commit to alternating Slot A / Slot B
tracker.update(
    shard_id=1,
    byte_offset=4096,
    row_offset_in_page=32,
    global_row_idx=28800,
    batch_id=450,
)
coord = tracker.commit(batch_id=450)

# Resume on restart
resume_coord = tracker.read()
print(f"Resuming from Shard {resume_coord.shard_id}, Global Row {resume_coord.global_row_idx}")
```

For PyTorch `torch.utils.data.DataLoader` integration:

```python
from luminamoon.data import create_resumable_collate

collate_fn = create_resumable_collate(
    fused_pipeline=pipeline,
    return_string_tensor=True,
)
# PyTorch DataLoader automatically receives zero-copy StringTensor batches
```

---

## 11. Fuzzy Deduplication & PII Sanitization (`MinHashDeduplicator` & `PIIMasker`)

Beyond exact 64-bit hashing, `luminamoon.data` provides MinHash LSH fuzzy shingling for near-duplicate document filtering:

```python
from luminamoon.data import MinHashDeduplicator, PIIMasker, jaccard_similarity

# 1. Fuzzy near-duplicate shingling with permuted XXH3-64 hashes
minhash = MinHashDeduplicator(threshold=0.8, k_shingle=5, num_perm=64)
doc_a = "Numerical optimization library with SIMD vectorization."
doc_b = "Numerical optimization library with AVX vectorization."

is_unique_a = minhash.is_unique(doc_a)
similarity = jaccard_similarity(doc_a, doc_b, k_shingle=5, num_perm=64)
print(f"Estimated Jaccard Similarity: {similarity:.2f}")

# 2. Parallel regex PII masking
masker = PIIMasker()
cleaned = masker.mask("Contact support at admin@example.com with key sk-123456789")
print(cleaned)  # "Contact support at [EMAIL] with key [API_KEY]"
```
