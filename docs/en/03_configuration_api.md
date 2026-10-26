# Configuration & API Reference

Complete class and function signatures, parameter reference tables, public methods, validation rules, and error handling for both the **LuminaV Optimizer** and the **LuminaMoon DataLoader**.

---

## Part I: LuminaV Optimizer API

### 1. Constructor Signature (`LuminaV`)

```python
from luminamoon import LuminaV

# Constructor Signature:
# LuminaV(
#     params: Iterable[Tensor | np.ndarray],
#     lr: float = 8.0e-4,
#     betas: tuple[float, float] = (0.9, 0.999),
#     eps: float = 1.0e-8,
#     weight_decay: float = 8.0e-2,
#     tau: float = 0.8,
#     lam: float = 0.3,
#     cautious: bool = True,
#     cautious_clamp_min: float = 0.5,
#     buffer: int | str = 2,
#     centered_defect: bool = False,
#     stochastic_rounding: bool = True,
#     bound: bool = True,
#     bound_type: int | str = "coordinate",
#     bound_ratio: float = 0.05,
#     master_weights: bool | int | str = "none",
#     execution: int | str = "auto",
#     fused_single_pass: bool = True,
#     ema_decay: float | str | None = "auto",
#     fsdp_sync: bool = False,
#     seed: int = 1337,
#     version: str | None = "1.5.3",
# )

# Standard Instantiation:
optimizer = LuminaV(
    model.parameters(),
    lr=8.0e-4,
    betas=(0.9, 0.999),
    eps=1.0e-8,
    weight_decay=8.0e-2,
    tau=0.8,
    lam=0.3,
    cautious=True,
    cautious_clamp_min=0.5,
    buffer=2,
    centered_defect=False,
    stochastic_rounding=True,
    bound=True,
    bound_type="coordinate",
    bound_ratio=0.05,
    master_weights="none",
    execution="auto",
    fused_single_pass=True,
    ema_decay="auto",
    fsdp_sync=False,
    seed=1337,
)
```

### 2. Optimizer Parameter Reference Table

| Parameter | Type | Default | Allowed Values | Technical Description |
| :--- | :--- | :--- | :--- | :--- |
| `params` | `Iterable` | *(Required)* | Tensor list, NumPy array list, or parameter group dicts | Model parameter weights to optimize. |
| `lr` | `float` | `8.0e-4` | $\ge 0.0$ | Primary learning rate. |
| `betas` | `Tuple[float, float]` | `(0.9, 0.999)` | $0.0 \le \beta < 1.0$ | First-order ($\beta_1$) and second-order ($\beta_2$) momentum decay coefficients. |
| `eps` | `float` | `1.0e-8` | $> 0.0$ | Numerical stability constant preventing division by zero. |
| `weight_decay` | `float` | `8.0e-2` | $\ge 0.0$ | Decoupled weight decay coefficient (AdamW style). |
| `tau` | `float` | `0.8` | $> 0.0$ | Temperature parameter for Dual-Buffer Convex Lookahead Momentum (CLM) scaling. |
| `lam` | `float` | `0.3` | $\ge 0.0$ | Triangle defect ratio weight ($\lambda$) for the Triangle Inequality Envelope in Single-Buffer TD-CLM (`buffer=1`). |
| `cautious` | `bool` | `True` | `True`, `False` | Enables cautious masking (updates only applied when $u \cdot g > 0$). |
| `cautious_clamp_min` | `float` | `0.5` | $0.0 < x \le 1.0$ | Floor clamp for cautious mask ratio to prevent over-amplification. |
| `buffer` | `int \| str` | `2` | `1`, `2`, `"single"`, `"dual"`, `"clm"`, `"tdclm"` | Momentum buffer mode: `2` (Dual-Buffer Convex Lookahead Momentum / CLM) or `1` (Single-Buffer Triangle-Defect CLM / TD-CLM). |
| `centered_defect` | `bool` | `False` | `True`, `False` | Centers innovation defect $(g - m)$ across both Dual-Buffer CLM and Single-Buffer TD-CLM. |
| `stochastic_rounding` | `bool` | `True` | `True`, `False` | Enables stateless 32-bit MurmurHash3 bitwise stochastic rounding for FP16/BF16. |
| `bound` | `bool` | `True` | `True`, `False` | Enables gradient step bounding. |
| `bound_type` | `int \| str` | `"coordinate"` | `1`, `2`, `"coordinate"`, `"radial"` | Bounding mode: `"coordinate"` (per-element) or `"radial"` (hyperspherical norm). |
| `bound_ratio` | `float` | `0.05` | $> 0.0$ | Maximum update step ratio relative to parameter norm (to disable bounding, set `bound=False`). |
| `master_weights` | `bool \| int \| str` | `"none"` | `"none"` / `0`, `"semi"` / `1`, `"full"` / `2` (`"fp32"`) | Mixed-precision master weight strategy: `"none"` (0, low-precision states), `"semi"` (1, FP32 master weights with 16-bit states), or `"full"` (2, FP32 master weights and states). |
| `execution` | `int \| str` | `"auto"` | `"auto"`, `"vulkan"`, `"simd"`, `"scalar"`, `"triton"`, `"single"` | Target hardware backend execution preference. |
| `fused_single_pass` | `bool` | `True` | `True`, `False` | Blends parameter update and EMA tracking into a single kernel pass. |
| `ema_decay` | `float \| str` | `"auto"` | `"auto"` or float $[0.0, 1.0)$ | EMA tracking coefficient for mask and bounding. `"auto"` sets `0.95` (buffer 1) or `0.80` (buffer 2). |
| `fsdp_sync` | `bool` | `False` | `True`, `False` | Enables global collective synchronization across distributed GPU ranks. |
| `seed` | `int` | `1337` | Integer | Base seed for per-parameter stochastic rounding PRNG generators. |
| `version` | `str \| None` | `"1.5.3"` | `"1.5.3"`, `"latest"`, `"default"` | Canonical LuminaV algorithm specification version lock. Specifications prior to `1.5.0` are deprecated. |

### 3. Optimizer Public Methods

1. **`step(closure=None, grads=None)`**: Executes a single optimization step. `grads` is required when operating on standalone NumPy arrays.
2. **`zero_grad(set_to_none=True)`**: Resets gradients of all optimized parameters.
3. **`state_dict()`**: Returns a clean, serializable dictionary with transient hardware buffers stripped.
4. **`load_state_dict(state_dict)`**: Restores optimizer state from a saved checkpoint.
5. **`detect_hardware()`**: Static method returning host CPU SIMD, Vulkan UMA, and Triton AOT capability metadata.

### 4. Parameter Validation & Typo Suggestions

If an unrecognized string option is passed, `LuminaV` computes the Levenshtein distance and raises an actionable `ValueError`:

```python
LuminaV(model.parameters(), bound_type="radal")
# ValueError: Invalid bound_type: 'radal'. Expected 'coordinate' or 'radial'. Did you mean 'radial'?
```

---

## Part II: LuminaMoon Data Engine API (`luminamoon.data`)

### 1. `FusedPipeline`

Fused multi-stage text normalization, heuristic filtering, deduplication, and PII masking executed in parallel via Rayon and native SIMD kernels:

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
| `normalize_newlines` | `bool` | `True` | Normalizes CRLF (`\r\n`) and lone CR (`\r`) sequences into canonical `\n`. |
| `filter_quality` | `bool` | `False` | Enables heuristic quality filtering rules (entropy, repetition, symbols, length). |
| `min_chars` | `int \| None` | `None` | Minimum string character length boundary required to retain a record. |
| `max_chars` | `int \| None` | `None` | Maximum string character length boundary (`None` = unbounded). |
| `min_words` | `int \| None` | `None` | Minimum whitespace-separated word count threshold. |
| `min_entropy` | `float \| None` | `None` | Minimum Shannon character entropy threshold to reject low-variety text. |
| `max_repetition_ratio` | `float \| None` | `None` | Maximum allowed repetitive n-gram fraction threshold. |
| `max_symbol_ratio` | `float \| None` | `None` | Maximum allowed symbol-to-alphanumeric ratio before dropping. |
| `min_byte_len` | `int \| None` | `None` | Minimum UTF-8 byte length boundary. |
| `max_byte_len` | `int \| None` | `None` | Maximum UTF-8 byte length boundary (`None` = unbounded). |
| `truncate_over_max` | `bool` | `False` | Truncates strings exceeding `max_byte_len` instead of rejecting them. |
| `json_extract_key` | `str \| None` | `None` | Extracts a specific JSON field from raw string records before processing. |
| `deduplicator` | `StreamDeduplicator \| None` | `None` | Active exact deduplication SwissTable instance with WAL persistence. |
| `commit_dedup` | `bool` | `False` | Automatically commits SwissTable deduplication state to WAL upon batch processing. |

---

### 2. `StringTensor`

Dual-layout string container providing zero-copy memory interoperability across CPU and GPU runtime engines:

```python
tensor = pipeline(["Sample text line 1", "Sample text line 2"])

# Methods & Properties
length: int = len(tensor)
first_item: str = tensor[0]

# Zero-Copy Data Conversions
polars_series = tensor.to_polars()
arrow_array = tensor.to_arrow()
offsets_cpu, payload_cpu = tensor.to_numpy_gpu()
# offsets_gpu, payload_gpu = tensor.to_torch_gpu(device="cuda")
```

| Method | Return Type | Description |
| :--- | :--- | :--- |
| `to_polars()` | `polars.Series` | Exports to a Polars Series using Apache Arrow C-Data Interface without copying byte buffers. |
| `to_arrow()` | `pyarrow.Array` | Exports to an Arrow LargeString array. |
| `to_numpy_gpu()` | `tuple[np.ndarray, np.ndarray]` | Exports flat `uint32` offset array and contiguous `uint8` ASCII/UTF-8 byte buffer. |
| `to_torch_gpu(device)` | `tuple[torch.Tensor, torch.Tensor]` | Transcribes contiguous offsets and payload directly into PyTorch CUDA/MPS device memory. |

---

### 3. `StateTracker` (128-Byte Ping-Pong Crash Recovery)

Crash-proof atomic metadata management with IEEE CRC32 verification and write-ahead deduplication logging:

```python
from pathlib import Path
from luminamoon.data import StateTracker

tracker = StateTracker(checkpoint_path="./cache/train.meta")

# Commit current state atomically
tracker.commit(
    epoch=1,
    batch_index=150,
    sample_index=4800,
    custom_bytes=b"loss=0.241",
    new_hashes=[0x12345678, 0x87654321],
)

# Load verified state
state = tracker.load()
# Access state fields: state.epoch, state.batch_index, state.sample_index, state.custom_bytes
```

---

### 4. Modular Preprocessing & Heuristic Quality Components

```python
from luminamoon.data import (
    TextCleaner,
    QualityFilter,
    StreamDeduplicator,
    MinHashDeduplicator,
    PIIMasker,
    data_cpu_isa,
)

# 1. Hardware ISA Inspection
isa: str = data_cpu_isa()  # 'AVX-512', 'AVX2', 'NEON', or 'SCALAR'

# 2. Text Normalization
cleaner = TextCleaner(trim_whitespace=True, normalize_newlines=True)
clean_line = cleaner.clean("  text\r\n")

# 3. Shannon Quality Filter
qfilter = QualityFilter(min_chars=10, min_entropy=2.5, max_symbol_ratio=0.2)
passed = qfilter.is_high_quality("High quality prose text with rich vocabulary.")

# 4. Stream Deduplication
dedup = StreamDeduplicator(wal_path="./cache/dedup.wal")
is_new = dedup.is_unique("Document text content")

# 5. Fuzzy MinHash Deduplication
minhash = MinHashDeduplicator(num_perm=64, threshold=0.8)
sig = minhash.compute_signature("Document text content")

# 6. PII Masking
masker = PIIMasker()
redacted = masker.mask("Token: sk-abcdef123456789")
```

---

### 5. PyTorch Resumable DataLoader Adapters

```python
from luminamoon.data import ResumablePipeline, create_resumable_collate

# 1. Pipeline Wrapper
pipe = ResumablePipeline(
    source=raw_data_iterable,
    fused_pipeline=pipeline,
    batch_size=64,
    prefetch_batches=4,
    checkpoint_path="./cache/train.meta",
)
for batch in pipe:
    # Model training step...
    pipe.commit()

# 2. PyTorch collate_fn Adapter
collate_fn = create_resumable_collate(
    fused_pipeline=pipeline,
    checkpoint_path="./cache/train.meta",
)
```
