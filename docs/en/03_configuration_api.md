# Configuration & API Reference

Complete class and function signatures, parameter reference tables, public methods, validation rules, and error handling for both the **LuminaV Optimizer** and the **LuminaMoon DataLoader**.

---

## Part I: LuminaV Optimizer API

### 1. Constructor Signature (`LuminaV`)

```python
from luminamoon import LuminaV

optimizer = LuminaV(
    params,
    lr: float = 8.0e-4,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1.0e-8,
    weight_decay: float = 8.0e-2,
    tau: float = 0.8,
    lam: float = 0.3,
    cautious: bool = True,
    cautious_clamp_min: float = 0.5,
    buffer: Union[int, str] = 2,
    centered_defect: bool = False,
    stochastic_rounding: bool = True,
    bound: bool = True,
    bound_type: Union[int, str] = "coordinate",
    bound_ratio: float = 0.05,
    master_weights: Union[bool, int, str] = "none",
    execution: Union[int, str] = "auto",
    fused_single_pass: bool = True,
    ema_decay: Optional[Union[float, str]] = "auto",
    fsdp_sync: bool = False,
    seed: int = 1337,
)
```

### 2. Optimizer Parameter Reference Table

| Parameter | Type | Default | Allowed Values | Technical Description |
| :--- | :--- | :--- | :--- | :--- |
| `params` | `Iterable` | *(Required)* | Tensor list, NumPy array list, or parameter group dicts | Model parameter weights to optimize. |
| `lr` | `float` | `8.0e-4` | $> 0.0$ | Primary learning rate. |
| `betas` | `Tuple[float, float]` | `(0.9, 0.999)` | $0.0 \le \beta < 1.0$ | First-order ($\beta_1$) and second-order ($\beta_2$) momentum decay coefficients. |
| `eps` | `float` | `1.0e-8` | $> 0.0$ | Numerical stability constant preventing division by zero. |
| `weight_decay` | `float` | `8.0e-2` | $\ge 0.0$ | Decoupled weight decay coefficient (AdamW style). |
| `tau` | `float` | `0.8` | $> 0.0$ | Temperature parameter for CLM scaling (Dual-Buffer). |
| `lam` | `float` | `0.3` | $\ge 0.0$ | Temporal-difference defect penalty coefficient for Single-Buffer TD-CLM. |
| `cautious` | `bool` | `True` | `True`, `False` | Enables cautious masking (updates only applied when $u \cdot g > 0$). |
| `cautious_clamp_min` | `float` | `0.5` | $0.0 < x \le 1.0$ | Floor clamp for cautious mask ratio to prevent over-amplification. |
| `buffer` | `int \| str` | `2` | `1`, `2`, `"single"`, `"dual"`, `"clm"`, `"tdclm"` | Momentum buffer mode: `2` (Dual-Buffer CLM) or `1` (Single-Buffer TD-CLM). |
| `centered_defect` | `bool` | `False` | `True`, `False` | Centers temporal gradient difference $(g - m)$ across batch/tensor. |
| `stochastic_rounding` | `bool` | `True` | `True`, `False` | Enables hardware-independent PRNG stochastic rounding for FP16/BF16. |
| `bound` | `bool` | `True` | `True`, `False` | Enables gradient step bounding. |
| `bound_type` | `int \| str` | `"coordinate"` | `1`, `2`, `"coordinate"`, `"radial"` | Bounding mode: `"coordinate"` (per-element) or `"radial"` (hyperspherical norm). |
| `bound_ratio` | `float` | `0.05` | $\ge 0.0$ | Maximum update step ratio relative to parameter norm. `0.0` disables bounding. |
| `master_weights` | `bool \| int \| str` | `"none"` | `"none"` / `0`, `"fp32"` / `1`, `"pure_fp32"` / `2` | FP32 master weight strategy when training in FP16/BF16. |
| `execution` | `int \| str` | `"auto"` | `"auto"`, `"vulkan"`, `"simd"`, `"scalar"`, `"triton"`, `"single"` | Target hardware backend execution preference. |
| `fused_single_pass` | `bool` | `True` | `True`, `False` | Blends parameter update and EMA tracking into a single kernel pass. |
| `ema_decay` | `float \| str` | `"auto"` | `"auto"` or float $[0.0, 1.0)$ | EMA tracking coefficient for mask and bounding. `"auto"` sets `0.95` (buffer 1) or `0.80` (buffer 2). |
| `fsdp_sync` | `bool` | `False` | `True`, `False` | Enables global collective synchronization across distributed GPU ranks. |
| `seed` | `int` | `1337` | Integer | Base seed for per-parameter stochastic rounding PRNG generators. |

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

## Part II: LuminaMoon DataLoader API

### 1. `lm.stream(...)` & `DataStreamLoader`

```python
import luminamoon as lm

loader = lm.stream(
    sources: str | Path | Sequence[str | Path],
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = 64,
    preram_batches: int = 100,
    refill_trigger: int = 10,
    download_workers: int = 4,
    deduplicate: bool = True,
    auto_commit: bool = True,
    checkpoint_path: str | Path | None = None,
    json_extract_key: str | None = None,
    auth_token: str | None = None,
    tokenizer: Any = None,
    tokenizer_kwargs: Mapping[str, Any] | None = None,
    postprocess_fn: Callable[[Any], Any] | None = None,
    split: str | None = None,
    subset: str | None = None,
    **kwargs: Any,
)
```

| Parameter | Type | Default | Technical Description |
| :--- | :--- | :--- | :--- |
| `sources` | `str \| Path \| Sequence` | *(Required)* | Local file paths, `https://...` URLs, or `hf://datasets/<org>/<repo>[@rev]` URIs. |
| `columns` | `Sequence[str] \| None` | `None` | Column projection list. Multiple columns automatically route to `MultimodalParquetLoader`. |
| `batch_size` | `int` | `64` | Number of cleaned records or multimodal rows per yielded batch. |
| `preram_batches` | `int` | `100` | Maximum batches held in the background Pre-RAM watermark ring buffer. |
| `refill_trigger` | `int` | `10` | Refill delta ($\Delta$): background producer wakes when buffered batches $\le \text{preram}_{\text{batches}} - \text{refill}_{\text{trigger}}$. |
| `download_workers` | `int` | `4` | Number of concurrent Tokio HTTP 206 byte-range worker connections. |
| `deduplicate` | `bool` | `True` | Enables two-stage `xxh3_64` deduplication across staged and GPU-committed batches. |
| `auto_commit` | `bool` | `True` (`stream`) / `False` (`DataStreamLoader`) | Deferred auto-commit: persists batch $N$ to the 128B Ping-Pong `.meta` slot when batch $N+1$ is requested. |
| `checkpoint_path` | `str \| Path \| None` | `None` | Path to the 128B `.meta` file (defaults to `.luminamoon_cache/stream_<sha256>.meta`). |
| `parquet_column` | `str` | `"text"` | Target column name when streaming single-column Parquet files. |
| `json_extract_key` | `str \| None` | `None` | Extracts a specific string field from `.jsonl` lines in Rust before normalization. |
| `trim_whitespace` | `bool` | `True` | Trims leading/trailing Unicode whitespace in parallel via Rayon. |
| `normalize_newlines` | `bool` | `True` | Converts `\r\n` and `\r` to `\n` using SIMD `contains_byte_fast` pre-screening. |
| `min_byte_len` | `int` | `1` | Minimum UTF-8 byte length required to keep a record. |
| `max_byte_len` | `int` | `0` | Maximum UTF-8 byte length (`0` = unlimited). |
| `truncate_over_max` | `bool` | `False` | Truncates records exceeding `max_byte_len` on valid UTF-8 `is_char_boundary` points instead of dropping them. |
| `remote_chunk_bytes` | `int` | `262144` | HTTP 206 byte-range window size per worker (256 KB) for remote `.jsonl`/`.txt` streams. |
| `auth_token` | `str \| None` | `None` | HuggingFace Bearer token (auto-resolved from `os.environ`, Google Colab `userdata`, or Kaggle Secrets if `None`). |
| `tokenizer` | `Callable \| None` | `None` | Callable tokenizer (e.g., `AutoTokenizer`) invoked on each batch to populate `batch.tokens`. |
| `tokenizer_kwargs` | `Mapping \| None` | `None` | Keyword arguments passed to `tokenizer` (defaults to `{"padding": True, "truncation": True}`). |
| `postprocess_fn` | `Callable \| None` | `None` | Custom Python hook `fn(batch) -> batch` applied after tokenization. |
| `split` / `subset` | `str \| None` | `None` | Filters shard paths when auto-expanding a HuggingFace repository root URI. |

### 2. `lm.from_hf_dataset(...)`

Wraps any HuggingFace `datasets.Dataset`, `IterableDataset`, or Python iterable with GIL-free Rust SIMD + Rayon normalization and `xxh3_64` deduplication (`postprocess_and_dedup_records`):

```python
iterator = lm.from_hf_dataset(
    hf_dataset: Iterable[Any],
    *,
    column: str = "text",
    batch_size: int = 64,
    deduplicate: bool = True,
    trim_whitespace: bool = True,
    normalize_newlines: bool = True,
    min_byte_len: int = 1,
    max_byte_len: int = 0,
    truncate_over_max: bool = False,
    json_extract_key: str | None = None,
    tokenizer: Any = None,
    tokenizer_kwargs: Mapping[str, Any] | None = None,
    postprocess_fn: Callable[[Any], Any] | None = None,
) -> Iterator[DataBatch]
```

### 3. `lm.push_to_hub(...)` & `lm.serialize_table_to_bytes(...)`

Serializes `DataBatch`, `MultimodalBatch`, `dict[str, list]`, or `list[dict]` in Rust (`py.detach()`) to `.parquet` (with HuggingFace Dataset Viewer `struct<bytes: binary, path: utf8>` image compatibility), `.jsonl`, or `.json`, and uploads to the HuggingFace Hub:

```python
uri = lm.push_to_hub(
    repo: str,
    data: Any,
    path_in_repo: str = "data/part-00000.parquet",
    *,
    columns: Sequence[str] | None = None,
    rename: Mapping[str, str] | None = None,
    private: bool = True,
    revision: str = "main",
    commit_message: str = "Upload dataset shard via LuminaMoon",
    hf_image_struct: bool = True,
    token: str | None = None,
    username: str | None = None,
    org: str | None = None,
    endpoint: str | None = None,
) -> str
```

### 4. Batch Objects, Coordinates & Utility Functions

- **`DataBatch`**:
  - Properties: `batch_id: int`, `records: list[str]`, `tokens: Any`, `next_coord: StreamCoordinate`, `is_eof: bool`.
  - Methods: `batch.commit() -> StreamCoordinate`, `batch.to_dict() -> dict[str, list[str]]`, `batch.push_to_hub(repo, path_in_repo=...) -> str`, `len(batch)`, `batch[idx]`, `batch["text"]`, `batch["tokens"]`.
- **`MultimodalBatch`**:
  - Properties: `batch_id: int`, `num_rows: int`, `tokens: Any`, `next_coord: StreamCoordinate`, `is_eof: bool`.
  - Methods: `batch.column_names() -> list[str]`, `batch.column(name) -> list[Any]`, `batch.to_dict() -> dict[str, list[Any]]`, `batch.push_to_hub(repo, path_in_repo=...) -> str`, `batch[col_name]`, `batch[row_idx]`, `batch[slice]`.
- **`StreamCoordinate`**:
  - Fields: `shard_id`, `byte_offset`, `row_group_idx`, `page_idx`, `row_offset_in_page`, `global_row_idx`.
  - Methods: `coord.to_bytes() -> bytes` (32-byte little-endian), `StreamCoordinate.from_bytes(raw) -> StreamCoordinate`.
- **`lm.expand_hf_sources(sources, *, split=None, subset=None, token=None, endpoint=None) -> list[str]`**:
  - Auto-discovers `.parquet` and `.jsonl` shards inside `hf://datasets/...` repositories (with RFC 5988 `Link` pagination).
- **`lm.resolve_hf_secrets(token=None, username=None, org=None, endpoint=None) -> dict[str, str | None]`**:
  - Resolves HuggingFace credentials from explicit arguments, `os.environ`, Google Colab `userdata`, or Kaggle `UserSecretsClient`.
- **`lm.dataloader_cpu_isa() -> str`**:
  - Returns the active CPU SIMD byte-scanning instruction set (`"AVX-512"`, `"AVX2"`, `"NEON"`, or `"SCALAR"`).
