# `string_tensor.rs` & `adapters.py`: Dual-Layout `StringTensor` & Async Resumable Pipelines

[`data/src/string_tensor.rs`](https://github.com/cloverx-id/luminamoon/blob/main/data/src/string_tensor.rs) and [`python/luminamoon/data/adapters.py`](https://github.com/cloverx-id/luminamoon/blob/main/python/luminamoon/data/adapters.py) provide zero-copy CPU/GPU string memory layouts and asynchronous background batch prefetching for PyTorch and standalone training loops.

---

## 1. Dual-Layout `StringTensor` (`src/string_tensor.rs`)

`StringTensor` bridges CPU string processing and GPU tensor ingestion via two complementary memory representations:

### 1.1 CPU Layout: 16-Byte Umbra / DuckDB `StringView16`
```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StringView16 {
    pub len: u32,
    pub prefix_or_inline: [u8; 4],
    pub suffix_or_offset: [u8; 8],
}
```
- **Inlined Strings ($\text{len} \le 12$ bytes)**: Stored completely inside the 16-byte struct (`prefix_or_inline` + `suffix_or_offset`) with zero heap pointer dereferences.
- **Pool-Referenced Strings ($\text{len} > 12$ bytes)**: Stores the first 4 bytes in `prefix_or_inline` (enabling single-register 32-bit prefix rejection in `filter_by_prefix` and `fast_eq`), alongside a 4-byte `offset` and 4-byte `buffer_id` into `Arc<Vec<u8>>`.

### 1.2 GPU & Arrow Layout: Contiguous Offsets + Payload
- `to_gpu_buffers()` emits a contiguous `uint32` offset array `[0, off_1, ..., off_N]` of length $N + 1$ and a flat `uint8` UTF-8 byte buffer.
- Directly exported to PyTorch (`to_torch_gpu(device)`), NumPy (`to_numpy_gpu()`), Apache Arrow (`to_arrow(zero_copy=True)` via `pa.StringArray.from_buffers`), and Polars (`to_polars()`).

---

## 2. Asynchronous Prefetching & PyTorch Integration (`adapters.py`)

1. **`ResumablePipeline`**:
   - Wraps any Python iterable or streaming dataset with single-pass `FusedPipeline` execution, `StateTracker` 128-byte Ping-Pong coordinate tracking, and background thread prefetching (`prefetch_batches > 0` via bounded `queue.Queue`).
   - Calling `pipe.commit()` atomically persists both the active `StateTracker` batch coordinate (`Slot A` / `Slot B`) and `StreamDeduplicator` WAL.
2. **`create_resumable_collate`**:
   - Generates a drop-in `collate_fn` for `torch.utils.data.DataLoader` that extracts text fields, runs `FusedPipeline` (or modular `TextCleaner` -> `QualityFilter` -> `StreamDeduplicator` -> `PIIMasker`), and returns either tokenized PyTorch tensors or a `StringTensor`.
