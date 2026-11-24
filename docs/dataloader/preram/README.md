# StringTensor, Sequence Packing & Streaming Pipelines

[`data/src/string_tensor.rs`](https://github.com/cloverx-id/luminamoon/blob/main/data/src/string_tensor.rs) and the pipeline modules under `python/luminamoon/data/` (`loader.py`, `batch.py`, `packing.py`, `collate.py`, `pipeline.py`, `ring_buffer.py`) provide CPU/GPU string layouts, document-isolated sequence packing, and resumable streaming pipelines.

---

## 1. Dual-Layout `StringTensor` (`src/string_tensor.rs`)

`StringTensor` bridges CPU string processing, Apache Arrow C-Data Interface export, and flat GPU tensor buffers:

### 1.1 CPU Layout: 16-Byte Arrow `Utf8View`-Aligned `StringView16`
```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StringView16 {
    pub(crate) len: u32,
    pub(crate) prefix_or_inline: [u8; 4],
    pub(crate) suffix_or_offset: [u8; 8],
}
```

- **Inlined Strings ($\text{len} \le 12$ bytes)**: Stored completely inside the 16-byte struct (`prefix_or_inline` + `suffix_or_offset`) without heap pointer dereferences.
- **Pool-Referenced Strings ($\text{len} > 12$ bytes)**: Stores the first 4 bytes in `prefix_or_inline` (enabling single-register 32-bit prefix rejection in `filter_by_prefix` and `fast_eq`), alongside a 4-byte `buffer_id` and 4-byte `offset` matching Arrow `Utf8View` layout.

### 1.2 Arrow PyCapsule & GPU Contiguous Layout
- `__arrow_c_array__()` exports the `Utf8View` array via Apache Arrow C-Data Interface PyCapsules (`to_arrow()`, `to_polars()`).
- `to_gpu_buffers()` emits a contiguous `uint32` offset array `[0, off_1, ..., off_N]` of length $N + 1$ and a flat `uint8` UTF-8 byte buffer (`to_numpy_gpu()`, `to_torch_gpu(device)`).

---

## 2. Streaming Pipelines, Sequence Packing & Token Buffering

1. **`DataPipeline`, `SequencePacker`, and `TokenBatch` (`loader.py`, `packing.py`, `batch.py`)**:
   - Primary v2.0 entrypoint supporting multi-source mixing, rank/worker sharding, document curation, document-isolated packing (`"best_fit"`, `"concat"`, `"pad"`), resumable `.state.json` + $O(\Delta)$ `.dedup` WAL checkpointing (`commit()`), and multi-epoch iteration (`reset()`, `set_epoch()`).
   - `TokenBatch` provides `input_ids`, `labels`, `doc_ids`, `position_ids`, `cu_seqlens`, `attention_mask`, `causal_doc_mask()` (preventing cross-document attention leakage), `as_xy()`, and `.to(device)`.
   - Long documents split across multiple chunks only record `source_refs` on their final chunk (`is_final=True`), preventing tail loss on `commit()` + resume.
2. **`ResumablePipeline` (`pipeline.py`)**:
   - Low-level streaming text pipeline with `StateTracker` 256-byte `LMDL v2` Ping-Pong coordinate tracking, background thread prefetching (`prefetch_batches > 0`), and `state_dict()` / `load_state_dict()`.
3. **`create_resumable_collate` (`collate.py`)**:
   - Generates a `collate_fn` for `torch.utils.data.DataLoader` supporting `FusedPipeline`, modular curation stages, optional custom `filter_fn`, tokenization, or `StringTensor` output.
4. **`TokenRingBuffer` (`ring_buffer.py`)**:
   - Thread-safe circular `int64` token buffer supporting `push`, `pop_tokens`, `next_batch`, `pop_batch`, `wait_for_prefill`, and optional `pin_memory=True` for streaming `(input_ids, labels)` batches.
