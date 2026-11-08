from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._ffi import suggest_closest_matches
from .checkpoint import StateTracker, StreamCoordinate
from .cleaner import TextCleaner
from .dedup import StreamDeduplicator
from .masker import PIIMasker
from .quality import QualityFilter
from .string_tensor import StringTensor

if TYPE_CHECKING:
    from .fused import FusedPipeline


def _extract_text(item: Any, text_key: str = "text") -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    if isinstance(item, Mapping):
        if text_key not in item:
            available = [str(k) for k in item.keys()]
            suggestions = suggest_closest_matches(text_key, available, 3)
            hint = (
                f" Did you mean '{suggestions[0]}'?"
                if len(suggestions) == 1
                else (f" Did you mean one of {suggestions}?" if suggestions else "")
            )
            raise KeyError(
                f"Target column '{text_key}' not found in dataset record.{hint} Available columns: {available}"
            )
        val = item.get(text_key)
        return str(val) if val is not None else None
    if isinstance(item, (list, tuple)) and len(item) > 0:
        return str(item[0])
    return str(item) if item is not None else None


def create_resumable_collate(
    cleaner: TextCleaner | None = None,
    quality_filter: QualityFilter | None = None,
    deduplicator: StreamDeduplicator | None = None,
    masker: PIIMasker | None = None,
    fused_pipeline: FusedPipeline | None = None,
    tokenizer: Any | None = None,
    text_key: str = "text",
    max_length: int | None = None,
    padding: bool | str = True,
    truncation: bool = True,
    return_tensors: str = "pt",
    return_string_tensor: bool = False,
) -> Callable[[list[Any]], Any]:
    """Creates a high-performance `collate_fn` compatible with PyTorch DataLoader.

    Pipeline execution order:
    1. Text extraction from dicts, strings, or tuples.
    2. High-speed cleaning: Fused single-pass Rust kernel (`fused_pipeline`) or modular stages.
    3. PII sanitization (`PIIMasker`).
    4. Tokenization into PyTorch tensors, or direct zero-copy StringTensor return.
    """

    def collate_fn(batch: list[Any]) -> Any:
        raw_texts: list[str] = []
        for item in batch:
            t = _extract_text(item, text_key=text_key)
            if t is not None:
                raw_texts.append(t)

        if not raw_texts:
            return {} if tokenizer else ([] if not return_string_tensor else StringTensor([]))

        if fused_pipeline is not None:
            # High-throughput fused path in Rust (AVX-512 + Quality + SwissTable Dedup in 1 pass)
            st = fused_pipeline(raw_texts)
            texts = st.to_list()
            if masker and texts:
                texts = masker.mask_batch(texts)
        else:
            # 1. Clean
            texts = cleaner.clean_batch(raw_texts) if cleaner else raw_texts

            # 2. Quality Filter
            if quality_filter and texts:
                texts = quality_filter.filter_batch(texts)

            # 3. Dedup
            if deduplicator and texts:
                texts = deduplicator.filter_batch(texts)

            # 4. Masker
            if masker and texts:
                texts = masker.mask_batch(texts)

        if not texts:
            if tokenizer is not None:
                try:
                    return tokenizer([], padding=padding, truncation=truncation, return_tensors=return_tensors)
                except Exception:
                    return {}
            return [] if not return_string_tensor else StringTensor([])

        # 5. Tokenize
        if tokenizer is not None:
            if padding and getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            kwargs: dict[str, Any] = {
                "padding": padding,
                "truncation": truncation,
                "return_tensors": return_tensors,
            }
            if max_length is not None:
                kwargs["max_length"] = max_length
            return tokenizer(texts, **kwargs)

        if return_string_tensor:
            return StringTensor(texts)

        return texts

    return collate_fn


class ResumablePipeline:
    """A streaming data pipeline that wraps any dataset with deterministic resume and state tracking.

    Supports asynchronous background prefetching (`prefetch_batches`) to eliminate GPU compute stalls,
    and single-pass Rust fused preprocessing (`fused_pipeline`) for > 400,000 records/sec throughput.
    """

    def __init__(
        self,
        source: Iterable[Any],
        checkpoint_path: str | Path | None = None,
        cleaner: TextCleaner | None = None,
        quality_filter: QualityFilter | None = None,
        deduplicator: StreamDeduplicator | None = None,
        masker: PIIMasker | None = None,
        fused_pipeline: FusedPipeline | None = None,
        text_key: str = "text",
        batch_size: int = 64,
        prefetch_batches: int = 0,
        return_string_tensor: bool = False,
    ) -> None:
        self.source = source
        self.cleaner = cleaner
        self.quality_filter = quality_filter
        self.deduplicator = deduplicator
        self.masker = masker
        self.fused_pipeline = fused_pipeline
        self.text_key = text_key
        self.batch_size = max(1, batch_size)
        self.prefetch_batches = max(0, prefetch_batches)
        self.return_string_tensor = return_string_tensor

        self.tracker = StateTracker(checkpoint_path) if checkpoint_path else None
        self._current_shard = 0
        self._current_row = 0
        self._emitted_samples = 0
        self._batch_id = 0
        self._active_batch_id = 0

    @property
    def start_coordinate(self) -> StreamCoordinate | None:
        """Returns the resume point read from the on-disk checkpoint."""
        return self.tracker.read() if self.tracker else None

    def _generate_batches(self) -> Iterator[tuple[list[str] | StringTensor, int]]:
        resume_coord = self.start_coordinate
        target_shard = resume_coord.shard_id if resume_coord else 0
        target_row = resume_coord.row_offset_in_page if resume_coord else 0
        if resume_coord is not None:
            self._current_shard = resume_coord.shard_id
            self._emitted_samples = resume_coord.global_row_idx
            if self.tracker is not None:
                self._batch_id = int(self.tracker.committed_batch_id)
                self._active_batch_id = self._batch_id

        buffer: list[str] = []

        if self.fused_pipeline is not None:
            # Fused execution path via Rust Rayon
            raw_chunk: list[str] = []
            chunk_size = self.batch_size * 2

            for idx, item in enumerate(self.source):
                if idx < target_row:
                    continue

                raw = _extract_text(item, self.text_key)
                if raw is not None:
                    raw_chunk.append(raw)
                    self._current_row = idx

                if len(raw_chunk) >= chunk_size:
                    st = self.fused_pipeline(raw_chunk)
                    cleaned_items = st.to_list()
                    raw_chunk.clear()

                    for c in cleaned_items:
                        if self.masker:
                            c = self.masker.mask(c)
                        buffer.append(c)
                        self._emitted_samples += 1
                        if len(buffer) >= self.batch_size:
                            self._batch_id += 1
                            if self.tracker:
                                self.tracker.update(
                                    shard_id=self._current_shard,
                                    row_offset_in_page=self._current_row + 1,
                                    global_row_idx=self._emitted_samples,
                                    batch_id=self._batch_id,
                                )
                            yield (StringTensor(buffer) if self.return_string_tensor else list(buffer), self._batch_id)
                            buffer.clear()

            if raw_chunk:
                st = self.fused_pipeline(raw_chunk)
                cleaned_items = st.to_list()
                for c in cleaned_items:
                    if self.masker:
                        c = self.masker.mask(c)
                    buffer.append(c)
                    self._emitted_samples += 1
                    if len(buffer) >= self.batch_size:
                        self._batch_id += 1
                        if self.tracker:
                            self.tracker.update(
                                shard_id=self._current_shard,
                                row_offset_in_page=self._current_row + 1,
                                global_row_idx=self._emitted_samples,
                                batch_id=self._batch_id,
                            )
                        yield (StringTensor(buffer) if self.return_string_tensor else list(buffer), self._batch_id)
                        buffer.clear()
        else:
            # Modular step-by-step path
            for idx, item in enumerate(self.source):
                if idx < target_row:
                    continue

                raw = _extract_text(item, self.text_key)
                if raw is None:
                    continue

                clean = self.cleaner.clean_single(raw) if self.cleaner else raw
                if clean is None:
                    continue

                if self.quality_filter and not self.quality_filter.is_high_quality(clean):
                    continue

                if self.deduplicator and not self.deduplicator.is_unique(clean):
                    continue

                if self.masker:
                    clean = self.masker.mask(clean)

                buffer.append(clean)
                self._current_row = idx
                self._emitted_samples += 1

                if len(buffer) >= self.batch_size:
                    self._batch_id += 1
                    if self.tracker:
                        self.tracker.update(
                            shard_id=self._current_shard,
                            row_offset_in_page=self._current_row + 1,
                            global_row_idx=self._emitted_samples,
                            batch_id=self._batch_id,
                        )
                    yield (StringTensor(buffer) if self.return_string_tensor else list(buffer), self._batch_id)
                    buffer.clear()

        if buffer:
            self._batch_id += 1
            if self.tracker:
                self.tracker.update(
                    shard_id=self._current_shard,
                    row_offset_in_page=self._current_row + 1,
                    global_row_idx=self._emitted_samples,
                    batch_id=self._batch_id,
                )
            yield (StringTensor(buffer) if self.return_string_tensor else list(buffer), self._batch_id)

    def __iter__(self) -> Iterator[list[str] | StringTensor]:
        if self.prefetch_batches <= 0:
            for batch_data, bid in self._generate_batches():
                self._active_batch_id = bid
                yield batch_data
            return

        import queue
        import threading

        q: queue.Queue[Any] = queue.Queue(maxsize=self.prefetch_batches)
        stop_event = threading.Event()
        sentinel = object()

        def _worker() -> None:
            try:
                for batch in self._generate_batches():
                    while not stop_event.is_set():
                        try:
                            q.put(batch, timeout=0.05)
                            break
                        except queue.Full:
                            continue
                    if stop_event.is_set():
                        break
                while not stop_event.is_set():
                    try:
                        q.put(sentinel, timeout=0.05)
                        break
                    except queue.Full:
                        continue
            except Exception as exc:
                while not stop_event.is_set():
                    try:
                        q.put(exc, timeout=0.05)
                        break
                    except queue.Full:
                        continue

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        try:
            while True:
                item = q.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                batch_data, bid = item
                self._active_batch_id = bid
                yield batch_data
        finally:
            stop_event.set()
            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    break

    def commit(self) -> None:
        """Persists the current batch coordinate and deduplication WAL atomically."""
        target_bid: int = int(self._active_batch_id if self._active_batch_id > 0 else self._batch_id)
        if self.tracker is not None and target_bid > 0:
            self.tracker.commit(target_bid)
        if self.deduplicator is not None:
            self.deduplicator.commit()


class TokenRingBuffer:
    """
    Pinned-Memory Contiguous Token Ring Buffer (`TokenRingBuffer` / `LumiRingBuffer`)
    with Asynchronous Background Prefill for Zero-Starvation LLM Training.

    Decouples network/disk streaming I/O from GPU compute by maintaining a circular
    reservoir of tokenized IDs in host RAM (optionally page-locked / pinned for direct
    PCIe DMA transfers) and enforcing an initial `auto_prefill_tokens` warmup barrier
    that eliminates burst-and-stall starvation on streaming datasets.
    """

    def __init__(
        self,
        source: Iterable[Any] | None = None,
        *,
        capacity_tokens: int = 1_000_000,
        auto_prefill_tokens: int = 500_000,
        batch_size: int = 4,
        seq_len: int = 2048,
        tokenizer: Any | None = None,
        text_key: str = "text",
        eos_token_id: int | None = None,
        pin_memory: bool = True,
        device: Any | None = None,
        producer_chunk_tokens: int = 16_384,
        prefill_timeout: float | None = 30.0,
    ) -> None:
        if capacity_tokens <= 0:
            raise ValueError(f"capacity_tokens must be > 0, got {capacity_tokens}.")
        if auto_prefill_tokens < 0:
            raise ValueError(f"auto_prefill_tokens must be >= 0, got {auto_prefill_tokens}.")
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError(
                f"batch_size ({batch_size}) and seq_len ({seq_len}) must be > 0."
            )

        import threading

        import numpy as np

        self.source = source
        self.capacity_tokens = int(capacity_tokens)
        self.auto_prefill_tokens = min(int(auto_prefill_tokens), self.capacity_tokens)
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.tokenizer = tokenizer
        self.text_key = text_key
        self.eos_token_id = eos_token_id
        self.pin_memory = bool(pin_memory)
        self.device = device
        self.producer_chunk_tokens = max(256, int(producer_chunk_tokens))
        self.prefill_timeout = prefill_timeout

        self._np = np
        self._torch: Any = None
        self._cuda_available = False
        try:
            import torch

            self._torch = torch
            self._cuda_available = bool(torch.cuda.is_available())
        except ImportError:
            pass

        self._ring = np.empty((self.capacity_tokens,), dtype=np.int64)
        self._read_pos = 0
        self._write_pos = 0
        self._size = 0
        self._total_written = 0
        self._total_read = 0
        self._eof = source is None
        self._error: Exception | None = None

        self._cond = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        if source is not None:
            self._start_producer()
            if self.auto_prefill_tokens > 0:
                self.wait_for_prefill(
                    target_tokens=self.auto_prefill_tokens,
                    timeout=self.prefill_timeout,
                )

    @property
    def available_tokens(self) -> int:
        """Returns the number of unread tokens currently stored in the ring buffer."""
        with self._cond:
            return self._size

    @property
    def is_eof(self) -> bool:
        """Returns True if the source stream has been completely ingested."""
        with self._cond:
            return self._eof

    def _extract_tokens_from_item(self, item: Any) -> list[int]:
        if isinstance(item, int):
            return [item]
        if self._torch is not None and isinstance(item, self._torch.Tensor):
            return item.detach().cpu().reshape(-1).tolist()
        if isinstance(item, self._np.ndarray):
            return item.reshape(-1).tolist()
        if isinstance(item, Mapping):
            if "input_ids" in item:
                raw_ids = item["input_ids"]
                if self._torch is not None and isinstance(raw_ids, self._torch.Tensor):
                    toks = raw_ids.detach().cpu().reshape(-1).tolist()
                elif isinstance(raw_ids, self._np.ndarray):
                    toks = raw_ids.reshape(-1).tolist()
                else:
                    toks = [int(t) for t in raw_ids]
                if self.eos_token_id is not None:
                    toks.append(int(self.eos_token_id))
                return toks
        if isinstance(item, (list, tuple)) and (not item or isinstance(item[0], int)):
            toks = [int(t) for t in item]
            if self.eos_token_id is not None and toks:
                toks.append(int(self.eos_token_id))
            return toks

        text = _extract_text(item, text_key=self.text_key)
        if not text:
            return []

        if self.tokenizer is not None:
            tok_fn: Any = self.tokenizer
            if hasattr(tok_fn, "encode"):
                encoded: Any = tok_fn.encode(text)
            elif callable(tok_fn):
                res: Any = tok_fn(text)
                encoded = res["input_ids"] if isinstance(res, Mapping) and "input_ids" in res else res
            else:
                raise TypeError(f"Unsupported tokenizer object: {type(self.tokenizer)}")
            toks = [int(t) for t in encoded]
        else:
            # Deterministic UTF-8 byte-level fallback when no external tokenizer is supplied
            toks = list(text.encode("utf-8", errors="replace"))

        if self.eos_token_id is not None:
            toks.append(int(self.eos_token_id))
        return toks

    def push(self, tokens: Any) -> int:
        """
        Writes a sequence or 1D array/tensor of token IDs into the ring buffer,
        blocking if the ring is currently full until space becomes available.
        """
        if self._torch is not None and isinstance(tokens, self._torch.Tensor):
            arr = tokens.detach().cpu().numpy().astype(self._np.int64, copy=False).reshape(-1)
        elif isinstance(tokens, self._np.ndarray):
            arr = tokens.astype(self._np.int64, copy=False).reshape(-1)
        else:
            arr = self._np.asarray(tokens, dtype=self._np.int64).reshape(-1)

        total = int(arr.size)
        offset = 0
        while offset < total and not self._stop_event.is_set():
            with self._cond:
                while self._size >= self.capacity_tokens and not self._stop_event.is_set():
                    self._cond.wait(timeout=0.05)
                if self._stop_event.is_set():
                    break

                free_space = self.capacity_tokens - self._size
                n_write = min(total - offset, free_space)
                first_part = min(n_write, self.capacity_tokens - self._write_pos)
                self._ring[self._write_pos : self._write_pos + first_part] = arr[
                    offset : offset + first_part
                ]
                second_part = n_write - first_part
                if second_part > 0:
                    self._ring[:second_part] = arr[
                        offset + first_part : offset + n_write
                    ]

                self._write_pos = (self._write_pos + n_write) % self.capacity_tokens
                self._size += n_write
                self._total_written += n_write
                offset += n_write
                self._cond.notify_all()
        return offset

    def _start_producer(self) -> None:
        import threading

        def _producer_loop() -> None:
            staging: list[int] = []
            try:
                assert self.source is not None
                for item in self.source:
                    if self._stop_event.is_set():
                        break
                    toks = self._extract_tokens_from_item(item)
                    if toks:
                        staging.extend(toks)
                    if len(staging) >= self.producer_chunk_tokens:
                        self.push(staging)
                        staging.clear()
                if staging and not self._stop_event.is_set():
                    self.push(staging)
                    staging.clear()
            except Exception as exc:
                with self._cond:
                    self._error = exc
                    self._cond.notify_all()
            finally:
                with self._cond:
                    self._eof = True
                    self._cond.notify_all()

        self._thread = threading.Thread(target=_producer_loop, daemon=True)
        self._thread.start()

    def wait_for_prefill(
        self,
        target_tokens: int | None = None,
        timeout: float | None = 30.0,
    ) -> int:
        """
        Blocks until the ring buffer contains at least `target_tokens` tokens,
        or until the upstream source reaches EOF.
        """
        import time

        goal = (
            min(int(target_tokens), self.capacity_tokens)
            if target_tokens is not None
            else self.auto_prefill_tokens
        )
        if goal <= 0:
            return self.available_tokens

        deadline = (time.monotonic() + float(timeout)) if timeout is not None else None
        with self._cond:
            while (
                self._size < goal
                and not self._eof
                and self._error is None
                and not self._stop_event.is_set()
            ):
                remaining = (deadline - time.monotonic()) if deadline is not None else 0.05
                if deadline is not None and remaining <= 0.0:
                    break
                self._cond.wait(timeout=min(0.05, max(0.001, remaining)))

            if self._error is not None:
                raise self._error
            return self._size

    def pop_tokens(self, num_tokens: int, *, timeout: float | None = None) -> Any:
        """
        Pops a contiguous 1D block of `num_tokens` token IDs from the ring buffer.
        Raises `StopIteration` if the source stream is exhausted before `num_tokens` are available.
        """
        import time

        n_req = int(num_tokens)
        if n_req <= 0:
            raise ValueError(f"num_tokens must be > 0, got {num_tokens}.")
        if n_req > self.capacity_tokens:
            raise ValueError(
                f"Requested num_tokens ({n_req}) exceeds ring capacity ({self.capacity_tokens})."
            )

        deadline = (time.monotonic() + float(timeout)) if timeout is not None else None
        with self._cond:
            while (
                self._size < n_req
                and not self._eof
                and self._error is None
                and not self._stop_event.is_set()
            ):
                remaining = (deadline - time.monotonic()) if deadline is not None else 0.05
                if deadline is not None and remaining <= 0.0:
                    raise TimeoutError(
                        f"Timed out waiting for {n_req} tokens in TokenRingBuffer (available={self._size})."
                    )
                self._cond.wait(timeout=min(0.05, max(0.001, remaining)))

            if self._error is not None:
                raise self._error
            if self._size < n_req:
                raise StopIteration

            out = self._np.empty((n_req,), dtype=self._np.int64)
            first_part = min(n_req, self.capacity_tokens - self._read_pos)
            out[:first_part] = self._ring[self._read_pos : self._read_pos + first_part]
            second_part = n_req - first_part
            if second_part > 0:
                out[first_part:] = self._ring[:second_part]

            self._read_pos = (self._read_pos + n_req) % self.capacity_tokens
            self._size -= n_req
            self._total_read += n_req
            self._cond.notify_all()

        if self._torch is not None:
            t = self._torch.from_numpy(out)
            if self.pin_memory and self._cuda_available:
                t = t.pin_memory()
            return t
        return out

    def next_batch(
        self,
        batch_size: int | None = None,
        seq_len: int | None = None,
        *,
        device: Any | None = None,
        non_blocking: bool = True,
    ) -> tuple[Any, Any]:
        """
        Fetches the next `(x, y)` causal language modeling batch of shape `(batch_size, seq_len)`.
        When `pin_memory=True` and a CUDA device is active, transfers the batch asynchronously
        via PCIe DMA (`non_blocking=True`).
        """
        B = int(batch_size) if batch_size is not None else self.batch_size
        T = int(seq_len) if seq_len is not None else self.seq_len
        target_dev = device if device is not None else self.device

        raw_1d = self.pop_tokens(B * (T + 1))
        if self._torch is not None and isinstance(raw_1d, self._torch.Tensor):
            block = raw_1d.view(B, T + 1)
            x = block[:, :-1].contiguous()
            y = block[:, 1:].contiguous()
            if self.pin_memory and self._cuda_available and not x.is_cuda:
                x = x.pin_memory()
                y = y.pin_memory()
            if target_dev is not None:
                x = x.to(device=target_dev, non_blocking=bool(non_blocking))
                y = y.to(device=target_dev, non_blocking=bool(non_blocking))
            return x, y

        block_np = raw_1d.reshape(B, T + 1)
        return block_np[:, :-1].copy(), block_np[:, 1:].copy()

    pop_batch = next_batch

    def __iter__(self) -> Iterator[tuple[Any, Any]]:
        while not self._stop_event.is_set():
            try:
                yield self.next_batch()
            except StopIteration:
                break

    def close(self) -> None:
        """Signals the background producer thread to stop and releases waiting consumers."""
        self._stop_event.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def __enter__(self) -> TokenRingBuffer:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


LumiRingBuffer = TokenRingBuffer

__all__ = [
    "LumiRingBuffer",
    "ResumablePipeline",
    "TokenRingBuffer",
    "create_resumable_collate",
]
