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


__all__ = ["ResumablePipeline", "create_resumable_collate"]
