from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .checkpoint import StateTracker, StreamCoordinate
from .cleaner import TextCleaner
from .collate import _extract_text
from .dedup import StreamDeduplicator
from .masker import PIIMasker
from .quality import QualityFilter
from .string_tensor import StringTensor

if TYPE_CHECKING:
    from .fused import FusedPipeline


class ResumablePipeline:
    """Streaming data pipeline with checkpoint resume and optional background prefetching."""

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
        """Return the resume coordinate read from the on-disk checkpoint."""
        return self.tracker.read() if self.tracker else None

    @property
    def _effective_deduplicator(self) -> StreamDeduplicator | None:
        if self.deduplicator is not None:
            return self.deduplicator
        if self.fused_pipeline is not None:
            return self.fused_pipeline.deduplicator
        return None

    def _finalize_batch(self, buffer: list[tuple[str, int]]) -> tuple[list[str] | StringTensor, int]:
        self._batch_id += 1
        last_row = buffer[-1][1]
        self._current_row = last_row
        if self.tracker:
            self.tracker.update(
                shard_id=self._current_shard,
                row_offset_in_page=last_row + 1,
                global_row_idx=self._emitted_samples,
                batch_id=self._batch_id,
            )
        texts = [text for text, _ in buffer]
        buffer.clear()
        if (
            self.fused_pipeline is not None
            and self.fused_pipeline.commit_dedup
            and self.tracker is None
            and self.prefetch_batches == 0
            and self._effective_deduplicator is not None
        ):
            self._effective_deduplicator.commit(batch_id=self._batch_id)
        payload: list[str] | StringTensor = StringTensor(texts) if self.return_string_tensor else texts
        return payload, self._batch_id

    def _drain_fused_chunk(
        self,
        raw_chunk: list[str],
        row_indices: list[int],
        buffer: list[tuple[str, int]],
    ) -> Iterator[tuple[list[str] | StringTensor, int]]:
        assert self.fused_pipeline is not None
        dedup = self._effective_deduplicator
        st, survivor_local_indices = self.fused_pipeline.process_indexed(
            raw_chunk,
            include_dedup=False,
        )
        cleaned_items = st.to_list()
        chunk_rows = list(row_indices)
        raw_chunk.clear()
        row_indices.clear()

        for c, local_idx in zip(cleaned_items, survivor_local_indices):
            pending_bid = self._batch_id + 1
            if dedup is not None and not dedup.is_unique(c, batch_id=pending_bid):
                continue
            if self.masker:
                c = self.masker.mask(c)
            buffer.append((c, chunk_rows[local_idx]))
            self._emitted_samples += 1
            if len(buffer) >= self.batch_size:
                yield self._finalize_batch(buffer)

    def _generate_batches(self) -> Iterator[tuple[list[str] | StringTensor, int]]:
        resume_coord = self.start_coordinate
        target_row = resume_coord.row_offset_in_page if resume_coord else 0
        if resume_coord is not None:
            self._current_shard = resume_coord.shard_id
            self._emitted_samples = resume_coord.global_row_idx
            if self.tracker is not None:
                self._batch_id = int(self.tracker.committed_batch_id)
                self._active_batch_id = self._batch_id

        buffer: list[tuple[str, int]] = []
        dedup = self._effective_deduplicator

        if self.fused_pipeline is not None:
            raw_chunk: list[str] = []
            row_indices: list[int] = []
            chunk_size = self.batch_size * 2

            for idx, item in enumerate(self.source):
                if idx < target_row:
                    continue
                raw = _extract_text(item, self.text_key)
                if raw is not None:
                    raw_chunk.append(raw)
                    row_indices.append(idx)
                if len(raw_chunk) >= chunk_size:
                    yield from self._drain_fused_chunk(raw_chunk, row_indices, buffer)

            if raw_chunk:
                yield from self._drain_fused_chunk(raw_chunk, row_indices, buffer)
        else:
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
                pending_bid = self._batch_id + 1
                if dedup is not None and not dedup.is_unique(clean, batch_id=pending_bid):
                    continue
                if self.masker:
                    clean = self.masker.mask(clean)

                buffer.append((clean, idx))
                self._emitted_samples += 1

                if len(buffer) >= self.batch_size:
                    yield self._finalize_batch(buffer)

        if buffer:
            yield self._finalize_batch(buffer)

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
        """Persist the current batch coordinate and deduplication WAL atomically."""
        target_bid: int = int(self._active_batch_id if self._active_batch_id > 0 else self._batch_id)
        if self.tracker is not None and target_bid > 0:
            self.tracker.commit(target_bid)
        dedup = self._effective_deduplicator
        if dedup is not None and target_bid > 0:
            dedup.commit(batch_id=target_bid)


__all__ = ["ResumablePipeline"]
