from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any

from .checkpoint import StateTracker, StreamCoordinate
from .cleaner import TextCleaner
from .collate import _extract_text
from .dedup import StreamDeduplicator
from .masker import PIIMasker
from .minhash import MinHashDeduplicator
from .quality import QualityFilter
from .string_tensor import StringTensor

if TYPE_CHECKING:
    from .fused import FusedPipeline


class ResumablePipeline:
    """Streaming text pipeline with crash-proof checkpoint resume, sharding, and prefetching."""

    def __init__(
        self,
        source: Iterable[Any],
        checkpoint_path: str | Path | None = None,
        cleaner: TextCleaner | None = None,
        quality_filter: QualityFilter | None = None,
        deduplicator: StreamDeduplicator | None = None,
        minhash: MinHashDeduplicator | None = None,
        masker: PIIMasker | None = None,
        fused_pipeline: FusedPipeline | None = None,
        text_key: str = "text",
        batch_size: int = 64,
        prefetch_batches: int = 0,
        return_string_tensor: bool = False,
        *,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if fused_pipeline is not None and (cleaner is not None or quality_filter is not None):
            raise ValueError(
                "Pass either `fused_pipeline` OR individual `cleaner`/`quality_filter`, not both."
            )
        if world_size <= 0 or not (0 <= rank < world_size):
            raise ValueError(f"Invalid distributed sharding: rank={rank}, world_size={world_size}.")

        self.source = source
        self.cleaner = cleaner
        self.quality_filter = quality_filter
        self.deduplicator = deduplicator
        self.minhash = minhash
        self.masker = masker
        self.fused_pipeline = fused_pipeline
        self.text_key = text_key
        self.batch_size = max(1, int(batch_size))
        self.prefetch_batches = max(0, int(prefetch_batches))
        self.return_string_tensor = bool(return_string_tensor)
        self.rank = int(rank)
        self.world_size = int(world_size)

        self.tracker = StateTracker(checkpoint_path) if checkpoint_path else None
        self._state_lock = threading.RLock()
        self._current_shard = 0
        self._current_row = 0
        self._emitted_samples = 0
        self._batch_id = 0
        self._active_batch_id = 0
        self._last_committed_bid = 0

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
        with self._state_lock:
            self._batch_id += 1
            bid = self._batch_id
            last_row = buffer[-1][1]
            self._current_row = last_row
            if self.tracker:
                self.tracker.update(
                    shard_id=self._current_shard,
                    row_offset_in_page=min(last_row + 1, 0xFFFF_FFFF),
                    global_row_idx=last_row + 1,
                    batch_id=bid,
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
                self._effective_deduplicator.commit_up_to(batch_id=bid)
                self._last_committed_bid = bid
        payload: list[str] | StringTensor = StringTensor(texts) if self.return_string_tensor else texts
        return payload, bid

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
            if self.minhash is not None and not self.minhash.is_unique(c):
                continue
            if self.masker is not None:
                c = self.masker.mask(c)
            buffer.append((c, chunk_rows[local_idx]))
            self._emitted_samples += 1
            if len(buffer) >= self.batch_size:
                yield self._finalize_batch(buffer)

    def _generate_batches(self) -> Iterator[tuple[list[str] | StringTensor, int]]:
        resume_coord = self.start_coordinate
        target_row = 0
        if resume_coord is not None:
            # Prefer 64-bit global_row_idx (stores exact next source index in v2) when populated.
            target_row = max(int(resume_coord.row_offset_in_page), int(resume_coord.global_row_idx))
            self._current_shard = resume_coord.shard_id
            if self.tracker is not None:
                self._batch_id = int(self.tracker.committed_batch_id)
                self._active_batch_id = self._batch_id
                self._last_committed_bid = self._batch_id

        buffer: list[tuple[str, int]] = []
        dedup = self._effective_deduplicator
        ws = self.world_size
        rk = self.rank

        if self.fused_pipeline is not None:
            raw_chunk: list[str] = []
            row_indices: list[int] = []
            chunk_size = self.batch_size * 2

            for idx, item in enumerate(self.source):
                if idx < target_row or (ws > 1 and (idx % ws) != rk):
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
                if idx < target_row or (ws > 1 and (idx % ws) != rk):
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
                if self.minhash is not None and not self.minhash.is_unique(clean):
                    continue
                if self.masker is not None:
                    clean = self.masker.mask(clean)

                buffer.append((clean, idx))
                self._emitted_samples += 1

                if len(buffer) >= self.batch_size:
                    yield self._finalize_batch(buffer)

        if buffer:
            yield self._finalize_batch(buffer)

    def _rollback_unconsumed(self, unconsumed_bids: list[int]) -> None:
        dedup = self._effective_deduplicator
        with self._state_lock:
            for bid in unconsumed_bids:
                if self.tracker is not None:
                    try:
                        self.tracker.rollback(bid)
                    except Exception:
                        pass
                if dedup is not None:
                    dedup.rollback(batch_id=bid)

    def __iter__(self) -> Iterator[list[str] | StringTensor]:
        if self.prefetch_batches <= 0:
            for batch_data, bid in self._generate_batches():
                self._active_batch_id = bid
                yield batch_data
            return

        import queue

        q: queue.Queue[Any] = queue.Queue(maxsize=self.prefetch_batches)
        stop_event = threading.Event()
        sentinel = object()
        error_tag = object()

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
                        self._rollback_unconsumed([batch[1]])
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
                        q.put((error_tag, exc), timeout=0.05)
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
                if isinstance(item, tuple) and len(item) == 2 and item[0] is error_tag:
                    raise item[1]
                batch_data, bid = item
                self._active_batch_id = bid
                yield batch_data
        finally:
            stop_event.set()
            unconsumed: list[int] = []
            while True:
                try:
                    leftover = q.get_nowait()
                    if (
                        isinstance(leftover, tuple)
                        and len(leftover) == 2
                        and leftover[0] is not error_tag
                        and isinstance(leftover[1], int)
                    ):
                        unconsumed.append(leftover[1])
                except queue.Empty:
                    break
            if thread.is_alive():
                thread.join(timeout=2.0)
            if unconsumed:
                self._rollback_unconsumed(unconsumed)

    def commit(self) -> None:
        """Atomically and idempotently persist all consumed batches up to `_active_batch_id`."""
        with self._state_lock:
            target_bid = int(self._active_batch_id if self._active_batch_id > 0 else self._batch_id)
            if target_bid <= 0 or target_bid <= self._last_committed_bid:
                return
            dedup = self._effective_deduplicator
            if dedup is not None:
                dedup.commit_up_to(batch_id=target_bid)
            if self.tracker is not None:
                self.tracker.commit_up_to(target_bid)
            self._last_committed_bid = target_bid

    def state_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot compatible with PyTorch Stateful / DCP."""
        with self._state_lock:
            dedup = self._effective_deduplicator
            return {
                "current_shard": self._current_shard,
                "current_row": self._current_row,
                "emitted_samples": self._emitted_samples,
                "active_batch_id": self._active_batch_id,
                "tracker": self.tracker.state_dict() if self.tracker is not None else None,
                "dedup": dedup.state_dict() if dedup is not None else None,
            }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore pipeline cursor and deduplication state from `state_dict()`."""
        with self._state_lock:
            self._current_shard = int(state.get("current_shard", 0))
            self._current_row = int(state.get("current_row", 0))
            self._emitted_samples = int(state.get("emitted_samples", 0))
            self._active_batch_id = int(state.get("active_batch_id", 0))
            if self.tracker is not None and state.get("tracker"):
                self.tracker.load_state_dict(state["tracker"])
            dedup = self._effective_deduplicator
            if dedup is not None and state.get("dedup"):
                dedup.load_state_dict(state["dedup"])


__all__ = ["ResumablePipeline"]
