"""Background prefetching and staged batch helpers for DataPipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import queue
import threading
from typing import Any

from .collate import _extract_text
from .packing import SequencePacker, TokenBatch
from .string_tensor import StringTensor


def iter_staged_batches(
    pipeline: Any,
    mixed: Iterator[tuple[str, int, Any]],
    stop_event: threading.Event | None = None,
    window_seen: set[int] | None = None,
) -> Iterator[
    tuple[
        TokenBatch | list[str] | StringTensor,
        list[tuple[str, int]],
        int,
        list[tuple[str, int]],
        list[tuple[str, int]],
    ]
]:
    """Yield staged batches along with source refs, skipped refs, and deferred dedup items."""
    seen_hashes: set[int] = window_seen if window_seen is not None else set()

    if pipeline.tokenizer is None:
        text_buf: list[tuple[str, int, str, str | None, int | None]] = []
        skipped_refs: list[tuple[str, int]] = []
        defer_dedup = pipeline.prefetch_batches > 0
        for name, idx, item in mixed:
            if pipeline._closed or (stop_event is not None and stop_event.is_set()):
                break
            raw = _extract_text(item, text_key=pipeline.text_key)
            with pipeline._lock:
                curated_res = (
                    pipeline._curate_text(
                        raw,
                        batch_id=pipeline._emitted_batches + 1,
                        defer_dedup_stage=defer_dedup,
                        window_seen=seen_hashes if defer_dedup else None,
                    )
                    if raw is not None
                    else None
                )
                if curated_res is not None and defer_dedup and curated_res[2] is not None:
                    seen_hashes.add(curated_res[2])
            if curated_res is None:
                skipped_refs.append((name, idx))
                continue
            curated, dedup_txt, dedup_h = curated_res
            text_buf.append((name, idx, curated, dedup_txt, dedup_h))
            if len(text_buf) >= pipeline.batch_size:
                refs = [(n, i) for n, i, _, _, _ in text_buf]
                texts = [t for _, _, t, _, _ in text_buf]
                dedup_items = (
                    [(dt, dh) for _, _, _, dt, dh in text_buf if dt is not None and dh is not None]
                    if (defer_dedup and pipeline.deduplicator is not None)
                    else []
                )
                text_buf.clear()
                payload: list[str] | StringTensor = StringTensor(texts) if pipeline.return_string_tensor else texts
                yield payload, refs, 0, skipped_refs, dedup_items
                skipped_refs = []
        if text_buf and not pipeline.drop_last and not pipeline._closed and (stop_event is None or not stop_event.is_set()):
            refs = [(n, i) for n, i, _, _, _ in text_buf]
            texts = [t for _, _, t, _, _ in text_buf]
            dedup_items = (
                [(dt, dh) for _, _, _, dt, dh in text_buf if dt is not None and dh is not None]
                if (defer_dedup and pipeline.deduplicator is not None)
                else []
            )
            text_buf.clear()
            payload = StringTensor(texts) if pipeline.return_string_tensor else texts
            yield payload, refs, 0, skipped_refs, dedup_items
        return

    packer = SequencePacker(
        batch_size=pipeline.batch_size,
        seq_len=pipeline.seq_len,
        mode=pipeline.packing,
        pack_window=pipeline.pack_window,
        pad_token_id=pipeline.pad_token_id,
        ignore_index=pipeline.ignore_index,
        drop_last=pipeline.drop_last,
        device=pipeline.device,
        non_blocking=pipeline.non_blocking,
    )
    pending_dedup: dict[tuple[str, int], tuple[str, int]] = {}
    skipped_refs = []

    def _extract_batch_dedup(batch: TokenBatch) -> list[tuple[str, int]]:
        if pipeline.deduplicator is None:
            return []
        items: list[tuple[str, int]] = []
        for ref in batch.source_refs:
            item_dedup = pending_dedup.pop(ref, None)
            if item_dedup is not None:
                items.append(item_dedup)
        return items

    for name, idx, item in mixed:
        if pipeline._closed or (stop_event is not None and stop_event.is_set()):
            break
        raw = _extract_text(item, text_key=pipeline.text_key)
        with pipeline._lock:
            curated_res = (
                pipeline._curate_text(
                    raw,
                    batch_id=pipeline._emitted_batches + 1,
                    defer_dedup_stage=True,
                    window_seen=seen_hashes,
                )
                if raw is not None
                else None
            )
        if curated_res is None:
            skipped_refs.append((name, idx))
            continue
        curated, dedup_txt, dedup_h = curated_res
        toks = pipeline._encode_text(curated)
        if not toks:
            skipped_refs.append((name, idx))
            continue
        if pipeline.deduplicator is not None and dedup_txt is not None and dedup_h is not None:
            with pipeline._lock:
                seen_hashes.add(dedup_h)
            pending_dedup[(name, idx)] = (dedup_txt, dedup_h)
        packer.add_document(toks, source_name=name, row_idx=idx)
        for batch in packer.pop_ready_batches():
            n_toks = int(batch.attention_mask.sum().item()) if hasattr(batch.attention_mask, "sum") else 0
            yield batch, list(batch.source_refs), n_toks, skipped_refs, _extract_batch_dedup(batch)
            skipped_refs = []

    if not pipeline._closed and (stop_event is None or not stop_event.is_set()):
        for batch in packer.flush():
            n_toks = int(batch.attention_mask.sum().item()) if hasattr(batch.attention_mask, "sum") else 0
            yield batch, list(batch.source_refs), n_toks, skipped_refs, _extract_batch_dedup(batch)
            skipped_refs = []


def run_prefetched_batches(
    pipeline: Any,
    produce_fn: Callable[[threading.Event], Iterator[Any]],
    window_seen: set[int],
) -> Iterator[TokenBatch | list[str] | StringTensor]:
    """Run a background producer queue while finalizing cursor and dedup state on consumer yield."""
    buf: queue.Queue[Any] = queue.Queue(maxsize=pipeline.prefetch_batches)
    stop_event = threading.Event()
    sentinel = object()
    error_holder: list[BaseException] = []

    def _producer() -> None:
        try:
            for record in produce_fn(stop_event):
                while not stop_event.is_set() and not pipeline._closed:
                    try:
                        buf.put(record, timeout=0.05)
                        break
                    except queue.Full:
                        continue
                if stop_event.is_set() or pipeline._closed:
                    break
        except BaseException as exc:
            error_holder.append(exc)
        finally:
            while True:
                try:
                    buf.put(sentinel, timeout=0.05)
                    break
                except queue.Full:
                    if stop_event.is_set() or pipeline._closed:
                        try:
                            buf.get_nowait()
                        except queue.Empty:
                            pass

    worker = threading.Thread(target=_producer, daemon=True)
    worker.start()
    try:
        while not pipeline._closed:
            try:
                item = buf.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is sentinel:
                if error_holder:
                    raise error_holder[0]
                break
            payload, refs, n_toks, skipped, dedup_items = item
            pipeline._finalize_yielded_batch(refs, n_toks, skipped, dedup_items, window_seen=window_seen)
            yield payload
    finally:
        stop_event.set()
        while True:
            try:
                buf.get_nowait()
            except queue.Empty:
                break
        worker.join(timeout=2.0)


__all__ = ["iter_staged_batches", "run_prefetched_batches"]
