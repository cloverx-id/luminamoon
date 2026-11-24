from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
import json
from pathlib import Path
import threading
from typing import Any

from ._sources import (
    atomic_write_json,
    iter_mixed_records,
    normalize_sources,
    resolve_distributed_context,
)
from .cleaner import TextCleaner
from .collate import _extract_text, create_resumable_collate
from .dedup import StreamDeduplicator
from .masker import PIIMasker
from .minhash import MinHashDeduplicator
from .packing import SequencePacker, TokenBatch
from .quality import QualityFilter
from .string_tensor import StringTensor


class DataPipeline:
    """Unified, stateful, zero-skew streaming pipeline for LLM pretraining & curation (v2.0).

    Combines multi-source weighted mixing, distributed rank/worker sharding, deterministic
    shuffling, Rust-accelerated curation (clean -> quality -> dedup -> PII -> custom filter),
    document-isolated sequence packing (`best_fit` / `concat` / `pad`), and exact-boundary
    crash-proof checkpointing (`commit()`, `state_dict()`, `load_state_dict()`).
    """

    def __init__(
        self,
        source: Iterable[Any] | Mapping[str, Any],
        *,
        tokenizer: Any | None = None,
        batch_size: int = 4,
        seq_len: int = 2048,
        packing: str = "best_fit",
        pack_window: int = 64,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        ignore_index: int = -100,
        drop_last: bool = False,
        text_key: str = "text",
        clean: bool | TextCleaner = True,
        quality: str | QualityFilter | None = None,
        dedup: str | StreamDeduplicator | MinHashDeduplicator | None = None,
        mask_pii: bool | PIIMasker = False,
        filter_fn: Callable[[str], bool] | None = None,
        rank: int | None = None,
        world_size: int | None = None,
        seed: int | None = None,
        shuffle_buffer: int = 0,
        checkpoint: str | Path | None = None,
        device: Any | None = None,
        non_blocking: bool = True,
        return_string_tensor: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}.")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}.")

        self._sources, self._is_mixture = normalize_sources(source)
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.packing = packing
        self.pack_window = max(self.batch_size, int(pack_window))
        self.ignore_index = int(ignore_index)
        self.drop_last = bool(drop_last)
        self.text_key = text_key
        self.filter_fn = filter_fn
        self.seed = seed
        self.shuffle_buffer = max(0, int(shuffle_buffer))
        self.device = device
        self.non_blocking = bool(non_blocking)
        self.return_string_tensor = bool(return_string_tensor)

        self.eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else getattr(tokenizer, "eos_token_id", None)
        )
        inferred_pad = getattr(tokenizer, "pad_token_id", None)
        self.pad_token_id = (
            int(pad_token_id)
            if pad_token_id is not None
            else (int(inferred_pad) if inferred_pad is not None else 0)
        )

        self._explicit_rank = rank
        self._explicit_world_size = world_size
        self.rank, self.world_size = resolve_distributed_context(rank, world_size)

        self._checkpoint_base = Path(checkpoint) if checkpoint is not None else None
        self._wal_path: Path | None = None
        self._state_json_path: Path | None = None
        self._epoch = 0
        self._configure_checkpoint_paths(self.rank, self.world_size)

        self.cleaner: TextCleaner | None = (
            clean if isinstance(clean, TextCleaner) else (TextCleaner() if clean else None)
        )
        if quality is None:
            self.quality_filter: QualityFilter | None = None
        elif isinstance(quality, QualityFilter):
            self.quality_filter = quality
        elif isinstance(quality, str) and quality.lower() == "pretraining":
            self.quality_filter = QualityFilter.for_pretraining()
        else:
            raise ValueError(f"Unsupported quality preset '{quality}'. Use None, 'pretraining', or QualityFilter.")

        self.deduplicator: StreamDeduplicator | None = None
        self.minhash: MinHashDeduplicator | None = None
        if isinstance(dedup, StreamDeduplicator):
            self.deduplicator = dedup
        elif isinstance(dedup, MinHashDeduplicator):
            self.minhash = dedup
        elif isinstance(dedup, str):
            mode = dedup.lower()
            if mode == "exact":
                self.deduplicator = StreamDeduplicator(wal_path=self._wal_path)
            elif mode == "minhash":
                self.minhash = MinHashDeduplicator()
            elif mode == "hybrid":
                self.deduplicator = StreamDeduplicator(wal_path=self._wal_path)
                self.minhash = MinHashDeduplicator()
            else:
                raise ValueError(f"Unsupported dedup mode '{dedup}'. Use None, 'exact', 'minhash', or 'hybrid'.")
        elif dedup is not None:
            raise TypeError(f"Unsupported dedup object: {type(dedup)}")

        self.masker: PIIMasker | None = (
            mask_pii if isinstance(mask_pii, PIIMasker) else (PIIMasker() if mask_pii else None)
        )

        # Exact-once boundary tracking (zero cursor skew even under Best-Fit out-of-order packing).
        self._lock = threading.RLock()
        self._committed_cursors: dict[str, int] = {name: 0 for name, _, _ in self._sources}
        self._committed_above: dict[str, set[int]] = {name: set() for name, _, _ in self._sources}
        self._emitted_by_source: dict[str, set[int]] = {name: set() for name, _, _ in self._sources}
        self._read_cursors: dict[str, int] = {name: 0 for name, _, _ in self._sources}
        self._emitted_batches = 0
        self._emitted_tokens = 0
        self._committed_batches = 0
        self._committed_tokens = 0
        self._closed = False

        if self._state_json_path is not None and self._state_json_path.is_file():
            self.load_state_dict(json.loads(self._state_json_path.read_text(encoding="utf-8")))

    def _configure_checkpoint_paths(self, rank: int, world_size: int) -> None:
        if self._checkpoint_base is None:
            return
        stem = self._checkpoint_base / f"rank_{rank}" if world_size > 1 else self._checkpoint_base
        self._wal_path = Path(f"{stem}.dedup")
        self._state_json_path = Path(f"{stem}.state.json")

    def _encode_text(self, text: str) -> list[int]:
        if self.tokenizer is None:
            return list(text.encode("utf-8", errors="replace"))
        tok = self.tokenizer
        ids: Any
        if hasattr(tok, "encode"):
            ids = tok.encode(text)
        elif callable(tok):
            out = tok(text)
            ids = out["input_ids"] if isinstance(out, Mapping) and "input_ids" in out else out
        else:
            raise TypeError(f"Unsupported tokenizer object: {type(tok)}")
        if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        result = [int(t) for t in ids]
        if self.eos_token_id is not None and (not result or result[-1] != int(self.eos_token_id)):
            result.append(int(self.eos_token_id))
        return result

    def _curate_text(
        self,
        raw_text: str,
        *,
        batch_id: int,
        defer_dedup_stage: bool = False,
        window_seen: set[int] | None = None,
    ) -> str | None:
        text: str | None = self.cleaner.clean_single(raw_text) if self.cleaner else raw_text
        if text is None:
            return None
        if self.quality_filter is not None and not self.quality_filter.is_high_quality(text):
            return None
        if self.deduplicator is not None:
            if defer_dedup_stage:
                if self.deduplicator.is_seen(text):
                    return None
                h = self.deduplicator.compute_hash_128(text)
                if window_seen is not None:
                    if h in window_seen:
                        return None
                    window_seen.add(h)
            elif not self.deduplicator.is_unique(text, batch_id=batch_id):
                return None
        if self.minhash is not None and not self.minhash.is_unique(text):
            return None
        if self.masker is not None:
            text = self.masker.mask(text)
        if self.filter_fn is not None and not self.filter_fn(text):
            return None
        return text

    def _record_consumed_refs(self, refs: list[tuple[str, int]], num_tokens: int) -> None:
        with self._lock:
            self._emitted_batches += 1
            self._emitted_tokens += int(num_tokens)
            for name, row_idx in refs:
                self._emitted_by_source.setdefault(name, set()).add(row_idx)

    def _compute_exact_cursors(self) -> tuple[dict[str, int], dict[str, list[int]]]:
        next_cursors: dict[str, int] = {}
        above: dict[str, list[int]] = {}
        for name, _, _ in self._sources:
            base = self._committed_cursors.get(name, 0)
            seen = set(self._committed_above.get(name, set())) | self._emitted_by_source.get(name, set())
            read_limit = max(self._read_cursors.get(name, base), max(seen) + 1 if seen else base)
            ws, rk = self.world_size, self.rank
            cur = base
            while cur < read_limit:
                if ws > 1 and (cur % ws) != rk:
                    cur += 1
                    continue
                if cur in seen:
                    seen.discard(cur)
                    cur += 1
                else:
                    break
            next_cursors[name] = cur
            above[name] = sorted(x for x in seen if x >= cur)
        return next_cursors, above

    def __iter__(self) -> Iterator[TokenBatch | list[str] | StringTensor]:
        rank, world_size = resolve_distributed_context(self._explicit_rank, self._explicit_world_size)
        if (rank, world_size) != (self.rank, self.world_size):
            self.rank, self.world_size = rank, world_size
            self._configure_checkpoint_paths(rank, world_size)
        eff_seed = (self.seed + self._epoch) if self.seed is not None else None
        mixed = iter_mixed_records(
            self._sources,
            is_mixture=self._is_mixture,
            rank=rank,
            world_size=world_size,
            committed_cursors=self._committed_cursors,
            committed_above=self._committed_above,
            seed=eff_seed,
            shuffle_buffer=self.shuffle_buffer,
        )

        if self.tokenizer is None:
            text_buf: list[tuple[str, int, str]] = []
            for name, idx, item in mixed:
                if self._closed:
                    break
                self._read_cursors[name] = max(self._read_cursors.get(name, 0), idx + 1)
                raw = _extract_text(item, text_key=self.text_key)
                curated = self._curate_text(raw, batch_id=self._emitted_batches + 1) if raw is not None else None
                if curated is None:
                    self._emitted_by_source.setdefault(name, set()).add(idx)
                    continue
                text_buf.append((name, idx, curated))
                if len(text_buf) >= self.batch_size:
                    refs = [(n, i) for n, i, _ in text_buf]
                    texts = [t for _, _, t in text_buf]
                    text_buf.clear()
                    self._record_consumed_refs(refs, 0)
                    yield StringTensor(texts) if self.return_string_tensor else texts
            if text_buf and not self.drop_last and not self._closed:
                refs = [(n, i) for n, i, _ in text_buf]
                texts = [t for _, _, t in text_buf]
                text_buf.clear()
                self._record_consumed_refs(refs, 0)
                yield StringTensor(texts) if self.return_string_tensor else texts
            return

        packer = SequencePacker(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            mode=self.packing,
            pack_window=self.pack_window,
            pad_token_id=self.pad_token_id,
            ignore_index=self.ignore_index,
            drop_last=self.drop_last,
            device=self.device,
            non_blocking=self.non_blocking,
        )
        window_seen: set[int] = set()
        pending_dedup: dict[tuple[str, int], tuple[str, int]] = {}

        def _stage_emitted_batch(batch: TokenBatch) -> None:
            n_toks = int(batch.attention_mask.sum().item()) if hasattr(batch.attention_mask, "sum") else 0
            self._record_consumed_refs(batch.source_refs, n_toks)
            if self.deduplicator is not None:
                bid = self._emitted_batches
                for ref in batch.source_refs:
                    item_dedup = pending_dedup.pop(ref, None)
                    if item_dedup is not None:
                        txt, h = item_dedup
                        window_seen.discard(h)
                        self.deduplicator.is_unique(txt, batch_id=bid)

        for name, idx, item in mixed:
            if self._closed:
                break
            self._read_cursors[name] = max(self._read_cursors.get(name, 0), idx + 1)
            raw = _extract_text(item, text_key=self.text_key)
            curated = (
                self._curate_text(
                    raw,
                    batch_id=self._emitted_batches + 1,
                    defer_dedup_stage=True,
                    window_seen=window_seen,
                )
                if raw is not None
                else None
            )
            if curated is None:
                self._emitted_by_source.setdefault(name, set()).add(idx)
                continue
            toks = self._encode_text(curated)
            if not toks:
                self._emitted_by_source.setdefault(name, set()).add(idx)
                continue
            if self.deduplicator is not None:
                pending_dedup[(name, idx)] = (curated, self.deduplicator.compute_hash_128(curated))
            packer.add_document(toks, source_name=name, row_idx=idx)
            for batch in packer.pop_ready_batches():
                _stage_emitted_batch(batch)
                yield batch

        if not self._closed:
            for batch in packer.flush():
                _stage_emitted_batch(batch)
                yield batch

    def commit(self) -> None:
        """Atomically and idempotently commit exact consumed document & token boundaries."""
        with self._lock:
            if self._emitted_batches <= self._committed_batches:
                return
            next_cursors, above = self._compute_exact_cursors()
            self._committed_cursors = next_cursors
            self._committed_above = {k: set(v) for k, v in above.items()}
            for name in self._emitted_by_source:
                self._emitted_by_source[name].clear()
            self._committed_batches = self._emitted_batches
            self._committed_tokens = self._emitted_tokens

            if self.deduplicator is not None:
                self.deduplicator.commit_up_to(batch_id=self._committed_batches)

            if self._state_json_path is not None:
                # Keep binary .dedup WAL strictly O(delta) on disk instead of dumping all hashes as JSON.
                disk_state = self._build_state_dict(include_dedup_hashes=self._wal_path is None)
                atomic_write_json(self._state_json_path, disk_state)

    def _build_state_dict(self, *, include_dedup_hashes: bool = True) -> dict[str, Any]:
        return {
            "version": 2,
            "epoch": self._epoch,
            "rank": self.rank,
            "world_size": self.world_size,
            "committed_batches": self._committed_batches,
            "committed_tokens": self._committed_tokens,
            "cursors": dict(self._committed_cursors),
            "emitted_above": {k: sorted(v) for k, v in self._committed_above.items()},
            "dedup": (
                self.deduplicator.state_dict()
                if (include_dedup_hashes and self.deduplicator is not None)
                else None
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return PyTorch `Stateful` / DCP-compatible checkpoint state."""
        with self._lock:
            return self._build_state_dict(include_dedup_hashes=True)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore exact resume state from `state_dict()`."""
        with self._lock:
            self._epoch = int(state.get("epoch", 0))
            self._committed_batches = int(state.get("committed_batches", 0))
            self._emitted_batches = self._committed_batches
            self._committed_tokens = int(state.get("committed_tokens", 0))
            self._emitted_tokens = self._committed_tokens
            cursors = state.get("cursors") or {}
            self._committed_cursors = {
                name: int(cursors.get(name, 0)) for name, _, _ in self._sources
            }
            above = state.get("emitted_above") or {}
            self._committed_above = {
                name: {int(x) for x in above.get(name, ())} for name, _, _ in self._sources
            }
            self._emitted_by_source = {name: set() for name, _, _ in self._sources}
            self._read_cursors = dict(self._committed_cursors)
            if self.deduplicator is not None and state.get("dedup") is not None:
                self.deduplicator.load_state_dict(state["dedup"])

    def reset(self) -> None:
        """Reset stream cursors so the pipeline can be iterated for a fresh epoch."""
        with self._lock:
            self._closed = False
            self._committed_cursors = {name: 0 for name, _, _ in self._sources}
            self._committed_above = {name: set() for name, _, _ in self._sources}
            self._emitted_by_source = {name: set() for name, _, _ in self._sources}
            self._read_cursors = {name: 0 for name, _, _ in self._sources}

    def set_epoch(self, epoch: int) -> None:
        """Advance to a new epoch (updates deterministic shuffle/mix seed and resets cursors)."""
        with self._lock:
            self._epoch = int(epoch)
            self.reset()

    def as_collate_fn(self) -> Callable[[list[Any]], Any]:
        """Export this pipeline's curation stages as a PyTorch `DataLoader` `collate_fn`."""
        return create_resumable_collate(
            cleaner=self.cleaner,
            quality_filter=self.quality_filter,
            deduplicator=self.deduplicator,
            minhash=self.minhash,
            masker=self.masker,
            filter_fn=self.filter_fn,
            tokenizer=self.tokenizer,
            text_key=self.text_key,
            max_length=self.seq_len if self.tokenizer is not None else None,
            return_string_tensor=self.return_string_tensor,
        )

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> DataPipeline:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


__all__ = ["DataPipeline"]
