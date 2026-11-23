from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, overload

from ._ffi import StreamDeduplicator as _RawStreamDeduplicator


class StreamDeduplicator:
    """Stateful 128-bit stream deduplicator backed by Rust XXH3-128 and append-only WAL.

    Features:
    - Collision-resistant 128-bit XXH3 hashing in Rust memory.
    - Two-stage lifecycle: `stage` -> `commit` / `commit_up_to` / `rollback`.
    - O(delta) append-only `.dedup` Write-Ahead Log (LMWL v2) for crash recovery.
    - Zero Python GIL lock during batch filtering and disk I/O.
    """

    def __init__(
        self,
        wal_path: str | Path | None = None,
        initial_hashes: Sequence[int] | None = None,
        max_capacity: int | None = None,
    ) -> None:
        self._wal_path = Path(wal_path) if wal_path else None
        self.max_capacity = max_capacity
        try:
            self._raw = _RawStreamDeduplicator(initial_hashes, max_capacity)
        except TypeError:
            self._raw = _RawStreamDeduplicator(initial_hashes)
        if self._wal_path and self._wal_path.is_file():
            self._raw.load_wal(str(self._wal_path))

    @staticmethod
    def compute_hash(record: str | bytes) -> int:
        """Computes the 64-bit XXH3 hash for a string or byte slice."""
        payload = record.encode("utf-8") if isinstance(record, str) else record
        return _RawStreamDeduplicator.compute_hash(payload)

    @staticmethod
    def compute_hash_128(record: str | bytes) -> int:
        """Computes the 128-bit XXH3 hash for a string or byte slice."""
        payload = record.encode("utf-8") if isinstance(record, str) else record
        if hasattr(_RawStreamDeduplicator, "compute_hash_128"):
            return _RawStreamDeduplicator.compute_hash_128(payload)
        return _RawStreamDeduplicator.compute_hash(payload)

    def is_seen(self, record: str | bytes) -> bool:
        """Checks if a record's hash is already committed or staged."""
        payload = record.encode("utf-8") if isinstance(record, str) else record
        return self._raw.is_seen(payload)

    def is_unique(self, record: str | bytes, batch_id: int = 1) -> bool:
        """Tries to stage a record under `batch_id`. Returns True if unseen."""
        payload = record.encode("utf-8") if isinstance(record, str) else record
        return self._raw.try_add(payload, batch_id)

    def filter_batch(self, records: Sequence[str], batch_id: int = 1) -> list[str]:
        """Filters a batch of strings, staging unseen items under `batch_id`."""
        if not records:
            return []
        return self._raw.filter_batch(list(records), batch_id)

    def commit(self, batch_id: int = 1) -> int:
        """Finalizes staged items for `batch_id` into committed history."""
        count = self._raw.commit(batch_id)
        if count > 0 and self._wal_path:
            self.save_wal()
        return count

    def commit_up_to(self, batch_id: int = 1) -> int:
        """Finalizes all staged batches with ID `<= batch_id` atomically and idempotently."""
        if hasattr(self._raw, "commit_up_to"):
            count = self._raw.commit_up_to(batch_id)
        else:
            count = self._raw.commit(batch_id)
        if count > 0 and self._wal_path:
            self.save_wal()
        return count

    def rollback(self, batch_id: int = 1) -> int:
        """Discards staged items for `batch_id` without committing them."""
        return self._raw.rollback(batch_id)

    def clear_staged(self) -> None:
        """Clears all uncommitted staged records."""
        self._raw.clear()

    def clear_all(self) -> None:
        """Clears both committed and staged records."""
        if hasattr(self._raw, "clear_all"):
            self._raw.clear_all()
        else:
            self._raw.clear()

    @property
    def count(self) -> int:
        """Total number of committed unique hashes tracked in memory."""
        raw_count = self._raw.count
        return raw_count() if callable(raw_count) else int(raw_count)

    def save_wal(self, path: str | Path | None = None) -> int:
        """Flushes newly committed hashes to the binary WAL file (O(delta) append-only)."""
        target = Path(path) if path else self._wal_path
        if not target:
            raise ValueError("No WAL path specified")
        target.parent.mkdir(parents=True, exist_ok=True)
        return self._raw.save_wal(str(target))

    def load_wal(self, path: str | Path | None = None) -> int:
        """Loads committed hashes from a binary WAL file into memory."""
        target = Path(path) if path else self._wal_path
        if not target:
            raise ValueError("No WAL path specified")
        return self._raw.load_wal(str(target))

    def export_hashes(self) -> list[int]:
        """Exports all committed hashes as a list of integers."""
        return self._raw.export_hashes()

    def state_dict(self) -> dict[str, Any]:
        """Returns a serializable snapshot of committed deduplication hashes."""
        return {"hashes": self.export_hashes(), "max_capacity": self.max_capacity}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restores committed hashes from a `state_dict()` snapshot."""
        self.clear_all()
        hashes = state.get("hashes") or []
        if hashes:
            self._raw = _RawStreamDeduplicator(list(hashes), self.max_capacity)

    @overload
    def __call__(self, target: str, *, batch_id: int = 1) -> bool: ...  # pyright: ignore[reportOverlappingOverload]

    @overload
    def __call__(self, target: Sequence[str], *, batch_id: int = 1) -> list[str]: ...

    def __call__(self, target: str | Sequence[str], *, batch_id: int = 1) -> bool | list[str]:
        if isinstance(target, str):
            return self.is_unique(target, batch_id=batch_id)
        return self.filter_batch(target, batch_id=batch_id)


def hash_record(record: str | bytes) -> int:
    """Convenience function to compute XXH3-64 hash."""
    return StreamDeduplicator.compute_hash(record)


__all__ = ["StreamDeduplicator", "hash_record"]
