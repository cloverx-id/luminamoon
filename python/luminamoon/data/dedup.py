from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ._ffi import StreamDeduplicator as _RawStreamDeduplicator


class StreamDeduplicator:
    """Stateful 64-bit stream deduplicator backed by Rust XXH3-64 and SwissTable.

    Features:
    - Zero-overhead IdentityHasher lookup in Rust memory.
    - Two-stage lifecycle: `stage` -> `commit` / `rollback`.
    - Automatic `.dedup` Write-Ahead Log (WAL) sync for crash recovery.
    - Zero Python GIL lock during batch filtering.
    """

    def __init__(
        self,
        wal_path: str | Path | None = None,
        initial_hashes: Sequence[int] | None = None,
    ) -> None:
        self._wal_path = Path(wal_path) if wal_path else None
        self._raw = _RawStreamDeduplicator(initial_hashes)
        if self._wal_path and self._wal_path.is_file():
            self._raw.load_wal(str(self._wal_path))

    @staticmethod
    def compute_hash(record: str | bytes) -> int:
        """Computes the 64-bit XXH3 hash for a string or byte slice."""
        payload = record.encode("utf-8") if isinstance(record, str) else record
        return _RawStreamDeduplicator.compute_hash(payload)

    def is_seen(self, record: str | bytes) -> bool:
        """Checks if a record's hash has already been committed."""
        payload = record.encode("utf-8") if isinstance(record, str) else record
        return self._raw.is_seen(payload)

    def is_unique(self, record: str | bytes) -> bool:
        """Tries to stage a record. Returns True if record is unique (not seen before)."""
        payload = record.encode("utf-8") if isinstance(record, str) else record
        return self._raw.try_add(payload)

    def filter_batch(self, records: Sequence[str]) -> list[str]:
        """Filters a batch of strings, returning only previously unseen unique items.

        Note: Unseen items are staged in memory. Call `commit()` to finalize them,
        or `rollback()` if batch processing fails.
        """
        if not records:
            return []
        return self._raw.filter_batch(records)

    def commit(self) -> int:
        """Finalizes all currently staged items into committed history.

        If a `wal_path` was provided, automatically updates the on-disk WAL.
        Returns the number of newly committed items.
        """
        count = self._raw.commit()
        if count > 0 and self._wal_path:
            self._raw.save_wal(str(self._wal_path))
        return count

    def rollback(self) -> int:
        """Discards all currently staged items without committing them."""
        return self._raw.rollback()

    def clear_staged(self) -> None:
        """Clears all staged records."""
        self._raw.clear()

    @property
    def count(self) -> int:
        """Total number of committed unique hashes tracked in memory."""
        return self._raw.count()

    def save_wal(self, path: str | Path | None = None) -> int:
        """Saves all committed hashes into a binary WAL file."""
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
        """Exports all committed hashes as a list of 64-bit integers."""
        return self._raw.export_hashes()


def hash_record(record: str | bytes) -> int:
    """Convenience function to compute XXH3-64 hash."""
    return StreamDeduplicator.compute_hash(record)
