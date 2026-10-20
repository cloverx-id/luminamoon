from __future__ import annotations

from pathlib import Path

from ._ffi import StateTracker as _RawStateTracker, StreamCoordinate


class StateTracker:
    """Crash-proof Dual-Slot Ping-Pong checkpoint state manager backed by Rust.

    Features:
    - Alternating 64-byte Slot A / Slot B design (128-byte total file).
    - CRC32 checksum protection on every slot.
    - Zero file corruption on abrupt process termination or GPU preemption.
    - Tracks fine-grained shard index, row offset, byte offset, and global sample index.
    """

    def __init__(self, meta_path: str | Path) -> None:
        self.meta_path = Path(meta_path)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self._raw = _RawStateTracker(str(self.meta_path))

    def read(self) -> StreamCoordinate:
        """Reads the last validly committed coordinate from the Ping-Pong slots."""
        return self._raw.read()

    def update(
        self,
        shard_id: int = 0,
        byte_offset: int = 0,
        row_group_idx: int = 0,
        page_idx: int = 0,
        row_offset_in_page: int = 0,
        global_row_idx: int = 0,
        batch_id: int = 1,
    ) -> None:
        """Stages an in-flight coordinate update in RAM."""
        self._raw.update(
            shard_id=shard_id,
            byte_offset=byte_offset,
            row_group_idx=row_group_idx,
            page_idx=page_idx,
            row_offset_in_page=row_offset_in_page,
            global_row_idx=global_row_idx,
            batch_id=batch_id,
        )

    def commit(self, batch_id: int = 1) -> StreamCoordinate:
        """Atomically persists the staged coordinate to disk via alternating Slot A/B."""
        return self._raw.commit(batch_id)

    @property
    def committed_batch_id(self) -> int:
        """Returns the last committed batch ID."""
        return self._raw.committed_batch_id()


__all__ = ["StateTracker", "StreamCoordinate"]
