from __future__ import annotations

from pathlib import Path
from typing import Any

from ._ffi import StateTracker as _RawStateTracker, StreamCoordinate


class StateTracker:
    """Crash-proof Dual-Slot Ping-Pong checkpoint state manager backed by Rust (LMDL v2).

    Features:
    - Alternating 128-byte Slot A / Slot B design (256-byte total file, 64B cache-line aligned).
    - Full 64-bit `dedup_count` and `consumed_tokens` fields protected by CRC32.
    - Atomic write-before-mutate commit with `commit_up_to` idempotency.
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

    def commit_up_to(self, batch_id: int = 1) -> StreamCoordinate:
        """Atomically and idempotently persists all staged batches with ID `<= batch_id`."""
        if hasattr(self._raw, "commit_up_to"):
            return self._raw.commit_up_to(batch_id)
        return self._raw.commit(batch_id)

    def rollback(self, batch_id: int = 1) -> StreamCoordinate:
        """Discards a staged coordinate update for `batch_id` from RAM."""
        return self._raw.rollback(batch_id)

    def rollback_all(self) -> StreamCoordinate:
        """Discards all uncommitted staged coordinate updates from RAM."""
        return self._raw.rollback_all()

    @property
    def committed_batch_id(self) -> int:
        """Returns the last committed batch ID."""
        return self._raw.committed_batch_id()

    @property
    def consumed_tokens(self) -> int:
        """Returns the last committed consumed token count."""
        return self._raw.consumed_tokens() if hasattr(self._raw, "consumed_tokens") else 0

    @consumed_tokens.setter
    def consumed_tokens(self, tokens: int) -> None:
        if hasattr(self._raw, "set_consumed_tokens"):
            self._raw.set_consumed_tokens(max(0, int(tokens)))

    def state_dict(self) -> dict[str, Any]:
        c = self.read()
        return {
            "shard_id": c.shard_id,
            "byte_offset": c.byte_offset,
            "row_group_idx": c.row_group_idx,
            "page_idx": c.page_idx,
            "row_offset_in_page": c.row_offset_in_page,
            "global_row_idx": c.global_row_idx,
            "committed_batch_id": self.committed_batch_id,
            "consumed_tokens": self.consumed_tokens,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        bid = max(self.committed_batch_id + 1, int(state.get("committed_batch_id", 1)))
        self.consumed_tokens = int(state.get("consumed_tokens", 0))
        self.update(
            shard_id=int(state.get("shard_id", 0)),
            byte_offset=int(state.get("byte_offset", 0)),
            row_group_idx=int(state.get("row_group_idx", 0)),
            page_idx=int(state.get("page_idx", 0)),
            row_offset_in_page=int(state.get("row_offset_in_page", 0)),
            global_row_idx=int(state.get("global_row_idx", 0)),
            batch_id=bid,
        )
        self.commit_up_to(bid)


__all__ = ["StateTracker", "StreamCoordinate"]
