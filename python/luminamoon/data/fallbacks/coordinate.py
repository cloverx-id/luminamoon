from __future__ import annotations

import time
from typing import Dict

class StreamCoordinate:
    def __init__(
        self,
        shard_id: int = 0,
        byte_offset: int = 0,
        row_group_idx: int = 0,
        page_idx: int = 0,
        row_offset_in_page: int = 0,
        global_row_idx: int = 0,
        epoch: int = 0,
        record_idx: int = 0,
        rng_state: int = 0,
    ) -> None:
        self.shard_id = shard_id
        self.byte_offset = byte_offset
        self.row_group_idx = row_group_idx
        self.page_idx = page_idx
        self.row_offset_in_page = row_offset_in_page
        self.global_row_idx = global_row_idx if global_row_idx != 0 else record_idx
        self.epoch = epoch
        self.record_idx = self.global_row_idx
        self.rng_state = rng_state
        self.timestamp_ns = time.time_ns()

    def to_dict(self) -> Dict[str, int]:
        return {
            "shard_id": self.shard_id,
            "byte_offset": self.byte_offset,
            "row_group_idx": self.row_group_idx,
            "page_idx": self.page_idx,
            "row_offset_in_page": self.row_offset_in_page,
            "global_row_idx": self.global_row_idx,
        }

    def to_bytes(self) -> bytes:
        return (
            self.shard_id.to_bytes(4, "little")
            + self.byte_offset.to_bytes(8, "little")
            + self.row_group_idx.to_bytes(4, "little")
            + self.page_idx.to_bytes(4, "little")
            + self.row_offset_in_page.to_bytes(4, "little")
            + self.global_row_idx.to_bytes(8, "little")
        )

    @staticmethod
    def from_bytes(raw: bytes) -> StreamCoordinate:
        if len(raw) != 32:
            raise ValueError(f"Invalid coordinate length: {len(raw)}")
        return StreamCoordinate(
            shard_id=int.from_bytes(raw[0:4], "little"),
            byte_offset=int.from_bytes(raw[4:12], "little"),
            row_group_idx=int.from_bytes(raw[12:16], "little"),
            page_idx=int.from_bytes(raw[16:20], "little"),
            row_offset_in_page=int.from_bytes(raw[20:24], "little"),
            global_row_idx=int.from_bytes(raw[24:32], "little"),
        )

__all__ = ["StreamCoordinate"]
