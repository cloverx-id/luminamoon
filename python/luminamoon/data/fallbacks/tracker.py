from __future__ import annotations

import os
from pathlib import Path
import struct
from typing import Dict, Optional
import zlib

from .coordinate import StreamCoordinate

_SLOT_MAGIC = b"LMDL"
_SLOT_VERSION = 1
_SLOT_SIZE = 64
_TOTAL_META_SIZE = _SLOT_SIZE * 2


def _encode_slot(epoch: int, batch_id: int, coord: StreamCoordinate, dedup_count: int = 0) -> bytes:
    body = (
        _SLOT_MAGIC
        + struct.pack("<IQQ", _SLOT_VERSION, epoch, batch_id)
        + coord.to_bytes()
        + struct.pack("<I", dedup_count & 0xFFFFFFFF)
    )
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return body + struct.pack("<I", crc)


def _decode_slot(raw: bytes) -> Optional[tuple[int, int, StreamCoordinate, int]]:
    if len(raw) != _SLOT_SIZE or raw[0:4] != _SLOT_MAGIC:
        return None
    version, epoch, batch_id = struct.unpack("<IQQ", raw[4:24])
    if version != _SLOT_VERSION:
        return None
    expected_crc = struct.unpack("<I", raw[60:64])[0]
    if (zlib.crc32(raw[0:60]) & 0xFFFFFFFF) != expected_crc:
        return None
    coord = StreamCoordinate.from_bytes(raw[24:56])
    dedup_count = struct.unpack("<I", raw[56:60])[0]
    return epoch, batch_id, coord, dedup_count


class StateTracker:
    def __init__(self, meta_path: str, dedup_path: Optional[str] = None) -> None:
        self._meta_path = Path(meta_path)
        self._dedup_path = Path(dedup_path) if dedup_path else Path(f"{meta_path}.dedup")
        self._staged: Dict[int, StreamCoordinate] = {}
        self._last_coord: StreamCoordinate = StreamCoordinate()
        self._last_batch_id: int = 0
        self._epoch: int = 0
        self._dedup_count: int = 0
        self._init_or_recover()

    def _init_or_recover(self) -> None:
        self._meta_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._meta_path.exists() or self._meta_path.stat().st_size < _TOTAL_META_SIZE:
            initial = _encode_slot(0, 0, self._last_coord, 0)
            with open(self._meta_path, "wb") as f:
                f.write(initial + initial)
                f.flush()
                os.fsync(f.fileno())
            return

        raw = self._meta_path.read_bytes()[:_TOTAL_META_SIZE]
        slot_a = _decode_slot(raw[0:_SLOT_SIZE])
        slot_b = _decode_slot(raw[_SLOT_SIZE:_TOTAL_META_SIZE])
        if slot_a is not None and slot_b is not None:
            active = slot_a if slot_a[0] >= slot_b[0] else slot_b
        elif slot_a is not None:
            active = slot_a
        elif slot_b is not None:
            active = slot_b
        else:
            raise RuntimeError("Corrupted metadata checkpoint in both slots")
        self._epoch, self._last_batch_id, self._last_coord, self._dedup_count = active

    def _write_active_slot(self) -> None:
        encoded = _encode_slot(self._epoch, self._last_batch_id, self._last_coord, self._dedup_count)
        slot_offset = 0 if (self._epoch % 2 == 1) else _SLOT_SIZE
        with open(self._meta_path, "r+b") as f:
            f.seek(slot_offset)
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())

    def stage_batch(self, batch_id: int, coord: StreamCoordinate) -> None:
        self._staged[batch_id] = coord

    def commit_batch(self, batch_id: int) -> None:
        if batch_id not in self._staged:
            raise RuntimeError(f"Unknown staged batch id: {batch_id}")
        self._last_coord = self._staged.pop(batch_id)
        self._last_batch_id = batch_id
        self._epoch += 1
        self._write_active_slot()

    def commit(self, batch_id: int = 1) -> StreamCoordinate:
        self.commit_batch(batch_id)
        return self._last_coord

    def rollback(self, batch_id: int = 1) -> StreamCoordinate:
        if batch_id not in self._staged:
            raise RuntimeError(f"Unknown staged batch id: {batch_id}")
        self._staged.pop(batch_id)
        if self._staged:
            return self._staged[max(self._staged)]
        return self._last_coord

    def rollback_all(self) -> StreamCoordinate:
        self._staged.clear()
        return self._last_coord

    def committed_batch_id(self) -> int:
        return self._last_batch_id

    def last_committed_coord(self) -> Optional[StreamCoordinate]:
        return self._last_coord

    def last_committed_batch_id(self) -> int:
        return self._last_batch_id

    def epoch(self) -> int:
        return self._epoch

    def save(self, coord: StreamCoordinate) -> int:
        bid = self._last_batch_id + 1
        self.stage_batch(bid, coord)
        self.commit_batch(bid)
        return self._epoch

    def load(self) -> Optional[StreamCoordinate]:
        return self._last_coord

    def read(self) -> StreamCoordinate:
        return self._last_coord

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
        coord = StreamCoordinate(
            shard_id=shard_id,
            byte_offset=byte_offset,
            row_group_idx=row_group_idx,
            page_idx=page_idx,
            row_offset_in_page=row_offset_in_page,
            global_row_idx=global_row_idx,
        )
        self.stage_batch(batch_id, coord)


__all__ = ["StateTracker"]
