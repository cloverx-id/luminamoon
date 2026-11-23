from __future__ import annotations

import os
from pathlib import Path
import struct
import threading
from typing import Dict, Optional
import zlib

from .coordinate import StreamCoordinate

_SLOT_MAGIC = b"LMDL"
_SLOT_VERSION = 2
_SLOT_SIZE = 128
_TOTAL_META_SIZE = _SLOT_SIZE * 2
_HASH_WIDTH = 16


def _encode_slot(
    epoch: int,
    batch_id: int,
    coord: StreamCoordinate,
    dedup_count: int = 0,
    consumed_tokens: int = 0,
) -> bytes:
    body = (
        _SLOT_MAGIC
        + struct.pack("<IQQ", _SLOT_VERSION, epoch, batch_id)
        + coord.to_bytes()
        + struct.pack("<QQI", dedup_count, consumed_tokens, _HASH_WIDTH)
        + (b"\x00" * 48)
    )
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return body + struct.pack("<I", crc)


def _decode_slot(raw: bytes) -> Optional[tuple[int, int, StreamCoordinate, int, int]]:
    if len(raw) != _SLOT_SIZE or raw[0:4] != _SLOT_MAGIC:
        return None
    version, epoch, batch_id = struct.unpack("<IQQ", raw[4:24])
    if version != _SLOT_VERSION:
        return None
    expected_crc = struct.unpack("<I", raw[124:128])[0]
    if (zlib.crc32(raw[0:124]) & 0xFFFFFFFF) != expected_crc:
        return None
    coord = StreamCoordinate.from_bytes(raw[24:56])
    dedup_count, consumed_tokens, hash_width = struct.unpack("<QQI", raw[56:76])
    if hash_width != _HASH_WIDTH:
        return None
    return epoch, batch_id, coord, dedup_count, consumed_tokens


class StateTracker:
    def __init__(self, meta_path: str, dedup_path: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        self._meta_path = Path(meta_path)
        self._dedup_path = Path(dedup_path) if dedup_path else Path(f"{meta_path}.dedup")
        self._staged: Dict[int, StreamCoordinate] = {}
        self._last_coord: StreamCoordinate = StreamCoordinate()
        self._last_batch_id: int = 0
        self._epoch: int = 0
        self._dedup_count: int = 0
        self._consumed_tokens: int = 0
        with self._lock:
            self._init_or_recover()

    def _init_or_recover(self) -> None:
        self._meta_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._meta_path.exists() or self._meta_path.stat().st_size == 0:
            initial = _encode_slot(0, 0, self._last_coord, 0, 0)
            with open(self._meta_path, "wb") as f:
                f.write(initial + initial)
                f.flush()
                os.fsync(f.fileno())
            return

        size = self._meta_path.stat().st_size
        if size < _TOTAL_META_SIZE:
            if size >= _SLOT_SIZE:
                slot_a = _decode_slot(self._meta_path.read_bytes()[:_SLOT_SIZE])
                if slot_a is not None:
                    self._epoch, self._last_batch_id, self._last_coord, self._dedup_count, self._consumed_tokens = slot_a
                    return
            raise RuntimeError("Corrupted metadata checkpoint in both slots")

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
        self._epoch, self._last_batch_id, self._last_coord, self._dedup_count, self._consumed_tokens = active

    def _write_slot(self, epoch: int, batch_id: int, coord: StreamCoordinate) -> None:
        encoded = _encode_slot(epoch, batch_id, coord, self._dedup_count, self._consumed_tokens)
        slot_offset = 0 if (epoch % 2 == 1) else _SLOT_SIZE
        with open(self._meta_path, "r+b") as f:
            f.seek(slot_offset)
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())

    def stage_batch(self, batch_id: int, coord: StreamCoordinate) -> None:
        with self._lock:
            self._staged[batch_id] = coord

    def commit_batch(self, batch_id: int) -> None:
        with self._lock:
            if batch_id not in self._staged:
                raise RuntimeError(f"Unknown staged batch id: {batch_id}")
            next_coord = self._staged[batch_id]
            next_epoch = self._epoch + 1
            eff_bid = max(batch_id, self._last_batch_id)
            eff_coord = next_coord if batch_id >= self._last_batch_id else self._last_coord
            self._write_slot(next_epoch, eff_bid, eff_coord)
            self._staged.pop(batch_id, None)
            self._epoch = next_epoch
            self._last_batch_id = eff_bid
            self._last_coord = eff_coord

    def commit(self, batch_id: int = 1) -> StreamCoordinate:
        with self._lock:
            self.commit_batch(batch_id)
            return self._last_coord

    def commit_up_to(self, batch_id: int = 1) -> StreamCoordinate:
        with self._lock:
            keys = sorted(k for k in self._staged if k <= batch_id)
            if not keys:
                if batch_id <= self._last_batch_id:
                    return self._last_coord
                raise RuntimeError(f"Unknown staged batch id: {batch_id}")
            latest_bid = keys[-1]
            latest_coord = self._staged[latest_bid]
            next_epoch = self._epoch + 1
            eff_bid = max(latest_bid, self._last_batch_id)
            eff_coord = latest_coord if latest_bid >= self._last_batch_id else self._last_coord
            self._write_slot(next_epoch, eff_bid, eff_coord)
            for k in keys:
                self._staged.pop(k, None)
            self._epoch = next_epoch
            self._last_batch_id = eff_bid
            self._last_coord = eff_coord
            return self._last_coord

    def rollback(self, batch_id: int = 1) -> StreamCoordinate:
        with self._lock:
            if batch_id not in self._staged:
                raise RuntimeError(f"Unknown staged batch id: {batch_id}")
            self._staged.pop(batch_id)
            if self._staged:
                return self._staged[max(self._staged)]
            return self._last_coord

    def rollback_all(self) -> StreamCoordinate:
        with self._lock:
            self._staged.clear()
            return self._last_coord

    def committed_batch_id(self) -> int:
        with self._lock:
            return self._last_batch_id

    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    def consumed_tokens(self) -> int:
        with self._lock:
            return self._consumed_tokens

    def set_consumed_tokens(self, tokens: int) -> None:
        with self._lock:
            self._consumed_tokens = max(0, int(tokens))

    def read(self) -> StreamCoordinate:
        with self._lock:
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
