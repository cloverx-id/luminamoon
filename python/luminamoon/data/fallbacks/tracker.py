from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from .coordinate import StreamCoordinate

class StateTracker:
    def __init__(self, meta_path: str, dedup_path: Optional[str] = None) -> None:
        self._meta_path = Path(meta_path)
        self._dedup_path = Path(dedup_path) if dedup_path else Path(f"{meta_path}.dedup")
        self._staged: Dict[int, StreamCoordinate] = {}
        self._last_coord: Optional[StreamCoordinate] = None
        self._last_batch_id: int = 0
        self._epoch: int = 0
        if self._meta_path.exists():
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        try:
            data = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._epoch = int(data.get("epoch", 0))
            self._last_batch_id = int(data.get("batch_id", 0))
            c = data.get("coord")
            if isinstance(c, dict):
                self._last_coord = StreamCoordinate(**c)
        except Exception:
            pass

    def _save_to_disk(self) -> None:
        self._meta_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": self._epoch,
            "batch_id": self._last_batch_id,
            "coord": self._last_coord.to_dict() if self._last_coord else None,
        }
        self._meta_path.write_text(json.dumps(payload), encoding="utf-8")

    def stage_batch(self, batch_id: int, coord: StreamCoordinate) -> None:
        self._staged[batch_id] = coord

    def commit_batch(self, batch_id: int) -> None:
        coord = self._staged.pop(batch_id, None)
        if coord is not None:
            self._last_coord = coord
        self._last_batch_id = batch_id
        self._epoch += 1
        self._save_to_disk()

    def commit(self, batch_id: int = 1) -> StreamCoordinate:
        self.commit_batch(batch_id)
        return self.read()

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
        return self._last_coord or StreamCoordinate()

    def update(
        self,
        shard_id: int = 0,
        byte_offset: int = 0,
        row_group_idx: int = 0,
        page_idx: int = 0,
        row_offset_in_page: int = 0,
        global_row_idx: int = 0,
        batch_id: int = 0,
    ) -> None:
        coord = StreamCoordinate(
            shard_id=shard_id,
            byte_offset=byte_offset,
            row_group_idx=row_group_idx,
            page_idx=page_idx,
            row_offset_in_page=row_offset_in_page,
            global_row_idx=global_row_idx,
        )
        self._last_coord = coord
        self._last_batch_id = batch_id
        self._save_to_disk()

__all__ = ["StateTracker"]
