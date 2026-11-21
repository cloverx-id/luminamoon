from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

class StreamDeduplicator:
    def __init__(
        self,
        capacity: int = 1_000_000,
        persistence_path: Optional[str] = None,
        initial_hashes: Optional[Sequence[int]] = None,
        wal_path: Optional[str] = None,
    ) -> None:
        self._seen: set[int] = set(initial_hashes or ())
        self._staged: Dict[int, List[int]] = {}
        self._staged_lookup: set[int] = set()
        self._persistence_path = persistence_path or wal_path

    @staticmethod
    def compute_hash(text_or_payload: Any) -> int:
        raw = text_or_payload.encode("utf-8") if isinstance(text_or_payload, str) else bytes(text_or_payload)
        digest = hashlib.blake2b(raw, digest_size=8).digest()
        return int.from_bytes(digest, "little", signed=False)

    def is_seen(self, payload: Any) -> bool:
        h = self.compute_hash(payload)
        return (h in self._seen) or (h in self._staged_lookup)

    def is_duplicate(self, text: str) -> bool:
        h = self.compute_hash(text)
        return (h in self._seen) or (h in self._staged_lookup)

    def check_and_insert(self, text: str) -> bool:
        h = self.compute_hash(text)
        if h in self._seen or h in self._staged_lookup:
            return False
        self._seen.add(h)
        return True

    def is_unique(self, text: str, batch_id: int = 1) -> bool:
        return self.try_add(text.encode("utf-8") if isinstance(text, str) else text, batch_id=batch_id)

    def try_add(self, payload: bytes, batch_id: int = 1) -> bool:
        h = self.compute_hash(payload)
        if h in self._seen or h in self._staged_lookup:
            return False
        self._staged_lookup.add(h)
        self._staged.setdefault(batch_id, []).append(h)
        return True

    def stage_batch(self, batch_id: int, texts: Sequence[str]) -> List[bool]:
        mask: List[bool] = []
        batch_hashes: List[int] = []
        for t in texts:
            h = self.compute_hash(t)
            if h in self._seen or h in self._staged_lookup:
                mask.append(False)
            else:
                self._staged_lookup.add(h)
                batch_hashes.append(h)
                mask.append(True)
        self._staged.setdefault(batch_id, []).extend(batch_hashes)
        return mask

    def commit_batch(self, batch_id: int) -> int:
        hashes = self._staged.pop(batch_id, [])
        for h in hashes:
            self._staged_lookup.discard(h)
            self._seen.add(h)
        return len(hashes)

    def commit(self, batch_id: int = 1) -> int:
        return self.commit_batch(batch_id)

    def rollback(self, batch_id: int = 1) -> int:
        hashes = self._staged.pop(batch_id, [])
        for h in hashes:
            self._staged_lookup.discard(h)
        return len(hashes)

    def filter_batch(self, texts: Sequence[str], batch_id: int = 1) -> List[str]:
        out: List[str] = []
        for t in texts:
            if self.try_add(t.encode("utf-8") if isinstance(t, str) else t, batch_id=batch_id):
                out.append(t)
        return out

    def filter_indices(self, texts: Sequence[str], batch_id: int = 1) -> List[int]:
        out: List[int] = []
        for idx, t in enumerate(texts):
            if self.try_add(t.encode("utf-8") if isinstance(t, str) else t, batch_id=batch_id):
                out.append(idx)
        return out

    def committed_count(self) -> int:
        return len(self._seen)

    def staged_count(self) -> int:
        return len(self._staged_lookup)

    def count(self) -> int:
        return len(self._seen)

    def reset_staged(self) -> None:
        self._staged.clear()
        self._staged_lookup.clear()

    def clear(self) -> None:
        self._staged.clear()
        self._staged_lookup.clear()

    def clear_all(self) -> None:
        self._seen.clear()
        self._staged.clear()
        self._staged_lookup.clear()

    def save_wal(self, path: str) -> int:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(f"{p}.tmp.{os.getpid()}")
        tmp.write_bytes(b"".join(h.to_bytes(8, "little") for h in self._seen))
        os.replace(tmp, p)
        return len(self._seen)

    def load_wal(self, path: str) -> int:
        p = Path(path)
        if not p.exists():
            return 0
        data = p.read_bytes()
        if len(data) % 8 != 0:
            raise ValueError(f"Corrupted WAL file '{path}': byte length {len(data)} is not a multiple of 8")
        n = len(data) // 8
        for i in range(n):
            h = int.from_bytes(data[i * 8 : (i + 1) * 8], "little")
            self._seen.add(h)
        return n

    def export_hashes(self) -> List[int]:
        return list(self._seen)

    def __len__(self) -> int:
        return len(self._seen) + len(self._staged_lookup)

__all__ = ["StreamDeduplicator"]
