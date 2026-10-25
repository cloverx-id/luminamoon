from __future__ import annotations

import hashlib
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

    def is_unique(self, text: str) -> bool:
        h = self.compute_hash(text)
        if h in self._seen or h in self._staged_lookup:
            return False
        self._staged_lookup.add(h)
        self._staged.setdefault(0, []).append(h)
        return True

    def try_add(self, payload: bytes) -> bool:
        h = self.compute_hash(payload)
        if h in self._seen or h in self._staged_lookup:
            return False
        self._staged_lookup.add(h)
        self._staged.setdefault(0, []).append(h)
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

    def commit(self) -> int:
        count = len(self._staged_lookup)
        self._seen.update(self._staged_lookup)
        self._staged.clear()
        self._staged_lookup.clear()
        return count

    def rollback(self) -> int:
        count = len(self._staged_lookup)
        self._staged.clear()
        self._staged_lookup.clear()
        return count

    def filter_batch(self, texts: Sequence[str]) -> List[str]:
        return [t for t in texts if self.check_and_insert(t)]

    def filter_indices(self, texts: Sequence[str]) -> List[int]:
        return [idx for idx, t in enumerate(texts) if self.check_and_insert(t)]

    def committed_count(self) -> int:
        return len(self._seen)

    def staged_count(self) -> int:
        return len(self._staged_lookup)

    @property
    def count(self) -> int:
        return len(self._seen) + len(self._staged_lookup)

    def reset_staged(self) -> None:
        self._staged.clear()
        self._staged_lookup.clear()

    def clear(self) -> None:
        self._seen.clear()
        self._staged.clear()
        self._staged_lookup.clear()

    def save_wal(self, path: str) -> int:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"".join(h.to_bytes(8, "little") for h in self._seen))
        return len(self._seen)

    def load_wal(self, path: str) -> int:
        p = Path(path)
        if not p.exists():
            return 0
        data = p.read_bytes()
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
