from __future__ import annotations

import hashlib
import os
from pathlib import Path
import struct
import threading
from typing import Any, Dict, List, Optional, Sequence

# Distinct magic ('LMWP') separates pure-Python BLAKE2b-128 WALs from native Rust XXH3-128 ('LMWL').
_WAL_MAGIC = b"LMWP"
_NATIVE_WAL_MAGIC = b"LMWL"
_WAL_VERSION = 2
_WAL_HEADER = _WAL_MAGIC + struct.pack("<I", _WAL_VERSION)
_HASH_WIDTH = 16


class StreamDeduplicator:
    def __init__(
        self,
        initial_hashes: Optional[Sequence[int]] = None,
        max_capacity: Optional[int] = None,
        wal_path: Optional[str] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._seen: set[int] = set(initial_hashes or ())
        self._staged: Dict[int, List[int]] = {}
        self._staged_lookup: set[int] = set()
        self._max_capacity = max_capacity
        self._wal_target: Optional[Path] = None
        self._pending_wal: List[int] = []
        if wal_path and Path(wal_path).is_file():
            self.load_wal(wal_path)

    @staticmethod
    def compute_hash(text_or_payload: Any) -> int:
        raw = text_or_payload.encode("utf-8") if isinstance(text_or_payload, str) else bytes(text_or_payload)
        return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little", signed=False)

    @staticmethod
    def compute_hash_128(text_or_payload: Any) -> int:
        raw = text_or_payload.encode("utf-8") if isinstance(text_or_payload, str) else bytes(text_or_payload)
        return int.from_bytes(hashlib.blake2b(raw, digest_size=16).digest(), "little", signed=False)

    def is_seen(self, payload: Any) -> bool:
        h = self.compute_hash_128(payload)
        with self._lock:
            return (h in self._seen) or (h in self._staged_lookup)

    def is_unique(self, text: str, batch_id: int = 1) -> bool:
        return self.try_add(text.encode("utf-8") if isinstance(text, str) else text, batch_id=batch_id)

    def try_add(self, payload: bytes, batch_id: int = 1) -> bool:
        h = self.compute_hash_128(payload)
        with self._lock:
            if h in self._seen or h in self._staged_lookup:
                return False
            if self._max_capacity is not None and (len(self._seen) + len(self._staged_lookup)) >= self._max_capacity:
                return True
            self._staged_lookup.add(h)
            self._staged.setdefault(batch_id, []).append(h)
            return True

    def commit(self, batch_id: int = 1) -> int:
        with self._lock:
            hashes = self._staged.pop(batch_id, [])
            newly = []
            for h in hashes:
                self._staged_lookup.discard(h)
                if h not in self._seen:
                    self._seen.add(h)
                    newly.append(h)
            self._pending_wal.extend(newly)
            return len(newly)

    def commit_up_to(self, batch_id: int = 1) -> int:
        with self._lock:
            keys = sorted(k for k in self._staged if k <= batch_id)
            total = 0
            for k in keys:
                total += self.commit(k)
            return total

    def rollback(self, batch_id: int = 1) -> int:
        with self._lock:
            hashes = self._staged.pop(batch_id, [])
            for h in hashes:
                self._staged_lookup.discard(h)
            return len(hashes)

    def filter_batch(self, texts: Sequence[str], batch_id: int = 1) -> List[str]:
        out: List[str] = []
        with self._lock:
            for t in texts:
                if self.try_add(t.encode("utf-8") if isinstance(t, str) else t, batch_id=batch_id):
                    out.append(t)
        return out

    def count(self) -> int:
        with self._lock:
            return len(self._seen)

    def clear(self) -> None:
        with self._lock:
            self._staged.clear()
            self._staged_lookup.clear()

    def clear_all(self) -> None:
        with self._lock:
            self._seen.clear()
            self._staged.clear()
            self._staged_lookup.clear()
            self._pending_wal.clear()
            self._wal_target = None

    def save_wal(self, path: str) -> int:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._wal_target == p and p.exists():
                if self._pending_wal:
                    payload = b"".join(h.to_bytes(_HASH_WIDTH, "little") for h in self._pending_wal)
                    with open(p, "ab") as f:
                        f.write(payload)
                        f.flush()
                        os.fsync(f.fileno())
                    self._pending_wal.clear()
                return len(self._seen)

            snapshot = list(self._seen)
            payload = _WAL_HEADER + b"".join(h.to_bytes(_HASH_WIDTH, "little") for h in snapshot)
            tmp_path = p.with_suffix(p.suffix + ".tmp")
            with open(tmp_path, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, p)
            self._pending_wal.clear()
            self._wal_target = p
            return len(snapshot)

    def load_wal(self, path: str) -> int:
        p = Path(path)
        if not p.exists():
            return 0
        data = p.read_bytes()
        if len(data) >= 4 and data[:4] == _NATIVE_WAL_MAGIC:
            raise ValueError(
                f"WAL file '{path}' uses native Rust XXH3-128 format ('LMWL'), "
                "which is incompatible with the pure-Python BLAKE2b-128 fallback."
            )
        if len(data) < len(_WAL_HEADER) or data[: len(_WAL_HEADER)] != _WAL_HEADER:
            raise ValueError(f"Corrupted WAL file '{path}': invalid LMWP v{_WAL_VERSION} header")
        body = data[len(_WAL_HEADER) :]
        valid_body_len = (len(body) // _HASH_WIDTH) * _HASH_WIDTH
        if valid_body_len < len(body):
            # Recover from torn append by truncating incomplete trailing bytes.
            with open(p, "r+b") as f:
                f.truncate(len(_WAL_HEADER) + valid_body_len)
                f.flush()
                os.fsync(f.fileno())
            body = body[:valid_body_len]
        n = valid_body_len // _HASH_WIDTH
        parsed = [int.from_bytes(body[i * _HASH_WIDTH : (i + 1) * _HASH_WIDTH], "little") for i in range(n)]
        with self._lock:
            self._seen.update(parsed)
            self._pending_wal.clear()
            self._wal_target = p
        return n

    def truncate_wal(self, path: str, max_count: int) -> int:
        p = Path(path)
        if not p.exists():
            return 0
        size = p.stat().st_size
        if size < len(_WAL_HEADER):
            return 0
        current = (size - len(_WAL_HEADER)) // _HASH_WIDTH
        keep = min(current, max(0, int(max_count)))
        target_len = len(_WAL_HEADER) + keep * _HASH_WIDTH
        if target_len < size:
            with open(p, "r+b") as f:
                f.truncate(target_len)
                f.flush()
                os.fsync(f.fileno())
        return keep

    def export_hashes(self) -> List[int]:
        with self._lock:
            return list(self._seen)

    def __len__(self) -> int:
        with self._lock:
            return len(self._seen) + len(self._staged_lookup)


__all__ = ["StreamDeduplicator"]
