from __future__ import annotations

import struct
from typing import List, Optional, Sequence, Tuple

from .dedup import StreamDeduplicator

class RustStringTensor:
    def __init__(self, strings: Sequence[str]) -> None:
        self._strings = list(strings)
        encoded = [s.encode("utf-8") for s in self._strings]
        offsets = [0]
        for b in encoded:
            offsets.append(offsets[-1] + len(b))
        self._offsets = offsets
        self._payload = b"".join(encoded)

    @staticmethod
    def from_list(strings: Sequence[str]) -> "RustStringTensor":
        return RustStringTensor(strings)

    @staticmethod
    def from_gpu_buffers(offsets: Sequence[int], payload: bytes) -> "RustStringTensor":
        if not offsets:
            return RustStringTensor([])
        strings: List[str] = []
        for i in range(len(offsets) - 1):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            strings.append(payload[start:end].decode("utf-8", errors="replace"))
        return RustStringTensor(strings)

    def __len__(self) -> int:
        return len(self._strings)

    def __getitem__(self, index: int) -> str:
        return self._strings[index]

    def total_bytes(self) -> int:
        return len(self._payload)

    def to_list(self) -> List[str]:
        return list(self._strings)

    def lengths(self) -> List[int]:
        return [self._offsets[i + 1] - self._offsets[i] for i in range(len(self._strings))]

    def is_inlined_mask(self) -> List[bool]:
        return [(self._offsets[i + 1] - self._offsets[i]) <= 12 for i in range(len(self._strings))]

    def get(self, idx: int) -> Optional[str]:
        if 0 <= idx < len(self._strings):
            return self._strings[idx]
        return None

    def slice(self, start: int, end: int) -> "RustStringTensor":
        return RustStringTensor(self._strings[start:end])

    def filter_by_length(self, min_bytes: int = 0, max_bytes: int = 0) -> "RustStringTensor":
        return RustStringTensor(
            [
                s
                for s in self._strings
                if len(s.encode("utf-8")) >= min_bytes
                and (max_bytes == 0 or len(s.encode("utf-8")) <= max_bytes)
            ]
        )

    def filter_by_prefix(self, prefix: str) -> "RustStringTensor":
        return RustStringTensor([s for s in self._strings if s.startswith(prefix)])

    def hash_all(self) -> List[int]:
        return [StreamDeduplicator.compute_hash(s) for s in self._strings]

    def to_offsets(self) -> List[int]:
        return list(self._offsets)

    def to_payload(self) -> bytes:
        return self._payload

    def get_gpu_buffers(self) -> Tuple[bytes, bytes]:
        off_bytes = struct.pack(f"<{len(self._offsets)}I", *self._offsets)
        return off_bytes, self._payload

    def to_gpu_buffers(self) -> Tuple[bytes, bytes]:
        return self.get_gpu_buffers()

StringTensor = RustStringTensor

__all__ = ["RustStringTensor", "StringTensor"]
