from __future__ import annotations

import hashlib
from typing import Any, List, Optional, Sequence

class MinHashDeduplicator:
    def __init__(
        self,
        threshold: float = 0.80,
        shingle_size: int = 5,
        num_perm: int = 64,
        num_bands: int = 16,
        k_shingle: Optional[int] = None,
    ) -> None:
        self.threshold = threshold
        self.shingle_size = k_shingle if k_shingle is not None else max(1, shingle_size)
        self.num_perm = max(1, num_perm)
        self.num_bands = max(1, min(num_bands, self.num_perm))
        self.rows_per_band = max(1, self.num_perm // self.num_bands)
        self._buckets: List[set[int]] = [set() for _ in range(self.num_bands)]
        self._count = 0

    def _signature(self, text: str) -> List[int]:
        if not text:
            return [0] * self.num_perm
        k = min(self.shingle_size, len(text))
        shingles = [text[i : i + k].encode("utf-8") for i in range(len(text) - k + 1)]
        sig: List[int] = []
        for p in range(self.num_perm):
            prefix = p.to_bytes(4, "little")
            m = min(
                int.from_bytes(hashlib.blake2b(prefix + s, digest_size=8).digest(), "little")
                for s in shingles
            )
            sig.append(m)
        return sig

    def jaccard_estimate(self, text_a: str, text_b: str) -> float:
        sig_a = self._signature(text_a)
        sig_b = self._signature(text_b)
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / float(self.num_perm)

    def similarity(self, text_a: str, text_b: str) -> float:
        return self.jaccard_estimate(text_a, text_b)

    def jaccard_similarity(self, other: Any) -> float:
        return 0.0

    def check_and_insert(self, text: str) -> bool:
        sig = self._signature(text)
        band_hashes: List[int] = []
        for b in range(self.num_bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            band_tuple = tuple(sig[start:end])
            bh = hash(band_tuple)
            if bh in self._buckets[b]:
                return False
            band_hashes.append(bh)
        for b, bh in enumerate(band_hashes):
            self._buckets[b].add(bh)
        self._count += 1
        return True

    def is_near_duplicate(self, text: str) -> bool:
        return not self.check_and_insert(text)

    def is_unique(self, text: str) -> bool:
        return self.check_and_insert(text)

    def filter_batch(self, texts: Sequence[str]) -> List[str]:
        return [t for t in texts if self.check_and_insert(t)]

    def clear(self) -> None:
        for b in self._buckets:
            b.clear()
        self._count = 0

    def count(self) -> int:
        return self._count

    def __len__(self) -> int:
        return self._count

__all__ = ["MinHashDeduplicator"]
