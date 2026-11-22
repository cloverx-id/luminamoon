from __future__ import annotations

import hashlib
from typing import Any, List, Optional, Sequence

class MinHashDeduplicator:
    def __init__(
        self,
        threshold: float = 0.80,
        shingle_size: int = 5,
        num_perm: int = 64,
        num_bands: Optional[int] = None,
        k_shingle: Optional[int] = None,
    ) -> None:
        self.threshold = max(0.0, min(1.0, float(threshold)))
        self.shingle_size = max(1, k_shingle if k_shingle is not None else shingle_size)
        self.num_perm = max(8, int(num_perm))
        if num_bands is None:
            if self.num_perm % 16 == 0:
                self.num_bands = 16
            elif self.num_perm % 8 == 0:
                self.num_bands = 8
            elif self.num_perm % 4 == 0:
                self.num_bands = 4
            else:
                self.num_bands = 1
        else:
            nb = int(num_bands)
            if nb <= 0 or nb > self.num_perm or (self.num_perm % nb != 0):
                raise ValueError(
                    f"num_bands ({nb}) must be in [1, num_perm ({self.num_perm})] and divide num_perm evenly."
                )
            self.num_bands = nb
        self.rows_per_band = self.num_perm // self.num_bands
        self._signatures: List[List[int]] = []
        self._buckets: List[dict[tuple[int, ...], List[int]]] = [{} for _ in range(self.num_bands)]
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

    @staticmethod
    def _sig_jaccard(sig_a: List[int], sig_b: List[int]) -> float:
        if not sig_a or len(sig_a) != len(sig_b):
            return 0.0
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / float(len(sig_a))

    def jaccard_estimate(self, text_a: str, text_b: str) -> float:
        return self._sig_jaccard(self._signature(text_a), self._signature(text_b))

    def similarity(self, text_a: str, text_b: str) -> float:
        return self.jaccard_estimate(text_a, text_b)

    def jaccard_similarity(self, text_a: Any, text_b: Optional[str] = None) -> float:
        if text_b is not None:
            return self.jaccard_estimate(str(text_a), str(text_b))
        if isinstance(text_a, str):
            return self._max_similarity(self._signature(text_a))
        if isinstance(text_a, MinHashDeduplicator):
            a_set = {tuple(sig) for sig in self._signatures}
            b_set = {tuple(sig) for sig in text_a._signatures}
            union = a_set | b_set
            return (len(a_set & b_set) / float(len(union))) if union else 0.0
        raise TypeError(f"Unsupported argument type for jaccard_similarity: {type(text_a)}")

    def _max_similarity(self, sig: List[int]) -> float:
        if not self._signatures:
            return 0.0
        if len(self._signatures) <= 128:
            max_sim = 0.0
            for existing in self._signatures:
                sim = self._sig_jaccard(sig, existing)
                if sim > max_sim:
                    max_sim = sim
                    if max_sim >= self.threshold:
                        return max_sim
            return max_sim

        candidates: set[int] = set()
        for b in range(self.num_bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            key = tuple(sig[start:end])
            doc_ids = self._buckets[b].get(key)
            if doc_ids:
                candidates.update(doc_ids)

        max_sim = 0.0
        for cid in candidates:
            sim = self._sig_jaccard(sig, self._signatures[cid])
            if sim > max_sim:
                max_sim = sim
                if max_sim >= self.threshold:
                    return max_sim
        return max_sim

    def check_and_insert(self, text: str) -> bool:
        sig = self._signature(text)
        if self._max_similarity(sig) >= self.threshold:
            return False
        doc_id = len(self._signatures)
        for b in range(self.num_bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            key = tuple(sig[start:end])
            self._buckets[b].setdefault(key, []).append(doc_id)
        self._signatures.append(sig)
        self._count = len(self._signatures)
        return True

    def is_near_duplicate(self, text: str) -> bool:
        return not self.check_and_insert(text)

    def is_unique(self, text: str) -> bool:
        return self.check_and_insert(text)

    def filter_batch(self, texts: Sequence[str]) -> List[str]:
        return [t for t in texts if self.check_and_insert(t)]

    def clear(self) -> None:
        self._signatures.clear()
        for b in self._buckets:
            b.clear()
        self._count = 0

    def count(self) -> int:
        return self._count

    def __len__(self) -> int:
        return self._count

__all__ = ["MinHashDeduplicator"]
