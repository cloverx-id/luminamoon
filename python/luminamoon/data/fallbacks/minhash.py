from __future__ import annotations

import hashlib
import math
from typing import Any, List, Optional, Sequence


def _choose_optimal_lsh_params(num_perm: int, threshold: float) -> tuple[int, int]:
    t = max(0.05, min(0.99, threshold))
    best_b, best_r, best_err = 1, max(1, num_perm), float("inf")
    for b in range(1, num_perm + 1):
        r = num_perm // b
        if r == 0:
            continue
        inflection = (1.0 / b) ** (1.0 / r)
        err = abs(inflection - t)
        if err + 1e-9 < best_err or (abs(err - best_err) <= 1e-9 and b * r > best_b * best_r):
            best_err = err
            best_b = b
            best_r = r
    return best_b, best_r


class MinHashDeduplicator:
    def __init__(
        self,
        threshold: float = 0.85,
        shingle_size: int = 5,
        num_perm: int = 64,
        num_bands: Optional[int] = None,
        k_shingle: Optional[int] = None,
    ) -> None:
        t = float(threshold)
        self.threshold = 0.85 if math.isnan(t) else max(0.0, min(1.0, t))
        self.shingle_size = max(1, k_shingle if k_shingle is not None else shingle_size)
        self.num_perm = max(8, int(num_perm))
        if num_bands is None:
            self.num_bands, self.rows_per_band = _choose_optimal_lsh_params(self.num_perm, self.threshold)
        else:
            nb = int(num_bands)
            if nb <= 0 or nb > self.num_perm or (self.num_perm % nb != 0):
                raise ValueError(
                    f"num_bands ({nb}) must be in [1, num_perm ({self.num_perm})] and divide num_perm evenly."
                )
            self.num_bands = nb
            self.rows_per_band = self.num_perm // self.num_bands
        self._signatures: List[tuple[List[int], bool]] = []
        self._buckets: List[dict[tuple[int, ...], List[int]]] = [{} for _ in range(self.num_bands)]
        self._count = 0

    def _signature(self, text: str) -> tuple[List[int], bool]:
        if not text:
            return ([0xFFFF_FFFF_FFFF_FFFF] * self.num_perm, True)
        k = min(self.shingle_size, len(text))
        mins = [0xFFFF_FFFF_FFFF_FFFF] * self.num_perm
        mask64 = 0xFFFF_FFFF_FFFF_FFFF
        for i in range(len(text) - k + 1):
            digest = hashlib.blake2b(text[i : i + k].encode("utf-8"), digest_size=16).digest()
            h1 = int.from_bytes(digest[:8], "little")
            h2 = int.from_bytes(digest[8:], "little") | 1
            acc = h1
            for p in range(self.num_perm):
                z = (acc ^ (acc >> 33)) & mask64
                z = (z * 0xFF51AFD7ED558CCD) & mask64
                z = (z ^ (z >> 33)) & mask64
                if z < mins[p]:
                    mins[p] = z
                acc = (acc + h2) & mask64
        return (mins, False)

    @staticmethod
    def _sig_jaccard(a: tuple[List[int], bool], b: tuple[List[int], bool]) -> float:
        sig_a, empty_a = a
        sig_b, empty_b = b
        if empty_a or empty_b:
            return 1.0 if (empty_a and empty_b) else 0.0
        if not sig_a or len(sig_a) != len(sig_b):
            return 0.0
        matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
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
            a_set = {tuple(sig) for sig, _ in self._signatures}
            b_set = {tuple(sig) for sig, _ in text_a._signatures}
            union = a_set | b_set
            return (len(a_set & b_set) / float(len(union))) if union else 0.0
        raise TypeError(f"Unsupported argument type for jaccard_similarity: {type(text_a)}")

    def _max_similarity(self, sig_pair: tuple[List[int], bool]) -> float:
        if not self._signatures:
            return 0.0
        if len(self._signatures) <= 128:
            max_sim = 0.0
            for existing in self._signatures:
                sim = self._sig_jaccard(sig_pair, existing)
                if sim > max_sim:
                    max_sim = sim
                    if max_sim >= self.threshold:
                        return max_sim
            return max_sim

        sig, _ = sig_pair
        candidates: set[int] = set()
        for b in range(self.num_bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            doc_ids = self._buckets[b].get(tuple(sig[start:end]))
            if doc_ids:
                candidates.update(doc_ids)

        max_sim = 0.0
        for cid in candidates:
            sim = self._sig_jaccard(sig_pair, self._signatures[cid])
            if sim > max_sim:
                max_sim = sim
                if max_sim >= self.threshold:
                    return max_sim
        return max_sim

    def check_and_insert(self, text: str) -> bool:
        sig_pair = self._signature(text)
        if self._signatures and self._max_similarity(sig_pair) >= self.threshold:
            return False
        doc_id = len(self._signatures)
        sig, _ = sig_pair
        for b in range(self.num_bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            self._buckets[b].setdefault(tuple(sig[start:end]), []).append(doc_id)
        self._signatures.append(sig_pair)
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
