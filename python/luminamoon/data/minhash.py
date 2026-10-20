from __future__ import annotations

from collections.abc import Sequence

from ._ffi import MinHashDeduplicator as _RawMinHashDeduplicator


class MinHashDeduplicator:
    """Fuzzy near-duplicate detector using MinHash and Locality Sensitive Hashing in Rust.

    Features:
    - Character shingling (n-grams) with XXH3-64 permutations.
    - Estimates Jaccard similarity without storing raw document text.
    - Filters out documents with similarity >= threshold (e.g. 0.85).
    """

    def __init__(
        self,
        threshold: float = 0.85,
        k_shingle: int = 5,
        num_perm: int = 64,
    ) -> None:
        self.threshold = float(threshold)
        self.k_shingle = int(k_shingle)
        self.num_perm = int(num_perm)
        self._raw = _RawMinHashDeduplicator(
            threshold=self.threshold,
            k_shingle=self.k_shingle,
            num_perm=self.num_perm,
        )

    def is_unique(self, text: str) -> bool:
        """Returns True if text is sufficiently unique (< threshold similarity to all seen)."""
        return self._raw.is_unique(text)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Estimates Jaccard similarity between two texts using MinHash signatures."""
        return self._raw.similarity(text_a, text_b)

    def filter_batch(self, texts: Sequence[str]) -> list[str]:
        """Filters a sequence of texts, returning only those that are not near-duplicates."""
        return [t for t in texts if self.is_unique(t)]

    @property
    def count(self) -> int:
        """Total number of unique document signatures currently tracked."""
        return self._raw.count()

    def clear(self) -> None:
        """Clears all stored signatures."""
        self._raw.clear()


def jaccard_similarity(
    text_a: str,
    text_b: str,
    k_shingle: int = 5,
    num_perm: int = 64,
) -> float:
    """Convenience helper to estimate Jaccard similarity between two texts."""
    dedup = MinHashDeduplicator(threshold=0.85, k_shingle=k_shingle, num_perm=num_perm)
    return dedup.similarity(text_a, text_b)


__all__ = ["MinHashDeduplicator", "jaccard_similarity"]
