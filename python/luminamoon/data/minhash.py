from __future__ import annotations

from collections.abc import Sequence
from typing import Any, overload

from ._ffi import MinHashDeduplicator as _RawMinHashDeduplicator


class MinHashDeduplicator:
    """Fuzzy near-duplicate detector using fast single-hash MinHash + auto-tuned LSH in Rust.

    Features:
    - Single 128-bit XXH3 hash per shingle expanded to `num_perm` universal permutations.
    - Auto-tuned `(num_bands, rows_per_band)` aligned to `threshold` on the LSH S-curve.
    - Parallel Rayon batch filtering without Python GIL (`filter_batch`).
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
        """Filters a sequence of texts in parallel using Rust Rayon + LSH."""
        if not texts:
            return []
        if hasattr(self._raw, "filter_batch"):
            return self._raw.filter_batch(list(texts))
        return [t for t in texts if self.is_unique(t)]

    @property
    def num_bands(self) -> int:
        raw_val: Any = getattr(self._raw, "num_bands", None)
        val: Any = raw_val() if callable(raw_val) else raw_val
        return int(val) if val else 8

    @property
    def rows_per_band(self) -> int:
        raw_val: Any = getattr(self._raw, "rows_per_band", None)
        val: Any = raw_val() if callable(raw_val) else raw_val
        return int(val) if val else max(1, self.num_perm // self.num_bands)

    @property
    def count(self) -> int:
        """Total number of unique document signatures currently tracked."""
        return self._raw.count()

    def clear(self) -> None:
        """Clears all stored signatures."""
        self._raw.clear()

    @overload
    def __call__(self, target: str) -> bool: ...  # pyright: ignore[reportOverlappingOverload]

    @overload
    def __call__(self, target: Sequence[str]) -> list[str]: ...

    def __call__(self, target: str | Sequence[str]) -> bool | list[str]:
        if isinstance(target, str):
            return self.is_unique(target)
        return self.filter_batch(target)


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
