from __future__ import annotations

from collections.abc import Sequence
from typing import Any, overload

from ._ffi import QualityFilter as _RawQualityFilter


class QualityFilter:
    """Heuristic quality filtering engine for pre-training corpora backed by Rust.

    Features:
    - Shannon Character Entropy ($H = -\\sum p \\log_2 p$) to detect gibberish/noise.
    - Word-level n-gram repetition ratio to eliminate degenerate looping.
    - Symbol-to-word ratio to eliminate code/log dumps and broken formatting.
    - Rayon multi-threaded parallel batch evaluation with Python GIL detached.
    """

    def __init__(
        self,
        min_chars: int | None = None,
        max_chars: int | None = None,
        min_words: int | None = None,
        min_entropy: float | None = None,
        max_repetition_ratio: float | None = None,
        max_symbol_ratio: float | None = None,
    ) -> None:
        self.min_chars = 0 if min_chars is None else max(0, min_chars)
        self.max_chars = 0 if max_chars is None else max(0, max_chars)
        self.min_words = 0 if min_words is None else max(0, min_words)
        self.min_entropy = 0.0 if min_entropy is None else float(min_entropy)
        self.max_repetition_ratio = 1.0 if max_repetition_ratio is None else float(max_repetition_ratio)
        self.max_symbol_ratio = 1.0 if max_symbol_ratio is None else float(max_symbol_ratio)
        self._raw = _RawQualityFilter(
            min_chars=self.min_chars,
            max_chars=self.max_chars,
            min_words=self.min_words,
            min_entropy=self.min_entropy,
            max_repetition_ratio=self.max_repetition_ratio,
            max_symbol_ratio=self.max_symbol_ratio,
        )

    @classmethod
    def for_pretraining(
        cls,
        min_chars: int | None = 20,
        max_chars: int | None = None,
        min_words: int | None = 3,
        min_entropy: float | None = 2.5,
        max_repetition_ratio: float | None = 0.35,
        max_symbol_ratio: float | None = 0.30,
    ) -> QualityFilter:
        """Pre-configured heuristic profile for LLM pre-training web scrape filtering."""
        return cls(
            min_chars=min_chars,
            max_chars=max_chars,
            min_words=min_words,
            min_entropy=min_entropy,
            max_repetition_ratio=max_repetition_ratio,
            max_symbol_ratio=max_symbol_ratio,
        )

    def is_high_quality(self, text: str) -> bool:
        """Evaluates whether a single text passes all quality thresholds."""
        return self._raw.is_high_quality(text)

    def metrics(self, text: str) -> dict[str, Any]:
        """Returns detailed mathematical quality metrics for a text."""
        char_count, word_count, entropy, repetition, symbols, passed = self._raw.metrics(text)
        return {
            "char_count": char_count,
            "word_count": word_count,
            "entropy": round(entropy, 4),
            "repetition_ratio": round(repetition, 4),
            "symbol_ratio": round(symbols, 4),
            "is_passed": passed,
        }

    def filter_batch(self, texts: Sequence[str]) -> list[str]:
        """Filters a batch of texts, returning only items passing all quality checks."""
        if not texts:
            return []
        return self._raw.filter_batch(texts)

    def filter_flags(self, texts: Sequence[str]) -> list[bool]:
        """Returns a boolean mask of pass/fail decisions for each text in the batch."""
        if not texts:
            return []
        return self._raw.filter_flags(texts)

    @overload
    def __call__(self, target: str) -> bool: ...  # pyright: ignore[reportOverlappingOverload]

    @overload
    def __call__(self, target: Sequence[str]) -> list[str]: ...

    def __call__(self, target: str | Sequence[str]) -> bool | list[str]:
        if isinstance(target, str):
            return self.is_high_quality(target)
        return self.filter_batch(target)


def is_high_quality(
    text: str,
    min_chars: int | None = None,
    max_chars: int | None = None,
    min_words: int | None = None,
    min_entropy: float | None = None,
    max_repetition_ratio: float | None = None,
    max_symbol_ratio: float | None = None,
) -> bool:
    """Convenience helper to evaluate a single text."""
    qf = QualityFilter(
        min_chars=min_chars,
        max_chars=max_chars,
        min_words=min_words,
        min_entropy=min_entropy,
        max_repetition_ratio=max_repetition_ratio,
        max_symbol_ratio=max_symbol_ratio,
    )
    return qf.is_high_quality(text)


def filter_quality(
    texts: Sequence[str],
    min_chars: int | None = None,
    max_chars: int | None = None,
    min_words: int | None = None,
    min_entropy: float | None = None,
    max_repetition_ratio: float | None = None,
    max_symbol_ratio: float | None = None,
) -> list[str]:
    """Convenience helper to filter a batch of texts."""
    qf = QualityFilter(
        min_chars=min_chars,
        max_chars=max_chars,
        min_words=min_words,
        min_entropy=min_entropy,
        max_repetition_ratio=max_repetition_ratio,
        max_symbol_ratio=max_symbol_ratio,
    )
    return qf.filter_batch(texts)


__all__ = ["QualityFilter", "filter_quality", "is_high_quality"]
