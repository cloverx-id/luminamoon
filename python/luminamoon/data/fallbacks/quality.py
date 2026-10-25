from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

class QualityFilter:
    def __init__(
        self,
        min_chars: int = 20,
        max_chars: int = 100_000,
        min_words: int = 3,
        min_entropy: float = 2.5,
        max_repetition_ratio: float = 0.35,
        max_symbol_ratio: float = 0.25,
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_words = min_words
        self.min_entropy = min_entropy
        self.max_repetition_ratio = max_repetition_ratio
        self.max_symbol_ratio = max_symbol_ratio

    def compute_metrics(self, text: str) -> Tuple[int, int, float, float, float]:
        char_len = len(text)
        if char_len == 0:
            return (0, 0, 0.0, 0.0, 0.0)
        words = text.split()
        word_count = len(words)
        counts: Dict[str, int] = {}
        for ch in text:
            counts[ch] = counts.get(ch, 0) + 1
        entropy = -sum((c / char_len) * math.log2(c / char_len) for c in counts.values())
        non_ws = [ch for ch in text if not ch.isspace()]
        if not non_ws:
            sym_ratio = 0.0
        else:
            sym_count = sum(1 for ch in non_ws if not ch.isalnum())
            sym_ratio = sym_count / len(non_ws)
        if word_count < 4:
            rep_ratio = 0.0
        else:
            total_trigrams = word_count - 2
            unique_trigrams = len(
                {(words[i], words[i + 1], words[i + 2]) for i in range(total_trigrams)}
            )
            rep_ratio = 1.0 - (unique_trigrams / total_trigrams)
        return (char_len, word_count, float(entropy), float(rep_ratio), float(sym_ratio))

    def passes(self, text: str) -> bool:
        char_len, word_count, entropy, rep_ratio, sym_ratio = self.compute_metrics(text)
        if self.min_chars > 0 and char_len < self.min_chars:
            return False
        if self.max_chars > 0 and char_len > self.max_chars:
            return False
        if self.min_words > 0 and word_count < self.min_words:
            return False
        if self.min_entropy > 0.0 and entropy < self.min_entropy:
            return False
        if self.max_repetition_ratio < 1.0 and rep_ratio > self.max_repetition_ratio:
            return False
        if self.max_symbol_ratio < 1.0 and sym_ratio > self.max_symbol_ratio:
            return False
        return True

    def is_high_quality(self, text: str) -> bool:
        return self.passes(text)

    def metrics(self, text: str) -> Tuple[int, int, float, float, float, bool]:
        char_len, word_count, entropy, rep_ratio, sym_ratio = self.compute_metrics(text)
        passed = self.passes(text)
        return (char_len, word_count, entropy, rep_ratio, sym_ratio, passed)

    def check(self, text: str) -> bool:
        return self.passes(text)

    def passes_batch(self, texts: Sequence[str]) -> List[bool]:
        return [self.passes(t) for t in texts]

    def filter_flags(self, texts: Sequence[str]) -> List[bool]:
        return [self.passes(t) for t in texts]

    def filter_batch(self, texts: Sequence[str]) -> List[str]:
        return [t for t in texts if self.passes(t)]

__all__ = ["QualityFilter"]
