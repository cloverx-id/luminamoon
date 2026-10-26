from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ._ffi import fused_pipeline_process_batch as _raw_fused_pipeline
from .string_tensor import StringTensor

if TYPE_CHECKING:
    from .dedup import StreamDeduplicator


def fused_process_batch(
    records: Sequence[str],
    *,
    trim_whitespace: bool = True,
    normalize_newlines: bool = True,
    min_byte_len: int | None = None,
    max_byte_len: int | None = None,
    truncate_over_max: bool = False,
    json_extract_key: str | None = None,
    filter_quality: bool = False,
    min_chars: int | None = None,
    max_chars: int | None = None,
    min_words: int | None = None,
    min_entropy: float | None = None,
    max_repetition_ratio: float | None = None,
    max_symbol_ratio: float | None = None,
    deduplicator: StreamDeduplicator | None = None,
    commit_dedup: bool = False,
) -> StringTensor:
    """Executes single-pass fused preprocessing in Rust without Python GIL.

    If any length, word, or heuristic bound is None, that rule is completely inactive.
    """
    raw_dedup = deduplicator._raw if deduplicator is not None and hasattr(deduplicator, "_raw") else None

    raw_result = _raw_fused_pipeline(
        records=list(records),
        trim_whitespace=trim_whitespace,
        normalize_newlines=normalize_newlines,
        min_byte_len=0 if min_byte_len is None else max(0, min_byte_len),
        max_byte_len=0 if max_byte_len is None else max(0, max_byte_len),
        truncate_over_max=truncate_over_max,
        json_extract_key=json_extract_key,
        filter_quality=filter_quality,
        min_chars=0 if min_chars is None else max(0, min_chars),
        max_chars=0 if max_chars is None else max(0, max_chars),
        min_words=0 if min_words is None else max(0, min_words),
        min_entropy=0.0 if min_entropy is None else float(min_entropy),
        max_repetition_ratio=1.0 if max_repetition_ratio is None else float(max_repetition_ratio),
        max_symbol_ratio=1.0 if max_symbol_ratio is None else float(max_symbol_ratio),
        deduplicator=raw_dedup,
        commit_dedup=commit_dedup,
    )

    if commit_dedup and deduplicator is not None and deduplicator._wal_path:
        deduplicator.save_wal()

    return StringTensor(raw_result)


class FusedPipeline:
    """Reusable high-throughput preprocessing pipeline executing entirely in Rust."""

    def __init__(
        self,
        *,
        trim_whitespace: bool = True,
        normalize_newlines: bool = True,
        min_byte_len: int | None = None,
        max_byte_len: int | None = None,
        truncate_over_max: bool = False,
        json_extract_key: str | None = None,
        filter_quality: bool = False,
        min_chars: int | None = None,
        max_chars: int | None = None,
        min_words: int | None = None,
        min_entropy: float | None = None,
        max_repetition_ratio: float | None = None,
        max_symbol_ratio: float | None = None,
        deduplicator: StreamDeduplicator | None = None,
        commit_dedup: bool = False,
    ) -> None:
        self.trim_whitespace = trim_whitespace
        self.normalize_newlines = normalize_newlines
        self.min_byte_len = min_byte_len
        self.max_byte_len = max_byte_len
        self.truncate_over_max = truncate_over_max
        self.json_extract_key = json_extract_key
        self.filter_quality = filter_quality
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_words = min_words
        self.min_entropy = min_entropy
        self.max_repetition_ratio = max_repetition_ratio
        self.max_symbol_ratio = max_symbol_ratio
        self.deduplicator = deduplicator
        self.commit_dedup = commit_dedup

    def __call__(self, records: Sequence[str]) -> StringTensor:
        """Processes a batch of records via fused single-pass execution."""
        return fused_process_batch(
            records,
            trim_whitespace=self.trim_whitespace,
            normalize_newlines=self.normalize_newlines,
            min_byte_len=self.min_byte_len,
            max_byte_len=self.max_byte_len,
            truncate_over_max=self.truncate_over_max,
            json_extract_key=self.json_extract_key,
            filter_quality=self.filter_quality,
            min_chars=self.min_chars,
            max_chars=self.max_chars,
            min_words=self.min_words,
            min_entropy=self.min_entropy,
            max_repetition_ratio=self.max_repetition_ratio,
            max_symbol_ratio=self.max_symbol_ratio,
            deduplicator=self.deduplicator,
            commit_dedup=self.commit_dedup,
        )


__all__ = ["FusedPipeline", "fused_process_batch"]
