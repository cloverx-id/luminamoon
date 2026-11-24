from __future__ import annotations

from typing import List, Optional, Sequence

from .cleaner import clean_single_record
from .dedup import StreamDeduplicator
from .quality import QualityFilter
from .string_tensor import RustStringTensor


def fused_pipeline_process_batch_indexed(
    records: Sequence[str],
    trim_whitespace: bool = True,
    normalize_newlines: bool = True,
    min_byte_len: int = 0,
    max_byte_len: int = 0,
    truncate_over_max: bool = False,
    json_extract_key: Optional[str] = None,
    filter_quality: bool = False,
    min_chars: int = 0,
    max_chars: int = 0,
    min_words: int = 0,
    min_entropy: float = 0.0,
    max_repetition_ratio: float = 1.0,
    max_symbol_ratio: float = 1.0,
    deduplicator: Optional[StreamDeduplicator] = None,
    commit_dedup: bool = False,
    batch_id: int = 1,
) -> tuple[RustStringTensor, List[int]]:
    qfilter = QualityFilter(
        min_chars=min_chars,
        max_chars=max_chars,
        min_words=min_words,
        min_entropy=min_entropy,
        max_repetition_ratio=max_repetition_ratio,
        max_symbol_ratio=max_symbol_ratio,
    )
    results: List[str] = []
    indices: List[int] = []
    for idx, raw in enumerate(records):
        cleaned = clean_single_record(
            raw,
            trim_whitespace=trim_whitespace,
            normalize_newlines=normalize_newlines,
            min_byte_len=min_byte_len,
            max_byte_len=max_byte_len,
            truncate_over_max=truncate_over_max,
            json_extract_key=json_extract_key,
        )
        if cleaned is None:
            continue
        if filter_quality and not qfilter.passes(cleaned):
            continue
        if deduplicator is not None and not deduplicator.try_add(
            cleaned.encode("utf-8"),
            batch_id=batch_id,
        ):
            continue
        results.append(cleaned)
        indices.append(idx)
    if commit_dedup and deduplicator is not None:
        deduplicator.commit(batch_id)
    return RustStringTensor(results), indices


def fused_pipeline_process_batch(
    records: Sequence[str],
    trim_whitespace: bool = True,
    normalize_newlines: bool = True,
    min_byte_len: int = 0,
    max_byte_len: int = 0,
    truncate_over_max: bool = False,
    json_extract_key: Optional[str] = None,
    filter_quality: bool = False,
    min_chars: int = 0,
    max_chars: int = 0,
    min_words: int = 0,
    min_entropy: float = 0.0,
    max_repetition_ratio: float = 1.0,
    max_symbol_ratio: float = 1.0,
    deduplicator: Optional[StreamDeduplicator] = None,
    commit_dedup: bool = False,
    batch_id: int = 1,
) -> RustStringTensor:
    st, _ = fused_pipeline_process_batch_indexed(
        records=records,
        trim_whitespace=trim_whitespace,
        normalize_newlines=normalize_newlines,
        min_byte_len=min_byte_len,
        max_byte_len=max_byte_len,
        truncate_over_max=truncate_over_max,
        json_extract_key=json_extract_key,
        filter_quality=filter_quality,
        min_chars=min_chars,
        max_chars=max_chars,
        min_words=min_words,
        min_entropy=min_entropy,
        max_repetition_ratio=max_repetition_ratio,
        max_symbol_ratio=max_symbol_ratio,
        deduplicator=deduplicator,
        commit_dedup=commit_dedup,
        batch_id=batch_id,
    )
    return st


__all__ = [
    "fused_pipeline_process_batch",
    "fused_pipeline_process_batch_indexed",
]
