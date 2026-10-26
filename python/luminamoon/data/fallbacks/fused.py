from __future__ import annotations

from typing import Any, List, Optional, Sequence

from .cleaner import TextCleaner, clean_single_record
from .dedup import StreamDeduplicator
from .masker import PIIMasker
from .quality import QualityFilter
from .string_tensor import RustStringTensor, StringTensor


def fused_pipeline_process_batch(
    raw_lines: Optional[Sequence[str]] = None,
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
    pii_masker: Optional[PIIMasker] = None,
    json_key: Optional[str] = None,
    records: Optional[Sequence[str]] = None,
) -> RustStringTensor:
    input_records = raw_lines if raw_lines is not None else (records or [])
    cleaner = TextCleaner(
        trim_whitespace=trim_whitespace,
        normalize_newlines=normalize_newlines,
        min_byte_len=min_byte_len,
        max_byte_len=max_byte_len,
        truncate_over_max=truncate_over_max,
        json_extract_key=json_extract_key or json_key,
    )
    qfilter = QualityFilter(
        min_chars=min_chars,
        max_chars=max_chars,
        min_words=min_words,
        min_entropy=min_entropy,
        max_repetition_ratio=max_repetition_ratio,
        max_symbol_ratio=max_symbol_ratio,
    )
    results: List[str] = []
    for raw in input_records:
        text: Optional[str] = raw
        if cleaner.json_extract_key is not None:
            text = cleaner.extract_json(raw, cleaner.json_extract_key)
            if text is None:
                continue
        cleaned = cleaner.clean(text)
        if not cleaned:
            continue
        if filter_quality and not qfilter.passes(cleaned):
            continue
        if deduplicator is not None and not deduplicator.check_and_insert(cleaned):
            continue
        if pii_masker is not None:
            cleaned = pii_masker.sanitize(cleaned)
        results.append(cleaned)
    return RustStringTensor(results)


def postprocess_and_dedup_records(
    records: Sequence[str],
    trim_whitespace: bool = True,
    normalize_newlines: bool = True,
    min_byte_len: int = 1,
    max_byte_len: int = 0,
    truncate_over_max: bool = False,
    json_extract_key: Optional[str] = None,
    seen_hashes: Optional[Sequence[int]] = None,
    deduplicate: bool = True,
) -> tuple[List[str], List[int]]:
    cleaned_all: List[str] = []
    for raw in records:
        cleaned = clean_single_record(
            raw,
            trim_whitespace=trim_whitespace,
            normalize_newlines=normalize_newlines,
            min_byte_len=min_byte_len,
            max_byte_len=max_byte_len,
            truncate_over_max=truncate_over_max,
            json_extract_key=json_extract_key,
        )
        if cleaned is not None:
            cleaned_all.append(cleaned)

    if not deduplicate:
        return (cleaned_all, [])

    seen: set[int] = set(seen_hashes or ())
    cleaned_unique: List[str] = []
    newly_committed: List[int] = []
    for cleaned in cleaned_all:
        h = StreamDeduplicator.compute_hash(cleaned)
        if h not in seen:
            seen.add(h)
            cleaned_unique.append(cleaned)
            newly_committed.append(h)

    return (cleaned_unique, newly_committed)


__all__ = ["fused_pipeline_process_batch", "postprocess_and_dedup_records"]
