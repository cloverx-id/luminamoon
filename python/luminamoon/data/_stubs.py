from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

class StreamCoordinate:
    shard_id: int
    byte_offset: int
    row_group_idx: int
    page_idx: int
    row_offset_in_page: int
    global_row_idx: int

    def __init__(
        self,
        shard_id: int = 0,
        byte_offset: int = 0,
        row_group_idx: int = 0,
        page_idx: int = 0,
        row_offset_in_page: int = 0,
        global_row_idx: int = 0,
    ) -> None: ...
    def to_bytes(self) -> bytes: ...
    @staticmethod
    def from_bytes(raw: bytes) -> StreamCoordinate: ...

class StreamDeduplicator:
    def __init__(self, initial_hashes: Sequence[int] | None = None) -> None: ...
    @staticmethod
    def compute_hash(payload: bytes) -> int: ...
    def is_seen(self, payload: bytes) -> bool: ...
    def try_add(self, payload: bytes) -> bool: ...
    def filter_batch(self, records: Sequence[str]) -> list[str]: ...
    def commit(self) -> int: ...
    def rollback(self) -> int: ...
    def clear(self) -> None: ...
    def count(self) -> int: ...
    def save_wal(self, path: str) -> int: ...
    def load_wal(self, path: str) -> int: ...
    def export_hashes(self) -> list[int]: ...

class StateTracker:
    def __init__(self, meta_path: str) -> None: ...
    def read(self) -> StreamCoordinate: ...
    def update(
        self,
        shard_id: int = 0,
        byte_offset: int = 0,
        row_group_idx: int = 0,
        page_idx: int = 0,
        row_offset_in_page: int = 0,
        global_row_idx: int = 0,
        batch_id: int = 1,
    ) -> None: ...
    def commit(self, batch_id: int = 1) -> StreamCoordinate: ...
    def committed_batch_id(self) -> int: ...

class QualityFilter:
    def __init__(
        self,
        min_chars: int = 20,
        max_chars: int = 0,
        min_words: int = 3,
        min_entropy: float = 2.5,
        max_repetition_ratio: float = 0.35,
        max_symbol_ratio: float = 0.30,
    ) -> None: ...
    def is_high_quality(self, text: str) -> bool: ...
    def metrics(self, text: str) -> tuple[int, int, float, float, float, bool]: ...
    def filter_batch(self, texts: Sequence[str]) -> list[str]: ...
    def filter_flags(self, texts: Sequence[str]) -> list[bool]: ...

class MinHashDeduplicator:
    def __init__(
        self,
        threshold: float = 0.85,
        k_shingle: int = 5,
        num_perm: int = 64,
    ) -> None: ...
    def is_unique(self, text: str) -> bool: ...
    def similarity(self, text_a: str, text_b: str) -> float: ...
    def count(self) -> int: ...
    def clear(self) -> None: ...

class StringTensor:
    @staticmethod
    def from_list(strings: Sequence[str]) -> StringTensor: ...
    @staticmethod
    def from_gpu_buffers(offsets: Sequence[int], payload: bytes) -> StringTensor: ...
    def to_gpu_buffers(self) -> tuple[bytes, bytes]: ...
    def to_offsets(self) -> list[int]: ...
    def to_payload(self) -> bytes: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> str: ...
    def get(self, index: int) -> str | None: ...
    def to_list(self) -> list[str]: ...
    def lengths(self) -> list[int]: ...
    def is_inlined_mask(self) -> list[bool]: ...
    def total_bytes(self) -> int: ...
    def filter_by_length(self, min_len: int = 0, max_len: int = 0) -> StringTensor: ...
    def filter_by_prefix(self, prefix: str) -> StringTensor: ...
    def hash_all(self) -> list[int]: ...

def dataloader_cpu_isa() -> str: ...
def suggest_closest_matches(
    target: str,
    candidates: Sequence[str],
    max_suggestions: int = 3,
) -> list[str]: ...
def serialize_table_to_bytes(
    columns: Any,
    format: str = "parquet",
    hf_image_struct: bool = True,
) -> bytes: ...
def clean_single_record(
    raw: str,
    trim_whitespace: bool = True,
    normalize_newlines: bool = True,
    min_byte_len: int = 1,
    max_byte_len: int = 0,
    truncate_over_max: bool = False,
    json_extract_key: str | None = None,
) -> str | None: ...
def clean_batch_records(
    records: Sequence[str],
    trim_whitespace: bool = True,
    normalize_newlines: bool = True,
    min_byte_len: int = 1,
    max_byte_len: int = 0,
    truncate_over_max: bool = False,
    json_extract_key: str | None = None,
) -> list[str]: ...
def postprocess_and_dedup_records(
    records: Sequence[str],
    trim_whitespace: bool = True,
    normalize_newlines: bool = True,
    min_byte_len: int = 1,
    max_byte_len: int = 0,
    truncate_over_max: bool = False,
    json_extract_key: str | None = None,
    seen_hashes: Sequence[int] | None = None,
    deduplicate: bool = True,
) -> tuple[list[str], list[int]]: ...
def fused_pipeline_process_batch(
    records: Sequence[str] | None = None,
    trim_whitespace: bool = True,
    normalize_newlines: bool = True,
    min_byte_len: int = 0,
    max_byte_len: int = 0,
    truncate_over_max: bool = False,
    json_extract_key: str | None = None,
    filter_quality: bool = False,
    min_chars: int = 0,
    max_chars: int = 0,
    min_words: int = 0,
    min_entropy: float = 0.0,
    max_repetition_ratio: float = 1.0,
    max_symbol_ratio: float = 1.0,
    deduplicator: Any = None,
    commit_dedup: bool = False,
    raw_lines: Sequence[str] | None = None,
) -> StringTensor: ...

__all__ = [
    "MinHashDeduplicator",
    "QualityFilter",
    "StateTracker",
    "StreamCoordinate",
    "StreamDeduplicator",
    "StringTensor",
    "clean_batch_records",
    "clean_single_record",
    "dataloader_cpu_isa",
    "fused_pipeline_process_batch",
    "postprocess_and_dedup_records",
    "serialize_table_to_bytes",
    "suggest_closest_matches",
]
