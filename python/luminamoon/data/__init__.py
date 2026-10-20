from __future__ import annotations

from .version import get_version, set_version

from ._ffi import (
    dataloader_cpu_isa as data_cpu_isa,
    postprocess_and_dedup_records,
    suggest_closest_matches,
)
from .adapters import ResumablePipeline, create_resumable_collate
from .checkpoint import StateTracker, StreamCoordinate
from .cleaner import TextCleaner, clean_batch, clean_text
from .dedup import StreamDeduplicator, hash_record
from .fused import FusedPipeline, fused_process_batch
from .masker import PIIMasker, mask_pii
from .minhash import MinHashDeduplicator, jaccard_similarity
from .quality import QualityFilter, filter_quality, is_high_quality
from .serialization import (
    serialize_to_arrow,
    serialize_to_jsonl,
    serialize_to_parquet,
)
from .string_tensor import StringTensor

__all__ = [
    "MinHashDeduplicator",
    "PIIMasker",
    "QualityFilter",
    "ResumablePipeline",
    "StateTracker",
    "StreamCoordinate",
    "StreamDeduplicator",
    "StringTensor",
    "TextCleaner",
    "clean_batch",
    "clean_text",
    "create_resumable_collate",
    "data_cpu_isa",
    "filter_quality",
    "fused_process_batch",
    "FusedPipeline",
    "hash_record",
    "is_high_quality",
    "jaccard_similarity",
    "mask_pii",
    "postprocess_and_dedup_records",
    "serialize_to_arrow",
    "serialize_to_jsonl",
    "serialize_to_parquet",
    "suggest_closest_matches",
    "get_version",
    "set_version",
]
