from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any


def _py_damerau_levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    width = lb + 1
    matrix = [0] * ((la + 1) * width)
    for i in range(la + 1):
        matrix[i * width] = i
    for j in range(lb + 1):
        matrix[j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            best = min(
                matrix[(i - 1) * width + j] + 1,
                matrix[i * width + (j - 1)] + 1,
                matrix[(i - 1) * width + (j - 1)] + cost,
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                best = min(best, matrix[(i - 2) * width + (j - 2)] + 1)
            matrix[i * width + j] = best
    return matrix[la * width + lb]


def _py_find_closest_matches(
    query: str,
    candidates: Sequence[str],
    max_results: int = 3,
) -> list[str]:
    q = query.strip().lower()
    if not q or max_results <= 0:
        return []
    q_len = len(q)
    threshold = 1 if q_len <= 3 else (2 if q_len <= 6 else 3)
    scored: list[tuple[int, str]] = []
    for cand in candidates:
        c_raw = str(cand).strip()
        if not c_raw:
            continue
        c = c_raw.lower()
        dist = _py_damerau_levenshtein(q, c)
        is_substr = len(q) >= 3 and (q in c or c in q)
        if dist <= threshold or is_substr:
            effective = min(dist, threshold) if is_substr else dist
            scored.append((effective, c_raw))
    scored.sort(key=lambda item: (item[0], item[1]))
    deduped: list[str] = []
    for _, name in scored:
        if name not in deduped:
            deduped.append(name)
            if len(deduped) >= max_results:
                break
    return deduped


if TYPE_CHECKING:

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

    class _RawDataBatch:
        batch_id: int
        records: list[str]
        next_coord: StreamCoordinate
        is_eof: bool

        def __len__(self) -> int: ...

    class _RawMultimodalBatch:
        batch_id: int
        num_rows: int
        next_coord: StreamCoordinate
        is_eof: bool

        def column_names(self) -> list[str]: ...
        def column(self, name: str) -> list[Any]: ...
        def to_dict(self) -> dict[str, list[Any]]: ...
        def __len__(self) -> int: ...

    class _RawDataStreamLoader:
        def __new__(cls, *args: Any, **kwargs: Any) -> Any: ...
        def committed_batch_id(self) -> int: ...
        def committed_coordinate(self) -> StreamCoordinate: ...
        def wait_for_refill(self, timeout_ms: int = 5000) -> int: ...
        def commit_gpu_batch(self, batch_id: int) -> StreamCoordinate: ...
        def next_batch(self) -> Any: ...
        def download_workers(self) -> int: ...
        def close(self) -> None: ...

    class _RawMultimodalParquetLoader:
        def __new__(cls, *args: Any, **kwargs: Any) -> Any: ...
        def next_batch(self) -> Any: ...
        def export_committed_hashes(self) -> list[int]: ...
        def import_committed_hashes(self, hashes: Sequence[int]) -> None: ...

    class HfHubUploader:
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...
        def create_repo(self, *args: Any, **kwargs: Any) -> str: ...
        def upload_file(self, *args: Any, **kwargs: Any) -> str: ...
        def upload_bytes(self, *args: Any, **kwargs: Any) -> str: ...
        def upload_table(self, *args: Any, **kwargs: Any) -> str: ...
        def upload_jsonl(self, *args: Any, **kwargs: Any) -> str: ...

    def dataloader_cpu_isa() -> str: ...
    def suggest_closest_matches(
        query: str,
        candidates: Sequence[str],
        max_results: int = 3,
    ) -> list[str]: ...
    def serialize_table_to_bytes(
        columns: Any,
        path_in_repo: str = "part.parquet",
        hf_image_struct: bool = True,
    ) -> bytes: ...
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

else:
    try:
        from luminamoon.optimizer._luminamoon_optimizer_rs import (
            DataBatch as _RawDataBatch,
            DataStreamLoader as _RawDataStreamLoader,
            HfHubUploader,
            MultimodalBatch as _RawMultimodalBatch,
            MultimodalParquetLoader as _RawMultimodalParquetLoader,
            StreamCoordinate,
            dataloader_cpu_isa,
            postprocess_and_dedup_records,
            serialize_table_to_bytes,
            suggest_closest_matches,
        )
    except ImportError:

        class StreamCoordinate:
            def __init__(
                self,
                shard_id: int = 0,
                byte_offset: int = 0,
                row_group_idx: int = 0,
                page_idx: int = 0,
                row_offset_in_page: int = 0,
                global_row_idx: int = 0,
            ) -> None:
                self.shard_id = shard_id
                self.byte_offset = byte_offset
                self.row_group_idx = row_group_idx
                self.page_idx = page_idx
                self.row_offset_in_page = row_offset_in_page
                self.global_row_idx = global_row_idx

        class _RawDataBatch:
            batch_id: int
            records: list[str]
            next_coord: StreamCoordinate
            is_eof: bool

            def __len__(self) -> int:
                return 0

        class _RawMultimodalBatch:
            batch_id: int
            num_rows: int
            next_coord: StreamCoordinate
            is_eof: bool

            def column_names(self) -> list[str]:
                return []

            def column(self, name: str) -> list[Any]:
                return []

            def to_dict(self) -> dict[str, list[Any]]:
                return {}

            def __len__(self) -> int:
                return 0

        class _RawDataStreamLoader:
            def __new__(cls, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError(
                    "luminamoon native Rust extension (_luminamoon_optimizer_rs) is not compiled"
                )

        class _RawMultimodalParquetLoader:
            def __new__(cls, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError(
                    "luminamoon native Rust extension (_luminamoon_optimizer_rs) is not compiled"
                )

            def export_committed_hashes(self) -> list[int]:
                return []

            def import_committed_hashes(self, hashes: Sequence[int]) -> None:
                pass

        class HfHubUploader:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError(
                    "luminamoon native Rust extension (_luminamoon_optimizer_rs) is not compiled"
                )

        def dataloader_cpu_isa() -> str:
            return "SCALAR"

        def suggest_closest_matches(
            query: str,
            candidates: Sequence[str],
            max_results: int = 3,
        ) -> list[str]:
            return _py_find_closest_matches(query, candidates, max_results)

        def serialize_table_to_bytes(
            columns: Any,
            path_in_repo: str = "part.parquet",
            hf_image_struct: bool = True,
        ) -> bytes:
            raise RuntimeError(
                "luminamoon native Rust extension (_luminamoon_optimizer_rs) is not compiled"
            )

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
        ) -> tuple[list[str], list[int]]:
            raise RuntimeError(
                "luminamoon native Rust extension (_luminamoon_optimizer_rs) is not compiled"
            )
