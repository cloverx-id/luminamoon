from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

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
        records: Sequence[str],
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
    ) -> StringTensor: ...

else:
    def _load_rs_module() -> Any:
        try:
            from luminamoon.optimizer import _luminamoon_optimizer_rs as mod
            return mod
        except ImportError:
            import importlib.machinery
            import importlib.util

            suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
            for entry in sys.path:
                if not entry:
                    continue
                opt_dir = Path(entry) / "luminamoon" / "optimizer"
                if not opt_dir.is_dir():
                    continue
                for candidate in opt_dir.iterdir():
                    if candidate.name.startswith("_luminamoon_optimizer_rs") and candidate.name.endswith(suffixes):
                        spec = importlib.util.spec_from_file_location(
                            "luminamoon.optimizer._luminamoon_optimizer_rs",
                            candidate,
                        )
                        if spec is not None and spec.loader is not None:
                            module = importlib.util.module_from_spec(spec)
                            sys.modules[spec.name] = module
                            spec.loader.exec_module(module)
                            return module
            raise

    try:
        _rs_mod = _load_rs_module()
        StreamCoordinate = _rs_mod.StreamCoordinate
        StreamDeduplicator = _rs_mod.StreamDeduplicator
        StateTracker = _rs_mod.StateTracker
        QualityFilter = _rs_mod.QualityFilter
        MinHashDeduplicator = _rs_mod.MinHashDeduplicator
        StringTensor = _rs_mod.StringTensor
        dataloader_cpu_isa = _rs_mod.dataloader_cpu_isa
        suggest_closest_matches = _rs_mod.suggest_closest_matches
        serialize_table_to_bytes = _rs_mod.serialize_table_to_bytes
        clean_single_record = _rs_mod.clean_single_record
        clean_batch_records = _rs_mod.clean_batch_records
        postprocess_and_dedup_records = _rs_mod.postprocess_and_dedup_records
        fused_pipeline_process_batch = _rs_mod.fused_pipeline_process_batch
    except Exception:
        # Fallback pure-Python stubs if C-extension not yet compiled
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

            def to_bytes(self) -> bytes:
                return (
                    self.shard_id.to_bytes(4, "little")
                    + self.byte_offset.to_bytes(8, "little")
                    + self.row_group_idx.to_bytes(4, "little")
                    + self.page_idx.to_bytes(4, "little")
                    + self.row_offset_in_page.to_bytes(4, "little")
                    + self.global_row_idx.to_bytes(8, "little")
                )

            @staticmethod
            def from_bytes(raw: bytes) -> StreamCoordinate:
                if len(raw) != 32:
                    raise ValueError(f"Invalid coordinate length: {len(raw)}")
                return StreamCoordinate(
                    shard_id=int.from_bytes(raw[0:4], "little"),
                    byte_offset=int.from_bytes(raw[4:12], "little"),
                    row_group_idx=int.from_bytes(raw[12:16], "little"),
                    page_idx=int.from_bytes(raw[16:20], "little"),
                    row_offset_in_page=int.from_bytes(raw[20:24], "little"),
                    global_row_idx=int.from_bytes(raw[24:32], "little"),
                )

        class StreamDeduplicator:
            def __init__(self, initial_hashes: Sequence[int] | None = None) -> None:
                self._committed = set(initial_hashes or ())
                self._staged = set()

            @staticmethod
            def compute_hash(payload: bytes) -> int:
                import hashlib
                return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")

            def is_seen(self, payload: bytes) -> bool:
                h = self.compute_hash(payload)
                return h in self._committed

            def try_add(self, payload: bytes) -> bool:
                h = self.compute_hash(payload)
                if h in self._committed or h in self._staged:
                    return False
                self._staged.add(h)
                return True

            def filter_batch(self, records: Sequence[str]) -> list[str]:
                out = []
                for r in records:
                    if self.try_add(r.encode("utf-8")):
                        out.append(r)
                return out

            def commit(self) -> int:
                count = len(self._staged)
                self._committed.update(self._staged)
                self._staged.clear()
                return count

            def rollback(self) -> int:
                count = len(self._staged)
                self._staged.clear()
                return count

            def count(self) -> int:
                return len(self._committed)

            def save_wal(self, path: str) -> int:
                p = Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("wb") as f:
                    for h in self._committed:
                        f.write(h.to_bytes(8, "little"))
                return len(self._committed)

            def load_wal(self, path: str) -> int:
                p = Path(path)
                if not p.is_file():
                    return 0
                count = 0
                with p.open("rb") as f:
                    while chunk := f.read(8):
                        if len(chunk) == 8:
                            self._committed.add(int.from_bytes(chunk, "little"))
                            count += 1
                return count

            def export_hashes(self) -> list[int]:
                return list(self._committed)

        class StateTracker:
            def __init__(self, meta_path: str) -> None:
                self._meta_path = Path(meta_path)
                self._meta_path.parent.mkdir(parents=True, exist_ok=True)
                self._coord = StreamCoordinate()
                self._batch_id = 0

            def read(self) -> StreamCoordinate:
                return self._coord

            def update(
                self,
                shard_id: int = 0,
                byte_offset: int = 0,
                row_group_idx: int = 0,
                page_idx: int = 0,
                row_offset_in_page: int = 0,
                global_row_idx: int = 0,
                batch_id: int = 1,
            ) -> None:
                self._coord = StreamCoordinate(
                    shard_id, byte_offset, row_group_idx, page_idx, row_offset_in_page, global_row_idx
                )
                self._batch_id = batch_id

            def commit(self, batch_id: int = 1) -> StreamCoordinate:
                return self._coord

            def committed_batch_id(self) -> int:
                return self._batch_id

        def dataloader_cpu_isa() -> str:
            return "SCALAR"

        def suggest_closest_matches(
            target: str,
            candidates: Sequence[str],
            max_suggestions: int = 3,
        ) -> list[str]:
            return list(candidates)[:max_suggestions]

        def serialize_table_to_bytes(
            columns: Any,
            format: str = "parquet",
            hf_image_struct: bool = True,
        ) -> bytes:
            return b""

        def clean_single_record(
            raw: str,
            trim_whitespace: bool = True,
            normalize_newlines: bool = True,
            min_byte_len: int = 1,
            max_byte_len: int = 0,
            truncate_over_max: bool = False,
            json_extract_key: str | None = None,
        ) -> str | None:
            text = raw
            if json_extract_key and text.startswith("{"):
                import json
                try:
                    obj = json.loads(text)
                    for key in json_extract_key.split("."):
                        obj = obj.get(key)
                    if obj is not None:
                        text = str(obj)
                except Exception:
                    return None
            if normalize_newlines:
                text = text.replace("\r\n", "\n").replace("\r", "\n")
            if trim_whitespace:
                text = text.strip()
            if len(text.encode("utf-8")) < min_byte_len:
                return None
            if max_byte_len > 0 and len(text.encode("utf-8")) > max_byte_len:
                if truncate_over_max:
                    text = text[:max_byte_len]
                else:
                    return None
            return text

        def clean_batch_records(
            records: Sequence[str],
            trim_whitespace: bool = True,
            normalize_newlines: bool = True,
            min_byte_len: int = 1,
            max_byte_len: int = 0,
            truncate_over_max: bool = False,
            json_extract_key: str | None = None,
        ) -> list[str]:
            out = []
            for r in records:
                res = clean_single_record(
                    r, trim_whitespace, normalize_newlines, min_byte_len, max_byte_len, truncate_over_max, json_extract_key
                )
                if res is not None:
                    out.append(res)
            return out

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
            cleaned = clean_batch_records(
                records, trim_whitespace, normalize_newlines, min_byte_len, max_byte_len, truncate_over_max, json_extract_key
            )
            return cleaned, []

        class QualityFilter:
            def __init__(
                self,
                min_chars: int = 20,
                max_chars: int = 0,
                min_words: int = 3,
                min_entropy: float = 2.5,
                max_repetition_ratio: float = 0.35,
                max_symbol_ratio: float = 0.30,
            ) -> None:
                self.min_chars = min_chars
                self.max_chars = max_chars
                self.min_words = min_words
                self.min_entropy = min_entropy
                self.max_repetition_ratio = max_repetition_ratio
                self.max_symbol_ratio = max_symbol_ratio

            def is_high_quality(self, text: str) -> bool:
                if len(text) < self.min_chars or (self.max_chars > 0 and len(text) > self.max_chars):
                    return False
                words = text.split()
                if len(words) < self.min_words:
                    return False
                return True

            def metrics(self, text: str) -> tuple[int, int, float, float, float, bool]:
                words = text.split()
                passed = self.is_high_quality(text)
                return (len(text), len(words), 3.0, 0.1, 0.1, passed)

            def filter_flags(self, texts: Sequence[str]) -> list[bool]:
                return [self.is_high_quality(t) for t in texts]

            def filter_batch(self, texts: Sequence[str]) -> list[str]:
                return [t for t in texts if self.is_high_quality(t)]

        class MinHashDeduplicator:
            def __init__(
                self,
                threshold: float = 0.85,
                k_shingle: int = 5,
                num_perm: int = 64,
            ) -> None:
                self.threshold = threshold
                self._seen: list[set[str]] = []

            def is_unique(self, text: str) -> bool:
                shingles = {text[i:i+5] for i in range(max(1, len(text)-4))}
                for prev in self._seen:
                    inter = len(shingles & prev)
                    union = len(shingles | prev)
                    if union > 0 and (inter / union) >= self.threshold:
                        return False
                self._seen.append(shingles)
                return True

            def similarity(self, text_a: str, text_b: str) -> float:
                sa = {text_a[i:i+5] for i in range(max(1, len(text_a)-4))}
                sb = {text_b[i:i+5] for i in range(max(1, len(text_b)-4))}
                union = len(sa | sb)
                return (len(sa & sb) / union) if union > 0 else 0.0

            def count(self) -> int:
                return len(self._seen)

            def clear(self) -> None:
                self._seen.clear()

        class StringTensor:
            def __init__(self, strings: Sequence[str] | None = None) -> None:
                self._strings = list(strings or [])

            @staticmethod
            def from_list(strings: Sequence[str]) -> StringTensor:
                return StringTensor(strings)

            @staticmethod
            def from_gpu_buffers(offsets: Sequence[int], payload: bytes) -> StringTensor:
                out = []
                for i in range(len(offsets) - 1):
                    s = payload[offsets[i]:offsets[i+1]].decode("utf-8", errors="replace")
                    out.append(s)
                return StringTensor(out)

            def to_gpu_buffers(self) -> tuple[bytes, bytes]:
                offsets = [0]
                payload_parts = []
                curr = 0
                for s in self._strings:
                    b = s.encode("utf-8")
                    payload_parts.append(b)
                    curr += len(b)
                    offsets.append(curr)
                offset_bytes = b"".join(o.to_bytes(4, "little") for o in offsets)
                return offset_bytes, b"".join(payload_parts)

            def to_offsets(self) -> list[int]:
                offsets = [0]
                curr = 0
                for s in self._strings:
                    curr += len(s.encode("utf-8"))
                    offsets.append(curr)
                return offsets

            def to_payload(self) -> bytes:
                return b"".join(s.encode("utf-8") for s in self._strings)

            def __len__(self) -> int:
                return len(self._strings)

            def __getitem__(self, index: int) -> str:
                return self._strings[index]

            def get(self, index: int) -> str | None:
                if 0 <= index < len(self._strings):
                    return self._strings[index]
                return None

            def to_list(self) -> list[str]:
                return list(self._strings)

            def lengths(self) -> list[int]:
                return [len(s.encode("utf-8")) for s in self._strings]

            def is_inlined_mask(self) -> list[bool]:
                return [len(s.encode("utf-8")) <= 12 for s in self._strings]

            def total_bytes(self) -> int:
                return sum(len(s.encode("utf-8")) for s in self._strings)

            def filter_by_length(self, min_len: int = 0, max_len: int = 0) -> StringTensor:
                out = [
                    s for s in self._strings
                    if len(s.encode("utf-8")) >= min_len and (max_len == 0 or len(s.encode("utf-8")) <= max_len)
                ]
                return StringTensor(out)

            def filter_by_prefix(self, prefix: str) -> StringTensor:
                return StringTensor([s for s in self._strings if s.startswith(prefix)])

            def hash_all(self) -> list[int]:
                import hashlib
                return [
                    int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")
                    for s in self._strings
                ]

        def fused_pipeline_process_batch(
            records: Sequence[str],
            trim_whitespace: bool = True,
            normalize_newlines: bool = True,
            min_byte_len: int = 1,
            max_byte_len: int = 0,
            truncate_over_max: bool = False,
            json_extract_key: str | None = None,
            filter_quality: bool = False,
            min_entropy: float = 1.5,
            max_repetition_ratio: float = 0.4,
            max_symbol_ratio: float = 0.4,
            min_words: int = 1,
            deduplicator: Any = None,
            commit_dedup: bool = False,
        ) -> StringTensor:
            survivors = []
            for r in records:
                c = clean_single_record(
                    r,
                    trim_whitespace=trim_whitespace,
                    normalize_newlines=normalize_newlines,
                    min_byte_len=min_byte_len,
                    max_byte_len=max_byte_len,
                    truncate_over_max=truncate_over_max,
                    json_extract_key=json_extract_key,
                )
                if c is None:
                    continue
                if deduplicator is not None and hasattr(deduplicator, "try_add"):
                    if not deduplicator.try_add(c.encode("utf-8")):
                        continue
                survivors.append(c)
            if commit_dedup and deduplicator is not None and hasattr(deduplicator, "commit"):
                deduplicator.commit()
            return StringTensor(survivors)
