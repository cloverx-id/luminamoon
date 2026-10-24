from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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
        import hashlib
        import json
        import math
        import re
        import struct
        import time

        class TextCleaner:
            def __init__(
                self,
                trim_whitespace: bool = True,
                normalize_newlines: bool = True,
                min_byte_len: int = 0,
                max_byte_len: int = 0,
                truncate_over_max: bool = False,
                json_extract_key: Optional[str] = None,
                collapse_whitespace: bool = True,
                strip_edges: bool = True,
            ) -> None:
                self.trim_whitespace = trim_whitespace
                self.normalize_newlines = normalize_newlines
                self.min_byte_len = min_byte_len
                self.max_byte_len = max_byte_len
                self.truncate_over_max = truncate_over_max
                self.json_extract_key = json_extract_key
                self.collapse_whitespace = collapse_whitespace
                self.strip_edges = strip_edges

            def clean(self, text: str) -> str:
                out = text.replace("\r\n", "\n").replace("\r", "\n")
                out = "".join(ch if (ch == "\n" or ord(ch) >= 0x20) else " " for ch in out)
                if self.collapse_whitespace:
                    out = re.sub(r"[ \t]+", " ", out)
                if self.strip_edges:
                    out = out.strip()
                return out

            def clean_batch(self, texts: Sequence[str]) -> List[str]:
                return [self.clean(t) for t in texts]

            def extract_json(self, json_str: str, key: str = "text") -> Optional[str]:
                try:
                    obj = json.loads(json_str)
                except Exception:
                    return None
                curr: Any = obj
                for part in key.split("."):
                    if isinstance(curr, dict) and part in curr:
                        curr = curr[part]
                    else:
                        return None
                return curr if isinstance(curr, str) else None

            def extract_json_batch(self, json_lines: Sequence[str], key: str = "text") -> List[str]:
                out: List[str] = []
                for line in json_lines:
                    extracted = self.extract_json(line, key)
                    if extracted is not None:
                        cleaned = self.clean(extracted)
                        if cleaned:
                            out.append(cleaned)
                return out

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

        class StreamDeduplicator:
            def __init__(
                self,
                capacity: int = 1_000_000,
                persistence_path: Optional[str] = None,
                initial_hashes: Optional[Sequence[int]] = None,
                wal_path: Optional[str] = None,
            ) -> None:
                self._seen: set[int] = set(initial_hashes or ())
                self._staged: Dict[int, List[int]] = {}
                self._staged_lookup: set[int] = set()
                self._persistence_path = persistence_path or wal_path

            @staticmethod
            def compute_hash(text_or_payload: Any) -> int:
                raw = text_or_payload.encode("utf-8") if isinstance(text_or_payload, str) else bytes(text_or_payload)
                digest = hashlib.blake2b(raw, digest_size=8).digest()
                return int.from_bytes(digest, "little", signed=False)

            def is_seen(self, payload: Any) -> bool:
                h = self.compute_hash(payload)
                return (h in self._seen) or (h in self._staged_lookup)

            def is_duplicate(self, text: str) -> bool:
                h = self.compute_hash(text)
                return (h in self._seen) or (h in self._staged_lookup)

            def check_and_insert(self, text: str) -> bool:
                h = self.compute_hash(text)
                if h in self._seen or h in self._staged_lookup:
                    return False
                self._seen.add(h)
                return True

            def is_unique(self, text: str) -> bool:
                h = self.compute_hash(text)
                if h in self._seen or h in self._staged_lookup:
                    return False
                self._staged_lookup.add(h)
                self._staged.setdefault(0, []).append(h)
                return True

            def try_add(self, payload: bytes) -> bool:
                h = self.compute_hash(payload)
                if h in self._seen or h in self._staged_lookup:
                    return False
                self._staged_lookup.add(h)
                self._staged.setdefault(0, []).append(h)
                return True

            def stage_batch(self, batch_id: int, texts: Sequence[str]) -> List[bool]:
                mask: List[bool] = []
                batch_hashes: List[int] = []
                for t in texts:
                    h = self.compute_hash(t)
                    if h in self._seen or h in self._staged_lookup:
                        mask.append(False)
                    else:
                        self._staged_lookup.add(h)
                        batch_hashes.append(h)
                        mask.append(True)
                self._staged.setdefault(batch_id, []).extend(batch_hashes)
                return mask

            def commit_batch(self, batch_id: int) -> int:
                hashes = self._staged.pop(batch_id, [])
                for h in hashes:
                    self._staged_lookup.discard(h)
                    self._seen.add(h)
                return len(hashes)

            def commit(self) -> int:
                count = len(self._staged_lookup)
                self._seen.update(self._staged_lookup)
                self._staged.clear()
                self._staged_lookup.clear()
                return count

            def rollback(self) -> int:
                count = len(self._staged_lookup)
                self._staged.clear()
                self._staged_lookup.clear()
                return count

            def filter_batch(self, texts: Sequence[str]) -> List[str]:
                return [t for t in texts if self.check_and_insert(t)]

            def filter_indices(self, texts: Sequence[str]) -> List[int]:
                return [idx for idx, t in enumerate(texts) if self.check_and_insert(t)]

            def committed_count(self) -> int:
                return len(self._seen)

            def staged_count(self) -> int:
                return len(self._staged_lookup)

            @property
            def count(self) -> int:
                return len(self._seen) + len(self._staged_lookup)

            def reset_staged(self) -> None:
                self._staged.clear()
                self._staged_lookup.clear()

            def clear(self) -> None:
                self._seen.clear()
                self._staged.clear()
                self._staged_lookup.clear()

            def save_wal(self, path: str) -> int:
                p = Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"".join(h.to_bytes(8, "little") for h in self._seen))
                return len(self._seen)

            def load_wal(self, path: str) -> int:
                p = Path(path)
                if not p.exists():
                    return 0
                data = p.read_bytes()
                n = len(data) // 8
                for i in range(n):
                    h = int.from_bytes(data[i * 8 : (i + 1) * 8], "little")
                    self._seen.add(h)
                return n

            def export_hashes(self) -> List[int]:
                return list(self._seen)

            def __len__(self) -> int:
                return len(self._seen) + len(self._staged_lookup)

        class MinHashDeduplicator:
            def __init__(
                self,
                threshold: float = 0.80,
                shingle_size: int = 5,
                num_perm: int = 64,
                num_bands: int = 16,
                k_shingle: Optional[int] = None,
            ) -> None:
                self.threshold = threshold
                self.shingle_size = k_shingle if k_shingle is not None else max(1, shingle_size)
                self.num_perm = max(1, num_perm)
                self.num_bands = max(1, min(num_bands, self.num_perm))
                self.rows_per_band = max(1, self.num_perm // self.num_bands)
                self._buckets: List[set[int]] = [set() for _ in range(self.num_bands)]
                self._count = 0

            def _signature(self, text: str) -> List[int]:
                if not text:
                    return [0] * self.num_perm
                k = min(self.shingle_size, len(text))
                shingles = [text[i : i + k].encode("utf-8") for i in range(len(text) - k + 1)]
                sig: List[int] = []
                for p in range(self.num_perm):
                    prefix = p.to_bytes(4, "little")
                    m = min(
                        int.from_bytes(hashlib.blake2b(prefix + s, digest_size=8).digest(), "little")
                        for s in shingles
                    )
                    sig.append(m)
                return sig

            def jaccard_estimate(self, text_a: str, text_b: str) -> float:
                sig_a = self._signature(text_a)
                sig_b = self._signature(text_b)
                matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
                return matches / float(self.num_perm)

            def similarity(self, text_a: str, text_b: str) -> float:
                return self.jaccard_estimate(text_a, text_b)

            def jaccard_similarity(self, other: Any) -> float:
                return 0.0

            def check_and_insert(self, text: str) -> bool:
                sig = self._signature(text)
                band_hashes: List[int] = []
                for b in range(self.num_bands):
                    start = b * self.rows_per_band
                    end = start + self.rows_per_band
                    band_tuple = tuple(sig[start:end])
                    bh = hash(band_tuple)
                    if bh in self._buckets[b]:
                        return False
                    band_hashes.append(bh)
                for b, bh in enumerate(band_hashes):
                    self._buckets[b].add(bh)
                self._count += 1
                return True

            def is_near_duplicate(self, text: str) -> bool:
                return not self.check_and_insert(text)

            def is_unique(self, text: str) -> bool:
                return self.check_and_insert(text)

            def filter_batch(self, texts: Sequence[str]) -> List[str]:
                return [t for t in texts if self.check_and_insert(t)]

            def clear(self) -> None:
                for b in self._buckets:
                    b.clear()
                self._count = 0

            @property
            def count(self) -> int:
                return self._count

            def __len__(self) -> int:
                return self._count

        class PIIMasker:
            _EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
            _IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
            _PHONE_RE = re.compile(r"(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}")

            def __init__(
                self,
                mask_email: bool = True,
                mask_ip: bool = True,
                mask_phone: bool = True,
                custom_patterns: Optional[Sequence[Tuple[str, str]]] = None,
                mask_token: Optional[str] = None,
            ) -> None:
                self.mask_email = mask_email
                self.mask_ip = mask_ip
                self.mask_phone = mask_phone
                self.mask_token = mask_token
                self.custom_compiled: List[Tuple[re.Pattern[str], str]] = []
                if custom_patterns:
                    for pat, repl in custom_patterns:
                        self.custom_compiled.append((re.compile(pat), repl))

            def sanitize(self, text: str) -> str:
                out = text
                if self.mask_email:
                    repl = self.mask_token if self.mask_token is not None else "[EMAIL]"
                    out = self._EMAIL_RE.sub(repl, out)
                if self.mask_ip:
                    repl = self.mask_token if self.mask_token is not None else "[IP_ADDRESS]"
                    out = self._IPV4_RE.sub(repl, out)
                if self.mask_phone:
                    repl = self.mask_token if self.mask_token is not None else "[PHONE]"
                    out = self._PHONE_RE.sub(repl, out)
                for reg, repl in self.custom_compiled:
                    out = reg.sub(repl, out)
                return out

            def sanitize_batch(self, texts: Sequence[str]) -> List[str]:
                return [self.sanitize(t) for t in texts]

            def mask(self, text: str) -> str:
                return self.sanitize(text)

            def mask_batch(self, texts: Sequence[str]) -> List[str]:
                return self.sanitize_batch(texts)

        class StreamCoordinate:
            def __init__(
                self,
                shard_id: int = 0,
                byte_offset: int = 0,
                row_group_idx: int = 0,
                page_idx: int = 0,
                row_offset_in_page: int = 0,
                global_row_idx: int = 0,
                epoch: int = 0,
                record_idx: int = 0,
                rng_state: int = 0,
            ) -> None:
                self.shard_id = shard_id
                self.byte_offset = byte_offset
                self.row_group_idx = row_group_idx
                self.page_idx = page_idx
                self.row_offset_in_page = row_offset_in_page
                self.global_row_idx = global_row_idx if global_row_idx != 0 else record_idx
                self.epoch = epoch
                self.record_idx = self.global_row_idx
                self.rng_state = rng_state
                self.timestamp_ns = time.time_ns()

            def to_dict(self) -> Dict[str, int]:
                return {
                    "shard_id": self.shard_id,
                    "byte_offset": self.byte_offset,
                    "row_group_idx": self.row_group_idx,
                    "page_idx": self.page_idx,
                    "row_offset_in_page": self.row_offset_in_page,
                    "global_row_idx": self.global_row_idx,
                }

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

        class StateTracker:
            def __init__(self, meta_path: str, dedup_path: Optional[str] = None) -> None:
                self._meta_path = Path(meta_path)
                self._dedup_path = Path(dedup_path) if dedup_path else Path(f"{meta_path}.dedup")
                self._staged: Dict[int, StreamCoordinate] = {}
                self._last_coord: Optional[StreamCoordinate] = None
                self._last_batch_id: int = 0
                self._epoch: int = 0
                if self._meta_path.exists():
                    self._load_from_disk()

            def _load_from_disk(self) -> None:
                try:
                    data = json.loads(self._meta_path.read_text(encoding="utf-8"))
                    self._epoch = int(data.get("epoch", 0))
                    self._last_batch_id = int(data.get("batch_id", 0))
                    c = data.get("coord")
                    if isinstance(c, dict):
                        self._last_coord = StreamCoordinate(**c)
                except Exception:
                    pass

            def _save_to_disk(self) -> None:
                self._meta_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "epoch": self._epoch,
                    "batch_id": self._last_batch_id,
                    "coord": self._last_coord.to_dict() if self._last_coord else None,
                }
                self._meta_path.write_text(json.dumps(payload), encoding="utf-8")

            def stage_batch(self, batch_id: int, coord: StreamCoordinate) -> None:
                self._staged[batch_id] = coord

            def commit_batch(self, batch_id: int) -> None:
                coord = self._staged.pop(batch_id, None)
                if coord is not None:
                    self._last_coord = coord
                self._last_batch_id = batch_id
                self._epoch += 1
                self._save_to_disk()

            def commit(self, batch_id: int = 1) -> StreamCoordinate:
                self.commit_batch(batch_id)
                return self.read()

            def committed_batch_id(self) -> int:
                return self._last_batch_id

            def last_committed_coord(self) -> Optional[StreamCoordinate]:
                return self._last_coord

            def last_committed_batch_id(self) -> int:
                return self._last_batch_id

            def epoch(self) -> int:
                return self._epoch

            def save(self, coord: StreamCoordinate) -> int:
                bid = self._last_batch_id + 1
                self.stage_batch(bid, coord)
                self.commit_batch(bid)
                return self._epoch

            def load(self) -> Optional[StreamCoordinate]:
                return self._last_coord

            def read(self) -> StreamCoordinate:
                return self._last_coord or StreamCoordinate()

            def update(
                self,
                shard_id: int = 0,
                byte_offset: int = 0,
                row_group_idx: int = 0,
                page_idx: int = 0,
                row_offset_in_page: int = 0,
                global_row_idx: int = 0,
                batch_id: int = 0,
            ) -> None:
                coord = StreamCoordinate(
                    shard_id=shard_id,
                    byte_offset=byte_offset,
                    row_group_idx=row_group_idx,
                    page_idx=page_idx,
                    row_offset_in_page=row_offset_in_page,
                    global_row_idx=global_row_idx,
                )
                self._last_coord = coord
                self._last_batch_id = batch_id
                self._save_to_disk()

        class RustStringTensor:
            def __init__(self, strings: Sequence[str]) -> None:
                self._strings = list(strings)
                encoded = [s.encode("utf-8") for s in self._strings]
                offsets = [0]
                for b in encoded:
                    offsets.append(offsets[-1] + len(b))
                self._offsets = offsets
                self._payload = b"".join(encoded)

            @staticmethod
            def from_list(strings: Sequence[str]) -> "RustStringTensor":
                return RustStringTensor(strings)

            @staticmethod
            def from_gpu_buffers(offsets: Sequence[int], payload: bytes) -> "RustStringTensor":
                if not offsets:
                    return RustStringTensor([])
                strings: List[str] = []
                for i in range(len(offsets) - 1):
                    start = int(offsets[i])
                    end = int(offsets[i + 1])
                    strings.append(payload[start:end].decode("utf-8", errors="replace"))
                return RustStringTensor(strings)

            def __len__(self) -> int:
                return len(self._strings)

            def __getitem__(self, index: int) -> str:
                return self._strings[index]

            def total_bytes(self) -> int:
                return len(self._payload)

            def to_list(self) -> List[str]:
                return list(self._strings)

            def lengths(self) -> List[int]:
                return [self._offsets[i + 1] - self._offsets[i] for i in range(len(self._strings))]

            def is_inlined_mask(self) -> List[bool]:
                return [(self._offsets[i + 1] - self._offsets[i]) <= 12 for i in range(len(self._strings))]

            def get(self, idx: int) -> Optional[str]:
                if 0 <= idx < len(self._strings):
                    return self._strings[idx]
                return None

            def slice(self, start: int, end: int) -> "RustStringTensor":
                return RustStringTensor(self._strings[start:end])

            def filter_by_length(self, min_bytes: int = 0, max_bytes: int = 0) -> "RustStringTensor":
                return RustStringTensor(
                    [
                        s
                        for s in self._strings
                        if len(s.encode("utf-8")) >= min_bytes
                        and (max_bytes == 0 or len(s.encode("utf-8")) <= max_bytes)
                    ]
                )

            def filter_by_prefix(self, prefix: str) -> "RustStringTensor":
                return RustStringTensor([s for s in self._strings if s.startswith(prefix)])

            def hash_all(self) -> List[int]:
                return [StreamDeduplicator.compute_hash(s) for s in self._strings]

            def to_offsets(self) -> List[int]:
                return list(self._offsets)

            def to_payload(self) -> bytes:
                return self._payload

            def get_gpu_buffers(self) -> Tuple[bytes, bytes]:
                off_bytes = struct.pack(f"<{len(self._offsets)}I", *self._offsets)
                return off_bytes, self._payload

            def to_gpu_buffers(self) -> Tuple[bytes, bytes]:
                return self.get_gpu_buffers()

        StringTensor = RustStringTensor

        def fused_pipeline_process_batch(
            raw_lines: Sequence[str],
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

        def clean_single_record(
            text: str,
            trim_whitespace: bool = True,
            normalize_newlines: bool = True,
            min_byte_len: int = 0,
            max_byte_len: int = 0,
            truncate_over_max: bool = False,
            json_extract_key: Optional[str] = None,
        ) -> Optional[str]:
            tc = TextCleaner(
                trim_whitespace=trim_whitespace,
                normalize_newlines=normalize_newlines,
                min_byte_len=min_byte_len,
                max_byte_len=max_byte_len,
                truncate_over_max=truncate_over_max,
                json_extract_key=json_extract_key,
            )
            t: Optional[str] = text
            if json_extract_key:
                t = tc.extract_json(text, json_extract_key)
                if t is None:
                    return None
            res = tc.clean(t)
            raw_len = len(res.encode("utf-8"))
            if min_byte_len > 0 and raw_len < min_byte_len:
                return None
            if max_byte_len > 0 and raw_len > max_byte_len:
                if truncate_over_max:
                    res = res.encode("utf-8")[:max_byte_len].decode("utf-8", errors="ignore")
                else:
                    return None
            return res

        def clean_batch_records(
            texts: Sequence[str],
            trim_whitespace: bool = True,
            normalize_newlines: bool = True,
            min_byte_len: int = 0,
            max_byte_len: int = 0,
            truncate_over_max: bool = False,
            json_extract_key: Optional[str] = None,
        ) -> List[str]:
            out = []
            for t in texts:
                c = clean_single_record(
                    t,
                    trim_whitespace=trim_whitespace,
                    normalize_newlines=normalize_newlines,
                    min_byte_len=min_byte_len,
                    max_byte_len=max_byte_len,
                    truncate_over_max=truncate_over_max,
                    json_extract_key=json_extract_key,
                )
                if c is not None:
                    out.append(c)
            return out

        def dataloader_cpu_isa() -> str:
            return "SCALAR"

        def suggest_closest_matches(query: str, candidates: Sequence[str], top_k: int = 3) -> List[str]:
            return list(candidates[:top_k])

        def postprocess_and_dedup_records(
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
            deduplicator: Any = None,
            commit_dedup: bool = False,
        ) -> List[str]:
            st = fused_pipeline_process_batch(
                raw_lines=records,
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
            )
            return st.to_list()

        def serialize_table_to_bytes(records: Sequence[str], format_name: str) -> bytes:
            return "\n".join(records).encode("utf-8")

        def serialize_to_jsonl(strings: Sequence[str], path: str, column: str = "text") -> int:
            with open(path, "w", encoding="utf-8") as f:
                for s in strings:
                    f.write(json.dumps({column: s}, ensure_ascii=False) + "\n")
            return len(strings)

        def serialize_to_parquet(
            strings: Sequence[str],
            path: str,
            column: str = "text",
            image_bytes: Optional[Sequence[Optional[bytes]]] = None,
            image_paths: Optional[Sequence[Optional[str]]] = None,
        ) -> int:
            import pyarrow as pa
            import pyarrow.parquet as pq

            arrays = [pa.array(strings, type=pa.string())]
            names = [column]
            if image_bytes is not None or image_paths is not None:
                n = len(strings)
                ib = list(image_bytes) if image_bytes is not None else [None] * n
                ip = list(image_paths) if image_paths is not None else [None] * n
                img_struct = pa.StructArray.from_arrays(
                    [pa.array(ib, type=pa.binary()), pa.array(ip, type=pa.string())],
                    names=["bytes", "path"],
                )
                arrays.append(img_struct)
                names.append("image")
            table = pa.Table.from_arrays(arrays, names=names)
            pq.write_table(table, path, compression="zstd")
            return len(strings)

        def serialize_to_arrow_ipc(
            strings: Sequence[str],
            path: str,
            column: str = "text",
            image_bytes: Optional[Sequence[Optional[bytes]]] = None,
            image_paths: Optional[Sequence[Optional[str]]] = None,
        ) -> int:
            import pyarrow as pa
            import pyarrow.feather as feather

            arrays = [pa.array(strings, type=pa.string())]
            names = [column]
            if image_bytes is not None or image_paths is not None:
                n = len(strings)
                ib = list(image_bytes) if image_bytes is not None else [None] * n
                ip = list(image_paths) if image_paths is not None else [None] * n
                img_struct = pa.StructArray.from_arrays(
                    [pa.array(ib, type=pa.binary()), pa.array(ip, type=pa.string())],
                    names=["bytes", "path"],
                )
                arrays.append(img_struct)
                names.append("image")
            table = pa.Table.from_arrays(arrays, names=names)
            feather.write_feather(table, path)
            return len(strings)

        def serialize_to_csv(strings: Sequence[str], path: str, column: str = "text") -> int:
            import csv

            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([column])
                for s in strings:
                    writer.writerow([s])
            return len(strings)