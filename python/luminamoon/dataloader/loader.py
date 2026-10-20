from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from ._ffi import (
    StreamCoordinate,
    _RawDataStreamLoader,
    _RawMultimodalParquetLoader,
    postprocess_and_dedup_records,
)
from .batch import (
    DataBatch,
    MultimodalBatch,
    _apply_batch_hooks,
    _CoalescedMultimodalRawBatch,
    _InMemoryRawBatch,
)
from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_DIR,
    DEFAULT_DOWNLOAD_WORKERS,
    DEFAULT_PRERAM_BATCHES,
    DEFAULT_REFILL_TRIGGER,
    DEFAULT_REMOTE_CHUNK_BYTES,
    DEFAULT_SHUFFLE_BUFFER_SIZE,
    DEFAULT_SHUFFLE_SEED,
    VALID_DATA_STREAM_LOADER_KWARGS,
    VALID_FROM_HF_DATASET_KWARGS,
    VALID_INTERLEAVE_KWARGS,
    VALID_INTERLEAVE_STRATEGIES,
    VALID_MULTIMODAL_LOADER_KWARGS,
    VALID_STREAM_KWARGS,
    WEBDATASET_TAR_EXTENSIONS,
)
from .diagnostics import (
    _diagnose_uri_typo_py,
    _format_did_you_mean,
    _validate_known_kwargs,
)
from .hub import (
    _normalize_sources_list,
    expand_hf_sources,
    resolve_hf_secrets,
)


def _default_checkpoint_path(
    normalized_sources: Sequence[str],
    parquet_column: str = "text",
    min_score: float | None = None,
    score_column: str | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> Path:
    key = "\n".join(normalized_sources)
    if parquet_column != "text":
        key = f"{key}\n#col={parquet_column}"
    if min_score is not None:
        key = f"{key}\n#min_score={score_column or 'auto'}:{min_score}"
    if world_size > 1:
        key = f"{key}\n#ddp={rank}/{world_size}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"stream_{digest}.meta"


def _partition_sources_for_ddp(
    sources: Sequence[str],
    rank: int,
    world_size: int,
) -> tuple[list[str], bool]:
    if world_size <= 1:
        return list(sources), False
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"Invalid DDP rank={rank} for world_size={world_size}: expected 0 <= rank < {world_size}"
        )
    if len(sources) >= world_size:
        return list(sources[rank::world_size]), False
    return list(sources), True


class DataStreamLoader(_RawDataStreamLoader):
    def __new__(
        cls,
        sources: Any,
        checkpoint_path: str | Path | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        preram_batches: int = DEFAULT_PRERAM_BATCHES,
        refill_trigger: int = DEFAULT_REFILL_TRIGGER,
        deduplicate: bool = True,
        auto_refill: bool = True,
        parquet_column: str = "text",
        json_extract_key: str | None = None,
        trim_whitespace: bool = True,
        normalize_newlines: bool = True,
        min_byte_len: int = 1,
        max_byte_len: int = 0,
        truncate_over_max: bool = False,
        auth_token: str | None = None,
        remote_chunk_bytes: int = DEFAULT_REMOTE_CHUNK_BYTES,
        download_workers: int = DEFAULT_DOWNLOAD_WORKERS,
        auto_commit: bool = False,
        tokenizer: Any = None,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        postprocess_fn: Callable[[Any], Any] | None = None,
        split: str | None = None,
        subset: str | None = None,
        min_score: float | None = None,
        score_column: str | None = None,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = False,
        seed: int = DEFAULT_SHUFFLE_SEED,
        shuffle_buffer_size: int = DEFAULT_SHUFFLE_BUFFER_SIZE,
        **kwargs: Any,
    ):
        _validate_known_kwargs("DataStreamLoader", kwargs, VALID_DATA_STREAM_LOADER_KWARGS)
        secrets = resolve_hf_secrets(token=auth_token)
        resolved_token = secrets["token"]
        expanded_sources = expand_hf_sources(
            sources,
            split=split,
            subset=subset,
            token=resolved_token,
            endpoint=secrets["endpoint"],
        )
        if shuffle and len(expanded_sources) > 1:
            rng = random.Random(seed)
            expanded_sources = list(expanded_sources)
            rng.shuffle(expanded_sources)

        rank_sources, stride_batches = _partition_sources_for_ddp(
            expanded_sources, rank=rank, world_size=world_size
        )

        resolved_ckpt = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else _default_checkpoint_path(
                rank_sources,
                parquet_column=parquet_column,
                min_score=min_score,
                score_column=score_column,
                rank=rank,
                world_size=world_size,
            )
        )
        if resolved_ckpt.parent:
            resolved_ckpt.parent.mkdir(parents=True, exist_ok=True)

        instance = super().__new__(
            cls,
            rank_sources,
            str(resolved_ckpt),
            batch_size=batch_size,
            preram_batches=preram_batches,
            refill_trigger=refill_trigger,
            deduplicate=deduplicate,
            auto_refill=auto_refill,
            parquet_column=parquet_column,
            json_extract_key=json_extract_key,
            trim_whitespace=trim_whitespace,
            normalize_newlines=normalize_newlines,
            min_byte_len=min_byte_len,
            max_byte_len=max_byte_len,
            truncate_over_max=truncate_over_max,
            auth_token=resolved_token,
            remote_chunk_bytes=remote_chunk_bytes,
            download_workers=download_workers,
            min_score=min_score,
            score_column=score_column,
        )
        instance._resolved_checkpoint_path = resolved_ckpt
        instance._stride_batches_for_ddp = stride_batches
        return instance

    def __init__(
        self,
        sources: Any,
        checkpoint_path: str | Path | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        preram_batches: int = DEFAULT_PRERAM_BATCHES,
        refill_trigger: int = DEFAULT_REFILL_TRIGGER,
        deduplicate: bool = True,
        auto_refill: bool = True,
        parquet_column: str = "text",
        json_extract_key: str | None = None,
        trim_whitespace: bool = True,
        normalize_newlines: bool = True,
        min_byte_len: int = 1,
        max_byte_len: int = 0,
        truncate_over_max: bool = False,
        auth_token: str | None = None,
        remote_chunk_bytes: int = DEFAULT_REMOTE_CHUNK_BYTES,
        download_workers: int = DEFAULT_DOWNLOAD_WORKERS,
        auto_commit: bool = False,
        tokenizer: Any = None,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        postprocess_fn: Callable[[Any], Any] | None = None,
        split: str | None = None,
        subset: str | None = None,
        min_score: float | None = None,
        score_column: str | None = None,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = False,
        seed: int = DEFAULT_SHUFFLE_SEED,
        shuffle_buffer_size: int = DEFAULT_SHUFFLE_BUFFER_SIZE,
        **kwargs: Any,
    ) -> None:
        self._checkpoint_path: Path = getattr(
            self,
            "_resolved_checkpoint_path",
            Path(checkpoint_path)
            if checkpoint_path is not None
            else _default_checkpoint_path(
                _normalize_sources_list(sources),
                parquet_column=parquet_column,
                min_score=min_score,
                score_column=score_column,
                rank=rank,
                world_size=world_size,
            ),
        )
        self._auto_commit = bool(auto_commit)
        self._last_uncommitted_batch_id: int | None = None
        self._tokenizer = tokenizer
        self._tokenizer_kwargs = tokenizer_kwargs
        self._postprocess_fn = postprocess_fn
        self._parquet_column = parquet_column
        self._rank = rank
        self._world_size = world_size
        self._stride_batches: bool = getattr(self, "_stride_batches_for_ddp", False)
        self._shuffle = bool(shuffle)
        self._rng = random.Random(seed + rank)
        self._shuffle_buffer_size = max(1, int(shuffle_buffer_size))

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def auto_commit(self) -> bool:
        return self._auto_commit

    def _flush_pending_auto_commit(self) -> None:
        if not self._auto_commit or self._last_uncommitted_batch_id is None:
            return
        pending_id = self._last_uncommitted_batch_id
        self._last_uncommitted_batch_id = None
        try:
            if super().committed_batch_id() < pending_id:
                super().commit_gpu_batch(pending_id)
        except BaseException:
            pass

    def commit_gpu_batch(self, batch_id: int) -> StreamCoordinate:
        if (
            self._last_uncommitted_batch_id is not None
            and batch_id >= self._last_uncommitted_batch_id
        ):
            self._last_uncommitted_batch_id = None
        return super().commit_gpu_batch(batch_id)

    def mark_in_gpu(self, batch_id: int) -> StreamCoordinate:
        return self.commit_gpu_batch(batch_id)

    def next_batch(self) -> DataBatch | None:
        while True:
            self._flush_pending_auto_commit()
            raw = super().next_batch()
            if raw is None:
                return None
            if self._auto_commit:
                self._last_uncommitted_batch_id = raw.batch_id
            if self._stride_batches and ((raw.batch_id - 1) % self._world_size != self._rank):
                continue
            if self._shuffle and len(raw.records) > 1:
                shuffled = list(raw.records)
                self._rng.shuffle(shuffled)
                raw = _InMemoryRawBatch(
                    batch_id=raw.batch_id,
                    records=shuffled,
                    next_coord=raw.next_coord,
                    is_eof=raw.is_eof,
                )
            batch: Any = DataBatch(raw, loader=self, column_name=self._parquet_column)
            return _apply_batch_hooks(
                batch,
                tokenizer=self._tokenizer,
                tokenizer_kwargs=self._tokenizer_kwargs,
                postprocess_fn=self._postprocess_fn,
                text_column=self._parquet_column,
            )  # type: ignore[return-value]

    def __iter__(self) -> DataStreamLoader:
        return self

    def __next__(self) -> DataBatch:
        batch = self.next_batch()
        if batch is None:
            raise StopIteration
        return batch

    def close(self) -> None:
        self._flush_pending_auto_commit()
        super().close()

    def __enter__(self) -> DataStreamLoader:
        return self

    def __exit__(
        self,
        exc_type: object = None,
        exc_value: object = None,
        traceback: object = None,
    ) -> bool:
        if exc_type is None:
            self._flush_pending_auto_commit()
        else:
            self._last_uncommitted_batch_id = None
        super().close()
        return False


class MultimodalParquetLoader(_RawMultimodalParquetLoader):
    def __new__(
        cls,
        path: str | Path,
        columns: Sequence[str] | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        deduplicate: bool = True,
        shard_id: int = 0,
        auth_token: str | None = None,
        download_workers: int = DEFAULT_DOWNLOAD_WORKERS,
        tokenizer: Any = None,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        postprocess_fn: Callable[[Any], Any] | None = None,
        min_score: float | None = None,
        score_column: str | None = None,
        shuffle: bool = False,
        seed: int = DEFAULT_SHUFFLE_SEED,
        **kwargs: Any,
    ):
        _validate_known_kwargs(
            "MultimodalParquetLoader", kwargs, VALID_MULTIMODAL_LOADER_KWARGS
        )
        p_str = str(path).strip()
        typo_msg = _diagnose_uri_typo_py(p_str)
        if typo_msg is not None:
            raise ValueError(typo_msg)
        norm_cols = [str(c) for c in columns] if columns is not None else None
        resolved_token = resolve_hf_secrets(token=auth_token)["token"]
        return super().__new__(
            cls,
            p_str,
            columns=norm_cols,
            batch_size=batch_size,
            deduplicate=deduplicate,
            shard_id=shard_id,
            auth_token=resolved_token,
            download_workers=download_workers,
            min_score=min_score,
            score_column=score_column,
        )

    def __init__(
        self,
        path: str | Path,
        columns: Sequence[str] | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        deduplicate: bool = True,
        shard_id: int = 0,
        auth_token: str | None = None,
        download_workers: int = DEFAULT_DOWNLOAD_WORKERS,
        tokenizer: Any = None,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        postprocess_fn: Callable[[Any], Any] | None = None,
        min_score: float | None = None,
        score_column: str | None = None,
        shuffle: bool = False,
        seed: int = DEFAULT_SHUFFLE_SEED,
        **kwargs: Any,
    ) -> None:
        self._tokenizer = tokenizer
        self._tokenizer_kwargs = tokenizer_kwargs
        self._postprocess_fn = postprocess_fn
        self._shuffle = bool(shuffle)
        self._rng = random.Random(seed)

    def export_committed_hashes(self) -> list[int]:
        return list(super().export_committed_hashes())

    def import_committed_hashes(self, hashes: Sequence[int]) -> None:
        super().import_committed_hashes([int(h) for h in hashes])

    def next_batch(self) -> MultimodalBatch | None:
        raw = super().next_batch()
        if raw is None:
            return None
        if self._shuffle and raw.num_rows > 1:
            col_names = raw.column_names()
            raw_dict = raw.to_dict()
            indices = list(range(raw.num_rows))
            self._rng.shuffle(indices)
            shuffled_cols = {
                c: [raw_dict[c][i] for i in indices] for c in col_names
            }
            raw = _CoalescedMultimodalRawBatch(
                batch_id=raw.batch_id,
                column_order=col_names,
                columns=shuffled_cols,
                num_rows=raw.num_rows,
                next_coord=raw.next_coord,
                is_eof=raw.is_eof,
            )
        batch: Any = MultimodalBatch(raw)
        return _apply_batch_hooks(
            batch,
            tokenizer=self._tokenizer,
            tokenizer_kwargs=self._tokenizer_kwargs,
            postprocess_fn=self._postprocess_fn,
        )  # type: ignore[return-value]

    def __iter__(self) -> MultimodalParquetLoader:
        return self

    def __next__(self) -> MultimodalBatch:
        batch = self.next_batch()
        if batch is None:
            raise StopIteration
        return batch

    def __enter__(self) -> MultimodalParquetLoader:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class _MultiShardMultimodalStream:
    def __init__(
        self,
        sources: Sequence[str],
        columns: Sequence[str] | None,
        batch_size: int,
        deduplicate: bool,
        auth_token: str | None,
        download_workers: int = DEFAULT_DOWNLOAD_WORKERS,
        tokenizer: Any = None,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        postprocess_fn: Callable[[Any], Any] | None = None,
        min_score: float | None = None,
        score_column: str | None = None,
        checkpoint_path: str | Path | None = None,
        coalesce_batches: bool = False,
        shuffle: bool = False,
        seed: int = DEFAULT_SHUFFLE_SEED,
    ) -> None:
        self._sources = list(sources)
        self._columns = list(columns) if columns is not None else None
        self._batch_size = max(1, int(batch_size))
        self._deduplicate = deduplicate
        self._auth_token = auth_token
        self._download_workers = download_workers
        self._tokenizer = tokenizer
        self._tokenizer_kwargs = tokenizer_kwargs
        self._postprocess_fn = postprocess_fn
        self._min_score = min_score
        self._score_column = score_column
        self._coalesce_batches = coalesce_batches
        self._shuffle = shuffle
        self._seed = seed
        self._checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )
        self._shard_idx = 0
        self._next_batch_id = 1
        self._committed_hashes: list[int] = []
        self._active_loader: MultimodalParquetLoader | None = None
        self._row_buffer: dict[str, list[Any]] = {}
        self._col_order: list[str] = []
        self._last_coord = StreamCoordinate()

        if self._checkpoint_path is not None and self._checkpoint_path.exists():
            try:
                state = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
                self._shard_idx = int(state.get("shard_idx", 0))
                self._next_batch_id = int(state.get("next_batch_id", 1))
                self._committed_hashes = [
                    int(h) for h in state.get("committed_hashes", [])
                ]
            except BaseException:
                pass

    def _save_checkpoint(self) -> None:
        if self._checkpoint_path is None:
            return
        if self._checkpoint_path.parent:
            self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "shard_idx": self._shard_idx,
                "next_batch_id": self._next_batch_id,
                "committed_hashes": self.export_committed_hashes(),
            }
        )
        tmp_file = self._checkpoint_path.with_suffix(".tmp")
        tmp_file.write_text(payload, encoding="utf-8")
        tmp_file.replace(self._checkpoint_path)

    def export_committed_hashes(self) -> list[int]:
        if self._active_loader is not None:
            return self._active_loader.export_committed_hashes()
        return list(self._committed_hashes)

    def import_committed_hashes(self, hashes: Sequence[int]) -> None:
        self._committed_hashes.extend(int(h) for h in hashes)
        if self._active_loader is not None:
            self._active_loader.import_committed_hashes(self._committed_hashes)

    def __iter__(self) -> _MultiShardMultimodalStream:
        return self

    def _open_next_shard(self) -> bool:
        if self._shard_idx >= len(self._sources):
            return False
        self._active_loader = MultimodalParquetLoader(
            self._sources[self._shard_idx],
            columns=self._columns,
            batch_size=self._batch_size,
            deduplicate=self._deduplicate,
            shard_id=self._shard_idx,
            auth_token=self._auth_token,
            download_workers=self._download_workers,
            tokenizer=None if self._coalesce_batches else self._tokenizer,
            tokenizer_kwargs=None if self._coalesce_batches else self._tokenizer_kwargs,
            postprocess_fn=None if self._coalesce_batches else self._postprocess_fn,
            min_score=self._min_score,
            score_column=self._score_column,
            shuffle=self._shuffle,
            seed=self._seed + self._shard_idx,
        )
        if self._deduplicate and self._committed_hashes:
            self._active_loader.import_committed_hashes(self._committed_hashes)
        self._shard_idx += 1
        return True

    def __next__(self) -> MultimodalBatch:
        if not self._coalesce_batches:
            while True:
                if self._active_loader is None:
                    if not self._open_next_shard():
                        self._save_checkpoint()
                        raise StopIteration

                assert self._active_loader is not None
                batch = self._active_loader.next_batch()
                if batch is not None:
                    if isinstance(batch, MultimodalBatch):
                        batch._batch_id_override = self._next_batch_id
                        if self._shard_idx < len(self._sources):
                            batch._is_eof_override = False
                    self._next_batch_id += 1
                    self._save_checkpoint()
                    return batch
                if self._deduplicate:
                    self._committed_hashes = (
                        self._active_loader.export_committed_hashes()
                    )
                self._active_loader = None

        while True:
            buffered_rows = (
                len(self._row_buffer[self._col_order[0]])
                if self._col_order and self._col_order[0] in self._row_buffer
                else 0
            )
            if buffered_rows >= self._batch_size:
                return self._emit_coalesced_batch(self._batch_size, is_eof=False)

            if self._active_loader is None:
                if not self._open_next_shard():
                    if buffered_rows > 0:
                        return self._emit_coalesced_batch(buffered_rows, is_eof=True)
                    self._save_checkpoint()
                    raise StopIteration

            assert self._active_loader is not None
            raw_b = self._active_loader.next_batch()
            if raw_b is None:
                if self._deduplicate:
                    self._committed_hashes = (
                        self._active_loader.export_committed_hashes()
                    )
                self._active_loader = None
                continue

            if not self._col_order:
                self._col_order = raw_b.column_names()
                for c in self._col_order:
                    self._row_buffer[c] = []
            b_dict = raw_b.to_dict()
            for c in self._col_order:
                self._row_buffer.setdefault(c, []).extend(b_dict.get(c, []))
            self._last_coord = raw_b.next_coord

    def _emit_coalesced_batch(self, count: int, is_eof: bool) -> MultimodalBatch:
        sliced_cols: dict[str, list[Any]] = {}
        for c in self._col_order:
            vals = self._row_buffer.get(c, [])
            sliced_cols[c] = vals[:count]
            del vals[:count]
        remaining = (
            len(self._row_buffer[self._col_order[0]]) if self._col_order else 0
        )
        final_eof = is_eof or (
            remaining == 0
            and self._active_loader is None
            and self._shard_idx >= len(self._sources)
        )
        raw_coalesced = _CoalescedMultimodalRawBatch(
            batch_id=self._next_batch_id,
            column_order=self._col_order,
            columns=sliced_cols,
            num_rows=count,
            next_coord=self._last_coord,
            is_eof=final_eof,
        )
        self._next_batch_id += 1
        self._save_checkpoint()
        batch: Any = MultimodalBatch(raw_coalesced)
        return _apply_batch_hooks(
            batch,
            tokenizer=self._tokenizer,
            tokenizer_kwargs=self._tokenizer_kwargs,
            postprocess_fn=self._postprocess_fn,
        )  # type: ignore[return-value]

    def close(self) -> None:
        self._save_checkpoint()
        self._active_loader = None
        self._shard_idx = len(self._sources)

    def __enter__(self) -> _MultiShardMultimodalStream:
        return self

    def __exit__(
        self,
        exc_type: object = None,
        exc_value: object = None,
        traceback: object = None,
    ) -> bool:
        self.close()
        return False


def stream(
    sources: Any,
    *,
    columns: Sequence[str] | str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    preram_batches: int = DEFAULT_PRERAM_BATCHES,
    refill_trigger: int = DEFAULT_REFILL_TRIGGER,
    download_workers: int = DEFAULT_DOWNLOAD_WORKERS,
    deduplicate: bool = True,
    auto_commit: bool = True,
    checkpoint_path: str | Path | None = None,
    json_extract_key: str | None = None,
    auth_token: str | None = None,
    tokenizer: Any = None,
    tokenizer_kwargs: Mapping[str, Any] | None = None,
    postprocess_fn: Callable[[Any], Any] | None = None,
    split: str | None = None,
    subset: str | None = None,
    min_score: float | None = None,
    score_column: str | None = None,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = False,
    seed: int = DEFAULT_SHUFFLE_SEED,
    shuffle_buffer_size: int = DEFAULT_SHUFFLE_BUFFER_SIZE,
    **kwargs: Any,
) -> DataStreamLoader | MultimodalParquetLoader | _MultiShardMultimodalStream:
    _validate_known_kwargs("stream", kwargs, VALID_STREAM_KWARGS)
    endpoint_kw = kwargs.pop("endpoint", None)
    secrets = resolve_hf_secrets(token=auth_token, endpoint=endpoint_kw)
    resolved_token = secrets["token"]
    expanded_sources = expand_hf_sources(
        sources,
        split=split,
        subset=subset,
        token=resolved_token,
        endpoint=secrets["endpoint"],
    )
    force_multimodal = bool(kwargs.pop("multimodal", False))
    coalesce_batches = bool(kwargs.pop("coalesce_batches", False))

    col_list: list[str] | None = None
    if columns is not None:
        if isinstance(columns, str):
            col_list = [columns]
        else:
            col_list = [str(c) for c in columns]

    is_webdataset_tar = any(
        s.split("?", 1)[0].lower().endswith(WEBDATASET_TAR_EXTENSIONS)
        for s in expanded_sources
    )

    if (
        force_multimodal
        or (col_list is not None and len(col_list) != 1)
        or (is_webdataset_tar and col_list is None)
    ):
        if shuffle and len(expanded_sources) > 1:
            rng = random.Random(seed)
            expanded_sources = list(expanded_sources)
            rng.shuffle(expanded_sources)
        rank_sources, _ = _partition_sources_for_ddp(
            expanded_sources, rank=rank, world_size=world_size
        )
        if len(rank_sources) == 1 and checkpoint_path is None and not coalesce_batches:
            return MultimodalParquetLoader(
                rank_sources[0],
                columns=col_list,
                batch_size=batch_size,
                deduplicate=deduplicate,
                auth_token=resolved_token,
                download_workers=download_workers,
                tokenizer=tokenizer,
                tokenizer_kwargs=tokenizer_kwargs,
                postprocess_fn=postprocess_fn,
                min_score=min_score,
                score_column=score_column,
                shuffle=shuffle,
                seed=seed + rank,
            )
        return _MultiShardMultimodalStream(
            rank_sources,
            columns=col_list,
            batch_size=batch_size,
            deduplicate=deduplicate,
            auth_token=resolved_token,
            download_workers=download_workers,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            postprocess_fn=postprocess_fn,
            min_score=min_score,
            score_column=score_column,
            checkpoint_path=checkpoint_path,
            coalesce_batches=coalesce_batches,
            shuffle=shuffle,
            seed=seed + rank,
        )

    if col_list is not None and len(col_list) == 1 and "parquet_column" not in kwargs:
        kwargs["parquet_column"] = col_list[0]

    return DataStreamLoader(
        expanded_sources,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        preram_batches=preram_batches,
        refill_trigger=refill_trigger,
        deduplicate=deduplicate,
        json_extract_key=json_extract_key,
        auth_token=resolved_token,
        download_workers=download_workers,
        auto_commit=auto_commit,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        postprocess_fn=postprocess_fn,
        min_score=min_score,
        score_column=score_column,
        rank=rank,
        world_size=world_size,
        shuffle=shuffle,
        seed=seed,
        shuffle_buffer_size=shuffle_buffer_size,
        **kwargs,
    )


def interleave(
    streams: Sequence[Iterable[Any]],
    *,
    probabilities: Sequence[float] | None = None,
    seed: int = DEFAULT_SHUFFLE_SEED,
    stopping_strategy: str = "first_exhausted",
    **kwargs: Any,
) -> Iterator[Any]:
    _validate_known_kwargs(
        "interleave",
        kwargs,
        VALID_INTERLEAVE_KWARGS,
    )
    if stopping_strategy not in VALID_INTERLEAVE_STRATEGIES:
        hint = _format_did_you_mean(stopping_strategy, VALID_INTERLEAVE_STRATEGIES)
        raise ValueError(
            f"Invalid stopping_strategy '{stopping_strategy}'.{hint} "
            f"Valid options: {list(VALID_INTERLEAVE_STRATEGIES)}"
        )
    if not streams:
        return
    iterators = [iter(s) for s in streams]
    active = list(range(len(iterators)))

    if probabilities is not None:
        if len(probabilities) != len(iterators):
            raise ValueError(
                f"probabilities length ({len(probabilities)}) must match streams length ({len(iterators)})"
            )
        weights = [float(p) for p in probabilities]
        if any(w < 0.0 for w in weights) or sum(weights) <= 0.0:
            raise ValueError("probabilities must be non-negative with a positive sum")
    else:
        weights = [1.0] * len(iterators)

    rng = random.Random(seed)
    while active:
        active_weights = [weights[i] for i in active]
        total_w = sum(active_weights)
        if total_w <= 0.0:
            chosen_pos = rng.randrange(len(active))
        else:
            r = rng.uniform(0.0, total_w)
            upto = 0.0
            chosen_pos = len(active) - 1
            for pos, w in enumerate(active_weights):
                upto += w
                if r <= upto:
                    chosen_pos = pos
                    break
        stream_idx = active[chosen_pos]
        try:
            yield next(iterators[stream_idx])
        except StopIteration:
            if stopping_strategy == "first_exhausted":
                return
            active.pop(chosen_pos)


def from_hf_dataset(
    hf_dataset: Iterable[Any],
    *,
    column: str = "text",
    batch_size: int = DEFAULT_BATCH_SIZE,
    deduplicate: bool = True,
    trim_whitespace: bool = True,
    normalize_newlines: bool = True,
    min_byte_len: int = 1,
    max_byte_len: int = 0,
    truncate_over_max: bool = False,
    json_extract_key: str | None = None,
    tokenizer: Any = None,
    tokenizer_kwargs: Mapping[str, Any] | None = None,
    postprocess_fn: Callable[[Any], Any] | None = None,
    **kwargs: Any,
) -> Iterator[DataBatch]:
    _validate_known_kwargs("from_hf_dataset", kwargs, VALID_FROM_HF_DATASET_KWARGS)
    target_size = max(1, int(batch_size))
    seen_hashes: list[int] = []
    raw_buffer: list[str] = []
    ready_records: list[str] = []
    batch_id = 1
    global_row_idx = 0

    def _flush_raw_buffer() -> None:
        nonlocal raw_buffer
        if not raw_buffer:
            return
        cleaned, new_hashes = postprocess_and_dedup_records(
            raw_buffer,
            trim_whitespace=trim_whitespace,
            normalize_newlines=normalize_newlines,
            min_byte_len=min_byte_len,
            max_byte_len=max_byte_len,
            truncate_over_max=truncate_over_max,
            json_extract_key=json_extract_key,
            seen_hashes=seen_hashes if deduplicate else None,
            deduplicate=deduplicate,
        )
        if deduplicate:
            seen_hashes.extend(new_hashes)
        ready_records.extend(cleaned)
        raw_buffer = []

    for item in hf_dataset:
        if isinstance(item, str):
            raw_buffer.append(item)
        elif isinstance(item, Mapping):
            if column in item:
                if item[column] is not None:
                    val = item[column]
                    raw_buffer.append(val if isinstance(val, str) else json.dumps(val))
            elif len(item) > 0:
                avail_keys = [str(k) for k in item.keys()]
                hint = _format_did_you_mean(column, avail_keys)
                raise KeyError(
                    f"Column '{column}' not found in dataset record.{hint} "
                    f"Available columns: {avail_keys}"
                )
        elif item is not None:
            raw_buffer.append(str(item))

        if len(raw_buffer) >= target_size:
            _flush_raw_buffer()
            while len(ready_records) >= target_size:
                slice_recs = ready_records[:target_size]
                del ready_records[:target_size]
                global_row_idx += len(slice_recs)
                coord = StreamCoordinate(
                    shard_id=0,
                    byte_offset=0,
                    row_group_idx=0,
                    page_idx=0,
                    row_offset_in_page=0,
                    global_row_idx=global_row_idx,
                )
                raw_obj = _InMemoryRawBatch(
                    batch_id=batch_id,
                    records=slice_recs,
                    next_coord=coord,
                    is_eof=False,
                )
                batch_id += 1
                batch: Any = DataBatch(raw_obj, column_name=column)
                yield _apply_batch_hooks(
                    batch,
                    tokenizer=tokenizer,
                    tokenizer_kwargs=tokenizer_kwargs,
                    postprocess_fn=postprocess_fn,
                    text_column=column,
                )  # type: ignore[misc]

    _flush_raw_buffer()
    while ready_records:
        slice_recs = ready_records[:target_size]
        del ready_records[:target_size]
        is_last = len(ready_records) == 0
        global_row_idx += len(slice_recs)
        coord = StreamCoordinate(
            shard_id=0,
            byte_offset=0,
            row_group_idx=0,
            page_idx=0,
            row_offset_in_page=0,
            global_row_idx=global_row_idx,
        )
        raw_obj = _InMemoryRawBatch(
            batch_id=batch_id,
            records=slice_recs,
            next_coord=coord,
            is_eof=is_last,
        )
        batch_id += 1
        batch = DataBatch(raw_obj, column_name=column)
        yield _apply_batch_hooks(
            batch,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            postprocess_fn=postprocess_fn,
            text_column=column,
        )  # type: ignore[misc]


__all__ = [
    "DataStreamLoader",
    "MultimodalParquetLoader",
    "_MultiShardMultimodalStream",
    "from_hf_dataset",
    "interleave",
    "stream",
]
