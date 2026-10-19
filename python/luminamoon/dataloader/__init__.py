from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import hashlib
import io
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
import urllib.request

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
        def commit_gpu_batch(self, batch_id: int) -> StreamCoordinate: ...
        def next_batch(self) -> Any: ...
        def download_workers(self) -> int: ...
        def close(self) -> None: ...

    class _RawMultimodalParquetLoader:
        def __new__(cls, *args: Any, **kwargs: Any) -> Any: ...
        def next_batch(self) -> Any: ...

    class HfHubUploader:
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...
        def create_repo(self, *args: Any, **kwargs: Any) -> str: ...
        def upload_file(self, *args: Any, **kwargs: Any) -> str: ...
        def upload_bytes(self, *args: Any, **kwargs: Any) -> str: ...
        def upload_table(self, *args: Any, **kwargs: Any) -> str: ...
        def upload_jsonl(self, *args: Any, **kwargs: Any) -> str: ...

    def dataloader_cpu_isa() -> str: ...
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

        class HfHubUploader:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError(
                    "luminamoon native Rust extension (_luminamoon_optimizer_rs) is not compiled"
                )

        def dataloader_cpu_isa() -> str:
            return "SCALAR"

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


def _lookup_secret_key(keys: Sequence[str]) -> str | None:
    for key in keys:
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip()

    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        for key in keys:
            try:
                val = userdata.get(key)
                if val and str(val).strip():
                    return str(val).strip()
            except BaseException:
                continue
    except BaseException:
        pass

    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        client = UserSecretsClient()
        for key in keys:
            try:
                val = client.get_secret(key)
                if val and str(val).strip():
                    return str(val).strip()
            except BaseException:
                continue
    except BaseException:
        pass

    return None


def resolve_hf_secrets(
    token: str | None = None,
    username: str | None = None,
    org: str | None = None,
    endpoint: str | None = None,
) -> dict[str, str | None]:
    resolved_token = (
        token.strip()
        if token and token.strip()
        else _lookup_secret_key(("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"))
    )
    resolved_username = (
        username.strip()
        if username and username.strip()
        else _lookup_secret_key(("HF_USERNAME",))
    )
    resolved_org = (
        org.strip() if org and org.strip() else _lookup_secret_key(("HF_ORG",))
    )
    resolved_endpoint = (
        endpoint.strip()
        if endpoint and endpoint.strip()
        else _lookup_secret_key(("HF_ENDPOINT",))
    )
    return {
        "token": resolved_token,
        "username": resolved_username,
        "org": resolved_org,
        "endpoint": resolved_endpoint,
    }


class _InMemoryRawBatch:
    def __init__(
        self,
        batch_id: int,
        records: list[str],
        next_coord: StreamCoordinate,
        is_eof: bool,
    ) -> None:
        self.batch_id = batch_id
        self.records = records
        self.next_coord = next_coord
        self.is_eof = is_eof

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        return (
            f"DataBatch(batch_id={self.batch_id}, len={len(self.records)}, "
            f"is_eof={self.is_eof})"
        )


class _BatchMeta(type):
    def __instancecheck__(cls, instance: object) -> bool:
        if super().__instancecheck__(instance):
            return True
        raw_cls = getattr(cls, "_raw_class", None)
        return raw_cls is not None and isinstance(instance, raw_cls)


class DataBatch(metaclass=_BatchMeta):
    _raw_class = _RawDataBatch

    def __init__(
        self,
        raw: _RawDataBatch | _InMemoryRawBatch,
        loader: DataStreamLoader | None = None,
        tokens: Any = None,
    ) -> None:
        self._raw = raw
        self._loader = loader
        self._committed = False
        self.tokens: Any = tokens

    @property
    def batch_id(self) -> int:
        return self._raw.batch_id

    @property
    def records(self) -> list[str]:
        return self._raw.records

    @property
    def next_coord(self) -> StreamCoordinate:
        return self._raw.next_coord

    @property
    def is_eof(self) -> bool:
        return self._raw.is_eof

    def to_dict(self) -> dict[str, list[str]]:
        return {"text": list(self._raw.records)}

    def commit(self) -> StreamCoordinate:
        if self._loader is None:
            return self._raw.next_coord
        coord = self._loader.mark_in_gpu(self._raw.batch_id)
        self._committed = True
        return coord

    def push_to_hub(
        self,
        repo: str,
        path_in_repo: str | None = None,
        **kwargs: Any,
    ) -> str:
        target_path = path_in_repo or f"data/part-{self.batch_id:05d}.parquet"
        return push_to_hub(repo, self, path_in_repo=target_path, **kwargs)

    def __len__(self) -> int:
        return len(self._raw)

    def __iter__(self) -> Iterator[str]:
        return iter(self._raw.records)

    def __getitem__(self, key: int | slice | str) -> Any:
        if isinstance(key, str):
            if key == "text":
                return list(self._raw.records)
            if key == "tokens" and self.tokens is not None:
                return self.tokens
            raise KeyError(key)
        return self._raw.records[key]

    def __repr__(self) -> str:
        return repr(self._raw)


class MultimodalBatch(metaclass=_BatchMeta):
    _raw_class = _RawMultimodalBatch

    def __init__(self, raw: _RawMultimodalBatch, tokens: Any = None) -> None:
        self._raw = raw
        self.tokens: Any = tokens

    @property
    def batch_id(self) -> int:
        return self._raw.batch_id

    @property
    def num_rows(self) -> int:
        return self._raw.num_rows

    @property
    def next_coord(self) -> StreamCoordinate:
        return self._raw.next_coord

    @property
    def is_eof(self) -> bool:
        return self._raw.is_eof

    def column_names(self) -> list[str]:
        return self._raw.column_names()

    def column(self, name: str) -> list[Any]:
        return self._raw.column(name)

    def to_dict(self) -> dict[str, list[Any]]:
        return self._raw.to_dict()

    def commit(self) -> StreamCoordinate:
        return self._raw.next_coord

    def push_to_hub(
        self,
        repo: str,
        path_in_repo: str | None = None,
        **kwargs: Any,
    ) -> str:
        target_path = path_in_repo or f"data/part-{self.batch_id:05d}.parquet"
        return push_to_hub(repo, self, path_in_repo=target_path, **kwargs)

    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, key: str | int | slice) -> Any:
        if isinstance(key, str):
            if key == "tokens" and self.tokens is not None:
                return self.tokens
            return self._raw.column(key)
        cols = self._raw.to_dict()
        names = self._raw.column_names()
        n = self._raw.num_rows
        if isinstance(key, int):
            idx = key + n if key < 0 else key
            if idx < 0 or idx >= n:
                raise IndexError("MultimodalBatch row index out of range")
            return {name: cols[name][idx] for name in names}
        if isinstance(key, slice):
            indices = range(*key.indices(n))
            return [{name: cols[name][i] for name in names} for i in indices]
        raise TypeError(f"Unsupported key type for MultimodalBatch: {type(key)!r}")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        cols = self._raw.to_dict()
        names = self._raw.column_names()
        for i in range(self._raw.num_rows):
            yield {name: cols[name][i] for name in names}

    def __repr__(self) -> str:
        return repr(self._raw)


def _normalize_sources_list(sources: str | Path | Sequence[str | Path]) -> list[str]:
    if isinstance(sources, (str, Path)):
        return [str(sources)]
    return [str(s) for s in sources]


_KNOWN_SHARD_EXTENSIONS = (".parquet", ".pq", ".jsonl", ".ndjson", ".json", ".txt")


def _is_loopback_endpoint(endpoint: str | None) -> bool:
    if not endpoint:
        return False
    ep = endpoint.strip().lower()
    return ep.startswith(("http://127.0.0.1", "http://localhost", "http://[::1]"))


def _parse_next_link_header(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        sections = part.split(";")
        if len(sections) >= 2 and any(
            s.strip() in ('rel="next"', "rel=next") for s in sections[1:]
        ):
            target = sections[0].strip()
            if target.startswith("<") and target.endswith(">"):
                return target[1:-1].strip()
    return None


def _list_hf_repo_files(
    org: str,
    repo: str,
    revision: str,
    token: str | None,
    endpoint: str | None,
) -> list[str]:
    base_ep = (endpoint or "https://huggingface.co").strip().rstrip("/")
    repo_id = f"{org}/{repo}"

    if not _is_loopback_endpoint(base_ep):
        try:
            from huggingface_hub import HfApi  # type: ignore[import-not-found]

            api = HfApi(token=token, endpoint=base_ep)
            files = api.list_repo_files(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
            )
            return [str(f) for f in files]
        except ImportError:
            pass
        except BaseException:
            pass

    next_url: str | None = (
        f"{base_ep}/api/datasets/{org}/{repo}/tree/{revision}?recursive=true"
    )
    out: list[str] = []
    pages_fetched = 0

    while next_url and pages_fetched < 50:
        pages_fetched += 1
        headers: dict[str, str] = {"Accept": "application/json"}
        if token and (
            next_url.lower().startswith("https://") or _is_loopback_endpoint(next_url)
        ):
            headers["Authorization"] = f"Bearer {token}"

        req = urllib.request.Request(next_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            link_hdr = resp.headers.get("Link") or resp.headers.get("link")

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, Mapping) and "path" in item:
                    if item.get("type", "file") != "directory":
                        out.append(str(item["path"]))
                elif isinstance(item, str):
                    out.append(item)

        next_url = _parse_next_link_header(link_hdr)

    return out


def expand_hf_sources(
    sources: str | Path | Sequence[str | Path],
    *,
    split: str | None = None,
    subset: str | None = None,
    token: str | None = None,
    endpoint: str | None = None,
) -> list[str]:
    normalized = _normalize_sources_list(sources)
    secrets = resolve_hf_secrets(token=token, endpoint=endpoint)
    resolved_token = secrets["token"]
    resolved_endpoint = secrets["endpoint"]

    expanded: list[str] = []
    for raw_src in normalized:
        trimmed = raw_src.strip()
        if not trimmed.startswith("hf://datasets/"):
            expanded.append(trimmed)
            continue

        rest = trimmed[len("hf://datasets/") :].strip("/")
        parts = [p for p in rest.split("/") if p]
        if len(parts) < 2:
            expanded.append(trimmed)
            continue

        org = parts[0]
        repo_and_rev = parts[1]
        if "@" in repo_and_rev:
            repo, rev = repo_and_rev.split("@", 1)
        else:
            repo, rev = repo_and_rev, "main"

        subpath = "/".join(parts[2:])
        clean_sub = subpath.split("?", 1)[0].lower()
        if clean_sub.endswith(_KNOWN_SHARD_EXTENSIONS):
            expanded.append(trimmed)
            continue

        all_files = _list_hf_repo_files(
            org=org,
            repo=repo,
            revision=rev,
            token=resolved_token,
            endpoint=resolved_endpoint,
        )

        candidates: list[str] = []
        for path in all_files:
            low_path = path.lower()
            if not low_path.endswith((".parquet", ".pq", ".jsonl", ".ndjson")):
                continue
            if subpath and not (
                path == subpath or path.startswith(subpath.rstrip("/") + "/")
            ):
                continue
            if subset and subset.lower() not in low_path:
                continue
            if split and split.lower() not in low_path:
                continue
            candidates.append(path)

        parquet_candidates = [
            p for p in candidates if p.lower().endswith((".parquet", ".pq"))
        ]
        selected = sorted(parquet_candidates if parquet_candidates else candidates)
        if not selected:
            raise ValueError(
                f"No matching dataset shards (.parquet/.jsonl) found in {trimmed}"
            )
        for rel_path in selected:
            expanded.append(f"hf://datasets/{org}/{repo}@{rev}/{rel_path}")

    return expanded


def _apply_batch_hooks(
    batch: DataBatch | MultimodalBatch,
    tokenizer: Any = None,
    tokenizer_kwargs: Mapping[str, Any] | None = None,
    postprocess_fn: Callable[[Any], Any] | None = None,
    text_column: str = "text",
) -> DataBatch | MultimodalBatch:
    if tokenizer is not None:
        tk_kwargs = (
            dict(tokenizer_kwargs)
            if tokenizer_kwargs is not None
            else {"padding": True, "truncation": True}
        )
        if isinstance(batch, DataBatch):
            texts = list(batch.records)
        else:
            col_names = batch.column_names()
            target_col = text_column if text_column in col_names else (col_names[0] if col_names else "")
            raw_vals = batch.column(target_col) if target_col else []
            texts = [str(v) if v is not None else "" for v in raw_vals]
        batch.tokens = tokenizer(texts, **tk_kwargs)

    if postprocess_fn is not None:
        res = postprocess_fn(batch)
        if isinstance(res, (DataBatch, MultimodalBatch)):
            return res
    return batch


def _default_checkpoint_path(normalized_sources: Sequence[str]) -> Path:
    digest = hashlib.sha256("\n".join(normalized_sources).encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(".luminamoon_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"stream_{digest}.meta"


class DataStreamLoader(_RawDataStreamLoader):
    def __new__(
        cls,
        sources: str | Path | Sequence[str | Path],
        checkpoint_path: str | Path | None = None,
        batch_size: int = 64,
        preram_batches: int = 100,
        refill_trigger: int = 10,
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
        remote_chunk_bytes: int = 262_144,
        download_workers: int = 4,
        auto_commit: bool = False,
        tokenizer: Any = None,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        postprocess_fn: Callable[[Any], Any] | None = None,
        split: str | None = None,
        subset: str | None = None,
    ):
        secrets = resolve_hf_secrets(token=auth_token)
        resolved_token = secrets["token"]
        expanded_sources = expand_hf_sources(
            sources,
            split=split,
            subset=subset,
            token=resolved_token,
            endpoint=secrets["endpoint"],
        )
        resolved_ckpt = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else _default_checkpoint_path(expanded_sources)
        )
        if resolved_ckpt.parent:
            resolved_ckpt.parent.mkdir(parents=True, exist_ok=True)

        instance = super().__new__(
            cls,
            expanded_sources,
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
        )
        instance._resolved_checkpoint_path = resolved_ckpt
        return instance

    def __init__(
        self,
        sources: str | Path | Sequence[str | Path],
        checkpoint_path: str | Path | None = None,
        batch_size: int = 64,
        preram_batches: int = 100,
        refill_trigger: int = 10,
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
        remote_chunk_bytes: int = 262_144,
        download_workers: int = 4,
        auto_commit: bool = False,
        tokenizer: Any = None,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        postprocess_fn: Callable[[Any], Any] | None = None,
        split: str | None = None,
        subset: str | None = None,
    ) -> None:
        self._checkpoint_path: Path = getattr(
            self,
            "_resolved_checkpoint_path",
            Path(checkpoint_path)
            if checkpoint_path is not None
            else _default_checkpoint_path(_normalize_sources_list(sources)),
        )
        self._auto_commit = bool(auto_commit)
        self._last_uncommitted_batch_id: int | None = None
        self._tokenizer = tokenizer
        self._tokenizer_kwargs = tokenizer_kwargs
        self._postprocess_fn = postprocess_fn
        self._parquet_column = parquet_column

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
        self._flush_pending_auto_commit()
        raw = super().next_batch()
        if raw is None:
            return None
        if self._auto_commit:
            self._last_uncommitted_batch_id = raw.batch_id
        batch: Any = DataBatch(raw, loader=self)
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
        batch_size: int = 64,
        deduplicate: bool = True,
        shard_id: int = 0,
        auth_token: str | None = None,
        download_workers: int = 4,
        tokenizer: Any = None,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        postprocess_fn: Callable[[Any], Any] | None = None,
    ):
        norm_cols = [str(c) for c in columns] if columns is not None else None
        resolved_token = resolve_hf_secrets(token=auth_token)["token"]
        return super().__new__(
            cls,
            str(path),
            columns=norm_cols,
            batch_size=batch_size,
            deduplicate=deduplicate,
            shard_id=shard_id,
            auth_token=resolved_token,
            download_workers=download_workers,
        )

    def __init__(
        self,
        path: str | Path,
        columns: Sequence[str] | None = None,
        batch_size: int = 64,
        deduplicate: bool = True,
        shard_id: int = 0,
        auth_token: str | None = None,
        download_workers: int = 4,
        tokenizer: Any = None,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        postprocess_fn: Callable[[Any], Any] | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._tokenizer_kwargs = tokenizer_kwargs
        self._postprocess_fn = postprocess_fn

    def next_batch(self) -> MultimodalBatch | None:
        raw = super().next_batch()
        if raw is None:
            return None
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
        download_workers: int = 4,
        tokenizer: Any = None,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        postprocess_fn: Callable[[Any], Any] | None = None,
    ) -> None:
        self._sources = list(sources)
        self._columns = list(columns) if columns is not None else None
        self._batch_size = batch_size
        self._deduplicate = deduplicate
        self._auth_token = auth_token
        self._download_workers = download_workers
        self._tokenizer = tokenizer
        self._tokenizer_kwargs = tokenizer_kwargs
        self._postprocess_fn = postprocess_fn
        self._shard_idx = 0
        self._active_loader: MultimodalParquetLoader | None = None

    def __iter__(self) -> _MultiShardMultimodalStream:
        return self

    def __next__(self) -> MultimodalBatch:
        while True:
            if self._active_loader is None:
                if self._shard_idx >= len(self._sources):
                    raise StopIteration
                self._active_loader = MultimodalParquetLoader(
                    self._sources[self._shard_idx],
                    columns=self._columns,
                    batch_size=self._batch_size,
                    deduplicate=self._deduplicate,
                    shard_id=self._shard_idx,
                    auth_token=self._auth_token,
                    download_workers=self._download_workers,
                    tokenizer=self._tokenizer,
                    tokenizer_kwargs=self._tokenizer_kwargs,
                    postprocess_fn=self._postprocess_fn,
                )
                self._shard_idx += 1

            batch = self._active_loader.next_batch()
            if batch is not None:
                return batch
            self._active_loader = None

    def close(self) -> None:
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
    sources: str | Path | Sequence[str | Path],
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = 64,
    preram_batches: int = 100,
    refill_trigger: int = 10,
    download_workers: int = 4,
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
    **kwargs: Any,
) -> DataStreamLoader | MultimodalParquetLoader | _MultiShardMultimodalStream:
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

    col_list: list[str] | None = None
    if columns is not None:
        if isinstance(columns, str):
            col_list = [columns]
        else:
            col_list = [str(c) for c in columns]

    if force_multimodal or (col_list is not None and len(col_list) > 1):
        if len(expanded_sources) == 1:
            return MultimodalParquetLoader(
                expanded_sources[0],
                columns=col_list,
                batch_size=batch_size,
                deduplicate=deduplicate,
                auth_token=resolved_token,
                download_workers=download_workers,
                tokenizer=tokenizer,
                tokenizer_kwargs=tokenizer_kwargs,
                postprocess_fn=postprocess_fn,
            )
        return _MultiShardMultimodalStream(
            expanded_sources,
            columns=col_list,
            batch_size=batch_size,
            deduplicate=deduplicate,
            auth_token=resolved_token,
            download_workers=download_workers,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            postprocess_fn=postprocess_fn,
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
        **kwargs,
    )


def from_hf_dataset(
    hf_dataset: Iterable[Any],
    *,
    column: str = "text",
    batch_size: int = 64,
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
) -> Iterator[DataBatch]:
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
            if column in item and item[column] is not None:
                val = item[column]
                raw_buffer.append(val if isinstance(val, str) else json.dumps(val))
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
                batch: Any = DataBatch(raw_obj)
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
        batch = DataBatch(raw_obj)
        yield _apply_batch_hooks(
            batch,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            postprocess_fn=postprocess_fn,
            text_column=column,
        )  # type: ignore[misc]


def _normalize_table_columns(
    data: Any,
    columns: Sequence[str] | None = None,
    rename: Mapping[str, str] | None = None,
) -> list[tuple[str, list[Any]]]:
    ordered: list[tuple[str, list[Any]]] = []

    if isinstance(data, DataBatch):
        ordered = [("text", list(data.records))]
    elif isinstance(data, MultimodalBatch):
        raw_dict = data.to_dict()
        ordered = [(col, list(raw_dict[col])) for col in data.column_names()]
    elif isinstance(data, Mapping):
        ordered = [(str(k), list(v)) for k, v in data.items()]
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        if len(data) == 0:
            ordered = []
        elif isinstance(data[0], Mapping):
            col_names: list[str] = []
            col_set: set[str] = set()
            for row in data:
                if not isinstance(row, Mapping):
                    raise ValueError("All elements in row-oriented data list must be dicts")
                for k in row.keys():
                    ks = str(k)
                    if ks not in col_set:
                        col_set.add(ks)
                        col_names.append(ks)
            ordered = [
                (name, [row.get(name, None) for row in data]) for name in col_names
            ]
        else:
            for item in data:
                if not isinstance(item, Sequence) or len(item) != 2:
                    raise ValueError(
                        "Expected (column_name, values) pairs or list[dict[str, Any]]"
                    )
                ordered.append((str(item[0]), list(item[1])))
    else:
        raise TypeError(
            "Unsupported data type for push_to_hub; expected DataBatch, MultimodalBatch, dict, or list of dicts"
        )

    if columns is not None:
        wanted = [str(c) for c in columns]
        lookup = dict(ordered)
        missing = [c for c in wanted if c not in lookup]
        if missing:
            raise ValueError(f"Requested columns not found in data: {missing}")
        ordered = [(c, lookup[c]) for c in wanted]

    if rename:
        rename_map = {str(k): str(v) for k, v in rename.items()}
        ordered = [(rename_map.get(col, col), vals) for col, vals in ordered]

    return ordered


def push_to_hub(
    repo: str,
    data: Any,
    path_in_repo: str = "data/part-00000.parquet",
    *,
    columns: Sequence[str] | None = None,
    rename: Mapping[str, str] | None = None,
    private: bool = True,
    revision: str = "main",
    commit_message: str = "Upload dataset shard via LuminaMoon",
    hf_image_struct: bool = True,
    token: str | None = None,
    username: str | None = None,
    org: str | None = None,
    endpoint: str | None = None,
) -> str:
    secrets = resolve_hf_secrets(
        token=token,
        username=username,
        org=org,
        endpoint=endpoint,
    )
    ordered_cols = _normalize_table_columns(data, columns=columns, rename=rename)
    payload_bytes = bytes(
        serialize_table_to_bytes(
            ordered_cols,
            path_in_repo=path_in_repo,
            hf_image_struct=hf_image_struct,
        )
    )

    resolved_endpoint = secrets["endpoint"]
    if not _is_loopback_endpoint(resolved_endpoint):
        try:
            from huggingface_hub import HfApi  # type: ignore[import-not-found]

            trimmed_repo = repo.strip().strip("/")
            if "/" in trimmed_repo:
                full_repo_id = trimmed_repo
            else:
                owner = secrets["org"] or secrets["username"]
                if not owner:
                    raise ValueError(
                        "Cannot resolve full repo_id without org/username or 'owner/repo' format"
                    )
                full_repo_id = f"{owner}/{trimmed_repo}"

            clean_path = path_in_repo.strip().lstrip("/")
            api = HfApi(token=secrets["token"], endpoint=resolved_endpoint)
            api.create_repo(
                repo_id=full_repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True,
            )
            api.upload_file(
                path_or_fileobj=io.BytesIO(payload_bytes),
                path_in_repo=clean_path,
                repo_id=full_repo_id,
                repo_type="dataset",
                revision=revision,
                commit_message=commit_message,
            )
            return f"hf://datasets/{full_repo_id}@{revision}/{clean_path}"
        except ImportError:
            pass

    uploader = HfHubUploader(
        token=secrets["token"],
        username=secrets["username"],
        org=secrets["org"],
        endpoint=resolved_endpoint,
    )
    uploader.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    return uploader.upload_bytes(
        repo,
        path_in_repo,
        payload_bytes,
        revision=revision,
        commit_message=commit_message,
    )


__all__ = [
    "DataBatch",
    "DataStreamLoader",
    "HfHubUploader",
    "MultimodalBatch",
    "MultimodalParquetLoader",
    "StreamCoordinate",
    "dataloader_cpu_isa",
    "expand_hf_sources",
    "from_hf_dataset",
    "postprocess_and_dedup_records",
    "push_to_hub",
    "resolve_hf_secrets",
    "serialize_table_to_bytes",
    "stream",
]
