from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ._ffi import StreamCoordinate, _RawDataBatch, _RawMultimodalBatch
from .diagnostics import _format_did_you_mean

if TYPE_CHECKING:
    from .loader import DataStreamLoader


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


class _CoalescedMultimodalRawBatch:
    def __init__(
        self,
        batch_id: int,
        column_order: Sequence[str],
        columns: dict[str, list[Any]],
        num_rows: int,
        next_coord: StreamCoordinate,
        is_eof: bool,
    ) -> None:
        self.batch_id = batch_id
        self._column_order = list(column_order)
        self._columns = columns
        self.num_rows = num_rows
        self.next_coord = next_coord
        self.is_eof = is_eof

    def column_names(self) -> list[str]:
        return list(self._column_order)

    def column(self, name: str) -> list[Any]:
        if name in self._columns:
            return list(self._columns[name])
        hint = _format_did_you_mean(name, self._column_order)
        avail = ", ".join(f"'{c}'" for c in self._column_order)
        raise KeyError(
            f"Column '{name}' not found in MultimodalBatch.{hint} Available columns: [{avail}]"
        )

    def to_dict(self) -> dict[str, list[Any]]:
        return {col: list(self._columns.get(col, [])) for col in self._column_order}

    def __len__(self) -> int:
        return self.num_rows

    def __repr__(self) -> str:
        return (
            f"MultimodalBatch(batch_id={self.batch_id}, num_rows={self.num_rows}, "
            f"columns={self._column_order!r}, is_eof={self.is_eof})"
        )


class _BatchMeta(type):
    def __instancecheck__(cls, instance: object) -> bool:
        if super().__instancecheck__(instance):
            return True
        raw_cls = getattr(cls, "_raw_class", None)
        return raw_cls is not None and isinstance(instance, raw_cls)


def _convert_column_to_numpy(values: Sequence[Any], np_mod: Any) -> Any:
    if not values:
        return np_mod.array([], dtype=object)
    first_non_null = next((v for v in values if v is not None), None)
    if isinstance(first_non_null, bool):
        if all(isinstance(v, bool) for v in values):
            return np_mod.asarray(values, dtype=np_mod.bool_)
    elif isinstance(first_non_null, int):
        if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            return np_mod.asarray(values, dtype=np_mod.int64)
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            return np_mod.asarray(values, dtype=np_mod.float64)
    elif isinstance(first_non_null, float):
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            return np_mod.asarray(values, dtype=np_mod.float64)
    elif isinstance(first_non_null, list):
        try:
            return np_mod.asarray(values)
        except BaseException:
            pass
    return np_mod.asarray(values, dtype=object)


def _convert_column_to_torch(
    values: Sequence[Any],
    torch_mod: Any,
    device: Any = None,
    dtype: Any = None,
) -> Any:
    if not values:
        return list(values)
    first_non_null = next((v for v in values if v is not None), None)
    if isinstance(first_non_null, bool) and all(isinstance(v, bool) for v in values):
        t = torch_mod.tensor(values, dtype=torch_mod.bool, device=device)
        return t
    if isinstance(first_non_null, int) and not isinstance(first_non_null, bool):
        if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            return torch_mod.tensor(values, dtype=torch_mod.long, device=device)
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            target_dtype = dtype if dtype is not None else torch_mod.float32
            return torch_mod.tensor(values, dtype=target_dtype, device=device)
    if isinstance(first_non_null, float) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
    ):
        target_dtype = dtype if dtype is not None else torch_mod.float32
        return torch_mod.tensor(values, dtype=target_dtype, device=device)
    if isinstance(first_non_null, list) and first_non_null:
        elem0 = first_non_null[0]
        if isinstance(elem0, (int, float)) and not isinstance(elem0, bool):
            try:
                is_float = any(
                    isinstance(x, float) for row in values if isinstance(row, list) for x in row
                )
                target_dtype = (
                    dtype
                    if dtype is not None
                    else (torch_mod.float32 if is_float else torch_mod.long)
                )
                return torch_mod.tensor(values, dtype=target_dtype, device=device)
            except BaseException:
                pass
    return list(values)


class DataBatch(metaclass=_BatchMeta):
    _raw_class = _RawDataBatch

    def __init__(
        self,
        raw: _RawDataBatch | _InMemoryRawBatch,
        loader: DataStreamLoader | None = None,
        tokens: Any = None,
        column_name: str = "text",
    ) -> None:
        self._raw = raw
        self._loader = loader
        self._committed = False
        self.tokens: Any = tokens
        self._column_name = column_name

    @property
    def column_name(self) -> str:
        return self._column_name

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
        out: dict[str, list[str]] = {"text": list(self._raw.records)}
        if self._column_name != "text":
            out[self._column_name] = list(self._raw.records)
        return out

    def to_numpy(self) -> dict[str, Any]:
        import numpy as np

        out: dict[str, Any] = {"text": _convert_column_to_numpy(self._raw.records, np)}
        if self._column_name != "text":
            out[self._column_name] = out["text"]
        if isinstance(self.tokens, Mapping):
            for k, v in self.tokens.items():
                out[str(k)] = np.asarray(v)
        return out

    def to_torch(self, device: Any = None, dtype: Any = None) -> dict[str, Any]:
        import torch  # type: ignore[import-not-found]

        out: dict[str, Any] = {
            "text": _convert_column_to_torch(
                self._raw.records, torch, device=device, dtype=dtype
            )
        }
        if self._column_name != "text":
            out[self._column_name] = out["text"]
        if isinstance(self.tokens, Mapping):
            for k, v in self.tokens.items():
                out[str(k)] = (
                    v.to(device=device)
                    if hasattr(v, "to")
                    else _convert_column_to_torch(v, torch, device=device, dtype=dtype)
                )
        return out

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
        from .hub import push_to_hub

        target_path = path_in_repo or f"data/part-{self.batch_id:05d}.parquet"
        return push_to_hub(repo, self, path_in_repo=target_path, **kwargs)

    def __len__(self) -> int:
        return len(self._raw)

    def __iter__(self) -> Iterator[str]:
        return iter(self._raw.records)

    def __getitem__(self, key: int | slice | str) -> Any:
        if isinstance(key, str):
            if key == "text" or key == self._column_name:
                return list(self._raw.records)
            if key == "tokens" and self.tokens is not None:
                return self.tokens
            available = ["text"]
            if self._column_name != "text":
                available.append(self._column_name)
            if self.tokens is not None:
                available.append("tokens")
            hint = _format_did_you_mean(key, available)
            avail_str = ", ".join(f"'{k}'" for k in available)
            raise KeyError(
                f"Column '{key}' not found in DataBatch.{hint} Available keys: [{avail_str}]"
            )
        return self._raw.records[key]

    def __repr__(self) -> str:
        return repr(self._raw)


class MultimodalBatch(metaclass=_BatchMeta):
    _raw_class = _RawMultimodalBatch

    def __init__(
        self,
        raw: _RawMultimodalBatch | _CoalescedMultimodalRawBatch,
        tokens: Any = None,
        batch_id_override: int | None = None,
        is_eof_override: bool | None = None,
    ) -> None:
        self._raw = raw
        self.tokens: Any = tokens
        self._batch_id_override = batch_id_override
        self._is_eof_override = is_eof_override

    @property
    def batch_id(self) -> int:
        if self._batch_id_override is not None:
            return self._batch_id_override
        return self._raw.batch_id

    @property
    def num_rows(self) -> int:
        return self._raw.num_rows

    @property
    def next_coord(self) -> StreamCoordinate:
        return self._raw.next_coord

    @property
    def is_eof(self) -> bool:
        if self._is_eof_override is not None:
            return self._is_eof_override
        return self._raw.is_eof

    def column_names(self) -> list[str]:
        return self._raw.column_names()

    def column(self, name: str) -> list[Any]:
        return self._raw.column(name)

    def to_dict(self) -> dict[str, list[Any]]:
        return self._raw.to_dict()

    def to_numpy(self) -> dict[str, Any]:
        import numpy as np

        raw_dict = self._raw.to_dict()
        out: dict[str, Any] = {
            col: _convert_column_to_numpy(raw_dict[col], np)
            for col in self._raw.column_names()
        }
        if isinstance(self.tokens, Mapping):
            for k, v in self.tokens.items():
                out[str(k)] = np.asarray(v)
        return out

    def to_torch(self, device: Any = None, dtype: Any = None) -> dict[str, Any]:
        import torch  # type: ignore[import-not-found]

        raw_dict = self._raw.to_dict()
        out: dict[str, Any] = {
            col: _convert_column_to_torch(
                raw_dict[col], torch, device=device, dtype=dtype
            )
            for col in self._raw.column_names()
        }
        if isinstance(self.tokens, Mapping):
            for k, v in self.tokens.items():
                out[str(k)] = (
                    v.to(device=device)
                    if hasattr(v, "to")
                    else _convert_column_to_torch(v, torch, device=device, dtype=dtype)
                )
        return out

    def commit(self) -> StreamCoordinate:
        return self._raw.next_coord

    def push_to_hub(
        self,
        repo: str,
        path_in_repo: str | None = None,
        **kwargs: Any,
    ) -> str:
        from .hub import push_to_hub

        target_path = path_in_repo or f"data/part-{self.batch_id:05d}.parquet"
        return push_to_hub(repo, self, path_in_repo=target_path, **kwargs)

    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, key: str | int | slice) -> Any:
        if isinstance(key, str):
            if key == "tokens" and self.tokens is not None:
                return self.tokens
            try:
                return self._raw.column(key)
            except ValueError as e:
                raise KeyError(str(e)) from e
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
            target_col = (
                text_column
                if text_column in col_names
                else (col_names[0] if col_names else "")
            )
            raw_vals = batch.column(target_col) if target_col else []
            texts = [str(v) if v is not None else "" for v in raw_vals]
        batch.tokens = tokenizer(texts, **tk_kwargs)

    if postprocess_fn is not None:
        res = postprocess_fn(batch)
        if isinstance(res, (DataBatch, MultimodalBatch)):
            return res
    return batch


__all__ = [
    "DataBatch",
    "MultimodalBatch",
    "_CoalescedMultimodalRawBatch",
    "_InMemoryRawBatch",
    "_apply_batch_hooks",
    "_convert_column_to_numpy",
    "_convert_column_to_torch",
]
