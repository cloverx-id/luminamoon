from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._ffi import serialize_table_to_bytes


def serialize_to_parquet(
    columns: Mapping[str, Sequence[Any]] | Sequence[tuple[str, Sequence[Any]]],
    output_path: str | Path | None = None,
    hf_image_struct: bool = True,
) -> bytes:
    """Serializes a columnar table into Apache Parquet binary bytes with ZSTD compression in Rust.

    The GIL is released during encoding. If `output_path` is provided, writes directly to disk.
    """
    raw_bytes = serialize_table_to_bytes(columns, format="parquet", hf_image_struct=hf_image_struct)
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw_bytes)
    return raw_bytes


def serialize_to_jsonl(
    columns: Mapping[str, Sequence[Any]] | Sequence[tuple[str, Sequence[Any]]],
    output_path: str | Path | None = None,
) -> bytes:
    """Serializes a columnar table into line-delimited JSONL UTF-8 bytes in Rust."""
    raw_bytes = serialize_table_to_bytes(columns, format="jsonl", hf_image_struct=False)
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw_bytes)
    return raw_bytes


def serialize_to_arrow(
    columns: Mapping[str, Sequence[Any]] | Sequence[tuple[str, Sequence[Any]]],
    output_path: str | Path | None = None,
) -> bytes:
    """Serializes a columnar table into Apache Arrow IPC / Feather binary bytes in Rust."""
    raw_bytes = serialize_table_to_bytes(columns, format="arrow", hf_image_struct=False)
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw_bytes)
    return raw_bytes


__all__ = [
    "serialize_to_arrow",
    "serialize_to_jsonl",
    "serialize_to_parquet",
]
