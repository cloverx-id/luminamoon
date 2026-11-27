from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import threading
from typing import Any

from ._ffi import serialize_table_to_bytes


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{path}.tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def serialize_to_parquet(
    columns: Mapping[str, Sequence[Any]] | Sequence[tuple[str, Sequence[Any]]],
    output_path: str | Path | None = None,
    hf_image_struct: bool = True,
) -> bytes:
    """Serializes a columnar table into Apache Parquet binary bytes with ZSTD compression in Rust."""
    raw_bytes = serialize_table_to_bytes(columns, format="parquet", hf_image_struct=hf_image_struct)
    if output_path:
        _write_bytes_atomic(Path(output_path), raw_bytes)
    return raw_bytes


def serialize_to_jsonl(
    columns: Mapping[str, Sequence[Any]] | Sequence[tuple[str, Sequence[Any]]],
    output_path: str | Path | None = None,
) -> bytes:
    """Serializes a columnar table into line-delimited JSONL UTF-8 bytes in Rust."""
    raw_bytes = serialize_table_to_bytes(columns, format="jsonl", hf_image_struct=False)
    if output_path:
        _write_bytes_atomic(Path(output_path), raw_bytes)
    return raw_bytes


def serialize_to_arrow(
    columns: Mapping[str, Sequence[Any]] | Sequence[tuple[str, Sequence[Any]]],
    output_path: str | Path | None = None,
) -> bytes:
    """Serializes a columnar table into Apache Arrow IPC / Feather binary bytes in Rust."""
    raw_bytes = serialize_table_to_bytes(columns, format="arrow", hf_image_struct=False)
    if output_path:
        _write_bytes_atomic(Path(output_path), raw_bytes)
    return raw_bytes


__all__ = [
    "serialize_to_arrow",
    "serialize_to_jsonl",
    "serialize_to_parquet",
]
