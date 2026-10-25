from __future__ import annotations

import json
from typing import Any, Optional, Sequence

def serialize_table_to_bytes(
    columns: Any,
    format: str = "parquet",
    hf_image_struct: bool = True,
) -> bytes:
    if isinstance(columns, dict):
        rows = [json.dumps(columns, ensure_ascii=False)]
    else:
        rows = [str(item) for item in columns]
    return "\n".join(rows).encode("utf-8")


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

__all__ = [
    "serialize_table_to_bytes",
    "serialize_to_arrow_ipc",
    "serialize_to_csv",
    "serialize_to_jsonl",
    "serialize_to_parquet",
]
