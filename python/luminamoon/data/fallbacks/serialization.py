from __future__ import annotations

from collections.abc import Mapping
import json
import math
from typing import Any, Optional, Sequence


def _normalize_columns(columns: Any) -> list[tuple[str, list[Any]]]:
    if isinstance(columns, Mapping):
        pairs = [(str(k), list(v)) for k, v in columns.items()]
    else:
        pairs = [(str(k), list(v)) for k, v in columns]
    if not pairs:
        raise ValueError("Cannot serialize empty column list")
    num_rows = len(pairs[0][1])
    for name, vals in pairs:
        if len(vals) != num_rows:
            raise ValueError(f"Column '{name}' length mismatch: expected {num_rows}, got {len(vals)}")
    return pairs


def _build_pyarrow_table(pairs: list[tuple[str, list[Any]]], hf_image_struct: bool) -> Any:
    import pyarrow as pa

    arrays: list[Any] = []
    names: list[str] = []
    for name, vals in pairs:
        has_bytes = any(isinstance(v, (bytes, bytearray)) for v in vals)
        if has_bytes and hf_image_struct:
            b_col = [bytes(v) if isinstance(v, (bytes, bytearray)) else None for v in vals]
            p_col = [v if isinstance(v, str) else None for v in vals]
            arr = pa.StructArray.from_arrays(
                [pa.array(b_col, type=pa.binary()), pa.array(p_col, type=pa.string())],
                names=["bytes", "path"],
            )
        elif has_bytes:
            arr = pa.array([bytes(v) if isinstance(v, (bytes, bytearray)) else None for v in vals], type=pa.binary())
        else:
            arr = pa.array(vals)
        arrays.append(arr)
        names.append(name)
    return pa.Table.from_arrays(arrays, names=names)


def serialize_table_to_bytes(
    columns: Any,
    format: str = "parquet",
    hf_image_struct: bool = True,
) -> bytes:
    pairs = _normalize_columns(columns)
    fmt = format.strip().lower()
    if fmt in ("jsonl", "ndjson") or fmt.endswith(".jsonl"):
        def _json_safe(v: Any) -> Any:
            if isinstance(v, (bytes, bytearray)):
                return "0x" + bytes(v).hex()
            if isinstance(v, float) and not math.isfinite(v):
                return None
            return v

        num_rows = len(pairs[0][1])
        lines: list[str] = []
        for idx in range(num_rows):
            row = {name: _json_safe(vals[idx]) for name, vals in pairs}
            lines.append(json.dumps(row, separators=(",", ":"), ensure_ascii=False))
        return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")

    if fmt in ("parquet",) or fmt.endswith((".parquet", ".pq")):
        import io

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "PyArrow is required for pure-Python Parquet fallback serialization. "
                "Install `pyarrow` or use the native Rust extension."
            ) from exc

        table = _build_pyarrow_table(pairs, hf_image_struct=hf_image_struct)
        sink = io.BytesIO()
        pq.write_table(table, sink, compression="zstd")
        return sink.getvalue()

    if fmt in ("arrow", "feather", "ipc") or fmt.endswith(".arrow"):
        import io

        try:
            import pyarrow.ipc as ipc
        except ImportError as exc:
            raise ImportError(
                "PyArrow is required for pure-Python Arrow IPC fallback serialization. "
                "Install `pyarrow` or use the native Rust extension."
            ) from exc

        table = _build_pyarrow_table(pairs, hf_image_struct=hf_image_struct)
        sink = io.BytesIO()
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
        return sink.getvalue()

    raise ValueError(
        f"Unsupported serialization format '{format}'. Expected one of: 'parquet', 'arrow', 'feather', 'ipc', 'jsonl', 'ndjson'."
    )


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
