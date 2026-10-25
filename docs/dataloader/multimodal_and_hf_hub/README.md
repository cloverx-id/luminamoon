# `serialize.rs` & `serialization.py`: GIL-Free Columnar Serialization (Parquet, Arrow IPC & JSONL)

[`data/src/serialize.rs`](https://github.com/cloverx-id/luminamoon/blob/main/data/src/serialize.rs) and [`python/luminamoon/data/serialization.py`](https://github.com/cloverx-id/luminamoon/blob/main/python/luminamoon/data/serialization.py) provide GIL-free binary serialization of Python dictionaries and columnar tables into Apache Parquet (with ZSTD compression), Apache Arrow IPC (`ARROW1`), and line-delimited JSONL (`NDJSON`).

---

## 1. Unified Columnar Type System (`DataValue`)

`data/src/serialize.rs` maps Python sequences and dictionaries into an 8-variant `DataValue` enum:

| Variant | Python Input Types | Arrow / Parquet Target Type (`hf_image_struct = true`) |
| :--- | :--- | :--- |
| `DataValue::Null` | `None` | Nullable slot in target column array |
| `DataValue::Text(String)` | `str` (non-JSON) | `DataType::Utf8` (`StringArray`) |
| `DataValue::Bytes(Vec<u8>)` | `bytes`, `bytearray` | `DataType::Struct(bytes: Binary, path: Utf8)` (or `DataType::Binary` when `hf_image_struct = false`) |
| `DataValue::Json(String)` | Valid JSON object/array `str` | `DataType::Utf8` in Parquet/Arrow; parsed JSON value in `.jsonl` |
| `DataValue::Int(i64)` | `int` | `DataType::Int64` (`Int64Array`) |
| `DataValue::Float(f64)` | `float` | `DataType::Float64` (`Float64Array`) |
| `DataValue::Bool(bool)` | `bool` | `DataType::Boolean` (`BooleanArray`) |
| `DataValue::List(Vec<DataValue>)` | Nested sequences | JSON-encoded array representation |

---

## 2. HuggingFace Dataset Viewer Image Struct Compatibility

When `hf_image_struct = True` (default in `serialize_to_parquet`), binary columns (`DataValue::Bytes`) are automatically encoded as an Arrow `StructArray` with schema:

$$\text{Struct}\langle\text{bytes}: \text{Binary},\ \text{path}: \text{Utf8}\rangle$$

This matches the canonical HuggingFace Dataset Viewer image/audio schema, enabling instant inline previews on `huggingface.co` without post-processing.

---

## 3. Public Python Serialization Helpers (`serialization.py`)

- **`serialize_to_parquet(columns, output_path=None, hf_image_struct=True) -> bytes`**: Encodes columnar data into ZSTD-compressed Apache Parquet bytes (`PAR1`) with the GIL released (`py.detach()`).
- **`serialize_to_arrow(columns, output_path=None) -> bytes`**: Encodes columnar data into Apache Arrow IPC File Format bytes (`ARROW1`).
- **`serialize_to_jsonl(columns, output_path=None) -> bytes`**: Encodes columnar data into UTF-8 line-delimited JSONL records.
