# `multimodal.rs`, `multimodal_formats.rs`, `multimodal_serialize.rs` & `hf_hub.rs`: Universal Multimodal Engine & HuggingFace Hub Security

[`data/src/serialize.rs`](../../../data/src/serialize.rs) provide universal multi-column Arrow/Parquet/JSONL/CSV/IPC/WebDataset streaming, transparent decompression (`.gz`, `.zst`, `.lz4`), GIL-free binary serialization (`Parquet`, `JSONL`, `JSON`), and hardened HuggingFace Hub repository management.

---

## 1. Unified Multimodal Type System (`MultimodalValue`)

`multimodal.rs` maps Apache Arrow physical column arrays and universal tabular records into a unified 8-variant enum:

| Variant | Arrow Physical Types Read | Parquet Serialization Target (`hf_image_struct = true`) |
| :--- | :--- | :--- |
| `MultimodalValue::Null` | Null bitmap across any array | Nullable slot in target column array |
| `MultimodalValue::Text(String)` | `StringArray`, `LargeStringArray`, `StringViewArray` | `DataType::Utf8` (`StringArray`) |
| `MultimodalValue::Bytes(Vec<u8>)` | `BinaryArray`, `LargeBinaryArray`, `BinaryViewArray`, `FixedSizeBinaryArray`, `StructArray{bytes, path}` | `DataType::Struct(bytes: Binary, path: Utf8)` (HF Dataset Viewer Image/Audio compatible) |
| `MultimodalValue::Json(String)` | JSON-formatted `Utf8` strings, `MapArray`, or arbitrary `StructArray` | `DataType::Utf8` (or structured JSON object in `.jsonl`/`.json`) |
| `MultimodalValue::Int(i64)` | `Int64..Int8`, `UInt64..UInt8`, `Date32`, `Date64`, `Timestamp*` | `DataType::Int64` (`Int64Array`) |
| `MultimodalValue::Float(f64)` | `Float64Array`, `Float32Array`, `Float16Array` (via `half::f16`) | `DataType::Float64` (`Float64Array`) |
| `MultimodalValue::Bool(bool)` | `BooleanArray` | `DataType::Boolean` (`BooleanArray`) |
| `MultimodalValue::List(Vec<MultimodalValue>)` | `ListArray`, `LargeListArray`, `FixedSizeListArray` | Recursive nested elements or JSON-encoded array |

---

## 2. HuggingFace Dataset Viewer Image Struct Compatibility

When `hf_image_struct = true` (default), binary columns containing image or audio bytes (`MultimodalValue::Bytes`) are automatically serialized into Parquet (`multimodal_serialize.rs`) as an Arrow `StructArray` with schema:

$$\text{Struct}\langle\text{bytes}: \text{Binary},\ \text{path}: \text{Utf8}\rangle$$

This matches the exact schema expected by the HuggingFace Dataset Viewer and `datasets.Image()`, allowing uploaded Parquet shards to render inline image previews on `huggingface.co` without post-upload conversion.

---

## 3. Credential Redaction & HTTPS Transport Guard (`src/hf_hub.rs`)

### 3.1 Secret Resolution Hierarchy
`HfCredentials::new` resolves credentials from explicit arguments or environment variables (`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` / `~/.cache/huggingface/token`, `HF_USERNAME`, `HF_ORG`, `HF_ENDPOINT`), while the Python layer (`resolve_hf_secrets`) extends this with automatic lookup in **Google Colab `userdata`** and **Kaggle `UserSecretsClient`**.

### 3.2 Redacted `Debug` & Plaintext HTTP Refusal
1. **Redacted Formatting**: Custom `fmt::Debug` implementation replaces any present token with `Some("[REDACTED]")` so credentials can never leak into stack traces or logs.
2. **HTTPS Enforcement (`validate_url_security`)**:
   ```rust
   pub fn validate_url_security(&self, url: &str) -> Result<()> {
       if self.token.is_none() || url.starts_with(HTTPS_SCHEME_PREFIX) {
           return Ok(());
       }
       if LOOPBACK_HTTP_PREFIXES
           .iter()
           .any(|prefix| url.starts_with(prefix))
       {
           return Ok(());
       }
       Err(DataLoaderError::InvalidConfig(
           "Refusing to transmit HF_TOKEN over unencrypted non-loopback HTTP connection".to_owned(),
       ))
   }
   ```
3. **Path Traversal Guard**: `HfHubClient::upload_bytes` rejects any `path_in_repo` that is empty or contains `".."`.
