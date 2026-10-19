# `multimodal.rs` & `hf_hub.rs`: Multimodal Parquet/JSONL/JSON Engine & HuggingFace Hub Security

[`dataloader/src/multimodal.rs`](../../../dataloader/src/multimodal.rs) and [`dataloader/src/hf_hub.rs`](../../../dataloader/src/hf_hub.rs) provide multi-column Arrow/Parquet streaming, GIL-free binary serialization (`Parquet`, `JSONL`, `JSON`), and hardened HuggingFace Hub repository management.

---

## 1. Unified Multimodal Type System (`MultimodalValue`)

`multimodal.rs` maps Apache Arrow physical column arrays into a unified 7-variant enum:

| Variant | Arrow Physical Types Read | Parquet Serialization Target (`hf_image_struct = true`) |
| :--- | :--- | :--- |
| `MultimodalValue::Null` | Null bitmap across any array | Nullable slot in target column array |
| `MultimodalValue::Text(String)` | `StringArray`, `LargeStringArray`, `StringViewArray` | `DataType::Utf8` (`StringArray`) |
| `MultimodalValue::Bytes(Vec<u8>)` | `BinaryArray`, `LargeBinaryArray`, `BinaryViewArray`, `StructArray{bytes, path}` | `DataType::Struct(bytes: Binary, path: Utf8)` (HF Dataset Viewer Image/Audio compatible) |
| `MultimodalValue::Json(String)` | JSON-formatted `Utf8` strings or arbitrary `StructArray` | `DataType::Utf8` (or structured JSON object in `.jsonl`/`.json`) |
| `MultimodalValue::Int(i64)` | `Int64Array`, `Int32Array` | `DataType::Int64` (`Int64Array`) |
| `MultimodalValue::Float(f64)` | `Float64Array`, `Float32Array` | `DataType::Float64` (`Float64Array`) |
| `MultimodalValue::Bool(bool)` | `BooleanArray` | `DataType::Boolean` (`BooleanArray`) |

---

## 2. HuggingFace Dataset Viewer Image Struct Compatibility

When `hf_image_struct = true` (default), binary columns containing image or audio bytes (`MultimodalValue::Bytes`) are automatically serialized into Parquet as an Arrow `StructArray` with schema:

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
       if self.token.is_none() || url.starts_with("https://") {
           return Ok(());
       }
       if url.starts_with("http://127.0.0.1")
           || url.starts_with("http://localhost")
           || url.starts_with("http://[::1]")
       {
           return Ok(());
       }
       Err(DataLoaderError::InvalidConfig(
           "Refusing to transmit HF_TOKEN over unencrypted non-loopback HTTP connection".to_owned(),
       ))
   }
   ```
3. **Path Traversal Guard**: `HfHubClient::upload_bytes` rejects any `path_in_repo` that is empty or contains `".."`.
