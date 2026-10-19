# `multimodal.rs` & `hf_hub.rs`: Mesin Multimodal Parquet/JSONL/JSON & Keamanan HuggingFace Hub

[`dataloader/src/multimodal.rs`](../../../dataloader/src/multimodal.rs) dan [`dataloader/src/hf_hub.rs`](../../../dataloader/src/hf_hub.rs) menyediakan streaming Arrow/Parquet multi-kolom, serialisasi biner bebas-GIL (`Parquet`, `JSONL`, `JSON`), serta manajemen repositori HuggingFace Hub berstandar keamanan tinggi.

[English Version (README.md)](README.md)

---

## 1. Sistem Tipe Multimodal Terpadu (`MultimodalValue`)

`multimodal.rs` memetakan array kolom fisik Apache Arrow ke dalam enum terpadu 7-varian:

| Varian | Tipe Fisik Arrow yang Dibaca | Target Serialisasi Parquet (`hf_image_struct = true`) |
| :--- | :--- | :--- |
| `MultimodalValue::Null` | Bitmap null pada array mana pun | Slot bernilai null pada array kolom target |
| `MultimodalValue::Text(String)` | `StringArray`, `LargeStringArray`, `StringViewArray` | `DataType::Utf8` (`StringArray`) |
| `MultimodalValue::Bytes(Vec<u8>)` | `BinaryArray`, `LargeBinaryArray`, `BinaryViewArray`, `StructArray{bytes, path}` | `DataType::Struct(bytes: Binary, path: Utf8)` (Kompatibel penuh dengan HF Dataset Viewer) |
| `MultimodalValue::Json(String)` | String `Utf8` berformat JSON atau `StructArray` umum | `DataType::Utf8` (atau objek JSON terstruktur pada `.jsonl`/`.json`) |
| `MultimodalValue::Int(i64)` | `Int64Array`, `Int32Array` | `DataType::Int64` (`Int64Array`) |
| `MultimodalValue::Float(f64)` | `Float64Array`, `Float32Array` | `DataType::Float64` (`Float64Array`) |
| `MultimodalValue::Bool(bool)` | `BooleanArray` | `DataType::Boolean` (`BooleanArray`) |

---

## 2. Kompatibilitas Struct Gambar HuggingFace Dataset Viewer

Ketika `hf_image_struct = true` (aktif secara bawaan), kolom biner yang memuat byte gambar atau audio (`MultimodalValue::Bytes`) secara otomatis diserialisasikan ke dalam berkas Parquet sebagai `StructArray` Arrow dengan skema:

$$\text{Struct}\langle\text{bytes}: \text{Binary},\ \text{path}: \text{Utf8}\rangle$$

Skema ini identik dengan spesifikasi resmi HuggingFace Dataset Viewer dan `datasets.Image()`, sehingga berkas shard Parquet yang diunggah dapat langsung menampilkan pratinjau visual di halaman `huggingface.co` tanpa konversi tambahan.

---

## 3. Redaksi Kredensial & Proteksi Transportasi HTTPS (`src/hf_hub.rs`)

### 3.1 Hierarki Resolusi Kredensial Rahasia
`HfCredentials::new` membaca kredensial dari parameter eksplisit atau variabel lingkungan (`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` / `~/.cache/huggingface/token`, `HF_USERNAME`, `HF_ORG`, `HF_ENDPOINT`), sedangkan lapisan Python (`resolve_hf_secrets`) memperluas dukungan ini dengan pembacaan otomatis dari **Google Colab `userdata`** dan **Kaggle `UserSecretsClient`**.

### 3.2 Redaksi `Debug` & Penolakan HTTP Teks Biasa
1. **Format Debug Tersensor**: Implementasi khusus `fmt::Debug` menyensor isi token menjadi `Some("[REDACTED]")` agar token rahasia tidak pernah bocor ke dalam log atau *stack trace*.
2. **Penegakan HTTPS (`validate_url_security`)**:
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
3. **Proteksi Path Traversal**: `HfHubClient::upload_bytes` menolak nilai `path_in_repo` yang kosong atau mengandung `".."`.
