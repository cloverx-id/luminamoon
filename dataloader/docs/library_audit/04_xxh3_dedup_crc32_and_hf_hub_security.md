# 4. `xxhash-rust`, `hashbrown`, `crc32fast`, `parking_lot`, dan Spesifikasi API HuggingFace Hub

## 4.1 Versi Stabil Terbaru
- **`xxhash-rust`**: `0.8.19` (`features = ["xxh3"]`)
- **`hashbrown`**: `0.17.1`
- **`crc32fast`**: `1.5.2`
- **`parking_lot`**: `0.12.5`
- **Protokol Eksternal HuggingFace Hub**: `huggingface_hub >= 0.28 / 1.x` (HTTP Commit NDJSON API & Dataset Viewer Parquet Spec)

---

## 4.2 Riwayat Perubahan pada Rentang Versi (`2024` - `2026`)

### A. `hashbrown` (`0.14.0` - `0.17.1`)
- **Apa yang berubah**: Mulai `hashbrown 0.15.0` hingga `0.17.1`, *default hasher* internal untuk `HashMap` dan `HashSet` diganti dari `ahash` menjadi `foldhash`.
- **Implikasi Teknis SwissTable**: `hashbrown` menggunakan arsitektur SwissTable yang membagi nilai hash 64-bit menjadi dua bagian: 57 bit bawah (`h1`) untuk pemilihan slot bucket, dan 7 bit teratas (`h2`, bit 57..63) sebagai *control byte* untuk pencarian grup SIMD. Apabila kunci tabel sudah berupa nilai `u64` yang dihasilkan oleh algoritma hash 64-bit berdistribusi seragam, menjalankan `foldhash` di atasnya merupakan duplikasi komputasi.

### B. `xxhash-rust` (`0.8.10` - `0.8.19`)
- **Apa yang berubah**: Peningkatan optimasi kompilasi `const fn` dan vektorisasi otomatis pada `xxh3_64(input: &[u8]) -> u64`.
- **Properti Distribusi Bit**: `xxh3_64` menghasilkan *avalanche* penuh di seluruh 64 bit (termasuk 7 bit teratas `h2`), sehingga aman dipadukan langsung dengan SwissTable tanpa *re-hashing*.

### C. `crc32fast` (`1.4.0` - `1.5.2`)
- **Apa yang berubah**: Rilis `1.5.x` mengaktifkan instruksi perangkat keras `vpclmulqdq` (Vector Carry-Less Multiplication pada AVX-512) di samping `pclmulqdq` klasik untuk mempercepat perhitungan checksum CRC32 IEEE.

### D. `parking_lot` (`0.12.1` - `0.12.5`)
- **Apa yang berubah**: Pembaruan pemeliharaan untuk kompatibilitas lint Rust Edition 2024 pada primitif sinkronisasi `Mutex<T>` dan `Condvar` tanpa alokasi OS mutex di jalur tidak terkontensi (*uncontended fast-path*).

### E. Konvensi Kredensial & Endpoint Eksternal HuggingFace Hub (`huggingface_hub 0.25` - `1.x`)
- **Variabel Lingkungan Standar**: Pustaka resmi `huggingface_hub` menstandarkan urutan resolusi token pada `HF_TOKEN` (menggantikan `HUGGING_FACE_HUB_TOKEN` lama sebagai prioritas pertama), lokasi berkas token lokal di `$HF_HOME/token` atau `~/.cache/huggingface/token`, serta variabel `HF_ENDPOINT` (bawaan `https://huggingface.co`).
- **Protokol Unggah `application/x-ndjson`**: Endpoint `POST /api/datasets/{owner}/{repo}/commit/{revision}` menerima aliran *Newline-Delimited JSON* dengan baris `{"key": "header", "value": {"summary": ...}}` diikuti baris `{"key": "file", "value": {"content": "<base64>", "path": ..., "encoding": "base64"}}`.
- **Konvensi Skema Citra Dataset Viewer**: Agar kolom gambar pada berkas Parquet langsung dirender oleh HuggingFace Dataset Viewer tanpa konfigurasi manual, kolom citra harus disimpan sebagai Arrow `Struct` dengan dua field: `bytes: Binary` dan `path: Utf8`.

---

## 4.3 Adaptasi pada Kode Sumber LuminaMoon

1. **Bypass `foldhash` dengan `IdentityBuildHasher` + `xxh3_64` (`dataloader/src/dedup.rs`)**:
   - Mendefinisikan `IdentityHasher` (`write_u64` langsung menyimpan nilai hash `xxh3_64`) untuk `HashSet<u64, IdentityBuildHasher>` dan `HashMap<u64, u32, IdentityBuildHasher>`, menghilangkan overhead *double-hashing* `foldhash` pada `hashbrown 0.17.1`.
2. **Validasi Slot Checkpoint dengan `crc32fast::hash` (`dataloader/src/metadata.rs`)**:
   - Menggunakan `crc32fast::hash(&buf[..60])` untuk memverifikasi integritas kedua slot Ping-Pong 64-byte (`Slot A` dan `Slot B`) pada berkas checkpoint.
3. **Pemisahan Kunci `parking_lot::Mutex` (`dataloader/src/preram.rs`)**:
   - Memisahkan `Mutex<QueueState>` dan `Condvar` dari `Mutex<DualMetadataManager>` agar thread pengisi ulang Pre-RAM dan pemanggil `commit_gpu_batch()` tidak saling memblokir pengambilan batch di memori.
4. **Kepatuhan Protokol HuggingFace Hub (`dataloader/src/hf_hub.rs` & `dataloader/src/multimodal.rs`)**:
   - `HfCredentials` membaca `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, berkas token lokal, `HF_ORG`, `HF_USERNAME`, dan `HF_ENDPOINT`, menyensor nilai token menjadi `"[REDACTED]"` pada implementasi `fmt::Debug`, serta memblokir pengiriman token di atas protokol `http://` non-loopback.
   - `serialize_columns_to_parquet_bytes_with_options` mengenkode kolom `MultimodalValue::Bytes` ke dalam `StructArray` `{bytes: Binary, path: Utf8}` yang kompatibel langsung dengan HuggingFace Dataset Viewer.
5. **Auto-Resolver Kredensial Colab/Kaggle, Paginasi RFC 5988 `Link`, dan Hybrid LFS/Xet Upload (`python/luminamoon/dataloader/__init__.py`)**:
   - `resolve_hf_secrets` secara otomatis memuat `HF_TOKEN`, `HF_ORG`, dan `HF_USERNAME` dari `google.colab.userdata` atau `kaggle_secrets.UserSecretsClient` ke `os.environ` jika belum tersedia.
   - `expand_hf_sources` mendukung auto-discovery shard `.parquet`/`.jsonl` dari root repositori `hf://datasets/org/repo[@revision]` menggunakan `huggingface_hub.HfApi.list_repo_files` atau endpoint REST `/api/datasets/.../tree/...?recursive=true` lengkap dengan paginasi header RFC 5988 `Link: <...>; rel="next"` dan validasi HTTPS per halaman.
   - `push_to_hub` mengombinasikan serialisasi Rust bebas-GIL (`serialize_table_to_bytes`) dengan `huggingface_hub.HfApi.upload_file` (untuk dukungan Git LFS / Xet multipart pada berkas besar) serta fallback otomatis ke mesin Rust `HfHubUploader` (`application/x-ndjson`).

