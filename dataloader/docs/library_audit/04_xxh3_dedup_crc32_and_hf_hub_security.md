# 4. `xxhash-rust`, `hashbrown`, `crc32fast`, `parking_lot`, and HuggingFace Hub API Specifications

## 4.1 Latest Stable Versions
- **`xxhash-rust`**: `0.8.19` (`features = ["xxh3"]`)
- **`hashbrown`**: `0.17.1`
- **`crc32fast`**: `1.5.2`
- **`parking_lot`**: `0.12.5`
- **External HuggingFace Hub Protocol**: `huggingface_hub >= 0.28 / 1.x` (HTTP Commit NDJSON API & Dataset Viewer Parquet Spec)

---

## 4.2 Version Range Changelog (`2024` - `2026`)

### A. `hashbrown` (`0.14.0` - `0.17.1`)
- **What changed**: Starting in `hashbrown 0.15.0` through `0.17.1`, the internal default hasher for `HashMap` and `HashSet` switched from `ahash` to `foldhash`.
- **SwissTable technical implications**: `hashbrown` uses the SwissTable architecture, splitting a 64-bit hash into two parts: the lower 57 bits (`h1`) for bucket slot selection and the upper 7 bits (`h2`, bits 57..63) as the control byte for SIMD group probing. When table keys are already uniformly distributed 64-bit hashes (`u64`), running `foldhash` on top of them performs redundant computation.

### B. `xxhash-rust` (`0.8.10` - `0.8.19`)
- **What changed**: Enhanced `const fn` compilation optimizations and auto-vectorization in `xxh3_64(input: &[u8]) -> u64`.
- **Bit distribution properties**: `xxh3_64` provides full avalanche across all 64 bits (including the upper 7 `h2` bits), making it safe to pair directly with SwissTable via an identity hasher without re-hashing.

### C. `crc32fast` (`1.4.0` - `1.5.2`)
- **What changed**: The `1.5.x` series enables `vpclmulqdq` hardware instructions (Vector Carry-Less Multiplication on AVX-512) alongside classic `pclmulqdq` to accelerate IEEE CRC32 checksum calculation.

### D. `parking_lot` (`0.12.1` - `0.12.5`)
- **What changed**: Maintenance updates for Rust Edition 2024 lint compatibility across `Mutex<T>` and `Condvar` synchronization primitives with zero OS mutex allocation on the uncontended fast path.

### E. HuggingFace Hub Credential & External Endpoint Conventions (`huggingface_hub 0.25` - `1.x`)
- **Standard environment variables**: The official `huggingface_hub` library standardized token resolution priority on `HF_TOKEN` (superseding legacy `HUGGING_FACE_HUB_TOKEN` as primary), local token files at `$HF_HOME/token` or `~/.cache/huggingface/token`, and `HF_ENDPOINT` (defaulting to `https://huggingface.co`).
- **`application/x-ndjson` Commit Protocol**: The `POST /api/datasets/{owner}/{repo}/commit/{revision}` endpoint accepts Newline-Delimited JSON streams beginning with `{"key": "header", "value": {"summary": ...}}` followed by `{"key": "file", "value": {"content": "<base64>", "path": ..., "encoding": "base64"}}`.
- **Dataset Viewer Image Schema Convention**: For image columns in Parquet files to render automatically in the HuggingFace Dataset Viewer without manual configuration, image columns must be stored as an Arrow `Struct` with two fields: `bytes: Binary` and `path: Utf8`.

---

## 4.3 Adaptations in LuminaMoon Source Code

1. **Bypassing `foldhash` with `IdentityBuildHasher` + `xxh3_64` (`dataloader/src/dedup.rs`)**:
   - Defines `IdentityHasher` (`write_u64` stores the `xxh3_64` hash directly) for `HashSet<u64, IdentityBuildHasher>` and `HashMap<u64, u32, IdentityBuildHasher>`, eliminating `foldhash` double-hashing overhead in `hashbrown 0.17.1`.
2. **Checkpoint Slot Validation via `crc32fast::Hasher` (`dataloader/src/metadata.rs`)**:
   - Uses `crc32fast` to verify the integrity of both 64-byte Ping-Pong checkpoint slots (`Slot A` and `Slot B`).
3. **Fine-Grained `parking_lot::Mutex` Separation (`dataloader/src/preram.rs`)**:
   - Separates `Mutex<QueueState>` and `Condvar` from `Mutex<DualMetadataManager>` so background Pre-RAM refill workers and `commit_gpu_batch()` callers never block in-memory batch pops.
4. **HuggingFace Hub Protocol Compliance (`dataloader/src/hf_hub.rs` & `dataloader/src/multimodal_serialize.rs`)**:
   - `HfCredentials` resolves `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, local token files, `HF_ORG`, `HF_USERNAME`, and `HF_ENDPOINT`, redacts token values as `"[REDACTED]"` in `fmt::Debug`, and blocks token transmission over non-loopback `http://` connections.
   - `serialize_columns_to_parquet_bytes_with_options` encodes `MultimodalValue::Bytes` columns into `{bytes: Binary, path: Utf8}` `StructArray` layouts directly compatible with the HuggingFace Dataset Viewer.
5. **Colab/Kaggle Credential Auto-Resolver, RFC 5988 `Link` Pagination, and Hybrid LFS/Xet Upload (`python/luminamoon/dataloader/hub.py`)**:
   - `resolve_hf_secrets` automatically resolves `HF_TOKEN`, `HF_ORG`, and `HF_USERNAME` from environment variables, `google.colab.userdata`, or `kaggle_secrets.UserSecretsClient`.
   - `expand_hf_sources` supports automatic shard discovery from repository roots `hf://datasets/org/repo[@revision]` via `huggingface_hub.HfApi.list_repo_files` or the REST `/api/datasets/.../tree/...?recursive=true` endpoint with RFC 5988 `Link: <...>; rel="next"` header pagination and per-page HTTPS validation.
   - `push_to_hub` combines GIL-free Rust serialization (`serialize_table_to_bytes`) with `huggingface_hub.HfApi.upload_file` (supporting Git LFS / Xet multipart uploads for large shards) and automatic fallback to the native Rust `HfHubUploader` (`application/x-ndjson`).
