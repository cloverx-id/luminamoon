# 5. Universal Formats (`arrow-ipc`, `half`), Stream Decompression (`flate2`, `zstd`, `lz4_flex`), and Typo Diagnostics

## 5.1 Latest Stable Versions (October 2026)
- **`arrow-ipc`**: `60.0.0` (synchronized with `arrow-array = "60.0.0"`, `arrow-schema = "60.0.0"`, `parquet = "60.0.0"`)
- **`half`**: `2.7.1` (`f16` / `bf16` IEEE 754-2008 half-precision floating point)
- **`flate2`**: `1.1.10` (`features = ["rust_backend"]` / `miniz_oxide 0.8.9` + `crc32fast 1.5.2`)
- **`zstd`**: `0.14.3` (`zstd-safe 8.2.0` + `zstd-sys 2.0.16+zstd.1.5.7`)
- **`lz4_flex`**: `0.14.0` (`safe-decode` & `frame` enabled by default, 100% `unsafe`-free decompression path)

---

## 5.2 Detailed Version-by-Version Changelog over a 1-Year Window (`2025` - `October 2026`)

### A. `arrow-ipc` (`53.0.0` - `60.0.0`)
| Version | Date / Era | API, ABI, and Internal Behavior Changes |
| :--- | :--- | :--- |
| **`53.0.0` - `54.0.0`** | Q4 2024 - Q1 2025 | Added full IPC serialization and deserialization support (`FileReader`, `StreamReader`, `FileWriter`, `StreamWriter`) for high-performance variable-length view types `Utf8View` (`StringViewArray`) and `BinaryView` (`BinaryViewArray`) with dedicated variadic buffers. |
| **`55.0.0` - `56.0.0`** | Q2 2025 | Strict 8-byte / 64-byte buffer alignment validation on IPC `RecordBatch` messages to prevent unaligned pointer reads when memory-mapping `.arrow` / `.feather` files (`mmap` or `Cursor<&[u8]>`). |
| **`57.0.0`** | Q3 2025 | Improved IPC body compression (`LZ4_FRAME` and `ZSTD`) and variadic buffer bounds checking on `BinaryView` to prevent out-of-bounds slice panics on corrupted IPC streams. |
| **`58.0.0`** | Q4 2025 | Deprecated `StreamReader::try_new_unbuffered`; unified internal buffering in `StreamReader::try_new` and raised MSRV to match modern Rust toolchains. |
| **`59.0.0` - `60.0.0`** | 2026 | Removed legacy methods deprecated in `57.x`/`58.x`, aligned `Field` and `Schema` structures with `arrow-schema 60.0.0`, and enabled zero-copy slicing across `ListArray`, `LargeListArray`, `FixedSizeListArray`, and `Float16Array`. |

### B. `lz4_flex` (`0.11.0` - `0.14.0`) & Mitigation of `CVE-2026-32829` (`RUSTSEC-2026-0041`)
| Version | Release Date | API Changes, Security Features, and CVE Mitigations |
| :--- | :--- | :--- |
| **`0.11.0` - `0.11.5`** | 2024 - 2025 | Inverted the feature flag from `checked-decode` to `unchecked-decode`. Previously, additive Cargo feature unification meant a transitive dependency disabling `checked-decode` could silently enable `unsafe` pointer decoding. Inverting the flag to `unchecked-decode` (off by default) with `safe-decode` (on by default) guarantees 100% `unsafe`-free decompression unless explicitly opted in. |
| **`0.11.6` & `0.12.0` - `0.12.1`** | **2026-05-09** | **Critical Security Mitigation `CVE-2026-32829` / `RUSTSEC-2026-0041` (CVSS 7.5 High)**: In `lz4_flex < 0.11.6` and `0.12.0`, when using APIs that write into uninitialized output buffers (`decompress_into` / `Sink`) on malformed LZ4 inputs containing an invalid match offset beyond already-written bytes, the decompressor could copy uninitialized memory bytes into the output buffer before returning an error (*information leak*). Fixed in `0.11.6` and `0.12.1` by validating match offset bounds against the actual output cursor prior to copying. |
| **`0.13.0`** | **2026-05-09** | Added external dictionary compression/decompression support (`compress_prepend_size_with_dict`, `decompress_size_prepended_with_dict`) for small LZ4 blocks. |
| **`0.13.1`** | **2026-05-09** | Fixed an out-of-bounds panic when an external dictionary smaller than 4 bytes (`dict.len() < 4`) was passed to dictionary compression functions. |
| **`0.14.0`** | **2026-07-14** | Explicitly separated the `alloc` feature flag from `no_std` paths, refined `frame::FrameDecoder` (`std::io::Read`) for framed `.lz4` streams (*LZ4 Frame Format*), and optimized `safe-decode` throughput. |

### C. `flate2` (`1.0.30` - `1.1.10`)
| Version | Release Date | Backend, Performance, and Feature Flag Changes |
| :--- | :--- | :--- |
| **`1.0.30` - `1.0.35`** | 2024 - Early 2025 | Upgraded `miniz_oxide` to the `0.8.x` series and introduced optional `zlib-rs` backend support (pure-Rust SIMD-accelerated Zlib). |
| **`1.1.0` - `1.1.4`** | Mid 2025 | Added `.reset()` and internal buffer reuse across decoders (`GzDecoder`, `ZlibDecoder`, `DeflateDecoder`), eliminating repeated heap allocations when decompressing thousands of small `.gz` shards. |
| **`1.1.5` - `1.1.8`** | Late 2025 - Early 2026 | Optimized `MultiGzDecoder` for concatenated `.jsonl.gz` streams and improved handling of trailing bytes after Gzip CRC32 footers. |
| **`1.1.9`** | Q2 2026 | Refactored internal `zlib-rs` and `crc32fast` feature flags, which temporarily disabled runtime CPU SIMD detection (`std`) when consumers customized default features. |
| **`1.1.10`** | **Q3 2026** | **Throughput Regression Fix (`runtime_detection`)**: Added the `runtime_detection` feature flag (`zlib-rs?/std`, `crc32fast?/std`) directly into `default` features, ensuring `crc32fast` (`vpclmulqdq` / `pclmulqdq` / `asimd`) and compression backends automatically detect CPU SIMD capabilities at runtime. |

### D. `zstd` (`0.13.0` - `0.14.3`) & `zstd-safe` (`7.2.0` - `8.2.0`)
| Version | Release Date | Memory Safety (`Soundness`) and C `libzstd` Engine Changes |
| :--- | :--- | :--- |
| **`0.13.0` - `0.13.3`** | 2024 - 2025 | Upgraded `zstd-sys` bindings to `libzstd 1.5.6` and `1.5.7` (faster large-frame decompression and rare long-distance matcher bug fixes). |
| **`0.14.0`** | **2026** | **`zstd-safe 8.0.0+` Soundness Fix (`CCtx` / `DCtx`)**: Resolved a memory-safety soundness bug where streaming wrappers allowed direct `&mut CCtx` / `&mut DCtx` borrows (`DerefMut`), permitting callers to `std::mem::swap` contexts while holding raw pointers to short-lived dictionary buffers (*lifetime violation*). In `0.14.x`, direct mutable context access is restricted in favor of safe delegation methods on `Encoder` and `Decoder`. |
| **`0.14.1` - `0.14.3`** | **2026** | Updated dependencies to `zstd-safe ^8.2.0` (`zstd 1.5.7`), refined `zstd::stream::read::Decoder` and `zstd::stream::encode_all` for multi-frame `.zst` streams (including HuggingFace Xet/Parquet skippable metadata frames), and eliminated compiler warnings on Rust `1.99.1`. |

### E. `half` (`2.4.0` - `2.7.1`)
| Version | Release Date | `f16` / `bf16` Conversion and Hardware Intrinsic Changes |
| :--- | :--- | :--- |
| **`2.4.0` - `2.5.0`** | 2024 - 2025 | Stabilized batch slice conversions (`f16::to_f32_slice`, `bf16::to_f32_slice`) and full `bytemuck` and `serde` integration. |
| **`2.6.0`** | 2025 | Raised MSRV to Rust `1.81+`, utilizing `f16c` / `avx512fp16` intrinsics on `x86_64` and `fp16` on `aarch64` on stable compilers without nightly feature flags. |
| **`2.7.0` - `2.7.1`** | **2026** | Full alignment with `arrow-array 60.0.0` (`Float16Array` exports `half::f16` elements), branchless `to_f64()` optimization for subnormal and `NaN` values, and full compliance with Rust Edition 2024 `unsafe_op_in_unsafe_fn` rules. |

### F. Self-Contained WebDataset `.tar` Parser & Typo Diagnostics (Zero External Crate Bloat)
- **Why External `tar` Crates Were Avoided (`CVE-2025-62518` / TARmageddon)**:
  In late 2025, a critical vulnerability (**`CVE-2025-62518` / TARmageddon**) was disclosed across async TAR parsers in the Rust ecosystem (`tokio-tar` / `astral-tokio-tar`) caused by desynchronization between PAX extended headers and GNU longname headers, enabling file smuggling and path traversal.
- **LuminaMoon's Self-Contained Solution (`parse_webdataset_tar_bytes` in `data/src/multimodal_formats.rs`)**:
  Because WebDataset ingestion only groups paired in-memory payloads (e.g., `sample_001.jpg`, `sample_001.txt`, `sample_001.json`) into tabular rows without ever extracting files to disk, LuminaMoon implements a pure in-memory 512-byte USTAR block parser that strictly rejects directory/symlink entries and deterministically groups sample stems.

---

## 5.3 Adaptations in LuminaMoon Source Code

1. **Universal Format Support (`data/src/multimodal_formats.rs`, `data/src/multimodal_serialize.rs`, `data/src/preram.rs`, `data/src/hf_hub.rs`)**:
   - `InMemoryMultimodalTable::from_bytes_auto` automatically detects and parses:
     * **Apache Arrow IPC / Feather v2** (`.arrow`, `.feather`, `.ipc`, or `ARROW1` magic bytes) via `arrow_ipc::reader::{FileReader, StreamReader}`.
     * **WebDataset TAR** (`.tar`, `.tgz`, `.tar.gz`, `.tar.zst`, `.tar.lz4`), grouping sample files by stem (`00001.png` + `00001.txt` + `00001.json` $\rightarrow$ 1 multimodal row).
     * **CSV & TSV** (`.csv`, `.tsv`) with an RFC 4180 state-machine parser (escaped quotes, automatic `Int`/`Float`/`Bool`/`Json`/`Text` type inference, and `min_score` filtering).
     * **JSON Array / Object Wrapper & JSONL** (`.json`, `.jsonl`, `.ndjson`).
   - `serialize_columns_by_path_format` (`data/src/hf_hub.rs`) serializes tables into `.parquet`, `.arrow`/`.feather`/`.ipc`, `.jsonl`/`.ndjson`, `.json`, `.csv`, and `.tsv`, transparently wrapping output in `.gz` (`flate2::write::GzEncoder`) or `.zst` (`zstd::stream::encode_all`) compression based on `path_in_repo`.

2. **Transparent Local & Remote Stream Decompression (`data/src/source_local.rs` & `data/src/source_hf.rs`)**:
   - `decompress_bytes_if_compressed` detects compression via both file extensions (`.gz`, `.zst`/`.zstd`, `.lz4`) and header magic bytes:
     * **Gzip (`[0x1F, 0x8B]`)**: `flate2::read::MultiGzDecoder` (`1.1.10`).
     * **Zstandard (`[0x28, 0xB5, 0x2F, 0xFD]`)**: `zstd::stream::decode_all` (`0.14.3`).
     * **LZ4 Frame (`[0x04, 0x22, 0x4D, 0x18]`)**: `lz4_flex::frame::FrameDecoder` (`0.14.0`, protected against `CVE-2026-32829`).

3. **Full Arrow Type Extraction & Nested JSON Paths (`data/src/multimodal.rs` & `data/src/postprocess.rs`)**:
   - `MultimodalValue::List(Vec<MultimodalValue>)` recursively extracts `ListArray`, `LargeListArray`, and `FixedSizeListArray` (supporting multi-turn chat `messages` and embedding vectors).
   - `Float16Array` values are converted losslessly via `half::f16::to_f64()`.
   - `extract_json_value_by_path` supports nested dot-paths and array indices (e.g., `json_extract_key="messages.0.content"` or `"meta.quality.score"`).

4. **Typo Diagnostic Engine & Developer-Friendly Errors (`data/src/error.rs` & `python/luminamoon/data/diagnostics.py`)**:
   - **Damerau-Levenshtein Optimal String Alignment (`damerau_levenshtein`, `find_closest_matches`)**: Computes edit distance including adjacent character transpositions (e.g., `"txet"` $\rightarrow$ `"text"`, `"prmopt"` $\rightarrow$ `"prompt"`).
   - **Fake File / Git LFS Pointer Detection (`inspect_magic_bytes_hint`)**: Detects when a `.parquet`/`.arrow` file that fails to open is actually an unfetched Git LFS text pointer (`version https://git-lfs.github.com/spec/v1`), an HTML error page (`<!DOCTYPE html>`), or a misnamed JSON/Gzip/Zstd file.
   - **Sibling Filename & URI Suggestions (`open_local_file_with_diagnostics`, `diagnose_uri_typo`)**: Suggests similar filenames in the same directory on `FileNotFound`, corrects URI scheme typos (`hf://dataset/...` $\rightarrow$ `hf://datasets/...`, `httpss://`), and validates unknown Python `**kwargs` with closest-parameter suggestions.
