# 6. Universal DataLoader Libraries (`arrow-ipc`, `half`, `flate2`, `zstd`, `lz4_flex`)

## 6.1 Latest Stable Versions (October 2026)
- **`arrow-ipc`**: `60.0.0` (synchronized with `arrow-array = "60.0.0"`, `arrow-schema = "60.0.0"`, `parquet = "60.0.0"`)
- **`half`**: `2.7.1` (`f16` / `bf16` IEEE 754-2008 half-precision floating point)
- **`flate2`**: `1.1.10` (`features = ["rust_backend"]` / `miniz_oxide 0.8.9` + `crc32fast 1.5.2`)
- **`zstd`**: `0.14.3` (`zstd-safe 8.2.0` + `zstd-sys 2.0.16+zstd.1.5.7`)
- **`lz4_flex`**: `0.14.0` (`safe-decode` & `frame` enabled by default)

---

## 6.2 Detailed Version-by-Version Changelog over a 1-Year Window (`2025` - `October 2026`)

### A. `arrow-ipc` (`53.0.0` - `60.0.0`)
- **`53.0.0` - `54.0.0`**: Added IPC serialization and deserialization support (`FileReader`, `StreamReader`, `FileWriter`) for `StringViewArray` (`Utf8View`) and `BinaryViewArray` (`BinaryView`).
- **`55.0.0` - `57.0.0`**: Strict 8-byte/64-byte buffer alignment validation and variadic buffer bounds checking on `.arrow` / `.feather` v2 files to prevent out-of-bounds panics on corrupted inputs.
- **`58.0.0` - `60.0.0`**: Deprecated and removed `StreamReader::try_new_unbuffered`, aligned schemas with `arrow-schema 60.0.0`, and enabled extraction of `ListArray`, `LargeListArray`, `FixedSizeListArray`, and `Float16Array`.

### B. `lz4_flex` (`0.11.0` - `0.14.0`): Mitigation of `CVE-2026-32829` (`RUSTSEC-2026-0041`)
- **`0.11.0` - `0.11.5`**: Inverted the `checked-decode` feature flag into `unchecked-decode` (off by default) so Cargo feature unification always preserves `safe-decode` mode (100% `unsafe`-free).
- **`0.11.6` & `0.12.1` (2026-05-09)**: **Mitigated `CVE-2026-32829` / `RUSTSEC-2026-0041` (CVSS 7.5 High)** — fixed an uninitialized memory read/leak during LZ4 block decompression when malformed input specified an invalid match offset beyond the output cursor.
- **`0.13.0` - `0.13.1` (2026-05-09)**: External dictionary compression support and out-of-bounds panic fix for dictionaries smaller than `< 4` bytes.
- **`0.14.0` (2026-07-14)**: Explicit separation of the `alloc` feature from `no_std` and optimized `frame::FrameDecoder` for `.lz4` streaming.

### C. `flate2` (`1.0.30` - `1.1.10`)
- **`1.1.0` - `1.1.8`**: Internal decoder buffer reuse in `GzDecoder` / `MultiGzDecoder` to eliminate repeated heap allocations.
- **`1.1.9` - `1.1.10` (Q3 2026)**: Added the `runtime_detection` feature flag (`zlib-rs?/std`, `crc32fast?/std`) into `default` features in `1.1.10`, resolving a throughput regression where runtime CPU SIMD detection was inadvertently omitted in `1.1.9`.

### D. `zstd` (`0.13.0` - `0.14.3`) & `zstd-safe` (`8.2.0+zstd.1.5.7`)
- **`0.14.0`**: Fixed a soundness bug in `zstd-safe 8.0.0+` by removing direct `DerefMut` access to `CCtx`/`DCtx` inside streaming wrappers, which previously allowed `std::mem::swap` to violate dictionary/buffer reference lifetimes.
- **`0.14.1` - `0.14.3`**: Upgraded to `zstd-safe ^8.2.0` (`libzstd 1.5.7`), supporting multi-frame `.zst` streams and skippable metadata frames in `zstd::stream::read::Decoder`.

### E. `half` (`2.4.0` - `2.7.1`)
- **`2.4.0` - `2.6.0`**: Stabilized `f16`/`bf16` slice conversions and hardware intrinsics (`f16c` / `avx512fp16` on `x86_64`, `fp16` on `aarch64`).
- **`2.7.0` - `2.7.1`**: Direct integration with `arrow-array 60.0.0` (`Float16Array`) and Rust Edition 2024 `unsafe_op_in_unsafe_fn` compliance.

---

## 6.3 Complete Details & Source Code Adaptations
Module-by-module adaptations in `luminamoon-dataloader` are documented in full in [`data/docs/library_audit/05_universal_formats_compression_and_diagnostics.md`](06_universal_dataloader_formats_and_compression.md).
