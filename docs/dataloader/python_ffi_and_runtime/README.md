# `py_bridge.rs` & `python/luminamoon/data/`: PyO3 `abi3t` FFI Bridge & Modular Python Runtime

[`data/src/py_bridge.rs`](https://github.com/cloverx-id/luminamoon/blob/main/data/src/py_bridge.rs) and the modular [`python/luminamoon/data/`](https://github.com/cloverx-id/luminamoon/blob/main/python/luminamoon/data/__init__.py) package unite the native Rust SIMD and columnar engine with Python under CPython Stable ABI (`abi3-py310` & free-threaded `abi3t` PEP 703).

---

## 1. Native PyO3 Classes & Functions (`src/py_bridge.rs`)

All native symbols are compiled into `_luminamoon_optimizer_rs` and re-exported through `python/luminamoon/data/_ffi.py`:

| Rust PyO3 Symbol | Python Wrapper (`luminamoon.data`) | GIL Detached (`py.detach`) | Core Responsibility |
| :--- | :--- | :--- | :--- |
| `PyStreamCoordinate` | `StreamCoordinate` (`checkpoint.py`) | N/A (32B value) | 32-byte little-endian stream coordinate with `to_bytes()` / `from_bytes()`. |
| `PyStateTracker` | `StateTracker` (`checkpoint.py`) | Yes | 128-byte Dual-Slot Ping-Pong (`Slot A` / `Slot B`) atomic checkpoint manager (`read`, `update`, `commit`). |
| `PyStreamDeduplicator` | `StreamDeduplicator` (`dedup.py`) | Yes | Two-stage `xxh3_64` SwissTable deduplicator with `.dedup` WAL (`is_unique`, `filter_batch`, `commit`, `rollback`). |
| `PyMinHashDeduplicator` | `MinHashDeduplicator` (`minhash.py`) | Yes | 64-permutation MinHash LSH near-duplicate detector (`is_unique`, `similarity`, `filter_batch`). |
| `PyQualityFilter` | `QualityFilter` (`quality.py`) | Yes | Parallel Rayon Shannon entropy, repetition, and symbol ratio filter (`is_high_quality`, `metrics`, `filter_batch`). |
| `PyStringTensor` | `StringTensor` (`string_tensor.py`) | Yes | Dual-layout 16B CPU `StringView16` and contiguous GPU/Arrow buffer container. |
| `fused_pipeline_process_batch` | `FusedPipeline` (`fused.py`) | Yes | Single-pass Rayon + SIMD text cleaning, quality filtering, and deduplication returning `StringTensor`. |
| `serialize_table_to_bytes` | `serialize_to_*` (`serialization.py`) | Yes | Columnar serialization to Parquet (`PAR1`), Arrow IPC (`ARROW1`), and JSONL. |

---

## 2. Modular Pure-Python Fallback Layer (`python/luminamoon/data/fallbacks/`)

When running in environments without the compiled native binary, `_ffi.py` transparently falls back to the modular `fallbacks/` subpackage:

- `cleaner.py`, `quality.py`, `dedup.py`, `minhash.py`, `masker.py`, `coordinate.py`, `tracker.py`, `string_tensor.py`, `fused.py`, `serialization.py`, and `helpers.py`.
- Preserves 100% API and binary format compatibility (including 32-byte `StreamCoordinate` and 128-byte IEEE CRC32 Ping-Pong `.meta` checkpoints).
