# Architecture & Hardware Acceleration

Technical architecture overview of **LuminaMoon (`luminamoon`)**, covering the **LuminaV Optimizer** compute hierarchy, the **LuminaMoon Data Engine** text preprocessing and SIMD pipeline, zero-copy memory protocols, and free-threaded Python (`abi3t` / PEP 703) execution.

---

## 1. LuminaV Optimizer Architecture & Dispatch (`optimizer/`)

```mermaid
flowchart TD
    PyTorch["PyTorch Tensors (FP32 / BF16 / FP16)"] --> FFI["LuminaV PyO3 abi3t FFI (Zero-Copy)"]
    NumPy["NumPy 2.x Arrays (Buffer Protocol)"] --> FFI
    FFI --> Dispatcher["Hardware Capability Dispatcher"]
    Dispatcher --> Triton["Triton AOT/JIT GPU (CUDA / ROCm)"]
    Dispatcher --> Vulkan["Vulkan 1.3/1.4 SPIR-V (UMA, N >= 16,384)"]
    Dispatcher --> SIMD["Rayon CPU SIMD (x86_64, N < 16,384)"]
    Dispatcher --> Scalar["Reference Scalar Core (Fallback)"]
    Triton & Vulkan & SIMD & Scalar --> Math["CLM, TD-CLM & Triangle Family (T1/T2) Update"]
```

### Optimizer Rust Workspace Crates (`optimizer/crates/`)

1. **`luminav-core`**: Mathematical formulas (CLM, TD-CLM, and Triangle Family `T1`/`T2`), state structs (`ParamState`), mode enums (`BufferMode`, `BoundType`, `DTypeMode`, `ExecutionMode`), XORShift/PCG PRNG generators, and numeric conversions (`f32_to_f16_bits_rne`, `f16_bits_to_f32`).
2. **`luminav-simd`**: Explicit x86_64 SIMD vectorization via `__m512` (`AVX-512` F, DQ, BW, VL: 16 floats/register) and `__m256` (`AVX2 + FMA`: 8 floats/register) parallelized across CPU cores via Rayon (`ELEM_CHUNK_SIZE = 2,097,152`-element chunks in `src/parallel/mod.rs`).
3. **`luminav-vulkan`**: Vulkan 1.3 / 1.4 compute shader pipeline via `ash` bindings, using Unified Memory Architecture (UMA) zero-copy host buffers and Wave32 subgroup reductions (`subgroupAdd`).
4. **`luminav-dispatch`**: Runtime hardware capability detector (`detect_hardware`) and dynamic router (`N >= 16,384` elements routes to Triton AOT or Vulkan UMA when available; `< 16,384` routes to Rayon CPU SIMD).
5. **`luminav-triton-aot`**: Ahead-of-Time (AOT) kernel runner with dynamic `libloading` CUDA (`nvcuda`) and ROCm HIP (`amdhip64_7`) driver bindings and a standalone C-API (`luminav_triton_standalone.h`).
6. **`optimizer/src/` (`_luminamoon_optimizer_rs`)**: Unified PyO3 0.30.1 `abi3t` extension linking both the Optimizer and Data Engine crates, releasing the GIL via `py.detach()`.

---

## 2. LuminaMoon Data Engine Architecture & SIMD Core (`data/`)

```mermaid
flowchart LR
    Raw["Raw Text Batch / Document Stream"] --> Cleaner["1. TextCleaner (AVX-512/AVX2/NEON SIMD)"]
    Cleaner --> Quality["2. QualityFilter (Shannon Entropy & Symbol Ratio)"]
    Quality --> Dedup["3. StreamDeduplicator (XXH3-64 WAL) & MinHash"]
    Dedup --> Masker["4. PIIMasker (Multi-Pattern Sanitizer)"]
    Masker --> Tensor["5. StringTensor (Dual-Layout CPU StringView / GPU Contiguous)"]
    Tensor --> Export["Arrow IPC / Polars / PyTorch GPU Tensors"]
    Tensor --> Checkpoint["6. StateTracker (128B Dual-Slot Ping-Pong CRC32)"]
```

### Data Engine Rust Modules (`data/src/`)

1. **`constants.rs` & `error.rs`**: Centralized endpoints, buffer bounds, URI typo tables, and `DataLoaderError` with Damerau-Levenshtein suggestions (`did_you_mean`).
2. **`coord.rs` & `metadata.rs`**: 32-byte little-endian `StreamCoordinate` (`shard_id`, `byte_offset`, `row_group_idx`, `page_idx`, `row_offset_in_page`, `global_row_idx`) and 128-byte Dual-Slot Ping-Pong `.meta` checkpoints (`Slot A` / `Slot B` protected by IEEE CRC32) with write-ahead `.dedup` logging.
3. **`dedup.rs` & `minhash.rs`**: 64-bit `xxh3_64` deduplication via SwissTable `IdentityBuildHasher` and 64-permutation MinHash LSH fuzzy shingling for near-duplicate rejection.
4. **`simd_cpu.rs` & `postprocess.rs`**: Self-contained CPU SIMD byte-scanning kernels (`AVX-512BW` 64-byte ZMM, `AVX2` 32-byte YMM, `ARM64 NEON` 16-byte) for whitespace trimming and canonical newline normalization, backed by 64-byte cache-line aligned buffers.
5. **`string_tensor.rs`**: Dual-layout container maintaining CPU inlined `StringView` (<= 12 bytes inline with 4-byte prefix comparison) and contiguous GPU flat `uint32` offsets + `uint8` byte payload buffers.
6. **`quality.rs`**: Parallel Rayon heuristic quality filter computing Shannon Character Entropy, repetitiveness fraction, and symbol-to-word ratios.
7. **`serialize.rs`**: GIL-detached serialization to Apache Arrow IPC (`LargeString`), Parquet, and NDJSON format buffers.
8. **`py_bridge.rs`**: Native PyO3 (`abi3t`) bindings exposing `FusedPipeline`, `StringTensor`, `StateTracker`, `clean_batch`, and `filter_quality` to Python with `py.detach()` GIL release.

---

## 3. Zero-Copy Memory Protocols

1. **Optimizer Tensor Pointers**: Extracts physical memory addresses directly via NumPy `__array_interface__['data'][0]` or PyTorch `tensor.data_ptr()`, validating C-contiguity and pointer anti-aliasing before mutating `&mut [f32]` slices in-place.
2. **Data Engine Dual-Layout `StringTensor`**: Maintains 16-byte Umbra CPU `StringView16` views alongside contiguous `uint32` offset + `uint8` UTF-8 payload buffers for zero-copy export to Apache Arrow (`to_arrow`), Polars (`to_polars`), NumPy (`to_numpy_gpu`), and PyTorch GPU tensors (`to_torch_gpu`).

---

## 4. CPython Stable ABI & Free-Threaded Runtime (`abi3t`)

The `luminamoon` binary wheel is compiled against CPython Stable ABI standards:

- **`abi3-py310`**: A single wheel compiled for Python 3.10 runs without recompilation across Python 3.10 through 3.15+.
- **`abi3t` (PEP 703)**: Compatible with Python 3.13+ free-threaded builds (`Py_GIL_DISABLED`). Both the Optimizer and Data Engine execute compute and data processing inside `py.detach()`, allowing multi-core execution without holding the GIL.

---

## 5. References

- **Optimizer Mathematical Foundations**: L. Maligranda, *Banach J. Math. Anal.*, 2008 (Triangle Inequality Defect); I. Loshchilov and F. Hutter, *ICLR*, 2019 (`AdamW`); M. R. Zhang *et al.*, *NeurIPS*, 2019 (`Lookahead`); J. Zhuang *et al.*, *NeurIPS*, 2020 (`AdaBelief`); S. Taniguchi *et al.*, *NeurIPS*, 2024 (`ADOPT`); K. Liang *et al.*, *ICLR*, 2026 (`Cautious Optimizers`); A. Brock *et al.*, *ICML*, 2021 (`AGC`).
- **Low-Precision Execution & Kernel Compilation**: S. Gupta *et al.*, *ICML*, 2015 (Stochastic Rounding); P. Zamirai *et al.*, *arXiv:2010.06192*, 2021 (`BFloat16`); A. Molina *et al.*, *ICLR*, 2020 (Padé Approximants); P. Tillet *et al.*, *MAPL*, 2019 (Triton Compiler).
- **Data Engine String & Deduplication Structures**: T. Neumann and M. Freitag, *CIDR*, 2020 (Umbra 16-byte `StringView` prefix inlining); A. Z. Broder, *Compression and Complexity of Sequences*, 1997 (MinHash LSH).
