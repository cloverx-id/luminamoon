# 1. Rust Toolchain, `std::arch` SIMD Intrinsics, `memmap2`, and Core Crates

## 1.1 Latest Stable Versions
- **Rust Compiler (`rustc` & `cargo`)**: `1.99.1` (`edition = "2024"`, `resolver = "3"`)
- **`memmap2`**: `0.9.11`
- **`rayon`**: `1.12.0`
- **`thiserror`**: `2.0.21`
- **`serde`**: `1.0.229`
- **`serde_json`**: `1.0.154`

---

## 1.2 Version Range Changelog (`Rust 1.85.0` - `1.99.1` & Supporting Crates)

### A. Rust 1.85.0 (`edition = "2024"`)
- **What changed**: The `unsafe_op_in_unsafe_fn` lint is enabled by default in Rust Edition 2024. Declaring a function as `unsafe fn` no longer implicitly permits calling `unsafe` operations inside the function body without an explicit `unsafe { ... }` block.
- **Syntax changes**: Stabilized `let-chains` (`if let Some(a) = opt && a > 0`) inside `if` and `while` conditions, now enforced by `clippy::collapsible_if`.

### B. Rust 1.86.0 - 1.89.0 (`std::arch` Target Features & `avx512bw` Stabilization)
- **Rust 1.86.0 - 1.87.0**: Value-based SIMD intrinsics (such as `_mm512_set1_epi8`, `_mm512_cmpeq_epi8_mask`, `_mm256_set1_epi8`, `_mm256_cmpeq_epi8`, `_mm256_movemask_epi8`, `vdupq_n_u8`, `vceqq_u8`, `vmaxvq_u8`) no longer require an `unsafe` block when called inside a `#[target_feature]` function. Wrapping value intrinsics in `unsafe` triggers an `unused_unsafe` compiler warning. Only raw pointer dereference intrinsics (`_mm512_loadu_si512`, `_mm256_loadu_si256`, `vld1q_u8`) and pointer arithmetic (`ptr.add(offset)`) remain `unsafe`.
- **Rust 1.89.0**: All `avx512f` and `avx512bw` target features and 512-bit byte intrinsics (`_mm512_cmpeq_epi8_mask`) were stabilized on the stable compiler channel without requiring `#![feature(stdarch_x86_avx512)]`.

### C. Rust 1.88.0 - 1.95.0 (`[T]::as_chunks` & `std::cfg_select!`)
- **Rust 1.88.0**: Stabilized the standard library method `slice::as_chunks::<N>()` to partition slices into fixed-size array chunks `&[[T; N]]` plus a remainder slice, replacing legacy `chunks_exact(N)` patterns (enforced by `clippy::chunks_exact_to_as_chunks`).
- **Rust 1.95.0**: Added the standard macro `std::cfg_select!` for cross-architecture conditional compilation without requiring the external `cfg-if` crate.

### D. `memmap2` (`0.9.0` - `0.9.11`): Mitigation of `RUSTSEC-2026-0186`
- **What changed**: In `memmap2 < 0.9.11`, internal memory-mapping end-boundary calculations used `ptr.add(len)`. Under modern Rust/LLVM pointer provenance rules, calling `ptr.add` on an empty mapping or at the edge of the address space violates `inbounds` invariants and triggers Undefined Behavior (UB).
- **Upstream fix (`0.9.11`)**: Replaced internal pointer boundary calculations with `wrapping_add` and stabilized `Advice::PopulateRead` (`MADV_POPULATE_READ` on Linux) for page pre-faulting.

### E. `thiserror` (`1.0` - `2.0.21`), `rayon` (`1.10` - `1.12.0`), and `serde_json` (`1.0.120` - `1.0.154`)
- **`thiserror 2.0.21`**: Migrated to the stabilized `core::error::Error` trait in `core`, removing mandatory reliance on internal `std::error::Error` paths and improving variable format expansion in `#[error("...")]` macros.
- **`rayon 1.12.0`**: Improved work-stealing scheduling efficiency on indexed parallel iterators (`into_par_iter().enumerate()`).
- **`serde_json 1.0.154`**: Optimized direct byte-buffer serialization (`serde_json::to_writer(&mut Vec<u8>, ...)`) without intermediate `String` allocations.

---

## 1.3 Adaptations in LuminaMoon Source Code

1. **Architecture Detection via `std::cfg_select!` and Scoped `unsafe` Intrinsic Blocks (`dataloader/src/simd_cpu.rs`)**:
   - Uses `std::cfg_select!` inside `detect_cpu_isa()` to select `x86`/`x86_64`, `aarch64`, or `Scalar` fallback branches without any `cfg-if` dependency.
   - In `find_byte_avx512`, `rfind_byte_avx512`, `find_byte_avx2`, `rfind_byte_avx2`, `find_byte_neon`, and `rfind_byte_neon`, `unsafe { ... }` blocks are restricted strictly to `_mm512_loadu_si512`, `_mm256_loadu_si256`, `vld1q_u8`, and `ptr.add(...)`:
     ```rust
     while offset + 64 <= len {
         let chunk = unsafe { _mm512_loadu_si512(ptr.add(offset).cast()) };
         let mask = _mm512_cmpeq_epi8_mask(chunk, v_needle);
         if mask != 0 {
             return Some(offset + mask.trailing_zeros() as usize);
         }
         offset += 64;
     }
     ```
2. **Adoption of `slice::as_chunks::<3>()` (`dataloader/src/multimodal_serialize.rs`)**:
   - The `encode_base64` function uses `let (chunks, rem) = input.as_chunks::<3>();` to process 3-byte blocks directly without per-iteration bounds checks.
3. **Safe Memory Mapping with `memmap2 0.9.11` (`dataloader/src/source_local.rs`)**:
   - `LocalMmapLineSource::open` maps local files using `memmap2 0.9.11` (patched against `RUSTSEC-2026-0186`) and enforces `start_coord.byte_offset.min(total_len)` bounds checks prior to slicing.
