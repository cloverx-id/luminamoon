# 1. Rust Toolchain, SIMD Intrinsics, and Ecosystem Crates

## 1.1 Latest Stable Versions
- **Rust Compiler (`rustc` & `cargo`)**: `1.99.1` (`edition = "2024"`, next beta release `1.100.0`)
- **`rayon`**: `1.12.0`
- **`thiserror`**: `2.0.21`
- **`libloading`**: `0.9.0`
- **`serde`**: `1.0.229`
- **`serde_json`**: `1.0.154`
- **`cfg-if`**: Removed from dependencies (replaced by the standard library macro `std::cfg_select!`)

---

## 1.2 Version Range Changelog (`Rust 1.85.0` - `1.99.1`)

### A. Rust 1.85.0 (`edition = "2024"`)
- **What changed**: The `unsafe_op_in_unsafe_fn` lint is enabled by default in Rust Edition 2024. Declaring an `unsafe fn` no longer implicitly permits calling `unsafe` operations inside its body without an explicit `unsafe {}` block.
- **Syntax changes**: Stabilized `let-chains` (`if let Some(x) = a && x > 0`) inside `if` and `while` conditions.

### B. Rust 1.86.0 - 1.87.0 (Target Features & SIMD Intrinsics)
- **Rust 1.86.0**: Functions annotated with `#[target_feature(enable = "...")]` can be called safely from other functions that enable an identical or superset target feature set.
- **Rust 1.87.0**: Value-based SIMD intrinsics (such as `_mm256_add_ps`, `_mm512_fmadd_ps`, `_mm512_cmp_ps_mask`) no longer require an `unsafe` block when called inside a function with matching `#[target_feature]` attributes. Only raw pointer dereference intrinsics (`_mm256_loadu_ps`, `_mm256_storeu_ps`, `_mm512_loadu_ps`, `_mm512_storeu_ps`) remain `unsafe`.

### C. Rust 1.89.0 (AVX-512 Target Feature Stabilization)
- **What changed**: All `avx512f`, `avx512dq`, `avx512bw`, and `avx512vl` target features and `std::arch::x86_64::_mm512_*` intrinsics were fully stabilized on the stable compiler channel without requiring `#![feature(stdarch_x86_avx512)]`.

### D. Rust 1.95.0 (`std::cfg_select!`)
- **What changed**: Added the standard library macro `std::cfg_select!` for compile-time configuration branching.
- **Impact**: Eliminates the need for the third-party `cfg-if` crate.

---

## 1.3 Adaptations in LuminaMoon Source Code

### 1. Replacing `cfg-if` with `std::cfg_select!`
- **Files**: `Cargo.toml`, `optimizer/crates/luminav-simd/Cargo.toml`, `optimizer/crates/luminav-simd/src/lib.rs`
- **Changes**:
  Removed the external `cfg-if` dependency from `Cargo.toml`. Updated `detect_simd_backend()` in `optimizer/crates/luminav-simd/src/lib.rs` to use the standard macro:
  ```rust
  pub fn detect_simd_backend() -> SimdBackend {
      std::cfg_select! {
          target_arch = "x86_64" => {
              if std::arch::is_x86_feature_detected!("avx512f")
                  && std::arch::is_x86_feature_detected!("avx512dq")
                  && std::arch::is_x86_feature_detected!("avx512bw")
                  && std::arch::is_x86_feature_detected!("avx512vl")
                  && std::arch::is_x86_feature_detected!("fma")
              {
                  SimdBackend::Avx512
              } else if std::arch::is_x86_feature_detected!("avx2")
                  && std::arch::is_x86_feature_detected!("fma")
              {
                  SimdBackend::Avx2
              } else {
                  SimdBackend::Scalar
              }
          }
          _ => {
              SimdBackend::Scalar
          }
      }
  }
  ```

### 2. Separating `SimdBackend::Scalar` and `SimdBackend::Avx2` Dispatch Paths
- **File**: `optimizer/crates/luminav-simd/src/parallel/{mod, dual, tdclm}.rs`
- **Previous issue**: The `match backend` arm in chunk dispatch helpers grouped `SimdBackend::Avx2 | SimdBackend::Scalar` into `avx2::*_avx2` calls. Even though top-level callers intercepted `SimdBackend::Scalar` earlier, combining those arms risked an illegal instruction trap (`SIGILL`) if a helper were invoked directly on a CPU without AVX2.
- **Fix**: Explicitly separated `SimdBackend::Scalar` from `SimdBackend::Avx2` across all chunk dispatch functions in `parallel/dual.rs` and `parallel/tdclm.rs`.
