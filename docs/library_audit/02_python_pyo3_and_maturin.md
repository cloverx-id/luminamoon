# 2. Python, PyO3, and Maturin

## 2.1 Latest Stable Versions
- **Python**: `3.15.0` (released October 9, 2026)
- **`pyo3`**: `0.30.1`
- **`maturin`**: `1.15.0`

---

## 2.2 Version Range Changelog (`pyo3 0.24.0` - `0.30.1`)

### A. PyO3 0.25.0 - 0.26.0: Deprecation of `extension-module` Feature
- **What changed**: The Cargo feature `pyo3/extension-module` was deprecated in `pyo3 0.26.0`.
- **Upstream technical rationale**: Cargo features are additive across a workspace. When `extension-module` is enabled in `Cargo.toml` to build `.pyd`/`.so` modules, running `cargo test` in the same workspace fails at link time because `libpython` symbols are omitted when `extension-module` is active.
- **New idiom**: Extension module linking is controlled by the build system (`maturin >= 1.8`) or via `PYO3_BUILD_EXTENSION_MODULE=1` during manual compilation, rather than `features = ["extension-module"]` in `Cargo.toml`.

### B. PyO3 0.26.0 - 0.28.0: Removal of `Python::allow_threads` $\rightarrow$ `Python::detach`
- **What changed**: `py.allow_threads(|| ...)` was deprecated in `pyo3 0.26.0` and **completely removed** in `pyo3 0.28.0`.
- **Upstream technical rationale**: With free-threaded Python (PEP 703 / PEP 803 without the GIL on Python 3.13–3.15), `allow_threads` is a misnomer because Python threads already run concurrently without a GIL; the operation actually detaches the Rust thread from the Python runtime.
- **New idiom**: Use `py.detach(|| ...)`.

### C. PyO3 0.24.0 - 0.30.1: Declarative `#[pymodule] mod` Declarations
- **What changed**: Legacy procedural `fn m(m: &Bound<'_, PyModule>) -> PyResult<()>` blocks with `m.add_function(wrap_pyfunction!(..., m)?)?` were replaced by declarative `mod`-based module definitions.
- **New idiom**:
  ```rust
  #[pymodule]
  mod _luminamoon_optimizer_rs {
      #[pymodule_export]
      use super::step_pyo3;

      #[pymodule_export]
      use super::detect_hardware_pyo3;
  }
  ```

### D. PyO3 0.29.0 - 0.30.1 & Python 3.15: Stable ABI `abi3t` (PEP 803)
- **What changed**: Added the `abi3t` feature alongside `abi3-py310`.
- **Upstream technical rationale**: PEP 803 extends the Stable ABI (`abi3`) to cover free-threaded Python builds (`python3.14t` / `python3.15t`) in addition to classic GIL builds.

---

## 2.3 Adaptations in LuminaMoon Source Code

1. **`Cargo.toml` & `pyproject.toml`**:
   - Removed `"extension-module"` from `pyo3` features.
   - Configured `pyo3 = { version = "0.30.1", features = ["abi3-py310", "abi3t"] }`.
   - Updated `requires = ["maturin>=1.15,<2.0"]` in `pyproject.toml`.
2. **`optimizer/src/python_ffi.rs`**:
   - Replaced `py.allow_threads(...)` in `step_pyo3` with `py.detach(...)`:
     ```rust
     let result = py.detach(|| {
         engine.step_batch(&mut views, &config, &steps, &param_indices)
     });
     ```
3. **`optimizer/src/lib.rs`**:
   - Migrated imperative module initialization to a declarative `#[pymodule] mod _luminamoon_optimizer_rs` block with `#[pymodule_export]` and `#[pymodule_init]`.

---

## 2.4 Forward-Compatibility Architecture (Python 3.10 - 3.15+)

To guarantee zero-failure execution across future Python versions or environments lacking a Rust/C++ toolchain:

1. **Dual-Tier Forward Compatibility**:
   - **Tier 1 (Stable ABI `abi3` & `abi3t`)**: The CPython extension is compiled with PyO3 `features = ["abi3-py310", "abi3t"]`. A single binary wheel built against the CPython 3.10 Stable ABI is automatically forward-compatible across Python 3.11, 3.12, 3.13, 3.14, and 3.15 in both standard GIL and free-threaded (`Py_GIL_DISABLED`) modes.
   - **Tier 2 (Universal Pure-PyTorch / NumPy Fallback Engine)**: If the compiled `_luminamoon_optimizer_rs` binary is absent, corrupted, or running on an alternative interpreter, `python/luminamoon/optimizer/luminav.py` transparently falls back to its pure-PyTorch/NumPy execution engine (`_step_gpu_group`) with identical mathematical convergence and stochastic rounding.
