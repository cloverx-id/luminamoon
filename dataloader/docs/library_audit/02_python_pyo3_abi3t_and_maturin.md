# 2. Python, PyO3 (`abi3t`), and Maturin

## 2.1 Latest Stable Versions
- **Python**: `3.15.0` (supports both standard CPython and free-threaded `Py_GIL_DISABLED` builds)
- **`pyo3`**: `0.30.1` (`features = ["abi3-py310", "abi3t"]`)
- **`maturin`**: `1.15.0` (`requires = ["maturin>=1.15,<2.0"]`)

---

## 2.2 Version Range Changelog (`pyo3 0.24.0` - `0.30.1` & `maturin 1.8` - `1.15.0`)

### A. PyO3 0.25.0 - 0.26.0: Deprecation of `extension-module` Feature
- **What changed**: The Cargo feature `pyo3/extension-module` was deprecated starting in `pyo3 0.26.0`.
- **Upstream technical rationale**: Cargo features are additive across an entire workspace. Enabling `extension-module` in `Cargo.toml` caused `cargo test` to fail during linking because `libpython` symbols were intentionally omitted when that feature was active.
- **New idiom**: Extension module linking is controlled directly by `maturin >= 1.8` (or the environment variable `PYO3_BUILD_EXTENSION_MODULE=1`), rather than static declarations in `Cargo.toml`.

### B. PyO3 0.26.0 - 0.28.0: Removal of `Python::allow_threads` $\rightarrow$ `Python::detach`
- **What changed**: `py.allow_threads(|| ...)` was deprecated in `pyo3 0.26.0` and **completely removed** in `pyo3 0.28.0`.
- **Upstream technical rationale**: On free-threaded Python runtimes (PEP 703 / PEP 803 without the GIL), Python threads already execute in parallel; releasing the context actually detaches the OS thread from the CPython runtime state.
- **New idiom**: Use `py.detach(|| ...)`.

### C. PyO3 0.27.0 - 0.30.1: Removal of `Bound::downcast` $\rightarrow$ `Bound::cast::<T>()`
- **What changed**: `Bound<'py, PyAny>::downcast::<T>()` and `downcast_into::<T>()` were deprecated and removed in `pyo3 0.30.1`, replaced by `Bound<'py, PyAny>::cast::<T>()` and `cast_into::<T>()`.
- **Upstream technical rationale**: Unified smart-reference type conversion naming across the `Bound<'py, T>` API.

### D. PyO3 0.29.0 - 0.30.1 & Python 3.15: Stable ABI `abi3t` (PEP 803) and `#[pyclass(frozen, subclass)]`
- **What changed**:
  1. Added the `abi3t` feature alongside `abi3-py310` to support the cross-version Stable ABI across Python 3.10–3.15 on both standard GIL builds and free-threaded builds (`python3.14t` / `python3.15t`).
  2. Rust classes exposed to Python via `#[pyclass]` no longer permit Python-side subclassing by default unless the `subclass` attribute is explicitly specified, and `frozen` is recommended for lock-free concurrency without `PyCell` borrow-checking overhead in free-threaded mode.
  3. Inside declarative `#[pymodule] mod ...` blocks, external sub-components are registered using functions annotated with `#[pymodule_init]`.

---

## 2.3 Adaptations in LuminaMoon Source Code

1. **`Cargo.toml` & `dataloader/Cargo.toml`**:
   - Configured `pyo3 = { version = "0.30.1", features = ["abi3-py310", "abi3t"] }` in `[workspace.dependencies]` without `"extension-module"`, allowing `cargo test -p luminamoon-dataloader` to link and run standalone.
2. **Runtime Detachment via `py.detach` (`dataloader/src/py_bridge.rs`)**:
   - All disk I/O, `mmap` reads, HTTP 206 network streaming, Parquet schema inference, and HuggingFace Hub NDJSON uploads execute inside `py.detach(|| ...)`:
     ```rust
     let batch_opt = py
         .detach(|| self.inner.next_batch())
         .map_err(to_py_err)?;
     ```
3. **Type Casting via `Bound::cast::<T>()` and `PyBool` vs `PyInt` Hierarchy (`dataloader/src/py_bridge.rs`)**:
   - Uses `columns.cast::<PyDict>()`, `v.cast::<PySequence>()`, and `pair.cast::<PyTuple>()` inside `extract_py_columns_table`.
   - Because `bool` is a subclass of `int` in CPython, `py_any_to_multimodal_value` always checks `item.is_instance_of::<PyBool>()` before `item.is_instance_of::<PyInt>()` so boolean columns are never coerced into `1`/`0` integers.
4. **`#[pyclass(frozen, subclass)]` Declarations and `#[pymodule_init]` Registration (`dataloader/src/py_bridge.rs` & `optimizer/src/lib.rs`)**:
   - `PyDataStreamLoader`, `PyMultimodalParquetLoader`, and `PyHfHubUploader` are declared with `#[pyclass(frozen, subclass, ...)]` and synchronized on the Rust side using `parking_lot::Mutex`.
   - `register_dataloader_classes(m)` is invoked from `#[pymodule_init] fn init(m: &Bound<'_, PyModule>)` inside `optimizer/src/lib.rs`, registering all classes and module-level functions (`serialize_table_to_bytes`, `postprocess_and_dedup_records`, `suggest_closest_matches`).
5. **Automatic Row-to-Column Transposition & GIL-Free Table/Postprocess Functions (`dataloader/src/py_bridge.rs`)**:
   - `extract_py_columns_table` supports three Python input layouts simultaneously: `dict[str, Sequence[Any]]`, `Sequence[tuple[str, Sequence[Any]]]`, and `Sequence[dict[str, Any]]` (transposing row dictionaries into ordered columns by first-seen key order and filling missing keys with `MultimodalValue::Null`).
   - `serialize_table_to_bytes` and `postprocess_and_dedup_records` (with `deduplicate: bool = true`) execute Parquet/Arrow IPC/CSV/JSONL/JSON serialization and Rayon + SIMD normalization + `xxh3_64` deduplication completely outside the GIL via `py.detach(|| ...)`.
