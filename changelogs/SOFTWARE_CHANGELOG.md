# LuminaMoon Software Changelog

All notable software, packaging, FFI, and infrastructure changes to the `luminamoon` monorepo and PyPI distribution (`pip install luminamoon`) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For the mathematical and kernel evolution of the **LuminaV Optimizer** specification (`v1.0.0` to `v1.5.3`), see [`LUMINAV_ALGORITHM_CHANGELOG.md`](LUMINAV_ALGORITHM_CHANGELOG.md).

---

## [1.0.6] - 2026-10-26

### Added
- **Python 3.15 Classifier Support**: Added `Programming Language :: Python :: 3.15` to package metadata in `pyproject.toml`.
- **O(1) Deduplication Membership Check**: Added `TwoStageDeduplicator::is_seen` in `data/src/dedup.rs` and wired `PyStreamDeduplicator::is_seen` to avoid $O(N)$ hash-vector exports.

### Changed
- **PROJECT LOOMY Rebranding & Author Standardization**: Standardized project branding to **PROJECT LOOMY** (`Loomy(luminamoon)`) with **Silver Moon (`@cloverxion`)** listed first as lead author followed by **Lumina Moon (`@cloverx-id`) contributors** across documentation and manifests.
- **Optional Dependency Import Diagnostics**: Wrapped `numpy`, `torch`, `pyarrow`, and `polars` imports in `StringTensor` export methods with actionable installation hints.
- **Fallback Signature Parity**: Aligned `suggest_closest_matches` (with Damerau-Levenshtein edit-distance ranking) and `postprocess_and_dedup_records` return tuple `(cleaned_unique, newly_committed)` in `luminamoon.data.fallbacks` with Rust FFI and `_stubs.py`.

### Fixed
- **WAL & GPU Buffer Corruption Hardening**: Added byte-alignment validation (`% 8 == 0` for `.dedup` WAL files, `% 4 == 0` for GPU `uint32` offset buffers), exact pre-commit offset seeking in `DualMetadataManager::commit_gpu_batch`, and `StringView16` field encapsulation (`pub(crate)`), inline length clamping, and saturating `u32` offset arithmetic in `data/src/string_tensor.rs`.
- **Data Pipeline & Collate Hardening**: Fixed WAL double-commit skip in `fused_process_batch`, added missing column typo suggestions and `bytes` decoding in `_extract_text`, auto-populated `tokenizer.pad_token = tokenizer.eos_token` when missing, restored resume coordinates in `ResumablePipeline`, and eliminated background prefetcher queue timeout drops.

---

## [1.0.5.post1] - 2026-10-25

### Changed
- **Modular Data Processing Engine**: Decomposed the monolithic data processing fallback layer in `luminamoon.data` into dedicated submodules under `fallbacks/` (`cleaner.py`, `quality.py`, `dedup.py`, `minhash.py`, `masker.py`, `coordinate.py`, `tracker.py`, `string_tensor.py`, `fused.py`, `serialization.py`, `helpers.py`), preserving 100% backward compatibility.
- **Updated Algorithm Changelog**: Updated `LUMINAV_ALGORITHM_CHANGELOG.md` with detailed mathematical formulation and parameter comparisons between 1.3.3 and 1.5.3.
- **PEP 440 Post-Release Synchronization**: Bumped package distribution version to `1.0.5.post1`.

---

## [1.0.5] - 2026-10-24

### Added
- **Comprehensive In-Editor Docstrings (`luminav.py`)**: Added Google-style docstrings across `LuminaV.__init__`, `step`, `zero_grad`, `state_dict`, `load_state_dict`, and `detect_hardware`, providing full IDE autocomplete tooltips and explicit default hyperparameter values.
- **Hyperparameter Quick Reference (`README.md`)**: Added a structured default parameter reference table directly to the main repository README.
- **Dedicated Dual-Track Changelog Directory (`changelogs/`)**: Separated release documentation into `SOFTWARE_CHANGELOG.md` (package releases) and `LUMINAV_ALGORITHM_CHANGELOG.md` (algorithm specification history).
- **Explicit Algorithm Specification Constant (`LuminaV.ALGORITHM_SPEC`)**: Added `LuminaV.ALGORITHM_SPEC = "1.5.3"` to clearly distinguish the mathematical specification version from the software package version.

### Changed
- **Unified Software Version Synchronization (`1.0.5`)**: Synchronized `luminamoon.__version__`, `luminamoon.data.get_version()`, `pyproject.toml`, and `Cargo.toml` to a single source of truth (`1.0.5`).
- **100% Air-Gapped Offline Version Resolution (`version.py`)**: Removed runtime HTTP GitHub file fetching from `luminamoon.optimizer.version` and `luminamoon.data.version`, guaranteeing deterministic, zero-network execution in air-gapped HPC clusters.

### Security
- **Upgraded `setuptools` (`84.0.0` -> `84.1.0`)**: Patched development dependency `setuptools` in `requirements.txt` to resolve `GHSA-xmj3-9gf7-h52w` (`CWE-22`).

---

## [1.0.4] - 2026-10-24

### Added
- **Free-Threaded CPython (No-GIL / PEP 703) Support**: Verified compilation and runtime execution of native `abi3t` / free-threaded wheels on Python `3.14t` and `3.15t`.
- **Rust Core Invariant Test Suite**: Added `optimizer/crates/luminav-core/tests/core_tests.rs` covering PRNG seed determinism, Padé rational $\tanh$ bounds, scratch buffer EMA smoothing, and Levenshtein typo suggestions.
- **PyPI Project URLs & Contributors Metadata**: Added `Homepage`, `Repository`, `Issues`, `Documentation`, and `Changelog` links alongside `Lumina Moon (cloverx-id) Contributors` to `pyproject.toml`, `Cargo.toml`, and `LICENSE`.

### Changed
- **Rust Workspace Modularization**:
  - Decomposed `luminav-core/src/reference_cpu.rs` into `reference_cpu/{mod, dual, tdclm}.rs`.
  - Decomposed `luminav-simd` (`avx2.rs`, `avx512.rs`, `parallel.rs`) into dedicated submodule directories (`dual.rs`, `tdclm.rs`, `mod.rs`).
  - Extracted Vulkan UMA step dispatches into `luminav-vulkan/src/{dual, tdclm}.rs`.
  - Decomposed `optimizer/src/python_ffi.rs` into `python_ffi/{config_parser, extract, worker, mod}.rs`.
- **Python Package Modularization**:
  - Split `luminav.py` into `standalone.py`, `validation.py`, and `numpy_backend.py`.
  - Split `gpu_backend.py` into `fsdp_sync.py`, `torch_ops.py`, and `triton_kernels.py`.
  - Added cross-platform cache directory resolution (`LOCALAPPDATA` on Windows, `XDG_CACHE_HOME` on POSIX).
- **Reference Implementation Restructuring**:
  - Decomposed monolithic `optimizer/reference/luminav_v153.py` (3,996 lines) into `core/`, `kernels/`, `single_backend/`, `triton_backend/`, and `utils/`.

---

## [1.0.3] - 2026-10-21

### Added
- **Two-Stage Deferred Release Pipeline**: Implemented draft GitHub release artifact staging on tag push (`v*`) with automated PyPI publication triggered upon release publication.

### Fixed
- **SIMD Finite BF16 Clamps**: Hardened AVX-512 and AVX2 finite clamp boundaries prior to BFloat16 stochastic rounding.
- **Build Wheel Privacy Sanitizer**: Added automated local path scrubbing in `tools/build_wheel.py` without modifying compiled ELF/PE/Mach-O relocation tables.

---

## [1.0.2.post2] - 2026-10-20

### Fixed
- **Dynamic Tag Versioning**: Added `tools/sync_version.py` to synchronize `pyproject.toml` and `_version.py` directly from Git release tags during CI wheel builds.

---

## [1.0.2.post1] - 2026-10-20

### Fixed
- **PEP 440 Post-Release Compliance**: Standardized version regex and enabled `skip-existing: true` on PyPI publishing workflows.

---

## [1.0.2] - 2026-10-20

### Added
- **Modular `luminamoon.data` Engine**: Added SIMD text normalization (`TextCleaner`), Shannon entropy & symbol heuristic filtering (`QualityFilter`), two-stage XXH3-64 exact deduplication (`StreamDeduplicator`), 64-permutation MinHash LSH fuzzy deduplication (`MinHashDeduplicator`), PII masking (`PIIMasker`), 128-byte Ping-Pong crash recovery (`StateTracker`), and dual-layout `StringTensor` (CPU 16-byte Umbra `StringView` & GPU contiguous offset+payload buffers).
- **Canonical LuminaV v1.5.3 Engine Integration**: Integrated the Rust AVX-512/AVX2 SIMD, Vulkan 1.3/1.4 UMA compute, and Triton AOT/JIT backends.

---

## [1.0.1] - 2026-10-20

### Added
- Initial monorepo release unifying the `luminamoon` Universal DataLoader and `LuminaV` Optimizer under a single Rust + Python workspace.
