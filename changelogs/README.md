# LuminaMoon Changelogs

This directory separates release tracking into two distinct lineages to eliminate version ambiguity between the software distribution and the underlying mathematical specification:

| Document | Current Version | Scope |
| :--- | :--- | :--- |
| [**Software Changelog**](SOFTWARE_CHANGELOG.md) | **`v1.0.8.post1`** | Tracks the `luminamoon` package distributed on PyPI and GitHub (`pip install luminamoon`), covering the Rust workspace crates, CPython FFI bindings, Data Engine, LSO-Eval suite, and CI/CD infrastructure. |
| [**LuminaV Algorithm Changelog**](LUMINAV_ALGORITHM_CHANGELOG.md) | **`v1.5.5`** | Tracks the mathematical formulation, numerical invariants, stochastic rounding algorithms, and GPU/SIMD kernel evolution of the **LuminaV Optimizer** from `v1.0.0` through the canonical `v1.5.5` specification. |

## Version Introspection in Python

```python
import luminamoon
from luminamoon import LuminaV
import luminamoon.data as lmd

# 1. Software Package Version (PyPI / Cargo / Data Engine)
print(luminamoon.__version__)      # "1.0.8.post1"
print(lmd.get_version())           # "1.0.8.post1"

# 2. Canonical LuminaV Algorithm Specification
print(LuminaV.ALGORITHM_SPEC)      # "1.5.5"
print(LuminaV.getVersion())        # "1.5.5"
```
