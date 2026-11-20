# LuminaMoon Changelogs

This directory separates release tracking into two distinct lineages to eliminate version ambiguity between the software distribution and the underlying mathematical specification:

| Document | Current Version | Scope |
| :--- | :--- | :--- |
| [**Software Changelog**](SOFTWARE_CHANGELOG.md) | **`v1.1.1.post2`** | Tracks the `luminamoon` package distributed on PyPI and GitHub (`pip install luminamoon`), covering the Rust workspace crates, CPython FFI bindings, Data Engine, LSO-Eval, and CI/CD infrastructure. |
| [**LuminaV Algorithm Changelog**](LUMINAV_ALGORITHM_CHANGELOG.md) | **`v2.0.0`** | Tracks the mathematical formulation, numerical invariants, stochastic rounding algorithms, and GPU/SIMD kernel evolution of the **LuminaV Optimizer** from `v1.0.0` through the `v2.0.0` specification. |

## Version Introspection in Python

```python
import luminamoon
from luminamoon import LuminaV
from luminamoon.optimizer import MATH_VERSION, get_math_version

# 1. Software Package Version (PyPI / Cargo / Data Engine)
print(luminamoon.__version__)       # "1.1.1.post2"
print(luminamoon.get_version())     # "1.1.1.post2"

# 2. LuminaV Algorithm Math Version
print(LuminaV.MATH_VERSION)         # "2.0.0"
print(LuminaV.get_math_version())   # "2.0.0"
print(MATH_VERSION)                 # "2.0.0"
print(get_math_version())           # "2.0.0"
```
