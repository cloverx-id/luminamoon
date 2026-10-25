# Security Policy

## Supported Versions

**PROJECT LUMI** actively maintains and provides security updates for the latest releases of **LuminaMoon** (`luminamoon`) and the **LuminaV Optimizer** (`luminav`):

| Component / Package | Version | Security Support Status |
| :--- | :--- | :--- |
| **`luminamoon`** (Unified Rust + Python Suite) | `>= 1.0.5` | Supported (Active) |
| **`luminav`** (Standalone & Bridge Optimizer) | `>= 1.5.3` | Supported (Active) |
| **`luminamoon`** | `< 1.0.5` | Unsupported (Upgrade to `>= 1.0.5`) |
| **`luminav`** | `< 1.5.3` | Unsupported (Upgrade to `>= 1.5.3`) |

---

## Reporting a Vulnerability

We take the security, memory safety, and numerical integrity of **PROJECT LUMI** seriously. If you discover a security vulnerability, memory-safety defect in FFI/SIMD/Vulkan/Triton boundaries, or supply-chain issue:

1. **Private GitHub Security Advisory (Preferred)**:
   Submit a confidential report directly through [**GitHub Private Vulnerability Reporting**](https://github.com/cloverx-id/luminamoon/security/advisories/new).
2. **Direct Contact**:
   Email **Silver Moon** (`@cloverxion`) directly at [`leafmoon.idn@gmail.com`](mailto:leafmoon.idn@gmail.com) with the subject prefix `[SECURITY][luminamoon]`.

### Response & Disclosure Timeline
- **Initial Acknowledgment**: Within **48 hours** of report submission.
- **Triage & Root-Cause Verification**: Within **5 business days**.
- **Patch Release & Coordinated Disclosure**: Critical and high-severity fixes are prioritized for immediate patch release on GitHub, PyPI, and Hugging Face Hub alongside a published GitHub Security Advisory (GHSA / CVE).

Please do **not** open public GitHub issues for undisclosed security vulnerabilities.

---

## Automated Security & Verification Architecture

Every commit, pull request, and scheduled build in `cloverx-id/luminamoon` is continuously audited across multiple defensive layers (`.github/workflows/security.yml`):

1. **GitHub CodeQL SAST (CWE Static Analysis)**:
   Automated static AST dataflow analysis (`github/codeql-action@v4`) across Python modules and GitHub Actions workflows to detect Common Weakness Enumeration (**CWE**) vulnerabilities (command injection, path traversal, unsafe deserialization, and workflow privilege escalation).
2. **RustSec & OSV Supply-Chain CVE Auditing (`tools/security_audit.py` & `pip-audit`)**:
   Every locked Rust crate in `Cargo.lock` (`crates.io`) and Python dependency in `pyproject.toml` (`PyPI`) is verified against the **RustSec Advisory Database (`RUSTSEC-*`)**, **Open Source Vulnerabilities (`OSV`)**, **PyPA (`PYSEC-*`)**, and **National Vulnerability Database (`CVE-*` / `GHSA-*`)**.
3. **Pull Request Dependency Review Gate (`actions/dependency-review-action@v5`)**:
   Blocks pull requests that introduce dependencies with known `moderate`, `high`, or `critical` vulnerabilities.
4. **Secret Scanning & Push Protection**:
   Server-side GitHub Secret Scanning and Push Protection are enforced on the repository alongside tracked-file hygiene checks to prevent accidental credential exposure.
5. **Memory & FFI Safety Guarantees**:
   - Strict pointer anti-aliasing, alignment, and contiguity checks across `PyO3` (`abi3-py310` / `abi3t`) and NumPy/PyTorch zero-copy buffers.
   - Minimal dependency surface: only actively imported crates (`pyo3`, `ash`, `libloading`, `rayon`, `parquet`, `arrow-*`, `xxhash-rust`, `crc32fast`, `hashbrown`, `parking_lot`, `serde`, `serde_json`, `thiserror`) are locked in `Cargo.lock` (see [`docs/library_audit/`](docs/library_audit/README.md)).
