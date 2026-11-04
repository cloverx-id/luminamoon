# Security Policy

## Supported Versions

Security updates and bug fixes are maintained for the following versions of **LuminaMoon** (`luminamoon`) and the **LuminaV Optimizer** (`luminav`):

| Component / Package | Version | Security Support Status |
| :--- | :--- | :--- |
| **`luminamoon`** (Rust and Python Package) | `>= 1.1.0` | Supported (Active) |
| **`luminav`** (Algorithm Specification) | `>= 1.6.0` | Supported (Active) |
| **`luminamoon`** (Maintenance Line) | `1.0.8` – `1.0.9` | Supported (Maintenance) |
| **`luminav`** (Maintenance Line) | `1.5.5` | Supported (Maintenance) |
| **`luminamoon`** | `< 1.0.8` | Unsupported (Upgrade recommended) |
| **`luminav`** | `< 1.5.5` | Unsupported (Upgrade recommended) |

---

## Reporting a Vulnerability

If you discover a security vulnerability, memory-safety issue in FFI/SIMD/Vulkan/Triton boundaries, or dependency flaw:

1. **Private GitHub Security Advisory (Preferred)**:
   Submit a confidential report through [GitHub Private Vulnerability Reporting](https://github.com/cloverx-id/luminamoon/security/advisories/new).
2. **Direct Contact**:
   Email **Silver Moon** (`@cloverxion`) directly at [`leafmoon.idn@gmail.com`](mailto:leafmoon.idn@gmail.com) with the subject prefix `[SECURITY][luminamoon]`.

### Response & Disclosure Timeline
- **Initial Acknowledgment**: Within 48 hours of report submission.
- **Triage & Verification**: Within 5 business days.
- **Fix & Disclosure**: Critical fixes are prioritized for release on GitHub, PyPI, and Hugging Face Hub alongside a published GitHub Security Advisory (GHSA / CVE).

Please do not open public GitHub issues for undisclosed security vulnerabilities.

---

## Automated Security Checks

Every commit and pull request in `cloverx-id/luminamoon` is checked across automated workflows (`.github/workflows/security.yml`):

1. **Static Analysis (GitHub CodeQL)**:
   Automated dataflow analysis (`github/codeql-action@v4`) across Python modules and GitHub Actions workflows to catch injection, traversal, and unsafe deserialization issues.
2. **Dependency & Supply-Chain Scanning (`tools/security_audit.py` & `pip-audit`)**:
   Locked Rust crates in `Cargo.lock` and Python dependencies in `pyproject.toml` are checked against the RustSec Advisory Database, Open Source Vulnerabilities (OSV), PyPA, and the National Vulnerability Database.
3. **Dependency Review Gate (`actions/dependency-review-action@v5`)**:
   Blocks pull requests that introduce dependencies with known vulnerabilities.
4. **Secret Scanning & Push Protection**:
   GitHub Secret Scanning and Push Protection are active to prevent accidental credential commits.
5. **Memory & FFI Checks**:
   - Pointer alignment and contiguity verification across PyO3 (`abi3-py310` / `abi3t`) and NumPy/PyTorch buffers.
   - Minimal dependency footprint: only actively required crates are locked in `Cargo.lock` (see [`docs/library_audit/`](docs/library_audit/README.md)).
