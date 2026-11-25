# Security Policy

## Supported Versions

Security updates and bug fixes are maintained for the active **2.x** development and release line of **LuminaMoon** (`luminamoon`) and the **LuminaV Optimizer** (`luminav`). All versions below `2.0.0` (including the `1.x.x` series) have reached End of Life and are no longer maintained.

| Component / Package | Version | Security Support Status |
| :--- | :--- | :--- |
| **`luminamoon`** (Rust and Python Package) | `>= 2.0.0rc1` (`2.x` line) | Supported (Active — Pre-Release RC) |
| **`luminav`** (Algorithm Math) | `>= 2.0.0` | Supported (Active) |
| **`luminamoon`** | `< 2.0.0` (`1.x.x`) | Unsupported (End of Life; upgrade to `2.0.0` upon release) |
| **`luminav`** | `< 2.0.0` (`1.x.x`) | Unsupported (End of Life; upgrade to `2.0.0` upon release) |

---

## Reporting a Vulnerability

If you discover a security vulnerability, memory-safety issue in FFI/SIMD/CubeCL/Triton boundaries, or dependency flaw:

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

Security workflows (`.github/workflows/security.yml`) run on pushes to `main`, pull requests targeting `main` or `development`, weekly scheduled runs, and manual dispatches:

1. **Static Analysis (GitHub CodeQL)**:
   Automated dataflow analysis (`github/codeql-action@v4`) across Python modules and GitHub Actions workflows to catch injection, traversal, and unsafe deserialization issues.
2. **Dependency & Supply-Chain Scanning (`tools/security_audit.py` & `pip-audit`)**:
   Locked Rust crates in `Cargo.lock` are queried against the Open Source Vulnerabilities (`api.osv.dev`) database covering RustSec advisories (with non-security `unmaintained`/`notice` entries reported as warnings), and the installed Python package is checked with `pip-audit`. *(Optional extras in `[project.optional-dependencies]` such as `torch`, `pyarrow`, and `polars` are installed and audited in user environments.)*
3. **Dependency Review Gate (`actions/dependency-review-action@v5`)**:
   Flags and blocks pull requests introducing dependencies with known vulnerabilities at `moderate` severity or higher.
4. **Secret Scanning & Push Protection**:
   GitHub Secret Scanning and Push Protection are active to prevent accidental credential commits.
5. **Memory & FFI Checks**:
    - Pointer alignment and contiguity verification across PyO3 (`abi3-py310` / `abi3t`) and NumPy/PyTorch buffers.
    - Minimal dependency footprint: only actively required crates are locked in `Cargo.lock` (see [`docs/dependencies/`](docs/dependencies/README.md)).
