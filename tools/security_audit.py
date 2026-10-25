import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"
CARGO_PKG_RE = re.compile(
    r'\[\[package\]\]\s+name = "([^"]+)"\s+version = "([^"]+)"\s+source = "registry\+',
    re.MULTILINE,
)
FORBIDDEN_TRACKED_PATTERNS = (
    re.compile(r"(^|/)\.env$"),
    re.compile(r"\.(pem|key|pfx|p12)$", re.IGNORECASE),
)
SECRET_CONTENT_PATTERNS = (
    ("GitHub Personal/Fine-Grained Token", re.compile(r"\b(ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{82})\b")),
    ("Hugging Face Access Token", re.compile(r"\bhf_[A-Za-z0-9]{34,}\b")),
    ("PyPI API Token", re.compile(r"\bpypi-[A-Za-z0-9_-]{50,}\b")),
)


def audit_cargo_lock_osv(cargo_lock_path: Path) -> tuple[int, list[tuple[str, str, list[str]]]]:
    if not cargo_lock_path.exists():
        raise FileNotFoundError(f"Missing Cargo.lock at {cargo_lock_path}")

    content = cargo_lock_path.read_text(encoding="utf-8")
    packages = CARGO_PKG_RE.findall(content)
    if not packages:
        raise RuntimeError("No registry packages parsed from Cargo.lock")

    queries = [{"package": {"name": name, "ecosystem": "crates.io"}, "version": ver} for name, ver in packages]
    req = urllib.request.Request(
        OSV_BATCH_URL,
        data=json.dumps({"queries": queries}).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "LuminaMoon-Security-Audit/1.0.5"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    results = payload.get("results", [])
    findings: list[tuple[str, str, list[str]]] = []
    for (name, ver), res in zip(packages, results):
        vulns = res.get("vulns") or []
        if vulns:
            ids = [str(v.get("id", "UNKNOWN")) for v in vulns]
            findings.append((name, ver, ids))

    return len(packages), findings


def audit_tracked_files_for_secrets(repo_root: Path) -> tuple[int, list[str]]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    violations: list[str] = []

    for rel_path in tracked_files:
        for pat in FORBIDDEN_TRACKED_PATTERNS:
            if pat.search(rel_path):
                violations.append(f"Forbidden sensitive file tracked in git: {rel_path}")

        full_path = repo_root / rel_path
        if not full_path.is_file() or full_path.stat().st_size > 2_000_000:
            continue
        try:
            text = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for label, token_re in SECRET_CONTENT_PATTERNS:
            if token_re.search(text):
                violations.append(f"Potential {label} detected in tracked file: {rel_path}")

    return len(tracked_files), violations


def write_step_summary(
    crate_count: int,
    crate_findings: list[tuple[str, str, list[str]]],
    tracked_count: int,
    secret_violations: list[str],
) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    lines = [
        "### PROJECT LUMI Supply-Chain & Repository Security Audit",
        "",
        "| Security Check | Scope | Result |",
        "| :--- | :--- | :--- |",
        f"| **RustSec / OSV / CVE Crate Audit** | `{crate_count}` locked crates in `Cargo.lock` | {'PASS (`0` vulnerabilities)' if not crate_findings else f'FAIL (`{len(crate_findings)}` vulnerable crates)'} |",
        f"| **Tracked File & Secret Hygiene** | `{tracked_count}` git-tracked files | {'PASS (`0` leaked secrets)' if not secret_violations else f'FAIL (`{len(secret_violations)}` violations)'} |",
        "",
    ]
    Path(summary_path).open("a", encoding="utf-8").write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OSV/RustSec CVE and secret-hygiene security audit.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    crate_count, crate_findings = audit_cargo_lock_osv(repo_root / "Cargo.lock")
    tracked_count, secret_violations = audit_tracked_files_for_secrets(repo_root)
    write_step_summary(crate_count, crate_findings, tracked_count, secret_violations)

    print(f"[SECURITY AUDIT] Audited {crate_count} Rust crates in Cargo.lock against OSV/RustSec/CVE: {len(crate_findings)} advisories.")
    print(f"[SECURITY AUDIT] Audited {tracked_count} git-tracked files for credential hygiene: {len(secret_violations)} violations.")

    if crate_findings or secret_violations:
        for name, ver, ids in crate_findings:
            print(f"  [VULNERABILITY] Crate {name} v{ver}: {', '.join(ids)}", file=sys.stderr)
        for v in secret_violations:
            print(f"  [SECRET VIOLATION] {v}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
