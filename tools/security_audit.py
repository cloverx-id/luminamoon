import argparse
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"
OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{}"
NON_BLOCKING_INFORMATIONAL_TYPES = frozenset({"unmaintained", "notice"})
CARGO_PKG_RE = re.compile(
    r'\[\[package\]\]\s+name = "([^"]+)"\s+version = "([^"]+)"\s+source = "registry\+',
    re.MULTILINE,
)
DISALLOWED_PATH_RULES = (
    re.compile(r"(^|/)\.env$"),
    re.compile(r"\.(pem|pfx|p12)$", re.IGNORECASE),
)
DISALLOWED_CONTENT_RULES = (
    ("GitHub API Prefix (ghp_/github_pat_)", re.compile(r"\b(ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{82})\b")),
    ("Hugging Face Hub Prefix (hf_)", re.compile(r"\bhf_[A-Za-z0-9]{34,}\b")),
    ("PyPI Upload Prefix (pypi-)", re.compile(r"\bpypi-[A-Za-z0-9_-]{50,}\b")),
)


def _fetch_osv_advisory(vuln_id: str) -> dict | None:
    url = OSV_VULN_URL.format(urllib.parse.quote(vuln_id, safe=""))
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "LuminaMoon-Security-Audit/2.0.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data if isinstance(data, dict) else None
    except Exception:
        return None


def classify_osv_advisory(advisory: dict | None, pkg_name: str) -> str | None:
    """Return informational kind ('unmaintained'/'notice') if strictly non-security; else None (blocking)."""
    if not isinstance(advisory, dict):
        return None
    if advisory.get("aliases") or advisory.get("severity"):
        return None

    affected = advisory.get("affected")
    if not isinstance(affected, list) or not affected:
        return None

    matched = [
        entry
        for entry in affected
        if isinstance(entry, dict)
        and isinstance(entry.get("package"), dict)
        and entry["package"].get("name") == pkg_name
        and entry["package"].get("ecosystem") == "crates.io"
    ]
    if not matched:
        return None

    info_kinds: set[str] = set()
    for entry in matched:
        db_spec = entry.get("database_specific")
        if not isinstance(db_spec, dict):
            return None
        info = db_spec.get("informational")
        if info not in NON_BLOCKING_INFORMATIONAL_TYPES:
            return None
        if db_spec.get("categories") or db_spec.get("cvss"):
            return None
        info_kinds.add(str(info))

    return "/".join(sorted(info_kinds)) if info_kinds else None


def audit_cargo_lock_osv(
    cargo_lock_path: Path,
) -> tuple[int, list[tuple[str, str, list[str]]], list[tuple[str, str, list[str]]]]:
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
        headers={"Content-Type": "application/json", "User-Agent": "LuminaMoon-Security-Audit/2.0.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    results = payload.get("results", [])
    advisory_cache: dict[str, dict | None] = {}
    blocking_findings: list[tuple[str, str, list[str]]] = []
    warnings: list[tuple[str, str, list[str]]] = []
    for (name, ver), res in zip(packages, results):
        vulns = res.get("vulns") or []
        if not vulns:
            continue
        blocking_ids: list[str] = []
        warning_ids: list[str] = []
        for v in vulns:
            vid = str(v.get("id", "UNKNOWN"))
            if vid not in advisory_cache:
                advisory_cache[vid] = _fetch_osv_advisory(vid)
            info_kind = classify_osv_advisory(advisory_cache[vid], name)
            if info_kind is not None:
                warning_ids.append(f"{vid} ({info_kind})")
            else:
                blocking_ids.append(vid)
        if blocking_ids:
            blocking_findings.append((name, ver, blocking_ids))
        if warning_ids:
            warnings.append((name, ver, warning_ids))

    return len(packages), blocking_findings, warnings


def audit_tracked_file_hygiene(repo_root: Path) -> tuple[int, list[str]]:
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
        for pat in DISALLOWED_PATH_RULES:
            if pat.search(rel_path):
                violations.append(f"Disallowed file tracked in git: {rel_path}")

        full_path = repo_root / rel_path
        if not full_path.is_file() or full_path.stat().st_size > 2_000_000:
            continue
        try:
            text = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for rule_label, pattern_re in DISALLOWED_CONTENT_RULES:
            if pattern_re.search(text):
                violations.append(f"Disallowed pattern ({rule_label}) in tracked file: {rel_path}")

    return len(tracked_files), violations


def write_step_summary(
    crate_count: int,
    crate_findings: list[tuple[str, str, list[str]]],
    crate_warnings: list[tuple[str, str, list[str]]],
    tracked_count: int,
    hygiene_violations: list[str],
) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    if crate_findings:
        crate_status = f"FAIL (`{len(crate_findings)}` vulnerable crates)"
    elif crate_warnings:
        crate_status = f"PASS (`0` vulnerabilities, `{len(crate_warnings)}` informational warnings)"
    else:
        crate_status = "PASS (`0` vulnerabilities)"

    lines = [
        "### Loomy Supply-Chain & Repository Security Audit",
        "",
        "| Security Check | Scope | Result |",
        "| :--- | :--- | :--- |",
        f"| **RustSec / OSV / CVE Crate Audit** | `{crate_count}` locked crates in `Cargo.lock` | {crate_status} |",
        f"| **Tracked File Hygiene** | `{tracked_count}` git-tracked files | {'PASS (`0` policy violations)' if not hygiene_violations else f'FAIL (`{len(hygiene_violations)}` violations)'} |",
        "",
    ]
    Path(summary_path).open("a", encoding="utf-8").write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OSV/RustSec CVE and repository hygiene audit.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    crate_count, crate_findings, crate_warnings = audit_cargo_lock_osv(repo_root / "Cargo.lock")
    tracked_count, hygiene_violations = audit_tracked_file_hygiene(repo_root)
    write_step_summary(crate_count, crate_findings, crate_warnings, tracked_count, hygiene_violations)

    print(
        f"[SECURITY AUDIT] Audited {crate_count} Rust crates in Cargo.lock against OSV/RustSec/CVE: "
        f"{len(crate_findings)} vulnerabilities, {len(crate_warnings)} informational advisories."
    )
    print(f"[SECURITY AUDIT] Audited {tracked_count} git-tracked files for repository hygiene: {len(hygiene_violations)} violations.")

    for name, ver, ids in crate_warnings:
        print(f"  [WARNING] Crate {name} v{ver}: {', '.join(ids)}")

    if crate_findings or hygiene_violations:
        for name, ver, ids in crate_findings:
            print(f"  [VULNERABILITY] Crate {name} v{ver}: {', '.join(ids)}", file=sys.stderr)
        for item in hygiene_violations:
            print(f"  [HYGIENE VIOLATION] {item}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
