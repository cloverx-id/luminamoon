import argparse
import os
import re
import shutil
from pathlib import Path

GITHUB_BLOB_BASE = "https://github.com/cloverx-id/luminamoon/blob/main/"
GITHUB_TREE_BASE = "https://github.com/cloverx-id/luminamoon/tree/main/"

EXTERNAL_DOC_COPIES = {
    "README.md": "overview.md",
    "SECURITY.md": "security.md",
    "optimizer/README.md": "optimize/standalone_guide.md",
    "data/README.md": "dataloader/standalone_guide.md",
    "eval/README.md": "lso_eval.md",
    "changelogs/README.md": "changelogs/README.md",
    "changelogs/LUMINAV_ALGORITHM_CHANGELOG.md": "changelogs/LUMINAV_ALGORITHM_CHANGELOG.md",
    "changelogs/SOFTWARE_CHANGELOG.md": "changelogs/SOFTWARE_CHANGELOG.md",
}

REPO_TO_DOCS_MAP = {
    "README.md": "overview.md",
    "SECURITY.md": "security.md",
    "docs/README.md": "README.md",
    "optimizer/README.md": "optimize/standalone_guide.md",
    "data/README.md": "dataloader/standalone_guide.md",
    "data/docs/README.md": "dataloader/README.md",
    "data/docs/library_audit/README.md": "library_audit/README.md",
    "data/docs/library_audit/05_universal_formats_compression_and_diagnostics.md": (
        "library_audit/06_universal_dataloader_formats_and_compression.md"
    ),
    "eval/README.md": "lso_eval.md",
    "changelogs/README.md": "changelogs/README.md",
    "changelogs/LUMINAV_ALGORITHM_CHANGELOG.md": "changelogs/LUMINAV_ALGORITHM_CHANGELOG.md",
    "changelogs/SOFTWARE_CHANGELOG.md": "changelogs/SOFTWARE_CHANGELOG.md",
}

FENCED_MATH_RE = re.compile(
    r"^(?P<indent>[ \t]*)```math[ \t]*\r?\n(?P<body>.*?)^(?P=indent)```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
INLINE_GFM_MATH_RE = re.compile(r"\$`([^`\n]+)`\$")
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)")


def normalize_gfm_math(content: str) -> str:
    def _replace_fence(match: re.Match[str]) -> str:
        indent = match.group("indent")
        body = match.group("body").rstrip("\r\n")
        return f"{indent}$$\n{body}\n{indent}$$"

    content = FENCED_MATH_RE.sub(_replace_fence, content)
    content = INLINE_GFM_MATH_RE.sub(r"$\1$", content)
    return content


def rewrite_links(
    content: str,
    current_doc_path: Path,
    source_origin_path: Path,
    repo_root: Path,
    docs_root: Path,
) -> str:
    repo_docs_root = (repo_root / "docs").resolve()

    def _replace_link(match: re.Match[str]) -> str:
        label = match.group(1)
        raw_target = match.group(2)
        if raw_target.startswith(("http://", "https://", "mailto:", "#")):
            return match.group(0)

        parts = raw_target.split("#", 1)
        target_path_str = parts[0]
        anchor = f"#{parts[1]}" if len(parts) > 1 else ""
        if not target_path_str:
            return match.group(0)

        resolved_from_origin = (source_origin_path.parent / target_path_str).resolve()
        for candidate_docs_root in (docs_root, repo_docs_root):
            try:
                rel_to_docs = resolved_from_origin.relative_to(candidate_docs_root)
                rel_link = os.path.relpath(docs_root / rel_to_docs, current_doc_path.parent).replace("\\", "/")
                return f"[{label}]({rel_link}{anchor})"
            except ValueError:
                pass

        try:
            rel_to_repo = resolved_from_origin.relative_to(repo_root).as_posix()
        except ValueError:
            return match.group(0)

        if rel_to_repo in REPO_TO_DOCS_MAP:
            mapped_doc = docs_root / REPO_TO_DOCS_MAP[rel_to_repo]
            rel_link = os.path.relpath(mapped_doc, current_doc_path.parent).replace("\\", "/")
            return f"[{label}]({rel_link}{anchor})"

        if resolved_from_origin.is_dir():
            return f"[{label}]({GITHUB_TREE_BASE}{rel_to_repo}{anchor})"
        return f"[{label}]({GITHUB_BLOB_BASE}{rel_to_repo}{anchor})"

    return MARKDOWN_LINK_RE.sub(_replace_link, content)


def prepare_docs(repo_root: Path, docs_root: Path) -> None:
    repo_docs_root = (repo_root / "docs").resolve()
    origin_lookup: dict[Path, Path] = {}

    if docs_root != repo_docs_root:
        if docs_root.exists():
            shutil.rmtree(docs_root)
        shutil.copytree(repo_docs_root, docs_root)
        for md_file in docs_root.rglob("*.md"):
            rel = md_file.resolve().relative_to(docs_root)
            origin_lookup[md_file.resolve()] = (repo_docs_root / rel).resolve()

    for src_rel, dst_rel in EXTERNAL_DOC_COPIES.items():
        src_path = (repo_root / src_rel).resolve()
        dst_path = (docs_root / dst_rel).resolve()
        if src_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_path, dst_path)
            origin_lookup[dst_path] = src_path

    for md_file in sorted(docs_root.rglob("*.md")):
        md_resolved = md_file.resolve()
        source_origin = origin_lookup.get(md_resolved, md_resolved)
        raw = md_resolved.read_text(encoding="utf-8")
        updated = normalize_gfm_math(raw)
        updated = rewrite_links(updated, md_resolved, source_origin, repo_root, docs_root)
        md_resolved.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Markdown files in docs/ for MkDocs Material build.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--docs-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    docs_root = (args.docs_dir if args.docs_dir is not None else (repo_root / "docs")).resolve()
    prepare_docs(repo_root, docs_root)
    print(f"Prepared MkDocs directory: {docs_root}")


if __name__ == "__main__":
    main()
