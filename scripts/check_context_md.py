#!/usr/bin/env python3
"""Context-MD harness checker (read-only).

Enforces the two harness rules across all three tiers:
  1. Every managed Markdown file is <= 100 lines (guardrail docs must stay scannable).
  2. The root CLAUDE.md index stays in sync: every managed file is listed in it,
     and every CLAUDE.md / docs/context path it mentions actually resolves.

Managed files = root CLAUDE.md + every <subdir>/CLAUDE.md + every docs/context/*.md.
Exit code is non-zero if any warning fires (usable in CI / a Stop hook).
"""
import re
import sys
from pathlib import Path

LINE_CAP = 100
SKIP_DIRS = {".git", "venv", ".venv", "env", "node_modules", "__pycache__",
             "wandb", "models", "data", "outputs"}

REPO = Path(__file__).resolve().parent.parent
ROOT_INDEX = REPO / "CLAUDE.md"


def managed_files():
    """All harness-managed Markdown files, repo-relative paths."""
    found = []
    for p in REPO.rglob("*.md"):
        if any(part in SKIP_DIRS for part in p.relative_to(REPO).parts):
            continue
        rel = p.relative_to(REPO)
        if p.name == "CLAUDE.md" or rel.parts[:2] == ("docs", "context"):
            found.append(rel)
    return sorted(found)


def main():
    warnings = []
    files = managed_files()

    # Rule 1: line cap
    for rel in files:
        n = sum(1 for _ in (REPO / rel).open(encoding="utf-8"))
        if n > LINE_CAP:
            warnings.append(f"LINE  {rel}: {n} lines (> {LINE_CAP}) — split by sub-topic")

    # Rule 2: index sync (only if a root index exists)
    if not ROOT_INDEX.exists():
        warnings.append("INDEX CLAUDE.md (root) is missing — it is the harness index")
    else:
        text = ROOT_INDEX.read_text(encoding="utf-8")
        # 2a. every managed file (except the root index itself) is referenced in the index
        for rel in files:
            if rel == Path("CLAUDE.md"):
                continue
            if rel.as_posix() not in text:
                warnings.append(f"INDEX {rel}: not listed in root CLAUDE.md")
        # 2b. every CLAUDE.md / docs/context path the index mentions must resolve
        refs = re.findall(r"[\w./-]+/CLAUDE\.md|docs/context/[\w./-]+\.md", text)
        for ref in sorted(set(refs)):
            if not (REPO / ref).exists():
                warnings.append(f"INDEX dead link in CLAUDE.md -> {ref} (no such file)")

    if warnings:
        print(f"check_context_md: {len(warnings)} warning(s)")
        for w in warnings:
            print("  " + w)
        return 1
    print(f"check_context_md: OK — {len(files)} managed file(s), all <= {LINE_CAP} lines, index in sync")
    return 0


if __name__ == "__main__":
    sys.exit(main())
