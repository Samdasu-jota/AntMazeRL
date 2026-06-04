#!/usr/bin/env python3
"""PostToolUse hook — enforce the Context-MD harness rules.

Wired in .claude/settings.json on Write|Edit. Reads the hook JSON from stdin,
and if the edited file is a harness-managed Markdown file (root CLAUDE.md, any
<dir>/CLAUDE.md, or docs/context/*.md) it runs scripts/check_context_md.py.

On a violation (a managed file > 100 lines, or root-index drift) it writes the
checker output to STDERR and exits 2 — Claude Code feeds that back to the model
so it fixes the harness in the same turn. Otherwise it exits 0 silently, so
edits to source files (or clean MD edits) are never interrupted.

Pure stdlib, no jq. Paths resolve from this file's location, so it is
cwd-independent. Any malformed/empty stdin → exit 0 (never block a non-hook call).
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
CHECKER = REPO / "scripts" / "check_context_md.py"


def is_managed(path_str: str) -> bool:
    """True if path_str is a harness-managed context Markdown file."""
    try:
        rel = Path(path_str).resolve().relative_to(REPO)
    except (ValueError, OSError):
        return False  # outside the repo
    if rel.name == "CLAUDE.md":
        return True
    return rel.parts[:2] == ("docs", "context") and rel.suffix == ".md"


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # not a valid hook invocation — do not block
    file_path = (payload.get("tool_input") or {}).get("file_path")
    if not file_path or not is_managed(file_path):
        return 0
    result = subprocess.run(
        [sys.executable, str(CHECKER)], capture_output=True, text=True
    )
    if result.returncode != 0:
        sys.stderr.write(
            "[context-md-guard] Context-MD harness rule violated — fix before continuing "
            "(keep every CLAUDE.md / docs/context file <= 100 lines and listed in the root "
            "CLAUDE.md index; split at the cap):\n"
        )
        sys.stderr.write((result.stdout or "") + (result.stderr or ""))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
