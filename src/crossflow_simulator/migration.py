"""Checks for the M0 layout migration.

After the legacy snapshot was verified, the original top-level ``python/``
tree must disappear and nothing executable may still point at it. Historical
audit records (the implementation plan and the recorded review JSON) may
mention the old path and are excluded explicitly.
"""
from __future__ import annotations

import re
from pathlib import Path

_OLD = "python"
# The original source or tests path (either slash), not preceded by "legacy/".
PATTERN = re.compile(r"(?<!legacy/)(?<!legacy\\)(?<![\w.-])" + _OLD + r"[/\\](?:src|tests)\b")
SCANNED_SUFFIXES = {".py", ".toml", ".cfg", ".ini", ".json", ".spec", ".bat", ".ps1", ".cmd", ".yml", ".yaml", ".txt"}
EXCLUDED_DIRS = {".git", ".venv", "legacy", "build", "dist", "__pycache__", ".pytest_cache", "results"}
HISTORICAL = {"docs/implementation_plan.md", "docs/review/benchmark_results.json"}


def find_references(root: Path) -> list[dict]:
    hits = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SCANNED_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if relative.parts and (relative.parts[0] in EXCLUDED_DIRS or relative.parts[0].startswith(".venv")):
            continue
        if relative.as_posix() in HISTORICAL:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if PATTERN.search(line):
                hits.append({"file": relative.as_posix(), "line": number, "text": line.strip()[:160]})
    return hits


def migration_status(root: str | Path) -> dict:
    root = Path(root)
    original = root / _OLD
    references = find_references(root)
    present = original.exists()
    return {
        "ok": not present and not references,
        "original_python_dir_present": present,
        "references_to_original_paths": references,
        "note": "Remove the original tree with `git rm -r python` once the snapshot is verified."
        if present else "Original python/ tree removed.",
    }
